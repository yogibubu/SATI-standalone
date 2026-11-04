"""Native coupled-F adapter for assignment, blends and progressive refinement.

The inner fit uses a fixed Galerkin LAM span; outer stages may enlarge it.
All Hamiltonians and analytic directions come from the production assembler.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import numpy as np


def _model(value):
    from .spin_hyperfine import SpinHyperfineModel
    return value if isinstance(value,SpinHyperfineModel) else SpinHyperfineModel.from_dict(value)


def _parse(surface,model,request):
    from .lam_fit import _lam_variable
    from .spectral_fit import _get_parameter
    from .general_lam_operator import normalize_general_lam_operator_manifest
    defaults=({} if request.get('operator_manifest') is None else
              normalize_general_lam_operator_manifest(request['operator_manifest']).parameters_MHz)
    parsed=[]
    for i,raw in enumerate(request.get('variables',())):
        path=raw['path']
        if path.startswith(('electron.','nucleus:','pair:')):
            value=_get_parameter(None,model,path)
            multiplier=float(raw.get('multiplier',1.))
            if not np.isfinite(multiplier) or multiplier==0:raise ValueError('finite nonzero spin multiplier required')
            p=dict(name=raw.get('name',f'p{i+1}'),path=path,
                initial=float(raw.get('initial',value/multiplier)),scale=float(raw.get('scale',max(abs(value),1.))),
                lower=float(raw.get('lower',-np.inf)),upper=float(raw.get('upper',np.inf)),
                multiplier=multiplier,prior=deepcopy(raw.get('prior')))
        else:p=_lam_variable(raw,surface,i,defaults)
        if (not np.isfinite(p['initial']) or not np.isfinite(p['scale']) or p['scale']<=0
                or not np.isfinite(p['multiplier']) or not p['lower']<p['upper']
                or not p['lower']<=p['initial']<=p['upper']):
            raise ValueError('invalid native assignment variable')
        parsed.append(p)
    if len({p['name'] for p in parsed})!=len(parsed) or len({p['path'] for p in parsed})!=len(parsed):
        raise ValueError('unique spin assignment variable names and paths required')
    return parsed


class _SpinContext:
    def __init__(self,surface,model,variables,values,*,fs,lam_states,options=None,basis=None,operator_manifest=None):
        from .lam_rotation import solve_lam_basis
        from .lam_fit import _apply_lam_variables
        self.surface=surface;self.model=model;self.variables=variables;self.fs=tuple(fs)
        self.options=options or {};self.cache={};self.reference=None
        from .general_lam_operator import normalize_general_lam_operator_manifest
        self.manifest=None if operator_manifest is None else normalize_general_lam_operator_manifest(operator_manifest)
        self.scalar=[i for i,p in enumerate(variables) if not p['path'].startswith(('electron.','nucleus:','pair:'))]
        changed=_apply_lam_variables(surface,[variables[i] for i in self.scalar],np.asarray(values)[self.scalar])
        self.basis=basis or solve_lam_basis(changed,state_count=lam_states,
            primitive_basis=self.options.get('primitive_basis','auto'),fourier_cutoffs=self.options.get('fourier_cutoffs'))
        self.units=None;self.statistical_weights=None
        self.reference=self.snapshot(values)

    def snapshot(self,values):
        from .lam_fit import _apply_lam_variables,_surface_parameter_direction,_operator_parameter_values
        from .spectral_fit import _set_parameter
        from .physical_spin_response import physical_spin_matrices
        from .native_spin_effective import native_spin_n_values
        from .spin_reduced_fit import native_spin_parameter_matrices
        from matrix_numerics import diagonalize_hermitian
        x=np.asarray(values,float);key=x.tobytes()
        if key in self.cache:return self.cache[key]
        scalar=[self.variables[i] for i in self.scalar]
        surface=_apply_lam_variables(self.surface,scalar,x[self.scalar]);model=self.model
        for i,p in enumerate(self.variables):
            if i not in self.scalar:_,model=_set_parameter(None,model,p['path'],float(x[i])*p['multiplier'])
        directions=[]
        for i in range(len(self.variables)):
            directions.append(_surface_parameter_direction(self.surface,scalar,x[self.scalar],self.scalar.index(i))
                              if i in self.scalar and not self.variables[i]['path'].startswith('operator.') else {})
        memory=self.options.get('memory_budget_bytes',512*1024**2)
        parameters=_operator_parameter_values(self.manifest,self.variables,x)
        operator_directions=[{p['path'][9:]:p['multiplier']} if p['path'].startswith('operator.') else {}
                             for p in self.variables]
        (h,dh,mu,dmu),blocks,_=physical_spin_matrices(surface,self.basis,directions or [{}],model,
            two_f_values=self.fs,response_states=0,memory_budget_bytes=memory,
            operator_manifest=self.manifest,operator_parameters_MHz=parameters,
            operator_directions=operator_directions or [{}])
        spin=[i for i in range(len(self.variables)) if i not in self.scalar]
        if spin:
            if self.units is None:
                ns=sorted({n for f in self.fs for n in native_spin_n_values(model,f)})
                k=self.basis.vectors.shape[1]
                dummy=SimpleNamespace(basis=self.basis,representation='Ir',blocks={
                    n:SimpleNamespace(energies_MHz=np.zeros(k*(2*n+1)),vectors=np.eye(k*(2*n+1))) for n in ns})
                self.units=native_spin_parameter_matrices(surface,dummy,model,two_f_values=self.fs,
                    parameters=[self.variables[i]['path'] for i in spin],memory_budget_bytes=memory)
            for f in self.fs:
                for j,i in enumerate(spin):dh[f][i]=self.units[f][j]*self.variables[i]['multiplier']
        output={}
        for f in self.fs:
            diagonal=diagonalize_hermitian(h[f],backend=self.options.get('diagonalizer_backend'),check_finite=True)
            energies=np.asarray(diagonal.eigenvalues);vectors=np.asarray(diagonal.eigenvectors)
            residual=h[f]@vectors-vectors*energies
            scaled=np.max(np.linalg.norm(residual,axis=0)/np.maximum(1.,np.linalg.norm(h[f],ord=np.inf)+abs(energies)),initial=0.)
            if scaled>1e-10 or np.max(abs(vectors.conj().T@vectors-np.eye(len(energies))),initial=0.)>1e-10:
                raise RuntimeError('native spin assignment eigensystem failed its residual/orthogonality checks')
            output[f]=dict(energies=energies,vectors=vectors,
                basis=blocks[f].basis,derivatives=dh[f][:len(self.variables)])
        if self.statistical_weights is None and (surface.nuclear_spin_weights or surface.permutation_inversion_symmetry is not None):
            from .lam_rotation import solve_lam_rotation
            from .spin_hyperfine import ElectronSpin,NuclearSpin,NuclearSpinSpinCoupling,SpinHyperfineModel
            from .native_spin_effective import build_native_lam_spin_operator
            ns=sorted({n for f in self.fs for n in native_spin_n_values(model,f)})
            lam=solve_lam_rotation(surface,j_values=ns,lam_basis=self.basis,
                lam_state_count=len(self.basis.energies_MHz),reproject_lam_core=True,
                operator_manifest=self.manifest,operator_parameters_MHz=parameters)
            spatial={n:(b.vectors*np.array([l.spin_weight for l in b.levels]))@b.vectors.conj().T
                     for n,b in lam.blocks.items()}
            k=len(self.basis.energies_MHz)
            raw=SimpleNamespace(basis=self.basis,representation='Ir',blocks={n:SimpleNamespace(
                energies_MHz=np.zeros(k*(2*n+1)),vectors=np.eye(k*(2*n+1))) for n in ns})
            zero=SpinHyperfineModel(electron=ElectronSpin(two_S=model.electron.two_S),
                nuclei=tuple(NuclearSpin(n.label,n.two_I) for n in model.nuclei),
                nuclear_spin_spin=tuple(NuclearSpinSpinCoupling(p.left,p.right) for p in model.nuclear_spin_spin))
            empty=replace(surface,spin_tensor_surfaces_MHz={key:np.zeros_like(v)
                          for key,v in (surface.spin_tensor_surfaces_MHz or {}).items()})
            self.statistical_weights={f:build_native_lam_spin_operator(empty,raw,zero,two_F=f,
                _spin_free_blocks=spatial,memory_budget_bytes=memory).hamiltonian_MHz for f in self.fs}
        result=dict(basis=self.basis,blocks=output,dipoles=mu,statistical_weights=self.statistical_weights,
                    dipole_derivatives={p:v[:len(self.variables)] for p,v in dmu.items()})
        self.cache={key:result};return result

    def evaluate(self,values,observations):
        from .assignment_search import assignment_components
        from .spin_reduced_fit import _projector_response
        from .native_rotor import MHZ_TO_KELVIN
        snap=self.snapshot(values)
        handoff=_spin_transport(self.reference,snap,observations,
            minimum_principal_overlap=self.options.get('minimum_principal_overlap',.9),
            minimum_state_overlap=self.options.get('minimum_state_overlap',.8),
            degeneracy_tolerance_MHz=self.options.get('degeneracy_tolerance_MHz',1e-7))
        if not handoff['ready']:raise ValueError(handoff['reason'])
        vals=[];jac=[];component_data=[]
        for row in handoff['observations']:
            frequencies=[];derivatives=[];weights=[];dw=[]
            native=row.get('weight_mode','fixed')=='native'
            for component in assignment_components(row):
                left=component['lower'];right=component['upper'];f=left['two_F'];g=right['two_F']
                a=snap['blocks'][f];b=snap['blocks'][g];i=left['reference_index'];j=right['reference_index']
                l,dl,de=_projector_response(a['energies'],a['vectors'],a['derivatives'],[i])
                r,dr,df=_projector_response(b['energies'],b['vectors'],b['derivatives'],[j])
                frequency=float(b['energies'][j]-a['energies'][i]);derivative=df-de
                w=float(component.get('weight',1.));d=np.zeros(len(values))
                if native:
                    pair=(f,g)
                    if pair in snap['dipoles']:
                        op=snap['dipoles'][pair];dop=snap['dipole_derivatives'][pair]
                    elif (g,f) in snap['dipoles']:
                        op=snap['dipoles'][g,f].conj().T;dop=snap['dipole_derivatives'][g,f].conj().transpose(0,2,1)
                    else:raise ValueError('native E1 operator missing for blend component')
                    amplitude=l.conj().T@op@r
                    da=dl.conj().transpose(0,2,1)@op@r+l.conj().T@dop@r+l.conj().T@op@dr
                    strength=float(np.sum(abs(amplitude)**2))
                    ds=2*np.real(np.einsum('ij,pij->p',amplitude.conj(),da))
                    if snap['statistical_weights'] is not None:
                        statistical=snap['statistical_weights'][f]
                        population_weight=float(np.trace(l.conj().T@statistical@l).real)
                        dweight=2*np.real(np.einsum('ij,pij->p',l.conj(),statistical@dl))
                        ds=ds*population_weight+strength*dweight;strength*=population_weight
                    # A common thermal normalization cancels in each centroid.
                    temperature=self.options.get('temperature_K')
                    population=1.;dp=np.zeros(len(values))
                    if temperature is not None:
                        beta=MHZ_TO_KELVIN/float(temperature)
                        if not np.isfinite(beta) or beta<=0:raise ValueError('positive blend temperature required')
                        origin=min(np.min(b['energies']) for b in snap['blocks'].values())
                        lower=float(a['energies'][i])-origin;population=frequency*(-np.expm1(-beta*frequency))*np.exp(-beta*lower)
                        dp=np.exp(-beta*lower)*(((-np.expm1(-beta*frequency))+frequency*beta*np.exp(-beta*frequency))*derivative
                            -beta*frequency*(-np.expm1(-beta*frequency))*de)
                    d=w*(ds*population+strength*dp);w*=strength*population
                frequencies.append(frequency);derivatives.append(derivative);weights.append(w);dw.append(d)
            frequencies=np.asarray(frequencies);derivatives=np.asarray(derivatives);weights=np.asarray(weights)
            if not np.all(np.isfinite(weights)) or np.any(weights<0) or weights.sum()<=0:
                raise ValueError('blend has no finite positive native strength')
            component_data.append(dict(frequencies=frequencies,jacobian=derivatives,
                strengths=weights,strength_jacobian=np.asarray(dw)))
            norm=weights/weights.sum();center=float(norm@frequencies)
            vals.append(center);jac.append(norm@derivatives+(frequencies-center)@np.asarray(dw)/weights.sum())
        return dict(values=np.array(vals),jacobian=np.asarray(jac),components=component_data)

    def promote_accepted(self,values):
        # Trial eigenstates are always compared against a physical accepted
        # anchor. Stable indices are retained through a complete permutation.
        from .subspace_transport import transport_spectral_subspaces
        snap=self.snapshot(values);reference=self.reference
        blocks={}
        for f,a in reference['blocks'].items():
            b=snap['blocks'][f]
            tracked=transport_spectral_subspaces(a['energies'],b['energies'],a['vectors'].conj().T@b['vectors'],gap_tolerance=1e-7)
            permutation=[tracked['mapping'][i] for i in range(len(a['energies']))]
            blocks[f]={**b,'energies':b['energies'][permutation],'vectors':b['vectors'][:,permutation]}
        self.reference={**snap,'blocks':blocks}


def _spin_transport(reference,current,observations,*,minimum_principal_overlap=.9,
                    minimum_state_overlap=.8,degeneracy_tolerance_MHz=1e-7):
    from .assignment_search import assignment_components
    from .subspace_transport import transport_spectral_subspaces
    components=[c for r in observations for c in assignment_components(r)]
    selected={f:set() for f in reference['blocks']};ids={}
    for c in components:
        for side in ('lower','upper'):
            s=c[side];f=s['two_F'];i=s['reference_index'];state_id=s.get('state_id')
            if f not in selected or not isinstance(i,int) or isinstance(i,bool) or not 0<=i<len(reference['blocks'][f]['energies']):
                raise ValueError('spin state selector outside native F block')
            if not state_id or state_id in ids and ids[state_id]!=(f,i):
                raise ValueError('inconsistent persistent spin state identity')
            ids[state_id]=(f,i);selected[f].add(i)
    gram=reference['basis'].vectors.conj().T@current['basis'].vectors
    maps={};principal=1.;state=1.
    for f,required in selected.items():
        if not required:continue
        if f not in current['blocks']:raise ValueError('native F block missing during transport')
        a=reference['blocks'][f];b=current['blocks'][f]
        cross=np.zeros((len(a['basis']),len(b['basis'])),complex)
        groups={}
        for which,registry in enumerate((a['basis'],b['basis'])):
            for index,s in enumerate(registry):
                key=(s.N,s.two_J,s.nuclear_path,s.k_index%(2*s.N+1))
                pair=groups.setdefault(key,[[],[]]);pair[which].append((index,s.k_index//(2*s.N+1)))
        for left,right in groups.values():
            if left and right:
                li,lu=zip(*left,strict=True);ri,ru=zip(*right,strict=True)
                cross[np.ix_(li,ri)]=gram[np.ix_(lu,ru)]
        eig=a['vectors'].conj().T@cross@b['vectors']
        try:
            tracked=transport_spectral_subspaces(a['energies'],b['energies'],eig,
                diagnostic_indices=sorted(required),gap_tolerance=degeneracy_tolerance_MHz)
        except ValueError as exc:return dict(ready=False,reason=str(exc),observations=None)
        principal=min(principal,tracked['minimum_principal_overlap']);state=min(state,tracked['minimum_state_overlap'])
        maps[f]=tracked['mapping']
        for i in required:
            j=maps[f][i]
            if (np.count_nonzero(abs(a['energies']-a['energies'][i])<=degeneracy_tolerance_MHz)>1
                    or np.count_nonzero(abs(b['energies']-b['energies'][j])<=degeneracy_tolerance_MHz)>1):
                return dict(ready=False,reason='degenerate_spin_components_require_projector_measurements',observations=None)
    diagnostic=dict(minimum_principal_overlap=principal,minimum_state_overlap=state)
    if principal<minimum_principal_overlap or state<minimum_state_overlap:
        return dict(ready=False,reason='insufficient_spin_overlap',observations=None,diagnostic=diagnostic)
    result=deepcopy(observations)
    for c in [c for r in result for c in assignment_components(r)]:
        for side in ('lower','upper'):
            s=c[side];f=s['two_F'];s['reference_index']=int(maps[f][s['reference_index']])
            s['reference_energy_MHz']=float(current['blocks'][f]['energies'][s['reference_index']])
    return dict(ready=True,reason='native_coupled_F_overlap_preserved',observations=result,diagnostic=diagnostic)


def spin_exploration_catalog(surface,spin_model,*,j_values,two_f_values,lam_states,
        dipole_au,temperature_K,prediction_uncertainty_MHz,solve_options=None,operator_manifest=None):
    from .spin_hyperfine import coupled_nuclear_spin_paths
    model=_model(spin_model)
    if surface.dipole_au is None:
        if dipole_au is None:raise ValueError('physical dipole required for spin assignment')
        surface=replace(surface,dipole_au=np.broadcast_to(dipole_au,surface.grid_shape+(3,)).copy())
    if two_f_values is None:
        paths=coupled_nuclear_spin_paths(tuple(n.two_I for n in model.nuclei));fs=set()
        for n in j_values:
            for j in range(abs(2*n-model.electron.two_S),2*n+model.electron.two_S+1,2):
                for p in paths:
                    i=p[-1] if p else 0;fs.update(range(abs(j-i),j+i+1,2))
        two_f_values=sorted(fs)
    fs=tuple(sorted(set(two_f_values)))
    if not fs or any(isinstance(f,bool) or not isinstance(f,int) or f<0 for f in fs):
        raise ValueError('nonnegative doubled F values required')
    context=_SpinContext(surface,model,[],np.array([]),fs=fs,lam_states=lam_states,options=solve_options,
                         operator_manifest=operator_manifest)
    snap=context.reference;rows=[]
    from .native_rotor import MHZ_TO_KELVIN
    beta=None if temperature_K is None else MHZ_TO_KELVIN/temperature_K
    ground=min(np.min(b['energies']) for b in snap['blocks'].values())
    for (f,g),op in snap['dipoles'].items():
        a=snap['blocks'][f];b=snap['blocks'][g]
        strengths=abs(a['vectors'].conj().T@op@b['vectors'])**2
        for i,j in np.argwhere(strengths>1e-18):
            i=int(i);j=int(j)
            if f==g and j<=i:continue
            left=(f,i,float(a['energies'][i]));right=(g,j,float(b['energies'][j]))
            if left[2]>right[2]:left,right=right,left
            frequency=right[2]-left[2]
            if frequency<=1e-9:continue
            if any(np.count_nonzero(abs(snap['blocks'][k]['energies']-e)<=1e-7)>1 for k,_,e in (left,right)):
                continue
            strength=float(strengths[i,j]);intensity=strength
            if snap['statistical_weights'] is not None:
                f_low,i_low,_=left;v=snap['blocks'][f_low]['vectors'][:,i_low]
                intensity*=max(0.,float(np.vdot(v,snap['statistical_weights'][f_low]@v).real))
            if beta is not None:intensity*=frequency*(-np.expm1(-beta*frequency))*np.exp(-beta*(left[2]-ground))
            if intensity<=0:continue
            selectors={side:dict(two_F=k,reference_index=index,state_id=f'F2={k}:reference={index}',reference_energy_MHz=e)
                       for side,(k,index,e) in (('lower',left),('upper',right))}
            rows.append(dict(id=selectors['lower']['state_id']+'->'+selectors['upper']['state_id'],
                family=f'F2={left[0]}->{right[0]}',frequency_MHz=frequency,
                uncertainty_MHz=prediction_uncertainty_MHz,relative_intensity=intensity,**selectors))
    unique={r['id']:r for r in rows};rows=sorted(unique.values(),key=lambda r:r['frequency_MHz'])
    maximum=max((r['relative_intensity'] for r in rows),default=1.)
    for r in rows:r['relative_intensity']/=maximum
    return snap,rows


def refine_spin_assignments(surface,fit_request,assignment_report,reference,stages,*,
        minimum_principal_overlap=.9,minimum_state_overlap=.8,degeneracy_tolerance_MHz=1e-7,
        maximum_total_evaluations=60,checkpoint_path=None,resume_checkpoint=None,stage_limit=None,
        _resume_state=None,_checkpoint_callback=None):
    from .assignment_checkpoint import assignment_fingerprint,load_assignment_checkpoint,save_assignment_checkpoint
    from .assignment_search import assignments_to_observations
    from .assignment_covariance import assignment_fit_covariance
    from .assignment_statistics import fit_assignment_stage

    request=deepcopy(fit_request);stages=deepcopy(list(stages));model=_model(request['spin_model'])
    if request.get('schema')!='matrix.lock.lam_spectral_fit.v1':raise ValueError('native LAM/spin fit template required')
    if isinstance(maximum_total_evaluations,bool) or not isinstance(maximum_total_evaluations,int) or maximum_total_evaluations<1:
        raise ValueError('positive integer total fit budget required')
    if request.get('options',{}).get('loss','linear')!='linear':
        raise ValueError('correlated native spin assignment fitting requires a Gaussian objective')
    if request.get('structured_discrepancy') is not None:
        raise ValueError('discrepancy calibration uses separate rows after assignment')
    if surface.dipole_au is None and request.get('dipole_au') is not None:
        surface=replace(surface,dipole_au=np.broadcast_to(request['dipole_au'],surface.grid_shape+(3,)).copy())
    variables=_parse(surface,model,request)
    if not variables or not stages:raise ValueError('spin variables and nonempty stages required')
    paths=[v['path'] for v in variables];active=set();count=len(reference['basis'].energies_MHz)
    total=0;names=set()
    for s in stages:
        selected=set(s['active_paths']);states=s.get('lam_states',count);budget=s.get('max_nfev',20)
        if (not selected or selected-set(paths) or not active<=selected
                or len(selected)!=len(s['active_paths']) or not s['name'] or s['name'] in names):
            raise ValueError('unique stage names and nested native spin variable paths required')
        if (isinstance(states,bool) or not isinstance(states,int) or states<count
                or isinstance(budget,bool) or not isinstance(budget,int) or budget<1):
            raise ValueError('positive native spin stage budgets and nondecreasing LAM spaces required')
        total+=budget;count=states;active=selected;names.add(s['name'])
    if total>maximum_total_evaluations:raise ValueError('spin stages exceed the total fit budget')
    if stage_limit is not None and (isinstance(stage_limit,bool) or not isinstance(stage_limit,int) or stage_limit<1):
        raise ValueError('positive integer stage limit required')
    settings=dict(minimum_principal_overlap=minimum_principal_overlap,minimum_state_overlap=minimum_state_overlap,
                  degeneracy_tolerance_MHz=degeneracy_tolerance_MHz)
    if not all(np.isfinite(x) for x in settings.values()) or not 0<minimum_principal_overlap<=1 or not 0<minimum_state_overlap<=1 or degeneracy_tolerance_MHz<0:
        raise ValueError('finite positive overlap thresholds and nonnegative degeneracy tolerance required')
    contract=assignment_fingerprint('coupled_F',surface,request,assignment_report,stages,settings,maximum_total_evaluations)
    if resume_checkpoint is None and checkpoint_path is not None and Path(checkpoint_path).exists():
        resume_checkpoint=checkpoint_path
    if resume_checkpoint is not None:
        _resume_state=load_assignment_checkpoint(resume_checkpoint,contract=contract)
        if checkpoint_path is None:checkpoint_path=resume_checkpoint
    observations=assignments_to_observations(assignment_report)
    if not observations:
        return dict(status='no_accepted_assignments',completed=False,stages=[],observations=[],last_accepted_values={})
    request,audit=assignment_fit_covariance(request,assignment_report,observations)
    values=np.array([v['initial'] for v in variables]);records=[];start=0
    if _resume_state is not None:
        if _resume_state.get('contract')!=contract:raise ValueError('spin assignment checkpoint contract mismatch')
        if _resume_state.get('terminal_report') is not None:return deepcopy(_resume_state['terminal_report'])
        values=np.asarray(_resume_state['values']);observations=deepcopy(_resume_state['observations'])
        reference=_resume_state['reference'];records=deepcopy(_resume_state['records']);start=_resume_state['next_stage']
    def snapshot(next_stage,terminal=None):
        state=dict(contract=contract,values=values.copy(),reference=reference,observations=observations,
                   records=records,next_stage=next_stage,terminal_report=terminal)
        if checkpoint_path is not None:save_assignment_checkpoint(checkpoint_path,state,contract=contract)
        if _checkpoint_callback is not None:_checkpoint_callback(state)
    snapshot(start);completed=False;executed=0
    options={**request.get('options',{}),**settings}
    if request.get('temperature_K') is not None:options['temperature_K']=request['temperature_K']
    for index in range(start,len(stages)):
        if stage_limit is not None and executed>=stage_limit:break
        stage=stages[index];executed+=1
        context=_SpinContext(surface,model,variables,values,fs=sorted(reference['blocks']),
            lam_states=stage.get('lam_states',len(reference['basis'].energies_MHz)),options=options,
            operator_manifest=request.get('operator_manifest'))
        handoff=_spin_transport(reference,context.reference,observations,**settings)
        entry=dict(name=stage['name'],initial_values=dict(zip(paths,values.tolist(),strict=True)),
                   active_paths=stage['active_paths'],handoff=handoff);records.append(entry)
        if not handoff['ready']:entry['status']='assignment_transport_blocked';break
        try:
            fit=fit_assignment_stage(context,handoff['observations'],variables,values,stage['active_paths'],
                                     request,max_nfev=stage.get('max_nfev',20))
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            entry.update(status='fit_failed',message=str(exc));break
        entry['fit']=fit
        if not fit['success']:entry['status']='fit_not_converged';break
        trial=np.asarray(fit['values']);context.promote_accepted(trial)
        updated=_spin_transport(context.reference,context.reference,handoff['observations'],**settings)
        if not updated['ready']:entry['status']='postfit_transport_blocked';break
        entry['postfit_handoff']=updated;entry['status']='accepted'
        values=trial;reference=context.reference;observations=updated['observations'];snapshot(index+1)
    else:completed=True
    paused=not completed and records[-1]['status']=='accepted'
    report=dict(schema='matrix.lock.assignment_refinement.v1',backend='native_coupled_F',
        status='checkpointed' if paused else 'completed' if completed else records[-1]['status'],completed=completed,
        stages=records,observations=observations,last_accepted_values=dict(zip(paths,values.tolist(),strict=True)),
        measurement_covariance=audit,resumed_from_stage=start,executed_stages=executed,
        declared_evaluation_budget=total,scope='fixed-Galerkin native spin/LAM inner fits; precision basis qualification separate')
    snapshot(sum(r['status']=='accepted' for r in records),None if paused else report)
    return report
