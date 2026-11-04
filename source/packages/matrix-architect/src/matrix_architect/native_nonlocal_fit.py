"""Joint local/nonlocal native ZAFF fitting with observable-driven sparsity."""
from dataclasses import replace
import numpy as np
from matrix_chem import BOHR_TO_ANGSTROM
from matrix_gf.harmonic import CM_PER_HARTREE
from .native_pic_field import NativePicField
from .asymptotic_pic import AsymptoticPicCorrection,AsymptoticPicFactor
from .asymptotic_pic_hessian import NativeCorrectionHessianDesign
from .native_coupling_pruning import prune_native_couplings
from .native_modal_derivatives import native_factor_modal_two_jet,native_modal_derivative_design,complete_native_quartic_catalog
from .native_refinement import native_refinement_catalog
from .esalci_reconstruction import _independent_scaled_fit
from .zaff4 import baseline_modal_derivatives


def compress_native_harmonic_parent(parent,reference):
    """Check every quadratic mixed coupling against the complete reference G/H."""
    factors=[tuple(AsymptoticPicFactor(i,'native') for i in c.primitive_indices) for c in parent.field.retained_couplings]
    coefficients=[c.force_constant for c in parent.field.retained_couplings];fixed=[]
    for c in parent.corrections:
        if len({t.primitive_index for t in c.factors})>1 and sum(t.power for t in c.factors)==2:
            factors.append(c.factors);coefficients.append(c.coefficient)
        else:fixed.append(c)
    factors=tuple(factors);coefficients=np.array(coefficients)
    kernel=NativeCorrectionHessianDesign(parent.primitives,parent.field,[tuple((t.primitive_index,t.form,t.power,t.harmonic) for t in f) for f in factors])
    _,g,h=kernel.value(reference);triangle=np.triu_indices(np.size(reference))
    design=np.vstack([g*BOHR_TO_ANGSTROM,h[triangle]*BOHR_TO_ANGSTROM**2])
    reduced=prune_native_couplings(design,design@coefficients,coefficients,row_tolerances=1e-10,total_removal_budget=.1)
    sparse=np.diag(parent.field.diagonal_force_constants)
    field=replace(parent.field,retained_couplings=(),sparse_typed_hessian=sparse)
    corrections=tuple(fixed)+tuple(AsymptoticPicCorrection(float(c),f) for c,f in zip(reduced.coefficients,factors) if c!=0)
    child=NativePicField(parent.primitives,field,parent.nonbonded_evaluator,corrections=corrections,energy_offset_hartree=parent.energy_offset_hartree,metadata=getattr(parent,'metadata',{}))
    before=parent(reference);after=child(reference)
    error=max(np.max(abs(before.gradient_hartree_per_bohr-after.gradient_hartree_per_bohr)),np.max(abs(before.hessian_hartree_per_bohr2-after.hessian_hartree_per_bohr2)))
    if error>1e-9 or not reduced.report['reduction_accepted']:raise RuntimeError('quadratic coupling reduction changed the reference jet')
    return child,dict(reduced.report,reference_jet_change_max_atomic_units=float(error))


def fit_native_local_and_nonlocal(parent,reference,displacements,observed_cubic,observed_quartic,point_coordinates,point_energies_hartree,point_gradients_bohr,point_hessians_bohr,*,cubic_tolerance_cm=.01,quartic_tolerance_cm=.05,energy_tolerance_cm=1.,gradient_tolerance_Eh_bohr=1e-4,hessian_tolerance_Eh_bohr2=1e-4,maximum_matrix_bytes=32*1024**3,include_full_point_hessians=True,fit_strategy="regularized_native_response",design_cache_directory=None):
    """Fit actual nonlocal E/G/H jointly with measured local F3/F4.

    Point targets are explicit producer-defined PES observations, not a QFF
    sampled off reference. Every candidate retains native functions and bond
    switches. Rank-selected complete terms are refitted and then tested for
    insignificant contributions over the entire calibrated set.
    """
    charge_source=str(getattr(parent.nonbonded_evaluator,'charge_source',''))
    if 'APOC' not in charge_source or 'CM5' not in charge_source:raise ValueError('joint ZAFF4 calibration requires complete APOC CM5 charges')
    if fit_strategy not in ('legacy_basic','equilibrated_native_response','regularized_native_response'):raise ValueError('unknown native fitting strategy')
    if fit_strategy=='regularized_native_response' and design_cache_directory is None:raise ValueError('regularized native fitting requires a recoverable design cache')
    if fit_strategy=='legacy_basic':parent,harmonic_pruning=compress_native_harmonic_parent(parent,reference)
    else:harmonic_pruning=dict(status='FROZEN_PARENT',reason='reference-only pruning does not preserve nonlocal observables',removed_couplings=0,reference_jet_change_max_atomic_units=0.)
    baseline3,baseline4,numerics=baseline_modal_derivatives(parent,reference,displacements,observed_cubic,observed_quartic)
    first,second=native_factor_modal_two_jet(parent.primitives,reference,displacements)
    factors=native_refinement_catalog(parent.primitives,maximum_active_primitives=3)
    factors,completion=complete_native_quartic_catalog(parent.primitives,first,factors,complete_cubic=True)
    nonlocal_support={}
    if fit_strategy in ('equilibrated_native_response','regularized_native_response'):
        from .native_nonlocal_support import extend_native_nonlocal_support
        axes=completion['selected_primitive_indices_zero_based']
        singles=tuple((AsymptoticPicFactor(i,'inverse_distance' if parent.primitives[i].kind=='bond' else 'native' if parent.primitives[i].kind=='angle' else 'sin'),) for i in axes)
        probe=NativeCorrectionHessianDesign(parent.primitives,parent.field,[tuple((t.primitive_index,t.form,t.power,t.harmonic) for t in term) for term in singles])
        amplitudes=np.max(abs(np.array([probe.value(xyz)[0] for xyz in point_coordinates])),axis=0)
        excited=[i for i,a in zip(axes,amplitudes) if a>1e-3*amplitudes.max()]
        factors,nonlocal_support=extend_native_nonlocal_support(parent.primitives,axes,factors,excited_indices=excited)
        nonlocal_support.update(excitation_relative_floor=1e-3,native_axis_calibration_amplitudes=amplitudes.tolist())
    print('native catalog',len(factors),'terms; strategy',fit_strategy,flush=True)
    count=len(factors);dx=np.asarray(displacements).reshape(len(displacements),-1);n=len(dx);triangle=np.triu_indices(n)
    rows_per_point=1+n+(len(triangle[0]) if include_full_point_hessians else 0)
    rows=len(observed_cubic)+len(observed_quartic)+len(point_coordinates)*rows_per_point
    if rows*count*8>maximum_matrix_bytes:raise MemoryError('joint native fit exceeds its declared design budget')
    design=np.empty((rows,count));target=np.empty(rows);offset=0;slices={}
    for degree,observations,baseline,tolerance in [(3,observed_cubic,baseline3,cubic_tolerance_cm),(4,observed_quartic,baseline4,quartic_tolerance_cm)]:
        keys=tuple(sorted(observations));size=len(keys);slices[f'local_F{degree}']=(offset,offset+size,tolerance)
        design[offset:offset+size]=native_modal_derivative_design(first,second,factors,keys,maximum_matrix_bytes=maximum_matrix_bytes)/tolerance
        target[offset:offset+size]=np.array([observations[k]-baseline[k] for k in keys])/tolerance;offset+=size
    kernel=NativeCorrectionHessianDesign(parent.primitives,parent.field,[tuple((t.primitive_index,t.form,t.power,t.harmonic) for t in f) for f in factors])
    gtol=gradient_tolerance_Eh_bohr*CM_PER_HARTREE*np.linalg.norm(dx,axis=1)/BOHR_TO_ANGSTROM
    htol=hessian_tolerance_Eh_bohr2*CM_PER_HARTREE*np.outer(np.linalg.norm(dx,axis=1),np.linalg.norm(dx,axis=1))/BOHR_TO_ANGSTROM**2
    for i,xyz in enumerate(point_coordinates):
        e,g,h=kernel.value(np.asarray(xyz));base=parent(xyz)
        start=offset
        design[offset]=e/energy_tolerance_cm;target[offset]=(point_energies_hartree[i]-base.energy_hartree)*CM_PER_HARTREE/energy_tolerance_cm;offset+=1
        design[offset:offset+n]=(dx@g)/gtol[:,None]
        target[offset:offset+n]=(dx@(point_gradients_bohr[i]-base.gradient_hartree_per_bohr))*CM_PER_HARTREE/BOHR_TO_ANGSTROM/gtol;offset+=n
        if include_full_point_hessians:
            modal=np.einsum('ai,ijk,bj->abk',dx,h,dx,optimize=True)
            design[offset:offset+len(triangle[0])]=modal[triangle]/htol[triangle][:,None]
            residual=dx@(point_hessians_bohr[i]-base.hessian_hartree_per_bohr2)@dx.T*CM_PER_HARTREE/BOHR_TO_ANGSTROM**2
            target[offset:offset+len(triangle[0])]=residual[triangle]/htol[triangle];offset+=len(triangle[0])
        slices[f'nonlocal_{i}']=(start,offset,1.)
        print('native nonlocal design',i+1,'of',len(point_coordinates),flush=True)
    if design_cache_directory is not None:
        from .native_nonlocal_support import save_native_observable_cache
        save_native_observable_cache(design_cache_directory,design,target,factors,parent,dict(slices=slices,completion=completion,nonlocal_support=nonlocal_support,fit_strategy=fit_strategy))
    print('native observable design complete; starting solve',design.shape,flush=True)
    solver_report={}
    if fit_strategy=='regularized_native_response':
        from .native_regularized_fit import regularize_native_observable_cache
        from pathlib import Path
        coefficients,solver_report=regularize_native_observable_cache(design_cache_directory,Path(design_cache_directory).parent/'regularization')
        rank=None
        reduced_report=dict(status='DEFERRED_UNTIL_INDEPENDENT_VALIDATION',removed_couplings=0,reference_only_pruning=False)
    elif fit_strategy=='equilibrated_native_response':
        from .native_nonlocal_support import equilibrated_minimum_norm_fit
        coefficients,solver_report=equilibrated_minimum_norm_fit(design,target)
        rank=solver_report['resolved_rank']
        # Do not destroy a minimum-norm solution by a basic-column refit.
        # Whole-term pruning is deferred until independent nonlocal gates pass.
        reduced_report=dict(status='DEFERRED_UNTIL_INDEPENDENT_VALIDATION',removed_couplings=0,reference_only_pruning=False)
    else:
        coefficients,rank,active=_independent_scaled_fit(design,target,1e-10,'gelsy')
        active=np.flatnonzero(coefficients!=0)
        eligible=np.array([len({t.primitive_index for t in factors[i]})>1 for i in active])
        reduced=prune_native_couplings(design[:,active],target,coefficients[active],row_tolerances=1.,eligible=eligible,rank_reduce=False,total_removal_budget=.1)
        coefficients[active]=reduced.coefficients
        reduced_report=reduced.report
    residual=design@coefficients-target
    errors={key:float(np.max(abs(residual[start:end]))*tol) for key,(start,end,tol) in slices.items()}
    passed=bool(np.max(abs(residual))<=1.)
    report=dict(schema='matrix.architect.zaff4_nonlocal_fit.v1',model_level='ZAFF4',point_count=len(point_coordinates),observations=rows,candidate_terms=count,resolved_rank=None if rank is None else int(rank),retained_anharmonic_terms=int(np.count_nonzero(coefficients)),maximum_normalized_residual=float(np.max(abs(residual))),residuals_by_observable=errors,local_and_nonlocal_gates_passed=passed,harmonic_coupling_pruning=harmonic_pruning,anharmonic_coupling_pruning=reduced_report,fit_strategy=fit_strategy,solver=solver_report,nonlocal_support=nonlocal_support,completion=completion,baseline_numerics=numerics,nonlocal_EGH_used_in_fit=True,unprinted_derivatives_used_as_zero=False,physical_field_qualified=False,independent_validation='separate points required after calibration')
    corrections=parent.corrections+tuple(AsymptoticPicCorrection(float(c)/CM_PER_HARTREE,f) for c,f in zip(coefficients,factors) if c!=0)
    metadata=dict(getattr(parent,'metadata',{}));metadata.update(model_level='ZAFF4',reference_coordinates_angstrom=np.asarray(reference).tolist(),construction_status='LOCAL_AND_NONLOCAL_MATCHED' if passed else 'REJECTED_JOINT_FIT',nonlocal_fit=report)
    child=NativePicField(parent.primitives,parent.field,parent.nonbonded_evaluator,corrections=corrections,energy_offset_hartree=parent.energy_offset_hartree,metadata=metadata)
    if design_cache_directory is not None:
        np.save(__import__('pathlib').Path(design_cache_directory)/'coefficients.npy',coefficients,allow_pickle=False)
    return child,report


def run_native_nonlocal_protocol(request_path):
    """Run a portable joint E/G/H and sparse local derivative calibration."""
    from pathlib import Path
    import hashlib,json,resource,socket
    from threadpoolctl import threadpool_limits
    from matrix_core import resolve_keymaker_protocol,ARCHITECT_ZAFF_REFERENCE_PROTOCOL_ID
    from .native_pic_field import load_native_pic_field,save_native_pic_field
    path=Path(request_path).resolve();request=json.loads(path.read_text());base=path.parent
    from matrix_core.exchange_contracts import validate_request_budget
    validate_request_budget(request)
    required={'schema','required_host','memory_gib','parent_field','observations','nonlocal_observations','output_directory'}
    optional={'workers','tolerances'}
    if not required<=set(request) or set(request)-required-optional or request['schema'] not in ('matrix.architect.zaff4_nonlocal_request.v1','matrix.architect.zaff4_nonlocal_request.v2','matrix.architect.zaff4_nonlocal_request.v3'):
        raise ValueError('unsupported or incomplete joint ZAFF4 request')
    if socket.gethostname().split('.')[0]!=request['required_host'].split('.')[0]:raise RuntimeError('joint ZAFF4 fit requires its declared remote host')
    if request['memory_gib']!=128:raise ValueError('joint ZAFF4 fit requires 128 GiB per process')
    resource.setrlimit(resource.RLIMIT_AS,(128*1024**3,)*2)
    workers=request.get('workers',16)
    if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=128:raise ValueError('invalid worker budget')
    files={key:(base/request[key]).resolve() for key in ('parent_field','observations','nonlocal_observations')}
    parent=load_native_pic_field(files['parent_field'])
    charge_source=str(getattr(parent.nonbonded_evaluator,'charge_source',''))
    if 'APOC' not in charge_source or 'CM5' not in charge_source:raise ValueError('ZAFF requires the complete KEYMAKER/APOC CM5 baseline; estimated charges are not admissible')
    with np.load(files['observations'],allow_pickle=False) as data:
        reference=data['reference_angstrom'];dx=data['displacements_angstrom'];meta=json.loads(str(data['metadata_json']))
        if reference.ndim!=2 or reference.shape[1]!=3 or dx.ndim!=3 or dx.shape[1:]!=reference.shape or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(dx)):raise ValueError('invalid derivative geometry/frame')
        observations=[]
        for degree in (3,4):
            keys=data[f'indices_{degree}'];values=data[f'values_{degree}_cm']
            if keys.shape!=(len(values),degree) or keys.dtype.kind not in 'iu' or np.any(keys<0) or np.any(keys>=len(dx)) or not np.all(np.isfinite(values)):raise ValueError('invalid measured derivative entries')
            table={tuple(sorted(int(i) for i in key)):float(v) for key,v in zip(keys,values)}
            if len(table)!=len(values):raise ValueError('duplicate derivative observation')
            observations.append(table)
    if meta.get('schema')!='matrix.architect.sparse_modal_derivative_observations.v1' or not meta.get('source_sha256') or meta.get('unprinted_as_zero') is not False:raise ValueError('incomplete local derivative provenance')
    with np.load(files['nonlocal_observations'],allow_pickle=False) as data:
        coordinates=data['coordinates_angstrom'];energies=data['energies_hartree'];gradients=data['gradients_bohr'];hessians=data['hessians_bohr'];point_meta=json.loads(str(data['metadata_json']))
        if not np.array_equal(data['reference_angstrom'],reference):raise ValueError('nonlocal and local reference frames differ')
    count=len(coordinates);ndim=reference.size
    if count==0 or coordinates.shape!=(count,*reference.shape) or energies.shape!=(count,) or gradients.shape!=(count,ndim) or hessians.shape!=(count,ndim,ndim):raise ValueError('invalid nonlocal E/G/H dimensions')
    if not all(np.all(np.isfinite(a)) for a in (coordinates,energies,gradients,hessians)):raise ValueError('nonfinite nonlocal observation')
    if not np.allclose(hessians,hessians.transpose(0,2,1),atol=1e-10,rtol=0):raise ValueError('nonlocal Hessians must be symmetric')
    if point_meta.get('schema')!='matrix.architect.nonlocal_egh_observations.v1' or point_meta.get('role')!='calibration' or not point_meta.get('source_sha256') or not point_meta.get('target_definition'):raise ValueError('nonlocal calibration provenance is incomplete')
    tolerances=request.get('tolerances',{})
    allowed={'cubic_tolerance_cm','quartic_tolerance_cm','energy_tolerance_cm','gradient_tolerance_Eh_bohr','hessian_tolerance_Eh_bohr2'}
    if set(tolerances)-allowed or any(isinstance(v,bool) or not np.isfinite(v) or v<=0 for v in tolerances.values()):raise ValueError('invalid physical fit tolerance')
    target=(base/request['output_directory']).resolve();target.mkdir(parents=True,exist_ok=True)
    if any(target.iterdir()):raise FileExistsError('joint ZAFF4 output must be a fresh directory')
    regularized=request['schema']=='matrix.architect.zaff4_nonlocal_request.v3'
    stable=regularized or request['schema']=='matrix.architect.zaff4_nonlocal_request.v2'
    protocol=resolve_keymaker_protocol('zaff4-native-coupling-fit.v4' if regularized else 'zaff4-native-coupling-fit.v3' if stable else 'zaff4-native-coupling-fit.v2')
    with threadpool_limits(limits=workers):candidate,report=fit_native_local_and_nonlocal(parent,reference,dx,*observations,coordinates,energies,gradients,hessians,fit_strategy="regularized_native_response" if regularized else "equilibrated_native_response" if stable else "legacy_basic",design_cache_directory=target/"observable-cache" if stable else None,**tolerances)
    report.update(protocol_id=protocol.protocol_id,protocol_sha256=protocol.sha256,hierarchy=ARCHITECT_ZAFF_REFERENCE_PROTOCOL_ID,request_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),input_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in files.items()},nonlocal_provenance=point_meta,execution_host=socket.gethostname(),memory_gib=128,workers=workers)
    candidate.metadata['owner_protocol']=protocol.protocol_id
    save_native_pic_field(target/('zaff4-calibrated-candidate.json' if report['local_and_nonlocal_gates_passed'] else 'zaff4-rejected-candidate.json'),candidate)
    from matrix_core.exchange_contracts import write_exchange_json
    write_exchange_json(target/'result.json', report)
    return report
