"""TDA-DFT and full TD-DFT with EPOC optical linear-response polarization.

This is nonequilibrium LR absorption, not relaxed excited-state SS-DFT.
The slow solvent remains prepared on the equilibrated ground state. The fast
kernel is the derivative of EPOC's electronic reaction operator, without a
second static contribution or an empirical shift of vacuum excitation energies.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from .cpcm import CPCMConfinement
from .nonequilibrium import NonequilibriumCPCM
from .solvent_scf_protocol import ElectronicSolvationStep, iterate_solvation, _ao_integrals

REQUEST_SCHEMA = 'matrix.epoc.solvent_td_request.v1'
RESULT_SCHEMA = 'matrix.epoc.solvent_td_result.v1'
PROTOCOL_ID = 'epoc-ellipsoidal-solvent-td-dft.v1'


def fast_electronic_response(model, operators, density_response):
    """Apply dV_e/dD = f_opt sum_m W_m I_m Tr(I_m dD).

    Batched and nonsymmetric real transition densities are allowed. The
    electronic response operator is negative semidefinite for singlet charge
    fluctuations, since W is the negative conductor kernel.
    """
    if operators.operator_sha256 != model.operator_sha256:
        raise ValueError('response integrals and EPOC cavity differ')
    if np.iscomplexobj(density_response):
        raise ValueError('this adapter requires a real density response')
    dm = np.asarray(density_response,dtype=float)
    integrals = operators.integrals
    if dm.ndim < 2 or dm.shape[-2:] != integrals.shape[-2:] or not np.all(np.isfinite(dm)):
        raise ValueError('finite AO response matrix or batch required')
    contractions = np.einsum('...ij,mji->...m',dm,integrals)
    return np.einsum('...m,m,mij->...ij',contractions,model.optical_factor*model.weights,integrals)


def attach_optical_response(td, model, operators):
    """Decorate a PySCF response solver, retaining its vacuum XC/HF kernel."""
    original = td.gen_response
    def generate(*args,**kwargs):
        vacuum = original(*args,**kwargs)
        singlet = kwargs.get('singlet',True)
        def respond(dm):
            response = vacuum(dm)
            if singlet is not False:
                response = response + fast_electronic_response(model,operators,dm)
            return response
        return respond
    td.gen_response = generate
    return td


def run_ellipsoidal_td_protocol(request_path):
    """KEYMAKER: explicit TDA-DFT or full TD-DFT vertical LR absorption."""
    from matrix_core import (CalculationResources,build_calculation_launch_plan,
                            require_calculation_launch_or_parent_authorization)
    request_path = Path(request_path).expanduser().resolve()
    request = json.loads(request_path.read_text())
    fields = {'schema','backend','response_model','basis','xc','molecule','cavity',
              'dielectrics','nstates','output_directory'}
    if not isinstance(request,dict) or set(request) != fields or request['schema'] != REQUEST_SCHEMA:
        raise ValueError('invalid EPOC solvent TDA/TD-DFT request')
    response_model = request['response_model']
    if request['backend'] != 'pyscf' or response_model not in {'TDA-DFT','TD-DFT'}:
        raise ValueError('explicit PySCF TDA-DFT or TD-DFT response model required')
    for key in ('basis','xc'):
        if not isinstance(request[key],str) or not request[key].strip():
            raise ValueError('explicit DFT functional and Gaussian basis required')
    nstates = request['nstates']
    if isinstance(nstates,bool) or not isinstance(nstates,int) or not 1 <= nstates <= 6:
        raise ValueError('request between one and six singlet roots')
    molecule = request['molecule']
    if (not isinstance(molecule,dict) or set(molecule) != {'atoms','unit','charge','spin'}
            or molecule['unit'] not in {'angstrom','bohr'} or molecule['spin'] != 0
            or isinstance(molecule['charge'],bool) or not isinstance(molecule['charge'],int)):
        raise ValueError('explicit closed-shell molecule, charge, geometry and units required')
    cavity, dielectrics = request['cavity'],request['dielectrics']
    if not isinstance(cavity,dict) or set(cavity) != {'center_bohr','semiaxes_bohr','rotation'}:
        raise ValueError('explicit fixed cavity geometry required')
    if not isinstance(dielectrics,dict) or set(dielectrics) != {'static','optical'}:
        raise ValueError('explicit static and optical dielectrics required')
    model = NonequilibriumCPCM.compile(CPCMConfinement(**cavity,maximum_degree=2),
        static_dielectric=dielectrics['static'],optical_dielectric=dielectrics['optical'])
    destination = request['output_directory']
    if not isinstance(destination,str) or not destination.strip():
        raise ValueError('an explicit new output directory is required')
    output = Path(destination).expanduser()
    if not output.is_absolute():output=request_path.parent/output
    output=output.resolve()
    if output.exists():raise FileExistsError('TDA/TD-DFT output directory already exists')
    plan = build_calculation_launch_plan(backend='PySCF/EPOC-'+response_model,
        input_path=request_path,command=('keymaker','--epoc-solvent-td',str(request_path)),workdir=output,
        resources=CalculationResources(process_count=1,threads_per_process=1,memory_per_job_gb=1.,concurrent_jobs=1))
    require_calculation_launch_or_parent_authorization(plan,None)
    from threadpoolctl import threadpool_limits
    from pyscf import gto,dft,lib,tdscf
    from matrix_apoc import electronic_state_from_pyscf
    output.mkdir(parents=True,exist_ok=False)
    with threadpool_limits(limits=1):
        lib.num_threads(1)
        mol=gto.M(atom=molecule['atoms'],unit=molecule['unit'],charge=molecule['charge'],spin=0,
                  basis=request['basis'],max_memory=512,verbose=0)
        if mol.nao > 80 or mol.nelectron % 2:
            raise ValueError('closed-shell systems with <=80 AOs required')
        def configure():
            mf=dft.RKS(mol);mf.xc=request['xc'];mf.grids.level=4
            mf.conv_tol=1e-12;mf.conv_tol_grad=1e-9;mf.max_cycle=80;mf.chkfile=None
            if mf.do_nlc():
                raise ValueError('nonlocal-correlation response requires a separately qualified adapter')
            return mf
        vacuum=configure();vacuum.kernel()
        if not vacuum.converged:raise RuntimeError('vacuum RKS did not converge')
        initial=electronic_state_from_pyscf(vacuum)
        operators=_ao_integrals(model,mol,initial)
        h0=vacuum.get_hcore()
        last_engine=None
        def solve(potential,previous):
            nonlocal last_engine
            mf=configure();mf.get_hcore=lambda *args:h0+potential
            mf.kernel(dm0=previous.density_ao())
            last_engine=mf
            density=mf.make_rdm1()
            work=float(np.einsum('ij,ji->',density,potential))
            return ElectronicSolvationStep(electronic_state_from_pyscf(mf),float(mf.e_tot)-work,bool(mf.converged))
        ground=iterate_solvation(model,operators,initial,solve)
        # The independently checked last engine retains the fully polarized
        # orbital energies. Its vacuum response kernel contains no EPOC term.
        mf=last_engine
        density=mf.make_rdm1()
        residual=float(np.max(np.abs(density-ground.state.density_ao())))
        if residual >= 1e-9:raise RuntimeError('TD reference differs from qualified ground fixed point')
        independent=float(vacuum.energy_tot(dm=ground.state.density_ao(),h1e=h0))
        work_error=abs(independent-ground.vacuum_energy_hartree)
        if work_error >= 1e-10:raise RuntimeError('independent ground-state work correction failed')
        nocc=int(np.count_nonzero(mf.mo_occ > 0))
        if nstates > nocc*(mol.nao-nocc):raise ValueError('too many roots for the declared AO space')
        if response_model == 'TDA-DFT':
            td=tdscf.rks.TDA(mf)
        else:
            # Explicit coupled X/Y solver; never substitute TDA or Casida here.
            td=tdscf.rks.TDDFT(mf)
        td.nstates=nstates;td.singlet=True;td.conv_tol=1e-9;td.max_cycle=80;td.chkfile=None
        attach_optical_response(td,model,operators)
        energies,amplitudes=td.kernel()
        if len(energies)!=nstates or not np.all(td.converged):
            raise RuntimeError('not all requested TDA/TD-DFT roots converged')
        energies=np.asarray(energies,dtype=float)
        if not np.all(np.isfinite(energies)) or np.any(energies <= 0):
            raise RuntimeError('invalid or unstable TDA/TD-DFT excitation roots')
        strengths=np.asarray(td.oscillator_strength(),dtype=float)
        if strengths.shape!=(nstates,) or not np.all(np.isfinite(strengths)) or np.any(strengths < -1e-12):
            raise RuntimeError('invalid oscillator strengths')
        orbo=mf.mo_coeff[:,mf.mo_occ > 0];orbv=mf.mo_coeff[:,mf.mo_occ == 0]
        states=[];arrays={}
        for root,((x,y),energy,strength) in enumerate(zip(amplitudes,energies,strengths,strict=True),1):
            x=np.asarray(x,dtype=float)
            y=np.zeros_like(x) if np.ndim(y)==0 else np.asarray(y,dtype=float)
            norm=2.*float(np.sum(x*x)-np.sum(y*y))
            if not np.isfinite(norm) or abs(norm-1.) > 1e-6:
                raise RuntimeError('TD amplitude metric normalization failed')
            transition=2.*orbo@(x+y)@orbv.T
            arrays[f'root_{root}_transition_density_ao']=transition
            arrays[f'root_{root}_X']=x;arrays[f'root_{root}_Y']=y
            states.append(dict(root=root,excitation_energy_hartree=float(energy),oscillator_strength=float(strength),
                               xy_metric_norm=norm,transition_density_kind='spin_summed_transition_not_state_density'))
        from pyscf.dft import gen_grid,numint
        grid=gen_grid.Grids(mol);grid.level=4;grid.build()
        b=model.conductor.confinement
        scaled=(grid.coords-b.center_bohr)@b.rotation/b.semiaxes_bohr
        outside=np.sum(scaled**2,axis=1)>=1.
        ao=numint.eval_ao(mol,grid.coords[outside])
        leakage=float(np.einsum('pi,ij,pj,p->',ao,ground.state.density_ao(),ao,grid.weights[outside]))
        if leakage >= 1e-5:raise RuntimeError('ground density containment gate failed')
        np.savez(output/'td_response.npz',**arrays,overlap_ao=mf.get_ovlp(),
                 mo_coeff=mf.mo_coeff,mo_energy_hartree=mf.mo_energy,mo_occ=mf.mo_occ,
                 ground_density_ao=mf.make_rdm1(),
                 schema=np.array('matrix.epoc.solvent_td_response_npz.v1'),
                 response_model=np.array(response_model),energy_unit=np.array('hartree'),
                 slow_reference_sha256=np.array(ground.frozen.sha256))
    payload=dict(schema=RESULT_SCHEMA,protocol_id=PROTOCOL_ID,accepted=True,backend='pyscf',
        response_model=response_model,solver_class=type(td).__name__,basis=request['basis'],xc=request['xc'],
        treatment='nonequilibrium_linear_response_absorption',excited_state_density_relaxed=False,
        slow_reference_sha256=ground.frozen.sha256,optical_dielectric=model.optical_dielectric,
        static_dielectric=model.static_dielectric,modal_backend=model.conductor.modal_backend,
        ground_iterations=len(ground.history),ground_density_residual=ground.final_density_residual,
        td_reference_density_residual=residual,ground_work_error_hartree=work_error,
        outside_cavity_ground_electrons=leakage,states=states,ground_history=list(ground.history),
        transition_artifact='td_response.npz',launch_plan_sha256=plan.sha256,
        state_tracking_policy='single geometry root labels; APOC matching required for continuation')
    (output/'result.json').write_text(json.dumps(payload,indent=2)+'\n')
    (output/'frozen_ground.json').write_text(json.dumps(ground.frozen.to_record(),indent=2)+'\n')
    return payload


__all__=['fast_electronic_response','attach_optical_response','run_ellipsoidal_td_protocol']
