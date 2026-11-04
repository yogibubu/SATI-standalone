"""Method-independent electronic/continuum iteration and KEYMAKER requests.

The installed JSON runner supports PySCF RHF/RKS ground states. Other owners
can use iterate_solvation with their density/energy/state-continuity adapter.
FCI is not a requirement and is isolated in the reference qualification module.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import json
import math

import numpy as np

from .cpcm import CPCMConfinement
from .nonequilibrium import NonequilibriumCPCM

REQUEST_SCHEMA = 'matrix.epoc.solvent_scf_request.v1'
RESULT_SCHEMA = 'matrix.epoc.solvent_scf_result.v1'
PROTOCOL_ID = 'epoc-ellipsoidal-solvent-scf.v1'


@dataclass(frozen=True)
class ElectronicSolvationStep:
    """An owner's converged electronic solution at a fixed reaction operator.

    Energy is <H_vac> plus nuclear repulsion, without reaction-potential work.
    Excited owners certify continuity and supply an overlap at each step.
    """
    state: object
    vacuum_energy_hartree: float
    converged: bool
    continuity_certified: bool = False
    state_overlap: float | None = None


@dataclass(frozen=True)
class SolventSCFResult:
    state: object
    source: object
    frozen: object
    vacuum_energy_hartree: float
    free_energy_hartree: float
    final_density_residual: float
    history: tuple[dict, ...]


def iterate_solvation(model, operators, initial_state, electronic_solver, *, frozen=None,
                       excited_state=False, maximum_iterations=240):
    """Converge state/solvent response without imposing an electronic method.

    electronic_solver(V_ao, previous_state) -> ElectronicSolvationStep.
    A supplied frozen field stays unchanged; otherwise solvent is equilibrated.
    Numerical gates are fixed in protocol v1. Failures never return success.
    """
    from matrix_apoc import apoc_solvation_source, solvation_fock_ao
    if not callable(electronic_solver):
        raise TypeError('an electronic-owner callback is required')
    if isinstance(maximum_iterations, bool) or not isinstance(maximum_iterations, int) or not 2 <= maximum_iterations <= 240:
        raise ValueError('iteration budget must lie between 2 and 240')
    if frozen is not None:
        model.validate_frozen(frozen)
    state = initial_state
    state_id = state.label
    previous_energy = None
    history = []

    def checked_step(potential, previous):
        step = electronic_solver(potential, previous)
        if (not isinstance(step, ElectronicSolvationStep) or step.converged is not True
                or isinstance(step.vacuum_energy_hartree, bool)
                or not math.isfinite(step.vacuum_energy_hartree)):
            raise RuntimeError('electronic owner did not return a converged state and vacuum expectation')
        if step.state.label != state_id:
            raise RuntimeError('electronic state identity changed')
        if excited_state and (step.continuity_certified is not True or step.state_overlap is None
                              or not math.isfinite(step.state_overlap) or not .9 <= step.state_overlap <= 1.00000001):
            raise RuntimeError('excited electronic owner did not certify state continuity')
        # Checks geometry, AO frame, electron count and integral binding.
        src = apoc_solvation_source(model, step.state, operators)
        return step, src

    for iteration in range(1, maximum_iterations+1):
        current = apoc_solvation_source(model, state, operators)
        slow = model.freeze(current) if frozen is None else frozen
        potential = solvation_fock_ao(model, current, slow, operators)
        step, src = checked_step(potential, state)
        target_slow = model.freeze(src) if frozen is None else frozen
        free = step.vacuum_energy_hartree + model.evaluate(src, target_slow).energy_hartree
        if len(step.state.mo_coefficients) != len(state.mo_coefficients):
            raise RuntimeError('electronic owner changed the number of spin channels')
        new_dm = tuple(step.state.density_ao(i) for i in range(len(state.mo_coefficients)))
        old_dm = tuple(state.density_ao(i) for i in range(len(state.mo_coefficients)))
        residual = max(float(np.max(np.abs(a-b))) for a,b in zip(new_dm,old_dm,strict=True))
        change = None if previous_energy is None else abs(free-previous_energy)
        history.append(dict(iteration=iteration, density_residual_max=residual,
                            free_energy_hartree=free, energy_change_hartree=change,
                            state_overlap=step.state_overlap))
        if residual < 1e-9 and change is not None and change < 1e-11:
            final_potential = solvation_fock_ao(model, src, target_slow, operators)
            check, _ = checked_step(final_potential, step.state)
            final_residual = max(float(np.max(np.abs(check.state.density_ao(i)-d))) for i,d in enumerate(new_dm))
            if final_residual >= 1e-9 or abs(check.vacuum_energy_hartree-step.vacuum_energy_hartree) >= 1e-10:
                raise RuntimeError('independent final solvent fixed-point reevaluation failed')
            return SolventSCFResult(step.state, src, target_slow, step.vacuum_energy_hartree,
                                   free, final_residual, tuple(history))
        mixed = tuple(.6*a+.4*b for a,b in zip(new_dm, old_dm, strict=True))
        state = replace(step.state, density_matrices_ao=mixed)
        previous_energy = free
    raise RuntimeError('electronic/solvent response did not converge within the declared budget')


def _ao_integrals(model, molecule, state):
    """Analytic AO modal integrals through degree two, including rotated cavities."""
    from matrix_apoc import ApocSolvationIntegrals
    cavity = model.conductor.confinement
    with molecule.with_common_orig((0.,0.,0.)):
        overlap = molecule.intor_symmetric('int1e_ovlp')
        first = molecule.intor('int1e_r', comp=3)
        second = molecule.intor('int1e_rr', comp=9).reshape(3,3,molecule.nao,molecule.nao)
    center, rotation = cavity.center_bohr, cavity.rotation
    shifted_first = first-center[:,None,None]*overlap
    shifted_second = (second-center[:,None,None,None]*first[None,:,:,:]
                       -center[None,:,None,None]*first[:,None,:,:]
                       +np.outer(center,center)[:,:,None,None]*overlap)
    local_first = np.einsum('ai,amn->imn', rotation, shifted_first)
    local_second = np.einsum('ai,bj,abmn->ijmn', rotation, rotation, shifted_second)
    values = np.zeros((len(model.conductor.modes), molecule.nao, molecule.nao))
    for m, mode in enumerate(model.conductor.modes):
        for powers, coefficient in zip(mode.exponents,mode.coefficients,strict=True):
            axes = [i for i,n in enumerate(powers) for _ in range(int(n))]
            if len(axes) == 0:
                integral = overlap
            elif len(axes) == 1:
                integral = local_first[axes[0]]
            elif len(axes) == 2:
                integral = local_second[axes[0],axes[1]]
            else:
                raise ValueError('installed analytic AO adapter supports spectral degree <=2')
            values[m] += coefficient*integral/np.prod(cavity.semiaxes_bohr**powers)
    return ApocSolvationIntegrals.bind(model,state,values,
        provenance='PySCF analytic Gaussian moments through degree 2; explicit translated/rotated EPOC frame')


def run_ellipsoidal_scf_protocol(request_path):
    """KEYMAKER JSON entry: configurable RHF/RKS, basis and fixed ellipsoid.

    Excited-state electronic relaxation requires an explicit owner adapter via
    iterate_solvation; no transition-density approximation is silently used.
    """
    from matrix_core import (CalculationResources, build_calculation_launch_plan,
                            require_calculation_launch_or_parent_authorization)
    request_path = Path(request_path).expanduser().resolve()
    request = json.loads(request_path.read_text())
    required = {'schema','backend','electronic_method','basis','mode','molecule','cavity',
                'dielectrics','output_directory'}
    if (not isinstance(request,dict) or not required <= set(request)
            or set(request)-required-{'xc'} or request['schema'] != REQUEST_SCHEMA):
        raise ValueError('invalid EPOC solvent SCF request fields/schema')
    method = request['electronic_method']
    if request['backend'] != 'pyscf' or method not in {'RHF','RKS'}:
        raise ValueError('installed JSON adapter supports PySCF RHF/RKS; other methods need an electronic-owner adapter')
    if request['mode'] != 'equilibrium_ground_state':
        raise ValueError('excited relaxation requires a state-continuity-certified electronic owner via iterate_solvation')
    if not isinstance(request['basis'],str) or not request['basis'].strip():
        raise ValueError('an explicit Gaussian basis is required')
    if (method == 'RKS' and (not isinstance(request.get('xc'),str) or not request['xc'].strip())) or (method == 'RHF' and 'xc' in request):
        raise ValueError('an XC functional is required for RKS only')
    molecule = request['molecule']
    if (not isinstance(molecule,dict) or set(molecule) != {'atoms','unit','charge','spin'}
            or molecule['unit'] not in {'angstrom','bohr'} or molecule['spin'] != 0
            or isinstance(molecule['charge'],bool) or not isinstance(molecule['charge'],int)):
        raise ValueError('explicit atoms, units, integer charge and closed-shell spin=0 required')
    cavity = request['cavity']
    if not isinstance(cavity,dict) or set(cavity) != {'center_bohr','semiaxes_bohr','rotation'}:
        raise ValueError('explicit frozen cavity geometry required')
    dielectrics = request['dielectrics']
    if not isinstance(dielectrics,dict) or set(dielectrics) != {'static','optical'}:
        raise ValueError('explicit static and optical dielectrics required')
    model = NonequilibriumCPCM.compile(CPCMConfinement(**cavity,maximum_degree=2),
        static_dielectric=dielectrics['static'],optical_dielectric=dielectrics['optical'])
    destination = request['output_directory']
    if not isinstance(destination,str) or not destination.strip():
        raise ValueError('an explicit new output directory is required')
    output = Path(destination).expanduser()
    if not output.is_absolute():
        output = request_path.parent/output
    output = output.resolve()
    if output.exists():
        raise FileExistsError('solvent SCF output directory already exists')
    plan = build_calculation_launch_plan(backend='PySCF/EPOC-ellipsoidal-solvent-scf-v1',
        input_path=request_path,command=('keymaker','--epoc-solvent-scf',str(request_path)),workdir=output,
        resources=CalculationResources(process_count=1,threads_per_process=1,memory_per_job_gb=1.,concurrent_jobs=1))
    require_calculation_launch_or_parent_authorization(plan,None)
    from pyscf import gto,scf,dft,lib
    from threadpoolctl import threadpool_limits
    from matrix_apoc import electronic_state_from_pyscf
    output.mkdir(parents=True,exist_ok=False)
    with threadpool_limits(limits=1):
        lib.num_threads(1)
        mol = gto.M(atom=molecule['atoms'],unit=molecule['unit'],charge=molecule['charge'],
                    spin=0,basis=request['basis'],verbose=0,max_memory=512)
        if mol.nao > 80 or mol.nelectron % 2:
            raise ValueError('v1 resource envelope supports closed-shell systems with <=80 AOs')
        factory = scf.RHF if method == 'RHF' else dft.RKS
        def configure():
            engine = factory(mol)
            if method == 'RKS':
                engine.xc = request['xc']; engine.grids.level = 4
            engine.conv_tol = 1e-12; engine.conv_tol_grad = 1e-9
            engine.max_cycle = 80; engine.chkfile = None
            return engine
        vacuum = configure(); vacuum.kernel()
        if not vacuum.converged:
            raise RuntimeError('vacuum electronic calculation did not converge')
        initial = electronic_state_from_pyscf(vacuum)
        operators = _ao_integrals(model,mol,initial)
        h0 = vacuum.get_hcore()
        def solve(potential,previous):
            engine = configure(); engine.get_hcore = lambda *args: h0+potential
            engine.kernel(dm0=previous.density_ao())
            density = engine.make_rdm1()
            work = float(np.einsum('ij,ji->',density,potential))
            state = electronic_state_from_pyscf(engine)
            return ElectronicSolvationStep(state,float(engine.e_tot)-work,bool(engine.converged))
        result = iterate_solvation(model,operators,initial,solve)
        independent = float(vacuum.energy_tot(dm=result.state.density_ao(),h1e=h0))
        error = abs(independent-result.vacuum_energy_hartree)
        if error >= 1e-10:
            raise RuntimeError('independent vacuum energy/work correction failed')
        from pyscf.dft import gen_grid,numint
        grid = gen_grid.Grids(mol); grid.level = 4; grid.build()
        boundary = model.conductor.confinement
        local = (grid.coords-boundary.center_bohr)@boundary.rotation/boundary.semiaxes_bohr
        outside = np.sum(local**2,axis=1) >= 1.
        ao = numint.eval_ao(mol,grid.coords[outside])
        outside_electrons = float(np.einsum('pi,ij,pj,p->',ao,result.state.density_ao(),ao,grid.weights[outside]))
        if outside_electrons >= 1e-5:
            raise RuntimeError('electronic density outside cavity exceeds 1e-5 electrons')
    payload = dict(schema=RESULT_SCHEMA,protocol_id=PROTOCOL_ID,accepted=True,
        backend='pyscf',electronic_method=method,basis=request['basis'],xc=request.get('xc'),
        mode=request['mode'],vacuum_energy_hartree=float(vacuum.e_tot),
        vacuum_expectation_hartree=result.vacuum_energy_hartree,
        free_energy_hartree=result.free_energy_hartree,iterations=len(result.history),
        final_density_residual=result.final_density_residual,
        independent_vacuum_energy_error_hartree=error,outside_cavity_electrons=outside_electrons,
        modal_backend=model.conductor.modal_backend,history=list(result.history),
        frozen_reference_sha256=result.frozen.sha256,launch_plan_sha256=plan.sha256)
    (output/'result.json').write_text(json.dumps(payload,indent=2)+'\n')
    (output/'frozen_ground.json').write_text(json.dumps(result.frozen.to_record(),indent=2)+'\n')
    return payload


__all__ = ['ElectronicSolvationStep','SolventSCFResult','iterate_solvation','run_ellipsoidal_scf_protocol']
