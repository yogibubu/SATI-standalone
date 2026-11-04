"""Execute a frozen, finite constrained Gaussian Opt/Freq acquisition.

The plan is data, including atoms, physical constraints and permitted hosts.
Completed jobs are reused. Incomplete checkpoints are preserved and never
implicitly restarted. This owner cannot add points or change a QM route.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import fcntl
import hashlib
import inspect
import json
import os
import re
import subprocess
import traceback

import numpy as np
from matrix_gaussian import read_gaussian_fchk, lower_to_symmetric
from matrix_gaussian.writers import write_gaussian_point_input
from matrix_gaussian.route_policy import validate_gaussian_route_policy
from matrix_smith.coordinate_kernel.primitives import Primitive, eval_primitives
from matrix_smith.coordinate_kernel.pipeline import b_matrix_analytic

BOHR = .529177210903


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, payload):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def acquire(root, point, plan, plan_digest, *, collect_only=False, use_actual_constraints=False):
    if use_actual_constraints and not collect_only:
        raise ValueError('Actual-coordinate collection cannot launch electronic jobs')
    folder = root / 'points' / point['id']
    folder.mkdir(parents=True, exist_ok=True)
    input_path = folder / 'input.gjf'
    log = folder / 'output.log'
    result_path = folder / 'result.json'
    primitives = [Primitive(c['kind'], tuple(c['atoms_zero_based'])) for c in plan['constraints']]
    x = np.asarray(point['geometry_angstrom'], float)
    expected_constraints = eval_primitives(primitives, x)
    request = dict(plan_sha256=plan_digest, point=point,
        host=os.uname().nodename, driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    request_path = folder / 'request.json'
    if collect_only:
        if not request_path.exists() or not log.exists():
            raise RuntimeError('Collection requires an existing acquisition request and log')
        original=json.loads(request_path.read_text())
        if original['plan_sha256']!=plan_digest or original['point']!=point:
            raise ValueError('Collection does not match the immutable acquisition request')
        request['mode']='collect_at_actual_constraints' if use_actual_constraints else 'collect_only'
        request['original_request_sha256']=hashlib.sha256(request_path.read_bytes()).hexdigest()
        request_path=folder/('actual_collection_request.json' if use_actual_constraints else 'collection_request.json')
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError('Immutable point acquisition request changed')
    save(request_path, request)
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get('stage') == 'complete':
            for filename, digest in result['sha256'].items():
                if hashlib.sha256((folder / filename).read_bytes()).hexdigest() != digest:
                    raise ValueError('Completed source file changed: ' + filename)
            return result
    gdv = Path(plan['gdv_directory'])
    env = dict(os.environ, GAUSS_EXEDIR=f'{gdv}/bsd:{gdv}', GDVBASIS=str(gdv/'basis'),
        GAUSS_SCRDIR=str(folder/'scratch'), GAUSS_PDEF=str(plan['cores_per_job']),
        OMP_NUM_THREADS=str(plan['cores_per_job']))
    env['PATH'] = f'{gdv}:{gdv}/bsd:' + env.get('PATH', '')
    (folder/'scratch').mkdir(exist_ok=True)
    if not log.exists():
        if collect_only:
            raise RuntimeError('Collection mode cannot launch a missing electronic job')
        if (folder/'state.chk').exists():
            raise RuntimeError('Existing checkpoint without log retained; explicit resume required')
        freezes = ['D ' + ' '.join(str(i+1) for i in c['atoms_zero_based']) + ' F'
            for c in plan['constraints']]
        old_checkpoint=point.get('source_checkpoint')
        restart_link=()
        if old_checkpoint is not None:
            old_path=Path(old_checkpoint).resolve()
            if hashlib.sha256(old_path.read_bytes()).hexdigest()!=point.get('source_checkpoint_sha256'):
                raise ValueError('Declared starting checkpoint changed')
            restart_link=(f'%oldchk={old_path}',)
        write_gaussian_point_input(input_path, plan['atoms'], x,
            route=plan['method_route'], ensure_force=False,
            title=f"Finite constrained acquisition {point['id']} {point['role']}",
            charge=plan['charge'], multiplicity=plan['multiplicity'],
            link0=restart_link+('%chk=state.chk', f"%mem={plan['memory_GB_per_job']}GB",
                f"%nprocshared={plan['cores_per_job']}"),
            basis_set_file=root/plan['basis_filename'], modredundant_lines=freezes)
        with (folder/'launch.log').open('w') as output:
            process = subprocess.Popen([str(gdv/'bsd/gdvtest'), '-p', str(plan['cores_per_job']),
                'input.gjf', 'output.log'], cwd=folder, env=env,
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
            save(folder/'process.json', dict(pid=process.pid, host=os.uname().nodename, started_utc=now()))
            exit_code = process.wait()
        if exit_code:
            raise RuntimeError(f'Gaussian exited {exit_code}; checkpoint retained')
    text = log.read_text(errors='replace')
    if 'Normal termination of Gaussian' not in text[-8000:]:
        raise RuntimeError('Incomplete electronic job retained; no implicit rerun')
    if 'Optimization completed' not in text:
        raise ValueError('No completed constrained optimization in Gaussian log')
    fchk = folder/'state.fchk'
    if not fchk.exists():
        with (folder/'formchk.log').open('w') as output:
            subprocess.run([str(gdv/'formchk'), 'state.chk', 'state.fchk'], cwd=folder,
                env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
    field = read_gaussian_fchk(fchk)
    if list(field.atomic_numbers) != plan['atomic_numbers']:
        raise ValueError('Electronic source atom identity/order changed')
    y = field.cartesian_coordinates_bohr * BOHR
    actual_constraints = eval_primitives(primitives, y)
    error = float(np.max(abs(np.angle(np.exp(1j*(actual_constraints-expected_constraints))))))
    if error > 2e-5 and not use_actual_constraints:
        raise ValueError(f'Constrained torsions changed by {error} radian')
    hessian = lower_to_symmetric(field.cartesian_hessian_lower)
    gradient = np.asarray(field.cartesian_gradient_hartree_per_bohr).reshape(-1)
    dipole_derivatives = np.asarray(field.dipole_derivatives_au_per_bohr)
    if hessian.shape != (3*len(y),3*len(y)) or dipole_derivatives.size != 9*len(y):
        raise ValueError('Missing complete Hessian or dipole derivatives')
    B = b_matrix_analytic(primitives, y) * BOHR
    multipliers = np.linalg.lstsq(B.T, gradient, rcond=None)[0]
    projected = gradient - B.T @ multipliers
    payload = dict(geometry_angstrom=y, atomic_numbers=field.atomic_numbers,
        masses_amu=field.masses_amu, energy_hartree=np.asarray(field.total_energy_hartree),
        gradient_hartree_per_bohr=gradient, hessian_hartree_per_bohr2=hessian,
        dipole_au=field.dipole_moment_au, dipole_derivatives_au_per_bohr=dipole_derivatives,
        q_radian=np.angle(np.exp(1j*(np.asarray(point['q_radian'])+actual_constraints-expected_constraints))),
        requested_q_radian=np.asarray(point['q_radian']), constraint_values_radian=actual_constraints,
        constraint_multipliers_hartree=multipliers,
        projected_gradient_hartree_per_bohr=projected)
    if any(not np.all(np.isfinite(v)) for v in payload.values()):
        raise ValueError('Nonfinite acquired property')
    np.savez_compressed(folder/'source_payload.npz', **payload)
    result = dict(stage='complete', id=point['id'], role=point['role'],
        plan_sha256=plan_digest, completed_utc=now(), energy_hartree=float(field.total_energy_hartree),
        constraint_error_radian=error, maximum_projected_gradient_hartree_per_bohr=float(abs(projected).max()),
        constraint_target_within_tolerance=error<=2e-5,
        coordinate_policy='Actual geometry and actual torsions; requested targets retained separately',
        sha256={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
            for name in ('input.gjf','output.log','state.chk','state.fchk','source_payload.npz')})
    save(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--collect-only', action='store_true',
        help='Collect existing logs/checkpoints with separate reader provenance; never launch QM')
    parser.add_argument('--actual-constraints', action='store_true',
        help='In collection-only mode retain actual angular coordinates, flag target drift without mislabelling the field')
    args = parser.parse_args(); root = args.root.resolve()
    if args.actual_constraints and not args.collect_only:
        raise ValueError('Actual-coordinate collection requires --collect-only')
    raw = (root/'acquisition_plan.json').read_bytes(); plan = json.loads(raw)
    if plan['schema'] != 'matrix.gaussian.finite_constrained_acquisition.v1':
        raise ValueError('Unsupported acquisition plan')
    if os.uname().nodename not in plan['allowed_hostnames']:
        raise RuntimeError('Host identity is not authorized for this electronic plan')
    if len(plan['points']) != plan['total_points'] or len({p['id'] for p in plan['points']}) != len(plan['points']):
        raise ValueError('Frozen electronic budget or point IDs invalid')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+',p['id']) for p in plan['points']):
        raise ValueError('Point ID must be a simple directory name')
    if any(c['kind'] != 'dihedral' or len(c['atoms_zero_based']) != 4 for c in plan['constraints']):
        raise ValueError('This finite acquisition owner accepts torsional constraints')
    validate_gaussian_route_policy(plan['method_route'])
    lock = (root/'acquisition.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    digest = hashlib.sha256(raw).hexdigest()
    if (root/'plan_sha256.txt').read_text().strip() != digest:
        raise ValueError('Frozen source plan hash changed')
    request=dict(plan_sha256=digest,
        basis_sha256=hashlib.sha256((root/plan['basis_filename']).read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        writer_sha256=hashlib.sha256(Path(inspect.getsourcefile(write_gaussian_point_input)).read_bytes()).hexdigest())
    request_file=root/('actual_collection_request.json' if args.actual_constraints else 'collection_request.json' if args.collect_only else 'acquisition_request.json')
    if args.collect_only:
        request['mode']='collect_at_actual_constraints' if args.actual_constraints else 'collect_only'
        original=root/'acquisition_request.json'
        request['original_acquisition_request_sha256']=hashlib.sha256(original.read_bytes()).hexdigest()
    if request_file.exists() and json.loads(request_file.read_text())!=request:
        raise ValueError('Immutable acquisition runtime, basis or plan changed')
    save(request_file,request)
    records = {}
    def status(stage):
        save(root/'status.json',dict(stage=stage,host=os.uname().nodename,pid=os.getpid(),
            updated_utc=now(),budget=plan['total_points'],workers=plan['workers'],
            plan_sha256=digest,completed=sum(v['stage']=='complete' for v in records.values()),
            failed=sum(v['stage']=='failed' for v in records.values()),
            completed_ids=[k for k,v in records.items() if v['stage']=='complete']))
    status('acquiring')
    with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
        futures={pool.submit(acquire,root,p,plan,digest,collect_only=args.collect_only,use_actual_constraints=args.actual_constraints):p for p in plan['points']}
        for future in as_completed(futures):
            point=futures[future]
            try:
                records[point['id']]=future.result()
            except Exception as exc:
                records[point['id']]=dict(stage='failed',error=str(exc),traceback=traceback.format_exc())
                save(root/'points'/point['id']/'failure.json',records[point['id']])
            save(root/'point_results.json',records);status('acquiring')
            print(now(),point['id'],records[point['id']]['stage'],flush=True)
    failed=any(v['stage']=='failed' for v in records.values())
    status('acquisition_failed' if failed else 'acquisition_complete')
    if failed:
        raise RuntimeError('Finite source acquisition has failed points; checkpoints retained')


if __name__=='__main__':
    main()
