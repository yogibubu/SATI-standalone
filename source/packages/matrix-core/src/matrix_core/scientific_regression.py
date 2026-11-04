"""Permanent small LINK/SATI/LOCK regressions and a resumable remote qualification payload."""
from __future__ import annotations

from importlib import resources
import json
from pathlib import Path
import platform
import resource
import socket
import time
from types import SimpleNamespace

from ._scientific_records import fingerprint, read_json, seal
from .atomic_io import atomic_json_write
from .owner_evidence import OwnerEvidenceCapture
from .scientific_artifacts import ScientificArtifactRegistry
from .scientific_campaign import ScientificCampaign, capture_runtime_snapshot
from .scientific_costs import ExecutionTimer
from .scientific_planning import SCIENTIFIC_REQUEST_SCHEMA, build_scientific_plan


def run_regressions(directory, *, previous_report=None, maximum_time_ratio=3., maximum_memory_ratio=2.):
    import numpy as np
    import scipy
    from threadpoolctl import threadpool_info, threadpool_limits
    from matrix_link.sonic_hessian import parallel_sycart_hessian
    from matrix_trinity.vibronic_local_states import solve_local_potential_ground_state
    from matrix_rovib import LAMRotationSurface, write_lam_rotation_surface
    from matrix_rovib.lock_workflow import LOCKWorkflowRequest, run_lock_workflow

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    registry = ScientificArtifactRegistry(directory/'project')
    identity = dict(molecule_id='analytic-regression-models', isotopologue_id='controlled-masses',
        electronic_state='model-ground', method='controlled-analytic', basis='explicit-finite-basis',
        coordinate_frame='fixed-cartesian', geometry_sha256=fingerprint('analytic-geometry-v1'),
        atom_order_sha256=fingerprint(['H', 'H']), units={'energy': 'model-specific-explicit-units'})
    def plan(identifier):
        return build_scientific_plan(dict(schema=SCIENTIFIC_REQUEST_SCHEMA,
            objective='Controlled numerical regression', protocol_id=identifier), library=registry.library)
    captures = {}
    def observer(identifier, label):
        capture = OwnerEvidenceCapture(registry, plan=plan(identifier), protocol_id=identifier,
            run_id='regression-'+label, identity=identity, output_paths={}, observation_only=True)
        captures[label] = capture
        return capture
    reference_path = resources.files('matrix_core').joinpath('data/scientific_regression_reference_v1.json')
    reference = json.loads(reference_path.read_text())
    values = {}
    with threadpool_limits(limits=1), ExecutionTimer() as timer:
        basis = np.zeros((6, 1)); basis[0, 0] = -1/np.sqrt(2); basis[3, 0] = 1/np.sqrt(2)
        class AnalyticCalculator:
            atoms = ('H', 'H')
            nn_enabled = True
            def evaluate_nn_hessian_analytic(self, _coordinates):
                return SimpleNamespace(hessian_hartree_per_bohr2=.5*basis@basis.T)
        with observer('link-sycart-hessian.v1', 'link'):
            result = parallel_sycart_hessian(AnalyticCalculator(), ('H', 'H'),
                np.array([[-.37, 0., 0.], [.37, 0., 0.]]), basis_cartesian_from_sycart=basis, irreps=('A',))
            values['link_sycart_hessian'] = float(result.hessian_hartree_per_coordinate2[0, 0])
        with observer('trinity-native-pes-variational.v1', 'sati'):
            result = solve_local_potential_ground_state([1000.], lambda q: 500*q[:, 0]**2,
                half_width=6., grid_points=39, maximum_total_quanta=16, workers=1)
            values['sati_harmonic_ground'] = float(result.energies_cm[0])
        q = np.arange(9)*2*np.pi/9
        surface = LAMRotationSurface(coordinates=(q,), potential_MHz=np.zeros(9),
            F_MHz=np.full((9, 1, 1), 90.), rho=np.zeros((9, 3, 1)),
            R_MHz=np.broadcast_to(np.diag([1000., 700., 500.]), (9, 3, 3)),
            periodic=(True,), dipole_au=np.broadcast_to([1., .2, .1], (9, 3)))
        surface_path = write_lam_rotation_surface(directory/'surface.npz', surface)
        lock_plan = plan('lock-rovibrational-workflow.v1')
        lock_protocol = lock_plan['selected_protocol_id']
        import matrix_rovib.lock_workflow as lock_module
        runtime = capture_runtime_snapshot(code_files={'lock_workflow': lock_module.__file__},
            backend_files={}, backend_name='LOCK', backend_version='controlled-v1',
            dependencies=['numpy', 'scipy'], configuration={'workers': 1, 'J': [0, 1], 'lam_states': 3})
        input_record = registry.register(surface_path, artifact_schema='matrix.lock.lam_rotation_surface.v1',
            identity=identity, protocol_id=lock_protocol, run_id='analytic-surface', role='input')
        spec = dict(plan=lock_plan, protocol_id=lock_protocol, parameters={'J': [0, 1], 'lam_states': 3},
            input_bindings={'surface': input_record['artifact_id']}, input_identities={'surface': identity},
            output_identity=identity, runtime=runtime)
        campaign = ScientificCampaign(registry)
        run = campaign.prepare(**spec)
        campaign.start(run['run_id'])
        with ExecutionTimer() as lock_timer:
            with OwnerEvidenceCapture(registry, plan=lock_plan, protocol_id=lock_protocol,
                    run_id=run['run_id'], identity=identity,
                    output_paths={'workflow_manifest': directory/'lock/lock_workflow.json'}) as capture:
                result = run_lock_workflow(LOCKWorkflowRequest(surface_path, directory/'lock', j_min=0, j_max=1,
                    lam_states=3, levels_per_j=9, workers=1, temperature_K=2., assembly_backend='python'))
        levels = read_json(result.levels)
        values['lock_free_rotor_ground'] = float(levels['blocks']['0'][0]['energy_MHz'])
        values['lock_rigid_j1_first'] = float(levels['blocks']['1'][0]['energy_MHz'])
        campaign.complete(run['run_id'], plan=lock_plan, outputs=capture.outputs,
            evidence_ids=capture.evidence_ids, expected_identities={lock_protocol: identity},
            measurements=lock_timer.measurements(work_units=1, unit='controlled LOCK workflow', workload={'J': [0, 1], 'lam_states': 3}))
        reuse = campaign.reuse_decision(run['run_id'], **spec)
    checks = {name: dict(value=values[name], expected=target['value'], unit=target['unit'],
                         absolute_error=abs(values[name]-target['value']),
                         tolerance=target['absolute_tolerance'],
                         passed=abs(values[name]-target['value']) <= target['absolute_tolerance'])
              for name, target in reference['cases'].items()}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if platform.system() == 'Darwin' else peak*1024)
    environment = dict(host=socket.gethostname(), system=platform.system(), architecture=platform.machine(),
                       python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, workers=1,
                       numerical_libraries=[{key: item.get(key) for key in ('internal_api', 'version', 'threading_layer')}
                                            for item in threadpool_info()])
    performance = dict(status='baseline_recorded', elapsed_seconds=timer.elapsed_seconds,
                       process_peak_memory_bytes=peak_bytes, scope='whole benchmark process high-water memory; elapsed excludes imports')
    if previous_report is not None:
        previous = read_json(previous_report)
        if previous['environment'] != environment:
            performance['status'] = 'incompatible_environment'
        else:
            performance.update(status='passed', time_ratio=timer.elapsed_seconds/previous['performance']['elapsed_seconds'],
                memory_ratio=peak_bytes/previous['performance']['process_peak_memory_bytes'],
                maximum_time_ratio=maximum_time_ratio, maximum_memory_ratio=maximum_memory_ratio)
            if performance['time_ratio'] > maximum_time_ratio or performance['memory_ratio'] > maximum_memory_ratio:
                performance['status'] = 'failed'
    report = seal(dict(schema='matrix.keymaker.scientific_regression.v1', environment=environment,
        reference_sha256=fingerprint(reference), checks=checks, performance=performance,
        passed=all(check['passed'] for check in checks.values()) and reuse['reusable'] and
            performance['status'] not in {'failed', 'incompatible_environment'},
        owner_observation_ids={label: c.observation_ids for label, c in captures.items()},
        lock_owner_evidence_ids=capture.evidence_ids, reuse=reuse, scope=reference['scope']), 'report_sha256')
    atomic_json_write(directory/'benchmark.json', report, allow_nan=False)
    return report


def main(argv=None):
    import argparse
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--previous-report', type=Path)
    parser.add_argument('--request', type=Path, default=os.environ.get('MATRIX_KEYMAKER_REQUEST'),
                        help='Authorized remote validation request')
    args = parser.parse_args(argv)
    if args.request:
        request = read_json(args.request)
        output = Path(request['expected_result'])
        # An explicit pause provides a controlled monitor-disconnect window;
        # it is not part of the scientific benchmark or its measured duration.
        delay = float(request.get('recovery_window_seconds', 0.))
        if not 0 <= delay <= 30:
            raise ValueError('recovery window must be between 0 and 30 seconds')
        time.sleep(delay)
        result = run_regressions(output.parent/'regression')
        atomic_json_write(output, result, allow_nan=False)
    else:
        if args.output is None:
            parser.error('--output or --request is required')
        result = run_regressions(args.output, previous_report=args.previous_report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
