"""Empirical cost calibration from completed, evidence-qualified campaign runs."""
from __future__ import annotations

from copy import deepcopy
from statistics import median
import time

from ._scientific_records import fields, fingerprint, number, seal, text, verify
from .scientific_campaign import RUNTIME_SNAPSHOT_SCHEMA
from .scientific_evidence import evaluate_scientific_plan
from .scientific_planning import build_scientific_plan

COST_CALIBRATION_SCHEMA = 'matrix.keymaker.cost_calibration.v1'
CALIBRATED_PLAN_SCHEMA = 'matrix.keymaker.calibrated_plan.v1'


def validate_measurements(measurements):
    fields(measurements, ('elapsed_seconds', 'cpu_seconds', 'peak_memory_bytes',
                          'work_units', 'unit', 'workload', 'measurement_scope'))
    number(measurements['elapsed_seconds'], 'elapsed_seconds', minimum=1e-300)
    if measurements['cpu_seconds'] is not None:
        number(measurements['cpu_seconds'], 'cpu_seconds')
    peak = measurements['peak_memory_bytes']
    if peak is not None and (isinstance(peak, bool) or not isinstance(peak, int) or peak < 1):
        raise ValueError('peak_memory_bytes must be null or a positive integer')
    count = measurements['work_units']
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('work_units must be a positive integer')
    text(measurements['unit'], 'work unit')
    text(measurements['measurement_scope'], 'measurement_scope')
    if not isinstance(measurements['workload'], dict) or not measurements['workload']:
        raise ValueError('an explicit workload descriptor is required')
    fingerprint(measurements)
    return deepcopy(measurements)


class ExecutionTimer:
    """Opt-in instrumentation around an already authorized caller execution.

    Wall time includes the enclosed operation. CPU is this Python process only,
    not its external jobs. Peak memory stays unknown unless the owner measures it.
    The timer does not call a backend or mark the run scientifically successful.
    """
    def __enter__(self):
        self.wall_start = time.perf_counter()
        self.cpu_start = time.process_time()
        self.finished = False
        return self

    def __exit__(self, *_):
        self.elapsed_seconds = time.perf_counter()-self.wall_start
        self.cpu_seconds = time.process_time()-self.cpu_start
        self.finished = True

    def measurements(self, *, work_units, unit, workload, peak_memory_bytes=None):
        if not getattr(self, 'finished', False):
            raise ValueError('the execution interval has not finished')
        return validate_measurements(dict(elapsed_seconds=self.elapsed_seconds,
                                         cpu_seconds=self.cpu_seconds,
                                         peak_memory_bytes=peak_memory_bytes, work_units=work_units,
                                         unit=unit, workload=workload,
                                         measurement_scope='enclosed wall time; CPU of calling Python process only'))


def calibrate_cost(campaign, *, protocol_id, runtime, workload, work_units, unit,
                   measurement_scope, minimum_samples=3):
    manifest = campaign.registry.library.resolve(protocol_id)
    if manifest.protocol_id != protocol_id or manifest.status != 'approved':
        raise ValueError('an exact approved protocol ID is required')
    verify(runtime, 'runtime_sha256', RUNTIME_SNAPSHOT_SCHEMA)
    if not isinstance(workload, dict) or not workload:
        raise ValueError('a workload descriptor is required')
    if isinstance(work_units, bool) or not isinstance(work_units, int) or work_units < 1:
        raise ValueError('work_units must be positive integer')
    if isinstance(minimum_samples, bool) or not isinstance(minimum_samples, int) or minimum_samples < 2:
        raise ValueError('minimum_samples must be an integer >= 2')
    text(unit, 'work unit')
    text(measurement_scope, 'measurement_scope')
    samples, excluded = [], []
    for run in campaign.runs():
        if run['spec']['protocol_id'] != protocol_id or run['status'] != 'completed':
            continue
        measure = validate_measurements(run['measurements'])
        if (run['spec']['protocol_sha256'] != manifest.sha256
                or run['spec']['runtime']['runtime_sha256'] != runtime['runtime_sha256']
                or fingerprint(measure['workload']) != fingerprint(workload) or measure['unit'] != unit
                or measure['measurement_scope'] != measurement_scope):
            excluded.append(dict(run_id=run['run_id'], reason='Protocol, runtime, workload or work unit differs'))
            continue
        try:
            for identifier in run['outputs'].values():
                campaign.registry.get(identifier, expected_identity=run['spec']['output_identity'])
            review = evaluate_scientific_plan(run['plan'], campaign.registry, evidence_ids=run['evidence_ids'],
                                               expected_identities=run['review']['expected_identities'])
            own = [c for c in review['criteria'] if c['protocol_id'] == protocol_id]
            if not own or any(c['status'] != 'passed' for c in own):
                raise ValueError('scientific gate evidence no longer passes')
        except (OSError, KeyError, ValueError) as exc:
            excluded.append(dict(run_id=run['run_id'], reason=str(exc)))
            continue
        samples.append(dict(run_id=run['run_id'], record_sha256=run['record_sha256'],
                            seconds_per_unit=measure['elapsed_seconds']/measure['work_units'],
                            peak_memory_bytes=measure['peak_memory_bytes'],
                            measurement_scope=measure['measurement_scope']))
    qualified = len(samples) >= minimum_samples
    rates = [s['seconds_per_unit'] for s in samples]
    memory = [s['peak_memory_bytes'] for s in samples if s['peak_memory_bytes'] is not None]
    return seal(dict(schema=COST_CALIBRATION_SCHEMA, protocol_id=protocol_id,
                     protocol_sha256=manifest.sha256, runtime_sha256=runtime['runtime_sha256'],
                     workload=deepcopy(workload), unit=unit, work_units=work_units,
                     measurement_scope=measurement_scope,
                     status='empirical_estimate' if qualified else 'insufficient_samples',
                     minimum_samples=minimum_samples, samples=samples, excluded=excluded,
                     seconds_per_unit=median(rates) if qualified else None,
                     estimated_seconds=number(median(rates)*work_units, 'estimated_seconds') if qualified else None,
                     observed_range_seconds=[number(min(rates)*work_units, 'lower estimate'),
                                             number(max(rates)*work_units, 'upper estimate')] if qualified else None,
                     observed_peak_memory_bytes=max(memory) if memory else None,
                     peak_memory_coverage=len(memory),
                     assumptions='Linear scaling in declared work units; exact runtime/workload match. '
                                 'Observed range is not a confidence interval or a bound on future jobs. '
                                 'Memory is an observed maximum, not a forecast.',
                     execution_authorized=False), 'calibration_sha256')


def build_calibrated_plan(request, campaign, *, workloads, minimum_samples=3):
    """Feed measured calibrations into the existing deterministic planner.

    Explicit caller cost estimates are retained. Every automatically populated
    estimate refers to a complete calibration report containing source run IDs.
    """
    request = deepcopy(request)
    original = build_scientific_plan(request, library=campaign.registry.library)
    if not original['selected_protocol_id']:
        return seal(dict(schema=CALIBRATED_PLAN_SCHEMA, plan=original, calibrations=[]), 'bundle_sha256')
    protocols = {s['protocol_id'] for s in original['steps']}
    if not isinstance(workloads, dict) or set(workloads)-protocols:
        raise ValueError('workload specifications must refer to exact planned protocol IDs')
    calibrations = []
    estimates = request.setdefault('cost_estimates', {})
    for protocol, parameters in workloads.items():
        if protocol in estimates:
            continue
        fields(parameters, ('runtime', 'workload', 'work_units', 'unit', 'measurement_scope'))
        result = calibrate_cost(campaign, protocol_id=protocol, minimum_samples=minimum_samples, **parameters)
        calibrations.append(result)
        if result['status'] == 'empirical_estimate':
            estimates[protocol] = dict(work_units=result['work_units'], unit=result['unit'],
                                       seconds_per_unit=result['seconds_per_unit'],
                                       calibration_reference='keymaker-calibration:'+result['calibration_sha256'])
    plan = build_scientific_plan(request, library=campaign.registry.library)
    return seal(dict(schema=CALIBRATED_PLAN_SCHEMA, plan=plan, calibrations=calibrations), 'bundle_sha256')
