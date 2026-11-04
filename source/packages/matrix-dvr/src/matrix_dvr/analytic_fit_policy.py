"""Immutable finite-acquisition contract for the registered analytic DVR owner.

Keeping this small contract separate lets mathematical surface/jet owners load
without importing the request dispatchers for unrelated analytic families.
The protocol remains fail-closed against the installed KeyMaker library.
"""
import hashlib
import json

POLICY_ID = 'dvr-sparse-analytic-fit.v1'
POLICY = {
    'protocol_id': POLICY_ID,
    'change_policy': 'new_version_and_explicit_user_approval',
    'sequence': ['freeze_finite_sample_budget', 'acquire_qualified_samples',
                 'fit_appropriate_analytic_function', 'validate_fit',
                 'generate_dense_grid_from_function_only', 'converge_DVR_without_QM'],
    'automatic_QM_midpoints': False,
    'QM_evaluations_on_dense_DVR_grid': False,
    'DVR_refinement_may_launch_QM': False,
}
POLICY_SHA256 = hashlib.sha256(json.dumps(POLICY, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_sample_plan(*, sample_count, sample_budget, adaptive_qm=False):
    if (isinstance(sample_budget, bool) or not isinstance(sample_budget, int)
            or sample_budget < 1 or sample_count < 1 or sample_count > sample_budget):
        raise ValueError('DVR requires a finite frozen QM sample budget; sample count exceeds it')
    from matrix_core.protocol_library import load_keymaker_protocol_library
    load_keymaker_protocol_library().resolve(POLICY_ID)
    if adaptive_qm:
        raise ValueError('DVR protocol forbids automatic QM midpoints: refine the analytic fit/grid instead')
