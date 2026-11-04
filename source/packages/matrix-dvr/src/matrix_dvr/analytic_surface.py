"""Immutable sparse-QM -> analytic fit -> generated DVR grid policy.

This owner accepts existing samples only; it has no electronic backend callback.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import itertools
import json
import numpy as np

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
    """Budget is a finite explicit acquisition limit, never a DVR grid size."""
    if (isinstance(sample_budget, bool) or not isinstance(sample_budget, int)
            or sample_budget < 1 or sample_count < 1 or sample_count > sample_budget):
        raise ValueError('DVR requires a finite frozen QM sample budget; sample count exceeds it')
    from matrix_core.protocol_library import load_keymaker_protocol_library
    load_keymaker_protocol_library().resolve(POLICY_ID)
    if adaptive_qm:
        raise ValueError('DVR protocol forbids automatic QM midpoints: refine the analytic fit/grid instead')


@dataclass(frozen=True)
class AnalyticSurface:
    coefficients: np.ndarray
    terms: tuple[tuple[int, ...], ...]
    center: np.ndarray
    scale: np.ndarray
    family: str
    period: float | None = None

    def design(self, coordinates):
        q = np.asarray(coordinates, dtype=float)
        if q.ndim == 1:
            q = q[:, None]
        if q.ndim != 2 or q.shape[1] != len(self.center) or not np.all(np.isfinite(q)):
            raise ValueError('finite sample x dimension coordinates required')
        x = (q - self.center) / self.scale
        if np.any(abs(x) > 1 + 1e-10):
            raise ValueError('analytic DVR grid cannot extrapolate beyond the sampled domain')
        if self.family == 'fourier':
            phase = 2 * np.pi * (q[:, 0] - (self.center[0] - self.scale[0])) / self.period
            return np.column_stack([np.ones(len(q)) if t[0] == 0 else
                (np.cos((t[0]+1)//2*phase) if t[0] % 2 else np.sin(t[0]//2*phase)) for t in self.terms])
        tables = [np.polynomial.chebyshev.chebvander(x[:, k], max(t[k] for t in self.terms)) for k in range(x.shape[1])]
        return np.column_stack([np.prod([tables[k][:, t[k]] for k in range(x.shape[1])], axis=0) for t in self.terms])

    def evaluate(self, coordinates):
        return self.design(coordinates) @ self.coefficients


def fit_analytic_surface(coordinates, values, *, degree, family='chebyshev'):
    """Fit one finite-dimensional analytic function; never interpolate via QM."""
    q = np.asarray(coordinates, float)
    if q.ndim == 1:q = q[:, None]
    y = np.asarray(values, float)
    if q.ndim != 2 or y.shape != (len(q),) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(y)):
        raise ValueError('finite training coordinates and one value per sample required')
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError('positive analytic fit degree required')
    lower, upper = q.min(axis=0), q.max(axis=0)
    if np.any(upper <= lower):raise ValueError('all fitted dimensions need nonzero sampled extent')
    if family == 'fourier':
        if q.shape[1] != 1:raise ValueError('Fourier owner currently supports one periodic coordinate')
        terms = tuple((k,) for k in range(2*degree+1));period = float(upper[0]-lower[0])
    elif family == 'chebyshev':
        terms = tuple(t for t in itertools.product(range(degree+1), repeat=q.shape[1]) if sum(t) <= degree);period = None
    else:raise ValueError('declare a supported analytic family: chebyshev or fourier')
    model = AnalyticSurface(np.zeros(len(terms)), terms, (upper+lower)/2, (upper-lower)/2, family, period)
    a = model.design(q)
    if len(q) <= len(terms) or np.linalg.matrix_rank(a) != len(terms):
        raise ValueError('insufficient independent samples for the declared analytic function')
    c = np.linalg.lstsq(a, y, rcond=1e-12)[0]
    return AnalyticSurface(c, terms, model.center, model.scale, family, period)


def run_analytic_surface_request(request_path):
    """KeyMaker owner: qualify a fit and emit all intermediate nodes from it."""
    request_path = Path(request_path).resolve();raw = request_path.read_bytes();r = json.loads(raw)
    if r.get('schema') != 'matrix.dvr.analytic_surface_request.v1':raise ValueError('unsupported analytic DVR request')
    if r.get('family')=='affine_periodic_hermite':
        from .affine_periodic import run_affine_periodic_request
        return run_affine_periodic_request(request_path)
    if r.get('workflow_id')=='coupled-local-torsions.v1':
        from .coupled_torsions import run_coupled_torsions_request
        return run_coupled_torsions_request(request_path)
    if r.get('family')=='periodic_hermite':
        from .periodic_hermite import run_periodic_hermite_request
        return run_periodic_hermite_request(request_path)
    if r.get('family')=='asymptotic_valley':
        from .asymptotic_valley import run_asymptotic_valley_request
        return run_asymptotic_valley_request(request_path)
    if r.get('family')=='asymptotic_channel_matrix':
        from .asymptotic_channels import run_asymptotic_channel_request
        return run_asymptotic_channel_request(request_path)
    if r.get('family')=='asymptotic_barrier':
        from .asymptotic_surface import run_asymptotic_surface_request
        return run_asymptotic_surface_request(request_path)
    q = np.asarray(r['training_coordinates'], float);q = q[:, None] if q.ndim == 1 else q
    y = np.asarray(r['energies_cm'], float)
    validate_sample_plan(sample_count=len(q), sample_budget=r['sample_budget'], adaptive_qm=r.get('adaptive_qm', False))
    mask = np.zeros(len(q), bool);indices = r['validation_indices']
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= len(q):
        raise ValueError('declare distinct in-budget held-out validation samples')
    mask[indices] = True
    degree = r['degree'];family = r.get('family', 'chebyshev')
    candidate = fit_analytic_surface(q[~mask], y[~mask], degree=degree, family=family)
    # Held-out points must lie within the training domain: reserve interior points.
    error = candidate.evaluate(q[mask]) - y[mask]
    tolerance = float(r['max_validation_error_cm'])
    if not np.isfinite(tolerance) or tolerance <= 0:raise ValueError('positive finite fit tolerance required')
    if max(abs(error)) > tolerance:raise ValueError('analytic DVR fit fails held-out validation; no QM refinement will be launched')
    model = fit_analytic_surface(q, y, degree=degree, family=family)
    residual = model.evaluate(q) - y
    if max(abs(residual)) > tolerance:raise ValueError('final analytic DVR fit fails its sample residual gate')
    axes = tuple(np.asarray(a, float) for a in r['grid_axes'])
    if len(axes) != q.shape[1] or any(a.ndim != 1 or len(a)<2 or np.any(np.diff(a)<=0) for a in axes):
        raise ValueError('ordered DVR grid axis required for each fitted dimension')
    nodes = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    dense = model.evaluate(nodes.reshape(-1, q.shape[1])).reshape(tuple(len(a) for a in axes))
    out = (request_path.parent / r.get('output_directory', 'analytic_surface')).resolve();out.mkdir(parents=True, exist_ok=True)
    artifact = out/'analytic_dvr_grid.npz'
    np.savez_compressed(artifact, coordinates=nodes, potential_cm=dense,
                        coefficients=model.coefficients, terms=np.asarray(model.terms),
                        center=model.center, scale=model.scale, family=model.family)
    report = {'schema':'matrix.dvr.analytic_surface_result.v1', 'policy_id':POLICY_ID,
              'policy_sha256':POLICY_SHA256, 'request_sha256':hashlib.sha256(raw).hexdigest(),
              'sample_count':len(q), 'sample_budget':r['sample_budget'], 'family':family,
              'degree':degree, 'validation_max_error_cm':float(max(abs(error))),
              'fit_max_error_cm':float(max(abs(residual))), 'grid_source':'analytic_function_only',
              'QM_evaluations_for_generated_grid':0, 'artifact':str(artifact),
              'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(), 'status':'fit_validated'}
    target = out/'analytic_surface_result.json';target.write_text(json.dumps(report, indent=2)+'\n');return target


def generate_analytic_values(coordinates, values, targets, *, periodic=False):
    """Legacy-grid adapter: select analytic order by leave-one-out error.

    It evaluates only an analytic function on the new nodes; no QM callbacks.
    The explicit KeyMaker owner supplies the stronger held-out acceptance gate.
    """
    q = np.asarray(coordinates, float);q = q[:, None] if q.ndim == 1 else q
    y = np.asarray(values, float);family = 'fourier' if periodic else 'chebyshev'
    candidates = []
    for degree in range(1, min(10, len(q)-2)+1):
        try:model = fit_analytic_surface(q, y, degree=degree, family=family)
        except ValueError:continue
        a = model.design(q)
        if len(q) < len(model.terms)+2:continue
        u, _, _ = np.linalg.svd(a, full_matrices=False);leverage = np.sum(u*u, axis=1)
        if np.any(leverage > 1-1e-8):continue
        error = (model.evaluate(q)-y)/(1-leverage)
        candidates.append((float(np.mean(error*error)), model))
    if not candidates:raise ValueError('no supported analytic DVR fit; provide qualified sparse samples')
    model = min(candidates, key=lambda pair:pair[0])[1]
    return model.evaluate(targets)
