"""Positive occupation-dependent rate kernels conditioned on a common energy map.

The photon factor and exact harmonic ladder dependence are explicit. A smooth
anisotropic kernel learns the remaining log reduced rate, using both integer
and effective occupations. Kernel relevance and smoothing are inferred from
calibration data only; no reference eigensolve enters prediction.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np
from scipy.linalg import cho_factor, cho_solve, qr
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from .gvpt2_rate_model import EffectiveQuantumRateModel, _channel, _ladder_factor
from .gvpt2_rates import GVPT2RateReference, _labels, _occupations

SCHEMA = 'matrix.trinity.gvpt2-conditional-rates.v1'


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _fit_kernel(x, y, *, maximum_iterations, length_bounds, smoothing_bounds):
    n, dimension = x.shape
    p = np.column_stack((np.ones(n), x))
    _, r, pivot = qr(p, pivoting=True, mode='economic')
    rank = min(np.linalg.matrix_rank(r), max(1, n-2))
    keep = np.array(sorted(pivot[:rank]), int)
    p = p[:, keep]
    differences = (x[:, None, :]-x[None, :, :])**2
    df = n-rank
    if df <= 0:
        raise ValueError('at least two measured pairs per allowed transition family required')
    cache = {}

    def objective(parameters):
        lengths, noise = np.exp(parameters[:dimension]), np.exp(parameters[-1])
        k = np.exp(-.5*np.sum(differences/lengths**2, axis=2))
        derivatives = k[:, :, None]*differences/lengths**2
        cf = cho_factor(k+noise*np.eye(n), lower=True)
        inverse = cho_solve(cf, np.eye(n))
        kip = inverse@p
        pf = cho_factor(p.T@kip, lower=True)
        beta = cho_solve(pf, p.T@inverse@y)
        alpha = inverse@(y-p@beta)
        projection = inverse-kip@cho_solve(pf, kip.T)
        variance = max(float(y@alpha/df), 1e-20)
        value = .5*(df*np.log(variance)+2*np.log(np.diag(cf[0])).sum()+
                    2*np.log(np.diag(pf[0])).sum())
        sensitivity = projection-np.outer(alpha, alpha)/variance
        gradient = np.r_[.5*np.einsum('ij,ijk->k', sensitivity, derivatives),
                         .5*noise*np.trace(sensitivity)]
        cache.update(lengths=lengths, smoothing=noise, alpha=alpha, trend=beta)
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError('conditional-rate likelihood exceeded numerical range')
        return value, gradient

    bounds = [tuple(np.log(length_bounds))]*dimension+[tuple(np.log(smoothing_bounds))]
    initial = np.r_[np.full(dimension, np.log(.5)), np.log(.1)]
    initial = np.clip(initial, np.array(bounds)[:, 0], np.array(bounds)[:, 1])
    result = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=bounds,
                      options=dict(maxiter=maximum_iterations, ftol=1e-10))
    objective(result.x)
    return dict(centers=x.tolist(), alpha=cache['alpha'].tolist(), trend=cache['trend'].tolist(),
        trend_columns=keep.tolist(), lengths=cache['lengths'].tolist(),
        smoothing=float(cache['smoothing']), likelihood=float(result.fun),
        likelihood_converged=bool(result.success), optimizer_message=str(result.message),
        iterations=int(result.nit), reference_pairs=n)


def _kernel_prediction(head, points):
    centers = np.asarray(head['centers'], float)
    alpha, trend = np.asarray(head['alpha']), np.asarray(head['trend'])
    lengths = np.asarray(head['lengths'])
    keep = np.asarray(head['trend_columns'], int)
    result = np.empty(len(points))
    # Bound the temporary pair-distance matrix independently of input batch size.
    block = max(1, (32*1024**2)//(8*len(centers)))
    for start in range(0, len(points), block):
        x = points[start:start+block]
        kernel = np.exp(-.5*cdist(x/lengths, centers/lengths, 'sqeuclidean'))
        result[start:start+len(x)] = kernel@alpha+np.column_stack((np.ones(len(x)), x))[:, keep]@trend
    return result


def _reduced_rate_prediction(head, points):
    if head.get('link') == 'coherent':
        values = np.zeros(len(points))
        for component in head['components']:
            values += _kernel_prediction(component, points)**2
        return head['rate_scale']*values
    response = _kernel_prediction(head, points)
    with np.errstate(over='raise', invalid='raise'):
        if head.get('link', 'log') == 'sqrt':
            return head['rate_scale']*response**2
        return np.exp(response)


@dataclass(frozen=True)
class ConditionalQuantumRateModel:
    """Common energy/effective-number model with a calibrated conditional rate map.

    Energies are inherited exactly from base. The conditional rate coefficients
    are calibrated afterward with its photon gaps held fixed. This is a two-stage
    model, not a new simultaneous refit of the base effective-number map.
    """
    base: EffectiveQuantumRateModel
    heads: dict
    diagnostics: dict

    def __post_init__(self):
        if not isinstance(self.base, EffectiveQuantumRateModel) or not self.heads:
            raise ValueError('a common energy model and measured rate families required')
        dimension = 2*len(self.base.scales)
        for key, head in self.heads.items():
            link = head.get('link', 'log')
            if (key not in self.base.channels or link not in ('log', 'sqrt', 'coherent')
                    or not np.isfinite(head.get('rate_scale', 1.)) or head.get('rate_scale', 1.) <= 0):
                raise ValueError('invalid conditional-rate family/link')
            parts = head.get('components', ()) if link == 'coherent' else (head,)
            if not parts:
                raise ValueError('coherent rate requires fitted component functions')
            for part in parts:
                centers = np.asarray(part['centers'], float)
                alpha, trend = np.asarray(part['alpha'], float), np.asarray(part['trend'], float)
                lengths = np.asarray(part['lengths'], float)
                keep = np.asarray(part['trend_columns'])
                if (centers.ndim != 2 or centers.shape[1] != dimension or len(centers) < 2
                        or alpha.shape != (len(centers),) or lengths.shape != (dimension,)
                        or np.any(lengths <= 0) or keep.dtype.kind not in 'iu' or keep.ndim != 1
                        or len(set(keep)) != len(keep) or np.any(keep < 0) or np.any(keep > dimension)
                        or trend.shape != keep.shape or not all(np.all(np.isfinite(v))
                            for v in (centers, alpha, trend, lengths))):
                    raise ValueError('invalid conditional-rate coefficients')
        # Detach nested caller-owned dictionaries, retaining a JSON-native artifact.
        object.__setattr__(self, 'heads', _freeze(self.heads))
        object.__setattr__(self, 'diagnostics', dict(self.diagnostics))

    def __getattr__(self, name):
        # The sampler consumes the unchanged compiled energy coefficient protocol.
        return getattr(self.base, name)

    @property
    def extended_rate_identity(self):
        return hashlib.sha256(json.dumps(_thaw(self.heads), sort_keys=True, allow_nan=False).encode()).hexdigest()

    def energies(self, occupations, **controls):
        return self.base.energies(occupations, **controls)

    def effective_numbers(self, occupations, **controls):
        return self.base.effective_numbers(occupations, **controls)

    def _features(self, occupations, *, sectors=None, branches=None, allow_extrapolation=False):
        q = self.base.effective_numbers(occupations, sectors=sectors, branches=branches,
                                        allow_extrapolation=allow_extrapolation)
        return np.column_stack((occupations/self.base.scales, q/self.base.scales))

    def rates(self, initial, final, *, allowed, initial_sectors=None, final_sectors=None,
              initial_branches=None, final_branches=None, allow_extrapolation=False):
        a, b = _occupations(initial), _occupations(final)
        mask = np.asarray(allowed)
        if a.shape != b.shape or mask.shape != (len(a),) or mask.dtype.kind != 'b':
            raise ValueError('matched endpoint batches and explicit boolean mask required')
        sa, sb = _labels(initial_sectors, len(a), 'scalar'), _labels(final_sectors, len(b), 'scalar')
        controls_a = dict(sectors=sa, branches=initial_branches, allow_extrapolation=allow_extrapolation)
        controls_b = dict(sectors=sb, branches=final_branches, allow_extrapolation=allow_extrapolation)
        gap = self.energies(a, **controls_a)-self.energies(b, **controls_b)
        features = (self._features(a, **controls_a)+self._features(b, **controls_b))/2
        values = np.zeros(len(a))
        groups = {}
        for row in np.flatnonzero(mask & (gap > 0)):
            key = _channel(a[row], b[row], sa[row], sb[row])
            if key not in self.heads:
                raise ValueError('unmeasured allowed transition family; extend calibration')
            groups.setdefault(key, []).append(row)
        for key, rows in groups.items():
            rows = np.asarray(rows)
            reduced_rate = _reduced_rate_prediction(self.heads[key], features[rows])
            with np.errstate(over='raise', invalid='raise'):
                values[rows] = gap[rows]**3*_ladder_factor(a[rows], b[rows])*reduced_rate
        if not np.all(np.isfinite(values)):
            raise FloatingPointError('conditional rates exceed numerical range')
        return values

    cascade_channels = EffectiveQuantumRateModel.cascade_channels

    def sample_wang_landau(self, **controls):
        from matrix_sentinel.effective_quantum_rewl import sample_effective_quantum_wang_landau
        return sample_effective_quantum_wang_landau(self, **controls)

    def to_dict(self):
        return dict(schema=SCHEMA, base=self.base.to_dict(), heads=_thaw(self.heads), diagnostics=self.diagnostics)

    def save(self, path):
        def encode(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(type(value).__name__)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False, default=encode)+'\n')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if set(data) != {'schema', 'base', 'heads', 'diagnostics'} or data['schema'] != SCHEMA:
            raise ValueError('unsupported conditional-rate artifact')
        if data['base'].get('schema') == 'matrix.trinity.gvpt2-polynomial-energy.v1':
            from .gvpt2_hybrid_rates import PolynomialQuantumEnergyModel
            base = PolynomialQuantumEnergyModel.from_dict(data['base'])
        else:
            base = EffectiveQuantumRateModel.from_dict(data['base'])
        return cls(base, data['heads'], data['diagnostics'])


def fit_conditional_quantum_rates(reference, *, base_model=None, maximum_iterations=150,
        length_bounds=(.05, 10.), smoothing_bounds=(1e-5, 10.), rate_floor_s_inverse=1e-12,
        maximum_family_pairs=2000, maximum_workspace_bytes=256*1024**2):
    """Fit positive rate kernels using only the supplied calibration reference.

    Restricted marginal likelihood learns one relevance length per integer and
    effective mode coordinate. The factored photon frequency always uses the
    common base energy function. No held-out energies or rates are consulted.
    """
    if not isinstance(reference, GVPT2RateReference):
        raise TypeError('GVPT2RateReference required')
    if base_model is None:
        from .gvpt2_rate_selection import fit_selected_gvpt2_rate_model
        base_model = fit_selected_gvpt2_rate_model(reference)
    if not isinstance(base_model, EffectiveQuantumRateModel):
        raise TypeError("EffectiveQuantumRateModel base required")
    for budget in (maximum_family_pairs, maximum_workspace_bytes):
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError("positive integer kernel budgets required")
    if (base_model.coordinate_id != reference.coordinate_id
            or isinstance(maximum_iterations, bool) or not isinstance(maximum_iterations, int) or maximum_iterations < 1
            or not np.isfinite(rate_floor_s_inverse) or rate_floor_s_inverse <= 0):
        raise ValueError('common coordinate identity and positive fit controls required')
    for bounds in (length_bounds, smoothing_bounds):
        if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or not 0 < bounds[0] < bounds[1]:
            raise ValueError('positive ordered kernel bounds required')
    n = reference.occupations
    q = base_model.effective_numbers(n, sectors=reference.sectors, branches=reference.branches)
    features = np.column_stack((n/base_model.scales, q/base_model.scales))
    energies = base_model.energies(n, sectors=reference.sectors, branches=reference.branches)
    rows = np.flatnonzero(reference.allowed)
    left, right = reference.pairs[rows].T
    gap = abs(energies[left]-energies[right])
    if np.any(gap <= 0):
        raise ValueError('positive calibrated emission gaps required')
    keys = [_channel(n[a], n[b], reference.sectors[a], reference.sectors[b]) for a, b in zip(left, right)]
    targets = np.log(np.maximum(reference.rates_s_inverse[rows], rate_floor_s_inverse)/
                     (gap**3*_ladder_factor(n[left], n[right])))
    midpoint = (features[left]+features[right])/2
    heads = {}
    for key in sorted(set(keys)):
        mask = np.array([k == key for k in keys])
        estimate = 8*int(sum(mask))**2*(2*midpoint.shape[1]+8)
        if sum(mask) > maximum_family_pairs or estimate > maximum_workspace_bytes:
            raise MemoryError('conditional-rate family exceeds declared pair budget')
        heads[key] = _fit_kernel(midpoint[mask], targets[mask], maximum_iterations=maximum_iterations,
            length_bounds=length_bounds, smoothing_bounds=smoothing_bounds)
    diagnostics = dict(schema=SCHEMA, external_test_used=False, energy_model_unchanged=True,
        calibration_states=len(n), calibration_allowed_pairs=len(rows),
        maximum_iterations=maximum_iterations, length_bounds=list(length_bounds),
        smoothing_bounds=list(smoothing_bounds), rate_floor_s_inverse=rate_floor_s_inverse,
        energy_origin_anchored=bool(base_model.diagnostics.get('fit_controls', {}).get('anchor_ground_energy')),
        likelihood_converged=all(head['likelihood_converged'] for head in heads.values()),
        zero_allowed_rates_approximated_with_floor=True, physical_domain_qualified=False)
    return ConditionalQuantumRateModel(base_model, heads, diagnostics)


def fit_transitionwise_quantum_rates(reference, *, base_model=None, folds=3,
        maximum_iterations=150, length_bounds=(.05, 10.), smoothing_bounds=(1e-5, 10.),
        rate_floor_s_inverse=1e-12, maximum_family_pairs=2000,
        maximum_workspace_bytes=256*1024**2):
    """Select the positive link per family on internal whole-state folds.

    Log and square-root links are compared on errors of individual Einstein
    coefficients, including a tail score. Both have nonnegative algebraic
    predictions. The square-root response is a regression variable, not a
    signed transition moment or wavefunction phase. No external test is read.
    Validation is conditional on the fixed supplied common energy model.
    """
    if not isinstance(reference, GVPT2RateReference):
        raise TypeError('GVPT2RateReference required')
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError('at least two integer folds required')
    if base_model is None:
        from .gvpt2_rate_selection import fit_selected_gvpt2_rate_model
        base_model = fit_selected_gvpt2_rate_model(reference)
    # Existing acquisition validates controls and provides a guaranteed log-link
    # fallback before any internal selection, including sparse family handling.
    original = fit_conditional_quantum_rates(reference, base_model=base_model,
        maximum_iterations=maximum_iterations, length_bounds=length_bounds,
        smoothing_bounds=smoothing_bounds, rate_floor_s_inverse=rate_floor_s_inverse,
        maximum_family_pairs=maximum_family_pairs, maximum_workspace_bytes=maximum_workspace_bytes)
    n = reference.occupations
    state_features = original._features(n, sectors=reference.sectors, branches=reference.branches)
    energies = original.energies(n, sectors=reference.sectors, branches=reference.branches)
    rows = np.flatnonzero(reference.allowed)
    left, right = reference.pairs[rows].T
    factor = abs(energies[left]-energies[right])**3*_ladder_factor(n[left], n[right])
    y = reference.rates_s_inverse[rows]
    features = (state_features[left]+state_features[right])/2
    keys = [_channel(n[a], n[b], reference.sectors[a], reference.sectors[b]) for a, b in zip(left, right)]
    fold_ids = np.array([int(hashlib.sha256(np.asarray(row, '<i8').tobytes()+
        (reference.sectors[i]+'\0'+reference.branches[i]).encode()).hexdigest()[:8], 16) % folds
        for i, row in enumerate(n)])
    anchors = np.count_nonzero(n, axis=1) <= 1
    heads, selections = {}, {}

    def fit_link(x, rates, factors, link):
        reduced = np.maximum(rates, rate_floor_s_inverse)/factors
        if not np.all(np.isfinite(reduced)) or np.any(reduced <= 0):
            raise FloatingPointError('factored transition rates exceed numerical range')
        scale = float(np.median(reduced))
        target = np.log(reduced) if link == 'log' else np.sqrt(reduced/scale)
        fitted = _fit_kernel(x, target, maximum_iterations=maximum_iterations,
            length_bounds=length_bounds, smoothing_bounds=smoothing_bounds)
        fitted.update(link=link, rate_scale=scale)
        return fitted

    for key in sorted(set(keys)):
        family = np.flatnonzero(np.array([k == key for k in keys]))
        records = []
        for link in ('log', 'sqrt'):
            fold_records, valid = [], True
            for f in range(folds):
                excluded = (fold_ids == f) & ~anchors
                validation = np.any(excluded[np.column_stack((left[family], right[family]))], axis=1)
                train, test = family[~validation], family[validation]
                if len(train) < 3 or not len(test):
                    valid = False
                    break
                try:
                    fitted = fit_link(features[train], y[train], factor[train], link)
                    prediction = factor[test]*_reduced_rate_prediction(fitted, features[test])
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    fold_records.append(dict(fold=f, failure=str(exc)))
                    valid = False
                    break
                symmetric = 2*abs(prediction-y[test])/(prediction+y[test]+rate_floor_s_inverse)
                aggregate = float(abs(prediction-y[test]).sum()/max(y[test].sum(), rate_floor_s_inverse))
                score = .5*float(symmetric.mean())+.25*float(np.quantile(symmetric, .9))+.25*aggregate
                fold_records.append(dict(fold=f, pairs=len(test), score=score,
                    mean_symmetric_relative_error=float(symmetric.mean()),
                    p90_symmetric_relative_error=float(np.quantile(symmetric, .9)),
                    aggregate_rate_error=aggregate))
            records.append(dict(link=link, folds=fold_records,
                score=float(np.mean([v['score'] for v in fold_records])) if valid else None))
        eligible = [r for r in records if r['score'] is not None]
        if eligible:
            chosen = min(eligible, key=lambda r: r['score'])['link']
            heads[key] = fit_link(features[family], y[family], factor[family], chosen)
        else:
            chosen = 'log'
            heads[key] = _thaw(original.heads[key])
        selections[key] = dict(chosen_link=chosen, candidates=records,
                               sparse_fallback=not bool(eligible))
    diagnostics = dict(original.diagnostics, transitionwise_link_selection=dict(
        folds=folds, external_test_used=False, conditional_on_fixed_energy_model=True,
        scoring_weights=dict(mean_symmetric=.5, p90_symmetric=.25, aggregate_rate=.25),
        families=selections), likelihood_converged=all(h['likelihood_converged'] for h in heads.values()))
    return ConditionalQuantumRateModel(base_model, heads, diagnostics)


def fit_coherent_quantum_rates(reference, transition_moments_au, *, base_model=None,
        phase_convention, maximum_iterations=150, length_bounds=(.05, 10.),
        smoothing_bounds=(1e-5, 10.), maximum_family_pairs=2000,
        maximum_workspace_bytes=256*1024**2):
    """Fit physical phase-aligned moment components; return transition rates.

    Moment phases must refer to consistently assigned states in one common
    gauge. The native adapter can fix individual assigned-state phases; it
    does not determine arbitrary rotations inside unresolved degenerate spaces.
    Only the supplied calibration moments enter the regression. An orthogonal
    numerical-rank reduction preserves their total strength to roundoff.
    """
    from scipy.constants import c, epsilon_0, hbar, e, physical_constants
    if not isinstance(reference, GVPT2RateReference):
        raise TypeError('GVPT2RateReference required')
    moments = np.asarray(transition_moments_au, complex)
    if (moments.shape != (len(reference.pairs), 3) or not np.all(np.isfinite(moments))
            or not isinstance(phase_convention, str) or not phase_convention.strip()):
        raise ValueError('three finite atomic-unit dipole components and an explicit common phase convention required')
    if base_model is None:
        from .gvpt2_rate_selection import fit_selected_gvpt2_rate_model
        base_model = fit_selected_gvpt2_rate_model(reference)
    if not isinstance(base_model, EffectiveQuantumRateModel) or base_model.coordinate_id != reference.coordinate_id:
        raise ValueError('common base energy model/coordinate identity required')
    for value in (maximum_iterations, maximum_family_pairs, maximum_workspace_bytes):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('positive integer coherent-fit budgets required')
    for bounds in (length_bounds, smoothing_bounds):
        if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or not 0 < bounds[0] < bounds[1]:
            raise ValueError('positive ordered kernel bounds required')
    conversion = (2*np.pi*c*100)**3*(e*physical_constants['Bohr radius'][0])**2/(3*np.pi*epsilon_0*hbar*c**3)
    rows = np.flatnonzero(reference.allowed)
    a, b = reference.pairs[rows].T
    gaps = abs(reference.energies_cm[a]-reference.energies_cm[b])
    expected = conversion*gaps**3*np.sum(abs(moments[rows])**2, axis=1)
    if not np.allclose(expected, reference.rates_s_inverse[rows], rtol=1e-8, atol=1e-12):
        raise ValueError('moment units, endpoint assignment or reference photon gaps disagree with the rate table')
    n = reference.occupations
    q = base_model.effective_numbers(n, sectors=reference.sectors, branches=reference.branches)
    features = np.column_stack((n/base_model.scales, q/base_model.scales))
    midpoint = (features[a]+features[b])/2
    ladder = _ladder_factor(n[a], n[b])
    reduced = moments[rows]/np.sqrt(ladder)[:, None]
    components = np.column_stack((reduced.real, reduced.imag))
    keys = [_channel(n[i], n[j], reference.sectors[i], reference.sectors[j]) for i, j in zip(a, b)]
    heads = {}
    for key in sorted(set(keys)):
        mask = np.array([k == key for k in keys])
        count = int(sum(mask))
        estimate = 8*count**2*(2*midpoint.shape[1]+8)
        if count > maximum_family_pairs or estimate > maximum_workspace_bytes:
            raise MemoryError('coherent-rate family exceeds declared workspace/pair budget')
        data = components[mask]
        _, singular, basis = np.linalg.svd(data, full_matrices=False)
        threshold = (singular[0] if len(singular) else 0)*max(data.shape)*np.finfo(float).eps
        rank = max(1, int(sum(singular > threshold)))
        targets = data@basis[:rank].T
        amplitude_scale = max(float(np.sqrt(np.mean(np.sum(targets**2, axis=1)))), np.sqrt(np.finfo(float).tiny))
        targets /= amplitude_scale
        parts = [_fit_kernel(midpoint[mask], targets[:, i], maximum_iterations=maximum_iterations,
                  length_bounds=length_bounds, smoothing_bounds=smoothing_bounds) for i in range(rank)]
        heads[key] = dict(link='coherent', rate_scale=conversion*amplitude_scale**2,
            components=parts, retained_component_rank=rank, cartesian_component_dimension=3,
            numerical_rank_threshold=float(threshold), polarization_basis=basis[:rank].tolist(),
            likelihood_converged=all(p['likelihood_converged'] for p in parts), reference_pairs=count)
    diagnostics = dict(schema=SCHEMA, external_test_used=False, energy_model_unchanged=True,
        coherent_moment_regression=True, phase_convention=phase_convention,
        input_property='phase_aligned_electric_dipole_atomic_units',
        acquisition_rate_consistency_maximum_error_s_inverse=float(max(abs(expected-reference.rates_s_inverse[rows]))),
        calibration_states=len(n), calibration_allowed_pairs=len(rows),
        likelihood_converged=all(h['likelihood_converged'] for h in heads.values()),
        physical_domain_qualified=False, reference_frequency_used_only_for_unit_consistency=True)
    return ConditionalQuantumRateModel(base_model, heads, diagnostics)
