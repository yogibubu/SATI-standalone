"""Exact low-occupation data with a polynomial energy and conditional rate tail.

The occupation cutoff is an input. Missing low transitions are errors, never
zero-filled or counted as predictions. Higher-energy predictions need no solve.
"""
from dataclasses import dataclass, fields, replace
from itertools import combinations_with_replacement
from math import comb
import hashlib
import json

import numpy as np
from scipy.linalg import svd

from .gvpt2_rate_model import EffectiveQuantumRateModel
from .gvpt2_rates import GVPT2RateReference, _occupations, _labels, _readonly

ENERGY_SCHEMA = 'matrix.trinity.gvpt2-polynomial-energy.v1'
SCHEMA = 'matrix.trinity.gvpt2-hybrid-rates.v1'


def _key(n, sector, branch):
    return tuple(int(i) for i in n), sector, branch


def _rank(n):
    total, rank = 0, 0
    for i, value in enumerate(n):
        total += int(value)
        rank += comb(total+i, i+1)
    return rank


def _powers(modes, degree, budget):
    if comb(modes+degree, degree)-1 > budget:
        raise MemoryError('polynomial exceeds declared coefficient budget')
    rows = []
    for order in range(1, degree+1):
        for indices in combinations_with_replacement(range(modes), order):
            rows.append(np.bincount(indices, minlength=modes))
    return np.asarray(rows, np.int64)


def _features(n, scales, powers):
    # Build term by term, avoiding a states x terms x modes temporary.
    x = n/scales
    result = np.ones((len(n), len(powers)))
    for j, power in enumerate(powers):
        for i in np.flatnonzero(power):
            result[:, j] *= x[:, i]**power[i]
    return result


@dataclass(frozen=True)
class PolynomialQuantumEnergyModel(EffectiveQuantumRateModel):
    """Regularized occupation-polynomial correction to the common energy map.

    Category-specific coefficients preserve resolved symmetry/branch identities.
    Exact supplied low energies replace polynomial values below the cutoff.
    """
    energy_powers: np.ndarray
    energy_residual_coefficients_cm: np.ndarray
    low_reference: GVPT2RateReference
    exact_quanta: int
    window_reference: GVPT2RateReference | None = None
    energy_window: dict | None = None

    def __post_init__(self):
        super().__post_init__()
        p = np.asarray(self.energy_powers)
        c = np.asarray(self.energy_residual_coefficients_cm, float)
        if (p.ndim != 2 or p.shape[1] != len(self.scales) or not len(p)
                or p.dtype.kind not in 'iu' or np.any(p < 0) or np.any(p.sum(axis=1) == 0)
                or c.shape != (len(self.categories), len(p)) or not np.all(np.isfinite(c))
                or isinstance(self.exact_quanta, bool) or not isinstance(self.exact_quanta, int)
                or self.exact_quanta < 1 or self.exact_quanta > self.maximum_total_quanta
                or not isinstance(self.low_reference, GVPT2RateReference)
                or self.low_reference.coordinate_id != self.coordinate_id
                or self.low_reference.occupations.shape[1] != len(self.scales)
                or np.any(self.low_reference.occupations.sum(axis=1) > self.exact_quanta)
                or len({tuple(row) for row in p}) != len(p)):
            raise ValueError('invalid polynomial or exact-low energy domain')
        if any(cat not in self.categories for cat in zip(self.low_reference.sectors, self.low_reference.branches)):
            raise ValueError('unknown low-state category')
        object.__setattr__(self, 'energy_powers', _readonly(p, np.int64))
        object.__setattr__(self, 'energy_residual_coefficients_cm', _readonly(c))
        exact = {
            _key(n, s, b): float(e) for n, s, b, e in zip(self.low_reference.occupations,
                self.low_reference.sectors, self.low_reference.branches, self.low_reference.energies_cm)}
        extra = self.window_reference
        if extra is not None:
            if (not isinstance(extra, GVPT2RateReference) or extra.coordinate_id != self.coordinate_id
                    or extra.occupations.shape[1] != len(self.scales) or not isinstance(self.energy_window, dict)):
                raise ValueError('compatible window reference and declared energy window required')
            for n, s, b, e in zip(extra.occupations, extra.sectors, extra.branches, extra.energies_cm):
                key = _key(n, s, b)
                if ((s, b) not in self.categories or np.any(n > self.maximum_occupations)
                        or n.sum(dtype=object) > self.maximum_total_quanta):
                    raise ValueError('exact window extends beyond declared model domain')
                if key in exact and exact[key] != e:
                    raise ValueError('inconsistent low/window reference energies')
                exact[key] = float(e)
        elif self.energy_window is not None:
            raise ValueError('energy window requires acquired reference data')
        object.__setattr__(self, '_exact_energies', exact)

    def energies(self, occupations, *, sectors=None, branches=None, allow_extrapolation=False):
        n = _occupations(occupations)
        s, b = _labels(sectors, len(n), 'scalar'), _labels(branches, len(n), 'reference')
        values = super().energies(n, sectors=s, branches=b, allow_extrapolation=allow_extrapolation)
        categories = np.array([self.categories.index(cat) for cat in zip(s, b)])
        values += np.einsum('ij,ij->i', _features(n, self.scales, self.energy_powers),
                            self.energy_residual_coefficients_cm[categories])
        for i in range(len(n)):
            key = _key(n[i], s[i], b[i])
            if key in self._exact_energies:
                values[i] = self._exact_energies[key]
            elif n[i].sum(dtype=object) <= self.exact_quanta:
                raise ValueError('missing exact low-state energy; acquire resolved reference')
        if not np.all(np.isfinite(values)):
            raise FloatingPointError('polynomial energies exceed numerical range')
        return values

    def rates(self, *args, **kwargs):
        raise TypeError('attach a conditional rate model before predicting rates with generalized energies')

    def compiled_energy_extension(self, categories, maximum_cache_bytes=64*1024**2):
        count = comb(len(self.scales)+self.exact_quanta, self.exact_quanta)
        if count*len(categories)*8 > maximum_cache_bytes:
            raise MemoryError('exact low-energy cache exceeds declared budget')
        cache = np.full((len(categories), count), np.nan)
        for (n, s, b), e in self._exact_energies.items():
            if (s, b) in categories and sum(n) <= self.exact_quanta:
                cache[categories.index((s, b)), _rank(n)] = e
        coefficients = self.energy_residual_coefficients_cm[
            [self.categories.index(tuple(cat)) for cat in categories]]
        anchors = sorted((categories.index((s, b)), n, e) for (n, s, b), e in self._exact_energies.items()
                         if (s, b) in categories and sum(n) > self.exact_quanta)
        anchor_n = np.array([n for _, n, _ in anchors], np.int64).reshape(-1, len(self.scales))
        anchor_categories = np.array([cat for cat, _, _ in anchors], np.int64)
        anchor_e = np.array([e for _, _, e in anchors], float)
        return self.energy_powers, coefficients, self.exact_quanta, cache, anchor_n, anchor_categories, anchor_e

    def to_dict(self):
        base = {f.name: getattr(self, f.name) for f in fields(EffectiveQuantumRateModel)}
        return dict(schema=ENERGY_SCHEMA, base=EffectiveQuantumRateModel(**base).to_dict(),
            powers=self.energy_powers.tolist(), coefficients_cm=self.energy_residual_coefficients_cm.tolist(),
            exact_quanta=self.exact_quanta, low_reference=self.low_reference.to_dict(),
            window_reference=None if self.window_reference is None else self.window_reference.to_dict(),
            energy_window=self.energy_window)

    @classmethod
    def from_dict(cls, data):
        required = {'schema', 'base', 'powers', 'coefficients_cm', 'exact_quanta', 'low_reference'}
        if not required <= set(data) or set(data)-required-{'window_reference', 'energy_window'} or data['schema'] != ENERGY_SCHEMA:
            raise ValueError('unsupported polynomial-energy artifact')
        base = EffectiveQuantumRateModel.from_dict(data['base'])
        low = dict(data['low_reference'])
        if low.pop('schema', None) != 'matrix.trinity.gvpt2-rate-reference.v1':
            raise ValueError('unsupported low-state reference')
        window = data.get('window_reference')
        if window is not None:
            window = dict(window)
            if window.pop('schema', None) != 'matrix.trinity.gvpt2-rate-reference.v1':
                raise ValueError('unsupported window reference')
            window = GVPT2RateReference(**window)
        return cls(**{f.name: getattr(base, f.name) for f in fields(EffectiveQuantumRateModel)},
            energy_powers=data['powers'], energy_residual_coefficients_cm=data['coefficients_cm'],
            low_reference=GVPT2RateReference(**low), exact_quanta=data['exact_quanta'],
            window_reference=window, energy_window=data.get('energy_window'))


def fit_polynomial_quantum_energies(reference, base_model, *, exact_quanta=2,
        degrees=(2, 3, 4), penalties=(1e-6, .001, .1, 10.), folds=3,
        maximum_terms=2000, maximum_workspace_bytes=256*1024**2):
    """Select degree and ridge penalty on internal high-state energy folds only.

    Polynomial coefficients fit residual energies, not transition rates. The
    exact-low reference must be included in the supplied calibration inventory.
    No external holdout table enters this function.
    """
    from .gvpt2_rate_selection import subset_reference
    if not isinstance(reference, GVPT2RateReference) or not isinstance(base_model, EffectiveQuantumRateModel):
        raise TypeError('reference and common base energy model required')
    if (reference.coordinate_id != base_model.coordinate_id or isinstance(exact_quanta, bool)
            or not isinstance(exact_quanta, int) or not 1 <= exact_quanta <= base_model.maximum_total_quanta
            or isinstance(folds, bool) or not isinstance(folds, int) or folds < 2
            or not degrees or any(isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in degrees)
            or not penalties or any(not np.isfinite(p) or p <= 0 for p in penalties)
            or any(isinstance(b, bool) or not isinstance(b, int) or b < 1
                   for b in (maximum_terms, maximum_workspace_bytes))):
        raise ValueError('invalid polynomial fit controls')
    n = reference.occupations
    low = n.sum(axis=1) <= exact_quanta
    low_reference = subset_reference(reference, low)
    labels = dict(sectors=reference.sectors, branches=reference.branches)
    residual = reference.energies_cm-base_model.energies(n, **labels)
    cats = np.array([base_model.categories.index(cat) for cat in zip(reference.sectors, reference.branches)])
    fold_id = np.array([int(hashlib.sha256(np.asarray(row, '<i8').tobytes()+
        (s+'\0'+b).encode()).hexdigest()[:8], 16) % folds
        for row, s, b in zip(n, reference.sectors, reference.branches)])
    # Keep an energy-origin witness for every category in all folds.
    anchors = low.copy()
    for cat in range(len(base_model.categories)):
        members = np.flatnonzero(cats == cat)
        if not len(members):
            raise ValueError('energy calibration missing a resolved category')
        anchors[members[np.argmin(n[members].sum(axis=1))]] = True
    records, choices = [], []
    for degree in degrees:
        powers = _powers(n.shape[1], degree, maximum_terms)
        if 8*(len(n)*len(powers)*4+len(powers)**2) > maximum_workspace_bytes:
            raise MemoryError('energy regression exceeds declared workspace budget')
        x = _features(n, base_model.scales, powers)
        errors = {p: [] for p in penalties}
        for f in range(folds):
            test = (fold_id == f) & ~anchors
            for cat in range(len(base_model.categories)):
                train, valid = (cats == cat) & ~test, (cats == cat) & test
                if not np.any(valid):
                    continue
                u, singular, vt = svd(x[train], full_matrices=False)
                projected = u.T@residual[train]
                for penalty in penalties:
                    coefficients = vt.T@((singular/(singular**2+penalty))*projected)
                    errors[penalty].extend((x[valid]@coefficients-residual[valid]).tolist())
        for penalty, values in errors.items():
            if not values:
                raise ValueError('no high-state energy validation rows above exact cutoff')
            rmse = float(np.sqrt(np.mean(np.square(values))))
            records.append(dict(degree=degree, penalty=penalty, energy_rmse_cm=rmse, validation_rows=len(values)))
            choices.append((rmse, degree, penalty))
    _, degree, penalty = min(choices)
    powers = _powers(n.shape[1], degree, maximum_terms)
    x = _features(n, base_model.scales, powers)
    coefficients = []
    for cat in range(len(base_model.categories)):
        selected = cats == cat
        u, singular, vt = svd(x[selected], full_matrices=False)
        coefficients.append(vt.T@((singular/(singular**2+penalty))*(u.T@residual[selected])))
    values = {f.name: getattr(base_model, f.name) for f in fields(EffectiveQuantumRateModel)}
    values['diagnostics'] = dict(base_model.diagnostics, polynomial_selection=dict(
        candidates=records, selected_degree=degree, selected_penalty=penalty,
        external_test_used=False, base_map_fixed_during_internal_validation=True))
    return PolynomialQuantumEnergyModel(**values, energy_powers=powers,
        energy_residual_coefficients_cm=coefficients, low_reference=low_reference, exact_quanta=exact_quanta)


class HybridQuantumRateModel:
    """Exact listed low transitions and a positive conditional kernel above.

    'Exact' means the supplied GVPT2 result, not experimental exactness. The
    integer cutoff makes this a discrete hybrid; no continuity across a moving
    geometry or changing reference calculation is inferred by this construction.
    """
    def __init__(self, tail):
        from .gvpt2_conditional_rates import ConditionalQuantumRateModel
        if not isinstance(tail, ConditionalQuantumRateModel) or not isinstance(tail.base, PolynomialQuantumEnergyModel):
            raise TypeError('conditional tail with generalized energies required')
        self.tail = tail
        ref = tail.base.low_reference
        keys = [_key(n, s, b) for n, s, b in zip(ref.occupations, ref.sectors, ref.branches)]
        self._exact_rates = {frozenset((keys[a], keys[b])): (float(rate), bool(allowed))
            for (a, b), rate, allowed in zip(ref.pairs, ref.rates_s_inverse, ref.allowed)}
        if tail.base.window_reference is not None:
            ref = tail.base.window_reference
            keys = [_key(n, s, b) for n, s, b in zip(ref.occupations, ref.sectors, ref.branches)]
            for (a, b), rate, allowed in zip(ref.pairs, ref.rates_s_inverse, ref.allowed):
                key, value = frozenset((keys[a], keys[b])), (float(rate), bool(allowed))
                if key in self._exact_rates and self._exact_rates[key] != value:
                    raise ValueError('inconsistent low/window reference rates')
                self._exact_rates[key] = value

    def __getattr__(self, name):
        return getattr(self.tail, name)

    @property
    def diagnostics(self):
        reference = self.tail.base.low_reference
        return dict(schema=SCHEMA, exact_quanta=self.exact_quanta,
            exact_low_states=len(reference.occupations), exact_listed_low_pairs=len(reference.pairs),
            energy_window=self.tail.base.energy_window, total_exact_listed_pairs=len(self._exact_rates),
            missing_allowed_low_pairs='reference_required',
            energy_selection=self.tail.base.diagnostics['polynomial_selection'],
            rate_selection=self.tail.diagnostics, physical_domain_qualified=False)

    @property
    def extended_rate_identity(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest()

    def rates(self, initial, final, *, allowed, initial_sectors=None, final_sectors=None,
              initial_branches=None, final_branches=None, allow_extrapolation=False):
        a, b = _occupations(initial), _occupations(final)
        mask = np.asarray(allowed)
        if a.shape != b.shape or mask.shape != (len(a),) or mask.dtype.kind != 'b':
            raise ValueError('matched endpoints and explicit boolean mask required')
        sa, sb = _labels(initial_sectors, len(a), 'scalar'), _labels(final_sectors, len(b), 'scalar')
        ba, bb = _labels(initial_branches, len(a), 'reference'), _labels(final_branches, len(b), 'reference')
        ca = dict(sectors=sa, branches=ba, allow_extrapolation=allow_extrapolation)
        cb = dict(sectors=sb, branches=bb, allow_extrapolation=allow_extrapolation)
        gap = self.energies(a, **ca)-self.energies(b, **cb)
        exact = ((a.sum(axis=1, dtype=object) <= self.exact_quanta)
                 & (b.sum(axis=1, dtype=object) <= self.exact_quanta))
        if np.any(np.all(a == b, axis=1) & (np.asarray(sa) == sb) & (np.asarray(ba) == bb)):
            raise ValueError('distinct resolved endpoints required')
        result = np.zeros(len(a))
        handled = exact.copy()
        for i in range(len(a)):
            key = frozenset((_key(a[i], sa[i], ba[i]), _key(b[i], sb[i], bb[i])))
            if key not in self._exact_rates:
                if exact[i] and mask[i]:
                    raise ValueError('missing exact low transition; acquire reference pair')
                continue
            handled[i] = True
            rate, permitted = self._exact_rates[key]
            if permitted != mask[i]:
                raise ValueError('low-transition symmetry mask differs from reference')
            if gap[i] > 0:
                result[i] = rate
        rows = np.flatnonzero(~handled)
        if len(rows):
            result[rows] = self.tail.rates(a[rows], b[rows], allowed=mask[rows],
                initial_sectors=tuple(sa[i] for i in rows), final_sectors=tuple(sb[i] for i in rows),
                initial_branches=tuple(ba[i] for i in rows), final_branches=tuple(bb[i] for i in rows),
                allow_extrapolation=allow_extrapolation)
        return result

    cascade_channels = EffectiveQuantumRateModel.cascade_channels
    sample_wang_landau = EffectiveQuantumRateModel.sample_wang_landau

    def to_dict(self):
        return dict(schema=SCHEMA, tail=self.tail.to_dict())

    save = EffectiveQuantumRateModel.save

    @classmethod
    def load(cls, path):
        from pathlib import Path
        from .gvpt2_conditional_rates import ConditionalQuantumRateModel
        data = json.loads(Path(path).read_text())
        if set(data) != {'schema', 'tail'} or data['schema'] != SCHEMA:
            raise ValueError('unsupported hybrid-rate artifact')
        tail = data['tail']
        if set(tail) != {'schema', 'base', 'heads', 'diagnostics'} or tail['schema'] != 'matrix.trinity.gvpt2-conditional-rates.v1':
            raise ValueError('unsupported conditional tail')
        return cls(ConditionalQuantumRateModel(PolynomialQuantumEnergyModel.from_dict(tail['base']),
                                              tail['heads'], tail['diagnostics']))


def fit_hybrid_quantum_rates(reference, *, exact_quanta=2, base_model=None,
        energy_controls=None, rate_controls=None):
    """Fit a configurable exact-low/predicted-high model from calibration only.

    Supply every low resolved energy needed by the intended domain and every
    allowed low transition that will be queried. Low reference values are exact
    lookups, so they must be excluded from prediction-accuracy statistics.
    """
    from .gvpt2_rate_selection import fit_selected_gvpt2_rate_model
    from .gvpt2_conditional_rates import fit_transitionwise_quantum_rates
    if base_model is None:
        base_model = fit_selected_gvpt2_rate_model(reference)
    energy = fit_polynomial_quantum_energies(reference, base_model, exact_quanta=exact_quanta,
                                           **({} if energy_controls is None else energy_controls))
    tail = fit_transitionwise_quantum_rates(reference, base_model=energy,
                                          **({} if rate_controls is None else rate_controls))
    return HybridQuantumRateModel(tail)


def with_exact_energy_window(model, reference, *, initial_energy_cm, half_width_cm,
        maximum_exact_states=512, maximum_exact_pairs=10000):
    """Attach supplied GVPT2 states around an initial energy and their endpoints.

    Select all listed downward transitions whose upper state is in the window,
    retaining both endpoint energies, even when the final state is far below it.
    No eigensolve is launched. Completeness concerns the supplied inventory;
    unavailable states/transitions are not implied by the window declaration.
    Existing additional windows are replaced; the exact-low domain is retained.
    """
    from .gvpt2_conditional_rates import ConditionalQuantumRateModel
    if not isinstance(model, HybridQuantumRateModel) or not isinstance(reference, GVPT2RateReference):
        raise TypeError('hybrid model and resolved rate reference required')
    if (reference.coordinate_id != model.coordinate_id or not np.isfinite(initial_energy_cm)
            or not np.isfinite(half_width_cm) or half_width_cm < 0
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                   for v in (maximum_exact_states, maximum_exact_pairs))):
        raise ValueError('compatible reference and finite window/budgets required')
    source = abs(reference.energies_cm-initial_energy_cm) <= half_width_cm
    a, b = reference.pairs.T
    high = np.where(reference.energies_cm[a] >= reference.energies_cm[b], a, b)
    selected_pairs = source[high]
    if not np.any(selected_pairs):
        raise ValueError('no listed transitions originate in the requested window')
    selected_states = source.copy()
    selected_states[reference.pairs[selected_pairs].ravel()] = True
    if selected_states.sum() > maximum_exact_states or selected_pairs.sum() > maximum_exact_pairs:
        raise MemoryError('exact initial-energy window exceeds declared budget')
    indices = np.full(len(source), -1, int)
    indices[selected_states] = np.arange(sum(selected_states))
    extra = GVPT2RateReference(reference.occupations[selected_states], reference.energies_cm[selected_states],
        indices[reference.pairs[selected_pairs]], reference.rates_s_inverse[selected_pairs],
        reference.allowed[selected_pairs], reference.coordinate_id,
        tuple(np.asarray(reference.sectors)[selected_states]), tuple(np.asarray(reference.branches)[selected_states]),
        dict(reference.provenance))
    window = dict(initial_energy_cm=float(initial_energy_cm), half_width_cm=float(half_width_cm),
        source_states=int(sum(source)), endpoint_inclusive_states=int(sum(selected_states)),
        listed_outgoing_pairs=int(sum(selected_pairs)), maximum_exact_states=maximum_exact_states,
        maximum_exact_pairs=maximum_exact_pairs, inventory_completeness_asserted=False)
    energy = replace(model.tail.base, window_reference=extra, energy_window=window)
    return HybridQuantumRateModel(ConditionalQuantumRateModel(energy, model.tail.heads,
        dict(model.tail.diagnostics, exact_energy_window=window)))
