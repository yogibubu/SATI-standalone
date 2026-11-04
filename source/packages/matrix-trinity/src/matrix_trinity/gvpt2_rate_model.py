"""Joint algebraic GVPT2 energy/rate surrogate in effective quantum numbers.

Fits resolved spontaneous-emission A coefficients, not spectral intensities.
Integer occupations remain the state-counting grid. A shared explicit map q(n,
symmetry,branch) enters both energies and nonnegative rate expressions. No HDC
prescription or state-specific spectral solve enters training or inference.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import gammaln

from .gvpt2_rates import GVPT2RateReference, _labels, _occupations, _readonly

SCHEMA = "matrix.trinity.gvpt2-effective-quantum-rates.v1"


def _channel(a, b, sa, sb):
    delta = b-a
    nonzero = np.flatnonzero(delta)
    if len(nonzero) and delta[nonzero[0]] < 0:
        delta = -delta
    return json.dumps([delta.tolist(), sorted((sa, sb))], separators=(",", ":"))


def _ladder_factor(a, b):
    log_factor = np.sum(gammaln(np.maximum(a, b)+1.)-gammaln(np.minimum(a, b)+1.), axis=1)
    with np.errstate(over="raise", invalid="raise"):
        return np.exp(log_factor)


def _latent_design(n, sectors, branches, categories, pairs, scales):
    modes = n.shape[1]
    columns, targets = [], []
    for i in range(modes):
        columns.append(n[:, i]*(n[:, i]-1)/scales[i])
        targets.append(i)
    for i, j in pairs:
        columns.extend((n[:, i]*n[:, j]/scales[j], n[:, i]*n[:, j]/scales[i]))
        targets.extend((i, j))
    labels = tuple(zip(sectors, branches))
    if any(label not in categories for label in labels):
        raise ValueError("unknown symmetry/branch category; reference calibration required")
    for category in categories[1:]:
        indicator = np.array([label == category for label in labels], float)
        for i in range(modes):
            columns.append(indicator)
            targets.append(i)
    return np.column_stack(columns), np.asarray(targets, int)


def _effective(n, design, targets, parameters):
    q = n.astype(float).copy()
    for i in range(n.shape[1]):
        selected = np.flatnonzero(targets == i)
        q[:, i] += design[:, selected] @ parameters[selected]
    return q


def _energy(q, coefficients, scales, pairs):
    z = q/scales
    features = np.column_stack([np.ones(len(q)), z, q*(q-1)/scales**2,
        *[z[:, i]*z[:, j] for i, j in pairs]])
    modes = q.shape[1]
    gradient = np.broadcast_to(coefficients[1:1+modes]/scales, q.shape).copy()
    gradient += coefficients[1+modes:1+2*modes]*(2*q-1)/scales**2
    for coefficient, (i, j) in zip(coefficients[1+2*modes:], pairs):
        gradient[:, i] += coefficient*q[:, j]/(scales[i]*scales[j])
        gradient[:, j] += coefficient*q[:, i]/(scales[i]*scales[j])
    return features @ coefficients, features, gradient


def _rate_features(qa, qb, scales, pairs, degree, coefficients=None):
    midpoint, difference = (qa+qb)/(2*scales), (qa-qb)/scales
    columns = [np.ones(len(qa)), midpoint]
    if degree == 2:
        columns.extend((midpoint**2, *[midpoint[:, i]*midpoint[:, j] for i, j in pairs], difference**2))
    features = np.column_stack(columns)
    if coefficients is None:
        return features
    modes = qa.shape[1]
    gm = coefficients[:, 1:1+modes].copy()
    gd = np.zeros_like(gm)
    if degree == 2:
        gm += 2*coefficients[:, 1+modes:1+2*modes]*midpoint
        start = 1+2*modes
        for k, (i, j) in enumerate(pairs):
            gm[:, i] += coefficients[:, start+k]*midpoint[:, j]
            gm[:, j] += coefficients[:, start+k]*midpoint[:, i]
        gd = 2*coefficients[:, start+len(pairs):]*difference
    amplitude = np.einsum("ij,ij->i", features, coefficients)
    return amplitude, features, (.5*gm+gd)/scales, (.5*gm-gd)/scales


@dataclass(frozen=True)
class EffectiveQuantumRateModel:
    coordinate_id: str
    scales: np.ndarray
    mode_pairs: tuple
    categories: tuple
    latent_parameters: np.ndarray
    energy_coefficients_cm: np.ndarray
    rate_coefficients: np.ndarray
    rate_scales: np.ndarray
    channels: tuple
    rate_degree: int
    maximum_occupations: np.ndarray
    maximum_total_quanta: int
    provenance: dict
    diagnostics: dict

    def __post_init__(self):
        for name in ("scales", "latent_parameters", "energy_coefficients_cm", "rate_coefficients",
                     "rate_scales", "maximum_occupations"):
            value = np.asarray(getattr(self, name))
            if not np.all(np.isfinite(value)):
                raise ValueError("finite model coefficients required")
            object.__setattr__(self, name, _readonly(value))
        m, p, c = len(self.scales), len(self.mode_pairs), len(self.categories)
        if (not m or self.scales.shape != (m,) or np.any(self.scales < 1)
                or self.rate_degree not in (1, 2) or not c
                or len(set(self.categories)) != c or len(set(self.channels)) != len(self.channels)
                or any(len(cat) != 2 or any(not isinstance(s, str) or not s for s in cat) for cat in self.categories)
                or any(not 0 <= i < j < m for i, j in self.mode_pairs)
                or len(set(self.mode_pairs)) != p):
            raise ValueError("invalid effective-number model structure")
        f = 1+m if self.rate_degree == 1 else 1+3*m+p
        if (self.latent_parameters.shape != (m+2*p+(c-1)*m,)
                or self.energy_coefficients_cm.shape != (1+2*m+p,)
                or self.rate_coefficients.shape != (len(self.channels), f)
                or self.rate_scales.shape != (len(self.channels),) or np.any(self.rate_scales <= 0)
                or self.maximum_occupations.shape != (m,) or self.maximum_occupations.dtype.kind not in "iu"
                or np.any(self.maximum_occupations < 0)
                or not isinstance(self.maximum_total_quanta, int) or self.maximum_total_quanta < 0
                or not isinstance(self.coordinate_id, str) or not self.coordinate_id):
            raise ValueError("inconsistent effective-number coefficient dimensions/domain")

    def effective_numbers(self, occupations, *, sectors=None, branches=None, allow_extrapolation=False):
        n = _occupations(occupations)
        if n.shape[1] != len(self.scales):
            raise ValueError("occupation mode count differs from fitted model")
        if not allow_extrapolation and (np.any(n > self.maximum_occupations)
                or np.any(n.sum(axis=1, dtype=object) > self.maximum_total_quanta)):
            raise ValueError("outside fitted occupation envelope; explicit extrapolation consent required")
        sectors = _labels(sectors, len(n), "scalar")
        branches = _labels(branches, len(n), "reference")
        design, targets = _latent_design(n.astype(float), sectors, branches,
            self.categories, self.mode_pairs, self.scales)
        q = _effective(n, design, targets, self.latent_parameters)
        if not np.all(np.isfinite(q)):
            raise FloatingPointError("effective quantum numbers exceed numerical range")
        return q

    def energies(self, occupations, *, sectors=None, branches=None, allow_extrapolation=False):
        q = self.effective_numbers(occupations, sectors=sectors, branches=branches,
                                  allow_extrapolation=allow_extrapolation)
        energies = _energy(q, self.energy_coefficients_cm, self.scales, self.mode_pairs)[0]
        if not np.all(np.isfinite(energies)):
            raise FloatingPointError("surrogate energies exceed numerical range")
        return energies

    def rates(self, initial, final, *, allowed, initial_sectors=None, final_sectors=None,
              initial_branches=None, final_branches=None, allow_extrapolation=False):
        """Downward A in s^-1; explicit symmetry masks may encode any group.

        Missing calibrated channels raise, rather than being interpreted as
        forbidden. A zero/negative emission gap gives zero spontaneous A.
        """
        a, b = _occupations(initial), _occupations(final)
        mask = np.asarray(allowed)
        if a.shape != b.shape or mask.shape != (len(a),) or mask.dtype.kind != "b":
            raise ValueError("matched endpoint batches and explicit boolean symmetry mask required")
        sa, sb = _labels(initial_sectors, len(a), "scalar"), _labels(final_sectors, len(b), "scalar")
        qa = self.effective_numbers(a, sectors=sa, branches=initial_branches, allow_extrapolation=allow_extrapolation)
        qb = self.effective_numbers(b, sectors=sb, branches=final_branches, allow_extrapolation=allow_extrapolation)
        gap = (_energy(qa, self.energy_coefficients_cm, self.scales, self.mode_pairs)[0]
               -_energy(qb, self.energy_coefficients_cm, self.scales, self.mode_pairs)[0])
        if not np.all(np.isfinite(gap)):
            raise FloatingPointError("surrogate emission gaps exceed numerical range")
        selected = np.flatnonzero(mask & (gap > 0))
        values = np.zeros(len(a))
        lookup = {key: i for i, key in enumerate(self.channels)}
        keys = [_channel(a[i], b[i], sa[i], sb[i]) for i in selected]
        if any(key not in lookup for key in keys):
            raise ValueError("unmeasured transition family; extend GVPT2 reference")
        if len(selected):
            indices = np.array([lookup[key] for key in keys])
            f = _rate_features(qa[selected], qb[selected], self.scales, self.mode_pairs, self.rate_degree)
            amplitude = np.einsum("ij,ij->i", f, self.rate_coefficients[indices])
            with np.errstate(over="raise", invalid="raise"):
                values[selected] = gap[selected]**3*_ladder_factor(a[selected], b[selected])*self.rate_scales[indices]*amplitude**2
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("surrogate A coefficients exceed numerical range")
        return values

    def cascade_channels(self, occupations, pairs, *, allowed, sectors=None, branches=None,
                         complete_radiative_endpoints=False, allow_extrapolation=False):
        """Conditional branching over an explicit endpoint list; dark totals stay zero.

        Lifetimes are returned only if the caller asserts that all radiative
        endpoints are present. This assertion is not a completeness proof.
        """
        n = _occupations(occupations)
        s, b = _labels(sectors, len(n), "scalar"), _labels(branches, len(n), "reference")
        e = self.energies(n, sectors=s, branches=b, allow_extrapolation=allow_extrapolation)
        pairs = np.asarray(pairs)
        if (pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu"
                or np.any(pairs < 0) or np.any(pairs >= len(n))
                or np.any(pairs[:, 0] == pairs[:, 1])
                or len({tuple(sorted(p)) for p in pairs}) != len(pairs)):
            raise ValueError("distinct measured endpoint pairs required")
        first, second = pairs.T
        high = np.where(e[first] >= e[second], first, second)
        low = np.where(e[first] >= e[second], second, first)
        rates = self.rates(n[high], n[low], allowed=allowed,
            initial_sectors=tuple(s[i] for i in high), final_sectors=tuple(s[i] for i in low),
            initial_branches=tuple(b[i] for i in high), final_branches=tuple(b[i] for i in low),
            allow_extrapolation=allow_extrapolation)
        totals = np.bincount(high, weights=rates, minlength=len(n))
        probabilities = np.divide(rates, totals[high], out=np.zeros_like(rates), where=totals[high] > 0)
        lifetimes = None
        if complete_radiative_endpoints:
            lifetimes = np.divide(1., totals, out=np.full_like(totals, np.inf), where=totals > 0)
        return dict(initial_rows=high, final_rows=low, rates_s_inverse=rates,
            branching_probabilities=probabilities, total_listed_rates_s_inverse=totals,
            radiative_lifetimes_s=lifetimes,
            endpoint_completeness_asserted=bool(complete_radiative_endpoints))

    def sample_wang_landau(self, **controls):
        """Sample this complete energy map with SENTINEL's compiled CPU backend.

        See sample_effective_quantum_wang_landau for explicit resolved-state
        category domains and frozen rate/property callback conventions.
        """
        from matrix_sentinel.effective_quantum_rewl import sample_effective_quantum_wang_landau
        return sample_effective_quantum_wang_landau(self, **controls)

    def to_dict(self):
        return dict(schema=SCHEMA, **{name: (getattr(self, name).tolist()
            if isinstance(getattr(self, name), np.ndarray) else getattr(self, name))
            for name in self.__dataclass_fields__})

    @classmethod
    def from_dict(cls, payload):
        if payload.get("schema") != SCHEMA or set(payload) != {"schema", *cls.__dataclass_fields__}:
            raise ValueError("unsupported or incomplete GVPT2 rate-model artifact")
        values = {key: value for key, value in payload.items() if key != "schema"}
        for key in ("mode_pairs", "categories"):
            values[key] = tuple(tuple(row) for row in values[key])
        values["channels"] = tuple(values["channels"])
        return cls(**values)

    def save(self, path):
        def encode(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(f"nonserializable provenance value: {type(value).__name__}")
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False, default=encode)+"\n")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit_gvpt2_rate_model(reference, *, mode_pairs=None, rate_degree=1,
                        energy_tolerance_cm=1., rate_log_tolerance=.05,
                        rate_floor_s_inverse=1e-12, latent_penalty=.01,
                        coefficient_penalty=1e-6, maximum_parameters=12000,
                        maximum_workspace_bytes=512*1024**2, max_nfev=300,
                        anchor_ground_energy=False, latent_parameter_bound=None):
    """Simultaneously fit effective occupations, quadratic energies and A rates.

    The analytic Jacobian reuses the same occupation and channel feature arrays.
    Fit precisions and Tikhonov penalties are numerical model controls, not
    resonance-selection thresholds. No new reference/QC calculation is launched.
    The returned envelope is a calibration bound, not a thermal-domain certificate.
    """
    if not isinstance(reference, GVPT2RateReference):
        raise TypeError("one common GVPT2RateReference required")
    controls = (energy_tolerance_cm, rate_log_tolerance, rate_floor_s_inverse, latent_penalty, coefficient_penalty)
    if any(not np.isfinite(v) or v <= 0 for v in controls) or rate_degree not in (1, 2):
        raise ValueError("positive finite fit controls and rate_degree=1 or 2 required")
    for value in (maximum_parameters, maximum_workspace_bytes, max_nfev):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("positive integer fit budgets required")
    if not isinstance(anchor_ground_energy, bool):
        raise ValueError("anchor_ground_energy must be boolean")
    if latent_parameter_bound is not None and (not np.isfinite(latent_parameter_bound) or latent_parameter_bound <= 0):
        raise ValueError("positive finite latent_parameter_bound required")
    n = reference.occupations.astype(float)
    m, ns = n.shape[1], len(n)
    pairs = tuple((i, j) for i in range(m) for j in range(i+1, m)) if mode_pairs is None else tuple(map(tuple, mode_pairs))
    if any(len(pair) != 2 or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in pair)
           or not 0 <= pair[0] < pair[1] < m for pair in pairs) or len(set(pairs)) != len(pairs):
        raise ValueError("distinct mode pairs with 0 <= i < j < modes required")
    anchor = (reference.sectors[int(np.argmin(reference.energies_cm))], reference.branches[int(np.argmin(reference.energies_cm))])
    categories = (anchor, *sorted(set(zip(reference.sectors, reference.branches))-{anchor}))
    scales = np.maximum(1., np.max(n, axis=0))
    selected = np.flatnonzero(reference.allowed)
    if not len(selected):
        raise ValueError("at least one allowed reference transition required")
    left, right = reference.pairs[selected].T
    keys = [_channel(reference.occupations[a], reference.occupations[b], reference.sectors[a], reference.sectors[b])
            for a, b in zip(left, right)]
    channels = tuple(sorted(set(keys)))
    lookup = {key: i for i, key in enumerate(channels)}
    ci = np.array([lookup[key] for key in keys])
    nl = m+2*len(pairs)+(len(categories)-1)*m
    ne = 1+2*m+len(pairs)
    nr = 1+m if rate_degree == 1 else 1+3*m+len(pairs)
    total = nl+ne+len(channels)*nr
    residual_count = ns+len(selected)+total
    workspace = 8*(5*residual_count*total+4*ns*nl+4*len(selected)*nr)
    if total > maximum_parameters or workspace > maximum_workspace_bytes:
        raise MemoryError(f"joint surrogate needs {total} parameters and about {workspace} bytes; reduce the declared feature basis or raise budgets")
    design, targets = _latent_design(n, reference.sectors, reference.branches, categories, pairs, scales)
    energy_features = _energy(n, np.zeros(ne), scales, pairs)[1]
    e0, _, energy_rank, _ = np.linalg.lstsq(energy_features, reference.energies_cm, rcond=None)
    if anchor_ground_energy:
        zeros = np.flatnonzero(np.all(n == 0, axis=1))
        ground = int(np.argmin(reference.energies_cm))
        if ground not in zeros:
            raise ValueError("ground anchoring requires a zero-occupation lowest reference state")
        e0[0] = reference.energies_cm[ground]
        e0[1:], _, energy_rank, _ = np.linalg.lstsq(
            energy_features[:, 1:], reference.energies_cm-e0[0], rcond=None)
    rate_features = _rate_features(n[left], n[right], scales, pairs, rate_degree)
    gaps = reference.energies_cm[left]-reference.energies_cm[right]
    ladder = _ladder_factor(n[left], n[right])
    y = reference.rates_s_inverse[selected]
    if not np.any(y > 0):
        raise ValueError("positive rates are required to identify the shared effective-number map")
    reduced = np.divide(y, abs(gaps)**3*ladder, out=np.zeros_like(y), where=gaps != 0)
    rate_scales, c0, ranks = np.ones(len(channels)), np.zeros((len(channels), nr)), []
    for channel in range(len(channels)):
        rows = ci == channel
        positive = reduced[rows][reduced[rows] > 0]
        if len(positive):
            rate_scales[channel] = np.median(positive)
        c0[channel], _, rank, _ = np.linalg.lstsq(rate_features[rows], np.sqrt(reduced[rows]/rate_scales[channel]), rcond=None)
        ranks.append(int(rank))
    initial = np.r_[np.zeros(nl), e0, c0.ravel()]
    energy_scale = max(1., float(np.ptp(reference.energies_cm)))
    regularizer = np.r_[np.full(nl, np.sqrt(latent_penalty)),
        np.full(ne, np.sqrt(coefficient_penalty)/energy_scale),
        np.full(total-nl-ne, np.sqrt(coefficient_penalty))]
    rate_precision = np.maximum(y, rate_floor_s_inverse)
    cache = {}

    def evaluate(parameters):
        if "x" in cache and np.array_equal(parameters, cache["x"]):
            return cache["residual"], cache["jacobian"]
        q = _effective(n, design, targets, parameters[:nl])
        energies, ef, eg = _energy(q, parameters[nl:nl+ne], scales, pairs)
        ec = eg[:, targets]*design
        coefficients = parameters[nl+ne:].reshape(len(channels), nr)
        amplitude, rf, ga, gb = _rate_features(q[left], q[right], scales, pairs, rate_degree, coefficients[ci])
        gap = energies[left]-energies[right]
        factor = ladder*rate_scales[ci]
        a = abs(gap)**3*factor*amplitude**2
        residual = np.r_[(energies-reference.energies_cm)/energy_tolerance_cm,
            (np.log1p(a/rate_precision)-np.log1p(y/rate_precision))/rate_log_tolerance,
            regularizer*(parameters-initial)]
        jacobian = np.zeros((residual_count, total))
        jacobian[:ns, :nl] = ec/energy_tolerance_cm
        jacobian[:ns, nl:nl+ne] = ef/energy_tolerance_cm
        dg = 3*gap*abs(gap)*factor*amplitude**2
        da = 2*abs(gap)**3*factor*amplitude
        denominator = (rate_precision+a)*rate_log_tolerance
        rows = slice(ns, ns+len(selected))
        jacobian[rows, :nl] = (dg[:, None]*(ec[left]-ec[right])
            +da[:, None]*(ga[:, targets]*design[left]+gb[:, targets]*design[right]))/denominator[:, None]
        jacobian[rows, nl:nl+ne] = dg[:, None]*(ef[left]-ef[right])/denominator[:, None]
        for channel in range(len(channels)):
            indices = np.flatnonzero(ci == channel)
            jacobian[ns+indices, nl+ne+channel*nr:nl+ne+(channel+1)*nr] = (da[indices]/denominator[indices])[:, None]*rf[indices]
        jacobian[ns+len(selected)+np.arange(total), np.arange(total)] = regularizer
        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(jacobian)):
            raise FloatingPointError("joint fit exceeded numerical range")
        cache.update(x=parameters.copy(), residual=residual, jacobian=jacobian,
                     energies=energies, rates=a, gap=gap)
        return residual, jacobian

    free = np.ones(total, bool)
    if anchor_ground_energy:
        free[nl] = False
    def expand(x):
        parameters = initial.copy()
        parameters[free] = x
        return parameters
    lower, upper = np.full(total, -np.inf), np.full(total, np.inf)
    if latent_parameter_bound is not None:
        lower[:nl], upper[:nl] = -latent_parameter_bound, latent_parameter_bound
    result = least_squares(lambda x: evaluate(expand(x))[0], initial[free],
        jac=lambda x: evaluate(expand(x))[1][:, free], bounds=(lower[free], upper[free]),
        method="trf", x_scale="jac", max_nfev=max_nfev)
    parameters = expand(result.x)
    evaluate(parameters)
    if not result.success:
        raise RuntimeError(f"joint GVPT2 energy/rate fit did not converge: {result.message}")
    if np.any((y > 0) & (cache["gap"]*gaps <= 0)):
        raise ValueError("fit reversed a reference emission ordering; revise the common feature basis")
    diagnostics = dict(schema=SCHEMA, reference_method="GVPT2", hdc_energies_used=False,
        observable="spontaneous_electric_dipole_A_s_inverse", jointly_optimized=True,
        analytic_jacobian=True, fit_success=True, optimizer_message=str(result.message),
        nfev=int(result.nfev), optimality=float(result.optimality), parameter_count=int(sum(free)),
        reference_state_count=ns, measured_pair_count=len(reference.pairs),
        energy_design_rank=int(energy_rank), energy_feature_count=ne,
        free_energy_coefficient_count=ne-int(anchor_ground_energy),
        rate_design_ranks=ranks, rate_feature_count=nr,
        zero_only_channels=[channels[k] for k in range(len(channels)) if not np.any(y[ci == k] > 0)],
        energy_rmse_cm=float(np.sqrt(np.mean((cache["energies"]-reference.energies_cm)**2))),
        rate_scaled_rmse=float(np.sqrt(np.mean(((cache["rates"]-y)/rate_precision)**2))),
        physical_domain_qualified=False, held_out_validation_performed=False,
        envelope_is_not_a_convergence_certificate=True,
        fit_controls=dict(energy_tolerance_cm=energy_tolerance_cm, rate_log_tolerance=rate_log_tolerance,
            rate_floor_s_inverse=rate_floor_s_inverse, latent_penalty=latent_penalty,
            coefficient_penalty=coefficient_penalty, anchor_ground_energy=anchor_ground_energy,
            latent_parameter_bound=latent_parameter_bound))
    return EffectiveQuantumRateModel(reference.coordinate_id, scales, pairs, categories,
        parameters[:nl], parameters[nl:nl+ne], parameters[nl+ne:].reshape(len(channels), nr),
        rate_scales, channels, rate_degree, reference.occupations.max(axis=0),
        int(reference.occupations.sum(axis=1, dtype=object).max()), dict(reference.provenance), diagnostics)
