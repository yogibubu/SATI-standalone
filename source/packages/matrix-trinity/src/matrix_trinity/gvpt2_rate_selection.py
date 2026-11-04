"""Calibration-only selection of stable effective-quantum energy/rate models."""
from dataclasses import replace
import hashlib

import numpy as np

from .gvpt2_rates import GVPT2RateReference
from .gvpt2_rate_model import fit_gvpt2_rate_model


def subset_reference(reference, selected):
    """Retain whole states and only measured transitions between those states."""
    selected = np.asarray(selected)
    if selected.shape != (len(reference.occupations),) or selected.dtype.kind != 'b':
        raise ValueError('one boolean per reference state required')
    edges = np.all(selected[reference.pairs], axis=1)
    mapping = np.full(len(selected), -1, int)
    mapping[selected] = np.arange(sum(selected))
    return GVPT2RateReference(reference.occupations[selected], reference.energies_cm[selected],
        mapping[reference.pairs[edges]], reference.rates_s_inverse[edges], reference.allowed[edges],
        reference.coordinate_id, tuple(np.asarray(reference.sectors)[selected]),
        tuple(np.asarray(reference.branches)[selected]), dict(reference.provenance))


def prediction_metrics(model, reference, selected_states, selected_pairs):
    """Energy, rate and conditional branching errors on fixed resolved endpoints."""
    n = reference.occupations
    labels = dict(sectors=reference.sectors, branches=reference.branches, allow_extrapolation=True)
    energies = model.energies(n, **labels)
    a, b = reference.pairs.T
    high = np.where(reference.energies_cm[a] > reference.energies_cm[b], a, b)
    low = np.where(high == a, b, a)
    rates = model.rates(n[high], n[low], allowed=reference.allowed,
        initial_sectors=tuple(reference.sectors[i] for i in high),
        final_sectors=tuple(reference.sectors[i] for i in low),
        initial_branches=tuple(reference.branches[i] for i in high),
        final_branches=tuple(reference.branches[i] for i in low), allow_extrapolation=True)
    errors = energies[selected_states]-reference.energies_cm[selected_states]
    ref_total = np.bincount(high, weights=reference.rates_s_inverse, minlength=len(n))
    pred_total = np.bincount(high, weights=rates, minlength=len(n))
    pref = np.divide(reference.rates_s_inverse, ref_total[high], out=np.zeros(len(high)), where=ref_total[high] > 0)
    pred = np.divide(rates, pred_total[high], out=np.zeros(len(high)), where=pred_total[high] > 0)
    tv = .5*np.bincount(high, weights=abs(pred-pref), minlength=len(n))
    tv[(ref_total > 0) & (pred_total == 0)] = 1.
    bright = selected_states & (ref_total > 0)
    denominator = reference.rates_s_inverse[selected_pairs].sum()
    if not len(errors) or denominator <= 0 or not np.any(bright):
        raise ValueError('validation requires excluded states with positive measured radiative rates')
    return dict(energy_rmse_cm=float(np.sqrt(np.mean(errors**2))),
        rate_relative_l1=float(abs(rates[selected_pairs]-reference.rates_s_inverse[selected_pairs]).sum()/denominator),
        branching_total_variation_mean=float(tv[bright].mean()))


def fit_selected_gvpt2_rate_model(reference, *, folds=3, candidates=None,
        energy_tolerance_cm=5., rate_relative_tolerance=.1, branching_tolerance=.05,
        rate_floor_s_inverse=1e-3, max_nfev=1000):
    """Select regularization on internal state folds, then refit calibration data.

    The caller's external test data never enter this function. Ground/pure-mode
    anchors and one witness edge per channel/category stay in every inner fit.
    Complexity is selected jointly on energies, rates and branching, not on
    the optimizer flag. Existing model/rate/artifact/sampler interfaces survive.
    """
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError('at least two integer folds required')
    if any(not np.isfinite(v) or v <= 0 for v in (energy_tolerance_cm, rate_relative_tolerance, branching_tolerance)):
        raise ValueError('positive finite validation tolerances required')
    from .gvpt2_rate_model import _channel
    n = reference.occupations
    anchors = np.count_nonzero(n, axis=1) <= 1
    for category in set(zip(reference.sectors, reference.branches)):
        members = [i for i, cat in enumerate(zip(reference.sectors, reference.branches)) if cat == category]
        anchors[min(members, key=lambda i: (sum(n[i]), tuple(n[i])))] = True
    witnesses = {}
    for a, b in reference.pairs[reference.allowed]:
        key = _channel(n[a], n[b], reference.sectors[a], reference.sectors[b])
        pair = (int(a), int(b))
        if key not in witnesses or (sum(n[a])+sum(n[b]), pair) < witnesses[key][0]:
            witnesses[key] = ((sum(n[a])+sum(n[b]), pair), pair)
    for _, pair in witnesses.values():
        anchors[list(pair)] = True
    fold = np.array([int(hashlib.sha256(np.asarray(row, '<i8').tobytes()+
        (reference.sectors[i]+'\0'+reference.branches[i]).encode()).hexdigest()[:8], 16) % folds
        for i, row in enumerate(n)])
    if candidates is None:
        candidates = [dict(rate_degree=1, latent_penalty=100., coefficient_penalty=.1),
                      dict(rate_degree=2, latent_penalty=100., coefficient_penalty=.1),
                      dict(rate_degree=2, latent_penalty=100., coefficient_penalty=10.)]
    common = dict(anchor_ground_energy=True, latent_parameter_bound=1.,
        energy_tolerance_cm=energy_tolerance_cm, rate_floor_s_inverse=rate_floor_s_inverse,
        max_nfev=max_nfev)
    records = []
    for configuration in candidates:
        configuration = dict(configuration)
        if set(configuration)-{'rate_degree', 'latent_penalty', 'coefficient_penalty'}:
            raise ValueError('candidate may select rate degree and the two regularization strengths')
        values, failure = [], None
        for f in range(folds):
            validation = (fold == f) & ~anchors
            pairs = np.any(validation[reference.pairs], axis=1)
            if not np.any(validation) or not np.any(pairs):
                raise ValueError('reference too small for the requested inner state folds')
            try:
                fitted = fit_gvpt2_rate_model(subset_reference(reference, ~validation), **common, **configuration)
                metrics = prediction_metrics(fitted, reference, validation, pairs)
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                failure = str(exc)
                break
            score = ((metrics['energy_rmse_cm']/energy_tolerance_cm)**2+
                     (metrics['rate_relative_l1']/rate_relative_tolerance)**2+
                     (metrics['branching_total_variation_mean']/branching_tolerance)**2)
            values.append(dict(fold=f, score=score, **metrics))
        records.append(dict(controls=configuration, folds=values, failure=failure,
            score=None if failure else float(np.mean([v['score'] for v in values]))))
    valid = [i for i, record in enumerate(records) if record['score'] is not None]
    if not valid:
        raise RuntimeError(f'no candidate passed internal calibration fits: {records}')
    winner = min(valid, key=lambda i: records[i]['score'])
    model = fit_gvpt2_rate_model(reference, **common, **records[winner]['controls'])
    diagnostics = dict(model.diagnostics, internal_model_selection=dict(
        folds=folds, state_anchors=int(sum(anchors)), selected_candidate=winner,
        external_test_used=False, candidates=records,
        energy_tolerance_cm=energy_tolerance_cm, rate_relative_tolerance=rate_relative_tolerance,
        branching_tolerance=branching_tolerance))
    return replace(model, diagnostics=diagnostics)
