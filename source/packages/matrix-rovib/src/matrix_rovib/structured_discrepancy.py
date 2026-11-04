"""Correlated discrepancy calibration with separate final validation blocks."""

from __future__ import annotations
import numpy as np
from scipy.linalg import solve_triangular, cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from .prediction_budget import _covariance


def _kernel(left, right, lengths, left_groups, right_groups):
    distance = np.sum(((left[:, None, :] - right[None, :, :]) / lengths) ** 2, axis=2)
    return np.exp(-0.5 * distance) * (left_groups[:, None] == right_groups[None, :])


def structured_model_discrepancy(
    features,
    residuals,
    known_covariance,
    prediction_features,
    *,
    lengthscales,
    groups=None,
    prediction_groups=None,
    validation_indices=(),
    calibration_indices=None,
    block_keys=None,
):
    """Calibrate an RBF covariance on calibration rows only.

    Residuals are model-minus-experiment. Report a model-error mean and
    covariance; the physical Hamiltonian and its fitted parameters stay fixed.
    Final validation rows never select the amplitude or kernel lengthscales.
    """
    x = np.asarray(features, float)
    xp = np.asarray(prediction_features, float)
    r = np.asarray(residuals, float)
    lengths = np.asarray(lengthscales, float)
    if (
        x.ndim != 2
        or xp.ndim != 2
        or xp.shape[1] != x.shape[1]
        or r.shape != (len(x),)
        or lengths.shape != (x.shape[1],)
        or np.any(lengths <= 0)
        or not all(np.all(np.isfinite(a)) for a in (x, xp, r, lengths))
    ):
        raise ValueError("invalid discrepancy features, residuals or lengthscales")
    cal = np.arange(len(x)) if calibration_indices is None else np.asarray(calibration_indices, int)
    val = np.asarray(validation_indices, int)
    if (
        len(cal) < 3
        or len(set(cal.tolist())) != len(cal)
        or np.any(cal < 0)
        or np.any(cal >= len(x))
    ):
        raise ValueError("at least three unique calibration rows are required")
    if (
        len(set(val.tolist())) != len(val)
        or np.any(val < 0)
        or np.any(val >= len(x))
        or set(cal) & set(val)
    ):
        raise ValueError("calibration and final-validation rows must be disjoint")
    if block_keys is not None:
        keys = np.asarray(block_keys)
        if keys.shape != (len(x),) or set(keys[cal]) & set(keys[val]):
            raise ValueError("a physical block occurs in both calibration and final validation")
    group = np.zeros(len(x), int) if groups is None else np.asarray(groups)
    pgroup = np.zeros(len(xp), int) if prediction_groups is None else np.asarray(prediction_groups)
    if group.shape != (len(x),) or pgroup.shape != (len(xp),):
        raise ValueError("one group label is required per row")
    known = _covariance(known_covariance, len(x), "discrepancy residual")
    cc = known[np.ix_(cal, cal)]
    chol = np.linalg.cholesky(cc)
    k = _kernel(x[cal], x[cal], lengths, group[cal], group[cal])
    whitened = solve_triangular(chol, k, lower=True)
    whitened = solve_triangular(chol, whitened.T, lower=True).T
    eigen, u = np.linalg.eigh((whitened + whitened.T) / 2)
    eigen = np.maximum(eigen, 0.0)
    z = u.T @ solve_triangular(chol, r[cal], lower=True)
    scale = max(float(np.mean(r[cal] ** 2)), float(np.max(np.diag(cc))), np.finfo(float).tiny)

    def objective(a):
        diagonal = 1.0 + a * scale * eigen
        return float(np.sum(np.log(diagonal) + z * z / diagonal))

    upper = 10.0
    while objective(upper) < objective(upper / 2):
        upper *= 2
        if upper > 1.0e12:
            raise ValueError("discrepancy variance is not numerically bounded")
    optimized = minimize_scalar(
        objective, bounds=(0.0, upper), method="bounded", options=dict(xatol=1.0e-11)
    )
    variance = 0.0 if objective(0.0) <= objective(optimized.x) else float(optimized.x * scale)
    factor = cho_factor(cc + variance * k, lower=True)
    alpha = cho_solve(factor, r[cal])
    cross = variance * _kernel(xp, x[cal], lengths, pgroup, group[cal])
    solved = cho_solve(factor, cross.T)
    mean = cross @ alpha
    diagonal = np.maximum(variance - np.sum(cross * solved.T, axis=1), 0.0)
    validation = None
    if len(val):
        kv = variance * _kernel(x[val], x[cal], lengths, group[val], group[cal])
        total_cross = kv + known[np.ix_(val, cal)]
        center = total_cross @ alpha
        cv = (
            known[np.ix_(val, val)]
            + variance * _kernel(x[val], x[val], lengths, group[val], group[val])
            - total_cross @ cho_solve(factor, total_cross.T)
        )
        check = cho_factor((cv + cv.T) / 2, lower=True)
        error = r[val] - center
        validation = dict(
            indices=val.tolist(),
            count=len(val),
            raw_rms_MHz=float(np.sqrt(np.mean(r[val] ** 2))),
            conditional_rms_MHz=float(np.sqrt(np.mean(error**2))),
            chi_square=float(error @ cho_solve(check, error)),
            marginal_95_coverage=float(
                np.mean(abs(error) <= 1.959963984540054 * np.sqrt(np.diag(cv)))
            ),
            role="final validation; excluded from calibration and kernel choices",
        )
    return dict(
        schema="matrix.lock.structured_discrepancy.v1",
        calibration_indices=cal.tolist(),
        variance_MHz2=variance,
        lengthscales=lengths.tolist(),
        kernel="RBF within declared groups",
        model_error_mean_MHz=mean.tolist(),
        model_error_standard_uncertainty_MHz=np.sqrt(diagonal).tolist(),
        final_validation=validation,
        physical_predictions_corrected=False,
        scope="conditional GP model-discrepancy diagnostic; hyperparameter uncertainty excluded",
    )


def spectral_block_splits(block_keys, *, eligible=None):
    """Whole physical blocks are withheld; deterministic labels and row order."""
    keys = tuple(str(k) for k in block_keys)
    allowed = set(range(len(keys))) if eligible is None else set(eligible)
    if not allowed or min(allowed) < 0 or max(allowed) >= len(keys):
        raise ValueError("invalid block-validation rows")
    groups = sorted({keys[k] for k in allowed})
    if len(groups) < 2:
        raise ValueError("block validation requires at least two physical blocks")
    return [
        dict(
            block=group,
            training=[i for i in sorted(allowed) if keys[i] != group],
            validation=[i for i in sorted(allowed) if keys[i] == group],
        )
        for group in groups
    ]


def cross_validate_lam_blocks(systems, request, block_keys):
    """Refit each training block split from the declared QM initial values."""
    from .lam_global_fit import fit_global_lam_spectra
    from .lam_checkpoint import copy_workspace_request

    observations = [o for d in request["datasets"] for o in d["request"]["observations"]]
    if len(block_keys) != len(observations):
        raise ValueError("one physical block per observation is required")
    eligible = [i for i, o in enumerate(observations) if o.get("role", "training") == "training"]
    reports = []
    for split in spectral_block_splits(block_keys, eligible=eligible):
        local = copy_workspace_request(request)
        for key in (
            "checkpoint_path",
            "resume_checkpoint",
            "workspace_checkpoint_path",
            "resume_workspace",
        ):
            local.get("options", {}).pop(key, None)
        for dataset in local["datasets"]:
            for key in (
                "checkpoint_path",
                "resume_checkpoint",
                "workspace_checkpoint_path",
                "resume_workspace",
                "optimizer_state",
                "linearization_path",
                "evaluation_log_path",
                "iteration_trace_path",
            ):
                dataset["request"].get("options", {}).pop(key, None)
        for key in ("structured_discrepancy", "qc_acquisition"):
            local.pop(key, None)
        flat = [o for d in local["datasets"] for o in d["request"]["observations"]]
        for i in eligible:
            flat[i]["role"] = "training" if i in split["training"] else "validation"
        result = fit_global_lam_spectra(systems, local)
        residual = np.array(
            [result["comparisons"][i]["calculated_minus_observed_MHz"] for i in split["validation"]]
        )
        reports.append(
            dict(
                split,
                success=result["success"],
                rms_MHz=float(np.sqrt(np.mean(residual**2))),
                variables=result["variables"],
            )
        )
    return dict(
        schema="matrix.lock.blocked_validation.v1",
        folds=reports,
        all_converged=all(r["success"] for r in reports),
        initialization="same declared QM values in every fold; no full-fit warm start",
    )


def fitted_spectral_discrepancy(
    request, fit, physical_jacobian, *, training_whitener, training_jacobian
):
    """Attach a calibrated diagnostic using only rows withheld from fitting.

    Feature columns (e.g. J, K, LAM state) and physical block/group labels are
    explicit data; no molecular naming rule is inferred. Calibration and final
    validation indices refer to the original observation ordering.
    """
    spec = dict(request["structured_discrepancy"])
    intervals = spec.pop("intervals", None)
    obs = request["observations"]
    n = len(obs)
    training = np.array(
        [i for i, o in enumerate(obs) if o.get("role", "training") == "training"], int
    )
    held = np.array(
        [i for i, o in enumerate(obs) if o.get("role", "training") == "validation"], int
    )
    cal = list(spec.pop("calibration_indices"))
    val = list(spec.pop("validation_indices"))
    if set(cal + val) - set(held):
        raise ValueError(
            "discrepancy calibration/final validation must be withheld from the physical fit"
        )
    if not fit.get("success"):
        raise ValueError("structured discrepancy requires a converged identifiable physical fit")
    j = np.asarray(physical_jacobian, float)
    cov = np.asarray(fit["covariance"], float)
    weight = np.sqrt([o.get("weight", 1.0) for o in obs])
    sig = np.array([o["uncertainty_MHz"] for o in obs]) / weight
    raw = request.get("covariance_MHz2")
    c = (
        np.diag(sig * sig)
        if raw is None
        else np.asarray(raw, float) / weight[:, None] / weight[None, :]
    )
    known = c[np.ix_(held, held)] + j[held] @ cov @ j[held].T
    cross = training_whitener.apply(c[np.ix_(training, held)])
    shared = j[held] @ cov @ np.asarray(training_jacobian).T @ cross
    known -= shared + shared.T
    if intervals is not None:
        from .spectral_discrepancy_intervals import spectroscopic_discrepancy_features, spectral_discrepancy_intervals
        if any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) for i in cal+val):
            raise ValueError('calibration/validation rows must contain integer indices')
        descriptors = ({} if all(k in spec for k in ('features','groups','block_keys'))
                       else spectroscopic_discrepancy_features(obs))
        names = spec.pop('feature_names', None if 'features' in spec else descriptors['feature_names'])
        features = np.asarray(spec.pop('features', descriptors.get('features')), float)
        keys = np.asarray(spec.pop('block_keys', descriptors.get('block_keys'))).astype(str)
        families = np.asarray(spec.pop('groups', descriptors.get('groups')))
        if features.ndim != 2 or len(features) != n or keys.shape != (n,) or families.shape != (n,):
            raise ValueError('one feature/family/block record per observation required')
        if set(keys[training]) & (set(keys[cal]) | set(keys[val])):
            raise ValueError('physical-fit blocks overlap discrepancy calibration or validation')
        positions = {int(row): k for k, row in enumerate(held)}
        options = dict(intervals)
        if 'numerical_envelope_MHz' in options:
            options['numerical_envelope_MHz'] = np.asarray(options['numerical_envelope_MHz'])[held]
        report = spectral_discrepancy_intervals(features[held],
            np.array([r['calculated_minus_observed_MHz'] for r in fit['comparisons']])[held],known,
            np.array([r['calculated_MHz'] for r in fit['comparisons']])[held],
            calibration_indices=[positions[i] for i in cal],validation_indices=[positions[i] for i in val],
            block_keys=keys[held],groups=families[held],feature_names=names,**spec,**options)
        report['original_calibration_indices']=cal
        report['original_validation_indices']=val
        report['original_prediction_indices']=held.tolist()
        return report
    features = np.asarray(spec.pop("features"), float)
    if features.ndim != 2 or len(features) != n:
        raise ValueError("one feature row per observation is required")
    options = {}
    for key in ("groups", "block_keys"):
        if key in spec:
            values = np.asarray(spec.pop(key))
            options[key] = values[held]
            if key == "groups":
                options["prediction_groups"] = values
    positions = {int(row): k for k, row in enumerate(held)}
    report = structured_model_discrepancy(
        features[held],
        np.array([r["calculated_minus_observed_MHz"] for r in fit["comparisons"]])[held],
        known,
        features,
        calibration_indices=[positions[i] for i in cal],
        validation_indices=[positions[i] for i in val],
        **options,
        **spec,
    )
    report["original_calibration_indices"] = cal
    report["original_validation_indices"] = val
    return report
