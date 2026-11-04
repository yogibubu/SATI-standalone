"""Nonlinear profiles and physical-block resampling for structural LAM fits."""

from __future__ import annotations
import numpy as np
from scipy.linalg import solve_triangular
from .fit_convergence import assess_statistical_convergence, statistical_trust_region
from .uncertainty import _profile_interval


def _refit(fun, jac, point, lo, hi, scales, count, *, fixed=None, max_nfev=50, data_transform=None):
    mask = np.ones(len(point), bool)
    template = np.array(point, copy=True)
    if fixed is not None:
        mask[fixed[0]] = False
        template[fixed[0]] = fixed[1]

    def full(z):
        value = template.copy()
        value[mask] = z
        return value

    if not mask.any():
        r = fun(template)
        return dict(success=True, values=template.tolist(), chi_square=float(r @ r))

    class Done(Exception):
        def __init__(self, z):
            self.z = z.copy()

    def r(z):
        return np.asarray(fun(full(z)))

    def j(z):
        derivative = np.asarray(jac(full(z)))[:, mask]
        check = assess_statistical_convergence(
            r(z),
            derivative,
            z,
            lo[mask],
            hi[mask],
            data_count=count,
            data_correction_transform=data_transform,
        )
        if check["converged"]:
            raise Done(np.asarray(z))
        return derivative

    try:
        result = statistical_trust_region(
            r, j, template[mask], lo[mask], hi[mask], scales[mask], max_nfev=max_nfev
        )
        z = result.x
    except Done as end:
        z = end.z
    value = full(z)
    residual = np.asarray(fun(value))
    derivative = np.asarray(jac(value))[:, mask]
    check = assess_statistical_convergence(
        residual,
        derivative,
        z,
        lo[mask],
        hi[mask],
        data_count=count,
        data_correction_transform=data_transform,
    )
    return dict(
        success=check["converged"],
        values=value.tolist(),
        chi_square=float(residual @ residual),
        convergence=check,
    )


def nonlinear_structure_uncertainty(
    fun,
    jac,
    predict,
    point,
    lo,
    hi,
    scales,
    *,
    covariance,
    experimental_covariance,
    observations,
    parameter_names,
    training_indices,
    options=None,
):
    """Reoptimize the actual native spectrum for every profile/resample.

    Full experimental covariance is retained. A block jackknife always uses
    the correct retained marginal. Whole-block bootstrap is offered only when
    blocks are statistically independent; correlated blocks use the jackknife
    and correlated parametric refits. Priors remain fixed (conditional results).
    """
    opts = dict(options or {})
    x = np.asarray(point, float)
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    scale = np.asarray(scales)
    cov = np.asarray(covariance)
    train = np.asarray(training_indices, int)
    n = len(train)
    c = np.asarray(experimental_covariance)[np.ix_(train, train)]
    chol = np.linalg.cholesky(c)
    transform = chol / np.sqrt(np.diag(c))[:, None]
    base = np.asarray(fun(x))
    basecost = float(base @ base)
    names = list(parameter_names)
    maximum = int(opts.get("max_nfev", 50))
    profiles = []
    requested = opts.get("profile_parameters", names)
    if set(requested) - set(names):
        raise ValueError("unknown profile parameter")
    for name in requested:
        k = names.index(name)
        sigma = np.sqrt(max(cov[k, k], 0.0))
        offset = np.asarray(opts.get("profile_sigma_offsets", [-3.0, -1.0, 0.0, 1.0, 3.0]), float)
        if (
            offset.ndim != 1
            or len(offset) < 3
            or not np.all(np.isfinite(offset))
            or not np.any(offset == 0.0)
        ):
            raise ValueError("profile offsets require at least three finite offsets including zero")
        grid = np.unique(np.clip(x[k] + sigma * offset, lo[k], hi[k]))
        cost = []
        complete = True
        for value in grid:
            start = np.clip(
                x + cov[:, k] / max(cov[k, k], np.finfo(float).tiny) * (value - x[k]), lo, hi
            )
            result = _refit(
                fun,
                jac,
                start,
                lo,
                hi,
                scale,
                n,
                fixed=(k, value),
                max_nfev=maximum,
                data_transform=transform,
            )
            complete &= result["success"]
            cost.append(result["chi_square"] - basecost)
        cost = np.maximum(cost, 0.0)

        def bracket(threshold):
            return bool(
                np.any((grid < x[k]) & (cost >= threshold))
                and np.any((grid > x[k]) & (cost >= threshold))
            )

        profiles.append(
            dict(
                parameter=name,
                grid=grid.tolist(),
                delta_chi_square=cost.tolist(),
                complete=complete,
                interval_68=_profile_interval(grid, cost, 1.0, x[k])
                if complete and bracket(1.0)
                else None,
                interval_95=_profile_interval(grid, cost, 3.841458820694124, x[k])
                if complete and bracket(3.841458820694124)
                else None,
                bracketed_68=bracket(1.0),
                bracketed_95=bracket(3.841458820694124),
            )
        )
    samples = int(opts.get("parametric_samples", 0))
    seed = int(opts.get("seed", 1729))
    rng = np.random.default_rng(seed)
    replicas = []
    if samples and samples < 2:
        raise ValueError("at least two parametric replicates required")
    center = np.asarray(predict(x))[train]
    target = np.array([observations[i]["frequency_MHz"] for i in train])
    default_prediction_rows = [
        i for i, o in enumerate(observations) if o.get("role", "training") == "validation"
    ] or list(range(len(observations)))
    prediction_rows = np.asarray(opts.get("prediction_rows", default_prediction_rows), int)
    if np.any(prediction_rows < 0) or np.any(prediction_rows >= len(observations)):
        raise ValueError("invalid nonlinear prediction rows")
    prediction_samples = []
    for i in range(samples):
        synthetic = center + chol @ rng.standard_normal(n)
        offset = solve_triangular(chol, synthetic - target, lower=True)

        def residual(z, offset=offset):
            r = np.array(fun(z), copy=True)
            r[:n] -= offset
            return r

        replicas.append(
            dict(
                sample=i,
                **_refit(
                    residual, jac, x, lo, hi, scale, n, max_nfev=maximum, data_transform=transform
                ),
            )
        )
        if replicas[-1]["success"]:
            prediction_samples.append(
                np.asarray(predict(np.asarray(replicas[-1]["values"])))[prediction_rows]
            )
    successful = [r["values"] for r in replicas if r["success"]]
    intervals = None
    if len(successful) >= 2:
        estimates = np.asarray(successful)
        intervals = dict(
            percentiles_2_5_50_97_5=np.percentile(estimates, [2.5, 50.0, 97.5], axis=0).tolist(),
            covariance=np.atleast_2d(np.cov(estimates, rowvar=False)).tolist(),
        )
    blocks = []
    block_samples = []
    jackknife_covariance = None
    keys = opts.get("block_keys")
    if keys is not None:
        if len(keys) != len(observations):
            raise ValueError("one physical block label per observation required")
        labels = np.array([str(keys[i]) for i in train])
        unique = sorted(set(labels))
        b = len(unique)
        if b < 2:
            raise ValueError("at least two physical blocks required")

        def transformed(keep, multiplicity=None):
            marginal = c[np.ix_(keep, keep)]
            root = np.linalg.cholesky(marginal)
            # Transform from the existing whitened data residual back to its
            # physical marginal and whiten there, preserving cross correlations.
            matrix = solve_triangular(root, chol[keep], lower=True)
            if multiplicity is not None:
                matrix *= np.sqrt(multiplicity)[:, None]

            def r(z):
                raw = np.asarray(fun(z))
                return np.r_[matrix @ raw[:n], raw[n:]]

            def j(z):
                raw = np.asarray(jac(z))
                return np.vstack((matrix @ raw[:n], raw[n:]))

            return r, j, root / np.sqrt(np.diag(marginal))[:, None]

        for name in unique:
            keep = np.flatnonzero(labels != name)
            r, j, marginal_transform = transformed(keep)
            blocks.append(
                dict(
                    omitted_block=name,
                    retained_count=len(keep),
                    **_refit(
                        r,
                        j,
                        x,
                        lo,
                        hi,
                        scale,
                        len(keep),
                        max_nfev=maximum,
                        data_transform=marginal_transform,
                    ),
                )
            )
        if all(row["success"] for row in blocks):
            estimates = np.array([row["values"] for row in blocks])
            delta = estimates - estimates.mean(axis=0)
            jackknife_covariance = ((b - 1) / b * delta.T @ delta).tolist()
        bootstrap = int(opts.get("block_bootstrap_samples", 0))
        if bootstrap < 0:
            raise ValueError("negative block bootstrap budget")
        if bootstrap:
            cross = np.where(labels[:, None] != labels[None, :], c, 0.0)
            if np.max(abs(cross), initial=0.0) > 1.0e-12 * np.max(np.diag(c)):
                raise ValueError(
                    "block bootstrap requires independent physical blocks; use correlated parametric refits and block jackknife"
                )
            for iteration in range(bootstrap):
                draw = rng.integers(0, b, b)
                counts = np.bincount(draw, minlength=b)
                weight = np.array([counts[unique.index(label)] for label in labels])
                keep = np.flatnonzero(weight)
                r, j, marginal_transform = transformed(keep, weight[keep])
                block_samples.append(
                    dict(
                        sample=iteration,
                        drawn_blocks=[unique[k] for k in draw],
                        **_refit(
                            r,
                            j,
                            x,
                            lo,
                            hi,
                            scale,
                            len(keep),
                            max_nfev=maximum,
                            data_transform=marginal_transform,
                        ),
                    )
                )
    complete = all(r["complete"] for r in profiles) and all(
        r["success"] for r in replicas + blocks + block_samples
    )
    return dict(
        schema="matrix.lock.structural_nonlinear_uncertainty.v1",
        complete=complete,
        profiles=profiles,
        parametric_replicates=replicas,
        parametric_summary=intervals,
        nonlinear_prediction_intervals=dict(
            rows=prediction_rows.tolist(),
            percentiles_2_5_50_97_5_MHz=np.percentile(
                prediction_samples, [2.5, 50.0, 97.5], axis=0
            ).tolist()
            if len(prediction_samples) >= 2
            else None,
            scope="uncertainty of physical predictions conditional on the model and fixed QM prior; no new observation noise",
        ),
        block_jackknife=blocks,
        block_jackknife_covariance=jackknife_covariance,
        block_bootstrap=block_samples,
        seed=seed,
        parameter_names=names,
        interpretation="Nonlinear refits conditional on fixed QM priors and model; unbracketed profile intervals are not reported. Jackknife covariance assumes representative approximately balanced blocks. Not a global posterior or a model-accuracy guarantee.",
    )
