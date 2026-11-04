"""Spectroscopy-directed QC energy/gradient acquisition with correlated noise."""

from __future__ import annotations
import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from .prediction_budget import _covariance


def select_qc_candidates(
    parameter_covariance,
    observable_jacobian,
    candidates,
    *,
    batch_size=1,
    observable_sigmas=None,
    noise_covariance=None,
    minimum_gain=0.0,
    parameter_observation_cross_covariance=None,
):
    """Greedy expected reduction of weighted spectroscopic variance per cost.

    Candidate design rows can be energies, gradients, or other QC quantities.
    Full cross-candidate observation noise is retained when supplied. Candidate
    selection produces requests; it never modifies the Hamiltonian or runs QC.
    """
    g = np.asarray(observable_jacobian, float)
    if g.ndim != 2 or not np.all(np.isfinite(g)):
        raise ValueError("invalid spectroscopic sensitivity matrix")
    c = _covariance(parameter_covariance, g.shape[1], "QC parameter")
    records = [dict(r) for r in candidates]
    ids = [str(r["id"]) for r in records]
    if (
        len(set(ids)) != len(ids)
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, (int, np.integer))
        or batch_size < 1
        or not np.isfinite(minimum_gain)
        or minimum_gain < 0
    ):
        raise ValueError("invalid QC acquisition IDs or budget")
    if observable_sigmas is not None:
        sigma = np.asarray(observable_sigmas, float)
        if sigma.shape != (len(g),) or np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
            raise ValueError("invalid target uncertainties")
        g = g / sigma[:, None]
    designs = []
    noises = []
    offsets = [0]
    for r in records:
        h = np.asarray(r["design"], float)
        noise = np.asarray(r["noise_covariance"], float)
        cost = float(r.get("cost", 1.0))
        if (
            h.ndim != 2
            or h.shape[1] != len(c)
            or not len(h)
            or not np.all(np.isfinite(h))
            or not np.isfinite(cost)
            or cost <= 0
        ):
            raise ValueError("invalid QC design or cost")
        r["cost"] = cost
        noise = _covariance(noise, len(h), "QC observation")
        np.linalg.cholesky(noise)
        designs.append(h)
        noises.append(noise)
        offsets.append(offsets[-1] + len(h))
    full_noise = (
        None
        if noise_covariance is None
        else _covariance(noise_covariance, offsets[-1], "joint QC observation")
    )
    if full_noise is not None:
        np.linalg.cholesky(full_noise)

    cross_noise = (np.zeros((len(c),offsets[-1]))
                   if parameter_observation_cross_covariance is None
                   else np.asarray(parameter_observation_cross_covariance,float))
    if cross_noise.shape != (len(c),offsets[-1]) or not np.all(np.isfinite(cross_noise)):
        raise ValueError("invalid parameter/conditional-QC-noise covariance")
    joint_noise = (block_diag(*noises) if noises else np.zeros((0,0))) if full_noise is None else full_noise
    # Validate the *joint* covariance: correlated noise cannot be arbitrarily
    # appended to a posterior parameter covariance.
    joint = np.block([[c,cross_noise],[cross_noise.T,joint_noise]])
    diagonal=np.sqrt(np.maximum(np.diag(joint),np.finfo(float).tiny))
    normalized=joint/diagonal[:,None]/diagonal[None,:]
    if np.min(np.linalg.eigvalsh(normalized),initial=0.) < -1.e-9:
        raise ValueError("conditional parameter/QC-noise covariance is not positive semidefinite")

    def posterior(selected):
        h = np.vstack([designs[k] for k in selected])
        rows = [i for k in selected for i in range(offsets[k], offsets[k + 1])]
        noise = (
            block_diag(*(noises[k] for k in selected))
            if full_noise is None
            else full_noise[np.ix_(rows, rows)]
        )
        correlated = cross_noise[:,rows]
        cross = c @ h.T + correlated
        innovation = h @ c @ h.T + h @ correlated + correlated.T @ h.T + noise
        updated = c - cross @ cho_solve(cho_factor(innovation, lower=True), cross.T)
        return (updated + updated.T) / 2

    score = lambda covariance: float(np.einsum("ij,jk,ik->", g, covariance, g, optimize=True))
    selected = []
    history = []
    current = c
    remaining = set(range(len(records)))
    for _ in range(min(batch_size, len(records))):
        trials = []
        for k in sorted(remaining, key=lambda i: ids[i]):
            updated = posterior([*selected, k])
            gain = max(0.0, score(current) - score(updated))
            trials.append((gain / records[k].get("cost", 1.0), ids[k], k, gain, updated))
        if not trials:
            break
        best = sorted(trials, key=lambda row: (-row[0], row[1]))[0]
        utility, _, k, gain, updated = best
        if gain <= minimum_gain:
            break
        selected.append(k)
        remaining.remove(k)
        history.append(dict(id=ids[k], variance_reduction=gain, utility_per_cost=utility))
        current = updated
    return dict(
        schema="matrix.lock.qc_acquisition.v1",
        selected_ids=[ids[k] for k in selected],
        selection=history,
        selected_requests=[
            {k: v for k, v in records[i].items() if k not in {"design", "noise_covariance"}}
            for i in selected
        ],
        posterior_parameter_covariance=current.tolist(),
        weighted_variance_before=score(c),
        weighted_variance_after=score(current),
        parameter_noise_correlation_retained=bool(np.any(cross_noise)),
        observation_noise="full correlated"
        if full_noise is not None
        else "independent candidate blocks",
        scope="expected linearized reduction under the declared covariance; not an experimental accuracy guarantee",
    )


def periodic_pes_qc_candidates(
    surface,
    request,
    points,
    *,
    reference_coordinate,
    energy_sigma_MHz,
    gradient_sigmas_MHz,
    geometry_builder=None,
    job_template=None,
):
    """Construct energy-difference and LAM-gradient designs from native fields.

    Grid fields are interpolated with their periodic Fourier representation or
    nonperiodic cubic spline. Geometry reconstruction can be supplied by SMITH.
    Electronic calculations remain explicit, reviewable job requests.
    """
    from scipy.interpolate import CubicSpline
    from .lam_fit import _lam_variable, _surface_parameter_direction
    from .general_lam_operator import normalize_general_lam_operator_manifest, _fourier_field

    raw = request.get("operator_manifest")
    manifest = None if raw is None else normalize_general_lam_operator_manifest(raw)
    defaults = {} if manifest is None else manifest.parameters_MHz
    variables = tuple(
        _lam_variable(v, surface, k, defaults) for k, v in enumerate(request["variables"])
    )
    values = np.array([v["initial"] for v in variables])
    directions = []
    for k, var in enumerate(variables):
        if var["path"].startswith("operator."):
            direction = np.zeros(surface.grid_shape)
            for term in manifest.terms:
                if (
                    term.parameter == var["path"][9:]
                    and not term.momentum_axes
                    and not term.rotation_word
                ):
                    direction += (
                        var["multiplier"]
                        * term.prefactor
                        * _fourier_field(surface, term).reshape(surface.grid_shape)
                    )
            directions.append(direction)
        else:
            directions.append(
                _surface_parameter_direction(surface, variables, values, k)["potential_MHz"]
            )
    sigma = np.r_[energy_sigma_MHz, np.asarray(gradient_sigmas_MHz, float)]
    if sigma.shape != (surface.dimension + 1,) or np.any(sigma <= 0):
        raise ValueError("one positive energy and per-coordinate gradient uncertainty is required")
    reference = np.asarray(reference_coordinate, float)
    if reference.shape != (surface.dimension,):
        raise ValueError("invalid QC reference coordinate")

    def evaluate(field, q, derivative=None):
        if field is None:
            return 0.0
        a = np.asarray(field, complex)
        for axis, (grid, periodic) in enumerate(
            zip(surface.coordinates, surface.periodic, strict=True)
        ):
            order = int(derivative == axis)
            if periodic:
                n = len(grid)
                waves = 2 * np.pi * np.fft.fftfreq(n, grid[1] - grid[0])
                coefficient = np.fft.fft(a, axis=0) / n
                phase = np.exp(1j * waves * (q[axis] - grid[0])) * (1j * waves) ** order
                a = np.tensordot(phase, coefficient, axes=(0, 0))
            else:
                if q[axis] < grid[0] or q[axis] > grid[-1]:
                    raise ValueError("QC design leaves a nonperiodic surface domain")
                a = CubicSpline(grid, a, axis=0)(q[axis], nu=order)
        if abs(np.imag(a)) > 1.0e-8 * max(1.0, abs(a)):
            raise ValueError("QC interpolation is not real; check Nyquist resolution")
        return float(np.real(a))

    reference_values = np.array([evaluate(f, reference) for f in directions])
    result = []
    seen = set()
    for index, raw in enumerate(points):
        q = np.array(raw, float, copy=True)
        if q.shape != reference.shape or not np.all(np.isfinite(q)):
            raise ValueError("invalid QC candidate coordinate")
        for axis, (grid, periodic, order) in enumerate(
            zip(surface.coordinates, surface.periodic, surface.symmetry_orders, strict=True)
        ):
            if periodic:
                period = (grid[1] - grid[0]) * len(grid) / order
                q[axis] = grid[0] + (q[axis] - grid[0]) % period
        key = tuple(np.round(q, 12))
        if key in seen:
            continue
        seen.add(key)
        h = np.vstack(
            [
                np.array([evaluate(f, q) for f in directions]) - reference_values,
                *[
                    np.array([evaluate(f, q, axis) for f in directions])
                    for axis in range(surface.dimension)
                ],
            ]
        )
        job = dict(job_template or {})
        job.update(
            engine=job.get("engine", "gdv"),
            coordinate=q.tolist(),
            reference_coordinate=reference.tolist(),
            observables=["relative_energy", "coordinate_gradient"],
            gradient_definition="Cartesian QC gradient projected with the SMITH coordinate tangent",
        )
        if geometry_builder is not None:
            geometry = geometry_builder(q)
            if isinstance(geometry, dict):
                job["geometry_angstrom"] = np.asarray(geometry["geometry_angstrom"], float).tolist()
                if "cartesian_tangents_angstrom" in geometry:
                    job["cartesian_tangents_angstrom"] = np.asarray(
                        geometry["cartesian_tangents_angstrom"], float
                    ).tolist()
            else:
                job["geometry_angstrom"] = np.asarray(geometry, float).tolist()
        result.append(
            dict(
                id=f"qc-{index:05d}",
                design=h.tolist(),
                noise_covariance=np.diag(sigma * sigma).tolist(),
                job=job,
                parameter_paths=[v["path"] for v in variables],
            )
        )
    return result
