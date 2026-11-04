"""Direct structural/PES calibration through Cartesian kinetic surfaces.

The producer contract is deliberately molecularly independent: SMITH supplies
geometries and coordinate tangents; QC/TRINITY supply an effective potential.
Isotopologues supply actual masses, rather than a fitted metric scale factor.
"""

from __future__ import annotations
from collections import OrderedDict
import numpy as np
from scipy.interpolate import CubicSpline
from matrix_numerics.least_squares import CovarianceWhitening, svd_covariance
from .lam_rotation import LAMRotationSurface
from .tunneling_coordinates import ROTATIONAL_FACTOR_MHZ_AMU_ANGSTROM2


def cartesian_lam_surface(
    coordinates,
    geometries_angstrom,
    tangents_angstrom,
    masses_amu,
    potential_MHz,
    *,
    periodic,
    **surface_fields,
):
    """Build the complete translation-free rotation/LAM metric from a chart.

    ``tangents_angstrom`` has grid + (atoms, 3, LAM dimensions). Coordinates
    and tangents must be expressed in the same smooth body frame. The body
    frame may have a nonzero mechanical connection; its cross metric is kept.
    The volume scalar is included after transformation to flat grid measure.
    Rank deficient (linear-rotor or redundant) charts are rejected explicitly.
    """
    grids = tuple(np.asarray(q, float) for q in coordinates)
    shape = tuple(len(q) for q in grids)
    d = len(grids)
    x = np.array(geometries_angstrom, float, copy=True)
    t = np.array(tangents_angstrom, float, copy=True)
    m = np.asarray(masses_amu, float)
    if (
        m.ndim != 1
        or np.any(m <= 0)
        or not np.all(np.isfinite(m))
        or x.shape != shape + (len(m), 3)
        or t.shape != shape + (len(m), 3, d)
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(t))
    ):
        raise ValueError("invalid Cartesian chart, tangent or atomic masses")
    if not 1 <= d <= 4 or any(
        len(q) < 3
        or not np.all(np.isfinite(q))
        or np.any(np.diff(q) <= 0)
        or not np.allclose(np.diff(q), q[1] - q[0], rtol=1.0e-9, atol=1.0e-12)
        for q in grids
    ):
        raise ValueError("one to four finite uniform coordinate grids are required")
    if len(periodic) != d:
        raise ValueError("one boundary condition per coordinate is required")
    x -= np.einsum("a,...ac->...c", m, x)[..., None, :] / m.sum()
    t -= np.einsum("a,...acd->...cd", m, t)[..., None, :, :] / m.sum()
    rotations = np.stack([np.cross(np.eye(3)[a], x) for a in range(3)], axis=-1)
    columns = np.concatenate((t, rotations), axis=-1)
    metric = np.einsum("...aci,a,...acj->...ij", columns, m, columns, optimize=True)
    eigen = np.linalg.eigvalsh(metric)
    if np.any(eigen[..., 0] <= 1.0e-12 * eigen[..., -1]):
        raise ValueError("rank deficient Cartesian rotation/LAM chart")
    inverse = np.linalg.inv(metric)
    factor = ROTATIONAL_FACTOR_MHZ_AMU_ANGSTROM2
    f = factor * inverse[..., :d, :d]
    rho = -inverse[..., d:, :d] @ np.linalg.inv(inverse[..., :d, :d])
    r = factor * inverse[..., d:, d:] - rho @ f @ np.swapaxes(rho, -1, -2)
    logw = 0.5 * np.linalg.slogdet(metric)[1]

    def derivative(a, axis):
        grid = grids[axis]
        if periodic[axis]:
            waves = 2 * np.pi * np.fft.fftfreq(len(grid), grid[1] - grid[0])
            view = [1] * a.ndim
            view[axis] = len(grid)
            return np.fft.ifft(
                np.fft.fft(a, axis=axis) * (1j * waves).reshape(view), axis=axis
            ).real
        return CubicSpline(grid, a, axis=axis).derivative()(grid)

    gradient = np.stack([derivative(logw, i) for i in range(d)], axis=-1)
    flux = np.einsum("...ij,...j->...i", f, gradient)
    scalar = 0.5 * sum(derivative(flux[..., i], i) for i in range(d)) + 0.25 * np.einsum(
        "...i,...i->...", gradient, flux
    )
    if "geometric_potential_MHz" in surface_fields:
        raise ValueError("Cartesian volume scalar is constructed, not supplied")
    return LAMRotationSurface(
        coordinates=grids,
        potential_MHz=np.asarray(potential_MHz, float),
        F_MHz=f,
        rho=rho,
        R_MHz=r,
        geometric_potential_MHz=scalar,
        periodic=tuple(periodic),
        **surface_fields,
    )


def smith_lam_surface(
    chart,
    coordinates,
    gic_displacements,
    gic_tangents,
    potential_MHz,
    *,
    periodic,
    masses_amu=None,
    **surface_fields,
):
    """Realize a SMITH GIC/Eckart grid and its implicit Cartesian tangents.

    Displacements have grid + (GIC rank,); tangents have grid + (rank, d).
    They may contain structural offsets and curvilinear LAM paths. The chart
    must cover the grid without coordinate singularities; chart failures are
    propagated. Isotopic masses are supplied explicitly when different from
    the chart reference. The body frame remains the chart's Eckart frame.
    """
    from matrix_smith.evaluation import build_gic_b_matrix

    shape = tuple(len(q) for q in coordinates)
    d = len(coordinates)
    rank = chart.definition.rank
    shifts = np.asarray(gic_displacements, float)
    tangents = np.asarray(gic_tangents, float)
    if shifts.shape != shape + (rank,) or tangents.shape != shape + (rank, d):
        raise ValueError("SMITH displacement/tangent grid shape mismatch")
    geometries = []
    cartesian_tangents = []
    for at in np.ndindex(shape):
        xyz = chart.geometry_at_displacement_angstrom(shifts[at])
        b = np.asarray(build_gic_b_matrix(chart.definition, coordinates_angstrom=xyz).rows)
        augmented = np.vstack((b, chart.rigid_constraint_rows))
        tangent = np.linalg.solve(augmented, np.vstack((tangents[at], np.zeros((6, d)))))
        geometries.append(xyz)
        cartesian_tangents.append(tangent.reshape(len(xyz), 3, d))
    return cartesian_lam_surface(
        coordinates,
        np.array(geometries).reshape(shape + xyz.shape),
        np.array(cartesian_tangents).reshape(shape + (len(xyz), 3, d)),
        chart.masses_amu if masses_amu is None else masses_amu,
        potential_MHz,
        periodic=periodic,
        **surface_fields,
    )


def _fit_structure_pes_spectra(builder=None, request=None, *, execution=None):
    """Joint fit of actual structure and PES variables through native spectra.

    ``builder(dict(name=value))`` returns one LAMRotationSurface per dataset.
    It must reconstruct the geometry/metric/PES consistently at every trial;
    Cartesian charts can use :func:`cartesian_lam_surface`. State assignments
    are tracked from the initial reference. Producers with derivatives() use
    analytic moving-subspace response; other callables retain bounded complete-
    model differences. Both paths receive an independent final check. All
    covariance blocks are absolute and validation rows remain outside the fit.
    """
    from scipy.optimize._numdiff import approx_derivative
    from .lam_global_fit import _backend, _measure
    from .lam_fit import _lam_level_energy, track_lam_rotation_states
    from .fit_convergence import statistical_trust_region, assess_statistical_convergence
    from .prediction_budget import _covariance

    if request is None and isinstance(builder, dict):
        request, builder = builder, None
    if builder is None:
        from .physical_producer import build_physical_lam_producer

        builder = build_physical_lam_producer(request["producer"])
    if request.get("schema") != "matrix.lock.structure_pes_fit.v1":
        raise ValueError("unsupported structural fit schema")
    if request.get("options", {}).get("numerical_accuracy") is not None:
        from .lam_accuracy import fit_structure_pes_with_accuracy

        return fit_structure_pes_with_accuracy(builder, request)
    variables = request["variables"]
    datasets = request["datasets"]
    names = [str(d["system"]) for d in datasets]
    if len(set(names)) != len(names) or not names or not variables:
        raise ValueError("unique systems and variables are required")
    if len({v["name"] for v in variables}) != len(variables) or any(
        v.get("kind") not in {"structure", "pes"} for v in variables
    ):
        raise ValueError("variables require unique names and structure/pes kind")
    x = np.array([v["initial"] for v in variables], float)
    saved_optimizer = request.get("options", {}).get("optimizer_state")
    if saved_optimizer is not None:
        if saved_optimizer.get("parameter_names") != [v["name"] for v in variables]:
            raise ValueError("structural restart parameter registry differs")
        x = np.asarray(saved_optimizer["values"], float)
        if x.shape != (len(variables),):
            raise ValueError("structural restart parameter count differs")
    scale = np.array([v["scale"] for v in variables], float)
    lo = np.array([v.get("lower", -np.inf) for v in variables])
    hi = np.array([v.get("upper", np.inf) for v in variables])
    if (
        np.any(scale <= 0)
        or np.any(lo >= hi)
        or np.any(x < lo)
        or np.any(x > hi)
        or not np.all(np.isfinite(np.r_[x, scale]))
    ):
        raise ValueError("invalid structural fit values, scales or bounds")
    observations = [o for d in datasets for o in d["request"]["observations"]]
    y = np.array([o["frequency_MHz"] for o in observations])
    roles = [o.get("role", "training") for o in observations]
    if any(role not in {"training", "validation"} for role in roles):
        raise ValueError("invalid observation role")
    train = np.array([i for i, r in enumerate(roles) if r == "training"], int)
    if not len(train):
        raise ValueError("no training measurements")
    sigma = np.array([o["uncertainty_MHz"] for o in observations])
    c = _covariance(request.get("covariance_MHz2", np.diag(sigma * sigma)), len(y), "spectral")
    weights = np.sqrt(np.asarray([o.get("weight", 1.0) for o in observations], float))
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("finite positive observation weights are required")
    sigma = sigma / weights
    c = c / weights[:, None] / weights[None, :]
    whitening = CovarianceWhitening(sigma[train], c[np.ix_(train, train)])
    priors = [(k, v["prior"]) for k, v in enumerate(variables) if "prior" in v]
    pw = (
        None
        if not priors
        else CovarianceWhitening([p["sigma"] for _, p in priors], request.get("prior_covariance"))
    )
    predicates = request.get("predicates", [])
    index = {v["name"]: k for k, v in enumerate(variables)}
    a = np.array(
        [[p.get("coefficients", {}).get(v["name"], 0.0) for v in variables] for p in predicates]
    )
    if any(set(p.get("coefficients", {})) - set(index) for p in predicates):
        raise ValueError("unknown QM predicate variable")
    qw = (
        None
        if not predicates
        else CovarianceWhitening(
            [p["uncertainty"] for p in predicates], request.get("predicate_covariance")
        )
    )
    reference = {}
    cache = OrderedDict()
    evaluations = 0
    solutions = {}
    jacobian_cache = {}
    spectral_sensitivities = {}

    def predict(values, *, _require_workspace=False):
        nonlocal evaluations
        key = np.asarray(values, float).tobytes()
        if key in cache and (not _require_workspace or key in solutions):
            return cache[key]
        systems = builder(dict(zip(index, map(float, values), strict=True)))
        if set(systems) != set(names):
            raise ValueError("builder must provide exactly the requested systems")
        def evaluate_dataset(name):
            dataset = next(d for d in datasets if str(d["system"]) == name)
            local = dataset["request"]
            if any(v.get("prior") for v in local.get("variables", [])) or local.get("predicates"):
                raise ValueError("declare QC information once on the outer structural variables")
            context = _backend(systems[name], local)
            _, surface, solved, _ = context["predict"](context["x0"])
            if name not in reference:
                reference[name] = (solved, context["observations"])
            ref, obs = reference[name]
            from .lam_fit import _lam_level_index

            selected = {J: set() for J in ref.blocks}
            for item in obs:
                for side in ("lower", "upper"):
                    selector = item[side]
                    selected[selector.J].add(_lam_level_index(selector, ref, {}))
            mapping, tracking = track_lam_rotation_states(
                ref, solved, diagnostic_indices=selected,
                subspace_gap_MHz=local.get("options", {}).get("tracking_subspace_gap_MHz", 1.e-6),
            )
            if tracking["minimum_principal_overlap"] < float(
                local.get("options", {}).get("tracking_minimum_principal_overlap", 0.7)
            ):
                raise RuntimeError("structural change loses the assigned rovibrational subspace")
            frequency = np.array(
                [
                    _lam_level_energy(o["upper"], solved, mapping)
                    - _lam_level_energy(o["lower"], solved, mapping)
                    for o in obs
                ]
            )
            shadow = dict(
                context,
                observations=obs,
                predict=lambda z, payload=(frequency, surface, solved, mapping): payload,
            )
            measured, _, _ = _measure(shadow, context["x0"], derivatives=False)
            return measured, (shadow, surface, solved, mapping, obs)

        if execution is None:
            evaluated = [evaluate_dataset(name) for name in names]
        else:
            from .resource_execution import LOCKTask
            tasks = []
            for dataset in datasets:
                name = str(dataset["system"])
                local = dataset["request"]
                nlam = int(local.get("lam_states", 12))
                primitive = int(np.prod(systems[name].grid_shape))
                js = {int(c[side]["J"]) for o in local["observations"]
                      for c in o.get("components", [o]) for side in ("lower","upper")}
                dimensions = [nlam*(2*j+1) for j in js]
                resident = 16*(primitive*nlam + sum(d*d for d in dimensions))
                workspace = 16*(8*primitive*nlam + 6*max(dimensions)**2)
                tasks.append(LOCKTask(name, workspace, resident, sum(d**3 for d in dimensions)))
            evaluated = execution.map(evaluate_dataset, tasks)
        result = np.concatenate([np.asarray(item[0]) for item in evaluated])
        details = {name:item[1] for name,item in zip(names,evaluated,strict=True)}
        evaluations += 1
        cache[key] = result
        solutions.clear()
        solutions[key] = details
        while len(cache) > max(16, 4 * len(values) + 1):
            removed, _ = cache.popitem(last=False)
            solutions.pop(removed, None)
        return result

    def fun(values):
        residual = list(whitening.apply((predict(values) - y)[train]))
        if pw is not None:
            residual.extend(
                pw.apply(values[[k for k, _ in priors]] - np.array([p["value"] for _, p in priors]))
            )
        if qw is not None:
            residual.extend(qw.apply(a @ values - np.array([p["value"] for p in predicates])))
        return np.array(residual)

    options = request.get("options", {})
    step = float(options.get("derivative_step", 1.0e-4))
    if not np.isfinite(step) or step <= 0:
        raise ValueError("positive derivative step required")

    available = callable(getattr(builder, "derivatives", None)) and getattr(
        builder, "analytic_derivatives_available", True
    )
    if options.get("jacobian", "auto") not in {"auto", "analytic", "numerical"}:
        raise ValueError("unknown structural Jacobian mode")
    if options.get("jacobian") == "analytic" and not available:
        raise ValueError("the physical producer does not provide the required analytic derivatives")
    analytic = available and options.get("jacobian", "auto") != "numerical"

    def jac(values):
        key = np.asarray(values, float).tobytes()
        if key in jacobian_cache:
            return jacobian_cache[key]
        if not analytic:
            return approx_derivative(
                fun, values, method="3-point", abs_step=step * scale, bounds=(lo, hi)
            )
        from .lam_surface_response import native_surface_response, complete_surface_direction
        from .lam_response import apply_factors, moving_ritz_derivatives
        from .lam_fit import (
            _lam_level_index,
            _projected_coupled_factors,
            _offset_surface,
            _apply_lam_variables,
        )

        predict(values, _require_workspace=True)
        details = solutions[np.asarray(values, float).tobytes()]
        physical = builder.derivatives(dict(zip(index, map(float, values), strict=True)))
        rows = []
        for dataset in datasets:
            name = str(dataset["system"])
            context, surface, solved, mapping, obs = details[name]
            native_variables = []
            native_rows = []
            response_workspace = {}
            for variable in variables:
                direction = physical[name][variable["name"]]
                # Fixed native affine field scales must also act on a physical tangent.
                direction = complete_surface_direction(surface, direction) | {
                    "dipole_au": direction.get("dipole_au")
                }
                if context["variables"]:
                    original = builder(dict(zip(index, map(float, values), strict=True)))[name]
                    shifted = _apply_lam_variables(
                        _offset_surface(
                            original, complete_surface_direction(original, direction), 1.0
                        ),
                        context["variables"],
                        context["x0"],
                    )
                    direction = {
                        field: None
                        if getattr(shifted, field) is None
                        else np.asarray(getattr(shifted, field))
                        - np.asarray(getattr(surface, field))
                        for field in direction
                        if field not in {"dipole_au", "spin_tensor_surfaces_MHz"}
                    } | {"dipole_au": direction.get("dipole_au")}
                factors, du = native_surface_response(
                    surface,
                    solved,
                    direction,
                    response_states=int(context["options"].get("basis_response_states", 48)),
                    manifest=context["operator_manifest"],
                    parameters=solved.operator_parameters_MHz,
                    operator_direction=physical[name][variable["name"]].get(
                        "operator_parameters_MHz"
                    ),
                    workspace=response_workspace,
                )
                energies = {}
                for J, block in solved.blocks.items():
                    required = np.array(
                        sorted(
                            {
                                _lam_level_index(o[side], solved, mapping)
                                for o in obs
                                for side in ("lower", "upper")
                                if o[side].J == J
                            }
                        ),
                        int,
                    )
                    vectors = block.vectors[:, required]
                    energies[J] = np.full(len(block.energies_MHz), np.nan)
                    if solved.k_contractions is not None:
                        base = _projected_coupled_factors(
                            surface,
                            solved.basis.vectors,
                            J,
                            solved.basis,
                            operator_manifest=context["operator_manifest"],
                            operator_parameters_MHz=solved.operator_parameters_MHz,
                            representation=solved.representation,
                        )
                        energies[J][required] = moving_ritz_derivatives(
                            solved.k_contractions[J],
                            base,
                            factors[J],
                            block.energies_MHz[required],
                            vectors,
                        )
                    else:
                        action = apply_factors(
                            factors[J], vectors, solved.basis.vectors.shape[1], 2 * J + 1
                        )
                        energies[J][required] = np.real(np.sum(vectors.conj() * action, axis=0))
                native_rows.append(
                    [
                        energies[o["upper"].J][_lam_level_index(o["upper"], solved, mapping)]
                        - energies[o["lower"].J][_lam_level_index(o["lower"], solved, mapping)]
                        for o in obs
                    ]
                )
                native_variables.append(
                    dict(
                        path="surface." + variable["name"],
                        _native_surface_response=(factors, du),
                        dipole_direction=direction.get("dipole_au"),
                    )
                )
            component_jac = np.asarray(native_rows).T
            shadow = dict(
                context,
                variables=native_variables,
                moving_k_contraction=True,
                jac=lambda z: component_jac,
            )
            _, measured_jac, _ = _measure(shadow, values, derivatives=True)
            rows.extend(measured_jac)
        spectral_sensitivities.clear()
        spectral_sensitivities[key] = np.asarray(rows)
        result = list(whitening.apply(np.asarray(rows)[train]))
        if pw is not None:
            result.extend(pw.apply(np.eye(len(values))[[k for k, _ in priors]]))
        if qw is not None:
            result.extend(qw.apply(a))
        result = np.asarray(result)
        jacobian_cache.clear()
        jacobian_cache[key] = result
        return result

    from .parameter_selection import select_spectral_parameters
    selection_options = dict(options.get("parameter_selection", {}))
    selection_mode = selection_options.pop("mode", "report")
    selection_source = selection_options.pop("source", "augmented")
    verify_selection_nonlinear = bool(selection_options.pop("verify_nonlinear", False))
    if selection_mode not in {"report", "refit"} or selection_source not in {"experimental", "augmented"}:
        raise ValueError("parameter selection requires mode report/refit and source experimental/augmented")
    initial_jacobian = jac(x)
    initial_selection = select_spectral_parameters(
        initial_jacobian[:len(train)], list(index), scale,
        augmented_jacobian=initial_jacobian, **selection_options,
    )
    free = (np.asarray(initial_selection[selection_source]["selected_indices"], int)
            if selection_mode == "refit" else np.arange(len(x)))
    if selection_mode == "refit" and saved_optimizer is not None and saved_optimizer.get("selected_parameter_names"):
        saved_names=saved_optimizer["selected_parameter_names"]
        if len(set(saved_names))!=len(saved_names) or set(saved_names)-set(index):
            raise ValueError("saved conditional parameter registry differs")
        free=np.asarray([index[n] for n in saved_names],int)
    if not len(free):
        raise ValueError("no supported parameter direction; supply informative data or justified QM predicates")
    fixed_values = x.copy()

    def expand(z):
        full = fixed_values.copy()
        full[free] = z
        return full

    class Stationary(Exception):
        pass

    state = {}
    accepted_reference_key = None

    def save(s):
        nonlocal accepted_reference_key
        state.update(s)
        state["values"] = expand(np.asarray(s["values"])).tolist()
        state["scales"] = scale.tolist()
        state["parameter_names"] = list(index)
        state["selected_parameter_names"] = [variables[k]["name"] for k in free]
        accepted_key=np.asarray(state["values"],float).tobytes()
        if accepted_key != accepted_reference_key and accepted_key in solutions:
            from dataclasses import replace
            from .lam_fit import _lam_level_index
            details=solutions[accepted_key]
            updated={}
            for name,(context,surface,solved,mapping,observed) in details.items():
                remapped=tuple({**row,**{side:replace(row[side],reference_index=int(
                    _lam_level_index(row[side],solved,mapping))) for side in ("lower","upper")}}
                    for row in observed)
                reference[name]=(solved,remapped)
                component_frequency=context["predict"](context["x0"])[0]
                shadow=dict(context,observations=remapped,
                    predict=lambda z,payload=(component_frequency,surface,solved,{}):payload)
                updated[name]=(shadow,surface,solved,{},remapped)
            # Rejected steps never promote their states. Preserve the accepted
            # eigensystems and frequencies while invalidating old label caches.
            accepted_frequency=cache[accepted_key]
            cache.clear();cache[accepted_key]=accepted_frequency
            solutions.clear();solutions[accepted_key]=updated
            jacobian_cache.clear();spectral_sensitivities.clear()
            accepted_reference_key=accepted_key

    def checked(z):
        values = expand(z)
        j = jac(values)[:, free]
        assessment = assess_statistical_convergence(
            fun(values),
            j,
            z,
            lo[free],
            hi[free],
            data_count=len(train),
            data_correction_transform=whitening.standardized_transform,
        )
        if assessment["converged"]:
            raise Stationary
        return j

    predict(x)
    try:
        optimized = statistical_trust_region(
            lambda z: fun(expand(z)),
            checked,
            x[free],
            lo[free],
            hi[free],
            scale[free],
            max_nfev=int(options.get("max_nfev", 100)),
            initial_radius=float(saved_optimizer.get("trust_radius", 1.0))
            if saved_optimizer is not None and np.array_equal(saved_optimizer.get("scales"), scale)
            else 1.0,
            initial_damping=float(saved_optimizer.get("damping", 0.0))
            if saved_optimizer is not None and np.array_equal(saved_optimizer.get("scales"), scale)
            else 0.0,
            state_callback=save,
        )
        x = expand(optimized.x)
        message = optimized.message
    except Stationary:
        x = np.asarray(state["values"])
        message = "Statistical convergence reached."
    j = jac(x)
    assessment = assess_statistical_convergence(
        fun(x),
        j[:, free],
        x[free],
        lo[free],
        hi[free],
        data_count=len(train),
        data_correction_transform=whitening.standardized_transform,
    )
    predict(x, _require_workspace=True)
    accepted_contexts = solutions[np.asarray(x, float).tobytes()]
    # Independent finer-step check also works at active bounds (one-sided).
    finer = approx_derivative(
        fun, x, method="3-point", abs_step=0.5 * step * scale, bounds=(lo, hi)
    )
    tolerance = float(options.get("derivative_relative_tolerance", 1.0e-3))
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("positive derivative verification tolerance required")
    column_errors = np.linalg.norm((j - finer) * scale, axis=0) / np.maximum(
        np.linalg.norm(finer * scale, axis=0), 1.0e-12
    )
    reference_steps = np.full(len(x), 0.5)
    # A very small PES sensitivity on a large rovibrational energy offset can
    # be unresolved by the first numerical reference. Enlarge only failed
    # columns, requiring agreement of successive independent differences as
    # well as agreement with the analytic response. No fit parameter changes.
    if analytic:
        for column in np.flatnonzero(column_errors > tolerance):
            previous = finer[:, column]

            def one_coordinate(z):
                point = x.copy()
                point[column] = z[0]
                return fun(point)

            for attempt in range(int(options.get("maximum_derivative_check_expansions", 8))):
                multiplier = 2.0**attempt
                current = approx_derivative(
                    one_coordinate,
                    x[column : column + 1],
                    method="3-point",
                    abs_step=multiplier * step * scale[column],
                    bounds=(lo[column : column + 1], hi[column : column + 1]),
                )[:, 0]
                denominator = max(np.linalg.norm(current) * scale[column], 1.0e-12)
                error = np.linalg.norm((j[:, column] - current) * scale[column]) / denominator
                stable = (
                    np.linalg.norm((current - previous) * scale[column]) / denominator <= tolerance
                )
                if stable and error <= tolerance:
                    finer[:, column] = current
                    column_errors[column] = error
                    reference_steps[column] = multiplier
                    break
                previous = current
    relative = float(np.max(column_errors, initial=0.0))
    qualified = bool(relative <= tolerance)
    cov, rank, singular = svd_covariance(j, scales=scale)
    _, experimental_rank, _ = svd_covariance(j[: len(train)], scales=scale)
    final_selection = select_spectral_parameters(
        j[:len(train)], list(index), scale, augmented_jacobian=j, **selection_options
    )
    conditional_covariance, conditional_rank, _ = svd_covariance(j[:,free], scales=scale[free])
    unrestricted_covariance=cov.copy()
    if len(free)!=len(x):
        cov=np.zeros_like(cov)
        cov[np.ix_(free,free)]=conditional_covariance
    pred = predict(x)
    result = dict(
        covariance_kind="full local" if len(free)==len(x) else "conditional on declared fixed complementary parameters",
        unrestricted_covariance=unrestricted_covariance.tolist(),
        parameter_selection=dict(mode=selection_mode, source=selection_source,
            initial=initial_selection, final=final_selection,
            refined_parameters=[variables[k]["name"] for k in free],
            fixed_values={variables[k]["name"]:float(x[k]) for k in range(len(x)) if k not in free},
            conditional_covariance=conditional_covariance.tolist(), conditional_rank=conditional_rank),
        schema="matrix.lock.structure_pes_fit_result.v1",
        success=bool(assessment["converged"] and qualified),
        message=message,
        variables=[
            dict(name=v["name"], kind=v["kind"], value=float(z))
            for v, z in zip(variables, x, strict=True)
        ],
        covariance=cov.tolist(),
        rank=rank,
        experimental_rank=experimental_rank,
        augmented_rank=rank,
        singular_values=singular.tolist(),
        convergence=assessment,
        comparisons=[
            dict(
                role=role, calculated_MHz=float(z), calculated_minus_observed_MHz=float(z - target)
            )
            for role, z, target in zip(roles, pred, y, strict=True)
        ],
        derivative_qualification=dict(
            method=(
                "analytic Cartesian, kinetic and moving-subspace product rule; independent bounded-difference check"
                if analytic
                else "bounded three-point complete-model differences; half-step comparison"
            ),
            qualified=qualified,
            maximum_relative_change=float(relative),
            reference_step_multipliers=reference_steps.tolist(),
        ),
        producer_evaluations=evaluations,
        optimizer_state=state,
        scope="joint local structure/PES calibration; mass-specific kinetic surfaces rebuilt from the supplied chart",
    )

    if selection_mode == "refit" and verify_selection_nonlinear:
        from .structure_uncertainty import _refit
        full = _refit(fun,jac,x,lo,hi,scale,len(train),
            max_nfev=int(options.get("max_nfev",100)),data_transform=whitening.standardized_transform)
        full_values=np.asarray(full["values"])
        full_prediction=predict(full_values)
        selected_residual=fun(x)
        full_residual=fun(full_values)
        validation=np.array([i for i,r in enumerate(roles) if r=="validation"],int)
        validation_comparison={}
        if len(validation):
            w=CovarianceWhitening(sigma[validation],c[np.ix_(validation,validation)])
            selected_validation=w.apply((pred-y)[validation])
            full_validation=w.apply((full_prediction-y)[validation])
            validation_comparison=dict(selected_chi_square=float(selected_validation@selected_validation),
                full_chi_square=float(full_validation@full_validation),count=len(validation))
        result["parameter_selection"]["nonlinear_comparison"]=dict(full_refit=full,
            selected_augmented_chi_square=float(selected_residual@selected_residual),
            full_augmented_chi_square=float(full_residual@full_residual),
            selected_experimental_chi_square=float(selected_residual[:len(train)]@selected_residual[:len(train)]),
            full_experimental_chi_square=float(full_residual[:len(train)]@full_residual[:len(train)]),
            validation=validation_comparison,
            scope="nested physical-model refit; validation rows remain outside optimization")
        result["producer_evaluations"]=evaluations

    if options.get("return_spectral_sensitivities",False):
        accepted_key=np.asarray(x,float).tobytes()
        raw=spectral_sensitivities.get(accepted_key)
        if raw is None:
            raw=approx_derivative(predict,x,method="3-point",abs_step=step*scale,bounds=(lo,hi))
        result["spectral_sensitivities"]=dict(parameter_names=list(index),
            values=x.tolist(),jacobian_MHz_per_parameter=np.asarray(raw).tolist(),
            observation_roles=roles,training_indices=train.tolist())

    from copy import deepcopy
    from .lam_fit import _lam_level_index

    restart = deepcopy(request)
    for variable, value in zip(restart["variables"], x, strict=True):
        variable["initial"] = float(value)
    for dataset in restart["datasets"]:
        _, _, solved, mapping, components = accepted_contexts[str(dataset["system"])]
        iterator = iter(components)
        for observation in dataset["request"]["observations"]:
            for component in observation.get("components", [observation]):
                source = next(iterator)
                for side in ("lower", "upper"):
                    selector = source[side]
                    component[side] = dict(
                        J=selector.J, reference_index=_lam_level_index(selector, solved, mapping)
                    )
    restart.setdefault("options", {})["optimizer_state"] = deepcopy(state)
    if hasattr(builder, "to_specification"):
        restart["producer"] = builder.to_specification()
    result["accepted_requests"] = {str(d["system"]): d["request"] for d in restart["datasets"]}
    result["restart_request"] = restart
    if options.get("nonlinear_uncertainty") is not None:
        if len(free) != len(x):
            raise ValueError("conditional parameter-selection refits require an explicitly reduced uncertainty request")
        if not result["success"]:
            raise ValueError("nonlinear uncertainty requires a converged qualified structural fit")
        from .structure_uncertainty import nonlinear_structure_uncertainty

        result["nonlinear_uncertainty"] = nonlinear_structure_uncertainty(
            fun,
            jac,
            predict,
            x,
            lo,
            hi,
            scale,
            covariance=cov,
            experimental_covariance=c,
            observations=observations,
            parameter_names=list(index),
            training_indices=train,
            options=options["nonlinear_uncertainty"],
        )
    return result


def fit_structure_pes_spectra(builder=None, request=None):
    """Fit native structural spectra, optionally under one execution budget."""
    from copy import deepcopy
    if request is None and isinstance(builder, dict):
        request, builder = builder, None
    if request is None:
        raise ValueError("a structural fit request is required")
    policy = request.get("options", {}).get("execution_budget")
    if request.get("options", {}).get("numerical_accuracy") is not None:
        return _fit_structure_pes_spectra(builder, request)
    if policy is None:
        return _fit_structure_pes_spectra(builder, request)
    from .resource_execution import LOCKResourceExecutor
    local = deepcopy(request)
    # Per-system native operations remain serial under the outer CPU budget.
    for dataset in local["datasets"]:
        dataset["request"].setdefault("options", {}).update(workers=1,block_workers=1,jacobian_workers=1)
    with LOCKResourceExecutor(**policy) as execution:
        result = _fit_structure_pes_spectra(builder, local, execution=execution)
        result["execution_budget"] = execution.report()
    return result
