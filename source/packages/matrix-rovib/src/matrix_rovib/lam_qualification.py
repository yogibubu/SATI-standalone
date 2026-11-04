"""Automatic numerical qualification of native LAM observables and responses."""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, fields, replace
import numpy as np
from scipy.signal import resample
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class LAMBasisQualification:
    surface: object
    request: dict
    report: dict


def refine_lam_grid(surface, sizes):
    """Refine the represented PES/tensors; this adds no new QC information."""
    if len(sizes) != surface.dimension or any(
        n < o for n, o in zip(sizes, surface.grid_shape, strict=True)
    ):
        raise ValueError("one nondecreasing size is required per LAM grid")
    grids = tuple(
        q[0] + np.arange(n) * (q[1] - q[0]) * len(q) / n
        if periodic
        else np.linspace(q[0], q[-1], n)
        for q, n, periodic in zip(surface.coordinates, sizes, surface.periodic, strict=True)
    )

    def interpolate(value):
        a = np.asarray(value)
        if a.shape[: surface.dimension] != surface.grid_shape:
            return a.copy()
        for axis, (q, new, periodic) in enumerate(
            zip(surface.coordinates, grids, surface.periodic, strict=True)
        ):
            a = resample(a, len(new), axis=axis) if periodic else CubicSpline(q, a, axis=axis)(new)
        return a.real if not np.iscomplexobj(value) else a

    updates = {
        f.name: interpolate(getattr(surface, f.name))
        for f in fields(surface)
        if isinstance(getattr(surface, f.name), np.ndarray)
    }
    updates["coordinates"] = grids
    if surface.spin_tensor_surfaces_MHz:
        updates["spin_tensor_surfaces_MHz"] = {
            k: interpolate(v) for k, v in surface.spin_tensor_surfaces_MHz.items()
        }
    return replace(surface, **updates)


def _transport_reference(reference, old, new):
    if old.grid_shape == new.grid_shape:
        return reference
    vectors = reference.basis.vectors.reshape(old.grid_shape + (-1,))
    for axis, (before, after, periodic) in enumerate(
        zip(old.coordinates, new.coordinates, old.periodic, strict=True)
    ):
        if periodic:
            vectors = resample(vectors, len(after), axis=axis) * np.sqrt(len(before) / len(after))
        else:
            vectors = CubicSpline(before, vectors, axis=axis)(after) * np.sqrt(
                (after[1] - after[0]) / (before[1] - before[0])
            )
    vectors = vectors.reshape(int(np.prod(new.grid_shape)), -1)
    gram = vectors.conj().T @ vectors
    values, rotation = np.linalg.eigh(gram)
    if np.min(values) < 1.0e-8:
        raise RuntimeError("grid transport loses the reference subspace")
    vectors = vectors @ ((rotation / np.sqrt(values)) @ rotation.conj().T)
    return replace(reference, basis=replace(reference.basis, vectors=vectors))


def _evaluate(surface, request, *, reference=None):
    from .lam_global_fit import _backend, _measure
    from .lam_fit import (
        track_lam_rotation_states,
        _lam_operator_response_jacobian,
        _lam_level_index,
    )
    from .lam_measurements import native_strengths

    context = _backend(surface, request)
    values = context["x0"]
    _, fitted, solved, mapping = context["predict"](values)
    if reference is not None:
        before, old_surface, old_context = reference
        transported = _transport_reference(before, old_surface, surface)
        selected = {J: set() for J in before.blocks}
        for item in old_context["observations"]:
            for side in ("lower", "upper"):
                selector = item[side]
                selected[selector.J].add(_lam_level_index(selector, before, old_context["mapping"]))
        mapping, diagnostic = track_lam_rotation_states(
            transported, solved, diagnostic_indices=selected
        )
        if diagnostic["minimum_principal_overlap"] < 0.7:
            raise RuntimeError("basis refinement loses assigned state identity")
        observations = []
        for item in old_context["observations"]:
            item = dict(item)
            for side in ("lower", "upper"):
                selector = item[side]
                old_index = _lam_level_index(selector, before, old_context["mapping"])
                item[side] = replace(selector, reference_index=mapping[selector.J][old_index])
            observations.append(item)
        mapping = {J: {k: k for k in range(len(b.levels))} for J, b in solved.blocks.items()}
        context["observations"] = tuple(observations)
        frequencies = np.array(
            [
                solved.blocks[o["upper"].J].energies_MHz[o["upper"].reference_index]
                - solved.blocks[o["lower"].J].energies_MHz[o["lower"].reference_index]
                for o in observations
            ]
        )
        context["predict"] = lambda x: (frequencies, fitted, solved, mapping)
        context["jac"] = lambda x: _lam_operator_response_jacobian(
            fitted,
            solved,
            context["variables"],
            values,
            context["observations"],
            mapping,
            operator_manifest=context["operator_manifest"],
            operator_parameters_MHz=solved.operator_parameters_MHz,
            workers=1,
            moving_k_contraction=context["moving_k_contraction"],
            response_states=int(context["options"].get("basis_response_states", 48)),
            parameter_reference_surface=surface,
        )
    context["mapping"] = mapping
    frequency, jac, _ = _measure(context, values, derivatives=bool(len(values)))
    if jac is None:
        jac = np.empty((len(frequency), 0))
    has_dipole = surface.dipole_au is not None or request.get("dipole_au") is not None
    intensity, dintensity = (
        native_strengths(context, values, np.arange(context["count"]), derivatives=bool(len(values)))
        if has_dipole
        else (None, None)
    )
    return dict(
        frequency=frequency,
        jacobian=jac,
        intensity=intensity,
        intensity_jacobian=dintensity,
        solved=solved,
        context=context,
        scales=context["scales"],
    )


def _probe(surface, request, axis, policy, *, lam_state_count=None):
    candidate = deepcopy(request)
    options = candidate.setdefault("options", {})
    new_surface = surface
    maximum = int(np.prod(surface.grid_shape))
    grow = float(policy.get("growth_factor", 1.5))

    def increase(value, limit):
        return min(limit, max(value + 1, int(np.ceil(value * grow))))

    if axis == "lam_states":
        count = max(int(candidate.get("lam_states", 12)), lam_state_count or 0)
        cuts = options.get("fourier_cutoffs")
        if cuts is not None:
            maximum = min(
                maximum,
                int(
                    np.prod(
                        [
                            len(q) if c is None else 2 * c + 1
                            for q, c in zip(surface.coordinates, cuts, strict=True)
                        ]
                    )
                ),
            )
        maximum = min(maximum, int(policy.get("maximum_lam_states", maximum)))
        if count >= maximum:
            return None
        candidate["lam_states"] = increase(count, maximum)
    elif axis == "k_contraction":
        count = options.get("k_contraction_states")
        if count is None:
            return None
        limit = int(candidate.get("lam_states", 12))
        options["k_contraction_states"] = increase(int(count), limit)
        if options["k_contraction_states"] >= limit:
            options.pop("k_contraction_states")
            options.pop("k_contraction_reference_values", None)
    elif axis == "response_states":
        count = int(options.get("basis_response_states", 48))
        limit = min(maximum - int(candidate.get("lam_states", 12)),
                    int(policy.get("maximum_response_states", maximum)))
        if count >= limit:
            return None
        options["basis_response_states"] = increase(count, limit)
    elif axis == "roots":
        count = candidate.get("levels_per_j")
        if count is None:
            return None
        candidate["levels_per_j"] = max(int(count) + 4, int(np.ceil(count * grow)))
    elif axis == "fourier_cutoffs":
        cuts = options.get("fourier_cutoffs")
        if cuts is None:
            return None
        updated = [
            None if c is None else increase(c, (len(q) - 1) // 2)
            for q, c in zip(surface.coordinates, cuts, strict=True)
        ]
        if updated == list(cuts):
            return None
        options["fourier_cutoffs"] = updated
    elif axis == "grid":
        limits = policy["maximum_grid_sizes"]
        sizes = tuple(
            increase(n, limit) for n, limit in zip(surface.grid_shape, limits, strict=True)
        )
        if sizes == surface.grid_shape:
            return None
        # Preserve Nyquist parity and every exact translation order. A grid
        # incompatible with the molecular action is not a refinement.
        from math import lcm
        sizes = tuple(
            old + min((limit-old)//lcm(2,order),
                      (n-old+lcm(2,order)-1)//lcm(2,order))*lcm(2,order)
            for n,old,limit,order in zip(sizes,surface.grid_shape,limits,surface.symmetry_orders,strict=True)
        )
        if sizes == surface.grid_shape:
            return None
        factory = policy.get("_grid_factory")
        if factory is None:
            new_surface = refine_lam_grid(surface, sizes)
        else:
            template = refine_lam_grid(surface, sizes)
            new_surface = factory(template.coordinates)
            if new_surface.grid_shape != tuple(sizes):
                raise ValueError("physical grid builder returned the wrong grid")
    else:
        raise ValueError(f"unknown LAM refinement axis: {axis}")
    return new_surface, candidate


def converge_lam_basis(surface, request, policy=None):
    """Refine axes independently until all requested observables pass.

    Jacobian errors are measured per declared parameter scale. Certification
    compares a basis to independently enlarged bases, at fixed physical
    parameters; it is not a bound on the accuracy of the underlying QC PES.
    """
    policy = deepcopy(policy or {})
    current_request = deepcopy(request)
    current_request.setdefault("options", {}).pop("basis_convergence", None)
    state = current_request["options"].pop("optimizer_state", None)
    if state is not None:
        for variable, value in zip(current_request["variables"], state["values"], strict=True):
            variable["initial"] = value
        current_request["options"]["initial_trust_radius"] = state["trust_radius"]
    policy.setdefault("maximum_grid_sizes", [2 * n + 1 for n in surface.grid_shape])
    if len(policy["maximum_grid_sizes"]) != surface.dimension:
        raise ValueError("one maximum grid size per coordinate is required")
    tolerance = float(
        policy.get(
            "frequency_tolerance_MHz",
            0.01 * min(o["uncertainty_MHz"] for o in request["observations"]),
        )
    )
    derivative_tolerance = float(policy.get("jacobian_tolerance_MHz", tolerance))
    relative = float(policy.get("intensity_relative_tolerance", 1.0e-3))
    absolute = float(policy.get("intensity_absolute_tolerance", 1.0e-10))
    if min(tolerance, derivative_tolerance, relative, absolute) <= 0 or not np.all(
        np.isfinite([tolerance, derivative_tolerance, relative, absolute])
    ):
        raise ValueError("basis tolerances must be positive and finite")
    axes = tuple(
        policy.get(
            "axes",
            ("fourier_cutoffs", "lam_states", "k_contraction", "roots", "response_states", "grid"),
        )
    )
    if not axes:
        raise ValueError("at least one numerical refinement axis is required")
    if len(set(axes)) != len(axes):
        raise ValueError("numerical refinement axes must be unique")
    total_budget = policy.get("frequency_budget_MHz")
    if total_budget is not None:
        total_budget = np.broadcast_to(np.asarray(total_budget, dtype=float),
                                       (len(request["observations"]),)).copy()
        if not np.all(np.isfinite(total_budget)) or np.any(total_budget <= 0):
            raise ValueError("frequency budgets must be finite and positive")
        # Independent axes share a conservative L1 budget. Errors cannot
        # cancel to manufacture numerical certification.
        tolerance = min(tolerance, float(np.min(total_budget)) / len(axes))
        derivative_tolerance = min(derivative_tolerance, tolerance)
    intensity_budget = policy.get('intensity_budget')
    if intensity_budget is not None:
        intensity_budget = np.asarray(intensity_budget, float)
        if not np.all(np.isfinite(intensity_budget)) or np.any(intensity_budget <= 0):
            raise ValueError('intensity budgets must be finite and positive')

    from .lam_errors import LAMTruncationError

    history = []
    for attempt in range(int(policy.get("maximum_rounds", 6))):
        try:
            current = _evaluate(surface, current_request)
            break
        except LAMTruncationError as error:
            probe = _probe(surface, current_request, error.axis, policy)
            if probe is None:
                raise
            surface, current_request = probe
            history.append(dict(round=-1, axis=error.axis, status="refined_initial_truncation"))
    else:
        raise RuntimeError("initial LAM response exceeds the basis budget")
    changed = bool(history)
    if intensity_budget is not None and current['intensity'] is None:
        raise ValueError('intensity qualification requires a dipole surface or vector')
    if intensity_budget is not None:
        intensity_budget = np.broadcast_to(intensity_budget, np.shape(current['intensity'])).copy()
    success = False
    for iteration in range(int(policy.get("maximum_rounds", 6))):
        failed = False
        for axis in axes:
            probe = _probe(surface, current_request, axis, policy,
                           lam_state_count=len(current["solved"].basis.energies_MHz))
            if probe is None:
                exact = axis != "grid"
                represented = int(np.prod(surface.grid_shape))
                if axis == "lam_states":
                    cuts = current_request["options"].get("fourier_cutoffs")
                    if cuts is not None:
                        represented = min(represented, int(np.prod([
                            len(q) if c is None else 2*c+1
                            for q,c in zip(surface.coordinates,cuts,strict=True)])))
                    exact = int(current_request.get("lam_states", 12)) >= represented
                elif axis == "response_states":
                    exact = int(current_request["options"].get("basis_response_states", 48)) >= (
                        represented-int(current_request.get("lam_states", 12)))
                history.append(
                    dict(
                        round=iteration,
                        axis=axis,
                        status="full_represented_space" if exact else "budget_exhausted",
                    )
                )
                if not exact:
                    failed = True
                continue
            next_surface, next_request = probe
            try:
                other = _evaluate(
                    next_surface,
                    next_request,
                    reference=(current["solved"], surface, current["context"]),
                )
            except RuntimeError as error:
                history.append(dict(round=iteration, axis=axis, status="failed", reason=str(error)))
                failed = True
                continue
            df = float(np.max(abs(other["frequency"] - current["frequency"]), initial=0.0))
            dj = float(
                np.max(
                    abs((other["jacobian"] - current["jacobian"]) * current["scales"]), initial=0.0
                )
            )
            di = dij = 0.0
            if current["intensity"] is not None:
                denominator = intensity_budget / len(axes) if intensity_budget is not None else absolute + relative * np.maximum(
                    abs(current["intensity"]), abs(other["intensity"])
                )
                di = float(
                    np.max(
                        abs(other["intensity"] - current["intensity"]) / denominator, initial=0.0
                    )
                )
                dij = float(
                    np.max(
                        abs(
                            (other["intensity_jacobian"] - current["intensity_jacobian"])
                            * current["scales"]
                        )
                        / denominator[:, None],
                        initial=0.0,
                    )
                )
            passed = df <= tolerance and dj <= derivative_tolerance and max(di, dij) <= 1.0
            history.append(
                dict(
                    round=iteration,
                    axis=axis,
                    status="passed" if passed else "refined",
                    frequency_changes_MHz=(other["frequency"] - current["frequency"]).tolist(),
                    maximum_frequency_change_MHz=df,
                    maximum_scaled_jacobian_change_MHz=dj,
                    intensity_error_ratio=di,
                    intensity_jacobian_error_ratio=dij,
                    intensity_changes=None if current['intensity'] is None else
                        (other['intensity']-current['intensity']).tolist(),
                )
            )
            if not passed:
                surface, current_request, current = next_surface, next_request, other
                # Carry the tracked assignments into the promoted request.
                iterator = iter(current["context"]["observations"])
                for observation in current_request["observations"]:
                    for component in observation.get("components", [observation]):
                        resolved = next(iterator)
                        for side in ("lower", "upper"):
                            component[side] = dict(
                                J=resolved[side].J, reference_index=resolved[side].reference_index
                            )
                changed = True
                failed = True
        if not failed:
            success = True
            break
    last_round = max((row["round"] for row in history), default=-1)
    last_checks = [row for row in history if row["round"] == last_round]
    summed = np.zeros(len(current["frequency"]))
    summed_intensity = np.zeros_like(current['intensity']) if current['intensity'] is not None else None
    for row in last_checks:
        summed += np.abs(row.get("frequency_changes_MHz", np.zeros_like(summed)))
        if summed_intensity is not None and row.get('intensity_changes') is not None:
            summed_intensity += abs(np.asarray(row['intensity_changes']))
    budget_report = dict(
        enabled=total_budget is not None,
        complete=success,
        axes=list(axes),
        summed_absolute_changes_MHz=summed.tolist(),
        frequency_budget_MHz=None if total_budget is None else total_budget.tolist(),
        maximum_budget_ratio=None if total_budget is None else float(np.max(summed / total_budget)),
        interpretation="independent refinement differences, not a rigorous truncation bound",
    )
    if total_budget is not None:
        success = bool(success and np.all(summed <= total_budget))
        budget_report["complete"] = success
    if intensity_budget is not None:
        success = bool(success and np.all(summed_intensity <= intensity_budget))
        budget_report.update(complete=success, intensity_budget=intensity_budget.tolist(),
            summed_absolute_intensity_changes=summed_intensity.tolist())
    return LAMBasisQualification(
        surface,
        current_request,
        dict(
            schema="matrix.lock.lam_basis_qualification.v1",
            converged=success,
            numerical_budget=budget_report,
            refined=changed,
            history=history,
            frequency_tolerance_MHz=tolerance,
            jacobian_tolerance_MHz=derivative_tolerance,
            observation_count=len(current["frequency"]),
            intensity_checked=current["intensity"] is not None,
            scope="numerical convergence of represented surfaces; no QC model-accuracy claim",
        ),
    )


def fit_lam_with_converged_basis(surface, request, *, postfit_diagnostic=None):
    """Qualify before and after fitting; a refinement resumes at fitted values."""
    from .lam_fit import fit_lam_assigned_spectrum, LAMFitResult

    policy = request["options"]["basis_convergence"]
    local = deepcopy(request)
    local["options"].pop("basis_convergence")
    reports = []
    for _ in range(int(policy.get("maximum_fit_passes", 3))):
        qualified = converge_lam_basis(surface, local, policy)
        reports.append(qualified.report)
        if not qualified.report["converged"]:
            raise RuntimeError("LAM basis qualification exhausted its budget")
        surface, local = qualified.surface, qualified.request
        result = fit_lam_assigned_spectrum(surface, local, postfit_diagnostic=postfit_diagnostic)
        if not result.payload["success"]:
            return LAMFitResult(dict(result.payload, basis_qualification=reports), result.surface)
        fitted_request = deepcopy(local)
        for variable, record in zip(
            fitted_request["variables"], result.payload["variables"], strict=True
        ):
            variable["initial"] = record["value"]
        iterator = iter(result.payload["component_models"])
        for observation in fitted_request["observations"]:
            for component in observation.get("components", [observation]):
                component.update(next(iterator))
        final = converge_lam_basis(surface, fitted_request, policy)
        reports.append(final.report)
        if final.report["converged"] and not final.report["refined"]:
            payload = dict(result.payload, basis_qualification=reports)
            if payload.get("uncertainty_budget") is not None:
                from .prediction_budget import attach_numerical_budget

                payload["uncertainty_budget"] = attach_numerical_budget(
                    payload["uncertainty_budget"], _numerical_checks(final.report)
                )
            return LAMFitResult(payload, result.surface)
        surface, local = final.surface, final.request
    raise RuntimeError(
        "LAM fit/refinement exhausted its budget; last fitted values retained in qualification request"
    )


def _numerical_checks(report):
    if not report["converged"]:
        return None
    last = max((r["round"] for r in report["history"]), default=0)
    return {
        r["axis"]: r.get("frequency_changes_MHz", [0.0] * report["observation_count"])
        for r in report["history"]
        if r["round"] == last and r.get("status") in {"passed", "full_represented_space"}
    }


def fit_global_lam_with_converged_bases(systems, request):
    """Qualify every mass-specific basis, then resume any necessary refit."""
    from .lam_global_fit import fit_global_lam_spectra

    policy = request["options"]["basis_convergence"]
    local = deepcopy(request)
    local["options"].pop("basis_convergence")
    current = dict(systems)
    reports = []
    byname = {str(d["system"]): d for d in local["datasets"]}
    for variable in local["variables"]:
        for target in variable["targets"]:
            for item in byname[target["system"]]["request"]["variables"]:
                if item["path"] == target["path"]:
                    item["initial"] = (
                        target.get("offset", 0.0) + target.get("factor", 1.0) * variable["initial"]
                    )
    for _ in range(int(policy.get("maximum_fit_passes", 3))):
        for dataset in local["datasets"]:
            name = str(dataset["system"])
            qualified = converge_lam_basis(current[name], dataset["request"], policy)
            reports.append(dict(qualified.report, system=name, stage="before_fit"))
            if not qualified.report["converged"]:
                raise RuntimeError("joint LAM basis qualification exhausted its budget")
            current[name] = qualified.surface
            dataset["request"] = qualified.request
        result = fit_global_lam_spectra(current, local)
        if not result["success"]:
            return dict(result, basis_qualification=reports)
        parameters = {(p["system"], p["path"]): p["value"] for p in result["local_parameters"]}
        changed = False
        numerics = {}
        for dataset in local["datasets"]:
            name = str(dataset["system"])
            native = dataset["request"]
            for v in native["variables"]:
                v["initial"] = parameters[name, v["path"]]
            components = iter(result["component_models"][name])
            for observation in native["observations"]:
                for component in observation.get("components", [observation]):
                    component.update(next(components))
            qualified = converge_lam_basis(current[name], native, policy)
            reports.append(dict(qualified.report, system=name, stage="after_fit"))
            if not qualified.report["converged"]:
                raise RuntimeError("final joint LAM basis qualification exhausted its budget")
            changed |= qualified.report["refined"]
            current[name] = qualified.surface
            dataset["request"] = qualified.request
            numerics[name] = _numerical_checks(qualified.report)
        if not changed:
            result["basis_qualification"] = reports
            if local.get("uncertainty_budget") is not None:
                # Export comparisons in the exact concatenated dataset ordering.
                combined = {}
                offset = 0
                total = len(result["comparisons"])
                for dataset in local["datasets"]:
                    name = str(dataset["system"])
                    count = len(dataset["request"]["observations"])
                    for axis, delta in (numerics[name] or {}).items():
                        values = np.zeros(total)
                        values[offset : offset + count] = delta
                        combined[f"{name}:{axis}"] = values.tolist()
                    offset += count
                result["numerical_comparisons"] = combined
                from .prediction_budget import attach_numerical_budget

                result["uncertainty_budget"] = attach_numerical_budget(
                    result["uncertainty_budget"], combined
                )
            return result
        for v, fitted in zip(local["variables"], result["variables"], strict=True):
            v["initial"] = fitted["value"]
        local["options"].pop("resume_checkpoint", None)
        local["options"]["initial_trust_radius"] = result["optimizer_state"]["trust_radius"]
        local["options"]["initial_damping"] = result["optimizer_state"]["damping"]
    raise RuntimeError("joint fit/refinement exhausted its declared budget")
