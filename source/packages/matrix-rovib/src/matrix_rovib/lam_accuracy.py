"""One observable-space budget for grid, Fourier, LAM, K and response errors."""

from __future__ import annotations
from copy import deepcopy
import numpy as np
from .lam_qualification import LAMBasisQualification, converge_lam_basis, _probe, _evaluate


def _carry_assignments(request, evaluated):
    iterator = iter(evaluated["context"]["observations"])
    for row in request["observations"]:
        for component in row.get("components", [row]):
            resolved = next(iterator)
            for side in ("lower", "upper"):
                selector = resolved[side]
                component[side] = dict(J=selector.J, reference_index=selector.reference_index)
    return request


def qualify_lam_accuracy(surface, request, policy=None, *, grid_builder=None):
    """Allocate frequency/intensity budgets and check simultaneous enlargement.

    The L1 envelope of independent refinements and a joint-refinement
    comparison must both pass. These are convergence estimators, not rigorous
    upper bounds or QC error estimates. A physical grid builder reconstructs
    metric and volume scalar; interpolation alone is explicitly distinguished.
    """
    policy = deepcopy(policy or {})
    axes = tuple(
        policy.get(
            "axes",
            ("grid", "fourier_cutoffs", "lam_states", "k_contraction", "roots", "response_states"),
        )
    )
    if not axes or len(set(axes)) != len(axes):
        raise ValueError("unique numerical refinement axes required")
    fraction = float(policy.get("fraction_of_experimental_sigma", 0.01))
    if not np.isfinite(fraction) or fraction <= 0:
        raise ValueError("positive numerical accuracy fraction required")
    sigma = np.array(
        [o["uncertainty_MHz"] / np.sqrt(o.get("weight", 1.0)) for o in request["observations"]]
    )
    purpose=policy.get('purpose','refine')
    if purpose not in {'explore','refine'}:
        raise ValueError('qualification purpose must be explore or refine')
    if purpose=='explore':
        if 'observable_budget_MHz' not in policy:
            raise ValueError('exploration requires an explicit observable_budget_MHz')
        budget=np.broadcast_to(np.asarray(policy['observable_budget_MHz'],float),sigma.shape).copy()
        budget=np.minimum(budget,float(policy.get('frequency_tolerance_MHz',np.inf)))
    else:
        budget = np.minimum(sigma * fraction, float(policy.get("frequency_tolerance_MHz", np.inf)))
    if np.any(budget <= 0) or not np.all(np.isfinite(budget)):
        raise ValueError("invalid line frequency budget")
    local_policy = dict(policy, frequency_tolerance_MHz=float(np.min(budget)) / len(axes))
    local_policy.setdefault("jacobian_tolerance_MHz", local_policy["frequency_tolerance_MHz"])
    local_policy.setdefault("maximum_grid_sizes", [2 * n + 1 for n in surface.grid_shape])
    local_policy["axes"] = axes
    intensity_budget = policy.get('intensity_budget')
    if intensity_budget is not None:
        intensity_budget = np.asarray(intensity_budget, float)
        if not np.all(np.isfinite(intensity_budget)) or np.any(intensity_budget <= 0):
            raise ValueError('positive finite intensity budgets required')
        local_policy['intensity_budget'] = intensity_budget
    if grid_builder is not None:
        local_policy["_grid_factory"] = grid_builder
    history = []
    local = deepcopy(request)
    changed = False
    for attempt in range(int(policy.get("maximum_coupled_rounds", 3))):
        result = converge_lam_basis(surface, local, local_policy)
        history.append(result.report)
        changed |= result.report["refined"]
        surface, local = result.surface, result.request
        if not result.report["converged"]:
            break
        base = _evaluate(surface, local)
        joint_surface = surface
        joint_request = deepcopy(local)
        envelope = np.zeros(len(sigma))
        intensity_envelope = None if base['intensity'] is None else np.zeros_like(base['intensity'])
        last_round = max(
            (item["round"] for item in result.report["history"] if item["round"] >= 0), default=0
        )
        for item in result.report["history"]:
            if item.get("round") == last_round and item.get("status") == "passed":
                envelope += abs(np.asarray(item["frequency_changes_MHz"]))
                if intensity_envelope is not None and item.get('intensity_changes') is not None:
                    intensity_envelope += abs(np.asarray(item['intensity_changes']))
        for axis in axes:
            proposed = _probe(joint_surface, joint_request, axis, local_policy)
            if proposed is not None:
                joint_surface, joint_request = proposed
        other = _evaluate(
            joint_surface, joint_request, reference=(base["solved"], surface, base["context"])
        )
        difference = other["frequency"] - base["frequency"]
        passed = bool(np.all(envelope <= budget) and np.all(abs(difference) <= budget))
        intensity_change = None if base['intensity'] is None else other['intensity']-base['intensity']
        if intensity_budget is not None:
            if intensity_change is None:
                raise ValueError('intensity qualification requires a dipole surface or vector')
            intensity_budget = np.broadcast_to(intensity_budget, np.shape(intensity_change)).copy()
            passed = bool(passed and np.all(intensity_envelope <= intensity_budget)
                          and np.all(abs(intensity_change) <= intensity_budget))
        checks = dict(
            independent_L1_estimate_MHz=envelope.tolist(),
            joint_frequency_change_MHz=difference.tolist(),
            line_budget_MHz=budget.tolist(),
            joint_passed=passed,
            independent_L1_intensity_estimate=None if intensity_change is None else intensity_envelope.tolist(),
            joint_intensity_change=None if intensity_change is None else intensity_change.tolist(),
            intensity_budget=None if intensity_budget is None else intensity_budget.tolist(),
            intensity_converged=bool(intensity_budget is not None and passed),
        )
        if passed:
            return LAMBasisQualification(
                surface,
                local,
                dict(
                    schema="matrix.lock.lam_accuracy.v1",
                    converged=True,
                    purpose=purpose,
                    refined=changed,
                    history=history,
                    **checks,
                    grid_metric="reconstructed physical chart"
                    if grid_builder
                    else "interpolated represented surface",
                    scope="independent and simultaneous numerical refinements; no rigorous error bound or QC accuracy claim",
                ),
            )
        surface, local = joint_surface, _carry_assignments(joint_request, other)
        changed = True
    return LAMBasisQualification(
        surface,
        local,
        dict(
            schema="matrix.lock.lam_accuracy.v1",
            converged=False,
            refined=changed,
            history=history,
            reason="numerical refinement budget exhausted",
        ),
    )


def fit_structure_pes_with_accuracy(builder, request):
    """Refine the numerical representation and refit from the accepted point."""
    from .structure_spectrum import fit_structure_pes_spectra

    local = deepcopy(request)
    policy = local["options"].pop("numerical_accuracy")
    uncertainty = local["options"].pop("nonlinear_uncertainty", None)
    reports = []
    last = None

    def restart_request():
        output = deepcopy(local)
        output.setdefault("options", {})["numerical_accuracy"] = deepcopy(policy)
        if uncertainty is not None:
            output["options"]["nonlinear_uncertainty"] = deepcopy(uncertainty)
        return output

    maximum = int(policy.get("maximum_fit_passes", 3))
    if maximum < 1:
        raise ValueError("positive structural refinement pass budget required")
    for iteration in range(maximum):
        last = fit_structure_pes_spectra(builder, local)
        local = deepcopy(last["restart_request"])
        if not last["success"]:
            return dict(
                last,
                numerical_accuracy=dict(
                    converged=False,
                    passes=reports,
                    reason="physical fit not converged",
                    restart_request=restart_request(),
                ),
            )
        values = {v["name"]: v["value"] for v in last["variables"]}
        surfaces = builder(values)
        promoted = False
        all_passed = True
        for dataset in local["datasets"]:
            name = str(dataset["system"])
            factory = None
            if hasattr(builder, "with_grids"):
                factory = lambda grids, n=name: builder.with_grids(n, grids)(values)[n]
            elif "grid" in policy.get("axes", ("grid",)):
                raise ValueError("structural grid refinement requires a producer with with_grids()")
            qualification = qualify_lam_accuracy(
                surfaces[name], dataset["request"], policy, grid_builder=factory
            )
            reports.append(dict(iteration=iteration, system=name, **qualification.report))
            all_passed &= qualification.report["converged"]
            promoted |= qualification.report["refined"]
            dataset["request"] = qualification.request
            if qualification.surface.grid_shape != surfaces[name].grid_shape:
                builder = builder.with_grids(name, qualification.surface.coordinates)
        if hasattr(builder, "to_specification"):
            local["producer"] = builder.to_specification()
        for variable in local["variables"]:
            variable["initial"] = values[variable["name"]]
        if all_passed and not promoted:
            if uncertainty is not None:
                local["options"]["nonlinear_uncertainty"] = uncertainty
                last = fit_structure_pes_spectra(builder, local)
            return dict(
                last,
                numerical_accuracy=dict(
                    converged=True,
                    passes=reports,
                    refit_initialization="last accepted physical parameters",
                ),
                qualified_request=restart_request(),
                restart_request=restart_request(),
            )
    return dict(
        last,
        success=False,
        numerical_accuracy=dict(
            converged=False,
            passes=reports,
            reason="fit/refinement budget exhausted",
            restart_request=restart_request(),
        ),
    )
