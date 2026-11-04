"""Joint native LAM fits with shared physical parameters and blend observables.

Each system supplies its own mass-specific kinetic and rotational surfaces.
Targets share physical PES parameters explicitly; no reduced-mass scaling of
an arbitrary LAM metric is inferred. QC penalties declared in a local request
remain local; shared QC information should be declared once at global level.
"""

from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.linalg import block_diag
from matrix_numerics.least_squares import CovarianceWhitening, svd_covariance
from .fit_convergence import assess_statistical_convergence, statistical_trust_region
from .lam_measurements import native_strengths

GLOBAL_LAM_FIT_SCHEMA = "matrix.lock.global_lam_spectral_fit.v1"


def _backend(surface, request):
    from .lam_fit import fit_lam_assigned_spectrum

    from .lam_checkpoint import copy_workspace_request
    flat = copy_workspace_request(request)
    if request.get("options",{}).get("_workspace_resume_state") is not None:
        flat.setdefault("options",{})["_workspace_resume_state"]=request["options"]["_workspace_resume_state"]
    components = []
    measurements = []
    for row, observation in enumerate(flat["observations"]):
        group = observation.get("components", [observation])
        start = len(components)
        if not group:
            raise ValueError("an unresolved measurement requires components")
        for k, item in enumerate(group):
            components.append(
                dict(
                    label=f"component-{row}-{k}",
                    lower=item["lower"],
                    upper=item["upper"],
                    frequency_MHz=0.0,
                    uncertainty_MHz=1.0,
                )
            )
        weights = np.array([float(item.get("weight", 1.0)) for item in group])
        mode = observation.get("weight_mode", "fixed")
        if (
            np.any(weights <= 0)
            or not np.all(np.isfinite(weights))
            or mode not in {"fixed", "native"}
        ):
            raise ValueError("invalid blend component weights or weight_mode")
        measurements.append((np.arange(start, len(components)), weights, mode))
    flat["observations"] = components
    flat.pop("covariance_MHz2", None)
    flat["options"] = {**flat.get("options", {})}
    flat["options"].setdefault("jacobian_scheme", "analytic")
    flat["options"].pop("basis_convergence", None)
    for key in (
        "checkpoint_path",
        "resume_checkpoint",
        "linearization_path",
        "evaluation_log_path",
        "optimizer_state",
    ):
        flat["options"].pop(key, None)

    def capture(fun, jac, x0, lo, hi, scale, **context):
        def memo(function):
            cache = {}

            def evaluate(values):
                key = np.asarray(values, dtype=float).tobytes()
                if key not in cache:
                    result = function(values)
                    cache.clear()
                    cache[key] = result
                return cache[key]

            return evaluate

        return dict(
            context,
            fun=memo(fun),
            jac=memo(jac),
            x0=x0,
            lower=lo,
            upper=hi,
            scales=scale,
            dipole_au=request.get("dipole_au"),
            temperature_K=request.get("temperature_K"),
            measurements=measurements,
            count=len(components),
        )

    return fit_lam_assigned_spectrum(surface, flat, _evaluator_callback=capture)


def _measure(context, values, *, derivatives):
    frequencies = context["predict"](values)[0]
    jac = context["jac"](values)[: context["count"]] if derivatives else None
    out = []
    rows = []
    weights_out = []
    native_indices = (
        np.concatenate([idx for idx, _, mode in context["measurements"] if mode == "native"])
        if any(mode == "native" for _, _, mode in context["measurements"])
        else np.array([], int)
    )
    native = {}
    if len(native_indices):
        unique = np.unique(native_indices)
        strengths, ds = native_strengths(context, values, unique, derivatives=derivatives)
        native = {int(k): (s, d) for k, s, d in zip(unique, strengths, ds, strict=True)}
    for indices, base, mode in context["measurements"]:
        s = base.copy()
        ds = np.zeros((len(s), len(values)))
        if mode == "native":
            s *= np.array([native[int(k)][0] for k in indices])
            if derivatives:
                ds = base[:, None] * np.array([native[int(k)][1] for k in indices])
        total = float(s.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("unresolved measurement has zero total strength")
        weights = s / total
        center = float(weights @ frequencies[indices])
        out.append(center)
        weights_out.append(weights.tolist())
        if derivatives:
            rows.append(weights @ jac[indices] + (frequencies[indices] - center) @ ds / total)
    return np.array(out), np.array(rows) if derivatives else None, weights_out


def _fit_global_lam_spectra(systems, request, *, _backends=None, executor=None, resource_plan=None):
    """Fit several native Fourier/DVR systems using affine shared targets.

    ``datasets`` contain ``system`` and native LAM ``request``. Each global
    variable has ``targets`` with system/path and optional factor/offset.
    Experimental covariance covers all measurements, in dataset order; its
    training marginal enters the objective. Priors and predicates are separate
    correlated Gaussian blocks. The returned covariance is absolute (not
    rescaled by reduced chi square) and directly usable in downstream fits.
    """
    if request.get("schema") != GLOBAL_LAM_FIT_SCHEMA:
        raise ValueError("unsupported global LAM fit schema")
    datasets = request.get("datasets", [])
    variables = request.get("variables", [])
    names = [str(d["system"]) for d in datasets]
    if not names or len(set(names)) != len(names) or not variables:
        raise ValueError("unique datasets and variables are required")
    if any(n not in systems for n in names):
        raise ValueError("unknown LAM system")

    def prepare(name):
        dataset = next(d for d in datasets if str(d["system"]) == name)
        backend=_backend(systems[name],dataset['request'])
        restored=request.get('options',{}).get('_workspace_resume_global')
        if restored is not None:backend['response_workspace']=restored['response'][name]
        return name,backend

    if _backends is not None:
        backends = _backends
    elif executor is None:
        backends = dict(prepare(n) for n in names)
    else:
        backends = dict(executor.map(prepare, resource_plan["deterministic_schedule"]))
    nvar = len(variables)
    x = np.array([v["initial"] for v in variables], float)
    loaded=request.get("options",{}).get("_workspace_resume_global")
    if loaded is not None:x=np.asarray(loaded["optimizer_state"]["values"],float)
    scales = np.array([v.get("scale", max(abs(v["initial"]), 0.1)) for v in variables], float)
    lo = np.array([v.get("lower", -np.inf) for v in variables], float)
    hi = np.array([v.get("upper", np.inf) for v in variables], float)
    if len({v["name"] for v in variables}) != nvar:
        raise ValueError("global variable names must be unique")
    maps = {n: np.zeros((len(b["x0"]), nvar)) for n, b in backends.items()}
    offsets = {n: b["x0"].copy() for n, b in backends.items()}
    controlled = set()
    for column, var in enumerate(variables):
        if not var.get("targets"):
            raise ValueError("each global variable requires physical targets")
        for target in var["targets"]:
            name = str(target["system"])
            path = str(target["path"])
            key = (name, path)
            if name not in backends or key in controlled:
                raise ValueError("unknown system or duplicate physical target")
            paths = [v["path"] for v in backends[name]["variables"]]
            if path not in paths:
                raise ValueError(f"undeclared local LAM variable: {key}")
            k = paths.index(path)
            factor = float(target.get("factor", 1.0))
            offset = float(target.get("offset", 0.0))
            if not np.isfinite(factor) or not factor or not np.isfinite(offset):
                raise ValueError("invalid affine physical target")
            a = (backends[name]["lower"][k] - offset) / factor
            b = (backends[name]["upper"][k] - offset) / factor
            lo[column] = max(lo[column], min(a, b))
            hi[column] = min(hi[column], max(a, b))
            maps[name][k, column] = factor
            offsets[name][k] = offset
            controlled.add(key)
    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(scales))
        or np.any(scales <= 0)
        or np.any(np.isnan(lo))
        or np.any(np.isnan(hi))
        or np.any(lo >= hi)
        or np.any(x < lo)
        or np.any(x > hi)
    ):
        raise ValueError("infeasible global initial values, scales or bounds")
    observations = [o for d in datasets for o in d["request"]["observations"]]
    y = np.array([o["frequency_MHz"] for o in observations], float)
    sigma = np.array(
        [o["uncertainty_MHz"] / np.sqrt(o.get("weight", 1.0)) for o in observations], float
    )
    roles = [o.get("role", "training") for o in observations]
    if any(r not in {"training", "validation"} for r in roles):
        raise ValueError("invalid measurement role")
    training = np.array([i for i, r in enumerate(roles) if r == "training"], int)
    if not len(training) or not np.all(np.isfinite(y)):
        raise ValueError("finite training measurements are required")
    covariance = request.get("covariance_MHz2")
    if covariance is None and any("covariance_MHz2" in d["request"] for d in datasets):
        blocks = []
        for d in datasets:
            obs = d["request"]["observations"]
            s = np.array([o["uncertainty_MHz"] for o in obs])
            w = np.sqrt([o.get("weight", 1.0) for o in obs])
            c = np.array(d["request"].get("covariance_MHz2", np.diag(s * s)), float)
            CovarianceWhitening(s, c)
            blocks.append(c / w[:, None] / w[None, :])
        covariance = block_diag(*blocks)
    elif covariance is not None:
        covariance = np.array(covariance, float)
        raw_sigma = np.array([o["uncertainty_MHz"] for o in observations])
        CovarianceWhitening(raw_sigma, covariance)
        weights = np.sqrt([o.get("weight", 1.0) for o in observations])
        covariance = covariance / weights[:, None] / weights[None, :]
    CovarianceWhitening(sigma)
    whitening = CovarianceWhitening(
        sigma[training], None if covariance is None else covariance[np.ix_(training, training)]
    )
    penalty = []
    for k, v in enumerate(variables):
        if v.get("prior") is not None:
            penalty.append(dict(index=k, **v["prior"]))
    names_var = [v["name"] for v in variables]
    predicates = []
    for item in request.get("predicates", []):
        if item["name"] not in names_var:
            raise ValueError("unknown QM predicate variable")
        predicates.append(
            dict(index=names_var.index(item["name"]), value=item["value"], sigma=item["sigma"])
        )
    penalty_blocks = []
    for items, key in ((penalty, "prior_covariance"), (predicates, "predicate_covariance")):
        if items:
            penalty_blocks.append(
                (items, CovarianceWhitening([i["sigma"] for i in items], request.get(key)))
            )
    cache = {}
    all_sensitivities = {}

    def evaluate(values, *, with_jacobian=False):
        key = np.asarray(values, dtype=float).tobytes()
        if key in cache and (not with_jacobian or cache[key][1] is not None):
            return cache[key]
        predictions = []
        derivatives = []
        pr = []
        pj = []
        component_weights = []

        def one_system(name):
            backend = backends[name]
            local = offsets[name] + maps[name] @ values
            pred, jac, w = _measure(backend, local, derivatives=with_jacobian)
            r = backend["fun"](local)[backend["count"] :]
            penalty_jac = (
                backend["jac"](local)[backend["count"] :] @ maps[name] if with_jacobian else None
            )
            return name, (pred, None if jac is None else jac @ maps[name], w, r, penalty_jac)

        computed = (
            dict(one_system(n) for n in names)
            if executor is None
            else dict(executor.map(one_system, resource_plan["deterministic_schedule"]))
        )
        for name in names:
            pred, jac, w, r, penalty_jac = computed[name]
            predictions.extend(pred)
            component_weights.extend(w)
            pr.extend(r)
            if with_jacobian:
                derivatives.extend(jac)
                pj.extend(penalty_jac)
        pred = np.array(predictions)
        residual = [*whitening.apply((pred - y)[training]), *pr]
        rows = None
        if with_jacobian:
            j = np.array(derivatives)
            all_sensitivities.clear()
            all_sensitivities[key] = j
            rows = [*whitening.apply(j[training]), *pj]
        for items, w in penalty_blocks:
            indices = [i["index"] for i in items]
            centers = np.array([i["value"] for i in items])
            residual.extend(w.apply(values[indices] - centers))
            if with_jacobian:
                rows.extend(w.apply(np.eye(nvar)[indices]))
        result = (
            np.array(residual),
            None if rows is None else np.array(rows),
            pred,
            component_weights,
        )
        cache.clear()
        cache[key] = result
        return result

    options = request.get("options", {})
    state = {} if loaded is None else dict(loaded["optimizer_state"])
    if options.get("loss", "linear") != "linear":
        raise ValueError("global statistical fitting requires linear loss")
    # Contract includes surfaces as well as assignments and all physical targets.
    from .lam_checkpoint import copy_workspace_request
    contract = copy_workspace_request(request)
    contract.pop("options", None)
    for dataset in contract["datasets"]:
        local_options = dataset["request"].get("options", {})
        for key in (
            "checkpoint_path",
            "resume_checkpoint",
            "max_nfev",
            "linearization_path",
            "evaluation_log_path",
            "workers",
            "jacobian_workers",
            "block_workers",
        ):
            local_options.pop(key, None)
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, default=str).encode())
    for name in names:
        s = systems[name]
        for grid in s.coordinates:
            digest.update(np.ascontiguousarray(grid).tobytes())
        digest.update(
            repr(
                (s.periodic, s.symmetry_orders, s.distortion_reduction, s.nuclear_spin_weights)
            ).encode()
        )
        if s.permutation_inversion_symmetry is not None:
            digest.update(
                json.dumps(s.permutation_inversion_symmetry.to_dict(), sort_keys=True).encode()
            )
        for field in (
            "potential_MHz",
            "F_MHz",
            "rho",
            "R_MHz",
            "dipole_au",
            "quartic_distortion_MHz",
            "sextic_distortion_MHz",
            "geometric_potential_MHz",
            "lam_quartic_kinetic_MHz",
            "p2_rotational_metric_MHz",
            "p_rotational_cubic_MHz",
            "p_rotational_anticommutator_MHz",
            "pi2_rotation_MHz",
        ):
            value = getattr(s, field)
            if value is None:
                digest.update(b"None")
                continue
            a = np.ascontiguousarray(value)
            digest.update(str((a.shape, a.dtype.str)).encode())
            digest.update(a.tobytes())
    fingerprint = digest.hexdigest()
    if options.get("resume_checkpoint"):
        stored = json.loads(Path(options["resume_checkpoint"]).read_text())
        if stored["contract"] != fingerprint:
            raise ValueError("global LAM checkpoint contract mismatch")
        state = stored["optimizer_state"]
        x = np.array(state["values"], float)
    checkpoint = options.get("checkpoint_path")

    def save(newstate):
        accepted_values=np.asarray(newstate["values"],float)
        for name in names:
            promote=backends[name].get("promote_accepted")
            if promote is not None:
                promote(offsets[name]+maps[name]@accepted_values)
        key=accepted_values.tobytes()
        for old in list(cache):
            if old!=key:cache.pop(old)
        state.update(newstate)
        if options.get('workspace_checkpoint_path'):
            from .lam_checkpoint import write_lam_workspace,snapshot_response_workspace,lam_restart_contract
            native={n:backends[n]['snapshot_workspace'](offsets[n]+maps[n]@np.asarray(state['values'])) for n in names}
            response={n:snapshot_response_workspace(backends[n].get('response_workspace',{})) for n in names}
            write_lam_workspace(options['workspace_checkpoint_path'],dict(native=native,response=response,optimizer_state=state),
                contract=lam_restart_contract(systems,request))
        if checkpoint:
            path = Path(checkpoint)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            checkpoint_payload=dict(contract=fingerprint,optimizer_state=state)
            accepted=cache.get(key)
            if accepted is not None:
                error=accepted[2]-y
                checkpoint_payload["metrics"]=dict(
                    training_rms_MHz=float(np.sqrt(np.mean(error[training]**2))),
                    validation_rms_MHz=(float(np.sqrt(np.mean(error[np.array(roles)=="validation"]**2)))
                        if "validation" in roles else None),
                    chi_square=float(accepted[0]@accepted[0]))
            temporary.write_text(json.dumps(checkpoint_payload,indent=2))
            temporary.replace(path)

    class Stationary(Exception):
        pass

    assessment = None

    def checked(values):
        nonlocal assessment
        r, j, _, _ = evaluate(values, with_jacobian=True)
        assessment = assess_statistical_convergence(
            r,
            j,
            values,
            lo,
            hi,
            data_count=len(training),
            data_correction_transform=whitening.standardized_transform,
            **{k: v for k, v in options.get("convergence", {}).items() if k != "mode"},
        )
        if assessment["converged"]:
            raise Stationary
        return j

    try:
        optimized = statistical_trust_region(
            lambda z: evaluate(z)[0],
            checked,
            x,
            lo,
            hi,
            scales,
            max_nfev=int(options.get("max_nfev", 100)),
            initial_radius=float(
                state.get("trust_radius", options.get("initial_trust_radius", 1.0))
            ),
            initial_damping=float(state.get("damping", options.get("initial_damping", 0.0))),
            state_callback=save,
        )
        x = optimized.x
        message = optimized.message
    except Stationary:
        x = np.array(state["values"])
        message = "Statistical convergence reached."
    r, j, pred, component_weights = evaluate(x, with_jacobian=True)
    assessment = assess_statistical_convergence(
        r,
        j,
        x,
        lo,
        hi,
        data_count=len(training),
        data_correction_transform=whitening.standardized_transform,
        **{k: v for k, v in options.get("convergence", {}).items() if k != "mode"},
    )
    cov, rank, singular = svd_covariance(j, scales=scales)
    comparisons = [
        dict(
            label=o.get("label", str(i)),
            role=roles[i],
            frequency_MHz=float(y[i]),
            calculated_MHz=float(pred[i]),
            calculated_minus_observed_MHz=float(pred[i] - y[i]),
            component_weights=component_weights[i],
        )
        for i, o in enumerate(observations)
    ]
    projection = np.vstack([maps[n] for n in names])
    local_parameters = [
        dict(system=n, path=v["path"], value=float(value))
        for n in names
        for v, value in zip(backends[n]["variables"], offsets[n] + maps[n] @ x, strict=True)
    ]
    metrics = dict(
        system_count=len(names),
        observation_count=len(y),
        training_count=len(training),
        validation_count=len(y) - len(training),
        rms_residual_MHz=float(np.sqrt(np.mean((pred - y) ** 2))),
        training_rms_residual_MHz=float(np.sqrt(np.mean((pred - y)[training] ** 2))),
    )
    rotational_projection = None
    manifold_records = []
    if request.get("rotational_observables"):
        from .lam_rotational_observables import lam_rotational_observables

        records = []
        sensitivities = []
        used = set()
        for specification in request["rotational_observables"]:
            name = str(specification["system"])
            if name not in names or name in used:
                raise ValueError("rotational projections require unique known systems")
            used.add(name)
            dataset = next(d for d in datasets if str(d["system"]) == name)
            local = offsets[name] + maps[name] @ x
            if specification.get("maximum_rotation_order",2)>2:
                from .ordered_downfold import lam_ordered_effective_hamiltonian
                model=lam_ordered_effective_hamiltonian(systems[name],dataset["request"],local,
                    active_states=specification.get("active_states",[0]),
                    maximum_order=int(specification["maximum_rotation_order"]),
                    include_gap_MHz=float(specification.get("include_gap_MHz",0.)),
                    maximum_terms=int(specification.get("maximum_terms",100000)),
                    memory_budget_bytes=int(specification.get("memory_budget_bytes",512*1024**2)))
                manifold_records.append(dict(model.to_dict(),system=name,
                    parameter_covariance=cov.tolist(),
                    uncertainty_propagated=False))
                continue
            if "active_states" in specification:
                from .lam_manifold import lam_effective_manifold
                model = lam_effective_manifold(
                    systems[name], dataset["request"], local,
                    active_states=specification["active_states"],
                    include_gap_MHz=float(specification.get("include_gap_MHz", 0.)),
                )
                manifold_records.append(dict(model.to_dict(), system=name,
                    local_to_global_jacobian=maps[name].tolist(),
                    parameter_covariance=cov.tolist()))
                continue
            record = lam_rotational_observables(
                systems[name],
                dataset["request"],
                local,
                lam_state=int(specification.get("lam_state", 0)),
            )
            sensitivities.extend(np.asarray(record["jacobian"]) @ maps[name])
            records.append(dict(record, system=name))
        sensitivity = np.asarray(sensitivities).reshape((-1, nvar))
        rotational_projection = dict(
            schema="matrix.lock.joint_rotational_observables.v1",
            records=records,
            global_jacobian=sensitivity.tolist(),
            covariance_MHz2=(sensitivity @ cov @ sensitivity.T).tolist(),
            ordering=[f"{r['system']}:{axis}" for r in records for axis in ("A", "B", "C")],
        )
    from .lam_fit import _lam_level_index

    component_models = {}
    for name in names:
        _, _, model, mapping = backends[name]["predict"](offsets[name] + maps[name] @ x)
        component_models[name] = [
            dict(
                (side, dict(J=o[side].J, reference_index=_lam_level_index(o[side], model, mapping)))
                for side in ("lower", "upper")
            )
            for o in backends[name]["observations"]
        ]
    result = dict(
        schema="matrix.lock.global_lam_spectral_fit_result.v1",
        success=assessment["converged"],
        message=message,
        status="converged"
        if assessment["converged"]
        else "rank_deficient"
        if rank < nvar
        else "failed",
        metrics=metrics,
        local_parameters=local_parameters,
        local_parameter_covariance=(projection @ cov @ projection.T).tolist(),
        covariance_note="Absolute Gaussian covariance from scaled Jacobian SVD; no chi-square rescaling. Rank deficiency must be inspected.",
        variables=[
            dict(name=v["name"], value=float(value), targets=v["targets"])
            for v, value in zip(variables, x, strict=True)
        ],
        covariance=cov.tolist(),
        rank=rank,
        singular_values=singular.tolist(),
        convergence=assessment,
        chi_square=float(r @ r),
        data_chi_square=float(
            whitening.apply((pred - y)[training]) @ whitening.apply((pred - y)[training])
        ),
        comparisons=comparisons,
        optimizer_state=state,
        systems=names,
        tracking={
            n: d["request"].get("options", {}).get("tracking", "overlap")
            for n, d in zip(names, datasets, strict=True)
        },
        component_models=component_models,
        response_diagnostics={
            n: dict(
                cache_hits=backends[n].get("response_workspace", {}).get("cache_hits", 0),
                warm_bytes=backends[n].get("response_workspace", {}).get("warm_bytes", 0),
                counters={
                    key: sum(
                        (solver["counts"] if isinstance(solver,dict) else solver.counts)[key]
                        for solver in backends[n]
                        .get("response_workspace", {})
                        .get("prepared", {})
                        .values()
                    )
                    for key in ("solves", "iterations", "matvecs", "warm_starts")
                },
            )
            for n in names
        },
        parallelism=resource_plan,
        rotational_observables=rotational_projection,
        manifold_observables=manifold_records,
        mass_treatment="explicit system-specific surfaces; shared affine physical targets",
    )

    if request.get("structured_discrepancy") is not None:
        from .structured_discrepancy import fitted_spectral_discrepancy
        discrepancy_observations=observations
        if request['structured_discrepancy'].get('intervals') is not None:
            discrepancy_observations=[dict(o,system=d['system']) for d in datasets for o in d['request']['observations']]
        flat = dict(observations=discrepancy_observations, structured_discrepancy=request["structured_discrepancy"])
        if covariance is not None:
            weights = np.sqrt([o.get("weight", 1.) for o in observations])
            flat["covariance_MHz2"] = (covariance * weights[:, None] * weights[None, :]).tolist()
        result["structured_discrepancy"] = fitted_spectral_discrepancy(
            flat, result, all_sensitivities[np.asarray(x, dtype=float).tobytes()],
            training_whitener=whitening, training_jacobian=j[:len(training)])
    if request.get("qc_acquisition") is not None:
        from .qc_acquisition import select_qc_candidates
        if not result["success"]:
            raise ValueError("QC acquisition requires a converged identifiable fit")
        spec = dict(request["qc_acquisition"])
        rows = spec.pop("target_rows", list(range(len(observations))))
        result["qc_acquisition"] = select_qc_candidates(
            cov, all_sensitivities[np.asarray(x, dtype=float).tobytes()][rows], **spec)

    if request.get("uncertainty_budget") is not None:
        from .prediction_budget import lam_prediction_budget

        flat_request = dict(
            observations=observations, uncertainty_budget=request["uncertainty_budget"]
        )
        if covariance is not None:
            weights = np.sqrt([o.get("weight", 1.0) for o in observations])
            flat_request["covariance_MHz2"] = (
                covariance * weights[:, None] * weights[None, :]
            ).tolist()
        result["uncertainty_budget"] = lam_prediction_budget(
            flat_request,
            result,
            all_sensitivities[np.asarray(x, dtype=float).tobytes()],
            training_whitener=whitening,
            training_jacobian=j[: len(training)],
        )
    return result


def fit_global_lam_spectra(systems, request, *, _backends=None):
    """Joint native fit under one deterministic CPU and memory budget."""
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import nullcontext
    from threadpoolctl import threadpool_limits
    from .resource_planner import plan_lam_fit_resources

    if request.get("options", {}).get("basis_convergence") is not None and _backends is None:
        from .lam_qualification import fit_global_lam_with_converged_bases

        return fit_global_lam_with_converged_bases(systems, request)
    normalized = deepcopy(request)
    plan = plan_lam_fit_resources(systems, normalized)
    if normalized.get('options',{}).get('resume_workspace'):
        from .lam_checkpoint import read_lam_workspace,lam_restart_contract
        restored=read_lam_workspace(normalized['options']['resume_workspace'],contract=lam_restart_contract(systems,normalized))
        normalized['options']['_workspace_resume_global']=restored
        for dataset in normalized['datasets']:
            dataset['request'].setdefault('options',{})['_workspace_resume_state']=restored['native'][dataset['system']]
    for dataset in normalized["datasets"]:
        options = dataset["request"].setdefault("options", {})
        inner = plan["workers_per_system"]
        jac = min(inner, int(options.get("jacobian_workers", inner)))
        options.update(
            workers=inner, jacobian_workers=jac, block_workers=max(1, inner // max(jac, 1))
        )
    pool = (
        ThreadPoolExecutor(max_workers=plan["dataset_workers"])
        if plan["dataset_workers"] > 1
        else nullcontext(None)
    )
    with threadpool_limits(limits=1), pool as executor:
        return _fit_global_lam_spectra(
            systems, normalized, _backends=_backends, executor=executor, resource_plan=plan
        )


def fit_lam_measurements(surface, request):
    """Single-system adapter; ordinary resolved fits retain their original path."""
    from .lam_fit import LAMFitResult, _apply_lam_variables

    backend = _backend(surface, request)
    variables = backend["variables"]
    global_variables = []
    for k, v in enumerate(variables):
        name = v.get("name", f"p{k + 1}")
        global_variables.append(
            dict(
                name=name,
                initial=v.get("initial", 1.0 if v["path"].endswith(".scale") else 0.0),
                scale=v.get("scale", max(abs(v.get("initial", 0.0)), 0.1)),
                lower=v.get("lower", -np.inf),
                upper=v.get("upper", np.inf),
                targets=[dict(system="system", path=v["path"])],
            )
        )
    global_request = dict(
        schema=GLOBAL_LAM_FIT_SCHEMA,
        datasets=[dict(system="system", request=request)],
        variables=global_variables,
        options=request.get("options", {}),
    )
    # All post-fit diagnostics use the shared global measurement residual and
    # Jacobian. The adapter owns no separate optimizer or sensitivity cache.
    for key in ("structured_discrepancy", "qc_acquisition", "uncertainty_budget"):
        if request.get(key) is not None:
            global_request[key] = request[key]
    result = fit_global_lam_spectra(
        {"system": surface}, global_request, _backends={"system": backend}
    )
    # Local QC penalties are already included by the native backend once.
    fitted = _apply_lam_variables(
        surface, backend["variables"], np.array([v["value"] for v in result["variables"]])
    )
    for record, local in zip(result["variables"], backend["variables"], strict=True):
        record.update(path=local["path"], initial=local["initial"], scale=local["scale"])
    values = np.array([v["value"] for v in result["variables"]])
    solved = backend["predict"](values)[2]
    result["operator_parameters_MHz"] = solved.operator_parameters_MHz
    result["tracking"] = request.get("options", {}).get("tracking", "overlap")
    result["schema"] = "matrix.lock.lam_spectral_fit_result.v1"
    return LAMFitResult(result, fitted)


def fit_isotopologue_tunneling_results(results, request):
    """Construct mass-specific surfaces with SMITH/Qim and fit a shared PES."""
    from .tunneling_coordinates import isotopologue_tunneling_surfaces_from_qim_results

    return fit_global_lam_spectra(
        isotopologue_tunneling_surfaces_from_qim_results(results), request
    )
