"""Nonlinear fitting of assigned DVR/LAM-rotation transitions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from fnmatch import fnmatchcase
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
from scipy.optimize import OptimizeResult, least_squares, linear_sum_assignment

from .fit_convergence import assess_statistical_convergence, bounded_gauss_newton, statistical_trust_region

from .adaptive_solver import track_eigenstates_by_subspace
from .native_rotor import centrifugal_distortion_operators
from .lam_rotation import (
    LAMBasis,
    LAMRotationResult,
    LAMRotationSurface,
    _apply_lam_hamiltonian,
    _apply_momentum,
    _contracted_lam_operators,
    _hermitian,
    _rotational_operators,
    solve_lam_basis,
    solve_lam_rotation,
)


LAM_SPECTRAL_FIT_SCHEMA = "matrix.lock.lam_spectral_fit.v1"
LAM_SPECTRAL_FIT_RESULT_SCHEMA = "matrix.lock.lam_spectral_fit_result.v1"


@dataclass(frozen=True)
class LAMLevelSelector:
    J: int
    Ka: int | None = None
    Kc: int | None = None
    torsional_state: int = 0
    lam_symmetry: str | None = None
    reference_index: int | None = None
    state_id: str | None = None
    reference_energy_MHz: float | None = None

    def __post_init__(self) -> None:
        if self.J < 0 or self.torsional_state < 0:
            raise ValueError("LAM quantum numbers must be nonnegative")
        if (self.Ka is None) != (self.Kc is None):
            raise ValueError("LAM selectors require both Ka and Kc or neither")
        if self.reference_index is not None and self.reference_index < 0:
            raise ValueError("LAM reference_index must be nonnegative")
        if self.state_id is not None and not self.state_id:
            raise ValueError("LAM state_id cannot be empty")
        if self.reference_energy_MHz is not None and not math.isfinite(self.reference_energy_MHz):
            raise ValueError("LAM reference energies must be finite")
        if self.reference_energy_MHz is not None and self.state_id is None:
            raise ValueError("reference_energy_MHz requires a persistent state_id")
        if self.reference_index is None and self.reference_energy_MHz is None and self.Ka is None:
            raise ValueError(
                "LAM selectors require Ka/Kc, reference_index, or a state_id reference energy"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LAMLevelSelector":
        return cls(
            J=int(payload["J"]),
            Ka=None if payload.get("Ka") is None else int(payload["Ka"]),
            Kc=None if payload.get("Kc") is None else int(payload["Kc"]),
            torsional_state=int(payload.get("torsional_state", 0)),
            lam_symmetry=None
            if payload.get("lam_symmetry") is None
            else str(payload["lam_symmetry"]),
            reference_index=None
            if payload.get("reference_index") is None
            else int(payload["reference_index"]),
            state_id=None if payload.get("state_id") is None else str(payload["state_id"]),
            reference_energy_MHz=None
            if payload.get("reference_energy_MHz") is None
            else float(payload["reference_energy_MHz"]),
        )


@dataclass(frozen=True)
class LAMFitResult:
    payload: Mapping[str, Any]
    surface: LAMRotationSurface

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def fit_lam_assigned_spectrum(
    surface: LAMRotationSurface,
    request: Mapping[str, Any],
    *,
    postfit_diagnostic=None,
    _evaluator_callback=None,
) -> LAMFitResult:
    """Fit potential, kinetic, rho, or rotational metric surface parameters."""
    if request.get("schema") != LAM_SPECTRAL_FIT_SCHEMA:
        raise ValueError(f"unsupported LAM fit schema: {request.get('schema')!r}")
    if _evaluator_callback is None and request.get('options',{}).get('basis_convergence') is not None:
        from .lam_qualification import fit_lam_with_converged_basis
        return fit_lam_with_converged_basis(surface,request,postfit_diagnostic=postfit_diagnostic)
    if any('components' in item for item in request.get('observations', ())):
        if _evaluator_callback is not None or postfit_diagnostic is not None:
            raise ValueError('blend diagnostics use the global LAM measurement interface')
        from .lam_global_fit import fit_lam_measurements
        return fit_lam_measurements(surface,request)
    operator_manifest = request.get("operator_manifest")
    normalized_operator_manifest = None
    operator_defaults: Mapping[str, float] = {}
    if operator_manifest is not None:
        from .general_lam_operator import normalize_general_lam_operator_manifest

        normalized_operator_manifest = normalize_general_lam_operator_manifest(
            operator_manifest
        )
        operator_defaults = normalized_operator_manifest.parameters_MHz
    variables = tuple(
        _lam_variable(item, surface, index, operator_defaults)
        for index, item in enumerate(request.get("variables", ()))
    )
    observations = tuple(
        _lam_observation(item, index) for index, item in enumerate(request.get("observations", ()))
    )
    if not observations or (not variables and _evaluator_callback is None):
        raise ValueError("LAM fit requires variables and observations")
    names = [item["name"] for item in variables]
    if len(names) != len(set(names)):
        raise ValueError("LAM fit variable names must be unique")
    paths = [item["path"] for item in variables]
    if len(paths) != len(set(paths)):
        raise ValueError("each LAM fit path may be controlled only once")
    predicates = _lam_predicates(request.get("predicates", ()), variables)
    j_values = tuple(
        sorted({selector.J for item in observations for selector in (item["lower"], item["upper"])})
    )
    lam_states = int(request.get("lam_states", 12))
    levels_per_j = request.get("levels_per_j")
    options = request.get("options", {})
    if not isinstance(options, Mapping):
        raise ValueError("LAM fit options must be an object")
    loss = str(options.get("loss", "linear"))
    convergence_options = options.get("convergence", {})
    if not isinstance(convergence_options, Mapping):
        raise ValueError("LAM convergence options must be an object")
    convergence_mode = str(convergence_options.get("mode", "audit" if loss == "linear" else "solver"))
    if convergence_mode not in {"audit", "statistical", "solver"}:
        raise ValueError("LAM convergence mode must be audit, statistical or solver")
    if loss != "linear" and convergence_mode != "solver":
        raise ValueError("whitened statistical convergence requires linear loss")
    convergence_history = []
    solver_trace = []
    solver_state = None
    assembly_backend = options.get("assembly_backend", "compiled")
    diagonalizer_backend = options.get("diagonalizer_backend")
    primitive_basis = options.get("primitive_basis", "auto")
    fourier_cutoffs = options.get("fourier_cutoffs")
    k_contraction_states = options.get("k_contraction_states")
    contraction_mode = options.get("lam_contraction", "moving")
    if contraction_mode not in {"moving", "fixed"}:
        raise ValueError("lam_contraction must be moving or fixed")
    fixed_variational_span = contraction_mode == "fixed"
    if fixed_variational_span:
        # A fixed Galerkin space has no basis response. The complete current
        # operator is reprojected; potential/kinetic parameters still vary.
        options = dict(options, basis_response_states=0)
    fixed_lam_basis = fixed_variational_span or not any(
        item["path"].startswith(("potential.", "F.", "lam_p4."))
        for item in variables
    )
    workers = int(options.get("workers", 1))
    if workers < 1:
        raise ValueError("LAM fit workers must be at least one")
    jacobian_workers = int(options.get("jacobian_workers", workers))
    if jacobian_workers < 1:
        raise ValueError("LAM fit Jacobian workers must be at least one")
    block_workers = int(options.get("block_workers", 1 if jacobian_workers > 1 else workers))
    if block_workers < 1:
        raise ValueError("LAM fit block workers must be at least one")
    jacobian_scheme = str(options.get("jacobian_scheme", "forward"))
    if jacobian_scheme not in {"analytic", "hybrid", "forward", "central"}:
        raise ValueError(
            "LAM fit Jacobian scheme must be analytic, hybrid, forward, or central"
        )
    tracking = str(options.get("tracking", "overlap"))
    if tracking not in {"labels", "overlap"}:
        raise ValueError("LAM tracking must be labels or overlap")
    x0 = np.asarray([item["initial"] for item in variables], dtype=float)
    from .lam_checkpoint import lam_restart_contract,read_lam_workspace,write_lam_workspace
    workspace_contract=(lam_restart_contract(surface,request)
                        if options.get("workspace_checkpoint_path") or options.get("resume_workspace") else None)
    resumed=options.get('_workspace_resume_state')
    if options.get('resume_workspace') is not None:
        resumed=read_lam_workspace(options['resume_workspace'],contract=workspace_contract)
    if resumed is not None:
        x0=np.asarray(resumed['values'],float)
        if options.get('optimizer_state') is None and resumed.get('optimizer_state') is not None:
            options=dict(options,optimizer_state=resumed['optimizer_state'])
    k_reference_values = np.asarray(options.get("k_contraction_reference_values", x0), dtype=float)
    if k_reference_values.shape != x0.shape or not np.all(np.isfinite(k_reference_values)):
        raise ValueError("k_contraction_reference_values must contain one finite value per fit variable")
    if "k_contraction_reference_values" in options and (k_contraction_states is None or not fixed_lam_basis):
        raise ValueError("a frozen J,K reference requires k_contraction_states and a fixed LAM core")
    scales = np.asarray([item["scale"] for item in variables], dtype=float)
    lower = np.asarray([item["lower"] for item in variables], dtype=float)
    upper = np.asarray([item["upper"] for item in variables], dtype=float)
    observed = np.asarray([item["frequency_MHz"] for item in observations])
    sigma = np.asarray(
        [item["uncertainty_MHz"] / math.sqrt(item["weight"]) for item in observations]
    )
    training = np.asarray([item["role"] == "training" for item in observations], dtype=bool)
    if not np.any(training):
        raise ValueError("LAM fit requires at least one training observation")
    prior_indices = np.asarray(
        [index for index, item in enumerate(variables) if item["prior"] is not None],
        dtype=int,
    )
    prior_values = np.asarray(
        [variables[index]["prior"]["value"] for index in prior_indices], dtype=float
    )
    prior_sigmas = np.asarray(
        [variables[index]["prior"]["sigma"] for index in prior_indices], dtype=float
    )
    predicate_indices = np.asarray([item["index"] for item in predicates], dtype=int)
    predicate_values = np.asarray([item["value"] for item in predicates], dtype=float)
    predicate_sigmas = np.asarray([item["sigma"] for item in predicates], dtype=float)
    from matrix_numerics.least_squares import CovarianceWhitening
    def whitener(sigmas, covariance, indices=None, weights=None):
        if covariance is not None:
            covariance=np.asarray(covariance,dtype=float)
            if weights is not None:
                covariance=covariance/np.sqrt(weights[:,None]*weights[None,:])
            if indices is not None:covariance=covariance[np.ix_(indices,indices)]
        return CovarianceWhitening(sigmas,covariance)
    observation_covariance=request.get("covariance_MHz2")
    if observation_covariance is not None:
        # Validate the complete input before taking training/validation marginals.
        CovarianceWhitening(sigma, np.asarray(observation_covariance)/np.sqrt(
            np.asarray([o["weight"] for o in observations])[:,None]*np.asarray([o["weight"] for o in observations])[None,:]))
    fit_whitener=whitener(sigma[training],observation_covariance,np.flatnonzero(training),
                          np.asarray([o["weight"] for o in observations]))
    prior_whitener=whitener(prior_sigmas,request.get("prior_covariance"))
    predicate_whitener=whitener(predicate_sigmas,request.get("predicate_covariance"))
    cache: dict[
        bytes,
        tuple[
            np.ndarray,
            LAMRotationSurface,
            LAMRotationResult,
            Mapping[int, Mapping[int, int]],
        ],
    ] = {}
    solve_cache: dict[bytes, tuple[LAMRotationSurface, LAMRotationResult]] = {}
    jacobian_cache: dict[bytes, np.ndarray] = {}
    tracking_diagnostics: dict[bytes, Mapping[str, float]] = {}
    cache_size = int(options.get("cache_size", 3))
    if cache_size < 1:
        raise ValueError("LAM fit cache_size must be positive")
    checkpoint_path = options.get("checkpoint_path")
    checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
    best_cost = math.inf
    evaluation_history = []
    prediction_count = 0
    solve_count = 1
    jacobian_count = 0

    reference_surface = _apply_lam_variables(surface, variables, x0)
    reference_operator_parameters = _operator_parameter_values(
        normalized_operator_manifest, variables, x0
    )
    if resumed is not None:
        reference_workspace=resumed['workspace'];reference=resumed['reference'];frozen_reference=None
        observations=tuple(resumed['observations']);solve_count=0
    else:
        reference_workspace = None
        if not fixed_variational_span and all(item["path"].startswith("operator.") for item in variables):
            from .lam_rotation import prepare_lam_rotation_workspace

            reference_workspace = prepare_lam_rotation_workspace(
                reference_surface, lam_state_count=lam_states,
                operator_manifest=normalized_operator_manifest,
                diagonalizer_backend=diagonalizer_backend,
                primitive_basis=primitive_basis, fourier_cutoffs=fourier_cutoffs,
            )
        frozen_reference = None
        if k_contraction_states is not None and fixed_lam_basis and not np.array_equal(k_reference_values, x0):
            # Reconstruct exactly the frozen reference used by an earlier run.
            # Updating the starting point must not change the fitted subspace.
            frozen_reference = solve_lam_rotation(
                _apply_lam_variables(surface, variables, k_reference_values),
                j_values=j_values, lam_state_count=lam_states, level_count=levels_per_j,
                lam_basis=None if reference_workspace is None else reference_workspace.basis,
                assembly_backend=assembly_backend,
                diagonalizer_backend=diagonalizer_backend,
                operator_manifest=normalized_operator_manifest,
                operator_parameters_MHz=_operator_parameter_values(normalized_operator_manifest, variables, k_reference_values),
                primitive_basis=primitive_basis, fourier_cutoffs=fourier_cutoffs,
                k_contraction_states=k_contraction_states, workers=workers,
            )
            solve_count += 1
        reference = solve_lam_rotation(
            reference_surface, j_values=j_values, lam_state_count=lam_states,
            level_count=levels_per_j, workspace=reference_workspace,
            lam_basis=None if frozen_reference is None else frozen_reference.basis,
            assembly_backend=assembly_backend, diagonalizer_backend=diagonalizer_backend,
            operator_manifest=normalized_operator_manifest,
            operator_parameters_MHz=reference_operator_parameters,
            primitive_basis=primitive_basis, fourier_cutoffs=fourier_cutoffs,
            k_contraction_states=k_contraction_states,
            k_contractions=None if frozen_reference is None else frozen_reference.k_contractions,
            workers=workers,
        )
    flattened = tuple(
        selector for item in observations for selector in (item["lower"], item["upper"])
    )
    resolved, state_assignments = assign_lam_reference_states(
        reference,
        flattened,
        label_prior_MHz=float(options.get("assignment_label_prior_MHz", 20.0)),
        label_degeneracy_MHz=float(options.get("assignment_label_degeneracy_MHz", 5.0)),
    )
    if tracking == "overlap":
        # Bind every public selector to the initial eigensystem once. Symbolic
        # labels identify input states; continuation follows their eigenvectors.
        resolved = tuple(replace(selector, reference_index=_lam_level_index(selector, reference, {}))
                         for selector in resolved)
    resolved_iterator = iter(resolved)
    observations = tuple(
        {
            **item,
            "lower": next(resolved_iterator),
            "upper": next(resolved_iterator),
        }
        for item in observations
    )
    continuation_values = x0.copy()
    accepted_mapping = {J:{i:i for i in range(len(block.levels))} for J,block in reference.blocks.items()}
    required_original = {J:{selector.reference_index for item in observations
        for selector in (item["lower"],item["upper"]) if selector.J==J} for J in j_values}
    solve_cache[np.asarray(x0, dtype=np.float64).tobytes()] = (reference_surface, reference)
    def solve_at(x, *, retain=True):
        nonlocal solve_count
        key = np.asarray(x, dtype=np.float64).tobytes()
        if key in solve_cache:
            return solve_cache[key]
        fitted_surface = _apply_lam_variables(surface, variables, x)
        fitted_operator_parameters = _operator_parameter_values(
            normalized_operator_manifest, variables, x
        )
        try:
            solved = solve_lam_rotation(
                fitted_surface,
                j_values=j_values,
                lam_state_count=lam_states,
                level_count=levels_per_j,
                assembly_backend=assembly_backend,
                diagonalizer_backend=diagonalizer_backend,
                operator_manifest=normalized_operator_manifest,
                operator_parameters_MHz=fitted_operator_parameters,
                lam_basis=reference.basis if fixed_lam_basis else None,
                workspace=reference_workspace,
                primitive_basis=primitive_basis, fourier_cutoffs=fourier_cutoffs,
                k_contraction_states=k_contraction_states,
                k_contractions=reference.k_contractions if fixed_lam_basis else None,
                workers=block_workers,
                reproject_lam_core=fixed_variational_span,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"LAM solve failed at parameters {np.asarray(x).tolist()}: {exc}"
            ) from exc
        result = (fitted_surface, solved)
        solve_count += 1
        if retain:
            solve_cache[key] = result
            while len(solve_cache) > cache_size:
                oldest = next(iter(solve_cache))
                solve_cache.pop(oldest)
        return result

    def continuation_mapping(x, solved):
        if tracking != "overlap":
            return {}, {"minimum_principal_overlap": 1.0, "minimum_state_overlap": 1.0,
                        "continuation_steps": 0}
        delta = np.asarray(x, dtype=float) - continuation_values
        minimum_required = float(options.get("tracking_minimum_principal_overlap", 0.8))
        if not 0.0 <= minimum_required <= 1.0:
            raise ValueError("tracking_minimum_principal_overlap must lie in [0,1]")
        maximum_steps = int(options.get("tracking_max_steps", 64))
        if maximum_steps < 1:
            raise ValueError("tracking_max_steps must be positive")
        points = [(0.0, reference)]
        if np.array_equal(np.asarray(x, dtype=float), continuation_values):
            points.append((1.0, reference))
        else:
            points.append((1.0, solved))
        def evaluate_segments():
            selected = {J:{accepted_mapping[J][i] for i in required_original[J]}
                        for J in j_values}
            segments = []
            for (_,left),(_,right) in zip(points[:-1],points[1:],strict=True):
                local,diagnostic = track_lam_rotation_states(
                    left,right,diagnostic_indices=selected,
                    degeneracy_tolerance=float(options.get("tracking_degeneracy_tolerance",1.e-8)),
                    subspace_gap_MHz=options.get("tracking_subspace_gap_MHz"))
                segments.append((local,diagnostic))
                selected={J:{local[J][i] for i in selected[J]} for J in j_values}
            return segments

        segments = evaluate_segments()
        while len(points) - 1 < maximum_steps:
            worst, edge = min(
                (item[1]["minimum_principal_overlap"], index)
                for index, item in enumerate(segments))
            if worst >= minimum_required:
                break
            fraction = 0.5 * (points[edge][0] + points[edge + 1][0])
            _middle_surface,middle = solve_at(continuation_values + fraction*delta,retain=False)
            points.insert(edge+1,(fraction,middle))
            segments = evaluate_segments()
        composed = {J:dict(mapping) for J,mapping in accepted_mapping.items()}
        minimum_principal, minimum_state = 1.0, 1.0
        for local, diagnostic in segments:
            composed = {
                J: {origin: local[J][current_index]
                    for origin, current_index in composed[J].items()
                    if current_index in local[J]}
                for J in j_values
            }
            minimum_principal = min(
                minimum_principal, diagnostic["minimum_principal_overlap"]
            )
            minimum_state = min(minimum_state, diagnostic["minimum_state_overlap"])
        return composed, {
            "minimum_principal_overlap": float(minimum_principal),
            "minimum_state_overlap": float(minimum_state),
            "continuation_steps": len(points) - 1,
        }

    analysis_mode = False

    def predict(x):
        nonlocal best_cost, prediction_count
        key = np.asarray(x, dtype=np.float64).tobytes()
        if key in cache:
            return cache[key]
        fitted_surface, solved = solve_at(x)
        mappings, diagnostic = continuation_mapping(x, solved)
        values = np.asarray(
            [
                _lam_level_energy(item["upper"], solved, mappings)
                - _lam_level_energy(item["lower"], solved, mappings)
                for item in observations
            ]
        )
        cache[key] = (values, fitted_surface, solved, mappings)
        tracking_diagnostics[key] = diagnostic
        prediction_count += 1
        while len(cache) > cache_size:
            cache.pop(next(iter(cache)))
        cost = float(np.sum(fit_whitener.apply(values[training] - observed[training]) ** 2))
        if len(prior_indices):
            cost += float(np.sum(prior_whitener.apply(np.asarray(x)[prior_indices] - prior_values) ** 2))
        if len(predicate_indices):
            cost += float(np.sum(predicate_whitener.apply(np.asarray(x)[predicate_indices] - predicate_values) ** 2))
        if cost < best_cost and not analysis_mode:
            best_cost = cost
            record = {
                "evaluation": prediction_count,
                "chi_square": cost,
                "training_rms_MHz": float(np.sqrt(np.mean((values[training] - observed[training]) ** 2))),
                "validation_rms_MHz": (
                    float(np.sqrt(np.mean((values[~training] - observed[~training]) ** 2)))
                    if np.any(~training) else None
                ),
                "values": np.asarray(x, dtype=float).tolist(),
            }
            evaluation_history.append(record)
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
                temporary.write_text(json.dumps({
                    "schema": "matrix.lock.lam_spectral_fit_checkpoint.v1",
                    "lam_states": lam_states,
                    "primitive_basis": primitive_basis,
                    "fourier_cutoffs": fourier_cutoffs,
                    "k_contraction_states": k_contraction_states,
                    "k_contraction_reference_values": k_reference_values.tolist() if fixed_lam_basis and k_contraction_states is not None else None,
                    "grid_shape": list(surface.grid_shape),
                    "variables": [
                        {"name": variable["name"], "path": variable["path"], "value": float(value)}
                        for variable, value in zip(variables, x, strict=True)
                    ],
                    "state_assignments": list(state_assignments),
                    "observations": [
                        {
                            **{name: value for name, value in item.items() if name not in {"lower", "upper"}},
                            **{
                                side: {
                                    **asdict(item[side]),
                                    "reference_index": _lam_level_index(item[side], solved, mappings),
                                }
                                for side in ("lower", "upper")
                            },
                        }
                        for item in observations
                    ],
                    "operator_manifest_digest": solved.operator_manifest_digest,
                    "tracking_diagnostics": diagnostic,
                    "history": evaluation_history,
                }, indent=2) + "\n", encoding="utf-8")
                temporary.replace(checkpoint_path)
        return cache[key]

    def residual(x):
        data = fit_whitener.apply((predict(x)[0] - observed)[training])
        rows = [data]
        if len(prior_indices):
            rows.append(prior_whitener.apply(np.asarray(x)[prior_indices] - prior_values))
        if len(predicate_indices):
            rows.append(predicate_whitener.apply(np.asarray(x)[predicate_indices] - predicate_values))
        return np.concatenate(rows)

    def analytic_jacobian(x):
        nonlocal jacobian_count
        key = np.asarray(x, dtype=np.float64).tobytes()
        if key in jacobian_cache:
            return jacobian_cache[key]
        _values, fitted_surface, solved, mappings = predict(x)
        data = (
            _lam_operator_response_jacobian(
                fitted_surface,
                solved,
                variables,
                np.asarray(x, dtype=float),
                observations,
                mappings,
                operator_manifest=normalized_operator_manifest,
                operator_parameters_MHz=_operator_parameter_values(
                    normalized_operator_manifest, variables, x
                ),
                operator_contraction=(None if reference_workspace is None else reference_workspace.operator_contraction),
                moving_k_contraction=not fixed_lam_basis,
                response_states=int(options.get("basis_response_states", 48)),
                workers=jacobian_workers,
                parameter_reference_surface=surface,
            )
        )
        data = fit_whitener.apply(data[training])
        rows = [data]
        if len(prior_indices):
            priors = np.zeros((len(prior_indices), len(variables)))
            priors[np.arange(len(prior_indices)), prior_indices] = 1.0
            rows.append(prior_whitener.apply(priors))
        if len(predicate_indices):
            predicate_jacobian = np.zeros((len(predicate_indices), len(variables)))
            predicate_jacobian[np.arange(len(predicate_indices)), predicate_indices] = (
                1.0
            )
            rows.append(predicate_whitener.apply(predicate_jacobian))
        result = np.vstack(rows)
        jacobian_count += 1
        jacobian_cache[key] = result
        while len(jacobian_cache) > cache_size:
            jacobian_cache.pop(next(iter(jacobian_cache)))
        return result

    def hybrid_jacobian(x):
        nonlocal jacobian_count
        key = np.asarray(x, dtype=np.float64).tobytes()
        if key in jacobian_cache:
            return jacobian_cache[key]
        affine = tuple(
            index
            for index, item in enumerate(variables)
            if item["path"].startswith("R.offset.")
        )
        nonlinear = tuple(index for index in range(len(variables)) if index not in affine)
        result = np.empty((len(residual(x)), len(variables)))
        if nonlinear:
            finite = _finite_jacobian(
                residual,
                np.asarray(x, dtype=float),
                scales,
                lower,
                upper,
                workers=jacobian_workers,
                scheme="forward",
                columns=nonlinear,
            )
            result[:, nonlinear] = finite[:, nonlinear]
        if affine:
            _values, fitted_surface, solved, mappings = predict(x)
            columns = _lam_operator_response_jacobian(
                fitted_surface, solved, tuple(variables[i] for i in affine), np.asarray(x)[list(affine)],
                observations, mappings, operator_manifest=normalized_operator_manifest,
                operator_parameters_MHz=_operator_parameter_values(normalized_operator_manifest, variables, x),
                response_states=0, workers=jacobian_workers, moving_k_contraction=not fixed_lam_basis,
                parameter_reference_surface=surface,
            )
            ndata=int(np.count_nonzero(training))
            result[:ndata,affine]=fit_whitener.apply(columns[training])
            for indices, whitening, offset in ((prior_indices,prior_whitener,ndata),
                    (predicate_indices,predicate_whitener,ndata+len(prior_indices))):
                penalties=np.asarray([[float(i==column) for column in affine] for i in indices]).reshape(len(indices),len(affine))
                result[offset:offset+len(indices),affine]=whitening.apply(penalties)
        jacobian_count += 1
        jacobian_cache[key] = result
        while len(jacobian_cache) > cache_size:
            jacobian_cache.pop(next(iter(jacobian_cache)))
        return result

    jacobian_function = (
        analytic_jacobian
        if jacobian_scheme == "analytic"
        else hybrid_jacobian
        if jacobian_scheme == "hybrid"
        else lambda x: _finite_jacobian(
            residual,
            x,
            scales,
            lower,
            upper,
            workers=jacobian_workers,
            scheme=jacobian_scheme,
        )
    )

    def promote_accepted(x):
        nonlocal reference,continuation_values,accepted_mapping
        values=np.asarray(x,dtype=float)
        if np.array_equal(values,continuation_values):
            return False
        key=np.asarray(values,dtype=np.float64).tobytes()
        prediction=predict(values)
        reference=prediction[2]
        accepted_mapping={J:dict(mapping) for J,mapping in prediction[3].items()}
        continuation_values=values.copy()
        # A rejected point must never become the continuation reference.
        # Only accepted-point caches remain valid for the new local path.
        for storage in (cache,jacobian_cache,tracking_diagnostics):
            for old in list(storage):
                if old!=key:storage.pop(old)
        return True

    def snapshot_workspace(x,optimizer_state=None):
        _,_,current,mapping=predict(x)
        from dataclasses import replace
        resolved=tuple(dict(item,**{side:replace(item[side],reference_index=_lam_level_index(item[side],current,mapping))
            for side in ('lower','upper')}) for item in observations)
        return dict(values=np.asarray(x).copy(),workspace=reference_workspace,reference=current,
                    observations=resolved,optimizer_state=optimizer_state)

    if _evaluator_callback is not None:
        analysis_mode=True
        return _evaluator_callback(residual,jacobian_function,x0,lower,upper,scales,
            predict=predict,variables=variables,observations=observations,parameter_reference_surface=surface,
            operator_manifest=normalized_operator_manifest,options=options,
            moving_k_contraction=not fixed_lam_basis,snapshot_workspace=snapshot_workspace,
            promote_accepted=promote_accepted)

    method = str(options.get("method", "trf"))
    if method not in {"trf", "dogbox", "gauss_newton", "trust_region", "qm_subspace"}:
        raise ValueError("LAM fit method must be trf, dogbox, gauss_newton, trust_region or qm_subspace")
    if method in {"gauss_newton", "trust_region", "qm_subspace"} and (loss != "linear" or convergence_mode != "statistical"):
        raise ValueError("statistical optimizers require linear loss and statistical convergence")
    def statistical_assessment(x, r, j):
        return assess_statistical_convergence(
            r, j, x, lower, upper, data_count=int(np.count_nonzero(training)),
            joint_tolerance=float(convergence_options.get("joint_tolerance", 0.1)),
            line_tolerance=float(convergence_options.get("line_tolerance", 0.01)),
            rank_rcond=float(options.get("rank_rcond", 1.0e-10)),
            data_correction_transform=fit_whitener.standardized_transform,
        )

    class StatisticalStop(Exception):
        def __init__(self, x, r, j):
            self.x, self.r, self.j = np.asarray(x).copy(), r, j

    def checked_jacobian(x):
        j = jacobian_function(x)
        if convergence_mode == "statistical":
            r = residual(x)
            diagnostic = statistical_assessment(x, r, j)
            convergence_history.append({
                "assessment": len(convergence_history) + 1,
                "chi_square": float(r @ r),
                "joint_correction_norm": diagnostic["joint_correction_norm"],
                "maximum_data_correction_sigma": diagnostic["maximum_data_correction_sigma"],
                "classification": diagnostic["classification"],
            })
            if diagnostic["converged"]:
                raise StatisticalStop(x, r, j)
        return j

    resume_state = options.get("optimizer_state")
    state_contract = dict(parameter_paths=paths, scales=scales.tolist(),
                          lower=[None if not np.isfinite(z) else float(z) for z in lower],
                          upper=[None if not np.isfinite(z) else float(z) for z in upper])
    if resume_state is not None:
        if (not isinstance(resume_state, Mapping) or method != "trust_region"
                or any(k not in resume_state for k in ("trust_radius", "damping"))
                or resume_state.get("schema") != "matrix.lock.lm_optimizer_state.v1"
                or resume_state.get("method") != method
                or any(resume_state.get(k) != v for k, v in state_contract.items())
                or resume_state.get("values") != x0.tolist()):
            raise ValueError("checkpoint optimizer state differs from fit values, bounds or parameter metric")

    def record_solver_state(state):
        nonlocal solver_state
        promote_accepted(state['values'])
        solver_state = dict(state, **{k: v for k, v in state_contract.items() if k != "scales"},
                            schema="matrix.lock.lm_optimizer_state.v1", method="trust_region")
        if options.get('workspace_checkpoint_path'):
            write_lam_workspace(options['workspace_checkpoint_path'],snapshot_workspace(state['values'],solver_state),contract=workspace_contract)
        if checkpoint_path is not None and checkpoint_path.exists():
            payload = json.loads(checkpoint_path.read_text())
            # A checkpoint records the best evaluated point, which can differ
            # from the accepted point after a marginally improving rejected trial.
            if [v["value"] for v in payload["variables"]] == state["values"]:
                payload["optimizer_state"] = solver_state
                temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
                temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                temporary.replace(checkpoint_path)

    def record_solver_trace(history):
        if options.get("iteration_trace_path") is not None:
            destination = Path(options["iteration_trace_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
            temporary.replace(destination)

    def record_qm_accepted(values):
        promote_accepted(values)
        if options.get('workspace_checkpoint_path'):
            write_lam_workspace(options['workspace_checkpoint_path'],
                snapshot_workspace(values), contract=workspace_contract)

    try:
        if method == "trust_region":
            optimized = statistical_trust_region(
                residual, checked_jacobian, x0, lower, upper, scales,
                max_nfev=int(options.get("max_nfev", 300)),
                initial_radius=float(resume_state["trust_radius"] if resume_state is not None else options.get("initial_trust_radius", 1.0)),
                initial_damping=float(resume_state["damping"] if resume_state is not None else 0.),
                trace=solver_trace, trace_callback=record_solver_trace, state_callback=record_solver_state,
            )
        elif method == "qm_subspace":
            from .qm_subspace_fit import fit_identifiable_qm_subspaces
            optimized = fit_identifiable_qm_subspaces(
                residual, jacobian_function, x0, lower, upper, scales,
                data_count=int(np.count_nonzero(training)),
                max_nfev=int(options.get("max_nfev", 300)),
                rank_rcond=float(options.get("rank_rcond", 1.e-10)),
                minimum_singular_value=float(options.get("minimum_experimental_singular_value", 1.)),
                joint_tolerance=float(convergence_options.get("joint_tolerance", .1)),
                line_tolerance=float(convergence_options.get("line_tolerance", .01)),
                data_correction_transform=fit_whitener.standardized_transform,
                accepted_callback=record_qm_accepted, parameter_names=names,
            )
        elif method == "gauss_newton":
            optimized = bounded_gauss_newton(
                residual, checked_jacobian, x0, lower, upper,
                data_count=int(np.count_nonzero(training)),
                max_nfev=int(options.get("max_nfev", 300)),
                rank_rcond=float(options.get("rank_rcond", 1.0e-10)),
            )
        else:
            optimized = least_squares(
                residual, x0, bounds=(lower, upper), method=method,
                jac=checked_jacobian,
                x_scale="jac" if options.get("x_scale") == "jac" else scales,
                loss=loss, max_nfev=int(options.get("max_nfev", 300)),
                ftol=float(options.get("ftol", 1.0e-8)),
                xtol=float(options.get("xtol", 1.0e-8)),
                gtol=float(options.get("gtol", 1.0e-8)),
                verbose=int(options.get("verbose", 0)),
            )
    except StatisticalStop as stopped:
        optimized = OptimizeResult(
            x=stopped.x, success=True, status=5,
            message="Statistical stationarity satisfied in whitened observable space.",
            nfev=None, njev=len(convergence_history), cost=0.5 * float(stopped.r @ stopped.r),
            optimality=float(np.linalg.norm(stopped.j.T @ stopped.r, ord=np.inf)),
        )
    values, fitted_surface, solved, _mappings = predict(optimized.x)
    jacobian = jacobian_function(optimized.x)
    final_residual = residual(optimized.x)
    convergence = (statistical_assessment(optimized.x, final_residual, jacobian)
                   if loss == "linear" else {
                       "criterion": "solver", "classification": "not_assessed",
                       "converged": None,
                       "scope": "statistical diagnostic is defined for linear loss only",
                   })
    convergence["mode"] = convergence_mode
    convergence["history"] = convergence_history
    accepted_success = bool(optimized.success and (
        convergence_mode == "solver" or convergence["converged"]))
    convergence["termination"] = (
        "converged" if accepted_success else "evaluation_limit"
        if optimized.status == 0 else "stalled"
        if optimized.success or optimized.status == -3 else "solver_failed"
    )
    linearization = None
    if options.get("linearization_path") is not None:
        import hashlib
        destination = Path(options["linearization_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, values=optimized.x, residual=final_residual,
                jacobian=jacobian, lower=lower, upper=upper, scales=scales,
                data_count=np.asarray(int(np.count_nonzero(training))),
                frequencies_MHz=values,
                **({"data_correction_transform": fit_whitener.standardized_transform}
                   if fit_whitener.standardized_transform is not None else {}),
            )
        temporary.replace(destination)
        linearization = {"path": str(destination),
                         "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                         "jacobian_shape": list(jacobian.shape),
                         "row_order": "training, parameter priors, QM predicates"}
    data_jacobian = jacobian[: int(np.count_nonzero(training))]
    scaled_jacobian = jacobian * scales[None, :]
    scaled_data_jacobian = data_jacobian * scales[None, :]
    _left, singular, right_h = np.linalg.svd(scaled_jacobian, full_matrices=False)
    _data_left, data_singular, data_right_h = np.linalg.svd(
        scaled_data_jacobian, full_matrices=False
    )
    rcond = float(options.get("rank_rcond", 1.0e-10))
    rank = int(np.count_nonzero(singular > (singular[0] * rcond if len(singular) else 0.0)))
    data_rank = int(
        np.count_nonzero(data_singular > (data_singular[0] * rcond if len(data_singular) else 0.0))
    )
    # Use the same scaled SVD as the rank diagnosis. Forming J.T @ J squares
    # the condition number and silently discards weak columns when physical
    # coefficients span MHz, barriers and high-order distortion constants.
    identifiable_modes = singular > (singular[0] * rcond if len(singular) else 0.0)
    retained_right = right_h[identifiable_modes]
    inverse_modes = retained_right.T / singular[identifiable_modes]
    covariance_scaled = inverse_modes @ inverse_modes.T
    covariance = scales[:, None] * covariance_scaled * scales[None, :]
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(errors, errors)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0.0,
    )
    raw = values - observed
    identifiable = rank == len(variables)
    payload = {
        "schema": LAM_SPECTRAL_FIT_RESULT_SCHEMA,
        "status": "converged"
        if accepted_success and identifiable
        else "rank_deficient"
        if optimized.success and (convergence_mode == "solver" or convergence.get("rank") != len(variables))
        else "failed",
        "success": bool(accepted_success and identifiable),
        "convergence": convergence,
        "linearization": linearization,
        "whitening": {"experimental_correlations": observation_covariance is not None,
                      "qm_correlations": request.get("predicate_covariance") is not None,
                      "split_policy": "training marginal; validation never conditions training",
                      "line_tolerance": "original marginal sigma units"},
        "tracking": tracking,
        "numerical_basis": {
            "lam_contraction": contraction_mode,
            "primitive_representation": reference.basis.primitive_representation,
            "primitive_size": reference.basis.primitive_size,
            "fourier_cutoffs": None if reference.basis.fourier_cutoffs is None else list(reference.basis.fourier_cutoffs),
            "k_contraction_states": k_contraction_states,
            "k_contraction_frozen": bool(fixed_lam_basis and k_contraction_states is not None),
            "k_contraction_reference_values": k_reference_values.tolist() if fixed_lam_basis and k_contraction_states is not None else None,
            "k_contractions": None if reference.k_contractions is None else
                {str(J): value.diagnostics for J, value in reference.k_contractions.items()},
        },
        "tracking_diagnostics": tracking_diagnostics.get(
            np.asarray(optimized.x, dtype=np.float64).tobytes(), {}
        ),
        "state_assignments": list(state_assignments),
        "optimizer": {
            "method": method,
            "iteration_trace": solver_trace,
            "state": solver_state,
            "status": int(optimized.status),
            "message": str(optimized.message),
            "nfev": None if optimized.nfev is None else int(optimized.nfev),
            "njev": None if optimized.njev is None else int(optimized.njev),
            "cost": float(optimized.cost),
            "optimality": float(optimized.optimality),
            "jacobian": (
                "analytic exact-affine operator response plus contracted-basis perturbation theory"
                if jacobian_scheme == "analytic"
                else "hybrid exact-affine response plus parallel forward differences"
                if jacobian_scheme == "hybrid"
                else f"{jacobian_scheme} finite difference"
            ),
        },
        "variables": [
            {
                "name": item["name"],
                "path": item["path"],
                "value": float(optimized.x[index]),
                "multiplier": item["multiplier"],
                "physical_coefficient": float(optimized.x[index] * item["multiplier"]),
                "standard_error": float(errors[index]) if identifiable else None,
                "prior": item["prior"],
            }
            for index, item in enumerate(variables)
        ],
        "comparisons": [
            {
                "label": item["label"],
                "role": item["role"],
                "observed_MHz": item["frequency_MHz"],
                "calculated_MHz": float(values[index]),
                "calculated_minus_observed_MHz": float(raw[index]),
                "standardized_residual": float(raw[index] / sigma[index]),
            }
            for index, item in enumerate(observations)
        ],
        "predicates": [
            {
                **{key: value for key, value in item.items() if key != "index"},
                "fitted_value": float(optimized.x[item["index"]]),
                "standardized_residual": float(
                    (optimized.x[item["index"]] - item["value"]) / item["sigma"]
                ),
            }
            for item in predicates
        ],
        "metrics": {
            "observation_count": len(observations),
            "training_count": int(np.count_nonzero(training)),
            "validation_count": int(np.count_nonzero(~training)),
            "free_parameter_count": len(variables),
            "rank": rank,
            "chi_square": float(np.sum(residual(optimized.x) ** 2)),
            "data_chi_square": float(np.sum(fit_whitener.apply(raw[training]) ** 2)),
            "prior_chi_square": float(
                np.sum(prior_whitener.apply(optimized.x[prior_indices] - prior_values) ** 2)
            ),
            "prior_count": int(len(prior_indices)),
            "predicate_count": int(len(predicate_indices)),
            "predicate_chi_square": float(
                np.sum(
                    predicate_whitener.apply(optimized.x[predicate_indices] - predicate_values) ** 2
                )
            ),
            "data_rank": data_rank,
            "augmented_rank": rank,
            "rms_residual_MHz": float(np.sqrt(np.mean(raw[training] * raw[training]))),
            "validation_rms_MHz": (
                float(np.sqrt(np.mean(raw[~training] * raw[~training])))
                if np.any(~training)
                else None
            ),
            "maximum_absolute_residual_MHz": float(np.max(np.abs(raw))),
            "hamiltonian_evaluations": prediction_count,
            "hamiltonian_solves": solve_count,
            "jacobian_evaluations": jacobian_count,
            "condition_number_scaled": (
                float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 0.0
                else None
            ),
        },
        "singular_values_scaled": singular.tolist(),
        "data_singular_values_scaled": data_singular.tolist(),
        "identifiability": _identifiability_report(
            variables, singular, right_h, data_singular, data_right_h, rcond
        ),
        "covariance": covariance.tolist(),
        "correlation": correlation.tolist(),
        "evaluation_history": evaluation_history,
        "assembly_backends": sorted(set(solved.assembly_backends)),
        "diagonalizer_backends": sorted(set(solved.diagonalizer_backends)),
        "operator_manifest_digest": solved.operator_manifest_digest,
        "operator_parameters_MHz": (
            None
            if solved.operator_parameters_MHz is None
            else dict(solved.operator_parameters_MHz)
        ),
        "parallelism": {
            "workers": workers,
            "jacobian_workers": jacobian_workers,
            "block_workers": block_workers,
            "jacobian_scheme": jacobian_scheme,
        },
    }
    if method == "qm_subspace":
        payload['optimizer']['identifiable_subspaces'] = optimized.subspace_report
    payload['component_models']=[dict((side,dict(J=item[side].J,reference_index=_lam_level_index(item[side],solved,_mappings)))
        for side in ('lower','upper')) for item in observations]
    if request.get('uncertainty_budget') is not None:
        from .prediction_budget import lam_prediction_budget
        sensitivity=_lam_operator_response_jacobian(fitted_surface,solved,variables,optimized.x,observations,_mappings,
            operator_manifest=normalized_operator_manifest,operator_parameters_MHz=solved.operator_parameters_MHz,
            response_states=int(options.get('basis_response_states',48)),workers=jacobian_workers,
            moving_k_contraction=not fixed_lam_basis,parameter_reference_surface=surface)
        payload['uncertainty_budget']=lam_prediction_budget(dict(request,observations=observations),payload,sensitivity,
            training_whitener=fit_whitener,training_jacobian=jacobian[:int(np.count_nonzero(training))])
    if postfit_diagnostic is not None:
        if not callable(postfit_diagnostic):
            raise ValueError("postfit_diagnostic must be callable")
        analysis_mode = True
        payload["postfit_analysis"] = postfit_diagnostic(
            residual, jacobian_function, optimized.x.copy(), lower.copy(), upper.copy(), scales.copy(),
            data_count=int(np.count_nonzero(training)), predicate_count=len(predicate_indices),
            parameter_names=names,
            **({"data_correction_transform": fit_whitener.standardized_transform}
               if fit_whitener.standardized_transform is not None else {}),
        )
    if options.get('kinetic_comparison') is not None:
        from .lam_checkpoint import copy_workspace_request
        from .kinetic_hierarchy import compare_lam_kinetic_models
        diagnostic_request = copy_workspace_request(dict(request,
            **({'operator_manifest': normalized_operator_manifest.to_dict()}
               if normalized_operator_manifest is not None else {})))
        diagnostic_request['variables'] = []
        diagnostic_request['predicates'] = []
        diagnostic_request.pop('prior_covariance', None)
        diagnostic_request.pop('predicate_covariance', None)
        diagnostic_request['observations'] = [
            dict(label=item['label'], frequency_MHz=item['frequency_MHz'],
                 uncertainty_MHz=item['uncertainty_MHz'],
                 **{side: dict(asdict(item[side]),
                     reference_index=_lam_level_index(item[side], solved, _mappings))
                    for side in ('lower', 'upper')}) for item in observations]
        diagnostic_request['options'].pop('k_contraction_reference_values', None)
        if normalized_operator_manifest is not None:
            # Materialize fitted operator coefficients for the zero-variable audit.
            from .general_lam_operator import normalize_general_lam_operator_manifest
            manifest_payload = normalized_operator_manifest.to_dict()
            manifest_payload['parameters_MHz'] = dict(solved.operator_parameters_MHz)
            normalize_general_lam_operator_manifest(manifest_payload)
            diagnostic_request['operator_manifest'] = manifest_payload
        payload['kinetic_comparison'] = compare_lam_kinetic_models(
            fitted_surface, diagnostic_request, **dict(options['kinetic_comparison']))
        payload['kinetic_comparison']['accepted_fit_frequencies_MHz'] = values.tolist()
        payload['kinetic_comparison']['maximum_full_minus_accepted_fit_MHz'] = float(np.max(
            np.abs(np.asarray(payload['kinetic_comparison']['full_frequencies_MHz']) - values), initial=0.))
        payload['kinetic_comparison']['full_reference'] = (
            'recontracted materialized surface; may differ from a frozen Galerkin fit span')
    if options.get('workspace_checkpoint_path'):
        write_lam_workspace(options['workspace_checkpoint_path'],snapshot_workspace(optimized.x,solver_state),contract=workspace_contract)
    payload['workspace_restart']=dict(reused=resumed is not None,reference_solves=solve_count)
    return LAMFitResult(payload, fitted_surface)


def _lam_variable(payload, surface, index, operator_defaults=None):
    if not isinstance(payload, Mapping):
        raise ValueError("each LAM variable must be an object")
    path = str(payload["path"])
    operator_defaults = operator_defaults or {}
    if path.startswith("operator."):
        parameter = path.removeprefix("operator.")
        if parameter not in operator_defaults:
            raise ValueError(f"unknown operator-manifest parameter: {parameter!r}")
    else:
        _validate_lam_path(path, surface)
    default_initial = (
        float(operator_defaults[path.removeprefix("operator.")])
        if path.startswith("operator.")
        else 1.0 if path.endswith(".scale") else 0.0
    )
    initial = float(payload.get("initial", default_initial))
    scale = float(payload.get("scale", max(abs(initial), 0.1)))
    multiplier = float(payload.get("multiplier", 1.0))
    lower = float(
        payload.get(
            "lower",
            1.0e-6 if path in {"F.scale", "R.scale"} or path.endswith(".scale") else -math.inf,
        )
    )
    upper = float(payload.get("upper", math.inf))
    if not lower <= initial <= upper or scale <= 0.0 or not math.isfinite(multiplier):
        raise ValueError("invalid LAM fit initial value, scale, or bounds")
    prior = payload.get("prior")
    if prior is not None:
        if not isinstance(prior, Mapping):
            raise ValueError("LAM fit priors must be objects")
        prior = {"value": float(prior["value"]), "sigma": float(prior["sigma"])}
        if not math.isfinite(prior["value"]) or prior["sigma"] <= 0.0:
            raise ValueError("LAM fit prior values must be finite and sigmas positive")
    return {
        "name": str(payload.get("name", f"p{index + 1}")),
        "path": path,
        "initial": initial,
        "scale": scale,
        "lower": lower,
        "upper": upper,
        "multiplier": multiplier,
        "prior": prior,
    }


def _lam_observation(payload, index):
    if not isinstance(payload, Mapping):
        raise ValueError("each LAM observation must be an object")
    sigma = float(payload["uncertainty_MHz"])
    weight = float(payload.get("weight", 1.0))
    if sigma <= 0.0 or weight <= 0.0:
        raise ValueError("LAM uncertainties and weights must be positive")
    role = str(payload.get("role", "training")).strip().casefold()
    if role not in {"training", "validation"}:
        raise ValueError("LAM observation role must be training or validation")
    return {
        "label": str(payload.get("label", f"line-{index + 1}")),
        "frequency_MHz": float(payload["frequency_MHz"]),
        "uncertainty_MHz": sigma,
        "weight": weight,
        "role": role,
        "lower": LAMLevelSelector.from_dict(payload["lower"]),
        "upper": LAMLevelSelector.from_dict(payload["upper"]),
    }


def _lam_predicates(payload, variables):
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError("LAM predicates must be an array")
    rows = []
    for predicate_index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise ValueError("LAM predicates must be objects")
        exact_path = raw.get("path")
        pattern = raw.get("path_pattern")
        if exact_path is None and pattern is None:
            raise ValueError("a LAM predicate requires path or path_pattern")
        matches = [
            index
            for index, item in enumerate(variables)
            if (
                item["path"] == str(exact_path)
                if exact_path is not None
                else fnmatchcase(item["path"], str(pattern))
                or fnmatchcase(item["name"], str(pattern))
            )
        ]
        if bool(raw.get("required", True)) and not matches:
            label = exact_path if exact_path is not None else pattern
            raise ValueError(f"LAM predicate {label!r} matches no fitted variable")
        value, sigma = float(raw["value"]), float(raw["sigma"])
        if not math.isfinite(value) or not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("LAM predicates require finite values and positive sigmas")
        for variable_index in matches:
            rows.append(
                {
                    "name": str(raw.get("name", f"predicate-{predicate_index + 1}")),
                    "path": variables[variable_index]["path"],
                    "path_pattern": None if pattern is None else str(pattern),
                    "value": value,
                    "sigma": sigma,
                    "source": str(raw.get("source", "qm")),
                    "index": variable_index,
                }
            )
    return tuple(rows)


def _validate_lam_path(path, surface):
    if path in {"potential.scale", "F.scale", "rho.scale", "R.scale"}:
        return
    if re_match := re.fullmatch(r"potential\.(cos|sin)\.(\d+)", path):
        if int(re_match.group(2)) < 1 or not surface.periodic[0]:
            raise ValueError("Fourier potential terms require a periodic first LAM coordinate")
        return
    if re_match := re.fullmatch(r"R\.(cos|sin|one_minus_cos)\.(\d+)\.(\d+)\.(\d+)", path):
        order, left, right = map(int, re_match.groups()[1:])
        if order < 1 or not surface.periodic[0] or left >= 3 or right >= 3:
            raise ValueError(
                "Fourier rotational terms require valid axes and a periodic coordinate"
            )
        return
    if re_match := re.fullmatch(r"R\.(cos|sin|one_minus_cos)\.(\d+)\.(isotropic|b_minus_c)", path):
        if int(re_match.group(2)) < 1 or not surface.periodic[0]:
            raise ValueError("Fourier rotational terms require a periodic first coordinate")
        return
    if re_match := re.fullmatch(r"R\.offset\.(\d+)\.(\d+)", path):
        if int(re_match.group(1)) >= 3 or int(re_match.group(2)) >= 3:
            raise ValueError("rotational offset axes must be in a,b,c")
        return
    if match := re.fullmatch(
        r"(potential|F|rho|R)\.fourier\.([a-z_,]+)\.([0-9,]+)(?:\.(\d+)\.(\d+))?",
        path,
    ):
        family, functions, harmonics, left, right = match.groups()
        _multi_lam_fourier_field(surface, functions, harmonics)
        if family == "potential":
            if left is not None or right is not None:
                raise ValueError("potential Fourier paths do not take tensor indices")
        else:
            if left is None or right is None:
                raise ValueError(f"{family} Fourier paths require two tensor indices")
            array = {"F": surface.F_MHz, "rho": surface.rho, "R": surface.R_MHz}[family]
            if int(left) >= array.shape[-2] or int(right) >= array.shape[-1]:
                raise ValueError(f"LAM component index is outside the {family} tensor")
        return
    if match := re.fullmatch(r"spin\.(.+)\.scale", path):
        if match.group(1) not in (surface.spin_tensor_surfaces_MHz or {}):
            raise ValueError(f"LAM fit path {path!r} requires a matching spin surface")
        return
    if match := re.fullmatch(r"pi2\.(J|K|minus)\.scale", path):
        if surface.pi2_rotation_MHz is None:
            raise ValueError(f"LAM fit path {path!r} requires pi2_rotation_MHz")
        return
    if re.fullmatch(r"pi2\.(J|K|minus)\.MHz", path):
        return
    optional_paths = (
        (r"lam_p4\.(\d+)\.scale", surface.lam_quartic_kinetic_MHz, (surface.dimension,)),
        (
            r"p2R\.(\d+)\.(\d+)\.(\d+)\.scale",
            surface.p2_rotational_metric_MHz,
            (surface.dimension, 3, 3),
        ),
        (
            r"pJ3\.(\d+)\.(\d+)\.scale",
            surface.p_rotational_cubic_MHz,
            (surface.dimension, 3),
        ),
        (
            r"pJaa\.(\d+)\.(\d+)\.(\d+)\.scale",
            surface.p_rotational_anticommutator_MHz,
            (surface.dimension, 3, 3),
        ),
        (r"quartic\.(\d+)\.scale", surface.quartic_distortion_MHz, (5,)),
        (r"sextic\.(\d+)\.scale", surface.sextic_distortion_MHz, (7,)),
    )
    for pattern, array, limits in optional_paths:
        if match := re.fullmatch(pattern, path):
            indices = tuple(map(int, match.groups()))
            if array is None or any(
                index >= limit for index, limit in zip(indices, limits, strict=True)
            ):
                raise ValueError(f"LAM fit path {path!r} requires a matching surface field")
            return
    if match := re.fullmatch(r"p2R\.(\d+)\.isotropic\.scale", path):
        if surface.p2_rotational_metric_MHz is None or int(match.group(1)) >= surface.dimension:
            raise ValueError(f"LAM fit path {path!r} requires p2_rotational_metric_MHz")
        return
    match = re.fullmatch(r"(F|R|rho)\.(\d+)\.(\d+)\.scale", path)
    if match:
        family, left, right = match.group(1), int(match.group(2)), int(match.group(3))
        array = {"F": surface.F_MHz, "R": surface.R_MHz, "rho": surface.rho}[family]
        if left >= array.shape[-2] or right >= array.shape[-1]:
            raise ValueError(f"LAM component index is outside the {family} tensor")
        return
    raise ValueError(f"unsupported LAM fit path: {path}")


def _operator_parameter_values(manifest, variables, values):
    if manifest is None:
        return None
    overrides = {}
    for variable, raw in zip(variables, values, strict=True):
        path = variable["path"]
        if path.startswith("operator."):
            overrides[path.removeprefix("operator.")] = (
                float(raw) * variable["multiplier"]
            )
    return manifest.parameters_with(overrides)


def _apply_lam_variables(surface, variables, values):
    if all(variable["path"].startswith("operator.") for variable in variables):
        return surface
    potential = np.array(surface.potential_MHz, copy=True)
    F, rho, R = (
        np.array(surface.F_MHz, copy=True),
        np.array(surface.rho, copy=True),
        np.array(surface.R_MHz, copy=True),
    )
    lam_p4 = (
        None
        if surface.lam_quartic_kinetic_MHz is None
        else np.array(surface.lam_quartic_kinetic_MHz, copy=True)
    )
    p2R = (
        None
        if surface.p2_rotational_metric_MHz is None
        else np.array(surface.p2_rotational_metric_MHz, copy=True)
    )
    pJ3 = (
        None
        if surface.p_rotational_cubic_MHz is None
        else np.array(surface.p_rotational_cubic_MHz, copy=True)
    )
    pJaa = (
        None
        if surface.p_rotational_anticommutator_MHz is None
        else np.array(surface.p_rotational_anticommutator_MHz, copy=True)
    )
    pi2 = (
        None if surface.pi2_rotation_MHz is None
        else np.array(surface.pi2_rotation_MHz, copy=True)
    )
    quartic = (
        None
        if surface.quartic_distortion_MHz is None
        else np.array(surface.quartic_distortion_MHz, copy=True)
    )
    sextic = (
        None
        if surface.sextic_distortion_MHz is None
        else np.array(surface.sextic_distortion_MHz, copy=True)
    )
    spin_surfaces = {
        name: np.array(field, copy=True)
        for name, field in (surface.spin_tensor_surfaces_MHz or {}).items()
    }
    selected = tuple(zip(variables, values, strict=True))
    for variable, raw in selected:
        value, path = float(raw) * variable["multiplier"], variable["path"]
        if path == "potential.scale":
            potential = np.asarray(surface.potential_MHz) * value
        elif path == "F.scale":
            F = np.asarray(surface.F_MHz) * value
        elif path == "rho.scale":
            rho = np.asarray(surface.rho) * value
        elif path == "R.scale":
            R = np.asarray(surface.R_MHz) * value
    for variable, raw in selected:
        value, path = float(raw) * variable["multiplier"], variable["path"]
        if (
            path.startswith("potential.")
            and path != "potential.scale"
            and not path.startswith("potential.fourier.")
        ):
            _head, function, order = path.split(".")
            primitive = (
                np.cos(int(order) * surface.coordinates[0])
                if function == "cos"
                else np.sin(int(order) * surface.coordinates[0])
            )
            shape = (len(primitive),) + (1,) * (surface.dimension - 1)
            potential += value * np.broadcast_to(primitive.reshape(shape), surface.grid_shape)
        elif match := re.fullmatch(r"R\.(cos|sin|one_minus_cos)\.(\d+)\.(\d+)\.(\d+)", path):
            function, order, left, right = match.groups()
            primitive = (
                np.sin(int(order) * surface.coordinates[0])
                if function == "sin"
                else np.cos(int(order) * surface.coordinates[0])
            )
            if function == "one_minus_cos":
                primitive = 1.0 - primitive
            shape = (len(primitive),) + (1,) * (surface.dimension - 1)
            field = value * np.broadcast_to(primitive.reshape(shape), surface.grid_shape)
            R[..., int(left), int(right)] += field
            R[..., int(right), int(left)] = R[..., int(left), int(right)]
        elif match := re.fullmatch(
            r"R\.(cos|sin|one_minus_cos)\.(\d+)\.(isotropic|b_minus_c)", path
        ):
            function, order, pattern = match.groups()
            primitive = (
                np.sin(int(order) * surface.coordinates[0])
                if function == "sin"
                else np.cos(int(order) * surface.coordinates[0])
            )
            if function == "one_minus_cos":
                primitive = 1.0 - primitive
            shape = (len(primitive),) + (1,) * (surface.dimension - 1)
            field = value * np.broadcast_to(primitive.reshape(shape), surface.grid_shape)
            if pattern == "isotropic":
                for axis in range(3):
                    R[..., axis, axis] += field
            else:
                R[..., 1, 1] += field
                R[..., 2, 2] -= field
        elif match := re.fullmatch(r"R\.offset\.(\d+)\.(\d+)", path):
            left, right = map(int, match.groups())
            R[..., left, right] += value
            R[..., right, left] = R[..., left, right]
        elif match := re.fullmatch(
            r"(potential|F|rho|R)\.fourier\.([a-z_,]+)\.([0-9,]+)(?:\.(\d+)\.(\d+))?",
            path,
        ):
            family, functions, harmonics, left, right = match.groups()
            field = value * _multi_lam_fourier_field(surface, functions, harmonics)
            if family == "potential":
                potential += field
            else:
                array = {"F": F, "rho": rho, "R": R}[family]
                i, j = int(left), int(right)
                array[..., i, j] += field
                if family in {"F", "R"}:
                    array[..., j, i] = array[..., i, j]
        elif match := re.fullmatch(r"spin\.(.+)\.scale", path):
            name = match.group(1)
            spin_surfaces[name] = np.asarray(surface.spin_tensor_surfaces_MHz[name]) * value
        elif match := re.fullmatch(r"lam_p4\.(\d+)\.scale", path):
            assert lam_p4 is not None
            lam_p4[..., int(match.group(1))] *= value
        elif match := re.fullmatch(r"p2R\.(\d+)\.(\d+)\.(\d+)\.scale", path):
            assert p2R is not None
            mu, left, right = map(int, match.groups())
            p2R[..., mu, left, right] *= value
            p2R[..., mu, right, left] = p2R[..., mu, left, right]
        elif match := re.fullmatch(r"p2R\.(\d+)\.isotropic\.scale", path):
            assert p2R is not None
            mu = int(match.group(1))
            for axis in range(3):
                p2R[..., mu, axis, axis] *= value
        elif match := re.fullmatch(r"pJ3\.(\d+)\.(\d+)\.scale", path):
            assert pJ3 is not None
            mu, axis = map(int, match.groups())
            pJ3[..., mu, axis] *= value
        elif match := re.fullmatch(r"pJaa\.(\d+)\.(\d+)\.(\d+)\.scale", path):
            assert pJaa is not None
            mu, squared_axis, linear_axis = map(int, match.groups())
            pJaa[..., mu, squared_axis, linear_axis] *= value
        elif match := re.fullmatch(r"pi2\.(J|K|minus)\.scale", path):
            assert pi2 is not None
            pi2[{"J": 0, "K": 1, "minus": 2}[match.group(1)]] *= value
        elif match := re.fullmatch(r"pi2\.(J|K|minus)\.MHz", path):
            if pi2 is None:
                pi2 = np.zeros(3)
            pi2[{"J": 0, "K": 1, "minus": 2}[match.group(1)]] += value
        elif match := re.fullmatch(r"quartic\.(\d+)\.scale", path):
            assert quartic is not None
            quartic[..., int(match.group(1))] *= value
        elif match := re.fullmatch(r"sextic\.(\d+)\.scale", path):
            assert sextic is not None
            sextic[..., int(match.group(1))] *= value
        elif path.startswith("operator."):
            continue
        elif path not in {"potential.scale", "F.scale", "rho.scale", "R.scale"}:
            family, left, right, _scale = path.split(".")
            array = {"F": F, "rho": rho, "R": R}[family]
            array[..., int(left), int(right)] *= value
            if family in {"F", "R"}:
                array[..., int(right), int(left)] = array[..., int(left), int(right)]
    return replace(
        surface,
        potential_MHz=potential,
        F_MHz=F,
        rho=rho,
        R_MHz=R,
        lam_quartic_kinetic_MHz=lam_p4,
        p2_rotational_metric_MHz=p2R,
        p_rotational_cubic_MHz=pJ3,
        p_rotational_anticommutator_MHz=pJaa,
        pi2_rotation_MHz=pi2,
        quartic_distortion_MHz=quartic,
        sextic_distortion_MHz=sextic,
        spin_tensor_surfaces_MHz=spin_surfaces,
    )


def _multi_lam_fourier_field(surface, functions, harmonics):
    """Return a product Fourier field spanning every LAM coordinate."""
    names = tuple(functions.split(","))
    orders = tuple(int(value) for value in harmonics.split(","))
    if len(names) != surface.dimension or len(orders) != surface.dimension:
        raise ValueError("product Fourier paths require one function and harmonic per LAM")
    allowed = {"constant", "cos", "sin", "one_minus_cos"}
    if any(name not in allowed for name in names) or any(order < 0 for order in orders):
        raise ValueError("invalid product Fourier function or harmonic")
    field = np.ones(surface.grid_shape, dtype=float)
    for axis, (name, order, grid, periodic) in enumerate(
        zip(names, orders, surface.coordinates, surface.periodic, strict=True)
    ):
        if name == "constant":
            if order != 0:
                raise ValueError("constant Fourier factors require harmonic zero")
            factor = np.ones(len(grid), dtype=float)
        else:
            if not periodic or order < 1:
                raise ValueError("nonconstant Fourier factors require a positive periodic harmonic")
            phase = order * np.asarray(grid, dtype=float)
            factor = np.sin(phase) if name == "sin" else np.cos(phase)
            if name == "one_minus_cos":
                factor = 1.0 - factor
        shape = [1] * surface.dimension
        shape[axis] = len(factor)
        field *= factor.reshape(shape)
    return field


def _surface_parameter_direction(surface, variables, values, column):
    """Return the exact affine surface derivative for one fit variable.

    Every public fit path is affine in its controlling coordinate.  A unit
    coefficient extraction gives the derivative without a truncation error.
    The extraction increment follows parameter magnitude/scale to avoid
    cancellation for large coefficients; no displaced spectrum is solved.
    """
    base_values = np.asarray(values, dtype=float)
    shifted_values = base_values.copy()
    increment=max(1.0,abs(float(base_values[column])),abs(float(variables[column].get("scale",1.0))))
    shifted_values[column] += increment
    base = _apply_lam_variables(surface, variables, base_values)
    shifted = _apply_lam_variables(surface, variables, shifted_values)

    def derivative(name):
        left, right = getattr(shifted, name), getattr(base, name)
        if left is None and right is None:
            return None
        if left is None or right is None:
            raise RuntimeError(f"LAM fit path changed optional field {name!r} discontinuously")
        return (np.asarray(left) - np.asarray(right))/increment

    spin = {}
    for name in surface.spin_tensor_surfaces_MHz or {}:
        spin[name] = (
            np.asarray(shifted.spin_tensor_surfaces_MHz[name])
            - np.asarray(base.spin_tensor_surfaces_MHz[name])
        )/increment
    return {
        name: derivative(name)
        for name in (
            "potential_MHz",
            "F_MHz",
            "rho",
            "R_MHz",
            "geometric_potential_MHz",
            "lam_quartic_kinetic_MHz",
            "p2_rotational_metric_MHz",
            "p_rotational_cubic_MHz",
            "p_rotational_anticommutator_MHz",
            "pi2_rotation_MHz",
            "quartic_distortion_MHz",
            "sextic_distortion_MHz",
        )
    } | {"spin_tensor_surfaces_MHz": spin}


def _offset_surface(surface, direction, amount):
    fields = {}
    for name in (
        "potential_MHz",
        "F_MHz",
        "rho",
        "R_MHz",
        "geometric_potential_MHz",
        "lam_quartic_kinetic_MHz",
        "p2_rotational_metric_MHz",
        "p_rotational_cubic_MHz",
        "p_rotational_anticommutator_MHz",
        "pi2_rotation_MHz",
        "quartic_distortion_MHz",
        "sextic_distortion_MHz",
    ):
        base, delta = getattr(surface, name), direction[name]
        if base is None:
            fields[name] = None if delta is None else amount * np.asarray(delta)
        else:
            fields[name] = np.asarray(base) + amount * delta
    fields["spin_tensor_surfaces_MHz"] = {
        name: np.asarray(field) + amount * direction["spin_tensor_surfaces_MHz"][name]
        for name, field in (surface.spin_tensor_surfaces_MHz or {}).items()
    }
    return replace(surface, **fields)


def _lam_direction_action(surface, direction, values):
    """Apply the exact linear derivative through the production kinetic kernel."""
    columns = np.asarray(values, dtype=complex).reshape((int(np.prod(surface.grid_shape)), -1))
    potential = np.asarray(direction["potential_MHz"]).reshape(-1).copy()
    if direction["geometric_potential_MHz"] is not None:
        potential += np.asarray(direction["geometric_potential_MHz"]).reshape(-1)
    return _apply_lam_hamiltonian(surface, columns, potential=potential, coefficient_direction=direction)


def _projected_coupled_hamiltonian(surface,vectors,J,basis_template,**kwargs):
    factors=_projected_coupled_factors(surface,vectors,J,basis_template,**kwargs)
    return _hermitian(sum(np.kron(left,right) for left,right in factors))


def _projected_coupled_factors(surface, vectors, J, basis_template, *, operator_manifest=None, operator_parameters_MHz=None, representation='Ir', maximum_rotation_order=None, rotation_algebra=None):
    potential = np.asarray(surface.potential_MHz, dtype=float).reshape(-1).copy()
    if surface.geometric_potential_MHz is not None:
        potential += np.asarray(surface.geometric_potential_MHz, dtype=float).reshape(-1)
    acted = _apply_lam_hamiltonian(surface, vectors, potential=potential)
    lam_hamiltonian = _hermitian(vectors.conj().T @ acted)
    basis = LAMBasis(energies_MHz=np.real(np.diag(lam_hamiltonian)), vectors=vectors, momentum_operators=basis_template.momentum_operators, hamiltonian_MHz=basis_template.hamiltonian_MHz, diagonalizer_backend=basis_template.diagonalizer_backend, symmetry_sectors=basis_template.symmetry_sectors, sector_state_indices=basis_template.sector_state_indices)
    contracted = _contracted_lam_operators(surface, basis)
    rotational = _rotational_operators(J, representation) if rotation_algebra is None else rotation_algebra[0]
    nrot = 2 * J + 1
    identity = np.eye(nrot) if rotation_algebra is None else rotation_algebra[1]
    factors = [(lam_hamiltonian, identity)]
    for axis in range(3):
        factors.append((contracted['linear'][axis], rotational[axis]))
    for a in range(3):
        for b in range(3):
            operator = 0.5 * (rotational[a] @ rotational[b] + rotational[b] @ rotational[a])
            factors.append((contracted['quadratic'][a, b], operator))
    if contracted['distortion'].shape[0] and (maximum_rotation_order is None or maximum_rotation_order>=4):
        if rotation_algebra is None:
            operators = centrifugal_distortion_operators(J, surface.distortion_reduction, representation)
        else:
            from .native_rotor import _centrifugal_distortion_algebra
            operators = _centrifugal_distortion_algebra(
                rotational, surface.distortion_reduction, representation,
                power=lambda a,n:a.power(n), freeze=False)
        for coefficient, operator in zip(contracted['distortion'], operators[:len(contracted['distortion'])], strict=True):
            factors.append((coefficient, operator))
    if contracted['p2_rotational'].shape[0]:
        for mu in range(surface.dimension):
            for a in range(3):
                for b in range(3):
                    operator = 0.5 * (rotational[a] @ rotational[b] + rotational[b] @ rotational[a])
                    factors.append((contracted['p2_rotational'][mu, a, b], operator))
    if contracted['p_rotational_cubic'].shape[0] and (maximum_rotation_order is None or maximum_rotation_order>=3):
        total_rotation2 = sum((operator @ operator for operator in rotational))
        for mu in range(surface.dimension):
            for axis in range(3):
                factors.append((contracted['p_rotational_cubic'][mu, axis], rotational[axis] @ total_rotation2))
    if contracted['p_rotational_anticommutator'].shape[0] and (maximum_rotation_order is None or maximum_rotation_order>=3):
        for mu in range(surface.dimension):
            for squared_axis in range(3):
                square = rotational[squared_axis] @ rotational[squared_axis]
                for linear_axis in range(3):
                    coefficient = contracted['p_rotational_anticommutator'][mu, squared_axis, linear_axis]
                    operator = 0.5 * (square @ rotational[linear_axis] + rotational[linear_axis] @ square)
                    factors.append((coefficient, operator))
    if surface.pi2_rotation_MHz is not None and np.any(surface.pi2_rotation_MHz):
        from .pi2_rotation import project_covariant_momentum_matrices, pi2_rotation_factors
        projected = project_covariant_momentum_matrices(vectors, np.asarray(surface.rho).reshape((-1, 3, surface.dimension)), lambda mu, columns: basis.momentum_operators[mu] @ columns if basis.momentum_operators else _apply_momentum(surface, mu, columns))
        families, _ = pi2_rotation_factors(*projected, rotational, np.ones(3), _operator_algebra=rotation_algebra is not None)
        for coefficient, family in zip(surface.pi2_rotation_MHz, families, strict=True):
            if maximum_rotation_order is None or maximum_rotation_order>=4:
                selected=family
            elif maximum_rotation_order>=3:
                selected=tuple(family[i] for i in (0,1,5,9))
            else:
                selected=family[:1]
            for scale, lam, rot in selected:
                factors.append((coefficient * scale * lam, rot))
    if operator_manifest is not None:
        from .general_lam_operator import compile_general_lam_factors
        if maximum_rotation_order is not None:
            from .general_lam_operator import truncate_rotational_order
            operator_manifest=truncate_rotational_order(operator_manifest,maximum_rotation_order)
        factors.extend(compile_general_lam_factors(surface, basis, J, operator_manifest, representation=representation, parameters_MHz=operator_parameters_MHz))
    return tuple(factors)

def _lam_level_index(selector, result, mappings):
    if selector.reference_index is not None:
        if selector.J not in mappings:
            return selector.reference_index
        if selector.reference_index not in mappings[selector.J]:
            raise ValueError(
                "assigned LAM state left the retained continuation subspace; "
                "increase lam_states"
            )
        return mappings[selector.J][selector.reference_index]
    matches = [
        index
        for index, level in enumerate(result.blocks[selector.J].levels)
        if selector.Ka is not None
        and level.Ka == selector.Ka
        and level.Kc == selector.Kc
        and level.torsional_state == selector.torsional_state
        and (selector.lam_symmetry is None or level.lam_symmetry == selector.lam_symmetry)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"LAM selector matched {len(matches)} levels; add symmetry or reference_index"
        )
    return matches[0]


def _lam_operator_response_jacobian(
    surface,
    solved,
    variables,
    values,
    observations,
    mappings,
    *,
    operator_manifest,
    operator_parameters_MHz,
    response_states,
    workers,
    operator_contraction=None,
    moving_k_contraction=False,
    parameter_reference_surface=None,
):
    """Analytic retained/discarded response with Hellmann-Feynman energies."""
    parameter_reference_surface=surface if parameter_reference_surface is None else parameter_reference_surface
    if response_states < 0:
        raise ValueError("basis_response_states must be nonnegative")
    operator_columns = tuple(
        index for index, variable in enumerate(variables)
        if variable["path"].startswith("operator.")
    )
    if operator_columns:
        from .general_lam_operator import ContractedGeneralLAMOperators, contract_general_lam_operators
        from .operator_engine import hellmann_feynman_factorized_jacobian

        contraction = (
            operator_contraction if operator_contraction is not None
            else contract_general_lam_operators(surface, solved.basis, operator_manifest)
        )
        combinations = {
            str(index): {
                variables[index]["path"].removeprefix("operator."):
                variables[index]["multiplier"]
            }
            for index in operator_columns
        }
        level_derivatives = {}
        required_levels = {J: set() for J in solved.blocks}
        for observation in observations:
            for side in ("lower", "upper"):
                selector = observation[side]
                required_levels[selector.J].add(_lam_level_index(selector, solved, mappings))
        invariant_sectors = (
            surface.symmetry_sector_count > 1
            and all(term.symmetry == "totally-symmetric" for term in contraction.manifest.terms)
        )
        sectors = np.asarray(solved.basis.symmetry_sectors)
        restrictions = []
        for sector in range(surface.symmetry_sector_count if invariant_sectors else 1):
            indices = np.flatnonzero(sectors == sector) if invariant_sectors else np.arange(len(sectors))
            if not len(indices):
                continue
            local = ContractedGeneralLAMOperators(
                contraction.manifest,
                tuple(matrix[np.ix_(indices, indices)] for matrix in contraction.lam_factors),
            )
            restrictions.append((sector, indices, local))
        for J, block in solved.blocks.items():
            derivatives = np.zeros((len(block.energies_MHz), len(operator_columns)))
            tensor = block.vectors.reshape(len(sectors), 2 * J + 1, -1)
            for sector, indices, local in restrictions:
                levels = (
                    np.asarray([level.index for level in block.levels
                                if level.symmetry_sector == sector and level.index in required_levels[J]], dtype=int)
                    if invariant_sectors else np.asarray(sorted(required_levels[J]), dtype=int)
                )
                if not len(levels):
                    continue
                model = local.compile(
                    J, representation=solved.representation,
                    parameters_MHz=operator_parameters_MHz,
                )
                local_vectors = tensor[np.ix_(indices, np.arange(2 * J + 1), levels)].reshape(len(indices) * (2 * J + 1), -1)
                _names, derivatives[levels] = hellmann_feynman_factorized_jacobian(
                    model, local_vectors, parameter_combinations=combinations,
                )
            if moving_k_contraction and solved.k_contractions is not None:
                from .lam_response import moving_ritz_derivatives,apply_factors
                from .general_lam_operator import compile_general_lam_factors
                base=_projected_coupled_factors(surface,solved.basis.vectors,J,solved.basis,
                    operator_manifest=operator_manifest,operator_parameters_MHz=operator_parameters_MHz,
                    representation=solved.representation)
                original=compile_general_lam_factors(surface,solved.basis,J,contraction,
                    representation=solved.representation,parameters_MHz=operator_parameters_MHz)
                for local_index,index in enumerate(operator_columns):
                    parameter=variables[index]["path"].removeprefix("operator.")
                    changed=dict(operator_parameters_MHz);changed[parameter]+=variables[index]["multiplier"]
                    df=compile_general_lam_factors(surface,solved.basis,J,contraction,
                        representation=solved.representation,parameters_MHz=changed)+tuple((-a,b) for a,b in original)
                    corrected=moving_ritz_derivatives(solved.k_contractions[J],base,df,block.energies_MHz,block.vectors)
                    derivatives[:,local_index]=corrected
            level_derivatives[J] = derivatives
        result = np.empty((len(observations), len(variables)))
        for row, observation in enumerate(observations):
            lower, upper = observation["lower"], observation["upper"]
            il = _lam_level_index(lower, solved, mappings)
            iu = _lam_level_index(upper, solved, mappings)
            result[row, operator_columns] = (
                level_derivatives[upper.J][iu] - level_derivatives[lower.J][il]
            )
        surface_columns = tuple(index for index in range(len(variables)) if index not in operator_columns)
        if surface_columns:
            result[:, surface_columns] = _lam_operator_response_jacobian(
                surface, solved, tuple(variables[index] for index in surface_columns),
                np.asarray(values)[list(surface_columns)], observations, mappings,
                operator_manifest=operator_manifest,
                operator_parameters_MHz=operator_parameters_MHz,
                response_states=response_states, workers=workers, moving_k_contraction=moving_k_contraction,
                parameter_reference_surface=parameter_reference_surface,
            )
        return result
    retained = solved.basis.vectors
    retained_count = retained.shape[1]
    grid_size = retained.shape[0]
    primitive_size = solved.basis.primitive_size or grid_size
    extended_count = min(primitive_size, retained_count + response_states)
    if extended_count>retained_count and any(v["path"].startswith(("potential.","F.","lam_p4.")) for v in variables):
        extended = solve_lam_basis(surface,state_count=extended_count,
            primitive_basis="dvr" if solved.basis.primitive_representation == "dvr" else "fourier",
            fourier_cutoffs=solved.basis.fourier_cutoffs)
        discarded=extended.vectors[:,retained_count:]
        discarded_energies=extended.energies_MHz[retained_count:]
    else:
        discarded=np.empty((grid_size,0),complex);discarded_energies=np.empty(0)
    retained_energies = solved.basis.energies_MHz
    selected_j = tuple(sorted(solved.blocks))

    # A fixed Galerkin span and constant kinetic/rho fields admit an exact
    # polynomial derivative in preprojected momenta. Share these projections
    # across all columns; do not repeatedly differentiate zero field products.
    d = surface.dimension
    kinetic = np.asarray(surface.F_MHz).reshape(grid_size, d, d)
    rho_field = np.asarray(surface.rho).reshape(grid_size, 3, d)
    constant_kinematics = (
        response_states == 0
        and np.all(kinetic == kinetic[0])
        and np.all(rho_field == rho_field[0])
    )
    momentum_projection = second_projection = None
    if constant_kinematics and any(v["path"].startswith(("F.", "rho.")) for v in variables):
        from .lam_rotation import _project_lam_values
        def momentum(axis, vectors):
            return (solved.basis.momentum_operators[axis] @ vectors
                    if solved.basis.momentum_operators else _apply_momentum(surface, axis, vectors))
        pu = [momentum(axis, retained) for axis in range(d)]
        def project_action(action):
            return _project_lam_values(solved.basis, action,
                preserve_sectors=surface.symmetry_sector_count > 1)
        momentum_projection = [project_action(action) for action in pu]
        second_projection = {(mu, nu): project_action(momentum(mu, pu[nu]))
                             for mu in range(d) for nu in range(d)}

    # All columns at this physical point share primitive actions. Construct
    # them once even when derivative columns run concurrently.
    from threading import Lock
    response_workspace = None
    response_workspace_lock = Lock()
    def shared_response_workspace():
        nonlocal response_workspace
        with response_workspace_lock:
            if response_workspace is None:
                from .lam_surface_response import prepare_surface_response_workspace
                response_workspace = prepare_surface_response_workspace(
                    surface, solved, extended=extended if discarded.shape[1] else None)
            return response_workspace

    def column_response(column):
        direction = _surface_parameter_direction(parameter_reference_surface, variables, values, column)
        active_fields = {
            name
            for name, field in direction.items()
            if name != "spin_tensor_surfaces_MHz"
            and field is not None
            and np.any(field)
        }
        if any(np.any(field) for field in direction["spin_tensor_surfaces_MHz"].values()):
            active_fields.add("spin_tensor_surfaces_MHz")
        def scalar_projection(field):
            if np.all(field == field[0]):
                return field[0]*np.eye(retained_count)
            from .lam_rotation import _project_lam_values
            return _project_lam_values(solved.basis,field[:,None]*retained,
                preserve_sectors=surface.symmetry_sector_count > 1)
        if active_fields in ({"quartic_distortion_MHz"},{"sextic_distortion_MHz"}):
            family=next(iter(active_fields))
            coefficients=np.asarray(direction[family]).reshape((grid_size,-1))
            projected={k:scalar_projection(coefficients[:,k]) for k in range(coefficients.shape[1]) if np.any(coefficients[:,k])}
            operators={}
            for J in selected_j:
                words=centrifugal_distortion_operators(J,surface.distortion_reduction,solved.representation)
                offset=0 if family=="quartic_distortion_MHz" else 5
                operators[J]=tuple((field,words[offset+k]) for k,field in projected.items())
            return column,operators
        if active_fields=={"R_MHz"}:
            dR=np.asarray(direction["R_MHz"]).reshape((grid_size,3,3))
            projected={(a,b):scalar_projection(dR[:,a,b]) for a in range(3) for b in range(3) if np.any(dR[:,a,b])}
            operators={}
            for J in selected_j:
                rot=_rotational_operators(J,solved.representation)
                operators[J]=tuple((field,.5*(rot[a]@rot[b]+rot[b]@rot[a])) for (a,b),field in projected.items())
            return column,operators
        if (constant_kinematics and momentum_projection is not None
                and active_fields <= {"potential_MHz", "F_MHz", "rho", "R_MHz"}
                and np.all(direction["F_MHz"] == np.asarray(direction["F_MHz"]).reshape(grid_size,d,d)[0])
                and np.all(direction["rho"] == np.asarray(direction["rho"]).reshape(grid_size,3,d)[0])):
            df = np.asarray(direction["F_MHz"]).reshape(grid_size,d,d)[0]
            dr = np.asarray(direction["rho"]).reshape(grid_size,3,d)[0]
            f, rho = kinetic[0], rho_field[0]
            h0 = np.asarray(scalar_projection(np.asarray(direction["potential_MHz"]).reshape(-1)),complex)
            for (mu,nu), matrix in second_projection.items():
                h0 += df[mu,nu] * matrix
            dl = -2. * (rho @ df + dr @ f)
            dq = dr @ f @ rho.T + rho @ df @ rho.T + rho @ f @ dr.T
            linear = [sum((dl[a,mu]*momentum_projection[mu] for mu in range(d)),
                          np.zeros((retained_count,retained_count),complex)) for a in range(3)]
            dR = np.asarray(direction["R_MHz"]).reshape(grid_size,3,3)
            quadratic = {(a,b): scalar_projection(dR[:,a,b]) + dq[a,b]*np.eye(retained_count)
                         for a in range(3) for b in range(3) if np.any(dR[:,a,b]) or dq[a,b] != 0.}
            operators = {}
            for J in selected_j:
                rot = _rotational_operators(J,solved.representation)
                operators[J] = ((h0,np.eye(2*J+1)),) + tuple(
                    (matrix,rot[a]) for a,matrix in enumerate(linear) if np.any(matrix)
                ) + tuple((matrix,.5*(rot[a]@rot[b]+rot[b]@rot[a]))
                          for (a,b),matrix in quadratic.items())
            return column,operators
        if "spin_tensor_surfaces_MHz" not in active_fields:
            # Use the same analytic product-rule operator response as the
            # structural fitter. Subtracting projected large Hamiltonians
            # loses weak derivatives even when the polynomial stencil is exact.
            from .lam_surface_response import native_surface_response
            operators,_=native_surface_response(surface,solved,direction,
                response_states=response_states,manifest=operator_manifest,
                parameters=operator_parameters_MHz,workspace=shared_response_workspace())
            return column,operators
        derivative_action = _lam_direction_action(surface, direction, retained)
        if discarded.shape[1]:
            gaps = retained_energies[None, :] - discarded_energies[:, None]
            if np.any(np.abs(gaps) < 1.0e-10):
                raise RuntimeError("LAM contraction response encountered a zero retained-state gap")
            response = discarded @ ((discarded.conj().T @ derivative_action) / gaps)
        else:
            response = np.zeros_like(retained)
        # The projected Hamiltonian is a polynomial of degree at most four in
        # a directional displacement: the surface is affine, rho enters at
        # most quadratically, and bra/ket responses are affine.  The symmetric
        # five-point coefficient extraction is consequently the exact first
        # polynomial coefficient (up to floating-point roundoff), rather than
        # a truncation-controlled finite difference.
        step = 1.0e-3 / max(1.0, float(np.max(np.abs(response), initial=0.0)))
        displaced = {
            multiplier: (
                _offset_surface(surface, direction, multiplier * step),
                retained + multiplier * step * response,
            )
            for multiplier in (-2.0, -1.0, 1.0, 2.0)
        }
        operators = {}
        for J in selected_j:
            matrices = {
                multiplier: _projected_coupled_factors(
                    candidate_surface, candidate_vectors, J, solved.basis,
                    operator_manifest=operator_manifest,
                    operator_parameters_MHz=operator_parameters_MHz,
                    representation=solved.representation,
                )
                for multiplier, (candidate_surface, candidate_vectors) in displaced.items()
            }
            operators[J] = tuple((coefficient*left,right)
                for multiplier,coefficient in ((-2.,1./(12*step)),(-1.,-8./(12*step)),
                    (1.,8./(12*step)),(2.,-1./(12*step)))
                for left,right in matrices[multiplier])
        return column, operators

    if workers == 1 or len(variables) == 1:
        columns = tuple(column_response(column) for column in range(len(variables)))
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(
            max_workers=min(workers, len(variables)),
            thread_name_prefix="lock-lam-response",
        ) as pool:
            columns = tuple(pool.map(column_response, range(len(variables))))
    columns = tuple(sorted(columns, key=lambda item: item[0]))
    from .lam_response import apply_factors, moving_ritz_derivatives
    level_derivatives={}
    required_levels = {J: set() for J in selected_j}
    for observation in observations:
        for side in ('lower', 'upper'):
            selector = observation[side]
            required_levels[selector.J].add(_lam_level_index(selector, solved, mappings))
    for J in selected_j:
        block=solved.blocks[J]
        indices = np.asarray(sorted(required_levels[J]), dtype=int)
        moving=solved.k_contractions is not None and moving_k_contraction
        factors=(_projected_coupled_factors(surface,retained,J,solved.basis,
            operator_manifest=operator_manifest,operator_parameters_MHz=operator_parameters_MHz,
            representation=solved.representation) if moving else None)
        derivatives=[]
        for _column,operators in columns:
            if moving:
                derivatives.append(moving_ritz_derivatives(solved.k_contractions[J],factors,operators[J],
                    block.energies_MHz,block.vectors))
            else:
                selected_vectors = block.vectors[:, indices]
                action=apply_factors(operators[J],selected_vectors,retained_count,2*J+1)
                column = np.zeros(len(block.energies_MHz))
                column[indices] = np.real(np.sum(selected_vectors.conj()*action,axis=0))
                derivatives.append(column)
        level_derivatives[J]=np.asarray(derivatives)
    jacobian = np.empty((len(observations), len(variables)))
    for row, observation in enumerate(observations):
        lower, upper = observation["lower"], observation["upper"]
        il = _lam_level_index(lower, solved, mappings)
        iu = _lam_level_index(upper, solved, mappings)
        jacobian[row] = level_derivatives[upper.J][:, iu] - level_derivatives[lower.J][:, il]
    return jacobian


def _primitive_lam_vectors(result: LAMRotationResult):
    mapped = {}
    for J, block in result.blocks.items():
        transform = np.kron(result.basis.vectors, np.eye(2 * J + 1))
        mapped[J] = (np.asarray(block.energies_MHz), transform @ block.vectors)
    return mapped


def track_lam_rotation_states(reference, current, *, diagnostic_indices=None, degeneracy_tolerance=1.0e-8, subspace_gap_MHz=None):
    """Track through an exact cross-Gram contraction of two DVR subspaces.

    Both bases must use the same primitive coordinate grid. Projecting the
    current states into the reference LAM subspace preserves every overlap
    exactly and avoids storing primitive grid times rotation wavefunctions.
    """
    if reference.basis.vectors.shape[0] != current.basis.vectors.shape[0]:
        raise ValueError("LAM tracking requires a common primitive coordinate grid")
    gram = (
        None if reference.basis is current.basis
        else reference.basis.vectors.conj().T @ current.basis.vectors
    )
    left, right, indices_by_key, diagnostics_by_key = {}, {}, {}, {}
    for J, before in reference.blocks.items():
        after = current.blocks[J]
        projected = after.vectors
        if gram is not None:
            tensor = after.vectors.reshape(len(current.basis.energies_MHz), 2 * J + 1, -1)
            projected = np.einsum("ab,brs->ars", gram, tensor, optimize=True).reshape(before.vectors.shape[0], -1)
        before_labels = np.asarray([level.symmetry_sector for level in before.levels])
        after_labels = np.asarray([level.symmetry_sector for level in after.levels])
        sector_ids = np.union1d(before_labels, after_labels)
        groups = []
        if len(sector_ids) > 1:
            nrot = 2*J+1
            before_basis_labels = np.repeat(reference.basis.symmetry_sectors, nrot)
            after_basis_labels = np.repeat(current.basis.symmetry_sectors, nrot)
            before_norms = np.sum(np.abs(before.vectors)**2, axis=0)
            after_norms = np.sum(np.abs(after.vectors)**2, axis=0)
            for sector in sector_ids:
                lcols = np.flatnonzero(before_labels == sector)
                rcols = np.flatnonzero(after_labels == sector)
                lrows = np.flatnonzero(before_basis_labels == sector)
                rrows = np.flatnonzero(after_basis_labels == sector)
                # Only use an invariant decomposition when the supplied states
                # really have pure translation sectors. Full molecular irreps
                # that mix components retain the unrestricted overlap route.
                lp = before.vectors[np.ix_(lrows, lcols)]
                rp = after.vectors[np.ix_(rrows, rcols)]
                if (np.max(np.abs(before_norms[lcols]-np.sum(np.abs(lp)**2,axis=0)), initial=0.) > 1.e-10
                    or np.max(np.abs(after_norms[rcols]-np.sum(np.abs(rp)**2,axis=0)), initial=0.) > 1.e-10):
                    groups = []
                    break
                groups.append((int(sector), lcols, rcols, lrows))
        if not groups:
            groups = [(None, np.arange(len(before.energies_MHz)),
                       np.arange(len(after.energies_MHz)), np.arange(len(before.vectors)))]
        if subspace_gap_MHz is not None:
            before_irreps=[level.rovibrational_irrep for level in before.levels]
            after_irreps=[level.rovibrational_irrep for level in after.levels]
            if any(label is not None for label in before_irreps+after_irreps):
                if any(label is None for label in before_irreps+after_irreps):
                    raise ValueError("incomplete molecular-irrep registry during projector tracking")
                split=[]
                for sector,lcols,rcols,rows in groups:
                    for label in sorted(set(before_irreps)|set(after_irreps)):
                        lc=np.array([i for i in lcols if before_irreps[i]==label],int)
                        rc=np.array([i for i in rcols if after_irreps[i]==label],int)
                        if len(lc):
                            if not len(rc):raise ValueError("assigned molecular irrep left the retained space")
                            split.append(((sector,label),lc,rc,rows))
                groups=split
        selected = None if diagnostic_indices is None else set(diagnostic_indices[J])
        for sector, lcols, rcols, rows in groups:
            key = (J, sector)
            left[key] = (before.energies_MHz[lcols], before.vectors[np.ix_(rows,lcols)])
            right[key] = (after.energies_MHz[rcols], projected[np.ix_(rows,rcols)])
            indices_by_key[key] = (lcols,rcols)
            diagnostics_by_key[key] = tuple(i for i, original in enumerate(lcols)
                                            if selected is None or original in selected)
    local, diagnostic = _overlap_mappings(
        left, right, diagnostic_indices=diagnostics_by_key,
        degeneracy_tolerance=degeneracy_tolerance, subspace_gap_MHz=subspace_gap_MHz,
    )
    mappings = {J:{} for J in reference.blocks}
    for key, mapping in local.items():
        lcols, rcols = indices_by_key[key]
        mappings[key[0]].update({int(lcols[i]):int(rcols[j]) for i,j in mapping.items()})
    return mappings, diagnostic


def assign_lam_reference_states(
    reference: LAMRotationResult,
    selectors: tuple[LAMLevelSelector, ...] | list[LAMLevelSelector],
    *,
    label_prior_MHz: float = 20.0,
    label_degeneracy_MHz: float = 5.0,
) -> tuple[tuple[LAMLevelSelector, ...], tuple[dict[str, Any], ...]]:
    """Bind persistent state IDs to reference eigenvalue indices globally.

    ``label_prior_MHz`` resolves candidates within
    ``label_degeneracy_MHz`` of the nearest energy with a soft
    spectroscopic-label prior.  Outside that ambiguity window the assignment
    is controlled only by the reference energy.
    """
    if not math.isfinite(label_prior_MHz) or label_prior_MHz < 0.0:
        raise ValueError("LAM assignment label prior must be finite and nonnegative")
    if not math.isfinite(label_degeneracy_MHz) or label_degeneracy_MHz < 0.0:
        raise ValueError("LAM assignment degeneracy window must be finite and nonnegative")
    by_state: dict[str, LAMLevelSelector] = {}
    for selector in selectors:
        if selector.state_id is None:
            continue
        previous = by_state.get(selector.state_id)
        if previous is not None:
            left = (
                previous.J,
                previous.lam_symmetry,
                previous.torsional_state,
                previous.reference_index,
                previous.reference_energy_MHz,
            )
            right = (
                selector.J,
                selector.lam_symmetry,
                selector.torsional_state,
                selector.reference_index,
                selector.reference_energy_MHz,
            )
            if left != right:
                raise ValueError(f"inconsistent metadata for LAM state_id {selector.state_id!r}")
        else:
            by_state[selector.state_id] = selector

    assigned_indices: dict[str, int] = {}
    groups: dict[tuple[int, str | None, int], list[LAMLevelSelector]] = {}
    for selector in by_state.values():
        if selector.reference_index is not None:
            block = reference.blocks[selector.J]
            if selector.reference_index >= len(block.energies_MHz):
                raise ValueError(f"reference_index outside J={selector.J} block")
            assigned_indices[selector.state_id or ""] = selector.reference_index
        else:
            if selector.reference_energy_MHz is None:
                raise ValueError(
                    f"persistent LAM state {selector.state_id!r} needs reference_energy_MHz"
                )
            groups.setdefault(
                (selector.J, selector.lam_symmetry, selector.torsional_state), []
            ).append(selector)

    for (J, symmetry, torsional_state), targets in groups.items():
        candidates = [
            level
            for level in reference.blocks[J].levels
            if level.torsional_state == torsional_state
            and (symmetry is None or level.lam_symmetry == symmetry)
        ]
        if len(candidates) < len(targets):
            raise ValueError(
                f"only {len(candidates)} candidates for {len(targets)} persistent states "
                f"in J={J}, symmetry={symmetry}, vt={torsional_state}"
            )
        energy_differences = (
            np.asarray([item.reference_energy_MHz for item in targets], dtype=float)[:, None]
            - np.asarray([item.energy_MHz for item in candidates], dtype=float)[None, :]
        )
        costs = energy_differences * energy_differences
        candidate_energies = np.asarray([item.energy_MHz for item in candidates], dtype=float)
        ambiguous = (
            np.count_nonzero(
                np.abs(candidate_energies[:, None] - candidate_energies[None, :])
                <= label_degeneracy_MHz,
                axis=1,
            )
            > 1
        )
        for row, target in enumerate(targets):
            if target.Ka is None:
                continue
            for column, candidate in enumerate(candidates):
                if not ambiguous[column]:
                    continue
                label_distance = abs(candidate.Ka - target.Ka) + abs(candidate.Kc - target.Kc)
                costs[row, column] += label_prior_MHz**2 * label_distance
        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows, columns, strict=True):
            assigned_indices[targets[int(row)].state_id or ""] = candidates[int(column)].index

    diagnostics = []
    for state_id, selector in sorted(by_state.items()):
        index = assigned_indices[state_id]
        energy = float(reference.blocks[selector.J].energies_MHz[index])
        diagnostics.append(
            {
                "state_id": state_id,
                "J": selector.J,
                "lam_symmetry": selector.lam_symmetry,
                "torsional_state": selector.torsional_state,
                "reference_index": index,
                "target_energy_MHz": selector.reference_energy_MHz,
                "assigned_energy_MHz": energy,
                "energy_mismatch_MHz": None
                if selector.reference_energy_MHz is None
                else energy - selector.reference_energy_MHz,
                "Ka": reference.blocks[selector.J].levels[index].Ka,
                "Kc": reference.blocks[selector.J].levels[index].Kc,
            }
        )
    resolved = tuple(
        selector
        if selector.state_id is None
        else replace(selector, reference_index=assigned_indices[selector.state_id])
        for selector in selectors
    )
    return resolved, tuple(diagnostics)


def _overlap_mappings(
    reference,
    current,
    *,
    diagnostic_indices=None,
    degeneracy_tolerance=1.0e-8,
    subspace_gap_MHz=None,
):
    mappings = {}
    minimum_principal = 1.0
    minimum_state = 1.0
    subspace_records = {}
    for J, (left_values, left_vectors) in reference.items():
        right_values, right_vectors = current[J]
        left_values = np.asarray(left_values, dtype=float)
        right_values = np.asarray(right_values, dtype=float)
        left_vectors = np.asarray(left_vectors, dtype=complex)
        right_vectors = np.asarray(right_vectors, dtype=complex)
        if left_vectors.shape[0] != right_vectors.shape[0]:
            raise ValueError("LAM continuation requires a common primitive basis")
        if subspace_gap_MHz is not None:
            from .subspace_transport import transport_spectral_subspaces
            tracked = transport_spectral_subspaces(
                left_values, right_values, left_vectors.conj().T @ right_vectors,
                gap_tolerance=float(subspace_gap_MHz),
                diagnostic_indices=None if diagnostic_indices is None else diagnostic_indices[J],
            )
            mappings[J] = tracked["mapping"]
            minimum_principal = min(minimum_principal, tracked["minimum_principal_overlap"])
            minimum_state = min(minimum_state, tracked["minimum_state_overlap"])
            subspace_records[str(J)] = tracked["subspaces"]
            continue
        if (
            left_vectors.shape[1] == right_vectors.shape[1]
            and left_values.shape == right_values.shape
        ):
            tracked = track_eigenstates_by_subspace(
                left_values,
                left_vectors,
                right_values,
                right_vectors,
                degeneracy_tolerance=degeneracy_tolerance,
            )
            mappings[J] = {
                index: current_index
                for index, current_index in enumerate(tracked.permutation)
            }
            selected = (
                tuple(diagnostic_indices[J])
                if diagnostic_indices is not None
                else tuple(range(len(tracked.permutation)))
            )
            selected_overlaps = [tracked.squared_overlaps[index] for index in selected]
            if selected_overlaps:
                local_minimum = float(np.min(selected_overlaps))
                minimum_state = min(minimum_state, local_minimum)
                minimum_principal = min(minimum_principal, math.sqrt(local_minimum))
            continue

        # Completing a degeneracy at the contraction boundary can change the
        # retained dimension by one between fit evaluations.  Match the common
        # subspace and leave only unmatched boundary states unmapped; assigned
        # spectroscopic states are required to remain in the common subspace.
        from scipy.optimize import linear_sum_assignment

        overlap = np.abs(left_vectors.conj().T @ right_vectors) ** 2
        rows, columns = linear_sum_assignment(-overlap)
        mappings[J] = {
            int(row): int(column) for row, column in zip(rows, columns, strict=True)
        }
        selected = (
            set(diagnostic_indices[J]) if diagnostic_indices is not None else set(map(int, rows))
        )
        diagnostic_pairs = [
            (int(row), int(column))
            for row, column in zip(rows, columns, strict=True)
            if int(row) in selected
        ]
        assigned = np.asarray(
            [overlap[row, column] for row, column in diagnostic_pairs], dtype=float
        )
        if assigned.size:
            minimum_state = min(minimum_state, float(np.min(assigned)))
            minimum_principal = min(
                minimum_principal, math.sqrt(float(np.min(assigned)))
            )
    return mappings, {
        "minimum_principal_overlap": float(minimum_principal),
        "minimum_state_overlap": float(minimum_state),
        **({"projector_subspaces": subspace_records} if subspace_records else {}),
    }


def _identifiability_report(
    variables, singular, right_h, data_singular, data_right_h, rcond
):
    """Describe weak physical combinations before and after QM predicates."""

    names = tuple(item["name"] for item in variables)

    def directions(values, vectors):
        if not len(values):
            return []
        cutoff = float(values[0] * rcond)
        rows = []
        for index, value in enumerate(values):
            if value > cutoff:
                continue
            vector = np.asarray(vectors[index], dtype=float)
            order = np.argsort(np.abs(vector))[::-1][: min(4, len(vector))]
            rows.append(
                {
                    "singular_value": float(value),
                    "components": [
                        {"name": names[item], "coefficient": float(vector[item])}
                        for item in order
                        if abs(vector[item]) > 1.0e-8
                    ],
                }
            )
        return rows

    data_cutoff = float(data_singular[0] * rcond) if len(data_singular) else 0.0
    augmented_cutoff = float(singular[0] * rcond) if len(singular) else 0.0
    return {
        "data_rank": int(np.count_nonzero(data_singular > data_cutoff)),
        "augmented_rank": int(np.count_nonzero(singular > augmented_cutoff)),
        "qm_predicates_restore_rank": bool(
            np.count_nonzero(singular > augmented_cutoff)
            > np.count_nonzero(data_singular > data_cutoff)
        ),
        "data_null_directions": directions(data_singular, data_right_h),
        "remaining_null_directions": directions(singular, right_h),
    }


def _lam_level_energy(selector, result, mappings):
    block = result.blocks[selector.J]
    if selector.reference_index is not None:
        index = mappings.get(selector.J, {}).get(selector.reference_index, selector.reference_index)
        return float(block.energies_MHz[index])
    matches = [
        index
        for index, level in enumerate(block.levels)
        if selector.Ka is not None
        and level.Ka == selector.Ka
        and level.Kc == selector.Kc
        and level.torsional_state == selector.torsional_state
        and (selector.lam_symmetry is None or level.lam_symmetry == selector.lam_symmetry)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"LAM selector matched {len(matches)} levels; add symmetry or reference_index"
        )
    return float(block.energies_MHz[matches[0]])


def _finite_jacobian(
    function, x, scales, lower, upper, *, workers=1, scheme="forward", columns=None
):
    base = np.asarray(function(x))
    result = np.empty((len(base), len(x)))

    def evaluate_column(column):
        step = np.cbrt(np.finfo(float).eps) * max(abs(x[column]), scales[column], 1.0)
        plus = x.copy()
        plus[column] = min(upper[column], x[column] + step)
        if scheme == "forward" and plus[column] > x[column]:
            values = (function(plus) - base) / (plus[column] - x[column])
        elif scheme == "forward":
            minus = x.copy()
            minus[column] = max(lower[column], x[column] - step)
            values = (base - function(minus)) / (x[column] - minus[column])
        else:
            minus = x.copy()
            minus[column] = max(lower[column], x[column] - step)
            values = (function(plus) - function(minus)) / (plus[column] - minus[column])
        return column, values

    selected_columns = tuple(range(len(x))) if columns is None else tuple(columns)
    if workers == 1 or len(selected_columns) == 1:
        evaluated = tuple(evaluate_column(column) for column in selected_columns)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(
            max_workers=min(workers, len(selected_columns)),
            thread_name_prefix="lock-fit-jacobian",
        ) as pool:
            evaluated = tuple(pool.map(evaluate_column, selected_columns))
    for column, values in evaluated:
        result[:, column] = values
    return result


__all__ = [
    "LAMFitResult",
    "LAMLevelSelector",
    "LAM_SPECTRAL_FIT_RESULT_SCHEMA",
    "LAM_SPECTRAL_FIT_SCHEMA",
    "assign_lam_reference_states",
    "fit_lam_assigned_spectrum",
    "track_lam_rotation_states",
]
