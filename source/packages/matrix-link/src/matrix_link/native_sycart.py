"""Compiled SYCART minimum optimizer boundary."""

from __future__ import annotations

import numpy as np

try:
    from . import _optimizer_native
except ImportError:  # source checkout without the optional local build
    _optimizer_native = None


def native_sycart_available() -> bool:
    return _optimizer_native is not None


def apply_native_sycart_preconditioner(preconditioner, vector):
    """Apply a frozen sparse Davidson--Krylov model entirely in C++."""

    if _optimizer_native is None:
        raise RuntimeError("the compiled LINK optimizer extension is unavailable")
    payload = (
        preconditioner
        if isinstance(preconditioner, dict)
        else _native_preconditioner_payload(preconditioner)
    )
    return _optimizer_native.apply_preconditioner(
        payload,
        np.ascontiguousarray(vector, dtype=float),
    )


def optimize_native_sycart(callback, coordinates_angstrom, directions_angstrom, **settings):
    """Run the compiled Cartesian optimizer with one frozen SYCART tangent.

    ``callback`` receives an ``(N,3)`` Angstrom array and returns energy in
    hartree plus its Cartesian gradient in hartree/bohr.  QR is deliberately
    outside the iteration loop; the C++ kernel owns all subsequent optimizer
    algebra and line-search decisions.
    """

    if _optimizer_native is None:
        raise RuntimeError("the compiled LINK optimizer extension is unavailable")
    rows = np.asarray(directions_angstrom, dtype=float).reshape(-1, np.size(coordinates_angstrom))
    basis, _ = np.linalg.qr(rows.T, mode="reduced")
    return _optimizer_native.optimize_sycart(
        callback,
        np.asarray(coordinates_angstrom, dtype=float),
        np.ascontiguousarray(basis),
        **settings,
    )


def optimize_native_sycart_matrix_gfn2(
    xyzin_path,
    *,
    method="MATRIX-GFN2",
    charge=0,
    multiplicity=1,
    processors=1,
    solver_mode="dense",
    davidson_preconditioner=True,
    preconditioner_refresh_interval=10,
    matrix_free_model=None,
    scaling_observables_path=None,
    **settings,
):
    """Run the compiled full-SYCART minimum path with resident MATRIX-GFN2.

    This is the low-overhead path for small and medium systems.  The frozen
    SYCART basis is built once; the C++ kernel then owns BFGS/L-BFGS, trust
    scaling, line search, and convergence.  Only one resident calculator call
    crosses the Python boundary per electronic evaluation.
    """

    from pathlib import Path
    from tempfile import TemporaryDirectory
    import time

    from matrix_chem import read_xyzin_geometry
    from .resident_calculators import create_resident_calculator

    from .optimizer import (
        GeometryEvaluationService,
        OptimizerSettings,
        _matrix_free_sycart_davidson_preconditioner,
        coordinate_model_from_xyzin,
    )

    path = Path(xyzin_path).expanduser().resolve()
    geometry = read_xyzin_geometry(path)
    model = coordinate_model_from_xyzin(
        path,
        kind="sycart",
        task_regime="MINIMUM",
        matrix_free=False,
    )
    preparation_started = time.perf_counter()
    temporary_directory = None
    preconditioner_audits = []
    if davidson_preconditioner:
        if matrix_free_model is None:
            matrix_free_model = coordinate_model_from_xyzin(
                path,
                kind="sycart",
                task_regime="MINIMUM",
                matrix_free=True,
            )
        temporary_directory = TemporaryDirectory(prefix="matrix-link-native-sycart-")
        service = GeometryEvaluationService(
            xyzin_path=path,
            run_dir=temporary_directory.name,
            coordinate_model=matrix_free_model,
            settings=OptimizerSettings(stationary_point="minimum"),
        )
        preconditioner, audit = _matrix_free_sycart_davidson_preconditioner(
            service,
            matrix_free_model,
        )
        if preconditioner is None:
            raise RuntimeError("the analytic SYCART Davidson preconditioner is unavailable")
        preconditioner_audits.append(audit)
        settings["preconditioner"] = _native_preconditioner_payload(preconditioner)
        interval = int(preconditioner_refresh_interval)
        # The analytic SH preconditioner is useful in both the dense and
        # matrix-free native solvers.  In particular, a dense SYCART basis
        # still changes as the geometry moves; disabling refresh solely
        # because the solver uses its dense algebra leaves the initial SH
        # frozen and can produce unnecessary late-iteration oscillations.
        if interval > 0:
            refresh_epoch = 0

            def refresh_preconditioner(coordinates):
                nonlocal preconditioner, refresh_epoch
                refresh_epoch += 1
                refreshed, refreshed_audit = _matrix_free_sycart_davidson_preconditioner(
                    service,
                    matrix_free_model,
                    coordinates_angstrom=np.asarray(coordinates, dtype=float),
                    refresh_epoch=refresh_epoch,
                    source_iteration=refresh_epoch * interval,
                    initial_ritz_vectors=preconditioner.eigenvectors,
                )
                if refreshed is None:
                    raise RuntimeError("SYCART Davidson refresh failed")
                preconditioner = refreshed
                preconditioner_audits.append(refreshed_audit)
                return _native_preconditioner_payload(refreshed)

            settings["preconditioner_refresh"] = refresh_preconditioner
            settings["preconditioner_refresh_interval"] = interval
    else:
        audit = None
    preparation_seconds = time.perf_counter() - preparation_started
    session = create_resident_calculator(
        "matrix_gfn2",
        geometry.atoms,
        geometry.coordinates_angstrom,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        processors=processors,
        solver_mode=solver_mode,
    )

    def evaluate(coordinates):
        result = session.evaluate(coordinates)
        return result.energy_hartree, result.gradient_hartree_per_bohr

    started = time.perf_counter()
    try:
        result = optimize_native_sycart(
            evaluate,
            geometry.coordinates_angstrom,
            model.directions_angstrom,
            **settings,
        )
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()
    result["wall_seconds"] = time.perf_counter() - started
    result["optimizer_overhead_seconds"] = max(
        0.0,
        result["wall_seconds"] - result["calculator_seconds"],
    )
    result["atoms"] = geometry.atoms
    result["coordinate_rank"] = len(model.labels)
    result["coordinate_labels"] = tuple(model.labels)
    result["final_q"] = np.asarray(model.directions_angstrom, dtype=float) @ (
        np.asarray(result["coordinates"], dtype=float)
        - np.asarray(geometry.coordinates_angstrom, dtype=float)
    ).reshape(-1)
    result["preconditioner_preparation_seconds"] = preparation_seconds
    result["preconditioner_audit"] = audit
    result["preconditioner_refresh_audits"] = tuple(preconditioner_audits[1:])
    if scaling_observables_path is not None:
        # Population analysis is intentionally evaluated once, at the accepted
        # final geometry.  Keeping it outside the optimization loop avoids the
        # Mayer prefactor on every geometry step while making geometry-only
        # KeyMaker jobs complete scientific records.
        result["scaling_observables_path"] = str(
            session.write_scaling_observables(
                scaling_observables_path,
                np.asarray(result["coordinates"], dtype=float),
            )
        )
    return result


def optimize_native_sycart_matrix_gfn2_batch(xyzin_paths, **settings):
    """Optimize several structures in one resident compiled SYCART process.

    Import and sparse-runtime initialization are paid once and reported
    separately.  Each molecular record retains its own analytic primitive
    preconditioner and resident MATRIX-GFN2 calculator; no electronic or topology
    state is shared between molecules.
    """

    import time

    initialization_started = time.perf_counter()
    # Eagerly initialize every lazily imported component used by the first
    # model build.  Otherwise the first molecule alone is charged for loading
    # SciPy and the compiled SMITH atlas, which obscures the steady resident
    # optimizer cost in small-molecule campaigns.
    from scipy.sparse import csr_matrix as _csr_matrix  # noqa: F401
    from matrix_smith import compiled_gic_primitive_csr as _compiled_csr  # noqa: F401
    from .optimizer import (  # noqa: F401
        GeometryEvaluationService as _GeometryEvaluationService,
        OptimizerSettings as _OptimizerSettings,
        _matrix_free_sycart_davidson_preconditioner as _preconditioner,
        coordinate_model_from_xyzin as _coordinate_model,
    )
    from .sycart_matrix_free import (  # noqa: F401
        ProjectedPrimitiveHessianOperator as _ProjectedPrimitiveHessianOperator,
        davidson_inverse_preconditioner as _davidson_inverse_preconditioner,
    )

    initialization_seconds = time.perf_counter() - initialization_started
    records = []
    for path in xyzin_paths:
        record = optimize_native_sycart_matrix_gfn2(path, **settings)
        record["resident_total_seconds"] = (
            float(record["preconditioner_preparation_seconds"])
            + float(record["wall_seconds"])
        )
        records.append(record)
    return {
        "runtime_initialization_seconds": initialization_seconds,
        "records": tuple(records),
    }


def _native_preconditioner_payload(preconditioner):
    """Freeze the Python Davidson object into arrays owned by the C++ kernel."""

    operator_method = getattr(preconditioner, "operator", None)
    operator = getattr(operator_method, "__self__", None)
    if operator is None or not hasattr(operator, "_csr"):
        raise TypeError("Davidson preconditioner has no projected sparse operator")
    runtime = operator.runtime
    csr = operator._csr
    return {
        "indptr": np.ascontiguousarray(csr.indptr, dtype=np.intp),
        "indices": np.ascontiguousarray(csr.indices, dtype=np.intp),
        "data": np.ascontiguousarray(csr.data, dtype=float),
        "curvatures": np.ascontiguousarray(operator.curvatures, dtype=float),
        "external_rows": np.ascontiguousarray(runtime.external_rows, dtype=float),
        "rotations": np.ascontiguousarray(np.stack(runtime.rotations), dtype=float),
        "permutations": np.ascontiguousarray(np.stack(runtime.permutations), dtype=np.intp),
        "eigenvalues": np.ascontiguousarray(preconditioner.eigenvalues, dtype=float),
        "eigenvectors": np.ascontiguousarray(preconditioner.eigenvectors, dtype=float),
        "baseline_inverse": float(preconditioner.baseline_inverse),
        "shift": float(preconditioner.regularization_shift),
        "tolerance": float(preconditioner.krylov_relative_tolerance),
        "maximum_iterations": int(preconditioner.krylov_maximum_iterations),
    }


__all__ = [
    "apply_native_sycart_preconditioner",
    "native_sycart_available",
    "optimize_native_sycart",
    "optimize_native_sycart_matrix_gfn2",
    "optimize_native_sycart_matrix_gfn2_batch",
]
