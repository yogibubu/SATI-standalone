from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import os
from time import perf_counter

import numpy as np

from matrix_chem import get_default_isotope, get_isotope
from matrix_chem.topology.elements import atomic_number
from matrix_numerics import resolve_compute_backend

from .vibin import didq_sym6, modes_from_hessian


@dataclass(frozen=True)
class CubicTransformDiagnostics:
    algorithm: str
    backend: str
    device: str
    unique_terms: int
    expanded_terms: int
    density: float
    zero_tolerance: float
    selection_reason: str
    elapsed_seconds: float
    first_stage_density: float
    second_stage_algorithm: str


_ALGORITHM_CACHE: dict[tuple[int, int, int, str], str] = {}


@dataclass(frozen=True)
class IsotopicCubicNormalField:
    """One Cartesian cubic force field expressed for one isotopic mass set."""

    masses_amu: np.ndarray
    frequencies_cm1: np.ndarray
    modes_mw: np.ndarray
    cubic_semidiagonal_f3ijj_mw: np.ndarray
    cubic_mw: np.ndarray | None
    coordinates_principal_A: np.ndarray
    modes_principal_mw: np.ndarray
    inertia_principal_amu_bohr2: np.ndarray
    inertia_derivatives_amu_sqrt_ang: np.ndarray
    coriolis: dict[str, np.ndarray]
    transform_diagnostics: CubicTransformDiagnostics

    @property
    def semidiagonal_f3ijj_mw(self) -> np.ndarray:
        return np.asarray(self.cubic_semidiagonal_f3ijj_mw, dtype=float)


@dataclass(frozen=True)
class IsotopologueRovibrationalKinematics:
    """Mass-specific principal-axis data required by Delta Bvib."""

    coordinates_principal_A: np.ndarray
    modes_principal_mw: np.ndarray
    inertia_principal_amu_bohr2: np.ndarray
    inertia_derivatives_amu_sqrt_ang: np.ndarray
    coriolis: dict[str, np.ndarray]


def rovibrational_kinematics_from_modes(
    coordinates_angstrom,
    masses_amu,
    modes_mw,
) -> IsotopologueRovibrationalKinematics:
    """Build the isotope-specific Eckart/principal-axis rovibrational data.

    ``modes_mw`` contains orthonormal mass-weighted normal modes with shape
    ``(nvib, natoms, 3)``.  The result is provider independent and can be used
    with either Cartesian or curvilinear potential derivatives.
    """

    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    masses = np.asarray(masses_amu, dtype=float).reshape(-1)
    modes = np.asarray(modes_mw, dtype=float)
    if coordinates.shape != (masses.size, 3):
        raise ValueError("coordinates and isotope masses disagree")
    if modes.ndim != 3 or modes.shape[1:] != (masses.size, 3):
        raise ValueError("mass-weighted modes must have shape nvib x natoms x 3")
    if np.any(masses <= 0.0) or not all(
        np.all(np.isfinite(value)) for value in (coordinates, masses, modes)
    ):
        raise ValueError("rovibrational kinematics require finite coordinates and positive masses")
    flattened = modes.reshape((modes.shape[0], -1))
    if not np.allclose(
        flattened @ flattened.T,
        np.eye(modes.shape[0]),
        rtol=2.0e-7,
        atol=2.0e-7,
    ):
        raise ValueError("mass-weighted normal modes are not orthonormal")
    principal_coords, principal_modes, inertia = _principal_axis_frame(
        coordinates, masses, modes
    )
    return IsotopologueRovibrationalKinematics(
        coordinates_principal_A=principal_coords,
        modes_principal_mw=principal_modes,
        inertia_principal_amu_bohr2=inertia,
        inertia_derivatives_amu_sqrt_ang=didq_sym6(
            masses, principal_coords, principal_modes
        ),
        coriolis=_coriolis_matrices(masses, principal_modes),
    )


def cartesian_cubic_from_normal_modes(
    cubic_normal_mw,
    normal_modes_cartesian,
    masses_amu,
    reduced_masses_amu,
    *,
    zero_tolerance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover a Cartesian cubic field from a complete vibrational F3.

    ``cubic_normal_mw`` is expressed in mass-weighted rectilinear normal
    coordinates.  Gaussian's printed/fchk displacement vectors are normalized
    in ordinary Cartesian space; multiplying them by ``sqrt(m/mu)`` restores
    the orthonormal mass-weighted eigenvectors.  Translational and rotational
    eigenvectors are implicitly reintroduced with zero potential derivatives,
    as in Mackie et al., J. Chem. Phys. 142, 244107 (2015).  This is the step
    that makes the normal-to-Cartesian transformation linear and permits a
    subsequent, isotope-specific Eckart projection.
    """

    phi = np.asarray(cubic_normal_mw, dtype=float)
    modes = np.asarray(normal_modes_cartesian, dtype=float)
    masses = np.asarray(masses_amu, dtype=float).reshape(-1)
    reduced = np.asarray(reduced_masses_amu, dtype=float).reshape(-1)
    nvib = modes.shape[0]
    dimension = 3 * masses.size
    if phi.shape != (nvib, nvib, nvib):
        raise ValueError("normal cubic tensor and normal-mode count disagree")
    if modes.shape != (nvib, dimension):
        raise ValueError("normal modes must have shape nvib x 3N")
    if reduced.shape != (nvib,) or np.any(reduced <= 0.0):
        raise ValueError("one positive reduced mass is required per normal mode")
    mass_coordinates = np.repeat(masses, 3)
    omega = modes * np.sqrt(mass_coordinates)[None, :] / np.sqrt(reduced)[:, None]
    gram = omega @ omega.T
    if not np.allclose(gram, np.eye(nvib), rtol=2.0e-7, atol=2.0e-7):
        raise ValueError("mass-weighted normal modes are not orthonormal")

    # F(Y) = Omega Phi Omega^T in Y=sqrt(M)X; recover F(X) afterwards.
    cubic_mw_cart = np.einsum(
        "ijk,ia,jb,kc->abc", phi, omega, omega, omega, optimize=True
    )
    roots = np.sqrt(mass_coordinates)
    cubic_cart = cubic_mw_cart * roots[:, None, None] * roots[None, :, None] * roots[None, None, :]
    return _unique_symmetric_terms(cubic_cart, float(zero_tolerance))


def transform_normal_cubic_for_isotopologue(
    symbols,
    coordinates_angstrom,
    hessian_hartree_per_bohr2,
    cubic_parent_qmw,
    parent_modes_cartesian,
    parent_reduced_masses_amu,
    parent_masses_amu,
    *,
    substitutions: dict[int, int] | None = None,
    zero_tolerance: float = 0.0,
    backend: str | None = None,
) -> IsotopicCubicNormalField:
    """Transform a complete parent normal F3 directly to isotope normal modes.

    The change-of-normal-coordinate matrix is built in the full 3N Cartesian
    metric, so the isotope-specific translational/rotational complement and
    Eckart frame are accounted for before the vibrational block is selected.
    This is the direct normal-to-normal form of the published 3N linear
    transformation and avoids materializing a dense Cartesian rank-3 tensor.
    """

    coords = np.asarray(coordinates_angstrom, dtype=float)
    hessian = np.asarray(hessian_hartree_per_bohr2, dtype=float)
    parent_modes = np.asarray(parent_modes_cartesian, dtype=float)
    parent_reduced = np.asarray(parent_reduced_masses_amu, dtype=float)
    parent_masses = np.asarray(parent_masses_amu, dtype=float)
    phi = np.asarray(cubic_parent_qmw, dtype=float)
    natoms = len(symbols)
    dimension = 3 * natoms
    nvib = parent_modes.shape[0]
    if parent_modes.shape != (nvib, dimension) or phi.shape != (nvib, nvib, nvib):
        raise ValueError("parent normal modes and cubic field disagree")
    if parent_reduced.shape != (nvib,) or parent_masses.shape != (natoms,):
        raise ValueError("parent reduced/atomic masses have incompatible dimensions")
    target_masses = isotopic_masses_amu(symbols, substitutions)
    packed_hessian = hessian[np.tril_indices(dimension)]
    frequencies, target_modes = modes_from_hessian(target_masses, packed_hessian, coords)
    if target_modes.shape[0] != nvib:
        raise ValueError("parent and isotope vibrational dimensions disagree")

    omega_parent = parent_modes * np.sqrt(np.repeat(parent_masses, 3))[None, :]
    omega_parent /= np.sqrt(parent_reduced)[:, None]
    omega_target = target_modes.reshape((nvib, dimension))
    metric_scale = np.sqrt(np.repeat(parent_masses / target_masses, 3))
    change = omega_parent @ (metric_scale[None, :] * omega_target).T
    unique_indices, unique_values = _unique_symmetric_terms(phi, float(zero_tolerance))
    semidiagonal, diagnostics = _transform_semidiagonal_adaptive(
        unique_indices,
        unique_values,
        nvib,
        change,
        requested_backend=backend,
        zero_tolerance=float(zero_tolerance),
    )
    kinematics = rovibrational_kinematics_from_modes(coords, target_masses, target_modes)
    return IsotopicCubicNormalField(
        masses_amu=target_masses,
        frequencies_cm1=frequencies,
        modes_mw=target_modes,
        cubic_semidiagonal_f3ijj_mw=semidiagonal,
        cubic_mw=None,
        coordinates_principal_A=kinematics.coordinates_principal_A,
        modes_principal_mw=kinematics.modes_principal_mw,
        inertia_principal_amu_bohr2=kinematics.inertia_principal_amu_bohr2,
        inertia_derivatives_amu_sqrt_ang=kinematics.inertia_derivatives_amu_sqrt_ang,
        coriolis=kinematics.coriolis,
        transform_diagnostics=diagnostics,
    )


def isotopic_masses_amu(
    symbols: tuple[str, ...] | list[str],
    substitutions: dict[int, int] | None = None,
) -> np.ndarray:
    """Return exact atomic masses; substitution indices are one based."""

    changes = substitutions or {}
    masses: list[float] = []
    for index, symbol in enumerate(symbols):
        number = atomic_number(symbol)
        if number is None:
            raise ValueError(f"unsupported atom label for isotope assignment: {symbol}")
        atom_index = index + 1
        isotope = (
            get_isotope(number, int(changes[atom_index]))
            if atom_index in changes
            else get_default_isotope(number)
        )
        if isotope is None:
            requested = changes.get(atom_index, "default")
            raise ValueError(f"no isotope mass for atom {atom_index} ({symbol}, A={requested})")
        masses.append(float(isotope.mass))
    unknown = sorted(set(changes) - set(range(1, len(symbols) + 1)))
    if unknown:
        raise ValueError(f"isotope substitution indices out of range: {unknown}")
    return np.asarray(masses, dtype=float)


def transform_cartesian_cubic_for_isotopologue(
    symbols: tuple[str, ...] | list[str],
    coordinates_angstrom,
    hessian_hartree_per_bohr2,
    cubic_hartree_per_bohr3=None,
    *,
    substitutions: dict[int, int] | None = None,
    zero_tolerance: float | None = None,
    backend: str | None = None,
    full_tensor: bool = False,
    cubic_unique_indices=None,
    cubic_unique_values_hartree_per_bohr3=None,
) -> IsotopicCubicNormalField:
    """Reuse Cartesian F2/F3 for an isotopologue under the BO approximation."""

    coords = np.asarray(coordinates_angstrom, dtype=float)
    hessian = np.asarray(hessian_hartree_per_bohr2, dtype=float)
    cubic = (
        None
        if cubic_hartree_per_bohr3 is None
        else np.asarray(cubic_hartree_per_bohr3, dtype=float)
    )
    natoms = len(symbols)
    dimension = 3 * natoms
    if coords.shape != (natoms, 3):
        raise ValueError("Cartesian coordinates must have shape natoms x 3")
    if hessian.shape != (dimension, dimension):
        raise ValueError("Cartesian Hessian shape does not match the molecule")
    if cubic is not None and cubic.shape != (dimension, dimension, dimension):
        raise ValueError("Cartesian cubic tensor shape does not match the molecule")
    masses = isotopic_masses_amu(symbols, substitutions)
    packed_hessian = hessian[np.tril_indices(dimension)]
    frequencies, modes_mw = modes_from_hessian(masses, packed_hessian, coords)

    # x = M^-1/2 L Q, where rows of modes_mw are the orthonormal columns L.
    cartesian_per_mw_q = modes_mw.reshape((-1, dimension)).T / np.sqrt(
        np.repeat(masses, 3)
    )[:, None]
    tolerance = (
        float(os.environ.get("MATRIX_CUBIC_ZERO_TOL", "1e-14"))
        if zero_tolerance is None
        else float(zero_tolerance)
    )
    if tolerance < 0.0:
        raise ValueError("cubic zero tolerance must be non-negative")
    if cubic_unique_indices is not None or cubic_unique_values_hartree_per_bohr3 is not None:
        if cubic_unique_indices is None or cubic_unique_values_hartree_per_bohr3 is None:
            raise ValueError("both sparse cubic indices and values are required")
        unique_indices = np.asarray(cubic_unique_indices, dtype=int).reshape((-1, 3))
        unique_values = np.asarray(
            cubic_unique_values_hartree_per_bohr3, dtype=float
        ).reshape(-1)
        if unique_indices.shape[0] != unique_values.size:
            raise ValueError("sparse cubic indices and values have different lengths")
        retained = np.abs(unique_values) > tolerance
        unique_indices, unique_values = unique_indices[retained], unique_values[retained]
    elif cubic is not None:
        unique_indices, unique_values = _unique_symmetric_terms(cubic, tolerance)
    else:
        raise ValueError("a dense or sparse Cartesian cubic tensor is required")
    semidiagonal, diagnostics = _transform_semidiagonal_adaptive(
        unique_indices,
        unique_values,
        dimension,
        cartesian_per_mw_q,
        requested_backend=backend,
        zero_tolerance=tolerance,
    )
    cubic_mw = None
    if full_tensor:
        cubic_mw = _transform_full_adaptive(
            unique_indices,
            unique_values,
            dimension,
            cartesian_per_mw_q,
            diagnostics,
        )

    kinematics = rovibrational_kinematics_from_modes(coords, masses, modes_mw)
    return IsotopicCubicNormalField(
        masses_amu=masses,
        frequencies_cm1=frequencies,
        modes_mw=modes_mw,
        cubic_semidiagonal_f3ijj_mw=semidiagonal,
        cubic_mw=cubic_mw,
        coordinates_principal_A=kinematics.coordinates_principal_A,
        modes_principal_mw=kinematics.modes_principal_mw,
        inertia_principal_amu_bohr2=kinematics.inertia_principal_amu_bohr2,
        inertia_derivatives_amu_sqrt_ang=kinematics.inertia_derivatives_amu_sqrt_ang,
        coriolis=kinematics.coriolis,
        transform_diagnostics=diagnostics,
    )


def _unique_symmetric_terms(
    cubic: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    size = cubic.shape[0]
    indices: list[tuple[int, int, int]] = []
    values: list[float] = []
    for i in range(size):
        for j in range(i + 1):
            for k in range(j + 1):
                value = float(cubic[i, j, k])
                if abs(value) > tolerance:
                    indices.append((i, j, k))
                    values.append(value)
    return np.asarray(indices, dtype=int).reshape((-1, 3)), np.asarray(values, dtype=float)


def _expanded_symmetric_terms(
    unique_indices: np.ndarray, unique_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    indices: list[tuple[int, int, int]] = []
    values: list[float] = []
    for index, value in zip(unique_indices, unique_values, strict=True):
        for permuted in sorted(set(permutations(tuple(int(item) for item in index)))):
            indices.append(permuted)
            values.append(float(value))
    return np.asarray(indices, dtype=int).reshape((-1, 3)), np.asarray(values, dtype=float)


def _transform_semidiagonal_adaptive(
    unique_indices: np.ndarray,
    unique_values: np.ndarray,
    dimension: int,
    transform: np.ndarray,
    *,
    requested_backend: str | None,
    zero_tolerance: float,
) -> tuple[np.ndarray, CubicTransformDiagnostics]:
    expanded_indices, expanded_values = _expanded_symmetric_terms(unique_indices, unique_values)
    density = float(expanded_values.size / max(dimension**3, 1))
    selected = resolve_compute_backend(
        requested_backend,
        workload_size=max(expanded_values.size * transform.shape[1], 1),
        require_float64=True,
    )
    scipy_sparse = _scipy_sparse_available()
    policy = os.environ.get("MATRIX_CUBIC_ALGORITHM", "auto").strip().casefold() or "auto"
    if policy not in {"auto", "sparse", "dense", "gpu"}:
        raise ValueError("MATRIX_CUBIC_ALGORITHM must be auto, sparse, dense or gpu")
    dense_bytes = dimension**3 * 8
    dense_limit = int(float(os.environ.get("MATRIX_CUBIC_DENSE_MEMORY_MB", "1024")) * 1024**2)
    if policy == "sparse" and not scipy_sparse:
        raise RuntimeError("sparse cubic transformation requested but SciPy is unavailable")
    if policy == "gpu" and not selected.accelerated:
        raise RuntimeError("GPU cubic transformation requested but no float64 GPU is available")

    def run(name: str) -> tuple[np.ndarray, float, str]:
        if name == "sparse":
            return _transform_semidiagonal_scipy(
                expanded_indices,
                expanded_values,
                dimension,
                transform,
                zero_tolerance=zero_tolerance,
            )
        if name == "gpu":
            return _transform_semidiagonal_torch(
                expanded_indices,
                expanded_values,
                dimension,
                transform,
                selected.device,
                zero_tolerance=zero_tolerance,
            )
        dense = _dense_from_expanded(expanded_indices, expanded_values, dimension)
        first_stage = np.einsum("abc,cj->abj", dense, transform, optimize=True)
        first_density = _array_density(first_stage, zero_tolerance)
        intermediate, second_algorithm = _contract_second_stage(
            first_stage, transform, zero_tolerance
        )
        return transform.T @ intermediate, first_density, second_algorithm

    if policy != "auto":
        chosen = policy
        start = perf_counter()
        result, first_stage_density, second_stage_algorithm = run(chosen)
        elapsed = perf_counter() - start
        selection_reason = f"explicit MATRIX_CUBIC_ALGORITHM={policy}"
    elif dense_bytes > dense_limit:
        if not scipy_sparse:
            raise MemoryError(
                "dense cubic contraction exceeds MATRIX_CUBIC_DENSE_MEMORY_MB and SciPy is unavailable"
            )
        chosen = "sparse"
        start = perf_counter()
        result, first_stage_density, second_stage_algorithm = run(chosen)
        elapsed = perf_counter() - start
        selection_reason = f"dense tensor would require {dense_bytes / 1024**2:.1f} MiB"
    else:
        density_bin = int(round(density * 20.0))
        key = (dimension, transform.shape[1], density_bin, selected.device)
        chosen = _ALGORITHM_CACHE.get(key, "")
        if chosen:
            start = perf_counter()
            result, first_stage_density, second_stage_algorithm = run(chosen)
            elapsed = perf_counter() - start
            selection_reason = "machine-local autotuning cache"
        else:
            candidates = ["dense"]
            if scipy_sparse:
                candidates.append("sparse")
            if selected.accelerated:
                candidates.append("gpu")
            timings: dict[str, float] = {}
            results: dict[str, tuple[np.ndarray, float, str]] = {}
            for candidate in candidates:
                start = perf_counter()
                results[candidate] = run(candidate)
                timings[candidate] = perf_counter() - start
            reference = results["dense"][0]
            for candidate, (candidate_result, _candidate_density, _second) in results.items():
                if not np.allclose(candidate_result, reference, rtol=1.0e-10, atol=1.0e-12):
                    raise RuntimeError(
                        f"cubic transformation backend {candidate} disagrees with CPU reference"
                    )
            chosen = min(timings, key=timings.get)
            _ALGORITHM_CACHE[key] = chosen
            result, first_stage_density, second_stage_algorithm = results[chosen]
            elapsed = timings[chosen]
            selection_reason = "measured machine-local autotuning: " + ", ".join(
                f"{name}={timings[name]:.6g}s" for name in sorted(timings)
            )
    if chosen == "sparse":
        algorithm, backend_name, device = "symmetric-coo-sparse", "scipy", "cpu"
    elif chosen == "gpu":
        algorithm, backend_name, device = (
            "dense-device-contraction",
            selected.name,
            selected.device,
        )
    else:
        algorithm, backend_name, device = "dense-blas-contraction", "numpy", "cpu"
    diagnostics = CubicTransformDiagnostics(
        algorithm=algorithm,
        backend=backend_name,
        device=device,
        unique_terms=int(unique_values.size),
        expanded_terms=int(expanded_values.size),
        density=density,
        zero_tolerance=zero_tolerance,
        selection_reason=selection_reason,
        elapsed_seconds=float(elapsed),
        first_stage_density=float(first_stage_density),
        second_stage_algorithm=second_stage_algorithm,
    )
    return np.asarray(result, dtype=float), diagnostics


def _transform_semidiagonal_scipy(
    indices: np.ndarray,
    values: np.ndarray,
    dimension: int,
    transform: np.ndarray,
    *,
    zero_tolerance: float,
) -> tuple[np.ndarray, float, str]:
    from scipy.sparse import coo_matrix

    rows = indices[:, 0] * dimension + indices[:, 1]
    flat = coo_matrix(
        (values, (rows, indices[:, 2])), shape=(dimension * dimension, dimension)
    ).tocsr()
    contracted_c = np.asarray(flat @ transform).reshape(
        (dimension, dimension, transform.shape[1])
    )
    first_density = _array_density(contracted_c, zero_tolerance)
    intermediate, second_algorithm = _contract_second_stage(
        contracted_c, transform, zero_tolerance
    )
    return transform.T @ intermediate, first_density, second_algorithm


def _transform_semidiagonal_torch(
    indices: np.ndarray,
    values: np.ndarray,
    dimension: int,
    transform: np.ndarray,
    device: str,
    *,
    zero_tolerance: float,
) -> tuple[np.ndarray, float, str]:
    import torch

    dense = torch.zeros((dimension, dimension, dimension), dtype=torch.float64, device=device)
    idx = torch.as_tensor(indices, dtype=torch.long, device=device)
    dense[idx[:, 0], idx[:, 1], idx[:, 2]] = torch.as_tensor(
        values, dtype=torch.float64, device=device
    )
    matrix = torch.as_tensor(transform, dtype=torch.float64, device=device)
    first_stage = torch.einsum("abc,cj->abj", dense, matrix)
    threshold = float(zero_tolerance)
    first_density = float(torch.count_nonzero(torch.abs(first_stage) > threshold).item()) / max(
        first_stage.numel(), 1
    )
    intermediate = torch.einsum("abj,bj->aj", first_stage, matrix)
    result = matrix.T @ intermediate
    return result.detach().cpu().numpy(), first_density, "device diagonal contraction"


def _transform_full_adaptive(
    unique_indices: np.ndarray,
    unique_values: np.ndarray,
    dimension: int,
    transform: np.ndarray,
    diagnostics: CubicTransformDiagnostics,
) -> np.ndarray:
    indices, values = _expanded_symmetric_terms(unique_indices, unique_values)
    dense = _dense_from_expanded(indices, values, dimension)
    if diagnostics.device in {"cuda", "mps"}:
        import torch

        tensor = torch.as_tensor(dense, dtype=torch.float64, device=diagnostics.device)
        matrix = torch.as_tensor(transform, dtype=torch.float64, device=diagnostics.device)
        result = torch.einsum("abc,ai,bj,ck->ijk", tensor, matrix, matrix, matrix)
        return _symmetrize_rank3(result.detach().cpu().numpy())
    return _symmetrize_rank3(
        np.einsum("abc,ai,bj,ck->ijk", dense, transform, transform, transform, optimize=True)
    )


def _dense_from_expanded(indices: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    dense = np.zeros((size, size, size), dtype=float)
    if values.size:
        dense[indices[:, 0], indices[:, 1], indices[:, 2]] = values
    return dense


def _array_density(values: np.ndarray, tolerance: float) -> float:
    array = np.asarray(values)
    return float(np.count_nonzero(np.abs(array) > tolerance) / max(array.size, 1))


def _contract_second_stage(
    first_stage: np.ndarray,
    transform: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, str]:
    density = _array_density(first_stage, tolerance)
    sparse_cutoff = float(os.environ.get("MATRIX_CUBIC_STAGE2_SPARSE_DENSITY", "0.35"))
    if density >= sparse_cutoff:
        return (
            np.einsum("abj,bj->aj", first_stage, transform, optimize=True),
            "BLAS diagonal contraction",
        )
    indices = np.argwhere(np.abs(first_stage) > tolerance)
    intermediate = np.zeros((first_stage.shape[0], first_stage.shape[2]), dtype=float)
    if indices.size:
        a, b, j = indices.T
        contributions = first_stage[a, b, j] * transform[b, j]
        np.add.at(intermediate, (a, j), contributions)
    return intermediate, "sparse indexed diagonal contraction"


def _scipy_sparse_available() -> bool:
    try:
        import scipy.sparse  # noqa: F401
    except Exception:
        return False
    return True


def _principal_axis_frame(
    coordinates_A: np.ndarray, masses: np.ndarray, modes_mw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.average(coordinates_A, axis=0, weights=masses)
    shifted = coordinates_A - center
    inertia_A2 = np.zeros((3, 3), dtype=float)
    for mass, vector in zip(masses, shifted, strict=True):
        inertia_A2 += mass * (
            np.dot(vector, vector) * np.eye(3) - np.outer(vector, vector)
        )
    moments, axes = np.linalg.eigh(inertia_A2)
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    principal_coords = shifted @ axes
    principal_modes = modes_mw @ axes
    angstrom_per_bohr = 0.52917721092
    return principal_coords, principal_modes, np.diag(moments / angstrom_per_bohr**2)


def _coriolis_matrices(masses: np.ndarray, modes_mw: np.ndarray) -> dict[str, np.ndarray]:
    nvib = modes_mw.shape[0]
    matrices = {axis: np.zeros((nvib, nvib), dtype=float) for axis in ("x", "y", "z")}
    for i in range(nvib):
        for j in range(i + 1, nvib):
            vector = np.sum(
                np.cross(modes_mw[i], modes_mw[j]), axis=0
            )
            for axis, value in zip(("x", "y", "z"), vector, strict=True):
                matrices[axis][i, j] = float(value)
                matrices[axis][j, i] = -float(value)
    return matrices


def _symmetrize_rank3(tensor: np.ndarray) -> np.ndarray:
    return (
        tensor
        + tensor.transpose(0, 2, 1)
        + tensor.transpose(1, 0, 2)
        + tensor.transpose(1, 2, 0)
        + tensor.transpose(2, 0, 1)
        + tensor.transpose(2, 1, 0)
    ) / 6.0
