"""Normal-mode isotropy subgroups for symmetry-lowered LINK branches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModeSymmetryAnalysis:
    """Action of a parent point group on one Cartesian normal mode."""

    parent_point_group: str
    preserving_operation_labels: tuple[str, ...]
    branch_exchanging_operation_labels: tuple[str, ...]
    broken_operation_labels: tuple[str, ...]
    relative_residuals: tuple[tuple[str, float, float], ...]
    tolerance: float

    @property
    def is_totally_symmetric(self) -> bool:
        return not self.branch_exchanging_operation_labels and not self.broken_operation_labels


@dataclass(frozen=True)
class OrderParameterDirection:
    """One symmetry-inequivalent ray in a degenerate unstable subspace."""

    coefficients: tuple[float, ...]
    mass_weighted_direction: np.ndarray
    cartesian_direction: np.ndarray
    mode_symmetry: ModeSymmetryAnalysis
    orbit_size: int
    sign_equivalent: bool
    selection_policy: str


@dataclass(frozen=True)
class OrderParameterSubspaceAnalysis:
    """Representation diagnostics for a clustered unstable subspace."""

    dimension: int
    invariant_residual: float
    character_norm: float | None
    reducible: bool
    representation_source: str


def _canonical_ray(vector: np.ndarray, *, tolerance: float) -> np.ndarray:
    result = np.asarray(vector, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(result))
    if not np.all(np.isfinite(result)) or norm <= tolerance:
        raise ValueError("order-parameter direction must be finite and nonzero")
    result = result / norm
    pivot = int(np.flatnonzero(np.abs(result) > tolerance)[0])
    if result[pivot] < 0.0:
        result = -result
    return result


def _same_ray(left: np.ndarray, right: np.ndarray, *, tolerance: float) -> bool:
    return abs(float(np.asarray(left) @ np.asarray(right))) >= 1.0 - tolerance


def _restricted_subspace_representation(
    symmetry,
    basis: np.ndarray,
    *,
    tolerance: float,
) -> tuple[tuple[np.ndarray, ...], float]:
    from matrix_chem import cartesian_operation_matrix

    matrices: list[np.ndarray] = []
    maximum_residual = 0.0
    projector = basis @ basis.T
    for operation in symmetry.operations:
        cartesian = cartesian_operation_matrix(
            operation.rotation,
            tuple(int(atom) - 1 for atom in operation.permutation),
            natoms=basis.shape[0] // 3,
        )
        transformed = cartesian @ basis
        maximum_residual = max(
            maximum_residual,
            float(np.linalg.norm(transformed - projector @ transformed, ord=2)),
        )
        restricted = basis.T @ transformed
        nearest_orthogonal_left, _singular, nearest_orthogonal_right = np.linalg.svd(restricted)
        matrices.append(nearest_orthogonal_left @ nearest_orthogonal_right)
    if maximum_residual > max(100.0 * tolerance, 1.0e-7):
        raise ValueError("degenerate unstable subspace is not invariant under the parent group")
    return tuple(matrices), maximum_residual


def classify_order_parameter_subspace(
    symmetry,
    mass_weighted_basis: np.ndarray,
    *,
    tolerance: float = 1.0e-5,
) -> OrderParameterSubspaceAnalysis:
    """Classify symmetry versus accidental degeneracy before branch selection.

    A finite-group character norm of one for a two-dimensional cluster denotes
    one genuine two-dimensional irrep; any other norm denotes a reducible
    accidental degeneracy (the exact value depends on repeated components).
    Linear groups are
    classified analytically as axial continuous representations; their finite
    operation samples are used only for numerical subgroup auditing.
    """

    basis = np.asarray(mass_weighted_basis, dtype=float)
    if basis.ndim != 2 or basis.shape[1] < 1 or basis.shape[0] % 3:
        raise ValueError("order-parameter basis must be a nonempty 3N x d array")
    gram = basis.T @ basis
    if not np.allclose(gram, np.eye(basis.shape[1]), rtol=0.0, atol=max(10.0 * tolerance, 1.0e-8)):
        basis, _ = np.linalg.qr(basis)
    representation, residual = _restricted_subspace_representation(
        symmetry, basis, tolerance=tolerance
    )
    point_group = str(symmetry.point_group).strip().upper().replace("∞", "INF")
    linear = point_group in {"DINFH", "CINFV"}
    characters = np.asarray([float(np.trace(matrix)) for matrix in representation])
    norm = None if linear else float(characters @ characters / len(representation))
    reducible = bool(
        not linear
        and basis.shape[1] == 2
        and abs(float(norm) - 1.0) > max(100.0 * tolerance, 1.0e-6)
    )
    return OrderParameterSubspaceAnalysis(
        dimension=int(basis.shape[1]),
        invariant_residual=float(residual),
        character_norm=norm,
        reducible=reducible,
        representation_source="analytic_linear_axial" if linear else "finite_group_characters",
    )


def anharmonic_order_parameter_scan(
    mass_weighted_basis: np.ndarray,
    atomic_masses: np.ndarray,
    coordinates_angstrom: np.ndarray,
    energy_evaluator,
    *,
    radius_angstrom: float = 0.05,
    samples: int = 24,
) -> dict[str, object]:
    """Scan the angular dependence of a two-dimensional order parameter.

    ``energy_evaluator`` receives one Cartesian ``natoms x 3`` geometry and
    returns a scalar energy.  The scan is deliberately backend-agnostic, so it
    can use a cheap model before expensive branch optimizations.  It reports
    the minimum angular sectors but never silently replaces symmetry orbit
    selection; the latter remains an auditable discrete operation.
    """

    basis = np.asarray(mass_weighted_basis, dtype=float)
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    masses = np.asarray(atomic_masses, dtype=float).reshape(-1)
    if basis.ndim != 2 or basis.shape[1] != 2 or basis.shape[0] != 3 * masses.size:
        raise ValueError("anharmonic order-parameter scan requires a 3N x 2 basis")
    if coordinates.shape != (masses.size, 3) or np.any(masses <= 0.0):
        raise ValueError("anharmonic scan geometry or masses are invalid")
    radius = float(radius_angstrom)
    count = int(samples)
    if not np.isfinite(radius) or radius <= 0.0 or count < 4:
        raise ValueError("anharmonic scan radius must be positive and samples >= 4")
    basis, _ = np.linalg.qr(basis)
    mass_scale = np.sqrt(np.repeat(masses, 3))
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    energies: list[float] = []
    for angle in angles:
        coefficients = np.asarray((np.cos(angle), np.sin(angle)), dtype=float)
        mass_weighted = basis @ coefficients
        cartesian = (mass_weighted / mass_scale).reshape(masses.size, 3)
        cartesian /= max(float(np.max(np.linalg.norm(cartesian, axis=1))), 1.0e-30)
        energy = float(energy_evaluator(coordinates + radius * cartesian))
        if not np.isfinite(energy):
            raise ValueError("anharmonic order-parameter scan returned a non-finite energy")
        energies.append(energy)
    minimum = int(np.argmin(energies))
    return {
        "policy": "angular_anharmonic_order_parameter_scan",
        "radius_angstrom": radius,
        "samples": count,
        "angles_radians": [float(value) for value in angles],
        "energies_hartree": energies,
        "minimum_sample_index": minimum,
        "minimum_angle_radians": float(angles[minimum]),
        "minimum_energy_hartree": float(energies[minimum]),
    }


def symmetry_inequivalent_order_parameter_directions(
    symmetry,
    mass_weighted_basis: np.ndarray,
    atomic_masses: np.ndarray,
    *,
    tolerance: float = 1.0e-5,
) -> tuple[tuple[OrderParameterDirection, ...], dict[str, object]]:
    """Enumerate maximal-isotropy rays of a degenerate unstable irrep.

    Finite two-dimensional irreps are resolved from the real eigenlines of
    their restricted group operations and quotienting those lines by the full
    parent-group orbit.  For a linear molecule, the transverse two-dimensional
    irrep has a continuous axial orbit, so one projector-anchored ray is the
    unique symmetry-inequivalent representative.  The sign is omitted whenever
    a parent operation (or the continuous axial group) exchanges the two signs.
    """

    threshold = float(tolerance)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("order-parameter tolerance must be positive")
    basis = np.asarray(mass_weighted_basis, dtype=float)
    if basis.ndim != 2 or basis.shape[1] != 2 or basis.shape[0] % 3:
        raise ValueError("order-parameter selection requires a 3N x 2 basis")
    gram = basis.T @ basis
    if not np.allclose(gram, np.eye(2), rtol=0.0, atol=max(10.0 * threshold, 1.0e-8)):
        basis, _ = np.linalg.qr(basis)
    representation, invariant_residual = _restricted_subspace_representation(
        symmetry, basis, tolerance=threshold
    )
    point_group = str(symmetry.point_group).strip().upper().replace("∞", "INF")
    linear_continuous = point_group in {"DINFH", "CINFV"}
    characters = np.asarray([float(np.trace(matrix)) for matrix in representation])
    character_norm = float(characters @ characters / len(representation))
    if not linear_continuous and abs(character_norm - 1.0) > max(100.0 * threshold, 1.0e-6):
        raise ValueError(
            "degenerate unstable subspace is reducible under the parent group; "
            "it is not a multidimensional unstable irrep"
        )

    candidates: list[np.ndarray] = []
    policy = "finite_group_maximal_isotropy_rays"
    if linear_continuous:
        policy = "linear_continuous_axial_orbit"
        projector = basis @ basis.T
        for axis in np.eye(basis.shape[0]):
            projected = projector @ axis
            if float(np.linalg.norm(projected)) > threshold:
                candidates.append(_canonical_ray(basis.T @ projected, tolerance=threshold))
                break
    else:
        identity = np.eye(2)
        for matrix in representation:
            if np.allclose(matrix, identity, rtol=0.0, atol=threshold) or np.allclose(
                matrix, -identity, rtol=0.0, atol=threshold
            ):
                continue
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            for index, eigenvalue in enumerate(eigenvalues):
                if abs(float(np.imag(eigenvalue))) > threshold:
                    continue
                if abs(abs(float(np.real(eigenvalue))) - 1.0) > threshold:
                    continue
                candidate = np.real(eigenvectors[:, index])
                candidate = _canonical_ray(candidate, tolerance=threshold)
                if not any(_same_ray(candidate, item, tolerance=threshold) for item in candidates):
                    candidates.append(candidate)
        if not candidates:
            candidates.append(np.asarray((1.0, 0.0)))

    representatives: list[np.ndarray] = []
    for candidate in candidates:
        if any(
            any(
                _same_ray(existing, matrix @ candidate, tolerance=threshold)
                for matrix in representation
            )
            for existing in representatives
        ):
            continue
        representatives.append(_canonical_ray(candidate, tolerance=threshold))

    masses = np.asarray(atomic_masses, dtype=float).reshape(-1)
    if masses.size * 3 != basis.shape[0] or np.any(masses <= 0.0):
        raise ValueError("order-parameter atomic masses do not match the Cartesian basis")
    mass_scale = np.sqrt(np.repeat(masses, 3))
    directions: list[OrderParameterDirection] = []
    orbit_audit: list[dict[str, object]] = []
    for coefficients in representatives:
        orbit: list[np.ndarray] = []
        sign_equivalent = linear_continuous
        for matrix in representation:
            image = np.asarray(matrix @ coefficients, dtype=float)
            if float(image @ coefficients) <= -1.0 + threshold:
                sign_equivalent = True
            ray = _canonical_ray(image, tolerance=threshold)
            if not any(_same_ray(ray, item, tolerance=threshold) for item in orbit):
                orbit.append(ray)
        mass_weighted = basis @ coefficients
        mass_weighted /= max(float(np.linalg.norm(mass_weighted)), 1.0e-30)
        cartesian = mass_weighted / mass_scale
        cartesian /= max(float(np.linalg.norm(cartesian)), 1.0e-30)
        analysis = analyze_mode_symmetry(symmetry, cartesian, tolerance=threshold)
        directions.append(
            OrderParameterDirection(
                coefficients=tuple(float(value) for value in coefficients),
                mass_weighted_direction=mass_weighted,
                cartesian_direction=cartesian,
                mode_symmetry=analysis,
                orbit_size=len(orbit),
                sign_equivalent=sign_equivalent,
                selection_policy=policy,
            )
        )
        orbit_audit.append(
            {
                "coefficients": [float(value) for value in coefficients],
                "orbit_size": len(orbit),
                "sign_equivalent": sign_equivalent,
                "preserving_operations": list(analysis.preserving_operation_labels),
                "branch_exchanging_operations": list(analysis.branch_exchanging_operation_labels),
            }
        )
    audit = {
        "policy": policy,
        "parent_point_group": str(symmetry.point_group),
        "subspace_dimension": 2,
        "restricted_characters": characters.tolist(),
        "restricted_character_norm": character_norm,
        "representation_source": "analytic_linear_axial" if linear_continuous else "finite_group_characters",
        "invariant_subspace_residual": invariant_residual,
        "candidate_ray_count": len(candidates),
        "inequivalent_ray_count": len(directions),
        "orbits": orbit_audit,
    }
    return tuple(directions), audit


def analyze_mode_symmetry(
    symmetry,
    cartesian_mode: np.ndarray,
    *,
    tolerance: float = 1.0e-5,
) -> ModeSymmetryAnalysis:
    """Classify parent operations as preserving, exchanging, or breaking a mode."""

    from matrix_chem import cartesian_operation_matrix

    threshold = float(tolerance)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("mode-symmetry tolerance must be positive")
    vector = np.asarray(cartesian_mode, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)) or norm <= 1.0e-14 or vector.size % 3:
        raise ValueError("Cartesian normal mode must be a finite nonzero 3N vector")
    natoms = vector.size // 3
    preserving: list[str] = []
    exchanging: list[str] = []
    broken: list[str] = []
    residuals: list[tuple[str, float, float]] = []
    for operation in symmetry.operations:
        matrix = cartesian_operation_matrix(
            operation.rotation,
            tuple(int(atom) - 1 for atom in operation.permutation),
            natoms=natoms,
        )
        transformed = matrix @ vector
        plus = float(np.linalg.norm(transformed - vector) / norm)
        minus = float(np.linalg.norm(transformed + vector) / norm)
        label = str(operation.label)
        residuals.append((label, plus, minus))
        if plus <= threshold:
            preserving.append(label)
        elif minus <= threshold:
            exchanging.append(label)
        else:
            broken.append(label)
    if not preserving:
        raise ValueError("mode symmetry analysis did not retain the identity operation")
    return ModeSymmetryAnalysis(
        parent_point_group=str(symmetry.point_group),
        preserving_operation_labels=tuple(preserving),
        branch_exchanging_operation_labels=tuple(exchanging),
        broken_operation_labels=tuple(broken),
        relative_residuals=tuple(residuals),
        tolerance=threshold,
    )


def _operation_matches(left, right, *, tolerance: float) -> bool:
    return bool(
        tuple(int(atom) for atom in left.permutation)
        == tuple(int(atom) for atom in right.permutation)
        and np.allclose(
            np.asarray(left.rotation, dtype=float),
            np.asarray(right.rotation, dtype=float),
            rtol=0.0,
            atol=tolerance,
        )
    )


def validate_detected_mode_subgroup(
    parent_symmetry,
    branch_symmetry,
    analysis: ModeSymmetryAnalysis,
) -> None:
    """Require ORACLE's detected branch group to equal the mode stabilizer."""

    preserving = tuple(
        operation
        for operation in parent_symmetry.operations
        if operation.label in analysis.preserving_operation_labels
    )
    detected = tuple(branch_symmetry.operations)
    tolerance = analysis.tolerance
    unmatched_parent = [
        operation.label
        for operation in preserving
        if not any(
            _operation_matches(operation, candidate, tolerance=tolerance) for candidate in detected
        )
    ]
    unmatched_branch = [
        operation.label
        for operation in detected
        if not any(
            _operation_matches(operation, candidate, tolerance=tolerance)
            for candidate in preserving
        )
    ]
    if unmatched_parent or unmatched_branch:
        raise ValueError(
            "ORACLE branch symmetry does not match the normal-mode isotropy subgroup: "
            f"missing={unmatched_parent}, unexpected={unmatched_branch}"
        )


__all__ = [
    "ModeSymmetryAnalysis",
    "OrderParameterDirection",
    "analyze_mode_symmetry",
    "symmetry_inequivalent_order_parameter_directions",
    "validate_detected_mode_subgroup",
]
