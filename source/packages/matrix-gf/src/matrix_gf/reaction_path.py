"""SONIC Wilson-GF tools for relaxed Qim-type reaction paths.

The path direction is represented in the frozen, non-redundant SONIC tangent
space.  At a non-stationary path point the direction is removed with the
Wilson kinetic metric before the transverse Hessian is diagonalized.  This is
the coordinate-space counterpart of constructing curvilinear normal modes
orthogonal to a relaxed reaction path.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np
from matrix_smith import procrustes_align_salc_matrix

from .harmonic import HESSIAN_EIGENVALUE_TO_CM


@dataclass(frozen=True)
class SonicQimDirection:
    """Unstable saddle-point direction normalized in the Wilson metric."""

    eigenvalue: float
    frequency_cm: float
    sonic_direction: np.ndarray
    constraint_covector: np.ndarray
    negative_mode_count: int


@dataclass(frozen=True)
class SonicTransverseModes:
    """Path-orthogonal SONIC modes at a stationary or non-stationary point."""

    path_tangent: np.ndarray
    transverse_basis: np.ndarray
    projected_force_constants: np.ndarray
    eigenvalues: np.ndarray
    frequencies_cm: np.ndarray
    sonic_modes: np.ndarray
    gradient_transverse_fraction: float | None
    maximum_path_hessian_coupling: float


@dataclass(frozen=True)
class CartesianPathTransverseModes:
    """Mass-weighted Cartesian modes orthogonal to a reaction path and rigid motion."""

    transverse_basis: np.ndarray
    eigenvalues: np.ndarray
    frequencies_cm: np.ndarray
    zero_point_energy_cm: float
    external_rank: int
    negative_mode_count: int


@dataclass(frozen=True)
class HarmonicPathChannelBasis:
    """Local harmonic channels in a common coordinate frame."""

    frequencies_cm: np.ndarray
    coordinate_frame: np.ndarray
    kinetic_coefficient_cm: float
    center: np.ndarray | None = None


def harmonic_path_channel_overlap(
    left: HarmonicPathChannelBasis,
    right: HarmonicPathChannelBasis,
    *,
    maximum_quanta: int = 1,
    active_modes: np.ndarray | None = None,
) -> np.ndarray:
    """Return exact overlaps of local harmonic product states.

    The coordinate frames may be rectangular Cartesian transverse bases or
    square transported frames.  Their polar overlap supplies the orthogonal
    Duschinsky map, while the local frequencies determine the Gaussian widths.
    The result contains every product state whose total excitation does not
    exceed ``maximum_quanta``.
    """
    left_frequency = np.asarray(left.frequencies_cm, dtype=float)
    right_frequency = np.asarray(right.frequencies_cm, dtype=float)
    left_frame = np.asarray(left.coordinate_frame, dtype=float)
    right_frame = np.asarray(right.coordinate_frame, dtype=float)
    left_center = (
        np.zeros(left_frame.shape[0]) if left.center is None
        else np.asarray(left.center, dtype=float)
    )
    right_center = (
        np.zeros(right_frame.shape[0]) if right.center is None
        else np.asarray(right.center, dtype=float)
    )
    if (left_frequency.ndim != 1 or right_frequency.shape != left_frequency.shape
            or left_frame.shape[1] != len(left_frequency)
            or right_frame.shape[1] != len(left_frequency)
            or left_frame.shape[0] != right_frame.shape[0]
            or np.any(left_frequency <= 0.0) or np.any(right_frequency <= 0.0)
            or left_center.shape != (left_frame.shape[0],)
            or right_center.shape != (right_frame.shape[0],)
            or left.kinetic_coefficient_cm <= 0.0
            or not np.isclose(left.kinetic_coefficient_cm, right.kinetic_coefficient_cm)):
        raise ValueError("harmonic path-channel bases are incompatible")
    raw_map = left_frame.T @ right_frame
    u_matrix, _singular, vh = np.linalg.svd(raw_map)
    duschinsky = u_matrix @ vh
    alpha_left = left_frequency / (2.0 * left.kinetic_coefficient_cm)
    alpha_right = right_frequency / (2.0 * right.kinetic_coefficient_cm)
    precision_left = np.diag(alpha_left)
    precision_right = duschinsky @ np.diag(alpha_right) @ duschinsky.T
    total = precision_left + precision_right
    covariance = np.linalg.inv(total)
    center_left = left_frame.T @ left_center
    center_right = left_frame.T @ right_center
    linear = precision_left @ center_left + precision_right @ center_right
    mean = covariance @ linear
    dimension = len(left_frequency)
    log_ground = (
        np.linalg.slogdet(precision_left)[1] / 4.0
        + np.linalg.slogdet(precision_right)[1] / 4.0
        + dimension * np.log(2.0) / 2.0
        - np.linalg.slogdet(total)[1] / 2.0
    )
    displacement_exponent = -0.5 * (
        center_left @ precision_left @ center_left
        + center_right @ precision_right @ center_right
        - linear @ covariance @ linear
    )
    ground = float(np.exp(log_ground + displacement_exponent))
    rows_left = np.diag(np.sqrt(2.0 * alpha_left))
    rows_right = np.diag(np.sqrt(2.0 * alpha_right)) @ duschinsky.T
    if maximum_quanta < 0:
        raise ValueError("maximum_quanta must be nonnegative")
    active = (np.arange(dimension, dtype=int) if active_modes is None
              else np.asarray(active_modes, dtype=int))
    if (active.ndim != 1 or len(np.unique(active)) != len(active)
            or np.any(active < 0) or np.any(active >= dimension)):
        raise ValueError("active harmonic-channel modes are invalid")

    occupations = harmonic_product_occupations(len(active), maximum_quanta)

    def polynomials(offset: int) -> list[dict[tuple[int, ...], float]]:
        one_mode = []
        for local_mode in range(len(active)):
            form = offset + local_mode
            values = [{(): 1.0}]
            if maximum_quanta:
                values.append({(form,): 1.0})
            for quantum in range(1, maximum_quanta):
                current: dict[tuple[int, ...], float] = {}
                for keys, coefficient in values[quantum].items():
                    extended = tuple(sorted(keys + (form,)))
                    current[extended] = (
                        current.get(extended, 0.0)
                        + coefficient / np.sqrt(quantum + 1.0)
                    )
                factor = np.sqrt(quantum / (quantum + 1.0))
                for keys, coefficient in values[quantum - 1].items():
                    current[keys] = current.get(keys, 0.0) - factor * coefficient
                values.append(current)
            one_mode.append(values)
        result = []
        for state in occupations:
            terms = {(): 1.0}
            for mode, quantum in enumerate(state):
                combined: dict[tuple[int, ...], float] = {}
                for left_keys, left_coefficient in terms.items():
                    for right_keys, right_coefficient in one_mode[mode][quantum].items():
                        keys = tuple(sorted(left_keys + right_keys))
                        combined[keys] = (
                            combined.get(keys, 0.0)
                            + left_coefficient * right_coefficient
                        )
                terms = combined
            result.append(terms)
        return result

    forms = np.vstack((rows_left[active], rows_right[active]))
    form_means = np.concatenate((
        rows_left[active] @ (mean - center_left),
        rows_right[active] @ (mean - center_right),
    ))
    moment_cache: dict[tuple[int, ...], float] = {(): 1.0}

    def gaussian_moment(indices: tuple[int, ...]) -> float:
        if indices in moment_cache:
            return moment_cache[indices]
        first, rest = indices[0], indices[1:]
        value = form_means[first] * gaussian_moment(rest)
        value += float(sum(
            forms[first] @ covariance @ forms[partner]
            * gaussian_moment(rest[:index] + rest[index + 1:])
            for index, partner in enumerate(rest)
        ))
        moment_cache[indices] = value
        return value

    left_polynomials = polynomials(0)
    right_polynomials = polynomials(len(active))
    left_monomials = sorted(
        set().union(*(terms.keys() for terms in left_polynomials)),
        key=lambda keys: (len(keys), keys),
    )
    right_monomials = sorted(
        set().union(*(terms.keys() for terms in right_polynomials)),
        key=lambda keys: (len(keys), keys),
    )
    left_index = {keys: index for index, keys in enumerate(left_monomials)}
    right_index = {keys: index for index, keys in enumerate(right_monomials)}
    left_coefficients = np.zeros((len(left_polynomials), len(left_monomials)))
    right_coefficients = np.zeros((len(right_polynomials), len(right_monomials)))
    for state, terms in enumerate(left_polynomials):
        for keys, coefficient in terms.items():
            left_coefficients[state, left_index[keys]] = coefficient
    for state, terms in enumerate(right_polynomials):
        for keys, coefficient in terms.items():
            right_coefficients[state, right_index[keys]] = coefficient
    moments = np.asarray([
        [gaussian_moment(tuple(sorted(left_keys + right_keys)))
         for right_keys in right_monomials]
        for left_keys in left_monomials
    ])
    return ground * left_coefficients @ moments @ right_coefficients.T


def harmonic_product_occupations(mode_count: int, maximum_quanta: int) -> np.ndarray:
    """Enumerate product-state occupations by total quantum number."""
    if mode_count < 0 or maximum_quanta < 0:
        raise ValueError("mode count and maximum quanta must be nonnegative")
    if mode_count == 0:
        return np.zeros((1, 0), dtype=int)
    states = []
    for total in range(maximum_quanta + 1):
        block = [
            values for values in product(range(total + 1), repeat=mode_count)
            if sum(values) == total
        ]
        states.extend(sorted(block, reverse=True))
    return np.asarray(states, dtype=int)


def cartesian_path_transverse_modes(
    cartesian_hessian: np.ndarray,
    coordinates: np.ndarray,
    masses_amu: np.ndarray,
    path_tangent: np.ndarray,
) -> CartesianPathTransverseModes:
    """Project a Cartesian Hessian into the path-transverse vibrational space.

    ``path_tangent`` is the derivative of Cartesian coordinates with respect
    to any monotonic path parameter.  Its scale and phase do not affect the
    result.  Translation, rotation, and the reaction direction are removed in
    mass-weighted Cartesian space before diagonalization.  This construction
    is isotope general: the electronic Hessian is unchanged while the mass
    metric, rigid subspace, and reaction direction are rebuilt for each
    isotopologue.
    """

    geometry = np.asarray(coordinates, dtype=float)
    masses = np.asarray(masses_amu, dtype=float).reshape(-1)
    tangent = np.asarray(path_tangent, dtype=float)
    if geometry.shape != (len(masses), 3) or tangent.shape != geometry.shape:
        raise ValueError("coordinates and path tangent must have shape (natoms, 3)")
    if np.any(masses <= 0.0) or not all(
        np.all(np.isfinite(value)) for value in (geometry, masses, tangent)
    ):
        raise ValueError("path geometry, tangent, and masses must be finite and positive")

    from matrix_qm import mass_weighted_cartesian_hessian

    weighted_hessian = mass_weighted_cartesian_hessian(cartesian_hessian, masses)
    sqrt_mass = np.sqrt(masses)
    center = np.average(geometry, axis=0, weights=masses)
    shifted = geometry - center
    constraints: list[np.ndarray] = []
    for axis in range(3):
        translation = np.zeros_like(geometry)
        translation[:, axis] = sqrt_mass
        constraints.append(translation.reshape(-1))
    for axis in np.eye(3):
        constraints.append((np.cross(axis, shifted) * sqrt_mass[:, None]).reshape(-1))
    weighted_tangent = (tangent * sqrt_mass[:, None]).reshape(-1)
    if float(np.linalg.norm(weighted_tangent)) <= 1.0e-14:
        raise ValueError("path tangent has zero mass-weighted norm")
    constraints.append(weighted_tangent)
    constraint_matrix = np.column_stack(constraints)
    u_matrix, singular_values, _vh = np.linalg.svd(constraint_matrix, full_matrices=False)
    threshold = np.finfo(float).eps * max(constraint_matrix.shape) * singular_values[0]
    rank = int(np.count_nonzero(singular_values > threshold))
    rigid_rank = rank - 1
    if rigid_rank not in {5, 6}:
        raise ValueError(f"unexpected translation/rotation rank {rigid_rank}")
    if rank >= weighted_hessian.shape[0]:
        raise ValueError("reaction path leaves no transverse vibrational coordinates")
    _u, _s, vh = np.linalg.svd(u_matrix[:, :rank].T, full_matrices=True)
    transverse = vh[rank:, :].T
    projected = transverse.T @ weighted_hessian @ transverse
    eigenvalues, vectors = np.linalg.eigh(0.5 * (projected + projected.T))
    order = np.argsort(eigenvalues)
    eigenvalues = np.asarray(eigenvalues[order], dtype=float)
    transverse_modes = transverse @ vectors[:, order]
    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
    frequencies *= HESSIAN_EIGENVALUE_TO_CM
    negative = int(np.count_nonzero(eigenvalues < 0.0))
    zpe = 0.5 * float(np.sum(frequencies[frequencies > 0.0]))
    return CartesianPathTransverseModes(
        transverse_basis=transverse_modes,
        eigenvalues=eigenvalues,
        frequencies_cm=frequencies,
        zero_point_energy_cm=zpe,
        external_rank=rigid_rank,
        negative_mode_count=negative,
    )


@dataclass(frozen=True)
class SonicReactiveModes:
    """Hessian modes inside a SONIC block complementary to conserved rows."""

    conserved_count: int
    conserved_indices: tuple[int, ...]
    reactive_indices: tuple[int, ...]
    reactive_basis: np.ndarray
    sonic_from_modes: np.ndarray
    eigenvalues: np.ndarray
    frequencies_cm: np.ndarray
    transition_mode_significance: np.ndarray
    reaction_subspace_coverage: float
    reaction_direction_coverages: tuple[float, ...]
    reaction_direction_curvatures: tuple[float, ...]
    projected_reaction_direction_curvatures: tuple[float | None, ...]
    reactive_instability_captured: bool
    priority_order: np.ndarray
    conserved_couplings: np.ndarray
    normalized_conserved_coupling_by_mode: np.ndarray
    reactive_off_diagonal_norm_before: float
    reactive_off_diagonal_norm_after: float
    conserved_coupling_norm: float
    conserved_coupling_invariance_residual: float
    metric_orthogonality_residual: float
    full_spectrum_residual: float
    reactive_metric_condition: float
    degenerate_blocks: tuple[tuple[int, ...], ...]
    reference_subspace_overlaps: tuple[float, ...]


def curvilinear_sonic_force_constants(
    cartesian_hessian: np.ndarray,
    cartesian_gradient: np.ndarray,
    coordinates: np.ndarray,
    evaluate_b_matrix: Callable[[np.ndarray], np.ndarray],
    *,
    masses: np.ndarray | None = None,
    internal_step: float = 1.0e-4,
    stationary_gradient_tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Transform a Cartesian Hessian at a non-stationary point to SONIC space.

    The derivative of the contravariant back-transform is retained by central
    finite differences of the internal gradient.  This is the gradient-times-
    coordinate-curvature term that distinguishes curvilinear and rectilinear
    generalized normal modes away from stationary points.  When the Cartesian
    gradient vanishes, the routine returns the ordinary Wilson transformation
    exactly, so the two definitions coincide at a stationary point.

    Coordinates, gradients, Hessians, and the Wilson B matrix must use one
    consistent Cartesian length unit.  ``masses`` may contain one value per
    atom or one value per Cartesian component.
    """

    coords = np.asarray(coordinates, dtype=float)
    shape = coords.shape
    flat = coords.reshape(-1)
    gradient = np.asarray(cartesian_gradient, dtype=float).reshape(-1)
    hessian = np.asarray(cartesian_hessian, dtype=float)
    if gradient.shape != flat.shape or hessian.shape != (flat.size, flat.size):
        raise ValueError("Cartesian coordinates, gradient, and Hessian have incompatible sizes")
    if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(hessian)):
        raise ValueError("Cartesian derivatives must be finite")
    step = float(internal_step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("internal finite-difference step must be positive")
    inverse_mass = _cartesian_inverse_mass(masses, flat.size)
    b_matrix = _validated_b_matrix(evaluate_b_matrix(coords), flat.size)
    backtransform = _wilson_backtransform(b_matrix, inverse_mass)
    ordinary = backtransform.T @ hessian @ backtransform
    ordinary = 0.5 * (ordinary + ordinary.T)
    if float(np.linalg.norm(gradient)) <= stationary_gradient_tolerance:
        return ordinary

    internal_gradient = backtransform.T @ gradient
    result = np.empty_like(ordinary)
    for column in range(b_matrix.shape[0]):
        displacement = step * backtransform[:, column]
        plus_coords = (flat + displacement).reshape(shape)
        minus_coords = (flat - displacement).reshape(shape)
        plus_b = _validated_b_matrix(evaluate_b_matrix(plus_coords), flat.size)
        minus_b = _validated_b_matrix(evaluate_b_matrix(minus_coords), flat.size)
        if plus_b.shape != b_matrix.shape or minus_b.shape != b_matrix.shape:
            raise ValueError("SONIC B-matrix dimension changed along curvilinear displacement")
        plus_backtransform = _wilson_backtransform(plus_b, inverse_mass)
        minus_backtransform = _wilson_backtransform(minus_b, inverse_mass)
        plus_gradient = gradient + hessian @ displacement
        minus_gradient = gradient - hessian @ displacement
        plus_internal_gradient = plus_backtransform.T @ plus_gradient
        minus_internal_gradient = minus_backtransform.T @ minus_gradient
        result[:, column] = (plus_internal_gradient - minus_internal_gradient) / (2.0 * step)
    result = 0.5 * (result + result.T)
    if not np.all(np.isfinite(result)):
        raise ValueError("curvilinear SONIC Hessian contains non-finite values")
    # Retain the variable to make the differentiated quantity explicit and to
    # guard against a future accidental replacement by B H B^T.
    del internal_gradient
    return result


def sonic_qim_direction(
    force_constants: np.ndarray,
    g_matrix: np.ndarray,
    *,
    require_first_order_saddle: bool = True,
) -> SonicQimDirection:
    """Return the rectilinear imaginary-mode direction in SONIC coordinates.

    The returned vector ``l`` obeys ``l.T @ inv(G) @ l = 1``.  Consequently a
    scalar displacement along it is a mass-scaled Wilson normal coordinate,
    as required by the Bowman Qim construction.
    """

    f_mat, g_mat, g_half, _g_inverse_half = _validated_fg(force_constants, g_matrix)
    symmetric = g_half @ f_mat @ g_half
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (symmetric + symmetric.T))
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    negative_count = int(np.count_nonzero(eigenvalues < 0.0))
    if negative_count == 0:
        raise ValueError("Qim construction requires an imaginary saddle-point mode")
    if require_first_order_saddle and negative_count != 1:
        raise ValueError(
            f"Qim construction requires a first-order saddle, found {negative_count} negative modes"
        )
    eigenvalue = float(eigenvalues[0])
    direction = g_half @ eigenvectors[:, 0]
    direction = _canonical_phase(direction)
    constraint = np.linalg.solve(g_mat, direction)
    frequency = -float(np.sqrt(abs(eigenvalue)) * HESSIAN_EIGENVALUE_TO_CM)
    return SonicQimDirection(
        eigenvalue=eigenvalue,
        frequency_cm=frequency,
        sonic_direction=direction,
        constraint_covector=constraint,
        negative_mode_count=negative_count,
    )


def sonic_transverse_modes(
    force_constants: np.ndarray,
    g_matrix: np.ndarray,
    path_tangent: np.ndarray,
    *,
    gradient: np.ndarray | None = None,
) -> SonicTransverseModes:
    """Diagonalize a curvilinear SONIC Hessian in the path-transverse subspace.

    ``force_constants`` must include the gradient-dependent coordinate-
    curvature term at a non-stationary point, as returned by
    :func:`curvilinear_sonic_force_constants`.  ``gradient`` is the covariant
    derivative ``dE/dq``.  Its transverse
    fraction is zero on a perfectly relaxed path and therefore supplies a
    direct convergence diagnostic at non-stationary points.
    """

    f_mat, g_mat, g_half, g_inverse_half = _validated_fg(force_constants, g_matrix)
    tangent = np.asarray(path_tangent, dtype=float).reshape(-1)
    if tangent.shape != (f_mat.shape[0],) or not np.all(np.isfinite(tangent)):
        raise ValueError("path tangent must be a finite vector in the SONIC basis")
    metric_tangent = g_inverse_half @ tangent
    norm = float(np.linalg.norm(metric_tangent))
    if norm <= 1.0e-14:
        raise ValueError("path tangent has zero Wilson-metric norm")
    metric_tangent /= norm
    tangent = g_half @ metric_tangent
    tangent = _canonical_phase(tangent)
    metric_tangent = g_inverse_half @ tangent

    # Complete QR gives an orthonormal complement in mass-scaled SONIC space.
    q_matrix, _ = np.linalg.qr(metric_tangent[:, None], mode="complete")
    transverse_orthogonal = q_matrix[:, 1:]
    transverse_basis = g_half @ transverse_orthogonal
    projected = transverse_basis.T @ f_mat @ transverse_basis
    projected = 0.5 * (projected + projected.T)
    eigenvalues, vectors = np.linalg.eigh(projected)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    vectors = vectors[:, order]
    modes = transverse_basis @ vectors
    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
    frequencies *= HESSIAN_EIGENVALUE_TO_CM

    tangent_curvature = float(tangent @ f_mat @ tangent)
    cross = tangent @ f_mat @ modes
    transverse_curvatures = np.diag(modes.T @ f_mat @ modes)
    denominators = np.sqrt(np.abs(tangent_curvature * transverse_curvatures))
    normalized_cross = np.divide(
        np.abs(cross),
        denominators,
        out=np.zeros_like(cross),
        where=denominators > 1.0e-14,
    )

    transverse_fraction: float | None = None
    if gradient is not None:
        covariant_gradient = np.asarray(gradient, dtype=float).reshape(-1)
        if covariant_gradient.shape != tangent.shape or not np.all(np.isfinite(covariant_gradient)):
            raise ValueError("gradient must be a finite covariant SONIC vector")
        orthogonal_gradient = g_half @ covariant_gradient
        total = float(np.linalg.norm(orthogonal_gradient))
        transverse = transverse_orthogonal.T @ orthogonal_gradient
        transverse_fraction = float(np.linalg.norm(transverse) / max(total, 1.0e-30))

    return SonicTransverseModes(
        path_tangent=tangent,
        transverse_basis=transverse_basis,
        projected_force_constants=projected,
        eigenvalues=eigenvalues,
        frequencies_cm=frequencies,
        sonic_modes=modes,
        gradient_transverse_fraction=transverse_fraction,
        maximum_path_hessian_coupling=(
            float(np.max(normalized_cross)) if normalized_cross.size else 0.0
        ),
    )


def sonic_reactive_modes(
    force_constants: np.ndarray,
    g_matrix: np.ndarray,
    conserved_coordinates: int | tuple[int, ...] | list[int] | np.ndarray,
    *,
    transition_modes: int | tuple[int, ...] | list[int] | np.ndarray = 0,
    reaction_directions: np.ndarray | None = None,
    reference_reactive_basis: np.ndarray | None = None,
    degeneracy_absolute_tolerance: float = 1.0e-10,
    degeneracy_relative_tolerance: float | None = None,
) -> SonicReactiveModes:
    """Diagonalize a reactive SONIC complement while retaining conserved rows.

    ``conserved_coordinates`` may be the number of leading conserved rows or
    their explicit indices. The complementary block is transformed to a
    Wilson-metric-orthonormal Hessian basis without discarding any direction.
    One or more Hessian modes, or explicit reaction directions, define mode
    significance. Numerically degenerate blocks are Procrustes-aligned to a
    reference gauge, making the result deterministic and trackable.
    """

    f_mat, g_mat, g_half, _g_inverse_half = _validated_fg(force_constants, g_matrix)
    size = f_mat.shape[0]
    conserved = _coordinate_indices(conserved_coordinates, size)
    conserved_set = frozenset(conserved)
    reactive = tuple(index for index in range(size) if index not in conserved_set)
    if not reactive:
        raise ValueError("conserved coordinates must leave a non-empty reactive block")
    mode_indices = _coordinate_indices(transition_modes, size, scalar_is_count=False)

    kinetic_metric = np.linalg.solve(g_mat, np.eye(size, dtype=float))
    reactive_kinetic = kinetic_metric[np.ix_(reactive, reactive)]
    reactive_g_half, reactive_g_inverse_half, metric_condition = (
        _positive_definite_inverse_metric_halves(reactive_kinetic)
    )
    reactive_f = f_mat[np.ix_(reactive, reactive)]
    metric_normalized_hessian = reactive_g_half @ reactive_f @ reactive_g_half
    metric_normalized_hessian = 0.5 * (metric_normalized_hessian + metric_normalized_hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(metric_normalized_hessian)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    absolute_tolerance = float(degeneracy_absolute_tolerance)
    relative_tolerance = (
        float(np.sqrt(np.finfo(float).eps))
        if degeneracy_relative_tolerance is None
        else float(degeneracy_relative_tolerance)
    )
    if (
        not np.isfinite(absolute_tolerance)
        or not np.isfinite(relative_tolerance)
        or absolute_tolerance < 0.0
        or relative_tolerance < 0.0
    ):
        raise ValueError("degeneracy tolerances must be finite and non-negative")
    reference_orthonormal = _reference_reactive_orthonormal_basis(
        reference_reactive_basis,
        reactive_g_inverse_half,
        reactive_kinetic,
    )
    full_symmetric = g_half @ f_mat @ g_half
    full_symmetric = 0.5 * (full_symmetric + full_symmetric.T)
    full_values, full_vectors = np.linalg.eigh(full_symmetric)
    full_order = np.argsort(full_values)
    if reaction_directions is None:
        directions = g_half @ full_vectors[:, full_order[np.asarray(mode_indices, dtype=int)]]
    else:
        directions = np.asarray(reaction_directions, dtype=float)
        if directions.ndim == 1:
            directions = directions[:, None]
        if (
            directions.ndim != 2
            or directions.shape[0] != size
            or directions.shape[1] == 0
            or not np.all(np.isfinite(directions))
        ):
            raise ValueError("reaction directions must be finite full-SONIC columns")
        directions = _metric_orthonormal_columns(directions, kinetic_metric)
    reaction_anchors = (
        reactive_g_half
        @ kinetic_metric[np.ix_(reactive, tuple(range(size)))]
        @ directions
    )
    reaction_curvatures = np.diag(directions.T @ f_mat @ directions)
    direction_coverages_unclipped = np.sum(reaction_anchors * reaction_anchors, axis=0)
    curvature_floor = (
        np.finfo(float).eps
        * max(size, 1)
        * max(float(np.linalg.norm(full_symmetric, ord=2)), 1.0)
    )
    projected_reaction_curvatures: list[float | None] = []
    reactive_instability_captured = True
    for direction_index, coverage in enumerate(direction_coverages_unclipped):
        if float(coverage) <= np.finfo(float).eps * max(len(reactive), 1):
            projected_curvature = None
        else:
            anchor = reaction_anchors[:, direction_index]
            projected_curvature = float(
                anchor @ metric_normalized_hessian @ anchor / float(coverage)
            )
        projected_reaction_curvatures.append(projected_curvature)
        if float(reaction_curvatures[direction_index]) < -curvature_floor and (
            projected_curvature is None
            or projected_curvature >= -curvature_floor
        ):
            reactive_instability_captured = False
    if conserved:
        conserved_kinetic = kinetic_metric[np.ix_(conserved, conserved)]
        conserved_basis, _conserved_metric_half, _conserved_condition = (
            _positive_definite_inverse_metric_halves(conserved_kinetic)
        )
        dimensionless_coupling_operator = (
            conserved_basis.T
            @ f_mat[np.ix_(conserved, reactive)]
            @ reactive_g_half
        )
    else:
        conserved_basis = np.zeros((0, 0), dtype=float)
        dimensionless_coupling_operator = np.zeros(
            (0, len(reactive)),
            dtype=float,
        )
    degenerate_blocks = _eigenvalue_blocks(
        eigenvalues,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    eigenvectors, subspace_overlaps = _orient_degenerate_eigenvectors(
        eigenvectors,
        degenerate_blocks,
        reaction_anchors,
        dimensionless_coupling_operator,
        reference_orthonormal,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    reactive_basis = reactive_g_half @ eigenvectors
    for column in range(reactive_basis.shape[1]):
        reactive_basis[:, column] = _canonical_phase(reactive_basis[:, column])

    transform = np.eye(size, dtype=float)
    transform[np.ix_(reactive, reactive)] = reactive_basis
    transformed_f = transform.T @ f_mat @ transform
    transformed_reactive = transformed_f[np.ix_(reactive, reactive)]

    reactive_coefficients = (
        reactive_basis.T
        @ kinetic_metric[np.ix_(reactive, tuple(range(size)))]
        @ directions
    )
    squared = np.sum(reactive_coefficients * reactive_coefficients, axis=1)
    direction_coverages = np.sum(reactive_coefficients * reactive_coefficients, axis=0)
    direction_coverages = np.clip(direction_coverages, 0.0, 1.0)
    significance = squared / max(float(np.sum(squared)), 1.0e-30)

    cross = f_mat[np.ix_(conserved, reactive)] @ reactive_basis
    if conserved:
        dimensionless_cross = dimensionless_coupling_operator @ eigenvectors
        conserved_curvature = (
            conserved_basis.T
            @ f_mat[np.ix_(conserved, conserved)]
            @ conserved_basis
        )
        conserved_scale = max(
            float(np.linalg.norm(conserved_curvature, ord=2)),
            np.finfo(float).eps,
        )
        coupling_denominator = np.sqrt(
            np.maximum(
                np.abs(eigenvalues) * conserved_scale,
                np.finfo(float).eps**2,
            )
        )
        normalized_coupling = (
            np.linalg.norm(dimensionless_cross, axis=0) / coupling_denominator
        )
    else:
        normalized_coupling = np.zeros(len(eigenvalues), dtype=float)
    priority = np.asarray(
        sorted(
            range(len(eigenvalues)),
            key=lambda index: (
                -float(significance[index]),
                float(normalized_coupling[index]),
                float(eigenvalues[index]),
            ),
        ),
        dtype=int,
    )
    reference_cross = f_mat[np.ix_(conserved, reactive)] @ reactive_g_half
    reference_norm = float(np.linalg.norm(reference_cross))
    coupling_norm = float(np.linalg.norm(cross))
    transformed_kinetic = transform.T @ kinetic_metric @ transform
    transformed_g = np.linalg.solve(transformed_kinetic, np.eye(size, dtype=float))
    _tf, _tg, transformed_g_half, _transformed_g_inverse_half = _validated_fg(
        transformed_f, transformed_g
    )
    transformed_symmetric = transformed_g_half @ transformed_f @ transformed_g_half
    transformed_values = np.linalg.eigvalsh(0.5 * (transformed_symmetric + transformed_symmetric.T))
    return SonicReactiveModes(
        conserved_count=len(conserved),
        conserved_indices=conserved,
        reactive_indices=reactive,
        reactive_basis=reactive_basis,
        sonic_from_modes=transform,
        eigenvalues=eigenvalues,
        frequencies_cm=(
            np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * HESSIAN_EIGENVALUE_TO_CM
        ),
        transition_mode_significance=significance,
        reaction_subspace_coverage=float(np.mean(direction_coverages)),
        reaction_direction_coverages=tuple(
            float(value) for value in direction_coverages
        ),
        reaction_direction_curvatures=tuple(
            float(value) for value in reaction_curvatures
        ),
        projected_reaction_direction_curvatures=tuple(projected_reaction_curvatures),
        reactive_instability_captured=reactive_instability_captured,
        priority_order=priority,
        conserved_couplings=cross,
        normalized_conserved_coupling_by_mode=normalized_coupling,
        reactive_off_diagonal_norm_before=float(
            np.linalg.norm(metric_normalized_hessian - np.diag(np.diag(metric_normalized_hessian)))
        ),
        reactive_off_diagonal_norm_after=float(
            np.linalg.norm(transformed_reactive - np.diag(np.diag(transformed_reactive)))
        ),
        conserved_coupling_norm=coupling_norm,
        conserved_coupling_invariance_residual=abs(coupling_norm - reference_norm),
        metric_orthogonality_residual=float(
            np.linalg.norm(
                reactive_basis.T @ reactive_kinetic @ reactive_basis
                - np.eye(len(reactive), dtype=float)
            )
        ),
        full_spectrum_residual=float(
            np.max(np.abs(np.sort(full_values) - np.sort(transformed_values)))
        ),
        reactive_metric_condition=metric_condition,
        degenerate_blocks=degenerate_blocks,
        reference_subspace_overlaps=subspace_overlaps,
    )


def _coordinate_indices(
    value: int | tuple[int, ...] | list[int] | np.ndarray,
    size: int,
    *,
    scalar_is_count: bool = True,
) -> tuple[int, ...]:
    if isinstance(value, (int, np.integer)):
        integer = int(value)
        indices = tuple(range(integer)) if scalar_is_count else (integer,)
    else:
        indices = tuple(int(index) for index in np.asarray(value, dtype=int).reshape(-1))
    if (not scalar_is_count and not indices) or len(set(indices)) != len(indices):
        raise ValueError("coordinate indices must be non-empty and unique")
    if any(index < 0 or index >= size for index in indices):
        raise ValueError("coordinate index is outside the SONIC Hessian")
    return indices


def _positive_definite_inverse_metric_halves(
    kinetic_metric: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    matrix = np.asarray(kinetic_metric, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    scale = max(float(np.max(np.abs(values), initial=0.0)), 1.0)
    floor = np.finfo(float).eps * max(matrix.shape) * scale
    if np.any(values <= floor):
        raise ValueError("reactive kinetic metric is not numerically positive definite")
    inverse_half = (vectors * (1.0 / np.sqrt(values))) @ vectors.T
    half = (vectors * np.sqrt(values)) @ vectors.T
    return inverse_half, half, float(values[-1] / values[0])


def _reference_reactive_orthonormal_basis(
    reference: np.ndarray | None,
    reactive_g_inverse_half: np.ndarray,
    reactive_kinetic: np.ndarray,
) -> np.ndarray:
    if reference is None:
        return np.eye(reactive_kinetic.shape[0], dtype=float)
    basis = np.asarray(reference, dtype=float)
    size = reactive_kinetic.shape[0]
    if basis.shape != (size, size) or not np.all(np.isfinite(basis)):
        raise ValueError("reference reactive basis must be a finite square matrix")
    orthonormal = reactive_g_inverse_half @ basis
    gram = orthonormal.T @ orthonormal
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    if np.any(values <= np.finfo(float).eps * max(size, 1)):
        raise ValueError("reference reactive basis is rank deficient")
    return orthonormal @ (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def _eigenvalue_blocks(
    eigenvalues: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[tuple[int, ...], ...]:
    values = np.asarray(eigenvalues, dtype=float).reshape(-1)
    if not values.size:
        return ()
    scale = max(float(np.max(np.abs(values))), 1.0)
    tolerance = max(absolute_tolerance, relative_tolerance * scale)
    blocks: list[tuple[int, ...]] = []
    start = 0
    for index in range(1, len(values)):
        if abs(float(values[index] - values[index - 1])) > tolerance:
            blocks.append(tuple(range(start, index)))
            start = index
    blocks.append(tuple(range(start, len(values))))
    return tuple(blocks)


def _orient_degenerate_eigenvectors(
    eigenvectors: np.ndarray,
    blocks: tuple[tuple[int, ...], ...],
    reaction_anchors: np.ndarray,
    conserved_coupling_operator: np.ndarray,
    reference: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Choose a physically ordered gauge inside exactly degenerate blocks.

    Hessian diagonalization fixes only the subspace of a degenerate block.
    Within that freedom, the gauge first diagonalizes the projector onto the
    supplied reaction subspace (largest content first), then the squared
    dimensionless conserved/reactive coupling (smallest first).  Procrustes
    alignment to the reference is restricted to the remaining tied subspaces,
    so continuity cannot undo either physical criterion.
    """

    aligned = np.array(eigenvectors, dtype=float, copy=True)
    overlaps: list[float] = []
    for block in blocks:
        indices = np.asarray(block, dtype=int)
        current = aligned[:, indices]
        overlaps.append(_subspace_reference_overlap(current, reference))
        if len(indices) == 1:
            continue

        projected_reaction = current.T @ reaction_anchors
        reaction_projector = projected_reaction @ projected_reaction.T
        reaction_values, reaction_vectors = np.linalg.eigh(
            0.5 * (reaction_projector + reaction_projector.T)
        )
        reaction_order = np.argsort(reaction_values)[::-1]
        reaction_values = reaction_values[reaction_order]
        current = current @ reaction_vectors[:, reaction_order]

        reaction_blocks = _eigenvalue_blocks(
            reaction_values,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        for reaction_block in reaction_blocks:
            local = np.asarray(reaction_block, dtype=int)
            if len(local) == 1:
                continue
            tied = current[:, local]
            projected_coupling = conserved_coupling_operator @ tied
            coupling_gram = projected_coupling.T @ projected_coupling
            coupling_values, coupling_vectors = np.linalg.eigh(
                0.5 * (coupling_gram + coupling_gram.T)
            )
            coupling_order = np.argsort(coupling_values)
            coupling_values = coupling_values[coupling_order]
            tied = tied @ coupling_vectors[:, coupling_order]

            coupling_blocks = _eigenvalue_blocks(
                coupling_values,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            for coupling_block in coupling_blocks:
                residual = np.asarray(coupling_block, dtype=int)
                if len(residual) > 1:
                    tied[:, residual] = _procrustes_subspace_to_reference(
                        tied[:, residual],
                        reference,
                    )
            current[:, local] = tied
        aligned[:, indices] = current
    return aligned, tuple(overlaps)


def _subspace_reference_overlap(current: np.ndarray, reference: np.ndarray) -> float:
    weights = np.linalg.norm(current.T @ reference, axis=0)
    anchors = np.argsort(-weights, kind="stable")[: current.shape[1]]
    target, _triangular = np.linalg.qr(reference[:, anchors], mode="reduced")
    singular = np.linalg.svd(current.T @ target, compute_uv=False)
    return float(np.min(singular)) if singular.size else 1.0


def _procrustes_subspace_to_reference(
    current: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    weights = np.linalg.norm(current.T @ reference, axis=0)
    anchors = np.argsort(-weights, kind="stable")[: current.shape[1]]
    target, triangular = np.linalg.qr(reference[:, anchors], mode="reduced")
    target *= np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
    aligned_rows, _singular = procrustes_align_salc_matrix(target.T, current.T)
    return aligned_rows.T


def _metric_orthonormal_columns(
    directions: np.ndarray,
    kinetic_metric: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(directions, dtype=float)
    gram = matrix.T @ kinetic_metric @ matrix
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    floor = (
        np.finfo(float).eps * max(gram.shape) * max(float(np.max(np.abs(values), initial=0.0)), 1.0)
    )
    if np.any(values <= floor):
        raise ValueError("reaction directions are linearly dependent in the Wilson metric")
    return matrix @ (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def _validated_fg(
    force_constants: np.ndarray, g_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f_mat = np.asarray(force_constants, dtype=float)
    g_mat = np.asarray(g_matrix, dtype=float)
    if f_mat.ndim != 2 or f_mat.shape[0] != f_mat.shape[1] or f_mat.shape != g_mat.shape:
        raise ValueError("F and G must be square matrices with the same shape")
    if not np.all(np.isfinite(f_mat)) or not np.all(np.isfinite(g_mat)):
        raise ValueError("F and G must contain only finite values")
    f_mat = 0.5 * (f_mat + f_mat.T)
    g_mat = 0.5 * (g_mat + g_mat.T)
    values, vectors = np.linalg.eigh(g_mat)
    if np.any(values <= 0.0):
        raise ValueError("G matrix must be positive definite")
    g_half = (vectors * np.sqrt(values)) @ vectors.T
    g_inverse_half = (vectors * (1.0 / np.sqrt(values))) @ vectors.T
    return f_mat, g_mat, g_half, g_inverse_half


def _canonical_phase(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=float).copy()
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0.0:
        result *= -1.0
    return result


def _cartesian_inverse_mass(masses: np.ndarray | None, size: int) -> np.ndarray:
    if masses is None:
        return np.ones(size, dtype=float)
    values = np.asarray(masses, dtype=float).reshape(-1)
    if values.size * 3 == size:
        values = np.repeat(values, 3)
    if values.size != size or np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("masses must be positive and match atoms or Cartesian components")
    return 1.0 / values


def _validated_b_matrix(b_matrix: np.ndarray, cartesian_size: int) -> np.ndarray:
    matrix = np.asarray(b_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != cartesian_size:
        raise ValueError("SONIC B matrix has incompatible Cartesian dimension")
    if not np.all(np.isfinite(matrix)) or np.linalg.matrix_rank(matrix) != matrix.shape[0]:
        raise ValueError("SONIC B matrix must be finite and have independent rows")
    return matrix


def _wilson_backtransform(b_matrix: np.ndarray, inverse_mass: np.ndarray) -> np.ndarray:
    metric = (b_matrix * inverse_mass[None, :]) @ b_matrix.T
    return (inverse_mass[:, None] * b_matrix.T) @ np.linalg.inv(metric)
