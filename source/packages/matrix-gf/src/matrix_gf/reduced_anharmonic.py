"""Reduced anharmonic potentials along curvilinear reaction paths."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np


def total_degree_powers(dimension: int, maximum_degree: int) -> tuple[tuple[int, ...], ...]:
    """Return nonconstant monomial powers in deterministic total-degree order."""
    if dimension < 1 or maximum_degree < 1:
        raise ValueError("polynomial dimension and maximum degree must be positive")
    powers = [
        values for values in product(range(maximum_degree + 1), repeat=dimension)
        if 1 <= sum(values) <= maximum_degree
    ]
    return tuple(sorted(powers, key=lambda values: (sum(values), values)))


@dataclass(frozen=True)
class ReducedAnharmonicPathSurface:
    """Polynomial correction to local harmonic transverse potentials."""

    path_coordinates: np.ndarray
    active_modes: np.ndarray
    coordinate_symmetry: np.ndarray
    powers: tuple[tuple[int, ...], ...]
    coefficients_cm: np.ndarray
    fit_rms_cm: np.ndarray
    fit_maximum_error_cm: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "matrix.lock.reduced_anharmonic_path_surface.v1",
            "path_coordinates_sqrt_amu_angstrom": self.path_coordinates.tolist(),
            "active_transverse_modes_one_based": (self.active_modes + 1).tolist(),
            "coordinate_symmetry": self.coordinate_symmetry.tolist(),
            "powers": [list(values) for values in self.powers],
            "coefficients_cm-1": self.coefficients_cm.tolist(),
            "fit_rms_cm-1": self.fit_rms_cm.tolist(),
            "fit_maximum_error_cm-1": self.fit_maximum_error_cm.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ReducedAnharmonicPathSurface:
        if data.get("schema") != "matrix.lock.reduced_anharmonic_path_surface.v1":
            raise ValueError("unsupported reduced anharmonic surface schema")
        return cls(
            path_coordinates=np.asarray(
                data["path_coordinates_sqrt_amu_angstrom"], dtype=float,
            ),
            active_modes=np.asarray(
                data["active_transverse_modes_one_based"], dtype=int,
            ) - 1,
            coordinate_symmetry=np.asarray(data["coordinate_symmetry"], dtype=int),
            powers=tuple(
                tuple(int(value) for value in values) for values in data["powers"]
            ),
            coefficients_cm=np.asarray(data["coefficients_cm-1"], dtype=float),
            fit_rms_cm=np.asarray(data["fit_rms_cm-1"], dtype=float),
            fit_maximum_error_cm=np.asarray(
                data["fit_maximum_error_cm-1"], dtype=float,
            ),
        )


def fit_reduced_anharmonic_path_surface(
    path_coordinates: Sequence[float],
    dimensionless_points: np.ndarray,
    energies_cm: np.ndarray,
    harmonic_frequencies_cm: np.ndarray,
    active_modes: Sequence[int],
    *,
    coordinate_symmetry: np.ndarray,
    maximum_degree: int = 4,
) -> ReducedAnharmonicPathSurface:
    """Fit a symmetry-adapted correction to the local harmonic potential.

    ``dimensionless_points`` contains the same transverse sampling grid at
    every path image.  The supplied energies may contain an arbitrary common
    zero.  Symmetry-related samples are averaged after applying the inferred
    parity of each transverse coordinate.
    """
    path = np.asarray(path_coordinates, dtype=float)
    points = np.asarray(dimensionless_points, dtype=float)
    energies = np.asarray(energies_cm, dtype=float)
    frequencies = np.asarray(harmonic_frequencies_cm, dtype=float)
    modes = np.asarray(active_modes, dtype=int)
    symmetry = np.asarray(coordinate_symmetry, dtype=int)
    if (path.ndim != 1 or points.ndim != 2
            or energies.shape != (len(path), len(points))
            or frequencies.shape != (len(path), points.shape[1])
            or modes.shape != (points.shape[1],)
            or symmetry.shape != (points.shape[1], points.shape[1])
            or not np.array_equal(symmetry.T @ symmetry, np.eye(points.shape[1]))
            or not np.array_equal(symmetry @ symmetry, np.eye(points.shape[1]))
            or not all(np.all(np.isfinite(value)) for value in (
                path, points, frequencies,
            )) or np.any(np.isinf(energies)) or np.any(frequencies <= 0.0)):
        raise ValueError("reduced anharmonic surface inputs are incompatible")
    if not np.allclose(path, -path[::-1], atol=1.0e-10, rtol=0.0):
        raise ValueError("reaction-path grid must contain symmetry-related pairs")
    zero = np.flatnonzero(np.all(np.isclose(points, 0.0, atol=1.0e-12), axis=1))
    if len(zero) != 1:
        raise ValueError("transverse sampling grid must contain one origin")
    partners = []
    for point in points:
        matches = np.flatnonzero(np.all(
            np.isclose(points, symmetry @ point, atol=1.0e-12), axis=1,
        ))
        if len(matches) != 1:
            raise ValueError("sampling grid is incomplete under path symmetry")
        partners.append(int(matches[0]))
    symmetric = energies.copy()
    for left in range((len(path) + 1) // 2):
        right = len(path) - 1 - left
        for sample, partner in enumerate(partners):
            pair = np.asarray([
                energies[left, sample], energies[right, partner],
            ])
            if np.any(np.isfinite(pair)):
                mean = float(np.nanmean(pair))
                symmetric[left, sample] = mean
                symmetric[right, partner] = mean
    powers = total_degree_powers(points.shape[1], maximum_degree)
    design = np.asarray([
        [float(np.prod(point ** power)) for power in powers]
        for point in points
    ])
    coefficients, rms, maximum = [], [], []
    origin = int(zero[0])
    for index in range(len(path)):
        if not np.isfinite(symmetric[index, origin]):
            raise ValueError("electronic origin is missing after symmetry completion")
        relative = symmetric[index] - symmetric[index, origin]
        harmonic = 0.5 * np.sum(frequencies[index] * points**2, axis=1)
        target = relative - harmonic
        retained = np.isfinite(target)
        fitted, _residuals, rank, _singular = np.linalg.lstsq(
            design[retained], target[retained], rcond=None,
        )
        if rank != len(powers):
            raise ValueError("anharmonic sampling design is rank deficient")
        error = design[retained] @ fitted - target[retained]
        coefficients.append(fitted)
        rms.append(float(np.sqrt(np.mean(error**2))))
        maximum.append(float(np.max(np.abs(error))))
    return ReducedAnharmonicPathSurface(
        path_coordinates=path,
        active_modes=modes,
        coordinate_symmetry=symmetry,
        powers=powers,
        coefficients_cm=np.asarray(coefficients),
        fit_rms_cm=np.asarray(rms),
        fit_maximum_error_cm=np.asarray(maximum),
    )


def fit_reduced_interaction_path_surface(
    path_coordinates: Sequence[float],
    dimensionless_points: np.ndarray,
    interaction_energies_cm: np.ndarray,
    active_modes: Sequence[int],
    *,
    coordinate_symmetry: np.ndarray,
    maximum_degree: int = 4,
) -> ReducedAnharmonicPathSurface:
    """Fit a pure n-mode interaction after lower-order subtraction.

    Every retained monomial contains all active coordinates.  Consequently the
    fitted increment vanishes whenever any coordinate is zero and can be added
    to lower-order one-mode and pair surfaces without double counting.
    """
    path = np.asarray(path_coordinates, dtype=float)
    points = np.asarray(dimensionless_points, dtype=float)
    energies = np.asarray(interaction_energies_cm, dtype=float)
    modes = np.asarray(active_modes, dtype=int)
    symmetry = np.asarray(coordinate_symmetry, dtype=int)
    dimension = points.shape[1] if points.ndim == 2 else 0
    if (path.ndim != 1 or dimension < 2
            or energies.shape != (len(path), len(points))
            or modes.shape != (dimension,)
            or symmetry.shape != (dimension, dimension)
            or not np.array_equal(symmetry.T @ symmetry, np.eye(dimension))
            or not np.array_equal(symmetry @ symmetry, np.eye(dimension))
            or not all(np.all(np.isfinite(value)) for value in (path, points))
            or np.any(np.isinf(energies))):
        raise ValueError("reduced interaction surface inputs are incompatible")
    if not np.allclose(path, -path[::-1], atol=1.0e-10, rtol=0.0):
        raise ValueError("reaction-path grid must contain symmetry-related pairs")
    partners = []
    for point in points:
        matches = np.flatnonzero(np.all(
            np.isclose(points, symmetry @ point, atol=1.0e-12), axis=1,
        ))
        if len(matches) != 1:
            raise ValueError("interaction grid is incomplete under path symmetry")
        partners.append(int(matches[0]))
    symmetric = energies.copy()
    for left in range((len(path) + 1) // 2):
        right = len(path) - 1 - left
        for sample, partner in enumerate(partners):
            pair = np.asarray([
                energies[left, sample], energies[right, partner],
            ])
            if np.any(np.isfinite(pair)):
                mean = float(np.nanmean(pair))
                symmetric[left, sample] = mean
                symmetric[right, partner] = mean
    powers = tuple(
        power for power in total_degree_powers(dimension, maximum_degree)
        if all(value > 0 for value in power)
    )
    design = np.asarray([
        [float(np.prod(point ** power)) for power in powers]
        for point in points
    ])
    coefficients, rms, maximum = [], [], []
    for target in symmetric:
        retained = np.isfinite(target)
        fitted, _residuals, rank, _singular = np.linalg.lstsq(
            design[retained], target[retained], rcond=None,
        )
        if rank != len(powers):
            raise ValueError("interaction sampling design is rank deficient")
        error = design[retained] @ fitted - target[retained]
        coefficients.append(fitted)
        rms.append(float(np.sqrt(np.mean(error**2))))
        maximum.append(float(np.max(np.abs(error))))
    return ReducedAnharmonicPathSurface(
        path_coordinates=path,
        active_modes=modes,
        coordinate_symmetry=symmetry,
        powers=powers,
        coefficients_cm=np.asarray(coefficients),
        fit_rms_cm=np.asarray(rms),
        fit_maximum_error_cm=np.asarray(maximum),
    )


def anharmonic_channel_matrix(
    coefficients_cm: Sequence[float],
    powers: Sequence[Sequence[int]],
    occupations: np.ndarray,
) -> np.ndarray:
    """Project a polynomial correction onto harmonic product states."""
    coefficients = np.asarray(coefficients_cm, dtype=float)
    operators = anharmonic_channel_operators(powers, occupations)
    if coefficients.shape != (len(operators),):
        raise ValueError("anharmonic coefficients do not match polynomial powers")
    result = np.tensordot(coefficients, operators, axes=1)
    return 0.5 * (result + result.T)


def anharmonic_channel_operators(
    powers: Sequence[Sequence[int]],
    occupations: np.ndarray,
) -> np.ndarray:
    """Build reusable monomial matrices in a harmonic product basis."""
    states = np.asarray(occupations, dtype=int)
    power_values = tuple(tuple(int(value) for value in item) for item in powers)
    if (states.ndim != 2 or np.any(states < 0)
            or any(len(item) != states.shape[1] for item in power_values)):
        raise ValueError("anharmonic channel projection inputs are incompatible")
    if not power_values:
        return np.zeros((0, len(states), len(states)))
    maximum_power = max(max(item) for item in power_values)
    maximum_state = int(np.max(states, initial=0))
    size = maximum_state + maximum_power + 1
    position = np.zeros((size, size))
    for state in range(size - 1):
        value = np.sqrt((state + 1) / 2.0)
        position[state, state + 1] = value
        position[state + 1, state] = value
    operators = [np.eye(size)]
    for _degree in range(maximum_power):
        operators.append(operators[-1] @ position)
    result = np.zeros((len(power_values), len(states), len(states)))
    for term, power in enumerate(power_values):
        for row, left in enumerate(states):
            for column, right in enumerate(states):
                result[term, row, column] = float(np.prod([
                    operators[degree][left[mode], right[mode]]
                    for mode, degree in enumerate(power)
                ]))
    return result


def contract_one_mode_anharmonic_ground_states(
    coefficients_cm: np.ndarray,
    powers: Sequence[Sequence[int]],
    frequencies_cm: Sequence[float],
    *,
    maximum_quanta: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Contract a one-mode path surface into local ground states.

    Returns the correction to the harmonic zero-point energy and the local
    harmonic expansion coefficients.  Consecutive eigenvectors are placed in
    a continuous real gauge.
    """
    coefficients = np.asarray(coefficients_cm, dtype=float)
    frequencies = np.asarray(frequencies_cm, dtype=float)
    if (maximum_quanta < 0 or frequencies.ndim != 1
            or coefficients.shape[0] != len(frequencies)
            or np.any(frequencies <= 0.0)):
        raise ValueError("one-mode contraction inputs are incompatible")
    occupations = np.arange(maximum_quanta + 1, dtype=int)[:, None]
    operators = anharmonic_channel_operators(powers, occupations)
    corrections = []
    vectors = []
    previous = None
    for frequency, local_coefficients in zip(
        frequencies, coefficients, strict=True,
    ):
        hamiltonian = np.diag(frequency * (occupations[:, 0] + 0.5))
        hamiltonian += np.tensordot(local_coefficients, operators, axes=1)
        values, eigenvectors = np.linalg.eigh(hamiltonian)
        ground = eigenvectors[:, 0]
        if previous is not None and float(previous @ ground) < 0.0:
            ground = -ground
        corrections.append(float(values[0] - 0.5 * frequency))
        vectors.append(ground)
        previous = ground
    return np.asarray(corrections), np.asarray(vectors)


def contract_pair_interaction_to_active_polynomial(
    pair_coefficients_cm: np.ndarray,
    pair_powers: Sequence[Sequence[int]],
    one_mode_coefficients_cm: np.ndarray,
    one_mode_powers: Sequence[Sequence[int]],
    contracted_frequencies_cm: Sequence[float],
    *,
    active_position: int,
    maximum_quanta: int = 16,
    effective_degree: int = 6,
    active_grid: Sequence[float] = (
        -1.5, -1.125, -0.75, -0.375, 0.0, 0.375, 0.75, 1.125, 1.5,
    ),
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray, np.ndarray]:
    """Adiabatically contract one coordinate of a pure pair interaction.

    The lower-order one-mode Hamiltonian is rediagonalized at every value of
    the retained coordinate.  The returned polynomial is therefore the local
    response of the contracted ground state, rather than a frozen expectation
    value of the pair potential.
    """
    pair_coefficients = np.asarray(pair_coefficients_cm, dtype=float)
    one_coefficients = np.asarray(one_mode_coefficients_cm, dtype=float)
    frequencies = np.asarray(contracted_frequencies_cm, dtype=float)
    grid = np.asarray(active_grid, dtype=float)
    pair_power_values = tuple(tuple(int(value) for value in item)
                              for item in pair_powers)
    one_power_values = tuple(tuple(int(value) for value in item)
                             for item in one_mode_powers)
    if active_position not in {0, 1}:
        raise ValueError("pair contraction active position must be zero or one")
    contracted_position = 1 - active_position
    if (pair_coefficients.ndim != 2
            or pair_coefficients.shape[1] != len(pair_power_values)
            or one_coefficients.shape != (len(pair_coefficients), len(one_power_values))
            or frequencies.shape != (len(pair_coefficients),)
            or any(len(power) != 2 for power in pair_power_values)
            or any(len(power) != 1 for power in one_power_values)
            or np.any(frequencies <= 0.0) or maximum_quanta < 1
            or effective_degree < 1 or len(grid) <= effective_degree
            or not np.any(np.isclose(grid, 0.0, atol=1.0e-12))):
        raise ValueError("pair contraction inputs are incompatible")
    occupations = np.arange(maximum_quanta + 1, dtype=int)[:, None]
    one_operators = anharmonic_channel_operators(one_power_values, occupations)
    contracted_degrees = tuple(sorted({
        power[contracted_position] for power in pair_power_values
    }))
    contracted_operators = anharmonic_channel_operators(
        tuple((degree,) for degree in contracted_degrees), occupations,
    )
    operator_by_degree = dict(zip(contracted_degrees, contracted_operators, strict=True))
    effective_powers = tuple((degree,) for degree in range(1, effective_degree + 1))
    design = np.asarray([[value**degree for (degree,) in effective_powers]
                         for value in grid])
    coefficients, errors = [], []
    for frequency, one_values, pair_values in zip(
        frequencies, one_coefficients, pair_coefficients, strict=True,
    ):
        base = np.diag(frequency * (occupations[:, 0] + 0.5))
        base += np.tensordot(one_values, one_operators, axes=1)
        reference = float(np.linalg.eigvalsh(base)[0])
        effective = []
        for coordinate in grid:
            interaction = np.zeros_like(base)
            for coefficient, power in zip(
                pair_values, pair_power_values, strict=True,
            ):
                interaction += (
                    coefficient * coordinate ** power[active_position]
                    * operator_by_degree[power[contracted_position]]
                )
            effective.append(float(np.linalg.eigvalsh(base + interaction)[0] - reference))
        fitted, _residuals, rank, _singular = np.linalg.lstsq(
            design, effective, rcond=None,
        )
        if rank != len(effective_powers):
            raise ValueError("effective pair-contraction fit is rank deficient")
        error = design @ fitted - effective
        coefficients.append(fitted)
        errors.append(float(np.max(np.abs(error))))
    return effective_powers, np.asarray(coefficients), np.asarray(errors)


def contract_two_mode_interaction_ground_correction(
    pair_coefficients_cm: np.ndarray,
    pair_powers: Sequence[Sequence[int]],
    first_coefficients_cm: np.ndarray,
    first_powers: Sequence[Sequence[int]],
    first_frequencies_cm: Sequence[float],
    second_coefficients_cm: np.ndarray,
    second_powers: Sequence[Sequence[int]],
    second_frequencies_cm: Sequence[float],
    *,
    maximum_quanta: int = 16,
) -> np.ndarray:
    """Contract a pure pair increment between two n-mode coordinates."""
    pair_coefficients = np.asarray(pair_coefficients_cm, dtype=float)
    first_coefficients = np.asarray(first_coefficients_cm, dtype=float)
    second_coefficients = np.asarray(second_coefficients_cm, dtype=float)
    first_frequencies = np.asarray(first_frequencies_cm, dtype=float)
    second_frequencies = np.asarray(second_frequencies_cm, dtype=float)
    pair_power_values = tuple(tuple(int(value) for value in item)
                              for item in pair_powers)
    first_power_values = tuple(tuple(int(value) for value in item)
                               for item in first_powers)
    second_power_values = tuple(tuple(int(value) for value in item)
                                for item in second_powers)
    path_points = len(pair_coefficients)
    if (maximum_quanta < 1 or pair_coefficients.ndim != 2
            or pair_coefficients.shape[1] != len(pair_power_values)
            or first_coefficients.shape != (path_points, len(first_power_values))
            or second_coefficients.shape != (path_points, len(second_power_values))
            or first_frequencies.shape != (path_points,)
            or second_frequencies.shape != (path_points,)
            or any(len(power) != 2 for power in pair_power_values)
            or any(len(power) != 1 for power in first_power_values)
            or any(len(power) != 1 for power in second_power_values)
            or np.any(first_frequencies <= 0.0)
            or np.any(second_frequencies <= 0.0)):
        raise ValueError("two-mode contraction inputs are incompatible")
    occupations = np.arange(maximum_quanta + 1, dtype=int)[:, None]
    first_operators = anharmonic_channel_operators(first_power_values, occupations)
    second_operators = anharmonic_channel_operators(second_power_values, occupations)
    first_degrees = tuple(sorted({power[0] for power in pair_power_values}))
    second_degrees = tuple(sorted({power[1] for power in pair_power_values}))
    first_pair_operators = dict(zip(
        first_degrees,
        anharmonic_channel_operators(
            tuple((degree,) for degree in first_degrees), occupations,
        ),
        strict=True,
    ))
    second_pair_operators = dict(zip(
        second_degrees,
        anharmonic_channel_operators(
            tuple((degree,) for degree in second_degrees), occupations,
        ),
        strict=True,
    ))
    identity = np.eye(maximum_quanta + 1)
    corrections = []
    for index in range(path_points):
        first = np.diag(first_frequencies[index] * (occupations[:, 0] + 0.5))
        first += np.tensordot(first_coefficients[index], first_operators, axes=1)
        second = np.diag(second_frequencies[index] * (occupations[:, 0] + 0.5))
        second += np.tensordot(second_coefficients[index], second_operators, axes=1)
        first_values, first_vectors = np.linalg.eigh(first)
        second_values, second_vectors = np.linalg.eigh(second)
        reference = float(first_values[0] + second_values[0])
        reference_vector = np.kron(first_vectors[:, 0], second_vectors[:, 0])
        hamiltonian = np.kron(first, identity) + np.kron(identity, second)
        for coefficient, power in zip(
            pair_coefficients[index], pair_power_values, strict=True,
        ):
            hamiltonian += coefficient * np.kron(
                first_pair_operators[power[0]],
                second_pair_operators[power[1]],
            )
        values, vectors = np.linalg.eigh(hamiltonian)
        overlaps = np.abs(vectors.T.conj() @ reference_vector) ** 2
        state = int(np.argmax(overlaps))
        corrections.append(float(values[state] - reference))
    return np.asarray(corrections)


def contract_mode_interactions_to_active_surface(
    one_mode_coefficients_cm: np.ndarray,
    one_mode_powers: Sequence[Sequence[int]],
    contracted_frequencies_cm: Sequence[float],
    pair_components: Sequence[
        tuple[np.ndarray, Sequence[Sequence[int]], int, int]
    ],
    *,
    active_dimension: int,
    maximum_quanta: int = 16,
    effective_degree: int = 6,
    active_grid: Sequence[float] = (
        -1.5, -1.125, -0.75, -0.375, 0.0, 0.375, 0.75, 1.125, 1.5,
    ),
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray, np.ndarray]:
    """Contract one mode in the joint field of all retained coordinates."""
    one_coefficients = np.asarray(one_mode_coefficients_cm, dtype=float)
    frequencies = np.asarray(contracted_frequencies_cm, dtype=float)
    one_power_values = tuple(tuple(int(value) for value in item)
                             for item in one_mode_powers)
    grid = np.asarray(active_grid, dtype=float)
    path_points = len(one_coefficients)
    if (active_dimension < 1 or maximum_quanta < 1 or effective_degree < 1
            or frequencies.shape != (path_points,)
            or one_coefficients.shape[1] != len(one_power_values)
            or np.any(frequencies <= 0.0)
            or not np.any(np.isclose(grid, 0.0, atol=1.0e-12))):
        raise ValueError("joint contracted-mode inputs are incompatible")
    normalized_components = []
    contracted_degrees = set()
    for coefficients, powers, active_coordinate, active_position in pair_components:
        coefficient_values = np.asarray(coefficients, dtype=float)
        power_values = tuple(tuple(int(value) for value in item) for item in powers)
        if (coefficient_values.shape != (path_points, len(power_values))
                or any(len(power) != 2 for power in power_values)
                or not 0 <= active_coordinate < active_dimension
                or active_position not in {0, 1}):
            raise ValueError("joint pair component is incompatible")
        contracted_position = 1 - active_position
        contracted_degrees.update(power[contracted_position] for power in power_values)
        normalized_components.append((
            coefficient_values, power_values, active_coordinate,
            active_position, contracted_position,
        ))
    if not normalized_components:
        raise ValueError("joint contraction requires at least one pair component")
    occupations = np.arange(maximum_quanta + 1, dtype=int)[:, None]
    one_operators = anharmonic_channel_operators(one_power_values, occupations)
    degrees = tuple(sorted(contracted_degrees))
    contracted_operators = dict(zip(
        degrees,
        anharmonic_channel_operators(
            tuple((degree,) for degree in degrees), occupations,
        ),
        strict=True,
    ))
    active_points = np.asarray(list(product(grid, repeat=active_dimension)), dtype=float)
    effective_powers = total_degree_powers(active_dimension, effective_degree)
    design = np.asarray([
        [float(np.prod(point ** power)) for power in effective_powers]
        for point in active_points
    ])
    coefficients, errors = [], []
    for path_index, frequency in enumerate(frequencies):
        base = np.diag(frequency * (occupations[:, 0] + 0.5))
        base += np.tensordot(
            one_coefficients[path_index], one_operators, axes=1,
        )
        base_values, base_vectors = np.linalg.eigh(base)
        reference = float(base_values[0])
        reference_vector = base_vectors[:, 0]
        effective = []
        for point in active_points:
            interaction = np.zeros_like(base)
            for (pair_coefficients, power_values, active_coordinate,
                 active_position, contracted_position) in normalized_components:
                coordinate = point[active_coordinate]
                for coefficient, power in zip(
                    pair_coefficients[path_index], power_values, strict=True,
                ):
                    interaction += (
                        coefficient * coordinate ** power[active_position]
                        * contracted_operators[power[contracted_position]]
                    )
            values, vectors = np.linalg.eigh(base + interaction)
            overlaps = np.abs(vectors.T.conj() @ reference_vector) ** 2
            state = int(np.argmax(overlaps))
            effective.append(float(values[state] - reference))
        fitted, _residuals, rank, _singular = np.linalg.lstsq(
            design, effective, rcond=None,
        )
        if rank != len(effective_powers):
            raise ValueError("joint contracted-mode fit is rank deficient")
        error = design @ fitted - effective
        coefficients.append(fitted)
        errors.append(float(np.max(np.abs(error))))
    return effective_powers, np.asarray(coefficients), np.asarray(errors)


def contract_nmode_vscf_to_active_surface(
    one_mode_coefficients_cm: Sequence[np.ndarray],
    one_mode_powers: Sequence[Sequence[Sequence[int]]],
    frequencies_cm: np.ndarray,
    active_pair_components: Sequence[
        tuple[int, np.ndarray, Sequence[Sequence[int]], int, int]
    ],
    contracted_pair_components: Sequence[
        tuple[int, int, np.ndarray, Sequence[Sequence[int]], int]
    ],
    *,
    active_dimension: int,
    maximum_quanta: int = 16,
    effective_degree: int = 4,
    active_grid: Sequence[float] = (
        -1.5, -1.125, -0.75, -0.375, 0.0, 0.375, 0.75, 1.125, 1.5,
    ),
    convergence_tolerance: float = 1.0e-10,
    maximum_iterations: int = 100,
    density_mixing: float = 0.5,
    _return_active_samples: bool = False,
    property_powers: Sequence[Sequence[int]] = (),
) -> tuple:
    """VSCF-contract a pair-truncated n-mode potential onto active modes.

    All contracted coordinates are optimized together, so a coordinate shared
    by several pair surfaces is relaxed once.  Maximum-overlap following keeps
    every modal connected to its lower-order ground state when a polynomial
    continuation creates an unrelated low eigenvector near the basis edge.
    Damped density iteration and nearest-neighbor continuation across the
    active grid stabilize soft or nearly degenerate modal fields.
    Requested coordinate-property moments are accumulated during this same
    iteration; the full grid of modal eigenvectors is not retained. Supplying
    property_powers appends their ordered moment record to the return tuple.
    """
    frequencies = np.asarray(frequencies_cm, dtype=float)
    mode_count = len(one_mode_coefficients_cm)
    if (mode_count < 1 or frequencies.ndim != 2
            or frequencies.shape[1] != mode_count
            or maximum_quanta < 1 or active_dimension < 1
            or effective_degree < 1 or convergence_tolerance <= 0.0
            or maximum_iterations < 1 or not 0.0 < density_mixing <= 1.0):
        raise ValueError("VSCF n-mode contraction inputs are incompatible")
    path_points = frequencies.shape[0]
    coefficients = [np.asarray(value, dtype=float)
                    for value in one_mode_coefficients_cm]
    powers = [tuple(tuple(int(x) for x in item) for item in values)
              for values in one_mode_powers]
    if any(value.shape != (path_points, len(power))
           for value, power in zip(coefficients, powers, strict=True)):
        raise ValueError("VSCF one-mode surfaces do not share the path grid")
    words=tuple(tuple(int(x) for x in word) for word in property_powers)
    if any(len(word)!=mode_count or any(raw!=integer or integer<0 for raw,integer in zip(original,word,strict=True))
           for original,word in zip(property_powers,words,strict=True)):
        raise ValueError("property powers must be nonnegative integer tuples over contracted modes")
    occupations = np.arange(maximum_quanta + 1, dtype=int)[:, None]
    moment_operators={(mode,degree):anharmonic_channel_operators(((degree,),),occupations)[0]
        for mode in range(mode_count) for degree in sorted({word[mode] for word in words if word[mode]})}
    property_samples=[]
    one_operators = [anharmonic_channel_operators(power, occupations)
                     for power in powers]

    active_terms = []
    for mode, values, term_powers, active_coordinate, active_position in (
        active_pair_components
    ):
        values = np.asarray(values, dtype=float)
        term_powers = tuple(tuple(int(x) for x in item) for item in term_powers)
        if (not 0 <= mode < mode_count or not 0 <= active_coordinate < active_dimension
                or active_position not in {0, 1}
                or values.shape != (path_points, len(term_powers))
                or any(len(power) != 2 for power in term_powers)):
            raise ValueError("VSCF active-pair component is incompatible")
        contracted_position = 1 - active_position
        degrees = tuple(sorted({power[contracted_position] for power in term_powers}))
        operators = dict(zip(
            degrees,
            anharmonic_channel_operators(
                tuple((degree,) for degree in degrees), occupations,
            ),
            strict=True,
        ))
        active_terms.append((
            mode, values, term_powers, active_coordinate,
            active_position, contracted_position, operators,
        ))

    pair_terms = []
    for first, second, values, term_powers, first_position in (
        contracted_pair_components
    ):
        values = np.asarray(values, dtype=float)
        term_powers = tuple(tuple(int(x) for x in item) for item in term_powers)
        if (not 0 <= first < mode_count or not 0 <= second < mode_count
                or first == second or first_position not in {0, 1}
                or values.shape != (path_points, len(term_powers))
                or any(len(power) != 2 for power in term_powers)):
            raise ValueError("VSCF contracted-pair component is incompatible")
        second_position = 1 - first_position
        first_degrees = tuple(sorted({power[first_position] for power in term_powers}))
        second_degrees = tuple(sorted({power[second_position] for power in term_powers}))
        first_operators = dict(zip(
            first_degrees,
            anharmonic_channel_operators(
                tuple((degree,) for degree in first_degrees), occupations,
            ), strict=True,
        ))
        second_operators = dict(zip(
            second_degrees,
            anharmonic_channel_operators(
                tuple((degree,) for degree in second_degrees), occupations,
            ), strict=True,
        ))
        pair_terms.append((
            first, second, values, term_powers, first_position, second_position,
            first_operators, second_operators,
        ))

    grid = np.asarray(active_grid, dtype=float)
    active_points = np.asarray(list(product(grid, repeat=active_dimension)), dtype=float)
    zero_matches = np.flatnonzero(np.all(
        np.isclose(active_points, 0.0, atol=1.0e-12), axis=1,
    ))
    if len(zero_matches) != 1:
        raise ValueError("VSCF active grid must contain one origin")
    zero_index = int(zero_matches[0])
    effective_powers = total_degree_powers(active_dimension, effective_degree)
    design = np.asarray([
        [float(np.prod(point ** power)) for power in effective_powers]
        for point in active_points
    ])
    fitted_coefficients, scalar_corrections, fit_errors = [], [], []
    sampled_effective_surfaces = []
    origin_vectors, iteration_counts = [], []

    for path_index in range(path_points):
        base_hamiltonians = []
        isolated_vectors = []
        for mode in range(mode_count):
            hamiltonian = np.diag(
                frequencies[path_index, mode] * (occupations[:, 0] + 0.5)
            )
            hamiltonian += np.tensordot(
                coefficients[mode][path_index], one_operators[mode], axes=1,
            )
            _values, vectors = np.linalg.eigh(hamiltonian)
            harmonic_ground = np.zeros(maximum_quanta + 1)
            harmonic_ground[0] = 1.0
            overlaps = np.abs(vectors.T @ harmonic_ground) ** 2
            state = int(np.argmax(overlaps))
            vector = vectors[:, state]
            if vector[0] < 0.0:
                vector = -vector
            base_hamiltonians.append(hamiltonian)
            isolated_vectors.append(vector)

        energies = []
        local_vectors = []
        local_iterations = []
        evaluation_order = [zero_index] + sorted(
            (index for index in range(len(active_points)) if index != zero_index),
            key=lambda index: (
                float(active_points[index] @ active_points[index]),
                tuple(active_points[index]),
            ),
        )
        solved_vectors: dict[int, list[np.ndarray]] = {}
        property_by_point=np.empty((len(active_points),len(words)))
        for point_index in evaluation_order:
            point = active_points[point_index]
            fields = [hamiltonian.copy() for hamiltonian in base_hamiltonians]
            for (mode, values, term_powers, active_coordinate,
                 active_position, contracted_position, operators) in active_terms:
                coordinate = point[active_coordinate]
                for coefficient, power in zip(
                    values[path_index], term_powers, strict=True,
                ):
                    fields[mode] += (
                        coefficient * coordinate ** power[active_position]
                        * operators[power[contracted_position]]
                    )
            if solved_vectors:
                nearest = min(
                    solved_vectors,
                    key=lambda index: float(np.sum(
                        (active_points[index] - point) ** 2
                    )),
                )
                vectors = [value.copy() for value in solved_vectors[nearest]]
            else:
                vectors = [value.copy() for value in isolated_vectors]
            densities = [np.outer(vector, vector) for vector in vectors]
            for iteration in range(1, maximum_iterations + 1):
                updated = []
                for mode in range(mode_count):
                    fock = fields[mode].copy()
                    for (first, second, values, term_powers, first_position,
                         second_position, first_operators,
                         second_operators) in pair_terms:
                        if mode not in {first, second}:
                            continue
                        for coefficient, power in zip(
                            values[path_index], term_powers, strict=True,
                        ):
                            if mode == first:
                                expectation = float(np.trace(
                                    densities[second] @ second_operators[
                                        power[second_position]
                                    ]
                                ))
                                fock += coefficient * expectation * first_operators[
                                    power[first_position]
                                ]
                            else:
                                expectation = float(np.trace(
                                    densities[first] @ first_operators[
                                        power[first_position]
                                    ]
                                ))
                                fock += coefficient * expectation * second_operators[
                                    power[second_position]
                                ]
                    _values, candidates = np.linalg.eigh(fock)
                    overlaps = np.abs(candidates.T @ vectors[mode]) ** 2
                    vector = candidates[:, int(np.argmax(overlaps))]
                    if float(vector @ vectors[mode]) < 0.0:
                        vector = -vector
                    updated.append(vector)
                proposed = [np.outer(vector, vector) for vector in updated]
                change = max(
                    float(np.linalg.norm(new - old))
                    for new, old in zip(proposed, densities, strict=True)
                )
                vectors = updated
                if change <= convergence_tolerance:
                    densities = proposed
                    break
                densities = [
                    (1.0 - density_mixing) * old + density_mixing * new
                    for old, new in zip(densities, proposed, strict=True)
                ]
            else:
                raise RuntimeError("n-mode VSCF contraction did not converge")
            energy = sum(float(vector @ field @ vector)
                         for vector, field in zip(vectors, fields, strict=True))
            for (first, second, values, term_powers, first_position,
                 second_position, first_operators, second_operators) in pair_terms:
                for coefficient, power in zip(
                    values[path_index], term_powers, strict=True,
                ):
                    energy += coefficient * float(
                        vectors[first] @ first_operators[power[first_position]]
                        @ vectors[first]
                    ) * float(
                        vectors[second] @ second_operators[power[second_position]]
                        @ vectors[second]
                    )
            if words:
                moments={key:float(vectors[key[0]] @ operator @ vectors[key[0]])
                    for key,operator in moment_operators.items()}
                property_by_point[point_index]=[np.prod([moments[mode,degree]
                    for mode,degree in enumerate(word) if degree]) for word in words]
            energies.append((point_index, energy))
            local_vectors.append((point_index, np.asarray(vectors)))
            solved_vectors[point_index] = vectors
            local_iterations.append(iteration)
        energy_by_point = np.empty(len(active_points))
        for point_index, energy in energies:
            energy_by_point[point_index] = energy
        vector_by_point = dict(local_vectors)
        origin_energy = float(energy_by_point[zero_index])
        harmonic_zpe = 0.5 * float(np.sum(frequencies[path_index]))
        scalar_corrections.append(origin_energy - harmonic_zpe)
        target = energy_by_point - origin_energy
        sampled_effective_surfaces.append(target)
        if words:property_samples.append(property_by_point)
        fitted, _residuals, rank, _singular = np.linalg.lstsq(
            design, target, rcond=None,
        )
        if rank != len(effective_powers):
            raise ValueError("VSCF effective-surface fit is rank deficient")
        fitted_coefficients.append(fitted)
        fit_errors.append(float(np.max(np.abs(design @ fitted - target))))
        origin_vectors.append(vector_by_point[zero_index])
        iteration_counts.append(max(local_iterations))
    result = (
        effective_powers,
        np.asarray(fitted_coefficients),
        np.asarray(scalar_corrections),
        np.asarray(origin_vectors),
        np.asarray([fit_errors, iteration_counts], dtype=float).T,
    )
    if _return_active_samples:
        result=(*result, active_points.copy(), np.asarray(sampled_effective_surfaces))
    if words:
        result=(*result, dict(powers=words,expectations=np.asarray(property_samples),
            scope="coordinate-multiplication moments in the converged product VSCF state"))
    return result


def contract_nmode_vscf_to_adaptive_surface(
    one_mode_coefficients_cm: Sequence[np.ndarray],
    one_mode_powers: Sequence[Sequence[Sequence[int]]],
    frequencies_cm: np.ndarray,
    active_pair_components: Sequence[
        tuple[int, np.ndarray, Sequence[Sequence[int]], int, int]
    ],
    contracted_pair_components: Sequence[
        tuple[int, int, np.ndarray, Sequence[Sequence[int]], int]
    ],
    *,
    active_dimension: int,
    maximum_quanta: int = 16,
    minimum_effective_degree: int = 2,
    maximum_effective_degree: int = 8,
    effective_fit_tolerance_cm: float = 1.0,
    active_grid: Sequence[float] = (
        -1.5, -1.125, -0.75, -0.375, 0.0, 0.375, 0.75, 1.125, 1.5,
    ),
    convergence_tolerance: float = 1.0e-10,
    maximum_iterations: int = 100,
    density_mixing: float = 0.5,
):
    """Contract n-mode fields once and select a qualified effective representation.

    A low-order polynomial is used only to satisfy the legacy sampling kernel;
    the returned representation is selected from the raw VSCF energies.  A
    complete tensor grid is retained directly when no polynomial passes the
    requested training and holdout errors.
    """

    from .adaptive_surface import fit_adaptive_effective_surface

    (_powers, _coefficients, scalar, vectors, diagnostics,
     active_points, sampled_surfaces) = contract_nmode_vscf_to_active_surface(
        one_mode_coefficients_cm,
        one_mode_powers,
        frequencies_cm,
        active_pair_components,
        contracted_pair_components,
        active_dimension=active_dimension,
        maximum_quanta=maximum_quanta,
        effective_degree=1,
        active_grid=active_grid,
        convergence_tolerance=convergence_tolerance,
        maximum_iterations=maximum_iterations,
        density_mixing=density_mixing,
        _return_active_samples=True,
    )
    surface = fit_adaptive_effective_surface(
        active_points,
        sampled_surfaces,
        minimum_degree=minimum_effective_degree,
        maximum_degree=maximum_effective_degree,
        tolerance=effective_fit_tolerance_cm,
    )
    return surface, scalar, vectors, diagnostics


__all__ = [
    "ReducedAnharmonicPathSurface",
    "anharmonic_channel_matrix",
    "anharmonic_channel_operators",
    "contract_one_mode_anharmonic_ground_states",
    "contract_pair_interaction_to_active_polynomial",
    "contract_two_mode_interaction_ground_correction",
    "contract_mode_interactions_to_active_surface",
    "contract_nmode_vscf_to_active_surface",
    "contract_nmode_vscf_to_adaptive_surface",
    "fit_reduced_anharmonic_path_surface",
    "fit_reduced_interaction_path_surface",
    "total_degree_powers",
]
