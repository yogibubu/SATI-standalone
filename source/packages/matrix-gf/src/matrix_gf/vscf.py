"""Vibrational self-consistent-field contractions in local DVR bases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DiagonalPairVSCFResult:
    """Converged modals and fluctuation potentials for a two-mode expansion."""

    modal_energies: tuple[np.ndarray, ...]
    modal_coefficients: tuple[np.ndarray, ...]
    residual_pair_potentials: dict[tuple[int, int], np.ndarray]
    double_counting_energy: float
    iterations: int
    maximum_density_change: float
    extra_potential_expectation: float = 0.0


def solve_diagonal_pair_vscf(
    one_mode_hamiltonians: Sequence[np.ndarray],
    pair_potentials: Mapping[tuple[int, int], np.ndarray],
    *,
    retained_modals: int,
    convergence_tolerance: float = 1.0e-10,
    maximum_iterations: int = 200,
    density_mixing: float = 0.5,
    extra_mean_field: Callable | None = None,
) -> DiagonalPairVSCFResult:
    """Solve ground-state VSCF for pair potentials diagonal in local DVRs.

    ``pair_potentials[(i, j)][x_i, x_j]`` is the pure two-mode increment after
    subtraction of the one-mode cuts.  Damped density iteration stabilizes
    soft-mode fields.  The returned residual potentials have zero expectation
    in the converged product state and can be added directly in a subsequent
    VCI calculation without double counting the VSCF mean field.

    ``extra_mean_field(densities)`` may supply conditional fields and the
    expectation of an additional full-dimensional residual potential. Each
    field includes the expectation over all other modes. Its contribution to
    double counting is (number_of_modes - 1) times that expectation. This
    callback must exclude the explicit one-mode and pair terms supplied here.
    Residual pair tables alone do not represent this additional operator in VCI;
    its full operator and the subtracted mean fields must also be retained.
    """
    hamiltonians = tuple(np.asarray(value, dtype=float)
                         for value in one_mode_hamiltonians)
    mode_count = len(hamiltonians)
    if (
        mode_count < 1
        or retained_modals < 1
        or convergence_tolerance <= 0.0
        or maximum_iterations < 1
        or not 0.0 < density_mixing <= 1.0
    ):
        raise ValueError("VSCF control values are invalid")
    dimensions = []
    for mode, hamiltonian in enumerate(hamiltonians):
        if (
            hamiltonian.ndim != 2
            or hamiltonian.shape[0] != hamiltonian.shape[1]
            or retained_modals > hamiltonian.shape[0]
            or not np.all(np.isfinite(hamiltonian))
            or not np.allclose(
                hamiltonian, hamiltonian.T, atol=1.0e-11, rtol=0.0
            )
        ):
            raise ValueError(f"one-mode Hamiltonian {mode} is invalid")
        dimensions.append(hamiltonian.shape[0])

    pairs: dict[tuple[int, int], np.ndarray] = {}
    for key, raw in pair_potentials.items():
        if len(key) != 2:
            raise ValueError("VSCF pair keys must contain two mode indices")
        first, second = map(int, key)
        if not 0 <= first < second < mode_count:
            raise ValueError("VSCF pair indices must be unique and ordered")
        potential = np.asarray(raw, dtype=float)
        if (
            potential.shape != (dimensions[first], dimensions[second])
            or not np.all(np.isfinite(potential))
            or (first, second) in pairs
        ):
            raise ValueError(f"VSCF pair potential {(first, second)} is invalid")
        pairs[first, second] = potential

    densities = []
    for hamiltonian in hamiltonians:
        _values, vectors = np.linalg.eigh(hamiltonian)
        densities.append(np.abs(vectors[:, 0]) ** 2)

    def mean_fields(current: Sequence[np.ndarray]):
        fields = [np.zeros(dimension) for dimension in dimensions]
        for (first, second), potential in pairs.items():
            fields[first] += potential @ current[second]
            fields[second] += potential.T @ current[first]
        extra_expectation = 0.0
        if extra_mean_field is not None:
            additional, extra_expectation = extra_mean_field(tuple(current))
            if len(additional) != mode_count or not np.isfinite(extra_expectation):
                raise ValueError("invalid full-potential mean-field result")
            for mode, additional_field in enumerate(additional):
                value = np.asarray(additional_field)
                if (value.shape != (dimensions[mode],) or np.iscomplexobj(value)
                        or not np.all(np.isfinite(value))):
                    raise ValueError("invalid full-potential conditional field")
                fields[mode] += value
        return fields, float(extra_expectation)

    maximum_change = float("inf")
    for iteration in range(1, maximum_iterations + 1):
        fields, _extra_expectation = mean_fields(densities)
        proposed = []
        for hamiltonian, field in zip(hamiltonians, fields, strict=True):
            _values, vectors = np.linalg.eigh(
                hamiltonian + np.diag(field)
            )
            proposed.append(np.abs(vectors[:, 0]) ** 2)
        maximum_change = max(
            float(np.max(np.abs(new - old)))
            for new, old in zip(proposed, densities, strict=True)
        )
        if maximum_change <= convergence_tolerance:
            densities = proposed
            break
        densities = [
            (1.0 - density_mixing) * old + density_mixing * new
            for old, new in zip(densities, proposed, strict=True)
        ]
    else:
        raise RuntimeError(
            "pair VSCF did not converge: maximum density change "
            f"{maximum_change:.6g} after {maximum_iterations} iterations"
        )

    fields, extra_expectation = mean_fields(densities)
    modal_energies, modal_coefficients = [], []
    for hamiltonian, field in zip(hamiltonians, fields, strict=True):
        values, vectors = np.linalg.eigh(hamiltonian + np.diag(field))
        modal_energies.append(values[:retained_modals])
        modal_coefficients.append(vectors[:, :retained_modals])

    residuals: dict[tuple[int, int], np.ndarray] = {}
    double_counting = (mode_count - 1) * extra_expectation
    for (first, second), potential in pairs.items():
        field_first = potential @ densities[second]
        field_second = potential.T @ densities[first]
        expectation = float(densities[first] @ field_first)
        double_counting += expectation
        residuals[first, second] = (
            potential
            - field_first[:, None]
            - field_second[None, :]
            + expectation
        )

    return DiagonalPairVSCFResult(
        modal_energies=tuple(modal_energies),
        modal_coefficients=tuple(modal_coefficients),
        residual_pair_potentials=residuals,
        double_counting_energy=double_counting,
        iterations=iteration,
        maximum_density_change=maximum_change,
        extra_potential_expectation=extra_expectation,
    )


__all__ = ["DiagonalPairVSCFResult", "solve_diagonal_pair_vscf"]
