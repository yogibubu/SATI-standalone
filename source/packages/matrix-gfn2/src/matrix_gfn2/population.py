"""Whole-molecule Hirshfeld/CM5 populations for the GFN2 valence density."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .basis import contracted_gaussian_values
from .molecular_basis import MolecularBasis
from .parameters import GFN2ParameterSet


def hirshfeld_cm5_charges(
    atomic_numbers: Sequence[int],
    coordinates_bohr: np.ndarray,
    density: np.ndarray,
    basis: MolecularBasis,
    parameters: GFN2ParameterSet,
    *,
    molecular_charge: int,
    multiplicity: int,
    grid_level: int = 3,
    block_size: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate GFN2 neutral-proatom Hirshfeld weights and apply CM5PAC."""

    try:
        from pyscf import dft, gto
    except ImportError as exc:  # pragma: no cover - optional acceleration runtime
        raise RuntimeError("GFN2 CM5 quadrature requires the PySCF grid runtime") from exc
    try:
        from matrix_qm import cm5_charges_from_hirshfeld
    except ImportError as exc:  # pragma: no cover - installation contract
        raise RuntimeError("GFN2 CM5 requires the MATRIX-QM CM5PAC mapping") from exc

    numbers = tuple(int(value) for value in atomic_numbers)
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    labels = [
        f"{parameters.elements[number].symbol}{atom}"
        for atom, number in enumerate(numbers)
    ]
    molecule = gto.Mole()
    molecule.atom = list(zip(labels, coordinates.tolist(), strict=True))
    molecule.unit = "Bohr"
    molecule.basis = {label: [[0, [1.0, 1.0]]] for label in labels}
    molecule.charge = int(molecular_charge)
    molecule.spin = int(multiplicity) - 1
    molecule.cart = False
    molecule.verbose = 0
    molecule.build(False, False)

    reference_density = np.zeros((len(numbers), basis.norbitals, basis.norbitals))
    shell_start = 0
    for atom, number in enumerate(numbers):
        for shell in parameters.elements[number].shells:
            angular = int(shell.label[-1] == "p") + 2 * int(shell.label[-1] == "d")
            degeneracy = 2 * angular + 1
            occupancy = shell.reference_occupation / degeneracy
            indices = np.arange(shell_start, shell_start + degeneracy)
            reference_density[atom, indices, indices] = occupancy
            shell_start += degeneracy

    grids = dft.gen_grid.Grids(molecule)
    grids.level = int(grid_level)
    grids.build(with_non0tab=True)
    populations = np.zeros(len(numbers))
    chunk = max(1, int(block_size))
    total_density = np.asarray(density, dtype=float)
    for start in range(0, len(grids.coords), chunk):
        stop = min(len(grids.coords), start + chunk)
        cartesian_ao = contracted_gaussian_values(
            basis.cartesian_orbitals, grids.coords[start:stop]
        )
        ao = cartesian_ao @ basis.cartesian_to_spherical
        molecular_density = np.einsum(
            "pi,ij,pj->p", ao, total_density, ao, optimize=True
        )
        proatoms = np.einsum(
            "pi,aij,pj->ap", ao, reference_density, ao, optimize=True
        )
        proatoms = np.maximum(proatoms, 0.0)
        denominator = np.sum(proatoms, axis=0)
        stockholder = np.divide(
            proatoms,
            denominator[None, :],
            out=np.zeros_like(proatoms),
            where=denominator[None, :] > np.finfo(float).tiny,
        )
        weighted_density = grids.weights[start:stop] * np.maximum(molecular_density, 0.0)
        populations += np.sum(stockholder * weighted_density[None, :], axis=1)
    valence = np.asarray(
        [
            sum(shell.reference_occupation for shell in parameters.elements[number].shells)
            for number in numbers
        ]
    )
    hirshfeld = valence - populations
    residual = float(molecular_charge) - float(np.sum(hirshfeld))
    if abs(residual) > 2.0e-3:
        raise ArithmeticError(
            f"GFN2 Hirshfeld quadrature charge residual is {residual:.3e} e"
        )
    hirshfeld += residual / len(numbers)
    cm5 = cm5_charges_from_hirshfeld(
        numbers, coordinates * 0.529177210903, hirshfeld
    )
    return hirshfeld, cm5


__all__ = ["hirshfeld_cm5_charges"]
