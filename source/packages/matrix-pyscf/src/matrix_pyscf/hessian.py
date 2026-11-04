"""In-process Cartesian Hessians for LINK lower-level guesses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .parsers import _resolve_pyscf_reference


BOHR_TO_ANGSTROM = 0.529177210903


@dataclass(frozen=True)
class PySCFHessianEvaluation:
    hessian_hartree_per_bohr2: np.ndarray
    gradient_evaluations: int
    derivative: str


def pyscf_cartesian_hessian(
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    *,
    method: str,
    basis: str,
    charge: int = 0,
    multiplicity: int = 1,
    processors: int = 1,
    finite_difference_step_bohr: float = 1.0e-3,
) -> PySCFHessianEvaluation:
    """Evaluate an HF/DFT Hessian or an MP2 gradient-difference Hessian."""

    from pyscf import dft, gto, lib, mp, scf

    symbols = tuple(str(atom) for atom in atoms)
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.shape != (len(symbols), 3):
        raise ValueError("PySCF Hessian coordinates must have shape natoms x 3")
    step = float(finite_difference_step_bohr)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("PySCF Hessian finite-difference step must be positive")
    lib.num_threads(max(1, int(processors)))
    normalized_method = str(method).strip() or "HF"
    correlated = normalized_method.upper() in {"MP2", "RMP2", "UMP2"}
    reference = _resolve_pyscf_reference(
        method="HF" if correlated else normalized_method,
        multiplicity=int(multiplicity),
        reference=None,
    )

    def mean_field_at(geometry_angstrom: np.ndarray):
        molecule = gto.M(
            atom=[
                [atom, tuple(float(value) for value in xyz)]
                for atom, xyz in zip(symbols, geometry_angstrom, strict=True)
            ],
            basis=str(basis).strip() or "sto-3g",
            charge=int(charge),
            spin=int(multiplicity) - 1,
            unit="Angstrom",
            symmetry=False,
            verbose=0,
        )
        constructors = {
            "RHF": scf.RHF,
            "ROHF": scf.ROHF,
            "UHF": scf.UHF,
            "RKS": dft.RKS,
            "ROKS": dft.ROKS,
            "UKS": dft.UKS,
        }
        mean_field = constructors[reference](molecule)
        mean_field.verbose = 0
        if reference.endswith("KS"):
            mean_field.xc = normalized_method
        mean_field.kernel()
        if not mean_field.converged:
            raise RuntimeError("PySCF lower-level Hessian SCF did not converge")
        return mean_field

    if not correlated:
        mean_field = mean_field_at(coordinates)
        tensor = np.asarray(mean_field.Hessian().kernel(), dtype=float)
        matrix = tensor.transpose(0, 2, 1, 3).reshape(coordinates.size, coordinates.size)
        return PySCFHessianEvaluation(
            hessian_hartree_per_bohr2=0.5 * (matrix + matrix.T),
            gradient_evaluations=0,
            derivative="analytic",
        )

    def mp2_gradient(geometry_angstrom: np.ndarray) -> np.ndarray:
        mean_field = mean_field_at(geometry_angstrom)
        calculation = mp.MP2(mean_field).run(verbose=0)
        if getattr(calculation, "converged", True) is False:
            raise RuntimeError("PySCF lower-level MP2 calculation did not converge")
        return np.asarray(calculation.nuc_grad_method().kernel(), dtype=float).reshape(-1)

    matrix = np.empty((coordinates.size, coordinates.size), dtype=float)
    for column in range(coordinates.size):
        atom_index, component = divmod(column, 3)
        plus = coordinates.copy()
        minus = coordinates.copy()
        displacement = step * BOHR_TO_ANGSTROM
        plus[atom_index, component] += displacement
        minus[atom_index, component] -= displacement
        matrix[:, column] = (mp2_gradient(plus) - mp2_gradient(minus)) / (2.0 * step)
    return PySCFHessianEvaluation(
        hessian_hartree_per_bohr2=0.5 * (matrix + matrix.T),
        gradient_evaluations=2 * coordinates.size,
        derivative="central_difference_of_analytic_MP2_gradients",
    )


__all__ = ["PySCFHessianEvaluation", "pyscf_cartesian_hessian"]
