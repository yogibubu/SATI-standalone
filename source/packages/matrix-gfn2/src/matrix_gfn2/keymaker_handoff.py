"""Resident GFN2 and SF-TDA results exposed through KEYMAKER."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .calculator import GFN2Calculator
from .sf_tda_contract import load_sf_tda_contract


def evaluate_fast_xtb(
    atoms: Sequence[str], coordinates_angstrom: np.ndarray, *,
    charge: int = 0, multiplicity: int = 1,
) -> dict[str, object]:
    """Expose observables from one resident GFN2 state without APOC analysis."""

    result = GFN2Calculator(atoms, charge=charge, multiplicity=multiplicity).evaluate(
        np.asarray(coordinates_angstrom, dtype=float), compute_bond_orders=True
    )
    if not result.converged:
        raise RuntimeError("resident MATRIX-GFN2 did not converge")
    charges = np.asarray(result.charges_electron, dtype=float)
    bond_orders = np.asarray(result.bond_orders, dtype=float)
    natoms = len(atoms)
    if charges.shape != (natoms,) or bond_orders.shape != (natoms, natoms):
        raise ValueError("resident MATRIX-GFN2 observables have inconsistent dimensions")
    if not np.all(np.isfinite(charges)) or not np.all(np.isfinite(bond_orders)):
        raise ValueError("resident MATRIX-GFN2 observables contain non-finite values")
    return {
        "schema": "matrix.gfn2.fast_xtb_result.v1",
        "calculator": "matrix_gfn2.GFN2Calculator",
        "energy_hartree": float(result.energy_hartree),
        "gradient_hartree_per_bohr": result.gradient_hartree_per_bohr.tolist(),
        "charges_electron": charges.tolist(),
        "charge_model": "GFN2-SCC",
        "bond_orders": bond_orders.tolist(),
        "bond_order_definition": "Mayer AO expression evaluated by MATRIX-GFN2",
        "observable_source": "same converged MATRIX-GFN2 state; no APOC or CM5 recalculation",
        "scf_iterations": int(result.iterations),
        "converged": True,
    }


def evaluate_sf_tda(
    atoms: Sequence[str], coordinates_angstrom: np.ndarray, *,
    charge: int = 0, multiplicity: int = 2, roots: int = 3,
) -> dict[str, object]:
    """Evaluate resident spin-flip roots and complete analytic gradients."""

    if multiplicity < 2:
        raise ValueError("SF-TDA requires an open-shell reference")
    if roots < 1:
        raise ValueError("SF-TDA requires at least one root")
    contract = load_sf_tda_contract()
    result = GFN2Calculator(
        atoms, charge=charge, multiplicity=multiplicity, temperature_kelvin=0.0
    ).evaluate_sf_tda(np.asarray(coordinates_angstrom, dtype=float), roots=roots)
    if not result.ground_state.converged or not result.sf_tda_result.converged:
        raise RuntimeError("resident SF-TDA did not converge")
    return {
        "schema": "matrix.gfn2.sf_tda_result.v1",
        "calculator": "matrix_gfn2.GFN2Calculator.evaluate_sf_tda",
        "scientific_contract_sha256": contract.sha256,
        "ground_energy_hartree": float(result.ground_state.energy_hartree),
        "excitation_energies_hartree": result.excitation_energies_hartree.tolist(),
        "total_energies_hartree": result.total_energies_hartree.tolist(),
        "total_gradients_hartree_per_bohr": result.total_gradients_hartree_per_bohr.tolist(),
        "converged": True,
    }
