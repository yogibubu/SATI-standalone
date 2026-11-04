"""CM5 and Mayer observables from the native APOC electronic state."""

from __future__ import annotations

import numpy as np

from matrix_chem.topology.elements import atomic_symbol
from matrix_core import (
    ElectronicObservableCompleteness,
    ElectronicObservableKind,
    ElectronicObservableOwner,
    ElectronicObservableProvenance,
)
from matrix_qm import (
    QMPopulationObservables,
    mayer_bond_order_matrix,
    mayer_spin_bond_order_matrix,
)

from .state import ElectronicState


APOC_CM5_PROVENANCE = ElectronicObservableProvenance(
    owner=ElectronicObservableOwner.APOC,
    observable=ElectronicObservableKind.CM5,
    completeness=ElectronicObservableCompleteness.COMPLETE,
    source="APOC CM5 full molecule",
)
APOC_MAYER_PROVENANCE = ElectronicObservableProvenance(
    owner=ElectronicObservableOwner.APOC,
    observable=ElectronicObservableKind.MAYER,
    completeness=ElectronicObservableCompleteness.COMPLETE,
    source="APOC Mayer full molecule",
)


def mayer_bond_orders_from_electronic_state(state: ElectronicState) -> np.ndarray:
    """Evaluate the canonical Mayer matrix without invoking a QM backend."""

    if state.basis is None:
        raise ValueError("Mayer analysis requires AO-to-atom ownership in the APOC basis")
    owners = state.basis.shell_nucleus_index[state.basis.ao_shell]
    if state.spin_channels == 1:
        return mayer_bond_order_matrix(
            state.overlap_ao,
            owners,
            density_total=state.density_ao(),
        )
    return mayer_bond_order_matrix(
        state.overlap_ao,
        owners,
        density_alpha=state.density_ao(0),
        density_beta=state.density_ao(1),
    )


def mayer_spin_bond_orders_from_electronic_state(
    state: ElectronicState,
) -> np.ndarray:
    """Evaluate the alpha-minus-beta Mayer map for an unrestricted state."""

    if state.basis is None:
        raise ValueError("spin Mayer analysis requires AO-to-atom ownership in the APOC basis")
    if state.spin_channels != 2:
        return np.zeros((state.atomic_numbers.size, state.atomic_numbers.size))
    owners = state.basis.shell_nucleus_index[state.basis.ao_shell]
    return mayer_spin_bond_order_matrix(
        state.overlap_ao,
        owners,
        density_alpha=state.density_ao(0),
        density_beta=state.density_ao(1),
    )


def population_observables_from_electronic_state(
    state: ElectronicState,
    *,
    grid_level: int = 4,
    block_size: int = 20_000,
) -> QMPopulationObservables:
    """Calculate APOC Hirshfeld/CM5 and Mayer from a backend-neutral state.

    PySCF is used only as the Gaussian-AO/grid numerical engine. The definitions,
    input density and returned contract are owned by APOC; no population model
    reported by the producing electronic-structure code is substituted.
    """

    from matrix_pyscf import population_observables_from_pyscf

    molecule = pyscf_molecule_from_electronic_state(state)
    density = (
        state.density_ao()
        if state.spin_channels == 1
        else (state.density_ao(0), state.density_ao(1))
    )
    result = population_observables_from_pyscf(
        molecule,
        density,
        grid_level=grid_level,
        block_size=block_size,
    )
    native_mayer = mayer_bond_orders_from_electronic_state(state)
    if not np.allclose(result.mayer_bond_orders, native_mayer, atol=2.0e-8, rtol=0.0):
        raise ArithmeticError("APOC Mayer implementations disagree for the same AO state")
    if state.multiplicity == 1 and result.spin_populations is not None:
        raise ArithmeticError("closed-shell APOC analysis emitted a spin population")
    if state.multiplicity > 1 and result.spin_populations is None:
        raise ArithmeticError("open-shell APOC analysis omitted its spin population")
    return result


def pyscf_molecule_from_electronic_state(state: ElectronicState):
    """Reconstruct the explicit Gaussian AO space solely for APOC quadrature."""

    if state.basis is None:
        raise ValueError("CM5 analysis requires an explicit Gaussian basis")
    try:
        from pyscf import gto
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("APOC CM5 quadrature requires the PySCF numerical runtime") from exc
    basis = state.basis
    labels = [
        f"{atomic_symbol(int(number))}{index}" for index, number in enumerate(state.atomic_numbers)
    ]
    atoms = list(zip(labels, state.coordinates_bohr.tolist(), strict=True))
    basis_by_atom: dict[str, list[list[object]]] = {label: [] for label in labels}
    for shell_index in range(basis.shell_count):
        atom_index = int(basis.shell_nucleus_index[shell_index])
        primitive_indices = np.flatnonzero(basis.primitive_shell_index == shell_index)
        shell: list[object] = [int(basis.shell_angular_momentum[shell_index])]
        for primitive_index in primitive_indices:
            shell.append(
                [
                    float(basis.primitive_exponent[primitive_index]),
                    float(basis.primitive_coefficient[primitive_index]),
                ]
            )
        basis_by_atom[labels[atom_index]].append(shell)
    molecule = gto.Mole()
    molecule.atom = atoms
    molecule.unit = "Bohr"
    molecule.basis = basis_by_atom
    molecule.charge = int(state.charge)
    molecule.spin = int(state.multiplicity) - 1
    molecule.cart = bool(basis.cartesian)
    molecule.verbose = 0
    molecule.build()
    overlap = np.asarray(molecule.intor_symmetric("int1e_ovlp"), dtype=float)
    if overlap.shape != state.overlap_ao.shape or not np.allclose(
        overlap, state.overlap_ao, atol=2.0e-7, rtol=0.0
    ):
        raise ValueError(
            "the producing backend AO order/normalization differs from the APOC "
            "quadrature engine; a direct adapter transformation is required"
        )
    return molecule
