"""CM5 and Mayer analysis from PySCF density matrices or Molden orbitals."""

from __future__ import annotations

from pathlib import Path
import math
import tempfile

import numpy as np

from matrix_qm import (
    QMPopulationObservables,
    cm5_charges_from_hirshfeld,
    mayer_bond_order_matrix,
    mayer_spin_bond_order_matrix,
)


def load_molden_wavefunction(path: Path | str):
    """Load a complete Molden wavefunction using one normalized convention.

    ORCA's ``orca_2mkl`` writer includes normalized primitive coefficients,
    whereas PySCF's Molden reader expects the conventional unnormalized
    coefficients.  Keeping that correction here gives APOC population and
    electronic-state import exactly the same AO/MO interpretation.
    """

    try:
        from pyscf.tools import molden
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("PySCF is required to import Molden orbitals") from exc
    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="replace")
    if "Molden file created by orca_2mkl" not in text:
        return molden.load(str(source))
    text = _denormalize_orca_molden_gto(text)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".molden.input",
        encoding="utf-8",
    ) as normalized:
        normalized.write(text)
        normalized.flush()
        return molden.load(normalized.name)


def population_observables_from_pyscf(
    mol,
    density_matrix,
    *,
    grid_level: int = 4,
    block_size: int = 20_000,
    charge_tolerance: float = 2.0e-4,
) -> QMPopulationObservables:
    """Calculate the canonical ZAFF observables from a PySCF AO density.

    Neutral, spherically averaged atomic Hartree--Fock densities in the same
    AO basis and ECP provide the Hirshfeld pro-atoms.  With an ECP, Hirshfeld
    partitioning is performed consistently in valence space:
    ``q_A = Z_eff,A - N_val,A``.  The physical atomic number
    ``Z_A = Z_eff,A + N_core,A`` is then used by the CM5 correction.  The
    molecular density may be a spin-summed matrix or a pair of alpha/beta
    matrices.
    """

    try:
        from pyscf import dft
        from pyscf.scf import atom_hf
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("PySCF is required for AO-grid population analysis") from exc

    total_density, alpha_density, beta_density = _split_density_matrix(density_matrix)
    nao = int(mol.nao_nr())
    expected_shape = (nao, nao)
    if total_density.shape != expected_shape:
        raise ValueError(
            f"PySCF density has shape {total_density.shape}; expected {expected_shape}"
        )

    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=float)
    ao_owners = _ao_atom_indices(mol)
    if alpha_density is None:
        mayer = mayer_bond_order_matrix(
            overlap,
            ao_owners,
            density_total=total_density,
        )
    else:
        mayer = mayer_bond_order_matrix(
            overlap,
            ao_owners,
            density_alpha=alpha_density,
            density_beta=beta_density,
        )

    open_shell = int(getattr(mol, "spin", 0)) != 0
    if open_shell and (alpha_density is None or beta_density is None):
        raise ValueError("open-shell APOC analysis requires explicit alpha and beta densities")
    populations, spin_populations = _stockholder_population_quadrature(
        mol,
        total_density,
        spin_density=(alpha_density - beta_density) if open_shell else None,
        grid_level=grid_level,
        block_size=block_size,
        dft=dft,
        atom_hf=atom_hf,
    )

    effective_nuclear_charges = np.asarray(mol.atom_charges(), dtype=int)
    core_electrons = np.asarray(
        [int(mol.atom_nelec_core(atom)) for atom in range(mol.natm)],
        dtype=int,
    )
    atomic_numbers = effective_nuclear_charges + core_electrons
    hirshfeld = effective_nuclear_charges.astype(float) - populations
    expected_charge = float(mol.charge)
    charge_residual = expected_charge - float(np.sum(hirshfeld))
    if abs(charge_residual) > float(charge_tolerance):
        raise ValueError(
            "PySCF Hirshfeld quadrature violates the molecular charge by "
            f"{charge_residual:.3e} e; increase grid_level"
        )
    hirshfeld += charge_residual / mol.natm
    coordinates_angstrom = np.asarray(mol.atom_coords(unit="Angstrom"), dtype=float)
    cm5 = cm5_charges_from_hirshfeld(
        atomic_numbers,
        coordinates_angstrom,
        hirshfeld,
    )
    explicit_electron_count = float(np.einsum("ij,ji->", total_density, overlap, optimize=True))
    if spin_populations is not None:
        spin_residual = float(mol.spin) - float(np.sum(spin_populations))
        if abs(spin_residual) > float(charge_tolerance):
            raise ValueError(
                "PySCF Hirshfeld quadrature violates the alpha-minus-beta "
                f"electron count by {spin_residual:.3e}"
            )
        spin_populations += spin_residual / mol.natm
    return QMPopulationObservables(
        hirshfeld_charges=hirshfeld,
        cm5_charges=cm5,
        mayer_bond_orders=mayer,
        electron_count=explicit_electron_count + float(np.sum(core_electrons)),
        charge=expected_charge,
        ecp_core_electrons=core_electrons,
        spin_populations=spin_populations,
    )


def spin_population_observables_from_pyscf(
    mol,
    density_matrix,
    *,
    grid_level: int = 4,
    block_size: int = 20_000,
    spin_tolerance: float = 2.0e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Hirshfeld atomic spin populations and alpha-minus-beta Mayer map."""

    try:
        from pyscf import dft
        from pyscf.scf import atom_hf
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("PySCF is required for AO-grid spin analysis") from exc
    _total, alpha_density, beta_density = _split_density_matrix(density_matrix)
    if alpha_density is None or beta_density is None:
        raise ValueError("spin population analysis requires alpha and beta densities")
    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=float)
    owners = _ao_atom_indices(mol)
    spin_mayer = mayer_spin_bond_order_matrix(
        overlap,
        owners,
        density_alpha=alpha_density,
        density_beta=beta_density,
    )
    _total_populations, populations = _stockholder_population_quadrature(
        mol,
        _total,
        spin_density=alpha_density - beta_density,
        grid_level=grid_level,
        block_size=block_size,
        dft=dft,
        atom_hf=atom_hf,
    )
    assert populations is not None
    expected_spin = float(mol.spin)
    residual = expected_spin - float(np.sum(populations))
    if abs(residual) > float(spin_tolerance):
        raise ValueError(
            "PySCF Hirshfeld quadrature violates the alpha-minus-beta electron "
            f"count by {residual:.3e}"
        )
    populations += residual / mol.natm
    return populations, spin_mayer


def _stockholder_population_quadrature(
    mol,
    total_density,
    *,
    spin_density,
    grid_level,
    block_size,
    dft,
    atom_hf,
):
    """Integrate total and optional spin density on one identical grid."""

    grids = dft.gen_grid.Grids(mol)
    grids.level = int(grid_level)
    grids.build(with_non0tab=True)
    proatom_matrices = _neutral_proatom_density_matrices(mol, atom_hf)
    populations = np.zeros(mol.natm, dtype=float)
    spin_populations = None if spin_density is None else np.zeros(mol.natm, dtype=float)
    chunk = max(1, int(block_size))
    for start in range(0, grids.coords.shape[0], chunk):
        stop = min(grids.coords.shape[0], start + chunk)
        coordinates = grids.coords[start:stop]
        weights = np.asarray(grids.weights[start:stop], dtype=float)
        ao = np.asarray(dft.numint.eval_ao(mol, coordinates), dtype=float)
        molecular_density = np.asarray(dft.numint.eval_rho(mol, ao, total_density), dtype=float)
        proatoms = np.zeros((mol.natm, stop - start), dtype=float)
        for atom, (p0, p1) in enumerate(_ao_slices(mol)):
            local_ao = ao[:, p0:p1]
            proatoms[atom] = np.einsum(
                "pi,ij,pj->p",
                local_ao,
                proatom_matrices[atom],
                local_ao,
                optimize=True,
            )
        denominator = np.sum(np.maximum(proatoms, 0.0), axis=0)
        active = denominator > np.finfo(float).tiny
        stockholder = np.divide(
            proatoms,
            denominator,
            out=np.zeros_like(proatoms),
            where=active[None, :],
        )
        populations += np.sum(
            stockholder * (weights * np.maximum(molecular_density, 0.0))[None, :],
            axis=1,
        )
        if spin_populations is not None:
            signed_density = np.asarray(
                dft.numint.eval_rho(mol, ao, spin_density),
                dtype=float,
            )
            spin_populations += np.sum(
                stockholder * (weights * signed_density)[None, :],
                axis=1,
            )
    return populations, spin_populations


def population_observables_from_molden(
    path: Path | str,
    *,
    grid_level: int = 4,
    block_size: int = 20_000,
    charge: int | None = None,
) -> QMPopulationObservables:
    """Use a backend-neutral Molden orbital export as the population source."""

    source = Path(path)
    mol, _energies, coefficients, occupations, _labels, _spins = load_molden_wavefunction(source)
    if coefficients is None or occupations is None:
        raise ValueError(f"Molden file contains no molecular orbitals: {path}")
    if charge is not None:
        mol.charge = int(charge)
    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=float)
    density = _density_from_molecular_orbitals(
        coefficients,
        occupations,
        overlap=overlap,
    )
    electronic_charge = float(np.sum(mol.atom_charges())) - _electron_count_from_mos(occupations)
    if charge is None:
        rounded = int(round(electronic_charge))
        if abs(electronic_charge - rounded) > 1.0e-8:
            raise ValueError("Molden occupations imply a non-integral molecular charge")
        mol.charge = rounded
    elif abs(electronic_charge - int(charge)) > 1.0e-8:
        raise ValueError("Explicit charge is inconsistent with the Molden orbital occupations")
    return population_observables_from_pyscf(
        mol,
        density,
        grid_level=grid_level,
        block_size=block_size,
    )


def _density_from_molecular_orbitals(coefficients, occupations, *, overlap):
    if isinstance(coefficients, tuple):
        if not isinstance(occupations, tuple) or len(coefficients) != 2:
            raise ValueError("Molden alpha/beta orbitals require two occupation vectors")
        return tuple(
            _density_from_one_spin_orbital_block(coeff, occ, overlap)
            for coeff, occ in zip(coefficients, occupations, strict=True)
        )
    return _density_from_one_spin_orbital_block(coefficients, occupations, overlap)


def _density_from_one_spin_orbital_block(coefficients, occupations, overlap):
    """Build a density after metric orthonormalization of the occupied space.

    Several external Molden writers, notably ORCA's ``orca_2mkl``, use AO
    normalization conventions that do not give ``C.T S C = I`` when read by
    another program.  Symmetric orthonormalization of the occupied block
    preserves its span and restores the exact electron count without mixing it
    with virtual orbitals.
    """

    coeff = np.asarray(coefficients, dtype=float)
    occ = np.asarray(occupations, dtype=float).reshape(-1)
    metric = np.asarray(overlap, dtype=float)
    if coeff.ndim != 2 or coeff.shape[1] != occ.size:
        raise ValueError("Molden coefficient and occupation dimensions differ")
    active = np.flatnonzero(occ > 1.0e-12)
    if active.size == 0:
        return np.zeros((coeff.shape[0], coeff.shape[0]), dtype=float)
    active_occupations = occ[active]
    if not np.allclose(active_occupations, active_occupations[0], atol=1.0e-10, rtol=0.0):
        raise ValueError("APOC Molden import does not yet support unequal fractional occupations")
    occupied = coeff[:, active]
    gram = 0.5 * (occupied.T @ metric @ occupied + occupied.T @ metric.T @ occupied)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if np.min(eigenvalues) <= 1.0e-10:
        raise ValueError("Molden occupied orbital space is linearly dependent")
    inverse_sqrt = (eigenvectors * (1.0 / np.sqrt(eigenvalues))[None, :]) @ eigenvectors.T
    occupied = occupied @ inverse_sqrt
    return (occupied * active_occupations) @ occupied.T


def _electron_count_from_mos(occupations) -> float:
    if isinstance(occupations, tuple):
        return float(sum(np.sum(np.asarray(item, dtype=float)) for item in occupations))
    return float(np.sum(np.asarray(occupations, dtype=float)))


def _denormalize_orca_molden_gto(text: str) -> str:
    """Translate ``orca_2mkl`` normalized primitives to Molden/PySCF form."""

    lines = text.splitlines()
    output = list(lines)
    in_gto = False
    primitive_rows_remaining = 0
    angular_momentum = 0
    shell_letters = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}
    for index, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_gto = stripped.upper() == "[GTO]"
            primitive_rows_remaining = 0
            continue
        if not in_gto or not stripped:
            continue
        fields = stripped.split()
        if primitive_rows_remaining:
            exponent = float(fields[0].replace("D", "E").replace("d", "e"))
            normalization = _normalized_cartesian_primitive_factor(
                exponent,
                angular_momentum,
            )
            coefficients = [
                float(value.replace("D", "E").replace("d", "e")) / normalization
                for value in fields[1:]
            ]
            output[index] = (
                "      " + f"{exponent:.12g}" + "".join(f" {value:.12g}" for value in coefficients)
            )
            primitive_rows_remaining -= 1
            continue
        shell = fields[0].lower()
        if shell in shell_letters and len(fields) >= 2:
            angular_momentum = shell_letters[shell]
            primitive_rows_remaining = int(fields[1])
    return "\n".join(output) + "\n"


def _normalized_cartesian_primitive_factor(exponent: float, angular_momentum: int) -> float:
    odd_double_factorial = 1
    for value in range(1, 2 * int(angular_momentum), 2):
        odd_double_factorial *= value
    return (
        (2.0 * float(exponent) / math.pi) ** 0.75
        * (4.0 * float(exponent)) ** (0.5 * int(angular_momentum))
        / math.sqrt(float(odd_double_factorial))
    )


def _split_density_matrix(density_matrix):
    if isinstance(density_matrix, (tuple, list)):
        if len(density_matrix) != 2:
            raise ValueError("spin-resolved density must contain alpha and beta matrices")
        alpha = np.asarray(density_matrix[0], dtype=float)
        beta = np.asarray(density_matrix[1], dtype=float)
        return alpha + beta, alpha, beta
    density = np.asarray(density_matrix, dtype=float)
    if density.ndim == 3 and density.shape[0] == 2:
        return density[0] + density[1], density[0], density[1]
    return density, None, None


def _ao_slices(mol) -> tuple[tuple[int, int], ...]:
    return tuple((int(row[2]), int(row[3])) for row in mol.aoslice_by_atom())


def _ao_atom_indices(mol) -> np.ndarray:
    owners = np.empty(mol.nao_nr(), dtype=int)
    for atom, (p0, p1) in enumerate(_ao_slices(mol)):
        owners[p0:p1] = atom
    return owners


def _neutral_proatom_density_matrices(mol, atom_hf_module) -> tuple[np.ndarray, ...]:
    atomic_scf = atom_hf_module.get_atm_nrhf(mol)
    matrices: list[np.ndarray] = []
    for atom, (p0, p1) in enumerate(_ao_slices(mol)):
        symbol = mol.atom_symbol(atom)
        try:
            _energy, _orbital_energies, coefficients, occupations = atomic_scf[symbol]
        except KeyError as exc:
            raise ValueError(f"no neutral pro-atom density is available for {symbol}") from exc
        coefficients = np.asarray(coefficients, dtype=float)
        occupations = np.asarray(occupations, dtype=float)
        density = (coefficients * occupations) @ coefficients.T
        if density.shape != (p1 - p0, p1 - p0) and bool(mol.cart):
            density = _spherical_proatom_density_in_cartesian_ao_basis(
                mol,
                atom,
                density,
            )
        if density.shape != (p1 - p0, p1 - p0):
            raise ValueError(
                f"neutral pro-atom AO dimension mismatch for atom {atom + 1} ({symbol})"
            )
        matrices.append(density)
    return tuple(matrices)


def _spherical_proatom_density_in_cartesian_ao_basis(
    mol,
    atom: int,
    density_spherical: np.ndarray,
) -> np.ndarray:
    """Embed PySCF's spherical atomic HF density in a Cartesian molecular AO basis."""

    from pyscf import gto

    blocks: list[np.ndarray] = []
    for shell in range(int(mol.nbas)):
        if int(mol.bas_atom(shell)) != int(atom):
            continue
        angular = int(mol.bas_angular(shell))
        transform = np.asarray(
            gto.cart2sph(angular, normalized="sp"),
            dtype=float,
        )
        for _contraction in range(int(mol.bas_nctr(shell))):
            blocks.append(transform)
    cartesian_count = sum(block.shape[0] for block in blocks)
    spherical_count = sum(block.shape[1] for block in blocks)
    if density_spherical.shape != (spherical_count, spherical_count):
        return density_spherical
    transform = np.zeros((cartesian_count, spherical_count), dtype=float)
    cartesian_offset = 0
    spherical_offset = 0
    for block in blocks:
        ncart, nspherical = block.shape
        transform[
            cartesian_offset : cartesian_offset + ncart,
            spherical_offset : spherical_offset + nspherical,
        ] = block
        cartesian_offset += ncart
        spherical_offset += nspherical
    return transform @ density_spherical @ transform.T
