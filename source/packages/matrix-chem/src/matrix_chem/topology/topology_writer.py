from pathlib import Path

from .contracts import MATRIX_XYZ_TOPOLOGY_SCHEMA
from .automorphisms import build_topology_automorphism_orbits, topology_automorphism_lines
from .bonding_roles import is_bridging_ligand



def write_topology_section(
    fh,
    *,
    cg,
    dg,
    ringset=None,
    aromaticity=None,
    synthons=None,
    spin_density=None,
    smiles=None,
):
    """
    Write topology-related sections to an open xyzin file handle.
    """

    # ========================================================
    # TOPOLOGY
    # ========================================================

    fh.write("\n#TOPOLOGY\n")
    fh.write(f"SCHEMA {MATRIX_XYZ_TOPOLOGY_SCHEMA}\n")
    fh.write("INDEXING ATOMS=ONE_BASED\n")
    bond_order_source = getattr(synthons, "_bond_order_source", "ORACLE Pauling estimate")
    fh.write(f"BOND_ORDER_SOURCE {bond_order_source}\n")
    fh.write("RING_BASIS_POLICY CHORDLESS_NONMETAL_MINIMUM_CYCLE_BASIS\n")
    diagnostics = getattr(ringset, "cycle_basis_diagnostics", None)
    if diagnostics is not None:
        excluded = (
            ",".join(str(atom + 1) for atom in diagnostics.excluded_atoms)
            if diagnostics.excluded_atoms
            else "NONE"
        )
        fh.write(f"RING_BASIS_ALGORITHM {diagnostics.algorithm}\n")
        fh.write(f"RING_BASIS_COMPLETE {'YES' if diagnostics.complete else 'NO'}\n")
        fh.write(f"RING_CANDIDATE_COUNT {diagnostics.candidate_cycle_count}\n")
        fh.write(f"RING_BASIS_RANK {diagnostics.cycle_rank}\n")
        fh.write(f"RING_BASIS_COUNT {diagnostics.selected_cycle_count}\n")
        fh.write(f"RING_BASIS_MAXIMUM_SIZE {diagnostics.maximum_selected_size}\n")
        fh.write(f"RING_BASIS_ALLOWED_ATOMS {diagnostics.allowed_atom_count}\n")
        fh.write(f"RING_BASIS_ALLOWED_EDGES {diagnostics.allowed_edge_count}\n")
        fh.write(f"RING_BASIS_EXCLUDED_ATOMS {excluded}\n")

    # ========================================================
    # ATOMS
    # ========================================================

    fh.write("[ATOMS]\n")
    fh.write("  i   Z    Zeff  Charge   Coval   Deloc  RadDisp  Strain   Sigma      Pi    PiPi    Spin\n")

    for i in range(dg.natoms):
        Z = cg.Z[i]
        Zeff = synthons.Zeff(i) if synthons is not None else 0.0
        charge = synthons.charge(i) if synthons is not None else 0.0
        coval = synthons.covalency(i) if synthons is not None else 0.0
        deloc = synthons.delocalization(i) if synthons is not None else 0.0
        radial = synthons.radial_dispersion(i) if synthons is not None else 0.0
        strain = synthons.strain(i) if synthons is not None else 0.0
        sigma = synthons.sigma_index(i) if synthons is not None else 0.0
        pi = synthons.pi_index(i) if synthons is not None else 0.0
        pi_pi = synthons.pi_pi_index(i) if synthons is not None else 0.0
        spin = spin_density[i] if spin_density is not None else 0.0

        fh.write(
            f"{i + 1:3d} {Z:3d} {Zeff:7.3f} {charge:7.3f} "
            f"{coval:7.3f} {deloc:7.3f} {radial:7.3f} {strain:7.3f} "
            f"{sigma:7.3f} {pi:7.3f} {pi_pi:7.3f} {spin:7.3f}\n"
        )

    # ========================================================
    # BONDS
    # ========================================================

    fh.write("\n[BONDS]\n")
    if dg.bonds:
        for i, j in dg.bonds:
            fh.write(f"{i + 1} {j + 1}\n")
    else:
        fh.write("NONE\n")

    transitional_contacts = tuple(getattr(dg, "transitional_contacts", ()))
    near_covalent_contacts = tuple(getattr(dg, "near_covalent_contacts", ()))
    shared_monovalent_candidates = tuple(
        (center, *sorted(dg.adjacency[center]))
        for center, atomic_number in enumerate(dg.Z)
        if is_bridging_ligand(int(atomic_number)) and len(dg.adjacency[center]) == 2
    )

    # This evidence is deliberately task-neutral.  The topology writer records
    # candidates only; ORACLE's versioned TS catalog owns classification and
    # chart prescription after a transition-state task is requested explicitly.
    fh.write("\n[TRANSITION_STATE_EVIDENCE]\n")
    fh.write("SOURCE SINGLE_GEOMETRY\n")
    fh.write(f"BREAKING_COUNT {len(transitional_contacts)}\n")
    fh.write(f"FORMING_COUNT {len(near_covalent_contacts)}\n")
    fh.write(f"SHARED_MONOVALENT_COUNT {len(shared_monovalent_candidates)}\n")
    for center, left, right in shared_monovalent_candidates:
        fh.write(f"SHARED_MONOVALENT {center + 1} {left + 1} {right + 1}\n")

    # Weak acyclic links are deliberately excluded from the constitutional
    # graph, but ORACLE must retain them as typed evidence for exact TS and
    # disconnected-fragment coordinate construction.  SMITH may use these
    # rows as conditioning candidates; they never become covalent bonds here.
    fh.write("\n[TRANSITIONAL_CONTACTS]\n")
    if transitional_contacts or near_covalent_contacts:
        for i, j in transitional_contacts:
            fh.write(f"{i + 1} {j + 1} KIND=WEAK_ACYCLIC\n")
        for i, j in near_covalent_contacts:
            fh.write(f"{i + 1} {j + 1} KIND=NEAR_COVALENT\n")
    else:
        fh.write("NONE\n")

    # ========================================================
    # BOND ORDERS
    # ========================================================

    fh.write("\n[BOND_ORDERS]\n")
    bond_order_rows = []
    if synthons is not None:
        for i, j in dg.bonds:
            try:
                bond_order_rows.append(f"{i + 1} {j + 1} {float(synthons.bond_order(i, j)):.10g}")
            except Exception:
                continue
    if bond_order_rows:
        fh.write("\n".join(bond_order_rows) + "\n")
    else:
        fh.write("NONE\n")

    fh.write("\n[BOND_ORDER_COMPONENTS]\n")
    fh.write(
        "COLUMNS ATOM_I ATOM_J BO BO_SIGMA BO_PI BO_PI_PI\n"
    )
    component_rows = []
    if synthons is not None:
        for i, j in dg.bonds:
            try:
                components = synthons.bond_order_components(i, j)
                component_rows.append(
                    f"{i + 1} {j + 1} {components.total:.10g} "
                    f"{components.sigma:.10g} {components.pi:.10g} {components.pi_pi:.10g}"
                )
            except Exception:
                continue
    fh.write("\n".join(component_rows) + "\n" if component_rows else "NONE\n")

    # ========================================================
    # RINGS
    # ========================================================

    if ringset is not None:
        fh.write("\n[RINGS]\n")
        if ringset.rings:
            for ir, ring in enumerate(ringset.rings, start=1):
                atoms = " ".join(str(i + 1) for i in ring.atoms)
                fh.write(f"{ir} SIZE={len(ring)} ATOMS={atoms}\n")
        else:
            fh.write("NONE\n")

        automorphism_orbits = (
            build_topology_automorphism_orbits(
                dg,
                ringset,
                synthons,
                tuple(int(value) for value in dg.Z),
                aromaticity=aromaticity,
            )
            if synthons is not None
            else ()
        )
        fh.write("\n" + "\n".join(topology_automorphism_lines(automorphism_orbits)) + "\n")

    # ========================================================
    # AROMATICITY
    # ========================================================

    if aromaticity is not None:
        fh.write("\n[AROMATICITY]\n")

        if aromaticity.aromatic_atoms:
            atoms = " ".join(str(i + 1) for i in sorted(aromaticity.aromatic_atoms))
            fh.write(f"ATOMS {atoms}\n")
        else:
            fh.write("ATOMS NONE\n")

        if aromaticity.aromatic_bonds:
            bonds = " ".join(f"{i + 1}-{j + 1}" for i, j in sorted(aromaticity.aromatic_bonds))
            fh.write(f"BONDS {bonds}\n")
        else:
            fh.write("BONDS NONE\n")
    # ========================================================
    # SMILES (optional)
    # ========================================================

    if smiles is not None:
        fh.write("\n[SMILES_Synthons]\n")
        if isinstance(smiles, (list, tuple)):
            for s in smiles:
                fh.write(f"{s}\n")
        else:
            fh.write(f"{smiles}\n")


# ======================================================================
# Helper functions for safe topology writing
# ======================================================================


def remove_topology_section(lines):
    """
    Remove the #TOPOLOGY section from a list of lines.

    The section is defined as starting from a line beginning with '#TOPOLOGY'
    and ending at the line immediately before the next line beginning with '#',
    or at EOF if no further section exists.

    Returns:
        before : list of lines before #TOPOLOGY
        after  : list of lines after the #TOPOLOGY section
    """
    start = None
    end = None

    for i, line in enumerate(lines):
        if line.strip().startswith("#TOPOLOGY"):
            start = i
            break

    if start is None:
        return lines[:], []

    for j in range(start + 1, len(lines)):
        if lines[j].strip().startswith("#"):
            end = j
            break

    if end is None:
        end = len(lines)

    before = lines[:start]
    after = lines[end:]

    return before, after


def write_topology_to_xyzin(
    xyz_filename,
    *,
    cg,
    dg,
    ringset=None,
    aromaticity=None,
    synthons=None,
    spin_density=None,
    smiles=None,
):
    """
    Write (or replace) the #TOPOLOGY section in an XYZ file.

    The file is always written to:
        <cwd>/working/<basename(xyz_filename)>

    Any existing #TOPOLOGY section is removed and replaced in the same position.
    """

    # ------------------------------------------------------
    # Resolve path according to ORACLE policy
    # ------------------------------------------------------
    basename = Path(xyz_filename).name
    xyz_path = Path.cwd() / "working" / basename

    # ------------------------------------------------------
    # Read existing file
    # ------------------------------------------------------
    with open(xyz_path, "r") as fh:
        lines = fh.readlines()

    before, after = remove_topology_section(lines)

    # ------------------------------------------------------
    # Write updated file
    # ------------------------------------------------------
    with open(xyz_path, "w") as fh:
        fh.writelines(before)

        if before and not before[-1].endswith("\n"):
            fh.write("\n")

        write_topology_section(
            fh,
            cg=cg,
            dg=dg,
            ringset=ringset,
            aromaticity=aromaticity,
            synthons=synthons,
            spin_density=spin_density,
            smiles=smiles,
        )

        if after:
            if not after[0].startswith("\n"):
                fh.write("\n")
            fh.writelines(after)
