"""
Topology construction pipeline for ORACLE.

This module centralizes the geometry-first construction of all
topological objects and descriptor/synthon objects, without performing any I/O
or workflow decisions.

Frozen contracts:
- Does NOT write files.
- Does NOT call RDKit.
- Does NOT alter topology semantics.
"""

from .continuous_graph import ContinuousGraph
from .discrete_graph import DiscreteGraph
from .ringset import RingSet
from .atomic_synthons import AtomicSynthons
from .aromaticity import Aromaticity

# ============================================================
# Public API
# ============================================================


def build_topology_objects(
    coords,
    Z,
    *,
    bond_order_overrides=None,
    external_charges=None,
    charge_source="ORACLE electronegativity estimate",
    bond_order_source="ORACLE Pauling estimate",
    force_aromatic=False,
    electronic_connectivity=False,
    candidate_pairs=None,
):
    """
    Build all topology-related objects from Cartesian coordinates.

    Parameters
    ----------
    coords : array-like, shape (N,3)
        Cartesian coordinates.
    Z : array-like, shape (N,)
        Atomic numbers.
    force_aromatic : bool, optional
        Passed to Aromaticity (no topology effect).

    Returns
    -------
    cg : ContinuousGraph
    dg : DiscreteGraph
    ringset : RingSet
    synthons : AtomicSynthons
    aromaticity : Aromaticity
    """

    # --------------------------------------------------------
    # Continuous topology (geometry-first)
    # --------------------------------------------------------
    # Perceive connectivity from geometry alone. Electronic observables never
    # create or delete graph edges.
    cg = ContinuousGraph(coords, Z, candidate_pairs_override=candidate_pairs)

    # --------------------------------------------------------
    # Discrete topology (H-robust)
    # --------------------------------------------------------
    dg = DiscreteGraph(cg)

    # Charges and bond orders are independent synthon observables.  A complete
    # CM5 vector is therefore retained even when Mayer data are unavailable,
    # and complete Mayer values are retained even when charges must fall back
    # to the scaled electronegativity estimate.  Each observable still falls
    # back as a whole vector, so partially populated QM arrays are never mixed
    # with geometry values atom by atom.
    candidate_orders = dict(bond_order_overrides or {})
    required_bonds = {tuple(sorted(pair)) for pair in dg.bonds}
    candidate_charges = dict(external_charges or {})
    orders_complete = required_bonds.issubset(candidate_orders)
    charges_complete = set(candidate_charges) == set(range(len(Z)))
    selected_orders = candidate_orders if orders_complete else {}
    selected_charges = candidate_charges if charges_complete else {}
    if selected_orders:
        cg = ContinuousGraph(
            coords,
            Z,
            bond_order_overrides=selected_orders,
            connectivity_overrides=(selected_orders if electronic_connectivity else None),
            candidate_pairs_override=candidate_pairs,
        )
        dg = DiscreteGraph(cg)

    # --------------------------------------------------------
    # Ring detection
    # --------------------------------------------------------
    ringset = RingSet(dg, coords=cg.coords)

    # --------------------------------------------------------
    # Atomic synthons (continuous descriptors)
    # --------------------------------------------------------
    neighbors = [list(dg.adjacency[i]) for i in range(dg.natoms)]
    synthons = AtomicSynthons(
        Z=cg.Z,
        coords=cg.coords,
        neighbors=neighbors,
        coordination_numbers=cg.coordination_numbers,
    )
    synthons._external_charges = selected_charges or None
    synthons._external_bond_orders = selected_orders or None
    synthons._charge_source = (
        charge_source if selected_charges else "ORACLE electronegativity estimate"
    )
    synthons._bond_order_source = (
        bond_order_source if selected_orders else "ORACLE Pauling estimate"
    )

    # --------------------------------------------------------
    # Aromaticity (ring-driven, discrete + geometry)
    # --------------------------------------------------------
    aromaticity = Aromaticity(
        graph=cg,
        discrete_graph=dg,
        ring_set=ringset,
        synthons=synthons,
        force_aromatic=force_aromatic,
    )

    return cg, dg, ringset, synthons, aromaticity


def perceive_aromatic_ring_atoms(coords, Z) -> tuple[tuple[int, ...], ...]:
    """Return geometry-first aromatic rings as zero-based atom tuples."""
    *_, aromaticity = build_topology_objects(coords, Z)
    return tuple(
        tuple(int(atom) for atom in assignment.atoms)
        for assignment in aromaticity.assignments
    )
