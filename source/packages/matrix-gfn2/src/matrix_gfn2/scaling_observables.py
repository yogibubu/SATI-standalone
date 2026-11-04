"""Self-consistent LINK-xTB descriptors used by transferable corrections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .sparse_observables import LazyAtomicPairMatrix


GFN2_SCALING_OBSERVABLES_SCHEMA = "matrix.gfn2.scaling_observables.v1"


def gfn2_scaling_observables(
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    result: Any,
) -> dict[str, Any]:
    """Serialize only descriptors evaluated by the resident GFN2 calculation.

    The artifact deliberately contains SCC charges rather than CM5 charges and
    Mayer bond orders from the same converged state.  It is therefore suitable
    for geometry/Hessian correction without importing observables from a
    higher-level electronic-structure calculation.
    """
    symbols = tuple(str(atom) for atom in atoms)
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.shape != (len(symbols), 3):
        raise ValueError("GFN2 scaling-observable geometry has inconsistent dimensions")
    charges = np.asarray(result.charges_electron, dtype=float).reshape(-1)
    if charges.shape != (len(symbols),):
        raise ValueError("GFN2 SCC charge vector has inconsistent dimensions")
    raw_bond_orders = result.bond_orders
    if isinstance(raw_bond_orders, LazyAtomicPairMatrix):
        bond_orders = raw_bond_orders.to_dense()
    else:
        bond_orders = np.asarray(raw_bond_orders, dtype=float)
    if bond_orders.shape != (len(symbols), len(symbols)):
        raise ValueError("GFN2 Mayer matrix has inconsistent dimensions")
    if not all(np.all(np.isfinite(item)) for item in (coordinates, charges, bond_orders)):
        raise ValueError("GFN2 scaling observables contain non-finite values")
    pairs = [
        [left + 1, right + 1, float(bond_orders[left, right])]
        for left in range(len(symbols))
        for right in range(left + 1, len(symbols))
    ]
    return {
        "schema": GFN2_SCALING_OBSERVABLES_SCHEMA,
        "backend": "LINK-XTB/MATRIX-GFN2",
        "charge_model": "GFN2-SCC",
        "bond_order_model": "Mayer",
        "atoms": list(symbols),
        "coordinates_angstrom": coordinates.tolist(),
        "charges_electron": charges.tolist(),
        "mayer_bond_orders": pairs,
        "iscf_iterations": int(result.iterations),
        "iscf_converged": bool(result.converged),
    }


def write_gfn2_scaling_observables(
    path: Path | str,
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    result: Any,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(gfn2_scaling_observables(atoms, coordinates_angstrom, result), indent=2)
        + "\n",
        encoding="utf-8",
    )
    return target


__all__ = [
    "GFN2_SCALING_OBSERVABLES_SCHEMA",
    "gfn2_scaling_observables",
    "write_gfn2_scaling_observables",
]
