"""Primitive grids with explicit, resolution-independent physical walls."""
from __future__ import annotations
import numpy as np

def confined_dvr_grid(lower: float, upper: float, count: int) -> np.ndarray:
    """Interior nodes for the LOCK positive Dirichlet flux operator.

    That operator puts zero ghost values one spacing outside the first and
    last nodes. Including the physical walls among the nodes would move those
    walls whenever the grid resolution changed.
    """
    if isinstance(count,bool) or not isinstance(count,(int,np.integer)) or count<3:
        raise ValueError('At least three confined DVR interior nodes required')
    if not np.isfinite(lower) or not np.isfinite(upper) or upper<=lower:
        raise ValueError('Finite, increasing physical wall positions required')
    return np.linspace(lower,upper,int(count)+2)[1:-1]
