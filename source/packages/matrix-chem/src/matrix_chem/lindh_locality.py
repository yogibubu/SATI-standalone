"""Conservative, density-qualified tail bound for the xTB model pair blocks.

The bound is in Cartesian atomic Hessian units, BEFORE ANC conditioning.
No far pairs are enumerated. Occupancy of fixed-size cells bounds the count
in every distant lattice shell. At fixed occupancy/tolerance the radius is
independent of atom count. This is not a bound on transformed/conditioned SH.
"""
import math
from collections import Counter
from itertools import product
import numpy as np
from . import lindh_hessian as ref


def certified_pair_radius(xyz_bohr, numbers, tolerance, *, s6=20., cell_width=4.):
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('pair error tolerance must be finite and positive')
    xyz = np.asarray(xyz_bohr)
    cells = np.floor(xyz / cell_width).astype(np.int64)
    occupancy = max(Counter(map(tuple, cells)).values())
    species = sorted(set(numbers))
    periods = [ref._period_row(z) for z in species]
    amin = min(ref._A_AV[i,j] for i in periods for j in periods)
    gaussian_prefactor = max(ref._STRETCH_FORCE * math.exp(ref._A_AV[i,j]*ref._R_AV[i,j]**2)
                             for i in periods for j in periods)
    radii = [float(ref._VDW_RADII_ANGSTROM[z-1]) if z <= len(ref._VDW_RADII_ANGSTROM) else 2.
             for z in species]
    r0min, r0max = 2*min(radii)/.52917721, 2*max(radii)/.52917721
    amax = 20/r0min
    cmax = s6*max(ref._D2_C6[z-1] for z in species)
    radius = max(math.sqrt(ref._PAIR_CUTOFF2_BOHR), math.sqrt(4/amin), r0max, 2*cell_width)
    def bound(r):
        # Each 3x3 D2 element is bounded by
        # C/r^8 * [54 + 14 a r e + 3 a^2 r^2 e], e=exp[-20(r/r0max-1)].
        # ||block||_2 <= 3 max|element|. For r>=r0max, r^p e decreases (p<=2).
        e = math.exp(-20*(r/r0max-1))
        d2 = 3*cmax*(54+14*amax*r*e+3*(amax*r)**2*e)
        stretch = gaussian_prefactor*r**8*math.exp(-amin*r*r)
        # Shell k contains 24 k^2+2 cells, each at r >= (k-1)h.
        # Include all near cells pessimistically at r=R; integrate the tail
        # using (24 k^2+2)/(k-1)^8 <= 98/(k-1)^6 for k>=2.
        k = math.ceil(r/cell_width)+1
        lattice_tail = (2*k+1)**3/r**8 + 98/(5*cell_width**8*(k-1)**5)
        # A pair contributes both an off-diagonal and a diagonal block.
        return float(2*occupancy*(stretch+d2)*lattice_tail)
    for _ in range(256):
        error = bound(radius)
        if error <= tolerance:
            return radius, {'raw_pair_error_bound_hartree_per_bohr2': error,
                'requested_raw_pair_error_tolerance': float(tolerance),
                'bound_kind': 'SYMMETRIC_BLOCK_ROW_SUM_CELL_OCCUPANCY',
                'density_cell_width_bohr': cell_width,
                'maximum_cell_occupancy': occupancy,
                'pair_radius_bohr': radius,
                'bound_applies_to': 'RAW_CARTESIAN_MODEL_ONLY'}
        radius *= 1.15
    raise RuntimeError('could not certify xTB pair radius')


def linked_cell_neighbors(xyz, radius):
    """Expected O(N+number of local pairs); no global tree/sort/all-pair loop."""
    cells = {}
    keys = []
    for i, point in enumerate(xyz):
        key = tuple(math.floor(float(v)/radius) for v in point)
        keys.append(key)
        cells.setdefault(key, []).append(i)
    neighbors = [set() for _ in xyz]
    r2 = radius*radius
    offsets = tuple(product((-1,0,1), repeat=3))
    for i, key in enumerate(keys):
        for offset in offsets:
            for j in cells.get(tuple(a+b for a,b in zip(key,offset)), ()):
                if j >= i:
                    continue
                delta = xyz[i]-xyz[j]
                if float(delta@delta) <= r2:
                    if float(delta@delta) <= np.finfo(float).eps**2:
                        raise ValueError('coincident atoms are invalid for the Lindh Hessian')
                    neighbors[i].add(j)
                    neighbors[j].add(i)
    return neighbors
