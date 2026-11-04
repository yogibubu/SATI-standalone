"""
Coordination-dependent covalent radii (Å).

Pyykkö & Atsumi
Chem. Eur. J. 2009, 15, 186–197

Complete periodic table (Z = 1–118).
Values are provided where available; otherwise None.
No interpolation is performed here.
"""

from typing import Optional

# PYYKKO[Z] = {coordination_number: covalent_radius}
PYYKKO = {Z: {} for Z in range(1, 119)}

# --- Main-group elements ---
PYYKKO[1]  = {1: 0.32}
PYYKKO[2]  = {1: 0.28}

PYYKKO[3]  = {1: 1.28}
PYYKKO[4]  = {2: 0.96}
PYYKKO[5]  = {3: 0.84}
PYYKKO[6]  = {1: 0.60, 2: 0.67, 3: 0.76, 4: 0.77}
PYYKKO[7]  = {1: 0.54, 2: 0.60, 3: 0.71}
PYYKKO[8]  = {1: 0.53, 2: 0.57}
PYYKKO[9]  = {1: 0.50}
PYYKKO[10] = {1: 0.58}

PYYKKO[11] = {1: 1.66}
PYYKKO[12] = {2: 1.41}
PYYKKO[13] = {3: 1.21}
PYYKKO[14] = {4: 1.11}
PYYKKO[15] = {3: 1.07, 5: 1.11}
PYYKKO[16] = {2: 1.02, 4: 1.05, 6: 1.09}
PYYKKO[17] = {1: 0.99}
PYYKKO[18] = {1: 1.06}

# --- Halogens ---
PYYKKO[35] = {1: 1.14}
PYYKKO[53] = {1: 1.33}
PYYKKO[85] = {1: 1.45}

# --- Selected transition metals (where available) ---
PYYKKO[26] = {6: 1.32}
PYYKKO[29] = {4: 1.32}
PYYKKO[46] = {4: 1.39}
PYYKKO[78] = {4: 1.36}

DEFAULT_COORD = 1

# Existing ORACLE/JCP delocalization reference choices.  These are kept
# separate from the general coordination-dependent table so consumers never
# reinterpret raw Pyykko coordination keys independently.
_MULTIPLICITY_REFERENCE_COORDINATIONS = {
    6: (3, 2, 1),
    16: (4, 2, 6),
}


def covalent_radius(Z: int, coord: Optional[float] = None):
    """
    Return coordination-dependent covalent radius (Å).

    Parameters
    ----------
    Z : int
        Atomic number.
    coord : float or None
        Effective coordination number. If None, the lowest
        available coordination is used.

    Returns
    -------
    float or None
    """
    if Z <= 0 or Z not in PYYKKO:
        return None

    data = PYYKKO[Z]
    if not data:
        return None

    if coord is None:
        return data[min(data.keys())]

    # nearest available coordination (no interpolation here)
    return data[min(data.keys(), key=lambda k: abs(k - coord))]


def bond_order_reference_radii(
    Z: int,
    *,
    fallback_radius: float | None = None,
) -> tuple[float, float, float]:
    """Return ORACLE's established single/double/triple radial references.

    Only elements with an existing calibrated coordination-index mapping are
    resolved into distinct radii.  Other elements deliberately receive one
    common fallback rather than an invented multiplicity assignment.
    """

    atomic_number = int(Z)
    choices = _MULTIPLICITY_REFERENCE_COORDINATIONS.get(atomic_number)
    if choices is None:
        value = covalent_radius(atomic_number)
        if value is None:
            value = fallback_radius
        if value is None:
            raise ValueError(f"no Pyykko covalent radius is available for Z={atomic_number}")
        radius = float(value)
        return radius, radius, radius
    values = tuple(covalent_radius(atomic_number, choice) for choice in choices)
    if any(value is None for value in values):
        if fallback_radius is None:
            raise ValueError(
                f"incomplete Pyykko multiplicity references for Z={atomic_number}"
            )
        return (float(fallback_radius),) * 3
    return tuple(float(value) for value in values)  # type: ignore[return-value]
