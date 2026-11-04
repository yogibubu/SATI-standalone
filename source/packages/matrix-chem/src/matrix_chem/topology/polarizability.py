"""
Atomic static isotropic polarizabilities (Å^3).

Gas-phase reference values.
Complete periodic table (Z = 1–118).
Values are None where not reliably available.
"""

POLARIZABILITY = {Z: None for Z in range(1, 119)}

# --- Light elements ---
POLARIZABILITY[1]  = 0.667
POLARIZABILITY[2]  = 0.205
POLARIZABILITY[3]  = 24.3
POLARIZABILITY[4]  = 5.6
POLARIZABILITY[5]  = 3.0
POLARIZABILITY[6]  = 1.76
POLARIZABILITY[7]  = 1.10
POLARIZABILITY[8]  = 0.80
POLARIZABILITY[9]  = 0.56
POLARIZABILITY[10] = 0.39

# --- Second and third row ---
POLARIZABILITY[11] = 24.1
POLARIZABILITY[12] = 10.6
POLARIZABILITY[13] = 6.8
POLARIZABILITY[14] = 5.38
POLARIZABILITY[15] = 3.63
POLARIZABILITY[16] = 2.90
POLARIZABILITY[17] = 2.18
POLARIZABILITY[18] = 1.64

# --- Halogens ---
POLARIZABILITY[35] = 3.05
POLARIZABILITY[53] = 5.35
POLARIZABILITY[85] = 7.40

# --- Selected metals ---
POLARIZABILITY[29] = 6.5
POLARIZABILITY[46] = 4.3
POLARIZABILITY[78] = 5.6
POLARIZABILITY[79] = 5.8
POLARIZABILITY[82] = 7.4


def polarizability(Z: int):
    """
    Return atomic static polarizability (Å^3).

    Parameters
    ----------
    Z : int
        Atomic number.

    Returns
    -------
    float or None
    """
    if Z <= 0:
        return None
    return POLARIZABILITY.get(Z, None)

