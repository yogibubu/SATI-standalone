# ORACLE physical constants.

from enum import IntEnum
from typing import Dict


class Phy(IntEnum):
    TO_ANG = 1        # Angstrom per Bohr
    TO_KG = 2         # kg per amu
    TO_E = 3          # Coulomb per electron (converted to ESU)
    PLANCK = 4        # Planck constant (J s)
    AVOGADRO = 5
    JP_CAL = 6        # J per cal
    M_PER_B = 7       # meters per Bohr
    HARTREE = 8       # J per Hartree
    C_LIGHT = 9       # cm / s
    BOLTZMANN = 10
    FINE_STRUCT = 11
    E_MASS_KG = 12
    MOLAR_VOL = 13
    E_MAG_MOM = 14
    PROTON_MASS = 15
    G_FREE = 16


_CODATA_INDEX = {
    1979: 0,
    1986: 1,
    1998: 2,
    2006: 3,
    2010: 4,
    0: 4,     # default
}


def get_physical_constants(codata: int = 2010) -> Dict[Phy, float]:
    """
    Returns physical constants for a given CODATA epoch.
    """

    if codata not in _CODATA_INDEX:
        raise ValueError(f"Unsupported CODATA year: {codata}")

    i = _CODATA_INDEX[codata]

    # Raw data table (faithful to Fortran)
    data = {
        Phy.TO_ANG:      [0.52917706, 0.529177249, 0.5291772083, 0.52917720859, 0.52917721092],
        Phy.TO_KG:       [1.6605655e-27, 1.6605402e-27, 1.66053873e-27, 1.660538782e-27, 1.660538921e-27],
        Phy.TO_E:        [1.6021890717477622e-19, 1.6021890717477622e-19,
                           1.602176462e-19, 1.602176487e-19, 1.602176565e-19],
        Phy.PLANCK:      [6.626176e-34, 6.6260755e-34, 6.62606876e-34,
                           6.62606896e-34, 6.62606957e-34],
        Phy.AVOGADRO:    [6.022045e23, 6.0221367e23, 6.02214199e23,
                           6.02214179e23, 6.02214129e23],
        Phy.JP_CAL:      [4.184]*5,
        Phy.HARTREE:     [4.359814e-18, 4.3597482e-18, 4.35974381e-18,
                           4.35974394e-18, 4.35974434e-18],
        Phy.C_LIGHT:     [2.99792458e10]*5,
        Phy.BOLTZMANN:   [1.380662e-23, 1.380658e-23, 1.3806503e-23,
                           1.3806504e-23, 1.3806488e-23],
        Phy.FINE_STRUCT: [137.03602, 137.0359895, 137.03599976,
                           137.035999679, 137.035999074],
        Phy.MOLAR_VOL:   [22.41383e-3, 22.41410e-3, 22.413996e-3,
                           22.413996e-3, 22.4139679e-3],
        Phy.E_MAG_MOM:   [9.2847701e-24, 9.2847701e-24,
                           928.476362e-26, 928.476377e-26, 928.476430e-26],
        Phy.PROTON_MASS: [1.672623e-27, 1.672623e-27, 1.67262158e-27,
                           1.672621637e-27, 1.672621777e-27],
        Phy.G_FREE:      [2.002319304386]*3 + [2.0023193043622, 2.00231930436153],
    }

    # Extract selected CODATA column
    const = {k: v[i] for k, v in data.items()}

    # Derived quantities (exactly as in PhyFil)
    const[Phy.TO_E] *= const[Phy.C_LIGHT] / 10.0
    const[Phy.M_PER_B] = const[Phy.TO_ANG] * 1e-10
    const[Phy.FINE_STRUCT] = 1.0 / const[Phy.FINE_STRUCT]
    const[Phy.E_MASS_KG] = (
        const[Phy.HARTREE] * 1e4 /
        (const[Phy.C_LIGHT] * const[Phy.FINE_STRUCT])**2
    )

    return const
