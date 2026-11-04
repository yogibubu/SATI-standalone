from __future__ import annotations

import math

from matrix_chem import Phy, get_physical_constants


def conversion_factor(key: str, codata: int = 2010) -> float:
    """Vibrational conversion factors from reference Fortran CnvFct."""
    if not key:
        raise ValueError("empty conversion factor key")

    k = key.strip().upper()
    phy = get_physical_constants(codata)
    to_ang = phy[Phy.TO_ANG]
    to_kg = phy[Phy.TO_KG]
    to_e = phy[Phy.TO_E]
    planck = phy[Phy.PLANCK]
    avogadro = phy[Phy.AVOGADRO]
    jp_cal = phy[Phy.JP_CAL]
    hartree = phy[Phy.HARTREE]
    c_light = phy[Phy.C_LIGHT]
    boltzmann = phy[Phy.BOLTZMANN]
    fine_struct = phy[Phy.FINE_STRUCT]
    electron_mass = phy[Phy.E_MASS_KG]

    pi = math.pi
    m2ang = 1.0e10
    hbar = planck * m2ang**2 / (2.0 * pi * to_kg)
    factg = 2.0 * pi * c_light / hbar
    hc = planck * c_light * 1.0e18

    if k == "FAC0AU":
        return hartree * 1.0e18 / hc
    if k in ("FAC1AU", "FACT1"):
        value = 1.0 / (math.sqrt(factg) * hc)
        return value * hartree * 1.0e18 / to_ang if k == "FAC1AU" else value
    if k in ("FAC2AU", "FACT2"):
        value = 1.0 / (factg * hc)
        return value * hartree * 1.0e18 / to_ang**2 if k == "FAC2AU" else value
    if k in ("FAC3AU", "FACT3"):
        value = 1.0 / (factg * math.sqrt(factg) * hc)
        return value * hartree * 1.0e18 / to_ang**3 if k == "FAC3AU" else value
    if k in ("FAC4AU", "FACT4"):
        value = 1.0 / (factg**2 * hc)
        return value * hartree * 1.0e18 / to_ang**4 if k == "FAC4AU" else value
    if k == "FACTG":
        return factg
    if k == "FACTA":
        return 1.0 / (2.0 * factg)
    if k == "FACTB":
        return planck * c_light / (2.0 * boltzmann)
    if k == "FACTC":
        return boltzmann * m2ang**2 / (c_light**2 * to_kg * to_ang**2)
    if k == "HBAR":
        return hbar
    if k == "TOUMA":
        return to_kg / electron_mass
    if k == "PICH12":
        return pi * math.sqrt(c_light * to_kg / planck / m2ang**2)
    if k == "AU2DEB":
        return 1.0e10 * to_e * to_ang
    if k == "MWQ2Q":
        return 1.0 / (math.sqrt(factg) * to_ang)
    if k == "AU2CM1":
        return hartree / (planck * c_light)
    if k == "AU2AMU":
        return electron_mass / to_kg
    if k == "AU2KJM":
        return hartree * avogadro / 1.0e3
    if k == "AU2KCAM":
        return hartree * avogadro / (1.0e3 * jp_cal)
    if k == "FACG0AU":
        return 1.0
    if k == "FACG1AU":
        return 1.0 / (to_ang * math.sqrt(factg))
    if k == "FACG2AU":
        return 1.0 / (to_ang**2 * factg)
    if k == "FINE_STRUCT":
        return fine_struct
    raise ValueError(f"unrecognized vibrational conversion factor: {key}")
