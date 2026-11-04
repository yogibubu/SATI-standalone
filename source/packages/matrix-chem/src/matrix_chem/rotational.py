import numpy as np

from .physical_constants import Phy, get_physical_constants
from .inertia import principal_moments
from .structure import Structure


# ============================================================
# Rotational constants
# ============================================================
def rotational_constants_MHz(
    structure: Structure,
    isotopic: bool = True,
):
    """
    Compute rotational constants A, B, C in MHz.

    Returned order is ALWAYS A >= B >= C.
    """
    pc = get_physical_constants()

    # principal moments in amu Å^2
    I = principal_moments(structure, isotopic=isotopic)

    # physical constants
    h = pc[Phy.PLANCK]  # J s
    amu_to_kg = pc[Phy.TO_KG]  # kg / amu
    ang_to_m = 1.0e-10  # m / Å

    Bvals = []
    for Ii in I:
        Ii_SI = Ii * amu_to_kg * ang_to_m**2
        if Ii_SI <= 0.0:
            Bi = 0.0
        else:
            Bi = h / (8.0 * np.pi**2 * Ii_SI) * 1.0e-6  # MHz
        Bvals.append(Bi)

    # Sort so that A >= B >= C
    A, B, C = sorted(Bvals, reverse=True)
    return float(A), float(B), float(C)


# ============================================================
# Ray asymmetry parameter
# ============================================================
def ray_asymmetry(A: float, B: float, C: float) -> float:
    """
    Ray's asymmetry parameter κ.

    κ = (2B - A - C) / (A - C)

    Defined for non-linear rotors.
    """
    if abs(A - C) < 1.0e-12:
        return 0.0
    return (2.0 * B - A - C) / (A - C)


# ============================================================
# Rotor classification
# ============================================================
def classify_rotor(
    A: float,
    B: float,
    C: float,
    eps_zero: float = 1.0e-6,
    eps_rel: float = 1.0e-3,
):
    """
    Classify rotor type from rotational constants (MHz).

    Returns
    -------
    rotor_type : str
        'linear'
        'spherical'
        'symmetric_prolate'
        'symmetric_oblate'
        'asymmetric_prolate'
        'asymmetric_oblate'
    """
    # Linear rotor: one constant ~ 0
    if C < eps_zero:
        return "linear"

    # Spherical top
    if abs(A - B) / A < eps_rel and abs(B - C) / B < eps_rel:
        return "spherical"

    # Symmetric tops
    if abs(B - C) / B < eps_rel:
        return "symmetric_prolate"  # A >> B ≈ C

    if abs(A - B) / A < eps_rel:
        return "symmetric_oblate"  # A ≈ B >> C

    # Asymmetric tops: classify via Ray's kappa
    kappa = ray_asymmetry(A, B, C)

    if kappa < 0.0:
        return "asymmetric_prolate"
    else:
        return "asymmetric_oblate"


# ============================================================
# Effective constants helpers
# ============================================================
def effective_B(A: float, B: float, C: float):
    """
    Effective B constant (MHz) for quasi-linear or approximate treatments.
    """
    return max(B, C)


# ============================================================
# High-level convenience wrapper
# ============================================================
def rotational_info(
    structure: Structure,
    isotopic: bool = True,
):
    """
    Compute rotational constants and classify the rotor.

    Returns
    -------
    dict with keys:
      A, B, C           (MHz)
      rotor_type        (string)
      kappa             (Ray asymmetry)
      Beff              (effective B, MHz)
    """
    A, B, C = rotational_constants_MHz(structure, isotopic=isotopic)
    rotor_type = classify_rotor(A, B, C)
    kappa = ray_asymmetry(A, B, C)
    Beff = effective_B(A, B, C)

    return {
        "A": A,
        "B": B,
        "C": C,
        "rotor_type": rotor_type,
        "kappa": kappa,
        "Beff": Beff,
    }
