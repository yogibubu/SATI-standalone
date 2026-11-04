"""
van der Waals radii (Å).
"""

# Merz–Kollman / Bondi (Gaussian-style)
MERZ_KOLLMAN = {
    1: 1.20,  2: 1.20,  3: 1.37,  4: 1.45,  5: 1.45,
    6: 1.50,  7: 1.50,  8: 1.40,  9: 1.35, 10: 1.30,
    11: 1.57, 12: 1.36, 13: 1.24, 14: 1.17, 15: 1.90,
    16: 1.85, 17: 1.80, 18: 1.88,
    19: 2.75, 20: None,
    # many transition metals intentionally undefined
    28: 1.63, 29: 1.40, 30: 1.39,
    31: 1.87, 32: 1.86, 33: 2.00, 34: 2.00, 35: 1.95,
    36: 2.02,
    46: 1.63, 47: 1.72, 48: 1.58,
    49: 1.93, 50: 2.17, 51: 2.20, 52: 2.20, 53: 2.15,
    54: 2.16,
    78: 1.72, 79: 1.66, 80: 1.55, 81: 1.96, 82: 1.02,
    92: 1.86,
}

# UFF (Rappe et al.) diameters converted to radii; ORACLE RVdW97 table.
_UFF_DIAMETERS = [
    2.886, 2.362, 2.451, 2.745, 4.083, 3.851, 3.660, 3.500, 3.364, 3.243,
    2.983, 3.021, 4.499, 4.295, 4.147, 4.035, 3.947, 3.868, 3.812, 3.399,
    3.295, 3.175, 3.144, 3.023, 2.961, 2.912, 2.872, 2.834, 3.495, 2.763,
    4.383, 4.280, 4.230, 4.205, 4.189, 4.141, 4.114, 3.641, 3.345, 3.124,
    3.165, 3.052, 2.998, 2.963, 2.929, 2.899, 3.148, 2.848, 4.463, 4.392,
    4.420, 4.470, 4.500, 4.404, 4.517, 3.703, 3.522, 3.556, 3.606, 3.575,
    3.547, 3.520, 3.493, 3.368, 3.451, 3.428, 3.409, 3.391, 3.374, 3.355,
    3.640, 3.141, 3.170, 3.069, 2.954, 3.120, 2.840, 2.754, 3.293, 2.705,
    4.347, 4.297, 4.370, 4.709, 4.750, 4.765, 4.900, 3.677, 3.478, 3.396,
    3.424, 3.395, 3.424, 3.424, 3.381, 3.326, 3.339, 3.313, 3.299, 3.286,
    3.274, 3.248, 3.236,
]
UFF = {Z: None for Z in range(1, 119)}
UFF.update({Z: 0.5 * value for Z, value in enumerate(_UFF_DIAMETERS, start=1)})

# UFF vdW well depths D_i in kcal/mol through Lr.  The values and the x_i
# diameters above are from Table I of Rappe et al., JACS 114, 10024 (1992),
# DOI:10.1021/ja00051a040 and checked against the UFF parameter table shipped
# by Open Babel; atom types of a given element share the same x_i and D_i.
_UFF_WELL_DEPTHS = (
    0.044, 0.056, 0.025, 0.085, 0.180, 0.105, 0.069, 0.060, 0.050, 0.042,
    0.030, 0.111, 0.505, 0.402, 0.305, 0.274, 0.227, 0.185, 0.035, 0.238,
    0.019, 0.017, 0.016, 0.015, 0.013, 0.013, 0.014, 0.015, 0.005, 0.124,
    0.415, 0.379, 0.309, 0.291, 0.251, 0.220, 0.040, 0.235, 0.072, 0.069,
    0.059, 0.056, 0.048, 0.056, 0.053, 0.048, 0.036, 0.228, 0.599, 0.567,
    0.449, 0.398, 0.339, 0.332, 0.045, 0.364, 0.017, 0.013, 0.010, 0.010,
    0.009, 0.008, 0.008, 0.009, 0.007, 0.007, 0.007, 0.007, 0.006, 0.228,
    0.041, 0.072, 0.081, 0.067, 0.066, 0.037, 0.073, 0.080, 0.039, 0.385,
    0.680, 0.663, 0.518, 0.325, 0.284, 0.248, 0.050, 0.404, 0.033, 0.026,
    0.022, 0.022, 0.019, 0.016, 0.014, 0.013, 0.013, 0.013, 0.012, 0.012,
    0.011, 0.011, 0.011,
)
UFF_WELL_DEPTH_KCAL = {
    atomic_number: value
    for atomic_number, value in enumerate(_UFF_WELL_DEPTHS, start=1)
}

def descriptor_vdw_radius(Z: int):
    """Return the unique radius used by ORACLE/SMITH descriptors."""
    if Z <= 0:
        return 0.0
    return MERZ_KOLLMAN.get(Z, None)


def uff_vdw_radius(Z: int):
    """Return the UFF radius used only with the UFF non-bonded potential."""
    if Z <= 0:
        return 0.0
    return UFF.get(Z, None)


def uff_well_depth_kcal(Z: int):
    return UFF_WELL_DEPTH_KCAL.get(int(Z))
