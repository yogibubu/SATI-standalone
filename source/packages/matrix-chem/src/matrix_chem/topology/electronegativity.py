"""
Pauling electronegativity scale.

Values are intended for qualitative analysis only
(e.g. bond polarity heuristics).
"""

PAULING = {
     1: 2.20,  2: None,
     3: 0.98,  4: 1.57,  5: 2.04,  6: 2.55,  7: 3.04,  8: 3.44,  9: 3.98, 10: None,
    11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 18: None,
    19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55,
    26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65,
    31: 1.81, 32: 2.01, 33: 2.18, 34: 2.55, 35: 2.96, 36: 3.00,
    37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33, 41: 1.60, 42: 2.16,
    43: 1.90, 44: 2.20, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69,
    49: 1.78, 50: 1.96, 51: 2.05, 52: 2.10, 53: 2.66, 54: 2.60,
    55: 0.79, 56: 0.89,
    57: 1.10, 58: 1.12, 59: 1.13, 60: 1.14, 61: 1.13, 62: 1.17,
    63: 1.20, 64: 1.20, 65: 1.10, 66: 1.22, 67: 1.23, 68: 1.24,
    69: 1.25, 70: 1.10, 71: 1.27,
    72: 1.30, 73: 1.50, 74: 2.36, 75: 1.90, 76: 2.20,
    77: 2.20, 78: 2.28, 79: 2.54, 80: 2.00, 81: 1.62,
    82: 2.33, 83: 2.02, 84: 2.00, 85: 2.20, 86: None,
    87: 0.70, 88: 0.90,
    89: 1.10, 90: 1.30, 91: 1.50, 92: 1.38,
    93: 1.36, 94: 1.28, 95: 1.13, 96: 1.28,
    97: None, 98: None, 99: None, 100: None,
    101: None, 102: None, 103: None,
    104: None, 105: None, 106: None, 107: None,
    108: None, 109: None, 110: None,
    111: None, 112: None, 113: None, 114: None,
    115: None, 116: None, 117: None, 118: None,
}

DEFAULT_SCALE = "pauling"


def electronegativity(Z: int):
    if Z <= 0:
        return None
    return PAULING.get(Z, None)


def bond_polarity(Z1: int, Z2: int):
    chi1 = electronegativity(Z1)
    chi2 = electronegativity(Z2)
    if chi1 is None or chi2 is None:
        return None
    return abs(chi1 - chi2)

