import math
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from sympy.physics.wigner import wigner_3j, wigner_6j
from sympy import S
import re

from functools import lru_cache

# This module is the spectroscopy engine loaded both by the browser UI
# (through Pyodide) and by the local fitting helpers. The guiding rule is to
# keep one Hamiltonian implementation that stays close to the operative
# SPCAT/SPFIT raw-card semantics used for prediction and fitting.

try:
    from matrix_numerics import eigh_arrays as _matrix_eigh_arrays

except Exception:
    _matrix_eigh_arrays = None


def _matrix_eigh(matrix):
    if _matrix_eigh_arrays is None:
        return np.linalg.eigh(matrix)
    return _matrix_eigh_arrays(matrix)

@lru_cache(maxsize=None)
def W3J(j1_2, j2_2, j3_2, m1_2, m2_2, m3_2) -> float:
    """Cached 3-j symbol helper using doubled quantum numbers as integers."""
    return float(
        wigner_3j(
            S(j1_2)/2, S(j2_2)/2, S(j3_2)/2,
            S(m1_2)/2, S(m2_2)/2, S(m3_2)/2
        )
    )

@lru_cache(maxsize=None)
def W6J(j1_2, j2_2, j3_2, j4_2, j5_2, j6_2) -> float:
    """Cached 6-j symbol helper using doubled quantum numbers as integers."""
    return float(
        wigner_6j(
            S(j1_2)/2, S(j2_2)/2, S(j3_2)/2,
            S(j4_2)/2, S(j5_2)/2, S(j6_2)/2
        )
    )

def clear_wigner_cache():
    """Reset cached Wigner symbols after large scans or fitting loops."""
    W3J.cache_clear()
    W6J.cache_clear()

H_PLANCK = 6.62607015e-34  # J s
K_BOLTZ  = 1.380649e-23    # J K^-1
TAU      = 1.0e6           # MHz -> Hz

REDUCTION = 'S'
REPRESENTATION = 'Ir'

STR0 = -12.0
STR1 = -10.0

FREQ_MIN = 0.0
FREQ_MAX = 20000.0 # MHz
STEP = 1e-4  # MHz

eeWt=1
eoWt=1
oeWt=1
ooWt=1

# Hyperfine quadrupole (direct-F / Pickett-style spin normalization)
I_NUC = 0.0
# Legacy compatibility knob. Quadrupole normalization is now handled
# analytically in Pickett style, so this value is ignored.
CHI_SCALE = 1.0
chi_aa = 0.0
chi_bb = 0.0
chi_cc = 0.0
chi_ab = 0.0
chi_ac = 0.0
chi_bc = 0.0
F_MAX = None
HFS_INTENSITY_MODE = "recoupled"  # "recoupled"|"full"|"6j2"|"sqrt"
S_D2_ID = "auto"  # "auto"|41000|50000
# Fix the gauge of quasi-degenerate asymmetric-rotor eigenvectors by
# diagonalizing K^2 inside tiny energy clusters. Pickett applies an
# equivalent ordering/fixing step; leaving the gauge free causes weak-line
# intensity drift in near-degenerate oblate S cases.
DEGEN_TOL = 1.0e-8  # MHz

QUAD_CART_KEYS = ("chi_aa", "chi_bb", "chi_cc", "chi_ab", "chi_ac", "chi_bc")

LAST_DF_HF = None
LAST_QROT_USED = None
LAST_QROT_SOURCE = None
LAST_SIM_ENGINE = None
LAST_WAVEFUNC_CACHE = None
LAST_SIM_CACHE = None
LAST_SIM_REUSED = False


def _float_or_default(value, default=0.0):
    """
    Convert a scalar-like value to float with a fallback.

    Parameters
    ----------
    value : Any
        Input object that should represent a numeric value.
    default : float
        Value returned when conversion fails.

    Returns
    -------
    float
        Parsed floating-point value, or ``default`` if parsing fails.
    """
    try:
        return float(value)
    except Exception:
        return float(default)

def _read_quadrupole_nucleus(index, rep=None):
    """
    Read one quadrupolar nucleus from the module globals and normalize it.

    The web UI writes user inputs into module globals before invoking the
    engine. Centralizing the normalization here makes the HFS path easier to
    audit: every caller sees the same Cartesian tensor, spherical tensor, and
    Pickett-style reduced-spin scaling.
    """
    if int(index) == 1:
        i_val = _float_or_default(globals().get("I_NUC_1", globals().get("I_NUC", I_NUC)), I_NUC)
        chi_cart = tuple(
            _float_or_default(globals().get(f"{key}_1", globals().get(key, globals()[key])), globals()[key])
            for key in QUAD_CART_KEYS
        )
    elif int(index) == 2:
        i_val = _float_or_default(globals().get("I_NUC_2", 0.0), 0.0)
        chi_cart = tuple(
            _float_or_default(globals().get(f"{key}_2", 0.0), 0.0)
            for key in QUAD_CART_KEYS
        )
    else:
        raise ValueError(f"Unsupported quadrupole nucleus index: {index}")

    chi_dict = chi_cart_to_sph_q(*chi_cart, rep=rep)
    pickett_spin_reduced = pickett_quadrupole_reduced_spin_factor(i_val)
    if pickett_spin_reduced > 0.0:
        chi_pickett_dict = {q: value / pickett_spin_reduced for q, value in chi_dict.items()}
    else:
        chi_pickett_dict = dict(chi_dict)
    active = (i_val > 0.5) and any(abs(v) > 1e-15 for v in chi_cart)
    return {
        "index": int(index),
        "I": float(i_val),
        "chi_cart": tuple(float(v) for v in chi_cart),
        "chi_dict": chi_dict,
        "chi_pickett_dict": chi_pickett_dict,
        "pickett_spin_reduced": float(pickett_spin_reduced),
        "active": bool(active),
    }

def _active_quadrupole_nuclei(rep=None):
    """Return only nuclei that really contribute to the Hamiltonian."""
    return [nuc for nuc in (_read_quadrupole_nucleus(1, rep=rep), _read_quadrupole_nucleus(2, rep=rep)) if nuc["active"]]


def _has_active_quadrupole(rep=None):
    """Cheap predicate used to route between pure-rotational and HFS paths."""
    return len(_active_quadrupole_nuclei(rep=rep)) > 0


def _snapshot_quadrupole_globals():
    """
    Capture the quadrupole-related module globals in a plain dictionary.

    The simulator temporarily rewrites these globals in the oblate internal
    remap path. This helper returns a restorable snapshot containing nuclear
    spins, F-limit, and all Cartesian chi components for both nuclei.
    """
    snap = {
        "I_NUC": globals().get("I_NUC", I_NUC),
        "F_MAX": globals().get("F_MAX", F_MAX),
    }
    for key in QUAD_CART_KEYS:
        snap[key] = globals().get(key, globals()[key])
    for idx in (1, 2):
        snap[f"I_NUC_{idx}"] = globals().get(f"I_NUC_{idx}", None)
        for key in QUAD_CART_KEYS:
            snap[f"{key}_{idx}"] = globals().get(f"{key}_{idx}", None)
    return snap


def _restore_quadrupole_globals(snap):
    """
    Restore quadrupole globals from a snapshot produced earlier.

    Parameters
    ----------
    snap : dict
        Snapshot returned by :func:`_snapshot_quadrupole_globals`.

    Returns
    -------
    None
        The module globals are updated in place.
    """
    for key, value in snap.items():
        if value is None:
            globals().pop(key, None)
        else:
            globals()[key] = value


def _apply_oblate_quadrupole_remap():
    """
    Apply the same a <-> c remap used by SPCAT in the oblate family.

    The symmetric-oblate shortcut temporarily rewrites rotational constants,
    dipoles, and quadrupole tensors into the internal convention used by the
    matrix builder. A snapshot is returned so the caller can restore the
    original user-facing globals afterwards.
    """
    snap = _snapshot_quadrupole_globals()
    for idx in (1, 2):
        nuc = _read_quadrupole_nucleus(idx)
        mapped = _spcat_oblate_remap_chi(*nuc["chi_cart"])
        for key, value in zip(QUAD_CART_KEYS, mapped):
            globals()[f"{key}_{idx}"] = float(value)
            if idx == 1:
                globals()[key] = float(value)
    if snap.get("I_NUC_1", None) is not None:
        globals()["I_NUC"] = snap["I_NUC_1"]
    return snap

# ================== Utils comuni ==================
#----------------------------------------------------------------------
# SPCAT constants for ID parsing
NDECPAR = 5      # number of digit pairs for parameter ID
VIBDEC = 1       # vibdec for nvib <= 9
NBCD = (NDECPAR + 1) + VIBDEC  # like setopt()


def _getbcd_digits(num_str, nbcd):
    """
    Mimic getbcd() digit packing used by SPCAT.
    """
    iv = [0] * nbcd
    kk = len(num_str)
    for k in range(1, nbcd):
        if kk <= 0:
            break
        kk -= 1
        low = int(num_str[kk]) if kk >= 0 else 0
        high = 0
        if kk - 1 >= 0:
            kk -= 1
            high = int(num_str[kk])
        iv[k] = (high << 4) | low
    return iv


def _bcd2i(b):
    """
    Decode one Pickett BCD-packed byte into the corresponding integer.

    Input is one byte with decimal digits stored as high/low nibbles; output
    is the corresponding base-10 integer used by the parameter-ID parser.
    """
    return (b & 0x0F) + 10 * ((b >> 4) & 0x0F)


def _spcat_zfac_quartic(ity, reduction=None):
    """
    SPCAT/Pickett zfac for the subset we need (quartic Watson terms).
    Mirrors spinv.c:idpari + pasort for sign/scale.
    """
    zfac = 1.0
    if ity > 3:
        itmp = ity
        if ity >= 92:
            itmp = (ity - 92) >> 1
        if itmp <= 11 and (itmp % 2 == 0):
            zfac = -zfac
        # factor of 2 for delta K != 0 (pasort)
        if ity < 12:
            zfac *= 2.0

    return zfac


def parse_id_quartic(pid, vibdec=VIBDEC, reduction=None):
    """
    Minimal ID parsing for quartic Watson Ir terms, mirroring SPCAT logic
    enough to reproduce matrix elements.

    Returns: nsx (power of N^2), ksq (power of K^2), ity, zfac
    """
    iv = _getbcd_digits(str(abs(pid)), NBCD)
    idval = iv[1 + vibdec:]  # skip length and vib field

    nsx = idval[0] & 0x0F
    ksq = (idval[0] >> 4) & 0x0F
    ity = _bcd2i(idval[1])

    zfac = _spcat_zfac_quartic(ity, reduction=reduction)

    return nsx, ksq, ity, zfac


REP_MAP = {
    "Ir":   ("a","b","c"),
    "IIr":  ("b","c","a"),
    "IIIr": ("c","a","b"),
    "Il":   ("a","c","b"),
    "IIl":  ("b","a","c"),
    "IIIl": ("c","b","a"),
}

# REP_MAP keeps the full six labels for I/O compatibility. The active
# Hamiltonian path below uses the reduced Pickett raw-card semantics, where
# only the handedness family matters for operator placement.
PICKETT_ACTIVE_AXES = {
    "r": ("a", "b", "c"),
    "l": ("c", "b", "a"),
}

def _normalize_rep_key(rep):
    """
    Normalize a user-facing representation token to the canonical label.

    Parameters
    ----------
    rep : Any
        Representation label such as ``Ir`` or ``IIIl``.

    Returns
    -------
    str
        Canonical representation name exactly as stored in ``REP_MAP``.
    """
    token = str(rep).strip().replace(" ", "")
    for key in REP_MAP:
        if token.lower() == key.lower():
            return key
    raise ValueError(f"Unknown representation '{rep}'. Use Ir, IIr, IIIr, Il, IIl, or IIIl.")


def _pickett_rep_handedness(rep):
    """
    Reduce a six-label representation to its operative handedness family.

    Returns ``"r"`` for the almost-prolate/right-handed family and ``"l"``
    for the almost-oblate/left-handed family used by the raw Pickett path.
    """
    rep = _normalize_rep_key(rep)
    return "l" if rep.lower().endswith("l") else "r"


def _pickett_rep_axes(rep):
    """
    SPCAT raw-card semantics collapse I/II/III within the same handedness.

    The full six-fold REP_MAP remains the source of truth for explicit
    representation transforms, but the active Hamiltonian/dipole axes used by
    SPCAT itself depend only on the raw handedness field.
    """
    hand = _pickett_rep_handedness(rep)
    return PICKETT_ACTIVE_AXES[hand]

def _active_rep_axes(rep):
    """
    Return the principal-axis assignment actually used by the active engine.

    The output is a tuple ``(z_axis, x_axis, y_axis)`` after collapsing the
    six user-facing labels onto the two Pickett operative families.
    """
    return _pickett_rep_axes(rep)

def _signature_rep_key(rep):
    """
    Normalize a representation for cache/signature keys without failing hard.

    Valid representations are canonicalized; unknown values are converted to
    strings so diagnostic signatures can still be built.
    """
    try:
        return _normalize_rep_key(rep)
    except Exception:
        return str(rep)

def _cart_tensor_tuple_to_matrix(aa, bb, cc, ab, ac, bc):
    """
    Build a symmetric 3x3 Cartesian tensor matrix from six components.

    Inputs are the independent ``aa, bb, cc, ab, ac, bc`` components in the
    principal-axis basis; output is a NumPy array ready for remapping or
    conversion to spherical rank-2 components.
    """
    return np.array(
        [
            [aa, ab, ac],
            [ab, bb, bc],
            [ac, bc, cc],
        ],
        dtype=float,
    )


def _cart_tensor_matrix_to_tuple(tensor):
    """
    Convert a symmetric 3x3 tensor matrix back to its six independent values.

    Parameters
    ----------
    tensor : array-like
        Symmetric Cartesian tensor.

    Returns
    -------
    tuple
        ``(aa, bb, cc, ab, ac, bc)`` as floats.
    """
    tensor = np.asarray(tensor, dtype=float)
    return (
        float(tensor[0, 0]),
        float(tensor[1, 1]),
        float(tensor[2, 2]),
        float(tensor[0, 1]),
        float(tensor[0, 2]),
        float(tensor[1, 2]),
    )

def _op_for_ity_repchange(ity, Ja2, Jb2, Jc2, Jx=None, Jy=None, oblate_family=False):
    """
    Build the elementary operator used by the legacy representation-match basis.

    Inputs are squared angular-momentum operators and, when needed, the
    transverse ``Jx/Jy`` pair. Output is the matrix associated with one Pickett
    ``ity`` code for operator-matching experiments.
    """
    if ity == 0:
        return np.eye(Ja2.shape[0], dtype=complex)
    if ity == 1:
        return Ja2
    if ity == 2:
        return Jb2
    if ity == 3:
        return Jc2
    if ity == 4:
        if oblate_family:
            return (Ja2 - Jb2)
        return (Jc2 - Jb2)
    if ity == 5:
        if Jx is None or Jy is None:
            raise ValueError("ity=5 requires Jx and Jy")
        Jp = Jx + 1.0j * Jy
        Jm = Jx - 1.0j * Jy
        return 0.5 * (Jp @ Jp @ Jp @ Jp + Jm @ Jm @ Jm @ Jm)
    if ity == 6:
        if Jx is None or Jy is None:
            raise ValueError("ity=6 requires Jx and Jy")
        Jp = Jx + 1.0j * Jy
        Jm = Jx - 1.0j * Jy
        return -0.5 * (np.linalg.matrix_power(Jp, 6) + np.linalg.matrix_power(Jm, 6))
    raise NotImplementedError(f"ity={ity} not implemented")


def _spcat_oblate_remap_abc(A, B, C, mu_a=None, mu_b=None, mu_c=None):
    """
    SPCAT oblate mode remaps principal-axis labels a <-> c internally.
    This mirrors revsym handling in spinv.c for oblate Hamiltonian/dipoles.
    """
    A_r, B_r, C_r = C, B, A
    if mu_a is None:
        return A_r, B_r, C_r
    return A_r, B_r, C_r, mu_c, mu_b, mu_a


def _spcat_oblate_remap_chi(chi_aa_v, chi_bb_v, chi_cc_v, chi_ab_v, chi_ac_v, chi_bc_v):
    """
    Quadrupole tensor remap for SPCAT oblate mode (a <-> c axis swap).
    """
    return (
        chi_cc_v,  # chi_aa'
        chi_bb_v,  # chi_bb'
        chi_aa_v,  # chi_cc'
        chi_bc_v,  # chi_ab'
        chi_ac_v,  # chi_ac'
        chi_ab_v,  # chi_bc'
    )

def _rot_ops_rep(J, rep="Ir"):
    """
    Build Ja/Jb/Jc in the current SPCAT/SPFIT-compatible raw representation.

    The K basis is always generated from Jz. What changes with the selected
    representation family is which principal-axis label ("a", "b", or "c")
    receives Jz and which two axes span the transverse plane used for Delta K
    operators.
    """
    rep = _normalize_rep_key(rep)
    J = int(J)
    K = np.arange(-J, J + 1, dtype=float)
    n = K.size

    # Ladder in K (projection on z)
    Jp = np.zeros((n, n), dtype=complex)
    for i, k in enumerate(K[:-1]):
        Jp[i + 1, i] = math.sqrt(J * (J + 1) - k * (k + 1))
    Jm = Jp.T
    Jz = np.diag(K)
    Jx = (Jp + Jm) / 2.0
    Jy = (Jp - Jm) / (2.0j)

    z_ax, x_ax, y_ax = _pickett_rep_axes(rep)

    # map: axis letter -> operator
    op_by_letter = {}
    op_by_letter[z_ax] = Jz
    op_by_letter[x_ax] = Jx
    op_by_letter[y_ax] = Jy

    Ja = op_by_letter["a"]
    Jb = op_by_letter["b"]
    Jc = op_by_letter["c"]

    return Ja, Jb, Jc, Jz   # <- ritorno anche Jz per K^2


@lru_cache(maxsize=None)
def wang_transform(J: int) -> np.ndarray:
    """
    Orthonormal Wang symmetrization matrix that maps K-basis -> Wang basis.
    Column order: K=0, then for K=1..J: (+), (-) combinations.
    """
    J = int(J)
    n = 2 * J + 1
    W = np.zeros((n, n), dtype=float)
    col = 0
    W[J, col] = 1.0  # K=0
    col += 1

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for k in range(1, J + 1):
        i_pos = J + k
        i_neg = J - k
        phase = -1.0 if (k % 2) else 1.0
        W[i_pos, col] = inv_sqrt2
        W[i_neg, col] = phase * inv_sqrt2
        col += 1
        W[i_pos, col] = inv_sqrt2
        W[i_neg, col] = -phase * inv_sqrt2
        col += 1

    return W


def _fix_degenerate_evecs_by_K2(E, U_w, J, W, tol=1e-6):
    """
    Rotate eigenvectors within (near-)degenerate clusters to diagonalize K^2.
    This matches SPCAT's tendency to pick K-pure states when energies are
    (almost) degenerate, improving intensity consistency for weak lines.
    """
    if tol <= 0 or U_w.shape[1] < 2:
        return U_w

    # K^2 in Wang basis
    K = np.arange(-J, J + 1, dtype=float)
    K2 = np.diag(K * K)
    K2_w = W.T @ K2 @ W

    n = E.shape[0]
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(E[j] - E[i]) < tol:
            j += 1
        if j - i > 1:
            U_sub = U_w[:, i:j]
            K2_sub = U_sub.T @ K2_w @ U_sub
            vals, vecs = _matrix_eigh(K2_sub)
            order = np.argsort(vals)
            U_w[:, i:j] = U_sub @ vecs[:, order]
        i = j

    return U_w


def _op_for_ity(ity, Ja2, Jb2, Jc2, Jx=None, Jy=None, oblate_family=False):
    """
    Return the active operator block associated with one Pickett ``ity`` code.

    This is the operator lookup used by the current Hamiltonian builder. It
    maps Pickett IDs onto matrices in the already selected raw representation
    family and returns the operator as a complex NumPy matrix.
    """
    if ity == 0:
        return np.eye(Ja2.shape[0], dtype=complex)
    if ity == 1:
        return Ja2
    if ity == 2:
        return Jb2
    if ity == 3:
        return Jc2
    if ity == 4:
        if Jx is None or Jy is None:
            raise ValueError("ity=4 requires Jx and Jy")
        return Jy @ Jy - Jx @ Jx
    if ity == 5:
        if Jx is None or Jy is None:
            raise ValueError("ity=5 requires Jx and Jy")
        Jp = Jx + 1.0j * Jy
        Jm = Jx - 1.0j * Jy
        return 0.5 * (Jp @ Jp @ Jp @ Jp + Jm @ Jm @ Jm @ Jm)
    if ity == 6:
        if Jx is None or Jy is None:
            raise ValueError("ity=6 requires Jx and Jy")
        Jp = Jx + 1.0j * Jy
        Jm = Jx - 1.0j * Jy
        return -0.5 * (np.linalg.matrix_power(Jp, 6) + np.linalg.matrix_power(Jm, 6))
    raise NotImplementedError(f"ity={ity} not implemented")


def _sym_ksq(op, K2, ksq):
    """
    SPCAT symksq: 0.5*(K^2 op + op K^2), for K2 diagonal.
    For ksq>1, use K^(2*ksq) with the same symmetrization.
    """
    if ksq <= 0:
        return op
    K2p = K2.copy()
    for _ in range(1, ksq):
        K2p = K2p @ K2
    return 0.5 * (K2p @ op + op @ K2p)

def H_from_ids(params, J, rep="Ir"):
    """
    Assemble one J block directly from Pickett-style parameter identifiers.

    Higher-level helpers translate Watson A/S constants into these IDs, but
    this function is where the actual operator algebra lives. Keeping this
    layer explicit makes it easier to compare WMS-Rot against SPCAT/SPFIT and
    to reason about which axis carries K in the chosen representation family.
    """
    Ja, Jb, Jc, Jz = _rot_ops_rep(J, rep=rep)
    Ja2, Jb2, Jc2 = Ja@Ja, Jb@Jb, Jc@Jc
    J2 = Ja2 + Jb2 + Jc2
    K2 = Jz @ Jz   # <-- sempre
    ops_by_axis = {"a": Ja, "b": Jb, "c": Jc}
    _z_ax, x_ax, y_ax = _pickett_rep_axes(rep)
    Jx = ops_by_axis[x_ax]
    Jy = ops_by_axis[y_ax]
    oblate_family = (_pickett_rep_handedness(rep) == "l")

    H = np.zeros_like(Ja2, dtype=complex)
    for pid, val in params.items():
        nsx, ksq, ity, zfac = parse_id_quartic(pid)
        op = _op_for_ity(ity, Ja2, Jb2, Jc2, Jx=Jx, Jy=Jy, oblate_family=oblate_family)
        op = _sym_ksq(op, K2, ksq)
        if nsx > 0:
            op = np.linalg.matrix_power(J2, nsx) @ op
        H = H + zfac * val * op
    return H


def H_rot_watson_A(
    A, B, C, DJ, DJK, DK, dJ, dK, J,
    rep="Ir",
    HJ=0.0, HJK=0.0, HKJ=0.0, HK=0.0, h1=0.0, h2=0.0, h3=0.0
):
    """
    Diagonalize one J block in Watson A reduction.

    Parameters are the rotational constants, quartic/sextic distortion
    constants, J value, and representation family. The function converts those
    inputs to Pickett IDs, builds the Hamiltonian through :func:`H_from_ids`,
    diagonalizes it, and returns sorted eigenvalues/eigenvectors.
    """
    params = {
        10000: A,
        20000: B,
        30000: C,
        200: DJ,
        1100: DJK,
        2000: DK,
        40100: dJ,
        41000: dK,
        300: HJ,
        1200: HJK,
        2100: HKJ,
        3000: HK,
        40200: h1,
        41100: h2,
        42000: h3,
    }
    H = H_from_ids(params, J, rep=rep)
    E, U = _matrix_eigh(H)
    idx = np.argsort(E.real)
    return E.real[idx], U[:, idx]


def H_rot_watson_S(
    A, B, C, DJ, DJK, DK, d1, d2, J,
    rep="Ir",
    HJ=0.0, HJK=0.0, HKJ=0.0, HK=0.0, h1=0.0, h2=0.0, h3=0.0
):
    """
    Diagonalize one J block in Watson S reduction.

    Inputs mirror :func:`H_rot_watson_A`, except ``d1/d2`` follow the S
    convention. The helper resolves the active S-style ``d2`` identifier,
    builds the Pickett-ID Hamiltonian, diagonalizes it, and returns sorted
    eigenvalues/eigenvectors.
    """
    d2_id = _resolve_s_d2_id()
    params = {
        10000: A,
        20000: B,
        30000: C,
        200: DJ,
        1100: DJK,
        2000: DK,
        40100: d1,
        d2_id: d2,
        300: HJ,
        1200: HJK,
        2100: HKJ,
        3000: HK,
        50100: h1,
        51000: h2,
        60000: h3,
    }
    H = H_from_ids(params, J, rep=rep)
    E, U = _matrix_eigh(H)
    idx = np.argsort(E.real)
    return E.real[idx], U[:, idx]
#----------------------------------------------------------------------
def H_build(
    A, B, C, DJ, DJK, DK, dJ, dK, J,
    reduction=None, rep=None,
    HJ=0.0, HJK=0.0, HKJ=0.0, HK=0.0, h1=0.0, h2=0.0, h3=0.0
):
    """
    Dispatch to the Watson A or Watson S block builder.

    If ``reduction`` or ``rep`` are omitted, the current module globals are
    used. Output matches :func:`H_rot_watson_A` / :func:`H_rot_watson_S`:
    sorted eigenvalues and eigenvectors for the requested J block.
    """
    # prendi i valori globali aggiornati via pyodide.globals.set(...)
    if reduction is None:
        reduction = globals().get("REDUCTION", "A")
    if rep is None:
        rep = globals().get("REPRESENTATION", "Ir")

    if str(reduction).upper() == 'S':
        # dJ = d1, dK = d2 nel tuo schema
        return H_rot_watson_S(
            A, B, C, DJ, DJK, DK, dJ, dK, J,
            rep=rep, HJ=HJ, HJK=HJK, HKJ=HKJ, HK=HK, h1=h1, h2=h2, h3=h3
        )
    else:
        return H_rot_watson_A(
            A, B, C, DJ, DJK, DK, dJ, dK, J,
            rep=rep, HJ=HJ, HJK=HJK, HKJ=HKJ, HK=HK, h1=h1, h2=h2, h3=h3
        )


def _resolve_s_d2_id():
    """
    Resolve the SPCAT identifier used for the Watson-S ``d2`` constant.

    Returns either 41000 or 50000 depending on explicit user selection or, in
    auto mode, on whether the current run is quadrupole-HFS or pure rotation.
    """
    raw = globals().get("S_D2_ID", S_D2_ID)
    if raw is not None:
        sval = str(raw).strip().lower()
        if sval in ("41000", "50000"):
            return int(sval)
    # Auto mode: most SPCAT non-HFS S datasets use 50000, while many HFS
    # (quadrupole-HFS) setups use 41000.
    has_hfs = _has_active_quadrupole()
    return 41000 if has_hfs else 50000


def H_matrix(
    A, B, C, DJ, DJK, DK, dJ, dK, J,
    reduction=None, rep=None,
    HJ=0.0, HJK=0.0, HKJ=0.0, HK=0.0, h1=0.0, h2=0.0, h3=0.0
):
    """
    Return the undiagonalized rotational Hamiltonian matrix for one J block.

    Inputs are the same physical constants accepted by :func:`H_build`. The
    output is the raw complex matrix before diagonalization, expressed in the
    active representation/reduction convention.
    """
    if reduction is None:
        reduction = globals().get("REDUCTION", "A")
    if rep is None:
        rep = globals().get("REPRESENTATION", "Ir")

    if str(reduction).upper() == 'S':
        d2_id = _resolve_s_d2_id()
        params = {
            10000: A,
            20000: B,
            30000: C,
            200: DJ,
            1100: DJK,
            2000: DK,
            40100: dJ,
            d2_id: dK,
            300: HJ,
            1200: HJK,
            2100: HKJ,
            3000: HK,
            50100: h1,
            51000: h2,
            60000: h3,
        }
    else:
        params = {
            10000: A,
            20000: B,
            30000: C,
            200: DJ,
            1100: DJK,
            2000: DK,
            40100: dJ,
            41000: dK,
            300: HJ,
            1200: HJK,
            2100: HKJ,
            3000: HK,
            40200: h1,
            41100: h2,
            42000: h3,
        }

    return H_from_ids(params, J, rep=rep)

# -------------------------------------------------------------
# Boltzmann
# -------------------------------------------------------------
"""
Rotational symmetry number utilities.

Maps molecular point-group labels to the rotational symmetry number σ.
Only proper rotations are considered, following standard
statistical thermodynamics conventions.
"""
def rotational_symmetry_number(point_group: str) -> int:
    """
    Return the rotational symmetry number σ for a given point group.

    Parameters
    ----------
    point_group : str
        Point group label (e.g. 'C2v', 'D2d', 'D3h', 'Td', 'Oh', 'Ih',
        'Cinfv', 'Dinfh').

    Returns
    -------
    sigma : int
        Rotational symmetry number.

    Notes
    -----
    - Improper operations (σ, i, S_n) do not contribute directly to σ.
    - Point groups differing only by v/h/d suffixes have the same σ.
    """
    if not point_group:
        raise ValueError("Point group must be specified to determine σ")

    pg = point_group.strip()

    # ------------------------------------------------------------
    # Linear molecules
    # ------------------------------------------------------------
    if pg == "Cinfv":
        return 1
    if pg == "Dinfh":
        return 2

    # ------------------------------------------------------------
    # Trivial groups
    # ------------------------------------------------------------
    if pg in {"C1", "Cs", "Ci"}:
        return 1

    # ------------------------------------------------------------
    # Platonic solids
    # ------------------------------------------------------------
    if pg == "Td":
        return 12
    if pg == "Oh":
        return 24
    if pg == "Ih":
        return 60

    # ------------------------------------------------------------
    # Dihedral groups: Dn, Dnh, Dnd  → σ = 2n
    # (includes D2d, D3d, etc.)
    # ------------------------------------------------------------
    m = re.match(r"D(\d+)", pg)
    if m:
        n = int(m.group(1))
        return 2 * n

    # ------------------------------------------------------------
    # Cyclic groups: Cn, Cnv, Cnh → σ = n
    # ------------------------------------------------------------
    m = re.match(r"C(\d+)", pg)
    if m:
        return int(m.group(1))

    # ------------------------------------------------------------
    # Improper rotation groups Sn
    # Sn has n/2 proper rotations (n even only)
    # Example: S4 → C2 → σ = 2
    # ------------------------------------------------------------
    m = re.match(r"S(\d+)", pg)
    if m:
        n = int(m.group(1))
        if n % 2 != 0:
            raise ValueError(f"Invalid improper rotation group '{pg}'")
        return n // 2

    # ------------------------------------------------------------
    # Unsupported / unknown
    # ------------------------------------------------------------
    raise ValueError(f"Unsupported or unknown point group '{point_group}'")

def _boltz(E_MHz, T):
    """
    Evaluate the Boltzmann factor for one energy expressed in MHz.

    Parameters
    ----------
    E_MHz : float
        Level energy in MHz.
    T : float
        Temperature in kelvin.
    """
    return math.exp(-H_PLANCK * E_MHz * TAU / (K_BOLTZ * T))

def _compute_Qrs(levels, T):
    """
    Compute the rotational partition function from a level dictionary.

    Inputs are the per-J level table produced by the simple/matrix builders and
    the temperature in kelvin. Output is ``Q_rot`` including symmetry-number
    and nuclear-statistics weights.
    """
    # Rotational partition function:
    # Qrot(T) = (1/sigma) * sum_J sum_alpha (2J+1) g_ns(Gamma_Jalpha) exp(-E_Jalpha/kT)
    if T <= 0:
        return 0.0

    groupSymmetry = globals().get("groupSymmetry", "C1")
    ee_wt = float(globals().get("eeWt", 1.0))
    eo_wt = float(globals().get("eoWt", 1.0))
    oe_wt = float(globals().get("oeWt", 1.0))
    oo_wt = float(globals().get("ooWt", 1.0))

    #TODO: Controlla per il simmetrico
    def _spin_weight(species):
        if species == 'ee': return ee_wt
        if species == 'eo': return eo_wt
        if species == 'oe': return oe_wt
        if species == 'oo': return oo_wt
        return 1.0

    try:
        sigma = rotational_symmetry_number(groupSymmetry)
    except Exception:
        sigma = 1
    if sigma <= 0:
        sigma = 1

    total = 0.0
    for J, rows in levels.items():
        gJ = (2 * J + 1)
        for (E, Ka, Kc, species) in rows:
            total += gJ * _spin_weight(species) * _boltz(E, T)

    return total / sigma


def _signature_value(value):
    """
    Normalize a value before inserting it into a cache/signature tuple.

    Booleans, strings, ``None``, and NaN are preserved in a stable form so two
    physically equivalent runs produce matching cache keys.
    """
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        fval = float(value)
    except Exception:
        return str(value)
    if math.isnan(fval):
        return "nan"
    return fval


def _quadrupole_signature_payload():
    """
    Serialize the active quadrupole nuclei into a cache-friendly tuple.

    Output contains nucleus index, spin, and Cartesian chi components for each
    active quadrupolar nucleus, and is used to invalidate caches when the HFS
    setup changes.
    """
    payload = []
    for nuc in _active_quadrupole_nuclei():
        payload.append((
            int(nuc["index"]),
            _signature_value(nuc["I"]),
            *(_signature_value(v) for v in nuc["chi_cart"]),
        ))
    return tuple(payload)


def _normalize_j_range(J_min, J_max):
    """
    Validate and normalize inclusive J-range bounds.

    Returns a pair of integers ``(J_min, J_max)`` and raises a descriptive
    error if the provided bounds are non-integral, negative, or reversed.
    """
    try:
        j_min = int(J_min)
        j_max = int(J_max)
    except Exception as exc:
        raise ValueError(f"J bounds must be integers. Got J_min={J_min}, J_max={J_max}.") from exc
    if j_min < 0:
        raise ValueError(f"J_min must be >= 0. Got J_min={J_min}.")
    if j_max < j_min:
        raise ValueError(f"J_max must be >= J_min. Got J_min={J_min}, J_max={J_max}.")
    return j_min, j_max


def _sorted_level_js(levels):
    """Return the available J shells from a level dictionary in ascending order."""
    return sorted(int(J) for J in levels.keys())


def _match_last_sim_cache(signature):
    """
    Compare a requested simulation signature against the latest stored cache.

    Returns the cached structure when its signature matches exactly, otherwise
    returns ``None`` so the caller can rebuild the structure.
    """
    cache = globals().get("LAST_SIM_CACHE", None)
    if isinstance(cache, dict) and cache.get("signature") == signature:
        return cache
    return None


def _store_last_sim_cache(cache, signature):
    """
    Save a simulation cache object together with its identifying signature.

    The function mutates the cache dictionary in place by storing the
    signature, then exposes it through the module-global ``LAST_SIM_CACHE``.
    """
    cache["signature"] = signature
    globals()["LAST_SIM_CACHE"] = cache
    return cache


def _restore_cached_outputs(cache):
    """
    Restore auxiliary outputs from a cached structure into module globals.

    This keeps the wavefunction viewer and HFS tables synchronized when a
    structure is reused instead of recomputed.
    """
    global LAST_DF_HF, LAST_WAVEFUNC_CACHE
    LAST_DF_HF = cache.get("df_hf", None)
    LAST_WAVEFUNC_CACHE = cache.get("wavefunc_cache", None)


def _simple_cache_signature(
    rotorType, A, B, C,
    DJ, DJK, DK, dJ, dK,
    HJ, HJK, HKJ, HK, h1, h2, h3,
    J_min, J_max, rep
):
    """
    Build the cache signature for the analytic simple-rotor branch.

    The tuple records rotor type, constants, J range, active representation,
    reduction, and S-d2 convention so a previously built simple structure can
    be reused safely.
    """
    return (
        "simple",
        str(rotorType),
        _signature_value(A),
        _signature_value(B),
        _signature_value(C),
        _signature_value(DJ),
        _signature_value(DJK),
        _signature_value(DK),
        _signature_value(dJ),
        _signature_value(dK),
        _signature_value(HJ),
        _signature_value(HJK),
        _signature_value(HKJ),
        _signature_value(HK),
        _signature_value(h1),
        _signature_value(h2),
        _signature_value(h3),
        int(J_min),
        int(J_max),
        _signature_rep_key(rep),
        "rawrep",
        str(globals().get("REDUCTION", REDUCTION)),
        _signature_rep_key(globals().get("REPRESENTATION", REPRESENTATION)),
        _signature_value(_resolve_s_d2_id()),
    )


def _matrix_cache_signature(
    rotorType, A, B, C,
    DJ, DJK, DK, dJ, dK,
    HJ, HJK, HKJ, HK, h1, h2, h3,
    J_min, J_max, wang_symmetry
):
    """
    Build the cache signature for the matrix-diagonalization branch.

    Compared with :func:`_simple_cache_signature`, this also stores Wang/HFS
    controls and the active quadrupole payload, because those alter both the
    basis and the resulting transition catalog.
    """
    return (
        "matrix",
        str(rotorType),
        _signature_value(A),
        _signature_value(B),
        _signature_value(C),
        _signature_value(DJ),
        _signature_value(DJK),
        _signature_value(DK),
        _signature_value(dJ),
        _signature_value(dK),
        _signature_value(HJ),
        _signature_value(HJK),
        _signature_value(HKJ),
        _signature_value(HK),
        _signature_value(h1),
        _signature_value(h2),
        _signature_value(h3),
        int(J_min),
        int(J_max),
        bool(wang_symmetry),
        "rawrep",
        str(globals().get("REDUCTION", REDUCTION)),
        _signature_rep_key(globals().get("REPRESENTATION", REPRESENTATION)),
        _signature_value(_resolve_s_d2_id()),
        _signature_value(globals().get("DEGEN_TOL", 0.0)),
        _signature_value(globals().get("F_MAX", F_MAX)),
        _quadrupole_signature_payload(),
    )

# -------------------------------------------------------------
# Dipole |J,K>
# -------------------------------------------------------------
def axis_role_in_rep(axis_letter, rep):
    """
    Classify one principal axis as the active z/x/y role in a representation.

    Parameters
    ----------
    axis_letter : {"a", "b", "c"}
        Principal-axis label requested by the user.
    rep : str
        Representation family to inspect.

    Returns
    -------
    str
        One of ``"z"``, ``"x"``, or ``"y"``.
    """
    z_ax, x_ax, y_ax = _active_rep_axes(rep)
    if axis_letter == z_ax: return "z"
    if axis_letter == x_ax: return "x"
    if axis_letter == y_ax: return "y"
    raise ValueError("bad axis")

def build_mu_matrix_rep(Jl, Ju, axis_letter, rep="Ir"):
    """
    Build the dipole-transition matrix in the active raw representation.

    Inputs are lower/upper J values, a principal-axis dipole label, and the
    selected representation family. Output is the matrix of reduced rotational
    dipole amplitudes in the K basis for that axis.
    """
    role = axis_role_in_rep(axis_letter, rep)

    K_l = np.arange(-Jl, Jl + 1, dtype=int)
    K_u = np.arange(-Ju, Ju + 1, dtype=int)
    mu = np.zeros((K_l.size, K_u.size), dtype=float)

    for i, Kl in enumerate(K_l):
        for j, Ku in enumerate(K_u):

            if role == "z":
                if Ku != Kl:
                    continue
                q = 0
                sign = 1.0
            else:
                q = Ku - Kl
                if abs(q) != 1:
                    continue
                # x: antisimm => cambia segno tra q=+1 e q=-1
                # y: simm    => stesso segno
                if role == "x":
                    sign = (-1.0 if q == -1 else 1.0)
                else: # "y"
                    sign = 1.0

            try:
                w = W3J(int(2*Ju), 2, int(2*Jl),
                        int(-2*Ku), int(2*q), int(2*Kl))
            except Exception:
                continue
            if abs(w) < 1e-12:
                continue

            prefactor = math.sqrt((2*Ju + 1)*(2*Jl + 1)) * w

            if role in ("x","y"):
                prefactor /= math.sqrt(2.0)
                prefactor *= sign

            mu[i, j] = prefactor

    return mu

def build_mu_matrix(Jl, Ju, axis):
    """
    Build the legacy dipole matrix in the canonical ``Ir``-style convention.

    This helper predates the explicit representation-aware path and is kept
    for compatibility with simple branches that assume ``a`` is the K axis.
    """
    K_l = np.arange(-Jl, Jl + 1, dtype=int)
    K_u = np.arange(-Ju, Ju + 1, dtype=int)
    mu = np.zeros((K_l.size, K_u.size), dtype=float)

    for i, Kl in enumerate(K_l):
        for j, Ku in enumerate(K_u):
            if axis == 'a':
                if Ku != Kl:
                    continue
                q = 0
            else:
                q = Ku - Kl
                if abs(q) != 1:
                    continue

            try:
                w = W3J(int(2*Ju), 2, int(2*Jl), int(-2*Ku), int(2*q), int(2*Kl))
                #w = float(wigner_3j(S(Ju), S(1), S(Jl), S(-Ku), S(q), S(Kl)))
            except Exception:
                continue
            if abs(w) < 1e-12:
                continue

            prefactor = math.sqrt((2*Ju + 1)*(2*Jl + 1)) * w

            if axis == 'b':
                prefactor /= math.sqrt(2.0)
                if q == -1:
                    prefactor *= -1
            elif axis == 'c':
                prefactor /= math.sqrt(2.0)

            mu[i, j] = prefactor
    return mu

# -------------------------------------------------------------
# Simple model (linear/prolate/oblate)
# -------------------------------------------------------------
def line_strength(Jl, Kl, Ju, Ku):
    """
    Return the symmetric-top line-strength factor for one K-resolved branch.

    Input quantum numbers are lower/upper ``(J,K)`` values; output is the
    reduced rotational strength before dipole moments and Boltzmann weights.
    """
    p = Ku - Kl
    try:
        w = W3J(int(2*Ju), 2, int(2*Jl), int(-2*Ku), int(2*p), int(2*Kl))
        #w = float(wigner_3j(S(Ju), S(1), S(Jl), S(-Ku), S(p), S(Kl)))
    except Exception:
        return 0.0
    return (2*Ju + 1) * (w*w)

def _centrifugal_sym(J, K, DJ, DJK, DK, HJ=0.0, HJK=0.0, HKJ=0.0, HK=0.0):
    """
    Evaluate diagonal symmetric-top quartic and sextic distortion shifts.

    Parameters are the rotational quantum numbers and centrifugal-distortion
    constants. Output is the additive energy correction in MHz.
    """
    JJ = J * (J + 1)
    return (
        +DJ  * J**2 * (J + 1)**2
        +DJK * JJ * K**2
        +DK  * K**4
        +HJ  * (JJ**3)
        +HJK * (JJ**2) * K**2
        +HKJ * JJ * K**4
        +HK  * K**6
    )

def _build_simple_structure(rotorType, A, B, C,
                            DJ, DJK, DK, dJ, dK,
                            HJ, HJK, HKJ, HK, h1, h2, h3,
                            J_max, rep="Ir", J_min=0):
    """
    Build analytic level tables for linear and symmetric-top shortcuts.

    This branch is used only when the selected model is simple enough that the
    closed-form formulas still mirror the matrix treatment. As soon as
    off-diagonal centrifugal terms or quadrupole HFS matter, the code routes
    to the asymmetric builder instead.
    """
    global LAST_DF_HF, LAST_WAVEFUNC_CACHE
    J_min, J_max = _normalize_j_range(J_min, J_max)
    levels = {}

    def _ka_kc_from_k(K):
        if rotorType == 'oblate':
            return 0, K
        if rotorType == 'linear':
            return 0, 0
        return K, 0

    def _species_from_ka_kc(Ka, Kc):
        pKa = 'e' if (Ka % 2 == 0) else 'o'
        pKc = 'e' if (Kc % 2 == 0) else 'o'
        return pKa + pKc

    for J in range(J_min, J_max + 1):
        rows = []
        if rotorType == 'linear':
            E0 = B * J * (J + 1)
            corrE = E0 - DJ * J**2 * (J + 1)**2 + HJ * (J * (J + 1))**3
            Ka, Kc = _ka_kc_from_k(0)
            rows.append((corrE, 0, Ka, Kc, _species_from_ka_kc(Ka, Kc)))
        elif rotorType == 'prolate':
            for K in range(-J, J + 1):
                Kabs = abs(K)
                E0 = B * J * (J + 1) + (A - B) * (Kabs ** 2)
                corrE = E0 + _centrifugal_sym(J, Kabs, DJ, DJK, DK, HJ, HJK, HKJ, HK)
                Ka, Kc = _ka_kc_from_k(Kabs)
                rows.append((corrE, K, Ka, Kc, _species_from_ka_kc(Ka, Kc)))
        elif rotorType == 'spherical':
            for K in range(-J, J + 1):
                Kabs = abs(K)
                E0 = B * J * (J + 1)
                JJ = J * (J + 1)
                corrE = E0 + DJ * (JJ ** 2) + HJ * (JJ ** 3)
                Ka, Kc = _ka_kc_from_k(Kabs)
                rows.append((corrE, K, Ka, Kc, _species_from_ka_kc(Ka, Kc)))
        elif rotorType == 'oblate':
            for K in range(-J, J + 1):
                Kabs = abs(K)
                E0 = B * J * (J + 1) + (C - B) * (Kabs ** 2)
                corrE = E0 + _centrifugal_sym(J, Kabs, DJ, DJK, DK, HJ, HJK, HKJ, HK)
                Ka, Kc = _ka_kc_from_k(Kabs)
                rows.append((corrE, K, Ka, Kc, _species_from_ka_kc(Ka, Kc)))
        levels[J] = rows

    wf_rot_levels = []
    wf_states = {}
    for J in range(J_min, J_max + 1):
        Ja, Jb, Jc, _ = _rot_ops_rep(J, rep=rep)
        Ja2_op = Ja @ Ja
        Jb2_op = Jb @ Jb
        Jc2_op = Jc @ Jc
        k_values = list(range(-J, J + 1))
        for alpha, (E, K, Ka, Kc, species) in enumerate(levels[J]):
            n = 2 * J + 1
            v = np.zeros(n, dtype=complex)
            v[int(K) + J] = 1.0
            axis_values = {
                "a": _expectation(v, Ja2_op),
                "b": _expectation(v, Jb2_op),
                "c": _expectation(v, Jc2_op),
            }
            coord_expect = _wavefunc_coords_from_axis_values(axis_values, rep)
            key = f"{J}:{alpha}"
            state = {
                "key": key,
                "J": int(J),
                "alpha": int(alpha),
                "K": int(K),
                "Ka": int(Ka),
                "Kc": int(Kc),
                "species": str(species),
                "energy_mhz": float(E),
                "energy_cm": float(E) / 29979.2458,
                "k_values": k_values,
                "coeff_re": [float(x) for x in np.real(v)],
                "coeff_im": [float(x) for x in np.imag(v)],
                "coord_expect": coord_expect,
            }
            wf_states[key] = state
            wf_rot_levels.append({
                "id": key,
                "kind": "rot",
                "J": int(J),
                "alpha": int(alpha),
                "Ka": int(Ka),
                "Kc": int(Kc),
                "species": str(species),
                "energy_mhz": float(E),
                "energy_cm": float(E) / 29979.2458,
                "label": f"J={int(J)} alpha={int(alpha)} Ka={int(Ka)} Kc={int(Kc)}",
                "parent_key": key,
                "purity": 1.0,
            })

    wavefunc_cache = {
        "meta": _make_wavefunc_cache_meta(
            rotorType,
            rep,
            globals().get("REDUCTION", REDUCTION),
            "simple",
            J_min,
            J_max
        ),
        "rot_levels": wf_rot_levels,
        "hfs_levels": [],
        "states": wf_states,
    }
    LAST_DF_HF = None
    LAST_WAVEFUNC_CACHE = wavefunc_cache
    return {
        "engine": "simple",
        "rotorType": str(rotorType),
        "levels": levels,
        "J_min": int(J_min),
        "J_max": int(J_max),
        "rep": rep,
        "wavefunc_cache": wavefunc_cache,
        "df_hf": None,
        "mu_axis_cache": {},
    }


def _simulate_simple_from_cache(structure, T, mu_a, mu_b, mu_c, intensity_cut):
    """
    Convert a prebuilt simple-rotor structure into a transition DataFrame.

    Inputs are the cached structure, temperature, dipole components, and an
    intensity cutoff. Output is the filtered stick catalog with frequencies,
    intensities, Ka/Kc labels, and branch labels.
    """
    _restore_cached_outputs(structure)
    rotorType = structure["rotorType"]
    levels = structure["levels"]
    rep = structure["rep"]
    mu_axis_cache = structure.setdefault("mu_axis_cache", {})
    J_values = _sorted_level_js(levels)
    if not J_values:
        return pd.DataFrame(columns=['Frequency (MHz)', 'Intensity', 'Relative intensity', 'LGINT', 'logS', '...'])
    ee_wt = float(globals().get("eeWt", 1.0))
    eo_wt = float(globals().get("eoWt", 1.0))
    oe_wt = float(globals().get("oeWt", 1.0))
    oo_wt = float(globals().get("ooWt", 1.0))

    def _spin_weight(species):
        if species == 'ee': return ee_wt
        if species == 'eo': return eo_wt
        if species == 'oe': return oe_wt
        if species == 'oo': return oo_wt
        return 1.0

    def _compute_Qrs_simple(levels_local, T_local):
        if T_local <= 0:
            return 0.0
        try:
            sigma = rotational_symmetry_number(globals().get("groupSymmetry", "C1"))
        except Exception:
            sigma = 1
        if sigma <= 0:
            sigma = 1
        total = 0.0
        for J, rows in levels_local.items():
            gJ = (2 * J + 1)
            for (E, K, Ka, Kc, species) in rows:
                total += gJ * _spin_weight(species) * _boltz(E, T_local)
        return total / sigma

    zero = 1.5e-38
    cmc = 29979.2458
    tmc = 1.43878

    qrot_override = globals().get("QROT_OVERRIDE", None)
    if qrot_override is not None:
        qrot = float(qrot_override)
    else:
        qrot = float(_compute_Qrs_simple(levels, T))
    if qrot <= 0.0:
        qrot = 1.0
    if qrot < 1.0:
        qrot = 1.0
    globals()["LAST_QROT_USED"] = float(qrot)
    globals()["LAST_QROT_SOURCE"] = "override" if qrot_override is not None else "auto"

    fac = 4.16231e-5 / qrot

    fqmin = float(globals().get("FREQ_MIN", 0.0))
    fqmax = float(globals().get("FREQ_MAX", 9999.99))
    freq_unit = str(globals().get("FREQ_UNIT", "auto")).lower()
    if freq_unit not in ("ghz", "mhz", "auto"):
        freq_unit = "auto"
    if freq_unit == "ghz" or (freq_unit == "auto" and fqmax > 0.0 and fqmax <= 1000.0):
        fqmax *= 1000.0
        if fqmin > 0.0:
            fqmin *= 1000.0
    if fqmin < 5e-5:
        fqmin = 5e-5
    if fqmax < fqmin:
        fqmax = fqmin

    thrsh = float(globals().get("STR0", -12.0))
    thrsh1 = float(globals().get("STR1", -10.0))
    thrsh1 -= 2.0 * math.log10(300000.0)

    starg = thrsh - math.log10(max(fqmax * fac, zero))
    scomp = -38.0
    strmn = pow(10.0, max(starg, scomp))

    tmq = -tmc / (T * cmc) if T > 0 else -1.0
    tmql = tmq * 0.43429448

    def allowed(axis, dJv, dK):
        if rotorType == 'linear':
            return axis == 'a' and dK == 0 and abs(dJv) == 1
        if rotorType == 'prolate':
            if axis == 'a':
                return dK == 0 and abs(dJv) == 1
            return abs(dK) == 1 and (abs(dJv) == 1 or dJv == 0)
        if rotorType == 'spherical':
            if axis == 'a':
                return dK == 0 and abs(dJv) == 1
            return abs(dK) == 1 and (abs(dJv) == 1 or dJv == 0)
        if rotorType == 'oblate':
            if axis == 'c':
                return dK == 0 and abs(dJv) == 1
            return abs(dK) == 1 and (abs(dJv) == 1 or dJv == 0)
        return False

    lines = []
    axes = [('a', mu_a), ('b', mu_b), ('c', mu_c)]

    if rotorType == 'linear':
        for J in J_values:
            for E_l, K_l, Ka_l, Kc_l, sp_l in levels[J]:
                if (J + 1) in levels:
                    for E_u, K_u, Ka_u, Kc_u, sp_u in levels[J + 1]:
                        dK = K_u - K_l
                        nu = abs(E_u - E_l)
                        if nu < 1e-9:
                            continue
                        for axis, mu in axes:
                            if mu == 0 or not allowed(axis, 1, dK):
                                continue
                            Sval = line_strength(J, K_l, J+1, K_u)
                            if dK != 0:
                                Sval *= 0.5
                            strr = Sval * (mu ** 2)
                            if strr <= 0:
                                continue
                            logS = math.log10(max(strr, 1e-300))
                            if nu < fqmin or nu > fqmax:
                                continue
                            if strr < strmn:
                                continue
                            dgn = _spin_weight(sp_l)
                            str_val = dgn * strr * fac * nu * (1.0 - math.exp(tmq * nu))
                            if str_val <= 0:
                                continue
                            LGINT = math.log10(str_val + zero) + tmql * E_l
                            if LGINT < thrsh:
                                continue
                            thrshf = thrsh1 + 2.0 * math.log10(nu + zero)
                            diff = thrsh - thrshf
                            if abs(diff) < 4.0:
                                thrshf = math.log10(pow(10.0, diff) + 1.0) + thrshf
                            if LGINT < thrshf:
                                continue
                            intensity = str_val * math.exp(tmq * E_l)
                            if intensity <= 0:
                                continue
                            lines.append((nu, intensity, LGINT, logS, J, Ka_l, Kc_l, sp_l, J+1, Ka_u, Kc_u, sp_u, 'R'))
    else:
        rep_sym = str(rep) if rep in REP_MAP else ("IIIr" if rotorType == "oblate" else "Ir")

        def axis_mu_mat(Jl, Ju, axis):
            key = (Jl, Ju, axis)
            mat = mu_axis_cache.get(key)
            if mat is None:
                mat = build_mu_matrix_rep(Jl, Ju, axis, rep=rep_sym)
                mu_axis_cache[key] = mat
            return mat

        for Jl in J_values:
            rows_l = levels[Jl]
            for Ju in (Jl - 1, Jl, Jl + 1):
                if Ju not in levels:
                    continue
                rows_u = levels[Ju]
                mu_mat = (
                    mu_a * axis_mu_mat(Jl, Ju, 'a')
                    + mu_b * axis_mu_mat(Jl, Ju, 'b')
                    + mu_c * axis_mu_mat(Jl, Ju, 'c')
                )
                branch = 'R' if Ju > Jl else ('P' if Ju < Jl else 'Q')

                for il, (E_l, K_l, Ka_l, Kc_l, sp_l) in enumerate(rows_l):
                    for iu, (E_u, K_u, Ka_u, Kc_u, sp_u) in enumerate(rows_u):
                        if E_u <= E_l:
                            continue
                        nu = E_u - E_l
                        if nu < 1e-9:
                            continue

                        amp = mu_mat[il, iu]
                        strr = float((amp.conjugate() * amp).real)
                        if strr <= 0.0:
                            continue

                        logS = math.log10(max(strr, 1e-300))
                        if nu < fqmin or nu > fqmax:
                            continue
                        if strr < strmn:
                            continue

                        dgn = _spin_weight(sp_l)
                        str_val = dgn * strr * fac * nu * (1.0 - math.exp(tmq * nu))
                        if str_val <= 0:
                            continue
                        LGINT = math.log10(str_val + zero) + tmql * E_l
                        if LGINT < thrsh:
                            continue

                        thrshf = thrsh1 + 2.0 * math.log10(nu + zero)
                        diff = thrsh - thrshf
                        if abs(diff) < 4.0:
                            thrshf = math.log10(pow(10.0, diff) + 1.0) + thrshf
                        if LGINT < thrshf:
                            continue

                        intensity = str_val * math.exp(tmq * E_l)
                        if intensity <= 0:
                            continue
                        lines.append((nu, intensity, LGINT, logS, Jl, Ka_l, Kc_l, sp_l, Ju, Ka_u, Kc_u, sp_u, branch))

    if not lines:
        return pd.DataFrame(columns=['Frequency (MHz)', 'Intensity', 'Relative intensity', 'LGINT', 'logS', '...'])

    Imax = max(l[1] for l in lines)
    if Imax <= 0:
        Imax = 1.0
    data = [(nu, I, I/Imax, LGINT, logS, Jl, Ka_l, Kc_l, sp_l, Ju, Ka_u, Kc_u, sp_u, branch)
            for (nu, I, LGINT, logS, Jl, Ka_l, Kc_l, sp_l, Ju, Ka_u, Kc_u, sp_u, branch) in lines
            if I/Imax >= intensity_cut]

    cols = ['Frequency (MHz)', 'Intensity', 'Relative intensity', 'LGINT', 'logS',
            'Jl', 'Ka_l', 'Kc_l', 'sp_l', 'Ju', 'Ka_u', 'Kc_u', 'sp_u', 'Branch']
    return pd.DataFrame(data, columns=cols).sort_values('Frequency (MHz)').reset_index(drop=True)


def _simulate_simple(rotorType, T, A, B, C,
                     DJ, DJK, DK, dJ, dK,
                     HJ, HJK, HKJ, HK, h1, h2, h3,
                     mu_a, mu_b, mu_c,
                     J_max, intensity_cut,
                     rep="Ir", J_min=0):
    """
    One-shot wrapper around the analytic simple-rotor simulator.

    It builds the simple structure from the requested constants and immediately
    evaluates the transition catalog. Output is the same DataFrame returned by
    :func:`_simulate_simple_from_cache`.
    """
    globals()["LAST_SIM_REUSED"] = False
    structure = _build_simple_structure(
        rotorType, A, B, C,
        DJ, DJK, DK, dJ, dK,
        HJ, HJK, HKJ, HK, h1, h2, h3,
        J_max, rep=rep, J_min=J_min
    )
    return _simulate_simple_from_cache(structure, T, mu_a, mu_b, mu_c, intensity_cut)


# -------------------------------------------------------------
# Stick plot
# -------------------------------------------------------------
def plot_stick_spectrum(df):
    """
    Draw a quick matplotlib stick spectrum from a transition catalog.

    Input is a DataFrame with at least ``Frequency (MHz)`` and
    ``Relative intensity`` columns. Output is the current matplotlib figure.
    """
    plt.figure(figsize=(12, 4))
    plt.clf()
    if df.empty:
        plt.text(0.5, 0.5, 'No lines above threshold', ha='center', va='center')
    else:
        for _, r in df.iterrows():
            plt.vlines(r['Frequency (MHz)'], 0, r['Relative intensity'])
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Rel. intensity')

# -------------------------------------------------------------
# Asymmetric simulator (con HFS full-F se chi != 0)
# -------------------------------------------------------------
def _expectation(v, Op):
    """<v|Op|v> with v complex column vector."""
    return float(np.vdot(v, Op @ v).real)

def _wavefunc_axis_labels(rep):
    """
    Map the active internal x/y/z roles back to user-facing axis labels.

    Returns a dictionary used by the wavefunction viewer metadata so plotted
    expectation values carry the correct axis names for the chosen family.
    """
    z_ax, x_ax, y_ax = _active_rep_axes(rep)
    return {"x": x_ax.upper(), "y": y_ax.upper(), "z": z_ax.upper()}

def _wavefunc_coords_from_axis_values(axis_values, rep):
    """
    Reorder axis expectation values into the active x/y/z display coordinates.

    Input is a dictionary keyed by ``a/b/c``. Output is a dictionary keyed by
    ``x/y/z`` following the current representation family.
    """
    z_ax, x_ax, y_ax = _active_rep_axes(rep)
    return {
        "x": float(axis_values.get(x_ax, 0.0)),
        "y": float(axis_values.get(y_ax, 0.0)),
        "z": float(axis_values.get(z_ax, 0.0)),
    }

def _infer_k_axis_label(rotor_type, rep):
    """
    Infer which principal axis should be described as the K axis to the user.

    For simple rotor types this is fixed analytically; for asymmetric tops the
    answer follows the active representation family's quantization axis.
    """
    rt = str(rotor_type).strip().lower()
    if rt in ("linear", "prolate"):
        return "A"
    if rt == "oblate":
        return "C"
    if rt == "spherical":
        return None
    z_ax, _, _ = _active_rep_axes(rep)
    return z_ax.upper()

def _make_wavefunc_cache_meta(rotor_type, rep, reduction, engine, J_min, J_max):
    """
    Assemble metadata consumed by the wavefunction viewer and exports.

    The returned dictionary summarizes rotor type, representation, reduction,
    J range, axis labels, K-axis label, and active quadrupole payload.
    """
    k_axis = _infer_k_axis_label(rotor_type, rep)
    nuclei = _active_quadrupole_nuclei()
    quad_payload = {"active": bool(nuclei), "count": len(nuclei), "nuclei": []}
    if len(nuclei) == 1:
        nuc = nuclei[0]
        quad_payload["i_nuc"] = float(nuc["I"])
        quad_payload.update({key: float(val) for key, val in zip(QUAD_CART_KEYS, nuc["chi_cart"])})
    else:
        quad_payload["i_nuc"] = None
    for nuc in nuclei:
        quad_payload["nuclei"].append({
            "index": int(nuc["index"]),
            "i_nuc": float(nuc["I"]),
            **{key: float(val) for key, val in zip(QUAD_CART_KEYS, nuc["chi_cart"])},
        })
    return {
        "rotor_type": str(rotor_type),
        "representation": str(rep),
        "reduction": str(reduction),
        "engine": str(engine),
        "J_min": int(J_min),
        "J_max": int(J_max),
        "axis_labels": _wavefunc_axis_labels(rep),
        "k_axis_abc": k_axis,
        "quadrupole": quad_payload,
    }

def assign_KaKc_tau_species_from_expectation(v, J, Ja2, Jc2):
    """
    Representation-invariant assignment of (Ka, Kc, tau, species)
    using expectations <Ja^2>, <Jc^2> on eigenvector v.

    Enforces Ka,Kc integers with Ka+Kc in {J, J+1}.
    Chooses (Ka,Kc) minimizing ( <Ja^2>-Ka^2 )^2 + ( <Jc^2>-Kc^2 )^2.
    """
    # expectations
    EJa2 = max(0.0, _expectation(v, Ja2))
    EJc2 = max(0.0, _expectation(v, Jc2))

    Ka_eff = math.sqrt(EJa2)
    Kc_eff = math.sqrt(EJc2)

    # candidate integers near eff values
    cand_Ka = sorted(set([
        int(math.floor(Ka_eff + 1e-12)),
        int(math.ceil (Ka_eff - 1e-12)),
    ]))
    cand_Ka = [k for k in cand_Ka if 0 <= k <= J]
    if not cand_Ka:
        cand_Ka = [0]

    best = None
    for Ka in cand_Ka:
        # only two allowed Kc values by standard correlation
        for Kc in (J - Ka, J - Ka + 1):
            if not (0 <= Kc <= J):
                continue
            err = (EJa2 - Ka*Ka)**2 + (EJc2 - Kc*Kc)**2
            if (best is None) or (err < best[0]):
                best = (err, Ka, Kc)

    # fallback safety (shouldn't happen)
    if best is None:
        Ka = int(round(Ka_eff))
        Ka = max(0, min(J, Ka))
        # choose Kc branch closer to EJc2
        Kc0 = J - Ka
        Kc1 = J - Ka + 1
        candidates = []
        if 0 <= Kc0 <= J: candidates.append(Kc0)
        if 0 <= Kc1 <= J: candidates.append(Kc1)
        if not candidates: candidates = [max(0, min(J, int(round(Kc_eff))))]
        Kc = min(candidates, key=lambda kk: abs(EJc2 - kk*kk))
    else:
        _, Ka, Kc = best

    tau = 'e' if (Ka + Kc == J) else 'o'   # (Ka+Kc is J or J+1)

    pKa = 'e' if (Ka % 2 == 0) else 'o'
    pKc = 'e' if (Kc % 2 == 0) else 'o'
    species = pKa + pKc

    return Ka, Kc, tau, species, EJa2, EJc2


def assign_KaKc_tau_species_from_wang(v_w, J, rep):
    """
    Assign (Ka, Kc, tau, species) from dominant Wang-basis component.
    Valid when the z-axis is a or c (K corresponds to Ka or Kc).
    """
    z_ax = _active_rep_axes(rep)[0]
    if z_ax not in ("a", "c"):
        raise ValueError("Wang-based labels require z-axis a or c")

    m = int(np.argmax(np.abs(v_w)))
    if m == 0:
        K = 0
        sym_sign = 0  # only one state for K=0
    else:
        K = (m + 1) // 2
        sym_sign = 0 if (m % 2 == 1) else 1  # + -> 0, - -> 1

    tau_int = sym_sign ^ (K % 2)

    if z_ax == "a":
        Ka = K
        Kc = J - Ka + tau_int
    else:
        Kc = K
        Ka = J - Kc + tau_int

    Ka = int(max(0, min(J, Ka)))
    Kc = int(max(0, min(J, Kc)))

    tau = 'e' if (Ka + Kc == J) else 'o'
    pKa = 'e' if (Ka % 2 == 0) else 'o'
    pKc = 'e' if (Kc % 2 == 0) else 'o'
    species = pKa + pKc

    return Ka, Kc, tau, species


def _chi_make_traceless(chi_aa, chi_bb, chi_cc):
    """
    Remove the isotropic trace from diagonal quadrupole components.

    Inputs are the three diagonal Cartesian components; output is the
    traceless diagonal triplet used before spherical-tensor conversion.
    """
    tr = (chi_aa + chi_bb + chi_cc) / 3.0
    return chi_aa - tr, chi_bb - tr, chi_cc - tr


def chi_cart_to_sph_q(chi_aa, chi_bb, chi_cc, chi_ab, chi_ac, chi_bc, rep=None):
    """
    Convert a Cartesian quadrupole tensor into rank-2 spherical components.

    Parameters are the six Cartesian tensor components in MHz and an optional
    representation family. Output is a dictionary keyed by ``q=-2..+2`` in the
    internal x/y/z frame used by the active representation.
    """
    rep_key = _normalize_rep_key(globals().get("REPRESENTATION", REPRESENTATION) if rep is None else rep)
    z_ax, x_ax, y_ax = _active_rep_axes(rep_key)
    tensor = _cart_tensor_tuple_to_matrix(chi_aa, chi_bb, chi_cc, chi_ab, chi_ac, chi_bc)
    traceless = tensor - np.eye(3, dtype=float) * (float(np.trace(tensor)) / 3.0)
    axis_index = {"a": 0, "b": 1, "c": 2}
    iz = axis_index[z_ax]
    ix = axis_index[x_ax]
    iy = axis_index[y_ax]
    chi_zz = float(traceless[iz, iz])
    chi_xx = float(traceless[ix, ix])
    chi_yy = float(traceless[iy, iy])
    chi_zx = float(traceless[iz, ix])
    chi_zy = float(traceless[iz, iy])
    chi_xy = float(traceless[ix, iy])
    chi0 = 0.5 * (2.0 * chi_zz - chi_xx - chi_yy)
    chi_p1 = -(chi_zx + 1j * chi_zy)
    chi_m1 = +(chi_zx - 1j * chi_zy)
    chi_p2 = 0.5 * ((chi_xx - chi_yy) + 2j * chi_xy)
    chi_m2 = 0.5 * ((chi_xx - chi_yy) - 2j * chi_xy)
    return {-2: chi_m2, -1: chi_m1, 0: chi0, +1: chi_p1, +2: chi_p2}


def _remap_quadrupole_nucleus_to_internal(rep, nucleus):
    """
    Re-express one normalized quadrupole nucleus in the active internal frame.

    Input is a nucleus payload returned by :func:`_read_quadrupole_nucleus`;
    output is the same payload shape, but with spherical components rebuilt in
    the requested representation family.
    """
    chi_cart_int = tuple(float(v) for v in nucleus["chi_cart"])
    chi_dict_int = chi_cart_to_sph_q(*chi_cart_int, rep=rep)
    pickett_spin_reduced = float(
        nucleus.get("pickett_spin_reduced", pickett_quadrupole_reduced_spin_factor(nucleus["I"]))
    )
    if pickett_spin_reduced > 0.0:
        chi_pickett_dict_int = {q: value / pickett_spin_reduced for q, value in chi_dict_int.items()}
    else:
        chi_pickett_dict_int = dict(chi_dict_int)
    return {
        "index": int(nucleus["index"]),
        "I": float(nucleus["I"]),
        "chi_cart": tuple(float(v) for v in chi_cart_int),
        "chi_dict": chi_dict_int,
        "chi_pickett_dict": chi_pickett_dict_int,
        "pickett_spin_reduced": pickett_spin_reduced,
        "active": bool((float(nucleus["I"]) > 0.5) and any(abs(v) > 1e-15 for v in chi_cart_int)),
    }


def build_C2_matrix_JJp(J, Jp, q):
    """
    Build one rank-2 rotational tensor block between J and J'.

    Parameters
    ----------
    J, Jp : int
        Lower and upper rotational quantum numbers.
    q : int
        Spherical tensor component index in ``[-2, 2]``.

    Returns
    -------
    ndarray
        Complex matrix in the direct K basis.
    """
    K = np.arange(-J, J + 1, dtype=int)
    M = np.zeros((2 * J + 1, 2 * Jp + 1), dtype=complex)
    pref = math.sqrt((2 * J + 1) * (2 * Jp + 1))

    for i, Ki in enumerate(K):
        Kj = Ki - q
        if abs(Kj) > Jp:
            continue
        j = int(Kj + Jp)
        try:
            w = W3J(int(2 * J), 4, int(2 * Jp), int(-2 * Ki), int(2 * q), int(2 * Kj))
        except Exception:
            continue
        if abs(w) < 1e-14:
            continue
        # SPCAT-like phase convention for rank-2 rotational tensor blocks.
        phase = -1.0 if ((J - Ki) & 1) else 1.0
        M[i, j] = phase * pref * w
    return M


def build_HQ_in_K_JJp(J, Jp, chi_dict):
    """
    Assemble the quadrupole Hamiltonian block in the K basis for one J/J' pair.

    Input is the dictionary of spherical quadrupole components. Output is the
    complex ``(2J+1) x (2J'+1)`` matrix before Wang or F-coupled transforms.
    """
    HQ = np.zeros((2 * J + 1, 2 * Jp + 1), dtype=complex)
    for q, chiq in chi_dict.items():
        if abs(chiq) == 0:
            continue
        HQ += chiq * build_C2_matrix_JJp(J, Jp, q)
    return HQ


def _twoj(value):
    """Return the doubled-integer representation of a spin or angular momentum."""
    return int(round(2.0 * float(value)))


def _iter_halfint_values(two_min, two_max):
    """
    Enumerate half-integer values between two doubled bounds.

    Input bounds are doubled integers; output is a Python list of floats such
    as ``[0.5, 1.5, 2.5]``.
    """
    return [0.5 * two for two in range(int(two_min), int(two_max) + 1, 2)]


def _coupled_total_spin_values(nuclei):
    """
    Return all allowed coupled total-spin values for the active nuclei list.

    Supports zero, one, or two quadrupolar nuclei and returns the list of
    allowed ``I12`` values as floats.
    """
    if not nuclei:
        return []
    if len(nuclei) == 1:
        return [float(nuclei[0]["I"])]
    if len(nuclei) != 2:
        raise ValueError(f"Only up to two quadrupolar nuclei are supported. Got {len(nuclei)}.")
    i1 = float(nuclei[0]["I"])
    i2 = float(nuclei[1]["I"])
    return _iter_halfint_values(abs(_twoj(i1) - _twoj(i2)), _twoj(i1) + _twoj(i2))


def _other_spin_for_nucleus(nuclei, nucleus_index):
    """
    Return the spin of the companion nucleus in a two-spin coupling scheme.

    For one-nucleus problems the function returns 0.0, which keeps later
    recoupling formulas well defined.
    """
    for nuc in nuclei:
        if int(nuc["index"]) != int(nucleus_index):
            return float(nuc["I"])
    return 0.0


def _pickett_f_block_code(F_max):
    """
    Convert the user-visible ``F_max`` limit into Pickett's integer block code.

    Returns ``None`` when no valid finite limit is supplied.
    """
    if F_max is None:
        return None
    try:
        fv = float(F_max)
    except Exception:
        return None
    if fv < 0.0:
        return None
    return int(math.floor(fv + 0.5 + 1e-12))


def _pickett_knmax_from_fmax(F_max):
    """Translate ``F_max`` into the Wang-basis truncation index used for HFS support."""
    return _pickett_f_block_code(F_max)


def _hfs_rotational_support_jmax(J_max, nuclei, F_max):
    """
    Extend the rotational support range needed by the HFS builder.

    The direct-F quadrupole Hamiltonian couples J blocks with ``Delta J=0,±2``.
    This helper returns the enlarged J ceiling required to avoid truncating the
    visible manifold near the upper edge of the request.
    """
    j_max = int(J_max)
    if not nuclei:
        return j_max
    support_max = j_max
    # The quadrupole operator connects J blocks with Delta J = 0, +/-2.
    # Keep two extra rotational shells so the upper edge of the requested
    # manifold is not artificially truncated by missing J+2 coupling partners.
    edge_margin = 2
    if F_max is None:
        return support_max + edge_margin
    i12_values = _coupled_total_spin_values(nuclei)
    if not i12_values:
        return support_max + edge_margin
    support_max = max(support_max, int(math.ceil(float(F_max) + max(i12_values) - 1e-12)))
    return support_max + edge_margin


def _wang_basis_k_value(w_idx):
    """
    Recover the |K| value associated with one Wang-basis column index.

    Input is the column number in the Wang ordering; output is the associated
    non-negative integer K quantum number.
    """
    w_idx = int(w_idx)
    if w_idx <= 0:
        return 0
    return (w_idx + 1) // 2


def _wang_basis_selection_by_knmax(J, knmax):
    """
    Select the Wang-basis columns compatible with a maximum |K| truncation.

    Returns the list of retained Wang indices for one J shell.
    """
    dim = 2 * int(J) + 1
    if knmax is None:
        return list(range(dim))
    return [idx for idx in range(dim) if _wang_basis_k_value(idx) <= int(knmax)]


def _quadrupole_rotational_tensor_scale(rotor_type):
    """
    SPCAT uses a slightly different rank-2 rotational normalization for the
    a-axis symmetric-top embeddings (prolate and linear K=0 projection) than
    the generic asymmetric-top direct-K tensor used here. The oblate branch is
    already aligned by the internal a<->c remap and does not require this
    extra factor.
    """
    rt = str(rotor_type).strip().lower()
    if rt in ("linear", "prolate"):
        return math.sqrt(5.0 / 6.0)
    return 1.0


def _expand_truncated_wang_vector(v_sub, full_indices, full_dim):
    """
    Reinsert a truncated Wang-space vector into the full Wang basis.

    Parameters are the reduced vector, the retained full-space indices, and
    the target full dimension. Output is the expanded complex vector.
    """
    if (full_indices is None) or (len(full_indices) == int(full_dim)):
        return v_sub
    out = np.zeros(int(full_dim), dtype=complex)
    out[np.asarray(full_indices, dtype=int)] = np.asarray(v_sub, dtype=complex)
    return out


def pickett_quadrupole_reduced_spin_factor(I_self):
    """
    Pickett's internal reduced-spin normalization for a nuclear quadrupole
    tensor, i.e. the product spfac[ii] * spfac2[ii] with ii = 2I from
    spinv.c:pasort/checksp().

    The user-facing chi_ij remain physical quadrupole constants in MHz.
    We factor this normalization analytically so the Hamiltonian no longer
    depends on the empirical CHI_SCALE knob.
    """
    I_self = float(I_self)
    if I_self <= 0.5:
        return 0.0
    ii = _twoj(I_self)
    dtmp = 0.5 * ii
    spfac = math.sqrt(dtmp * (dtmp + 1.0) * (ii + 1))
    spfac2 = 0.25 * math.sqrt((dtmp + 1.5) / (dtmp - 0.5)) / dtmp
    return spfac * spfac2


def hyperfine_factor_energy_coupled(J, Jp, I12, I12p, F):
    """
    Return the standard 6-j recoupling factor for quadrupole energy blocks.

    Inputs are rotational quantum numbers, coupled nuclear-spin labels, and
    total F. Output is the scalar prefactor multiplying the rotational tensor
    block in the coupled representation.
    """
    two = int(round(2 * (Jp + I12 + F)))
    phase = (-1) ** (two // 2)
    sixj = W6J(int(2 * J), int(2 * Jp), 4, _twoj(I12p), _twoj(I12), int(2 * F))
    return phase * sixj


def spin_tensor_recoupling_factor(I12, I12p, I_self, I_other, active_position=1):
    """
    Recoupling factor for a rank-2 quadrupole tensor acting on one nucleus
    inside the coupled spin basis |(I1 I2) I12>.

    The Pickett phase depends on whether the active tensor acts on the first
    or second spin in the coupling tree. For the first spin the phase anchor
    is I12p; for the second spin it is I12. Using a single formula for both
    nuclei flips the sign of some off-diagonal I12 couplings.
    """
    phase_anchor = I12p
    if (I_other > 0.0) and (int(active_position) == 2):
        phase_anchor = I12
    two = int(round(2 * (I_other + I_self + phase_anchor + 2.0)))
    phase = (-1) ** (two // 2)
    return phase * math.sqrt((2 * I12 + 1) * (2 * I12p + 1)) * W6J(
        _twoj(I_self), _twoj(I12), _twoj(I_other),
        _twoj(I12p), _twoj(I_self), 4
    )


def pickett_spin_tensor_factor(I12, I12p, I_self, I_other, active_position=1):
    """
    Combine Pickett's reduced-spin normalization with the recoupling factor.

    Output is the full scalar prefactor for one nucleus acting inside a
    coupled-spin basis.
    """
    return pickett_quadrupole_reduced_spin_factor(I_self) * spin_tensor_recoupling_factor(
        I12, I12p, I_self, I_other, active_position=active_position
    )


_SPIN_COMPONENT_CACHE = {}
_EXACT_HFS_COUPLING_CACHE = {}


def CG(j1_2, m1_2, j2_2, m2_2, J_2, M_2):
    """
    Compute one Clebsch-Gordan coefficient from cached Wigner-3j values.

    All quantum numbers are passed as doubled integers. Output is a float
    following the phase convention used elsewhere in this module.
    """
    if int(m1_2) + int(m2_2) != int(M_2):
        return 0.0
    phase_exp = (int(j1_2) - int(j2_2) + int(M_2)) // 2
    phase = -1.0 if (phase_exp & 1) else 1.0
    return phase * math.sqrt(int(J_2) + 1) * W3J(
        int(j1_2), int(j2_2), int(J_2), int(m1_2), int(m2_2), int(-M_2)
    )


def _single_spin_component_lookup(twoI, q, reduced):
    """
    Tabulate one single-nucleus rank-2 spin operator in the m_I basis.

    Inputs are doubled spin, spherical component q, and reduced-spin factor.
    Output is a sparse dictionary keyed by ``(m, m')`` doubled integers.
    """
    lookup = {}
    for two_m in range(-int(twoI), int(twoI) + 1, 2):
        phase_exp = (int(twoI) - int(two_m)) // 2
        phase = -1.0 if (phase_exp & 1) else 1.0
        for two_mp in range(-int(twoI), int(twoI) + 1, 2):
            val = phase * W3J(int(twoI), 4, int(twoI), int(-two_m), int(2 * q), int(two_mp)) * float(reduced)
            if abs(val) > 1e-14:
                lookup[(int(two_m), int(two_mp))] = complex(val)
    return lookup


def _spin_component_lookup(nuclei, active_position, q):
    """
    Build or reuse the coupled-spin operator lookup for one active nucleus.

    Depending on whether one or two nuclei are present, the function returns a
    dictionary of matrix elements in the ``|I12, M12>`` basis for a given q.
    """
    spins_key = tuple(_twoj(nuc["I"]) for nuc in nuclei)
    cache_key = (spins_key, int(active_position), int(q))
    if cache_key in _SPIN_COMPONENT_CACHE:
        return _SPIN_COMPONENT_CACHE[cache_key]

    if len(nuclei) == 1:
        twoI = int(spins_key[0])
        reduced = pickett_quadrupole_reduced_spin_factor(0.5 * twoI)
        single = _single_spin_component_lookup(twoI, q, reduced)
        lookup = {}
        for (two_m, two_mp), val in single.items():
            lookup[(twoI, two_m, twoI, two_mp)] = val
        _SPIN_COMPONENT_CACHE[cache_key] = lookup
        return lookup

    if len(nuclei) != 2:
        raise ValueError(f"Only up to two quadrupolar nuclei are supported. Got {len(nuclei)}.")

    twoI1, twoI2 = map(int, spins_key)
    target_twoI = twoI1 if int(active_position) == 1 else twoI2
    reduced = pickett_quadrupole_reduced_spin_factor(0.5 * target_twoI)
    single = _single_spin_component_lookup(target_twoI, q, reduced)

    product_basis = []
    for two_m1 in range(-twoI1, twoI1 + 1, 2):
        for two_m2 in range(-twoI2, twoI2 + 1, 2):
            product_basis.append((two_m1, two_m2))

    coupled_basis = []
    twoI12_min = abs(twoI1 - twoI2)
    twoI12_max = twoI1 + twoI2
    for twoI12 in range(twoI12_min, twoI12_max + 1, 2):
        for twoM in range(-twoI12, twoI12 + 1, 2):
            coupled_basis.append((twoI12, twoM))

    U = np.zeros((len(product_basis), len(coupled_basis)), dtype=float)
    for prod_idx, (two_m1, two_m2) in enumerate(product_basis):
        twoM = two_m1 + two_m2
        for coup_idx, (twoI12, twoM12) in enumerate(coupled_basis):
            if int(twoM) != int(twoM12):
                continue
            U[prod_idx, coup_idx] = CG(twoI1, two_m1, twoI2, two_m2, twoI12, twoM)

    op = np.zeros((len(product_basis), len(product_basis)), dtype=complex)
    for row, (two_m1, two_m2) in enumerate(product_basis):
        for col, (two_mp1, two_mp2) in enumerate(product_basis):
            if int(active_position) == 1:
                if int(two_m2) != int(two_mp2):
                    continue
                op[row, col] = single.get((two_m1, two_mp1), 0.0)
            else:
                if int(two_m1) != int(two_mp1):
                    continue
                op[row, col] = single.get((two_m2, two_mp2), 0.0)

    coupled_op = U.T @ op @ U
    lookup = {}
    for row, (twoI12, twoM) in enumerate(coupled_basis):
        for col, (twoI12p, twoMp) in enumerate(coupled_basis):
            val = coupled_op[row, col]
            if abs(val) > 1e-14:
                lookup[(int(twoI12), int(twoM), int(twoI12p), int(twoMp))] = complex(val)

    _SPIN_COMPONENT_CACHE[cache_key] = lookup
    return lookup


def exact_hfs_scalar_coupling_factor(J, Jp, I12, I12p, F, nuclei, active_position):
    """
    Evaluate the exact direct-F quadrupole recoupling scalar.

    This routine contracts rotational Wigner terms, coupled-spin matrix
    elements, and Clebsch-Gordan coefficients explicitly. Output is the complex
    scalar multiplying the ``J/J'`` quadrupole block.
    """
    spins_key = tuple(_twoj(nuc["I"]) for nuc in nuclei)
    cache_key = (
        int(J), int(Jp), _twoj(I12), _twoj(I12p), _twoj(F), spins_key, int(active_position)
    )
    if cache_key in _EXACT_HFS_COUPLING_CACHE:
        return _EXACT_HFS_COUPLING_CACHE[cache_key]

    twoF = _twoj(F)
    twoI12 = _twoj(I12)
    twoI12p = _twoj(I12p)
    total = 0.0 + 0.0j

    for p in range(-2, 3):
        spin_lookup = _spin_component_lookup(nuclei, active_position, -p)
        phase_p = -1.0 if (int(p) & 1) else 1.0
        for twoMJ in range(-2 * int(J), 2 * int(J) + 1, 2):
            twoMI = twoF - twoMJ
            if abs(twoMI) > twoI12:
                continue
            if ((twoMI + twoI12) & 1) != 0:
                continue

            twoMJp = twoMJ - 2 * int(p)
            if abs(twoMJp) > 2 * int(Jp):
                continue
            if ((twoMJp + 2 * int(Jp)) & 1) != 0:
                continue

            twoMIp = twoF - twoMJp
            if abs(twoMIp) > twoI12p:
                continue
            if ((twoMIp + twoI12p) & 1) != 0:
                continue

            cg_left = CG(2 * int(J), twoMJ, twoI12, twoMI, twoF, twoF)
            cg_right = CG(2 * int(Jp), twoMJp, twoI12p, twoMIp, twoF, twoF)
            if abs(cg_left) < 1e-14 or abs(cg_right) < 1e-14:
                continue

            w = W3J(2 * int(J), 4, 2 * int(Jp), int(-twoMJ), int(2 * p), int(twoMJp))
            if abs(w) < 1e-14:
                continue

            phase_m = -1.0 if ((((2 * int(J)) - int(twoMJ)) // 2) & 1) else 1.0
            spin_val = spin_lookup.get((twoI12, twoMI, twoI12p, twoMIp), 0.0)
            if abs(spin_val) < 1e-14:
                continue

            total += phase_p * cg_left * cg_right * phase_m * w * spin_val

    total = complex(np.real_if_close(total, tol=1000))
    if abs(total.imag) < 1e-12:
        total = total.real
    _EXACT_HFS_COUPLING_CACHE[cache_key] = total
    return total


def hyperfine_factor_intensity(Jl, Fl, Ju, Fu, I):
    """
    Return the optional HFS intensity prefactor configured by the current mode.

    In the default mode the factor is 1.0 because the recoupling is already
    included in the transition amplitude matrix.
    """
    # Recoupling via 6-j is already included in the dipole amplitude matrix
    # (Aamp). Keep default factor = 1 to avoid double counting.
    mode = str(globals().get("HFS_INTENSITY_MODE", HFS_INTENSITY_MODE)).lower()
    sixj = W6J(int(2 * Jl), int(2 * Fl), int(2 * I), int(2 * Fu), int(2 * Ju), 2)
    if mode == "full":
        return (2 * Fl + 1) * (2 * Fu + 1) * (sixj * sixj)
    if mode == "6j2":
        return sixj * sixj
    if mode == "sqrt":
        return math.sqrt((2 * Fl + 1) * (2 * Fu + 1)) * abs(sixj)
    return 1.0


def e1_recoupling_amp(Jl, Fl, Ju, Fu, I):
    """
    Return the E1 recoupling amplitude between two F-coupled states.

    Inputs are lower/upper ``(J,F)`` values and the spectator spin. Output is
    the square-root-weighted 6-j amplitude used in transition matrices.
    """
    sixj = W6J(int(2 * Jl), int(2 * Fl), int(2 * I), int(2 * Fu), int(2 * Ju), 2)
    return math.sqrt((2 * Fl + 1) * (2 * Fu + 1)) * sixj


def _build_wang_basis_labels(J, rep_user, W, rep_int="Ir", full_indices=None):
    """
    Build Ka/Kc/species labels for each Wang-basis basis vector.

    Depending on the active quantization axis, labels are assigned either from
    dominant Wang components or from expectation values in the K basis.
    Output is a list aligned with the Wang basis ordering.
    """
    dim = W.shape[1]
    eye = np.eye(dim, dtype=float)
    labels = []
    use_wang_labels = _active_rep_axes(rep_user)[0] in ("a", "c")
    if use_wang_labels:
        for idx in range(dim):
            v_w = eye[:, idx]
            v_w = _expand_truncated_wang_vector(v_w, full_indices, 2 * J + 1)
            Ka, Kc, _tau, species = assign_KaKc_tau_species_from_wang(v_w, J, rep_user)
            labels.append((int(Ka), int(Kc), str(species)))
        return labels

    Ja, Jb, Jc, _ = _rot_ops_rep(J, rep=rep_user)
    Ja2 = Ja @ Ja
    Jc2 = Jc @ Jc
    W_eff = np.eye(dim, dtype=float) if W is None else W
    for idx in range(dim):
        v_k = W_eff @ eye[:, idx]
        Ka, Kc, _tau, species, _eja2, _ejc2 = assign_KaKc_tau_species_from_expectation(v_k, J, Ja2, Jc2)
        labels.append((int(Ka), int(Kc), str(species)))
    return labels


def build_F_basis_direct(rot_blocks_w, wang_basis_labels, nuclei, F_max=None):
    """
    Construct the direct-F basis bookkeeping for all active F blocks.

    Outputs are ``basis_by_F`` and ``groups_by_F``, which map each F value to
    its basis states and to the slices belonging to each ``(J, I12)`` sector.
    """
    basis_by_F = {}
    groups_by_F = {}
    i12_values = _coupled_total_spin_values(nuclei)
    for J in sorted(int(v) for v in rot_blocks_w.keys()):
        dim = int(rot_blocks_w[J].shape[0])
        for I12 in i12_values:
            Fmin = abs(J - I12)
            Fmax = J + I12
            twoFmin = int(round(2 * Fmin))
            twoFmax = int(round(2 * Fmax))
            for twoF in range(twoFmin, twoFmax + 1, 2):
                F = 0.5 * twoF
                if (F_max is not None) and (F > F_max):
                    continue
                basis = basis_by_F.setdefault(F, [])
                start = len(basis)
                for w_idx in range(dim):
                    Ka, Kc, sp = wang_basis_labels[J][w_idx]
                    basis.append((int(J), int(w_idx), float(I12), int(Ka), int(Kc), str(sp)))
                stop = len(basis)
                groups_by_F.setdefault(F, []).append({
                    "J": int(J),
                    "I12": float(I12),
                    "slice": slice(start, stop),
                })
    return basis_by_F, groups_by_F


def build_and_diag_HF_fullF(
    levels,
    rot_blocks_w,
    rot_eigvecs_w,
    wang_transforms,
    wang_basis_labels,
    nuclei,
    F_max=None,
    rotor_type="asymmetric",
):
    """
    Build and diagonalize the full hyperfine Hamiltonian block-by-block in F.

    Inputs are precomputed rotational blocks, Wang transforms, basis labels,
    quadrupole nuclei, and the optional F limit. Outputs are the diagonalized
    HFS blocks plus label/basis bookkeeping used later for spectra and tables.
    """
    if not nuclei:
        return None, None, None, None

    if all(all(abs(v) < 1e-15 for v in nuc["chi_dict"].values()) for nuc in nuclei):
        return None, None, None, None

    available_js = _sorted_level_js(levels)
    if not available_js:
        return None, None, None, None

    basis_by_F, groups_by_F = build_F_basis_direct(rot_blocks_w, wang_basis_labels, nuclei, F_max=F_max)
    HQ_w_cache = {}
    hfF_blocks = {}
    hfF_labels = {}
    used_hyperfine = False
    rot_scale = _quadrupole_rotational_tensor_scale(rotor_type)

    for F, basis in basis_by_F.items():
        groups = groups_by_F.get(F, [])
        n = len(basis)
        if n == 0 or not groups:
            continue

        H = np.zeros((n, n), dtype=complex)

        for group in groups:
            sl = group["slice"]
            H[sl, sl] += rot_blocks_w[group["J"]]

        for nucleus in nuclei:
            chi_dict = nucleus["chi_pickett_dict"]
            if rot_scale != 1.0:
                chi_dict = {q: rot_scale * value for q, value in chi_dict.items()}
            if all(abs(v) < 1e-15 for v in chi_dict.values()):
                continue
            active_position = 1
            if len(nuclei) >= 2:
                # The exact direct-F recoupling uses the opposite spin-slot
                # ordering from the external nucleus numbering.
                active_position = 2 if int(nucleus["index"]) == int(nuclei[0]["index"]) else 1
            for group_i in groups:
                J = group_i["J"]
                I12 = group_i["I12"]
                sl_i = group_i["slice"]
                for group_j in groups:
                    Jp = group_j["J"]
                    if abs(Jp - J) > 2:
                        continue
                    I12p = group_j["I12"]
                    sl_j = group_j["slice"]
                    key = (int(nucleus["index"]), int(J), int(Jp))
                    if key not in HQ_w_cache:
                        HQ_K = build_HQ_in_K_JJp(J, Jp, chi_dict)
                        WJ = wang_transforms[J]
                        WJp = wang_transforms[Jp]
                        HQ_w_cache[key] = WJ.T @ HQ_K @ WJp
                    if np.max(np.abs(HQ_w_cache[key])) < 1e-14:
                        continue
                    fac = exact_hfs_scalar_coupling_factor(
                        J, Jp, I12, I12p, F, nuclei, active_position
                    )
                    if abs(fac) < 1e-14:
                        continue
                    used_hyperfine = True
                    H[sl_i, sl_j] += fac * HQ_w_cache[key]

        if not used_hyperfine:
            continue
        H = 0.5 * (H + H.conjugate().T)
        Ew, C = _matrix_eigh(H)
        idx = np.argsort(Ew.real)
        Ew = Ew.real[idx]
        C = C[:, idx]
        hfF_blocks[F] = (Ew, C)

        labels = []
        for alpha in range(n):
            coeff = C[:, alpha]
            coeff_w = np.abs(coeff) ** 2
            dom_idx = int(np.argmax(coeff_w)) if coeff_w.size else 0
            J_basis, _w_basis, I12_basis, Ka_basis, Kc_basis, sp_basis = basis[dom_idx]
            parent_weights = {}
            i12_weights = {}
            for group in groups:
                sl = group["slice"]
                blk = coeff[sl]
                blk_norm = float(np.vdot(blk, blk).real)
                if blk_norm <= 1e-18:
                    continue
                I12 = float(group["I12"])
                i12_weights[I12] = i12_weights.get(I12, 0.0) + blk_norm
                U_w = rot_eigvecs_w[group["J"]]
                rot_amp = U_w.conj().T @ blk
                rot_w = np.abs(rot_amp) ** 2
                for tau, wt in enumerate(rot_w):
                    if wt <= 1e-18:
                        continue
                    key = (int(group["J"]), int(tau))
                    parent_weights[key] = parent_weights.get(key, 0.0) + float(wt)

            if parent_weights:
                (Jd, taud), purity = max(parent_weights.items(), key=lambda item: item[1])
            else:
                Jd = int(J_basis)
                taud = 0
                purity = 0.0

            E_rot, _Ka_parent, _Kc_parent, _sp_parent = levels[Jd][taud]
            Ka_w = sum(wt * levels[Jt][tau][1] for (Jt, tau), wt in parent_weights.items()) if parent_weights else float(Ka_basis)
            Kc_w = sum(wt * levels[Jt][tau][2] for (Jt, tau), wt in parent_weights.items()) if parent_weights else float(Kc_basis)
            Ka = int(Ka_basis)
            Kc = int(Kc_basis)
            sp = str(sp_basis)
            I12_dom = float(I12_basis)
            labels.append((
                float(F), int(alpha), float(Ew[alpha]),
                int(Jd), int(taud), int(Ka), int(Kc), str(sp),
                float(purity), float(Ka_w), float(Kc_w), float(I12_dom)
            ))
        hfF_labels[F] = labels

    if not used_hyperfine:
        return None, None, None, None

    return hfF_blocks, hfF_labels, basis_by_F, groups_by_F


def make_hyperfine_levels_table(hfF_labels, to_cm1=False):
    """
    Convert the internal HFS label payload into a sortable DataFrame.

    Input is the ``hfF_labels`` dictionary produced by the direct-F builder.
    Output is a pandas table with one row per HFS level, optionally converted
    from MHz to cm^-1.
    """
    rows = []
    for F, lst in hfF_labels.items():
        for rec in lst:
            if len(rec) >= 12:
                (Fv, alpha, E_MHz, J_dom, tau_dom, Ka, Kc, sp, purity, Ka_w, Kc_w, I12_dom) = rec[:12]
            else:
                (Fv, alpha, E_MHz, J_dom, tau_dom, Ka, Kc, sp, purity, Ka_w, Kc_w) = rec
                I12_dom = np.nan
            E_out = E_MHz / 29979.2458 if to_cm1 else E_MHz
            rows.append((J_dom, Fv, alpha, E_out, Ka, Kc, sp, tau_dom, purity, Ka_w, Kc_w, I12_dom))
    cols = [
        'J_dom',
        'F',
        'alpha',
        'Energy(cm^-1)' if to_cm1 else 'Energy(MHz)',
        'Ka',
        'Kc',
        'Parity',
        'tau_dom',
        'purity',
        'Ka_w',
        'Kc_w',
        'I12_dom',
    ]
    sort_col = 'Energy(cm^-1)' if to_cm1 else 'Energy(MHz)'
    return pd.DataFrame(rows, columns=cols).sort_values(['J_dom', 'F', sort_col])


def _filter_hf_labels_by_jmax(hfF_labels, j_max_visible):
    """
    Drop HFS labels whose dominant parent J lies above the visible J ceiling.

    This is a lightweight filter used when only the dominant-parent label is
    needed and no basis-aware projection is required.
    """
    if hfF_labels is None:
        return None
    j_lim = int(j_max_visible)
    filtered = {}
    for F, recs in hfF_labels.items():
        keep = [rec for rec in recs if int(rec[3]) <= j_lim]
        if keep:
            filtered[F] = keep
    return filtered


def _filter_hf_labels_by_visible_subspace(hfF_labels, hfF_blocks, basis_by_F, j_max_visible):
    """
    Keep only the HFS states that overlap the requested visible rotational subspace.

    Compared with :func:`_filter_hf_labels_by_jmax`, this version uses the full
    F-block eigenvectors to rank states by their weight inside the retained
    rotational basis.
    """
    if hfF_labels is None:
        return None
    j_lim = int(j_max_visible)
    filtered = {}
    for F, recs in hfF_labels.items():
        if (F not in hfF_blocks) or (F not in basis_by_F):
            keep = [rec for rec in recs if int(rec[3]) <= j_lim]
            if keep:
                filtered[F] = keep
            continue
        _Ew, C = hfF_blocks[F]
        basis = basis_by_F[F]
        vis_mask = np.asarray([int(rec[0]) <= j_lim for rec in basis], dtype=bool)
        n_visible = int(np.count_nonzero(vis_mask))
        if n_visible <= 0:
            continue
        if n_visible >= len(recs):
            filtered[F] = list(recs)
            continue
        weights = np.sum(np.abs(C[vis_mask, :]) ** 2, axis=0)
        keep_alpha = {
            int(alpha)
            for alpha in sorted(range(len(recs)), key=lambda idx: (-float(weights[idx]), idx))[:n_visible]
        }
        keep = [rec for rec in recs if int(rec[1]) in keep_alpha]
        if keep:
            filtered[F] = keep
    return filtered


def _build_asymmetric_structure(
    A, B, C,
    DJ=0, DJK=0, DK=0, dJ=0, dK=0,
    HJ=0, HJK=0, HKJ=0, HK=0, h1=0, h2=0, h3=0,
    J_max=30, wang_symmetry=True,
    rotor_type="asymmetric", J_min=0
):
    """
    Precompute and cache all J blocks needed for an asymmetric-rotor run.

    The returned structure contains much more than eigenvalues: it stores the
    Wang basis transforms, optional HFS-resolved blocks, transition labels,
    and the metadata consumed later by the wavefunction viewer. Prediction and
    fitting reuse this cache so repeated intensity evaluations stay cheap.
    """
    global LAST_DF_HF, LAST_WAVEFUNC_CACHE
    J_min, J_max = _normalize_j_range(J_min, J_max)

    rep_user = _normalize_rep_key(globals().get("REPRESENTATION", "Ir"))
    reduction = str(globals().get("REDUCTION", REDUCTION)).strip().upper()
    A_int = float(A)
    B_int = float(B)
    C_int = float(C)
    rep_int = rep_user
    quartic_int = np.asarray([DJ, DJK, DK, dJ, dK], dtype=float)
    sextic_int = np.asarray([HJ, HJK, HKJ, HK, h1, h2, h3], dtype=float)

    levels = {}
    eigs = {}
    eigvecs_k = {}
    rot_blocks_w = {}
    rot_eigvecs_w = {}
    wang_transforms = {}
    wang_basis_labels = {}
    nuclei = list(_active_quadrupole_nuclei(rep=rep_user))
    F_max_raw = globals().get("F_MAX", F_MAX)
    F_max_loc = None
    if F_max_raw is not None:
        try:
            fv = float(F_max_raw)
            if fv > 0.0:
                F_max_loc = fv
        except Exception:
            F_max_loc = None
    hfs_knmax = _pickett_knmax_from_fmax(F_max_loc) if nuclei else None
    J_build_max = _hfs_rotational_support_jmax(J_max, nuclei, F_max_loc)

    for J in range(J_min, J_build_max + 1):
        H = H_matrix(
            A_int,
            B_int,
            C_int,
            quartic_int[0],
            quartic_int[1],
            quartic_int[2],
            quartic_int[3],
            quartic_int[4],
            J,
            reduction=reduction,
            rep=rep_int,
            HJ=sextic_int[0],
            HJK=sextic_int[1],
            HKJ=sextic_int[2],
            HK=sextic_int[3],
            h1=sextic_int[4],
            h2=sextic_int[5],
            h3=sextic_int[6],
        )
        W_full = wang_transform(J) if wang_symmetry else None
        full_basis_indices = None
        if W_full is not None:
            H_w_full = W_full.T @ H @ W_full
            if (hfs_knmax is not None) and (len(nuclei) > 0):
                full_basis_indices = _wang_basis_selection_by_knmax(J, hfs_knmax)
                if len(full_basis_indices) < H_w_full.shape[0]:
                    H_w = H_w_full[np.ix_(full_basis_indices, full_basis_indices)]
                    W = W_full[:, full_basis_indices]
                else:
                    full_basis_indices = list(range(H_w_full.shape[0]))
                    H_w = H_w_full
                    W = W_full
            else:
                H_w = H_w_full
                W = W_full
                full_basis_indices = list(range(H_w.shape[0]))
            E, U_w = _matrix_eigh(H_w)
        else:
            H_w = H
            W = None
            E, U_w = _matrix_eigh(H_w)

        idx = np.argsort(E.real)
        E = E.real[idx]
        U_w = U_w[:, idx]

        degen_tol = float(globals().get("DEGEN_TOL", 0.0))
        U_w = _fix_degenerate_evecs_by_K2(E, U_w, J, W if W is not None else np.eye(U_w.shape[0]), tol=degen_tol)

        U_k = (W @ U_w) if W is not None else U_w

        Ja, Jb, Jc, Jz = _rot_ops_rep(J, rep=rep_user)
        Ja2 = Ja @ Ja
        Jc2 = Jc @ Jc

        Ka_list, Kc_list, spec_list = [], [], []
        use_wang_labels = (W is not None and _active_rep_axes(rep_user)[0] in ("a", "c"))
        if use_wang_labels:
            for v_w in U_w.T:
                v_w_full = _expand_truncated_wang_vector(v_w, full_basis_indices, 2 * J + 1)
                Ka, Kc, tau, species = assign_KaKc_tau_species_from_wang(v_w_full, J, rep_user)
                Ka_list.append(Ka)
                Kc_list.append(Kc)
                spec_list.append(species)
        else:
            for v in U_k.T:
                Ka, Kc, tau, species, _, _ = assign_KaKc_tau_species_from_expectation(v, J, Ja2, Jc2)
                Ka_list.append(Ka)
                Kc_list.append(Kc)
                spec_list.append(species)

        levels[J] = list(zip(E, Ka_list, Kc_list, spec_list))
        eigs[J] = (E, U_w, W)
        eigvecs_k[J] = U_k
        rot_blocks_w[J] = H_w
        rot_eigvecs_w[J] = U_w
        wang_transforms[J] = np.eye(H_w.shape[0], dtype=float) if W is None else W
        wang_basis_labels[J] = _build_wang_basis_labels(
            J,
            rep_user,
            wang_transforms[J],
            rep_int=rep_int,
            full_indices=full_basis_indices,
        )

    use_hfs = False
    hfF_blocks = hfF_labels = basis_by_F = groups_by_F = None
    if nuclei:
        hf_pack = build_and_diag_HF_fullF(
            levels,
            rot_blocks_w,
            rot_eigvecs_w,
            wang_transforms,
            wang_basis_labels,
            nuclei,
            F_max=F_max_loc,
            rotor_type=rotor_type,
        )
        use_hfs = (hf_pack[0] is not None)
        if use_hfs:
            hfF_blocks, hfF_labels, basis_by_F, groups_by_F = hf_pack

    df_hf = make_hyperfine_levels_table(hfF_labels, to_cm1=True) if use_hfs else None
    LAST_DF_HF = df_hf

    wf_rot_levels = []
    wf_states = {}
    wf_hfs_levels = []
    for J in range(J_min, J_max + 1):
        U_k = eigvecs_k[J]
        Ja_i, Jb_i, Jc_i, _ = _rot_ops_rep(J, rep=rep_user)
        Ja2_i = Ja_i @ Ja_i
        Jb2_i = Jb_i @ Jb_i
        Jc2_i = Jc_i @ Jc_i
        k_values = list(range(-J, J + 1))

        for alpha, (E, Ka, Kc, species) in enumerate(levels[J]):
            v = U_k[:, alpha]
            EJa2 = _expectation(v, Ja2_i)
            EJb2 = _expectation(v, Jb2_i)
            EJc2 = _expectation(v, Jc2_i)
            axis_values = {"a": float(EJa2), "b": float(EJb2), "c": float(EJc2)}
            axis_expect = dict(axis_values)
            coord_expect = _wavefunc_coords_from_axis_values(axis_values, rep_user)
            key = f"{J}:{alpha}"
            state = {
                "key": key,
                "J": int(J),
                "alpha": int(alpha),
                "Ka": int(Ka),
                "Kc": int(Kc),
                "species": str(species),
                "energy_mhz": float(E),
                "energy_cm": float(E) / 29979.2458,
                "k_values": k_values,
                "coeff_re": [float(x) for x in np.real(v)],
                "coeff_im": [float(x) for x in np.imag(v)],
                "axis_expect": axis_expect,
                "coord_expect": coord_expect,
            }
            wf_states[key] = state
            wf_rot_levels.append({
                "id": key,
                "kind": "rot",
                "J": int(J),
                "alpha": int(alpha),
                "Ka": int(Ka),
                "Kc": int(Kc),
                "species": str(species),
                "energy_mhz": float(E),
                "energy_cm": float(E) / 29979.2458,
                "label": f"J={int(J)} alpha={int(alpha)} Ka={int(Ka)} Kc={int(Kc)}",
                "parent_key": key,
                "purity": 1.0,
            })

    if use_hfs:
        for F, recs in hfF_labels.items():
            for rec in recs:
                (
                    Fv, alpha_hf, E_hf_mhz,
                    J_dom, tau_dom, Ka, Kc, sp,
                    purity, Ka_w, Kc_w, *extra_rec
                ) = rec
                parent_key = f"{int(J_dom)}:{int(tau_dom)}"
                I12_dom = float(extra_rec[0]) if extra_rec else float("nan")
                wf_hfs_levels.append({
                    "id": f"F={float(Fv):g}:{int(alpha_hf)}",
                    "kind": "hfs",
                    "F": float(Fv),
                    "alpha_hf": int(alpha_hf),
                    "J": int(J_dom),
                    "alpha": int(tau_dom),
                    "Ka": int(Ka),
                    "Kc": int(Kc),
                    "species": str(sp),
                    "energy_mhz": float(E_hf_mhz),
                    "energy_cm": float(E_hf_mhz) / 29979.2458,
                    "Ka_w": float(Ka_w),
                    "Kc_w": float(Kc_w),
                    "I12_dom": I12_dom,
                    "purity": float(purity),
                    "parent_key": parent_key,
                    "label": (
                        f"J={int(J_dom)} F={float(Fv):g} alpha={int(alpha_hf)} "
                        f"Ka={int(Ka)} Kc={int(Kc)}"
                    ),
                })

    wavefunc_cache = {
        "meta": _make_wavefunc_cache_meta(
            rotor_type,
            rep_user,
            globals().get("REDUCTION", REDUCTION),
            "matrix",
            J_min,
            J_max
        ),
        "rot_levels": wf_rot_levels,
        "hfs_levels": wf_hfs_levels,
        "states": wf_states,
    }
    LAST_WAVEFUNC_CACHE = wavefunc_cache
    return {
        "engine": "matrix",
        "levels": levels,
        "eigs": eigs,
        "eigvecs_k": eigvecs_k,
        "rep_int": rep_int,
        "rep_user": rep_user,
        "J_min": int(J_min),
        "J_max": int(J_max),
        "use_hfs": bool(use_hfs),
        "hfF_blocks": hfF_blocks,
        "hfF_labels": hfF_labels,
        "basis_by_F": basis_by_F,
        "groups_by_F": groups_by_F,
        "nuclei": nuclei,
        "wavefunc_cache": wavefunc_cache,
        "df_hf": df_hf,
        "mu_tau_cache": {},
        "mu_wang_cache": {},
        "facF_cache": {},
        "hfs_int_cache": {},
        "label_map": None,
        "F_list_all": None,
    }


def _build_linear_hfs_structure(B, DJ=0.0, HJ=0.0, J_max=30, J_min=0):
    """
    Exact linear-rotor HFS support via the same direct-F machinery used for
    asymmetric tops, but projected onto the physical K=0 subspace.
    """
    global LAST_DF_HF, LAST_WAVEFUNC_CACHE
    J_min, J_max = _normalize_j_range(J_min, J_max)

    nuclei = _active_quadrupole_nuclei()
    F_max_raw = globals().get("F_MAX", F_MAX)
    F_max_loc = None
    if F_max_raw is not None:
        try:
            fv = float(F_max_raw)
            if fv > 0.0:
                F_max_loc = fv
        except Exception:
            F_max_loc = None
    J_build_max = _hfs_rotational_support_jmax(J_max, nuclei, F_max_loc)

    levels = {}
    eigs = {}
    eigvecs_k = {}
    rot_blocks_w = {}
    rot_eigvecs_w = {}
    wang_transforms = {}
    wang_basis_labels = {}

    wf_rot_levels = []
    wf_states = {}
    wf_hfs_levels = []

    for J in range(J_min, J_build_max + 1):
        JJ = J * (J + 1)
        E = float(B * JJ - DJ * (JJ ** 2) + HJ * (JJ ** 3))
        levels[J] = [(E, 0, 0, "ee")]

        W = np.zeros((2 * J + 1, 1), dtype=float)
        W[J, 0] = 1.0
        U_w = np.array([[1.0]], dtype=complex)
        U_k = np.zeros((2 * J + 1, 1), dtype=complex)
        U_k[J, 0] = 1.0

        eigs[J] = (np.array([E], dtype=float), U_w, W)
        eigvecs_k[J] = U_k
        rot_blocks_w[J] = np.array([[E]], dtype=complex)
        rot_eigvecs_w[J] = U_w
        wang_transforms[J] = W
        wang_basis_labels[J] = [(0, 0, "ee")]

        if J > J_max:
            continue

        Ja_i, Jb_i, Jc_i, _ = _rot_ops_rep(J, rep="Ir")
        v = U_k[:, 0]
        axis_values = {
            "a": _expectation(v, Ja_i @ Ja_i),
            "b": _expectation(v, Jb_i @ Jb_i),
            "c": _expectation(v, Jc_i @ Jc_i),
        }
        coord_expect = _wavefunc_coords_from_axis_values(axis_values, "Ir")
        k_values = list(range(-J, J + 1))
        key = f"{J}:0"
        state = {
            "key": key,
            "J": int(J),
            "alpha": 0,
            "K": 0,
            "Ka": 0,
            "Kc": 0,
            "species": "ee",
            "energy_mhz": E,
            "energy_cm": E / 29979.2458,
            "k_values": k_values,
            "coeff_re": [float(x) for x in np.real(v)],
            "coeff_im": [float(x) for x in np.imag(v)],
            "coord_expect": coord_expect,
        }
        wf_states[key] = state
        wf_rot_levels.append({
            "id": key,
            "kind": "rot",
            "J": int(J),
            "alpha": 0,
            "Ka": 0,
            "Kc": 0,
            "species": "ee",
            "energy_mhz": E,
            "energy_cm": E / 29979.2458,
            "label": f"J={int(J)} alpha=0 Ka=0 Kc=0",
            "parent_key": key,
            "purity": 1.0,
        })

    use_hfs = False
    hfF_blocks = hfF_labels = basis_by_F = groups_by_F = None
    if nuclei:
        hf_pack = build_and_diag_HF_fullF(
            levels,
            rot_blocks_w,
            rot_eigvecs_w,
            wang_transforms,
            wang_basis_labels,
            nuclei,
            F_max=F_max_loc,
            rotor_type="linear",
        )
        use_hfs = (hf_pack[0] is not None)
        if use_hfs:
            hfF_blocks, hfF_labels, basis_by_F, groups_by_F = hf_pack

    df_hf = make_hyperfine_levels_table(hfF_labels, to_cm1=True) if use_hfs else None
    LAST_DF_HF = df_hf

    if use_hfs:
        for F, recs in hfF_labels.items():
            for rec in recs:
                (
                    Fv, alpha_hf, E_hf_mhz,
                    J_dom, tau_dom, Ka, Kc, sp,
                    purity, Ka_w, Kc_w, *extra_rec
                ) = rec
                parent_key = f"{int(J_dom)}:{int(tau_dom)}"
                I12_dom = float(extra_rec[0]) if extra_rec else float("nan")
                wf_hfs_levels.append({
                    "id": f"F={float(Fv):g}:{int(alpha_hf)}",
                    "kind": "hfs",
                    "F": float(Fv),
                    "alpha_hf": int(alpha_hf),
                    "J": int(J_dom),
                    "alpha": int(tau_dom),
                    "Ka": int(Ka),
                    "Kc": int(Kc),
                    "species": str(sp),
                    "energy_mhz": float(E_hf_mhz),
                    "energy_cm": float(E_hf_mhz) / 29979.2458,
                    "Ka_w": float(Ka_w),
                    "Kc_w": float(Kc_w),
                    "I12_dom": I12_dom,
                    "purity": float(purity),
                    "parent_key": parent_key,
                    "label": (
                        f"J={int(J_dom)} F={float(Fv):g} alpha={int(alpha_hf)} "
                        f"Ka={int(Ka)} Kc={int(Kc)}"
                    ),
                })

    wavefunc_cache = {
        "meta": _make_wavefunc_cache_meta(
            "linear",
            "Ir",
            globals().get("REDUCTION", REDUCTION),
            "matrix",
            J_min,
            J_max
        ),
        "rot_levels": wf_rot_levels,
        "hfs_levels": wf_hfs_levels,
        "states": wf_states,
    }
    LAST_WAVEFUNC_CACHE = wavefunc_cache
    return {
        "engine": "matrix",
        "levels": levels,
        "eigs": eigs,
        "eigvecs_k": eigvecs_k,
        "rep_int": "Ir",
        "rep_user": "Ir",
        "J_min": int(J_min),
        "J_max": int(J_max),
        "use_hfs": bool(use_hfs),
        "hfF_blocks": hfF_blocks,
        "hfF_labels": hfF_labels,
        "basis_by_F": basis_by_F,
        "groups_by_F": groups_by_F,
        "nuclei": nuclei,
        "wavefunc_cache": wavefunc_cache,
        "df_hf": df_hf,
        "mu_tau_cache": {},
        "mu_wang_cache": {},
        "facF_cache": {},
        "hfs_int_cache": {},
        "label_map": None,
        "F_list_all": None,
    }


def _simulate_asymmetric_from_cache(
    structure, T, mu_a, mu_b, mu_c,
    intensity_cut=1e-9,
    groupSymmetry='C1',
    a_checked=True, b_checked=True, c_checked=True
):
    """
    Turn a cached rotational structure into a transition catalog.

    At this stage the expensive diagonalization is already done. The function
    focuses on physics layered on top of the eigenvectors: dipole selection
    rules, Boltzmann factors, optional quadrupole-HFS recoupling, intensity
    thresholds, and frequency-window filtering.
    """
    if not a_checked:
        mu_a = 0.0
    if not b_checked:
        mu_b = 0.0
    if not c_checked:
        mu_c = 0.0

    _restore_cached_outputs(structure)
    levels = structure["levels"]
    eigs = structure["eigs"]
    rep_int = structure["rep_int"]
    use_hfs = bool(structure["use_hfs"])
    hfF_blocks = structure["hfF_blocks"]
    hfF_labels = structure["hfF_labels"]
    basis_by_F = structure["basis_by_F"]
    nuclei = list(structure.get("nuclei", []))
    mu_cache = structure.setdefault("mu_tau_cache", {})
    mu_wang_cache = structure.setdefault("mu_wang_cache", {})
    J_values = _sorted_level_js(levels)
    if not J_values:
        return pd.DataFrame(columns=[
            'Frequency (MHz)', 'Intensity', 'Relative intensity', 'LGINT', 'logS', 'alpha L', 'alpha u',
            'Jl', 'Ka_l', 'Kc_l', 'sp_l',
            'Ju', 'Ka_u', 'Kc_u', 'sp_u', 'Branch'
        ])
    rotor_type_meta = str(
        structure.get("wavefunc_cache", {}).get("meta", {}).get("rotor_type", "asymmetric")
    ).strip().lower()
    use_freq_scaled_hfs_threshold = (rotor_type_meta == "asymmetric")
    J_visible_max = int(structure.get("J_max", max(J_values)))
    J_values_visible = [J for J in J_values if int(J) <= J_visible_max]
    Qrs = _compute_Qrs(levels, T)

    zero = 1.5e-38
    cmc = 29979.2458
    tmc = 1.43878

    qrot_override = globals().get("QROT_OVERRIDE", None)
    if qrot_override is not None:
        qrot = float(qrot_override)
    else:
        qrot = float(Qrs) if Qrs > 0 else 1.0
    if qrot <= 0.0:
        qrot = 1.0
    if qrot < 1.0:
        qrot = 1.0
    globals()["LAST_QROT_USED"] = float(qrot)
    globals()["LAST_QROT_SOURCE"] = "override" if qrot_override is not None else "auto"

    fac = 4.16231e-5 / qrot

    fqmin = float(globals().get("FREQ_MIN", 0.0))
    fqmax = float(globals().get("FREQ_MAX", 9999.99))
    freq_unit = str(globals().get("FREQ_UNIT", "auto")).lower()
    if freq_unit not in ("ghz", "mhz", "auto"):
        freq_unit = "auto"
    if freq_unit == "ghz" or (freq_unit == "auto" and fqmax > 0.0 and fqmax <= 1000.0):
        fqmax *= 1000.0
        if fqmin > 0.0:
            fqmin *= 1000.0
    if fqmin < 5e-5:
        fqmin = 5e-5
    if fqmax < fqmin:
        fqmax = fqmin

    thrsh = float(globals().get("STR0", -12.0))
    thrsh1 = float(globals().get("STR1", -10.0))
    thrsh1 -= 2.0 * math.log10(300000.0)

    starg = thrsh - math.log10(max(fqmax * fac, zero))
    scomp = -38.0
    strmn = pow(10.0, max(starg, scomp))

    tmq = -tmc / (T * cmc) if T > 0 else -1.0
    tmql = tmq * 0.43429448

    eeWt = float(globals().get("eeWt", 1.0))
    eoWt = float(globals().get("eoWt", 1.0))
    oeWt = float(globals().get("oeWt", 1.0))
    ooWt = float(globals().get("ooWt", 1.0))

    def _spin_weight(species, groupSymmetry, eeWt=1.0, eoWt=1.0, oeWt=1.0, ooWt=1.0):
        if species == 'ee': return eeWt
        if species == 'eo': return eoWt
        if species == 'oe': return oeWt
        if species == 'oo': return ooWt
        return 1.0

    def mu_tau(Jl, Ju, axis):
        key = (Jl, Ju, axis)
        if key not in mu_cache:
            mu_J = build_mu_matrix_rep(Jl, Ju, axis, rep=rep_int)
            UJl, Wl = eigs[Jl][1], eigs[Jl][2]
            UJu, Wu = eigs[Ju][1], eigs[Ju][2]
            if Wl is None:
                Wl = np.eye(mu_J.shape[0], dtype=float)
            if Wu is None:
                Wu = np.eye(mu_J.shape[1], dtype=float)
            mu_J = Wl.T @ mu_J @ Wu
            mu_cache[key] = UJl.conj().T @ mu_J @ UJu
        return mu_cache[key]

    def mu_wang(Jl, Ju, axis):
        key = (Jl, Ju, axis)
        if key not in mu_wang_cache:
            mu_J = build_mu_matrix_rep(Jl, Ju, axis, rep=rep_int)
            Wl = eigs[Jl][2]
            Wu = eigs[Ju][2]
            if Wl is None:
                Wl = np.eye(mu_J.shape[0], dtype=float)
            if Wu is None:
                Wu = np.eye(mu_J.shape[1], dtype=float)
            mu_wang_cache[key] = Wl.T @ mu_J @ Wu
        return mu_wang_cache[key]

    if not use_hfs:
        lines = []
        for J1 in J_values_visible:
            for J2 in (J1 - 1, J1, J1 + 1):
                if J2 not in levels:
                    continue
                if int(J2) > J_visible_max:
                    continue
                if J2 < J1:
                    continue

                mu_a_if = mu_b_if = mu_c_if = None
                if mu_a != 0.0:
                    mu_a_if = mu_tau(J1, J2, 'a')
                if mu_b != 0.0:
                    mu_b_if = mu_tau(J1, J2, 'b')
                if mu_c != 0.0:
                    mu_c_if = mu_tau(J1, J2, 'c')

                levels_J1 = levels[J1]
                levels_J2 = levels[J2]

                for i, (E1_i, Ka1_i, Kc1_i, sp1_i) in enumerate(levels_J1):
                    for f, (E2_f, Ka2_f, Kc2_f, sp2_f) in enumerate(levels_J2):
                        if J1 == J2 and f <= i:
                            continue

                        amp_sum = 0.0 + 0.0j
                        if mu_a_if is not None:
                            amp_sum += mu_a * mu_a_if[i, f]
                        if mu_b_if is not None:
                            amp_sum += mu_b * mu_b_if[i, f]
                        if mu_c_if is not None:
                            amp_sum += mu_c * mu_c_if[i, f]
                        if amp_sum == 0.0:
                            continue

                        strr = (amp_sum.real * amp_sum.real + amp_sum.imag * amp_sum.imag)
                        if strr <= 0.0:
                            continue

                        logS = math.log10(max(strr, 1e-300))

                        if E1_i <= E2_f:
                            E_i, J_i, Ka_l, Kc_l, sp_l = E1_i, J1, Ka1_i, Kc1_i, sp1_i
                            E_f, J_f, Ka_u, Kc_u, sp_u = E2_f, J2, Ka2_f, Kc2_f, sp2_f
                        else:
                            E_i, J_i, Ka_l, Kc_l, sp_l = E2_f, J2, Ka2_f, Kc2_f, sp2_f
                            E_f, J_f, Ka_u, Kc_u, sp_u = E1_i, J1, Ka1_i, Kc1_i, sp1_i

                        nu_MHz = E_f - E_i
                        if nu_MHz <= 1e-9:
                            continue
                        if nu_MHz < fqmin or nu_MHz > fqmax:
                            continue
                        if strr < strmn:
                            continue

                        dgn = _spin_weight(sp_l, groupSymmetry, eeWt, eoWt, oeWt, ooWt)
                        str_val = dgn * strr * fac * nu_MHz * (1.0 - math.exp(tmq * nu_MHz))
                        if str_val <= 0.0:
                            continue

                        LGINT = math.log10(str_val + zero) + tmql * E_i
                        if LGINT < thrsh:
                            continue

                        thrshf = thrsh1 + 2.0 * math.log10(nu_MHz + zero)
                        diff = thrsh - thrshf
                        if abs(diff) < 4.0:
                            thrshf = math.log10(pow(10.0, diff) + 1.0) + thrshf
                        if LGINT < thrshf:
                            continue

                        intensity = str_val * math.exp(tmq * E_i)

                        if J_f == J_i:
                            branch = 'Q'
                        elif J_f == J_i + 1:
                            branch = 'R'
                        elif J_f == J_i - 1:
                            branch = 'P'
                        else:
                            branch = ''

                        lines.append((
                            nu_MHz, intensity, LGINT, logS, i, f,
                            J_i, Ka_l, Kc_l, sp_l,
                            J_f, Ka_u, Kc_u, sp_u,
                            branch
                        ))

        if not lines:
            return pd.DataFrame(columns=[
                'Frequency (MHz)', 'Intensity', 'Relative intensity', 'LGINT', 'logS', 'alpha L', 'alpha u',
                'Jl', 'Ka_l', 'Kc_l', 'sp_l',
                'Ju', 'Ka_u', 'Kc_u', 'sp_u', 'Branch'
            ])

        Imax = max(l[1] for l in lines)
        if Imax <= 0:
            Imax = 1.0
        data = [
            (nu, I, I / Imax, LGINT, logS, alpha_l, alpha_u,
             Jl, Ka_l, Kc_l, sp_l,
             Ju, Ka_u, Kc_u, sp_u, branch)
            for (nu, I, LGINT, logS, alpha_l, alpha_u,
                 Jl, Ka_l, Kc_l, sp_l,
                 Ju, Ka_u, Kc_u, sp_u, branch) in lines
        ]
        cols = [
            'Frequency (MHz)', 'Intensity', 'Relative intensity', 'LGINT', 'logS', 'alpha L', 'alpha u',
            'Jl', 'Ka_l', 'Kc_l', 'sp_l',
            'Ju', 'Ka_u', 'Kc_u', 'sp_u', 'Branch'
        ]

        return (pd.DataFrame(data, columns=cols)
                  .sort_values('Frequency (MHz)')
                  .reset_index(drop=True))

    label_map = structure.get("label_map")
    if label_map is None:
        label_map = {}
        for F, lst in hfF_labels.items():
            d = {}
            for rec in lst:
                if len(rec) >= 12:
                    (Fv, alpha, E, J_dom, tau_dom, Ka, Kc, sp, purity, Ka_w, Kc_w, I12_dom) = rec[:12]
                else:
                    (Fv, alpha, E, J_dom, tau_dom, Ka, Kc, sp, purity, Ka_w, Kc_w) = rec
                    I12_dom = float("nan")
                d[int(alpha)] = (int(J_dom), int(Ka), int(Kc), sp, float(I12_dom))
            label_map[F] = d
        structure["label_map"] = label_map
    facF_cache = structure.setdefault("facF_cache", {})
    hfs_cache = structure.setdefault("hfs_int_cache", {})
    F_list_all = structure.get("F_list_all")
    if F_list_all is None:
        F_list_all = sorted(hfF_blocks.keys())
        structure["F_list_all"] = F_list_all

    def facF_hfs(Jl, Fl, Ju, Fu, I12):
        key = (Jl, Fl, Ju, Fu, I12)
        v = facF_cache.get(key)
        if v is None:
            v = e1_recoupling_amp(Jl, Fl, Ju, Fu, I12)
            facF_cache[key] = v
        return v

    def hfs_int(Jl, Fl, Ju, Fu):
        key = (Jl, Fl, Ju, Fu, len(nuclei))
        v = hfs_cache.get(key)
        if v is None:
            if len(nuclei) == 1:
                v = hyperfine_factor_intensity(Jl, Fl, Ju, Fu, float(nuclei[0]["I"]))
            else:
                v = 1.0
            hfs_cache[key] = v
        return v

    lines = []
    for Fl in F_list_all:
        Ew_l, C_l = hfF_blocks[Fl]
        basis_l = basis_by_F[Fl]
        n_l = Ew_l.size

        Jdom_l = np.array([label_map[Fl].get(a, (0, 0, 0, 'ee', np.nan))[0] for a in range(n_l)], dtype=int)
        Ka_l_d = np.array([label_map[Fl].get(a, (0, 0, 0, 'ee', np.nan))[1] for a in range(n_l)], dtype=int)
        Kc_l_d = np.array([label_map[Fl].get(a, (0, 0, 0, 'ee', np.nan))[2] for a in range(n_l)], dtype=int)
        sp_l_d = [label_map[Fl].get(a, (0, 0, 0, 'ee', np.nan))[3] for a in range(n_l)]
        I12_l_d = np.array([label_map[Fl].get(a, (0, 0, 0, 'ee', np.nan))[4] for a in range(n_l)], dtype=float)
        for Fu in F_list_all:
            if abs(Fu - Fl) > 1.0:
                continue
            if (Fl == 0.0 and Fu == 0.0):
                continue

            Ew_u, C_u = hfF_blocks[Fu]
            basis_u = basis_by_F[Fu]
            n_u = Ew_u.size

            Jdom_u = np.array([label_map[Fu].get(a, (0, 0, 0, 'ee', np.nan))[0] for a in range(n_u)], dtype=int)
            Ka_u_d = np.array([label_map[Fu].get(a, (0, 0, 0, 'ee', np.nan))[1] for a in range(n_u)], dtype=int)
            Kc_u_d = np.array([label_map[Fu].get(a, (0, 0, 0, 'ee', np.nan))[2] for a in range(n_u)], dtype=int)
            sp_u_d = [label_map[Fu].get(a, (0, 0, 0, 'ee', np.nan))[3] for a in range(n_u)]
            I12_u_d = np.array([label_map[Fu].get(a, (0, 0, 0, 'ee', np.nan))[4] for a in range(n_u)], dtype=float)
            M = np.zeros((n_l, n_u), dtype=complex)
            for i, (Jl, w_l, I12_l, *_restl) in enumerate(basis_l):
                for j, (Ju, w_u, I12_u, *_restu) in enumerate(basis_u):
                    if abs(Ju - Jl) > 1:
                        continue
                    if abs(I12_u - I12_l) > 1e-9:
                        continue

                    f = facF_hfs(Jl, Fl, Ju, Fu, I12_l)
                    if abs(f) < 1e-14:
                        continue

                    z = 0.0 + 0.0j
                    if mu_a != 0.0:
                        z += mu_a * mu_wang(Jl, Ju, 'a')[w_l, w_u]
                    if mu_b != 0.0:
                        z += mu_b * mu_wang(Jl, Ju, 'b')[w_l, w_u]
                    if mu_c != 0.0:
                        z += mu_c * mu_wang(Jl, Ju, 'c')[w_l, w_u]

                    if z != 0.0:
                        M[i, j] = f * z

            Aamp = C_l.conjugate().T @ M @ C_u

            for a1 in range(n_l):
                E1 = float(Ew_l[a1])
                a2_start = (a1 + 1) if (Fl == Fu) else 0

                for a2 in range(a2_start, n_u):
                    E2 = float(Ew_u[a2])
                    if abs(E2 - E1) <= 1e-9:
                        continue

                    amp = Aamp[a1, a2]
                    S_rot = (amp.real * amp.real + amp.imag * amp.imag)
                    if S_rot <= 0.0:
                        continue

                    hfs_fac = hfs_int(int(Jdom_l[a1]), Fl, int(Jdom_u[a2]), Fu)
                    if hfs_fac <= 0.0:
                        continue

                    S_tot = S_rot * hfs_fac
                    if E1 <= E2:
                        E_i, F_i = E1, Fl
                        E_f, F_f = E2, Fu
                        J_l, J_u = int(Jdom_l[a1]), int(Jdom_u[a2])
                        I12_l_vis, I12_u_vis = float(I12_l_d[a1]), float(I12_u_d[a2])
                        Ka_l, Kc_l, sp_l = int(Ka_l_d[a1]), int(Kc_l_d[a1]), sp_l_d[a1]
                        Ka_u, Kc_u, sp_u = int(Ka_u_d[a2]), int(Kc_u_d[a2]), sp_u_d[a2]
                    else:
                        E_i, F_i = E2, Fu
                        E_f, F_f = E1, Fl
                        J_l, J_u = int(Jdom_u[a2]), int(Jdom_l[a1])
                        I12_l_vis, I12_u_vis = float(I12_u_d[a2]), float(I12_l_d[a1])
                        Ka_l, Kc_l, sp_l = int(Ka_u_d[a2]), int(Kc_u_d[a2]), sp_u_d[a2]
                        Ka_u, Kc_u, sp_u = int(Ka_l_d[a1]), int(Kc_l_d[a1]), sp_l_d[a1]

                    nu_MHz = E_f - E_i
                    if nu_MHz <= 1e-9:
                        continue
                    if nu_MHz < fqmin or nu_MHz > fqmax:
                        continue

                    dgn = _spin_weight(sp_l, groupSymmetry, eeWt, eoWt, oeWt, ooWt)
                    str_val = dgn * S_tot * fac * nu_MHz * (1.0 - math.exp(tmq * nu_MHz))
                    if str_val <= 0.0:
                        continue

                    LGINT = math.log10(str_val + zero) + tmql * E_i
                    if LGINT < thrsh:
                        continue
                    if use_freq_scaled_hfs_threshold:
                        thrshf = thrsh1 + 2.0 * math.log10(nu_MHz + zero)
                        diff = thrsh - thrshf
                        if abs(diff) < 4.0:
                            thrshf = math.log10(pow(10.0, diff) + 1.0) + thrshf
                        if LGINT < thrshf:
                            continue

                    intensity = str_val * math.exp(tmq * E_i)
                    logS = math.log10(max(S_tot, 1e-300))

                    if J_u == J_l:
                        branch = 'Q'
                    elif J_u == J_l + 1:
                        branch = 'R'
                    elif J_u == J_l - 1:
                        branch = 'P'
                    else:
                        branch = ''

                    nu_cat = float(f"{nu_MHz:.4f}")
                    lower_state_id = (float(F_i), int(a1)) if E1 <= E2 else (float(F_i), int(a2))
                    upper_state_id = (float(F_f), int(a2)) if E1 <= E2 else (float(F_f), int(a1))
                    lines.append((
                        nu_MHz, nu_cat, intensity, LGINT, logS,
                        J_l, float(F_i), float(I12_l_vis), Ka_l, Kc_l, sp_l,
                        J_u, float(F_f), float(I12_u_vis), Ka_u, Kc_u, sp_u,
                        branch, lower_state_id, upper_state_id
                    ))

    if not lines:
        return pd.DataFrame(columns=[
            'Frequency (MHz)', 'Intensity', 'Relative intensity', 'LGINT', 'logS',
            'Jl', 'Fl', 'I12_l', 'Ka_l', 'Kc_l', 'sp_l',
            'Ju', 'Fu', 'I12_u', 'Ka_u', 'Kc_u', 'sp_u', 'Branch'
        ])

    lines_sorted = sorted(lines, key=lambda x: (x[1],) + x[5:17] + x[18:20])

    def same_transition_hfs(l1, l2):
        if l1[5:17] != l2[5:17]:
            return False
        return l1[18:20] == l2[18:20]

    grouped = []
    current = [lines_sorted[0]]
    for line in lines_sorted[1:]:
        if same_transition_hfs(line, current[0]) and (line[1] == current[0][1]):
            current.append(line)
        else:
            grouped.append(current)
            current = [line]
    grouped.append(current)

    merged = []
    for group in grouped:
        nu_ref = group[0][1]
        I_sum = sum(g[2] for g in group)
        LGINT_merged = math.log10(max(I_sum, 1e-300))
        g_max = max(group, key=lambda g: g[2])
        (_, _, _, _, logS,
         Jl, Fl, I12_l, Ka_l, Kc_l, sp_l,
         Ju, Fu, I12_u, Ka_u, Kc_u, sp_u,
         branch, _lower_state_id, _upper_state_id) = g_max
        merged.append((
            nu_ref, I_sum, LGINT_merged, logS,
            Jl, Fl, I12_l, Ka_l, Kc_l, sp_l,
            Ju, Fu, I12_u, Ka_u, Kc_u, sp_u,
            branch
        ))

    Imax = max(m[1] for m in merged)
    if Imax <= 0.0:
        Imax = 1.0

    data = [
        (nu, I, I / Imax, LGINT, logS,
         Jl, Fl, I12_l, Ka_l, Kc_l, sp_l,
         Ju, Fu, I12_u, Ka_u, Kc_u, sp_u, branch)
        for (nu, I, LGINT, logS,
             Jl, Fl, I12_l, Ka_l, Kc_l, sp_l,
             Ju, Fu, I12_u, Ka_u, Kc_u, sp_u, branch) in merged
    ]
    cols = [
        'Frequency (MHz)', 'Intensity', 'Relative intensity', 'LGINT', 'logS',
        'Jl', 'Fl', 'I12_l', 'Ka_l', 'Kc_l', 'sp_l',
        'Ju', 'Fu', 'I12_u', 'Ka_u', 'Kc_u', 'sp_u', 'Branch'
    ]

    return (pd.DataFrame(data, columns=cols)
              .sort_values('Frequency (MHz)')
              .reset_index(drop=True))


def _simulate_asymmetric(
    T, A, B, C,
    DJ=0, DJK=0, DK=0, dJ=0, dK=0,
    HJ=0, HJK=0, HKJ=0, HK=0, h1=0, h2=0, h3=0,
    mu_a=1.0, mu_b=0.0, mu_c=0.0,
    J_max=30, intensity_cut=1e-9,
    groupSymmetry='C1',
    a_checked=True, b_checked=True, c_checked=True,
    wang_symmetry=True, J_min=0
):
    """
    One-shot wrapper around the cached asymmetric/matrix simulator.

    It builds the full asymmetric structure for the requested constants and
    immediately evaluates the transition catalog, returning the resulting
    pandas DataFrame.
    """
    globals()["LAST_SIM_REUSED"] = False
    structure = _build_asymmetric_structure(
        A, B, C,
        DJ, DJK, DK, dJ, dK,
        HJ, HJK, HKJ, HK, h1, h2, h3,
        J_max, wang_symmetry=wang_symmetry,
        rotor_type=globals().get("rotorType", "asymmetric"),
        J_min=J_min
    )
    return _simulate_asymmetric_from_cache(
        structure, T, mu_a, mu_b, mu_c,
        intensity_cut=intensity_cut,
        groupSymmetry=groupSymmetry,
        a_checked=a_checked, b_checked=b_checked, c_checked=c_checked
    )

def simulate_rigid_spectrum(
    T, A, B, C,
    DJ=0, DJK=0, DK=0, dJ=0, dK=0,
    HJ=0, HJK=0, HKJ=0, HK=0, h1=0, h2=0, h3=0,
    mu_a=1.0, mu_b=0.0, mu_c=0.0,
    J_max=30, intensity_cut=1e-9,
    groupSymmetry='C1',
    rotorType='asymmetric',
    a_checked=True, b_checked=True, c_checked=True,
    wang_symmetry=True, J_min=0
):
    """
    Public entry point used by the browser UI and the fitting stack.

    The dispatcher validates the chosen rotor conventions, decides whether a
    simple analytic shortcut is still safe, applies the SPCAT oblate remap
    when needed, reuses the latest compatible cache, and finally returns the
    predicted transition list as a DataFrame.
    """
    globals()["LAST_SIM_REUSED"] = False
    J_min, J_max = _normalize_j_range(J_min, J_max)
    # Guard-rail: keep conventions consistent with spectroscopy inputs.
    # Strict ordering is required only for the asymmetric rotor case.
    eps = 1e-9
    try:
        A_f, B_f, C_f = float(A), float(B), float(C)
    except Exception:
        A_f = B_f = C_f = float("nan")
    if rotorType == "asymmetric":
        if not (A_f > B_f + eps and B_f > C_f + eps):
            raise ValueError(f"Asymmetric rotor requires A > B > C (MHz). Got A={A}, B={B}, C={C}.")
    else:
        if not (A_f + eps >= B_f and B_f + eps >= C_f):
            raise ValueError(f"Rotational constants must be ordered as A >= B >= C (MHz). Got A={A}, B={B}, C={C}.")

    if not a_checked: mu_a = 0.0
    if not b_checked: mu_b = 0.0
    if not c_checked: mu_c = 0.0

    # Decide if the closed-form symmetric branch is valid:
    # only diagonal symmetric terms and no active quadrupole-HFS.
    # Any off-diagonal Watson terms (dJ/dK/h1/h2/h3) or active quadrupole
    # requires full matrix diagonalization to match SPCAT.
    offdiag_terms = (dJ, dK, h1, h2, h3)
    has_offdiag = any(abs(float(v)) > 1e-15 for v in offdiag_terms)
    has_quadrupole = _has_active_quadrupole()
    use_simple_sym = (
        rotorType in ('prolate', 'oblate', 'spherical')
        and (not has_offdiag)
        and (not has_quadrupole)
    )

    # Always matrix for asymmetric. Symmetric tops switch to matrix whenever
    # off-diagonal Watson terms or quadrupole-HFS are active. Linear rotors
    # keep the dedicated closed form unless quadrupole-HFS is active, in which
    # case we project the exact direct-F Hamiltonian onto K=0.
    use_matrix_branch = (
        rotorType == 'asymmetric'
        or (rotorType in ('prolate', 'oblate', 'spherical') and not use_simple_sym)
        or (rotorType == 'linear' and has_quadrupole)
    )

    if use_matrix_branch:
        A_eff, B_eff, C_eff = A_f, B_f, C_f
        mu_a_eff, mu_b_eff, mu_c_eff = mu_a, mu_b, mu_c
        a_eff, b_eff, c_eff = a_checked, b_checked, c_checked
        quad_snap = None
        if rotorType == 'linear':
            A_eff = B_eff
            C_eff = B_eff
        elif rotorType == 'prolate':
            C_eff = B_eff
        elif rotorType == 'oblate':
            A_eff = B_eff
            A_eff, B_eff, C_eff, mu_a_eff, mu_b_eff, mu_c_eff = _spcat_oblate_remap_abc(
                A_eff, B_eff, C_eff, mu_a_eff, mu_b_eff, mu_c_eff
            )
            a_eff, b_eff, c_eff = c_checked, b_checked, a_checked
            quad_snap = _apply_oblate_quadrupole_remap()
        elif rotorType == 'spherical':
            A_eff = B_eff
            C_eff = B_eff

        try:
            globals()["LAST_SIM_ENGINE"] = "matrix"
            signature = _matrix_cache_signature(
                rotorType,
                A_eff, B_eff, C_eff,
                DJ, DJK, DK, dJ, dK,
                HJ, HJK, HKJ, HK, h1, h2, h3,
                J_min, J_max, wang_symmetry
            )
            cache = _match_last_sim_cache(signature)
            if cache is None:
                if rotorType == 'linear':
                    cache = _build_linear_hfs_structure(
                        B_eff,
                        DJ=DJ,
                        HJ=HJ,
                        J_max=J_max,
                        J_min=J_min,
                    )
                else:
                    cache = _build_asymmetric_structure(
                        A_eff, B_eff, C_eff,
                        DJ, DJK, DK, dJ, dK,
                        HJ, HJK, HKJ, HK, h1, h2, h3,
                        J_max, wang_symmetry=wang_symmetry,
                        rotor_type=rotorType,
                        J_min=J_min
                    )
                _store_last_sim_cache(cache, signature)
            else:
                globals()["LAST_SIM_REUSED"] = True
            return _simulate_asymmetric_from_cache(
                cache, T, mu_a_eff, mu_b_eff, mu_c_eff,
                intensity_cut=intensity_cut,
                groupSymmetry=groupSymmetry,
                a_checked=a_eff, b_checked=b_eff, c_checked=c_eff
            )
        finally:
            if quad_snap is not None:
                _restore_quadrupole_globals(quad_snap)
    globals()["LAST_SIM_ENGINE"] = "simple"
    rep_simple = _normalize_rep_key(globals().get("REPRESENTATION", REPRESENTATION))
    signature = _simple_cache_signature(
        rotorType,
        A_f, B_f, C_f,
        DJ, DJK, DK, dJ, dK,
        HJ, HJK, HKJ, HK, h1, h2, h3,
        J_min, J_max, rep_simple
    )
    cache = _match_last_sim_cache(signature)
    if cache is None:
        cache = _build_simple_structure(
            rotorType, A_f, B_f, C_f,
            DJ, DJK, DK, dJ, dK,
            HJ, HJK, HKJ, HK, h1, h2, h3,
            J_max, rep=rep_simple, J_min=J_min
        )
        _store_last_sim_cache(cache, signature)
    else:
        globals()["LAST_SIM_REUSED"] = True
    return _simulate_simple_from_cache(cache, T, mu_a, mu_b, mu_c, intensity_cut)


def _normalized_freq_window_bounds(freq_min=None, freq_max=None, freq_unit=None):
    """
    Normalize the frequency window to MHz using the same rules as the simulator.

    Inputs may be explicit values or ``None`` to reuse the current globals.
    Output is ``(freq_min_mhz, freq_max_mhz, normalized_unit_token)``.
    """
    raw_min = globals().get("FREQ_MIN", 0.0) if freq_min is None else freq_min
    raw_max = globals().get("FREQ_MAX", 9999.99) if freq_max is None else freq_max
    raw_unit = globals().get("FREQ_UNIT", "auto") if freq_unit is None else freq_unit
    fqmin = float(raw_min)
    fqmax = float(raw_max)
    unit = str(raw_unit).lower()
    if unit not in ("ghz", "mhz", "auto"):
        unit = "auto"
    if unit == "ghz" or (unit == "auto" and fqmax > 0.0 and fqmax <= 1000.0):
        fqmax *= 1000.0
        if fqmin > 0.0:
            fqmin *= 1000.0
    if fqmin < 5e-5:
        fqmin = 5e-5
    if fqmax < fqmin:
        fqmax = fqmin
    return fqmin, fqmax, unit


def _window_overlaps(lo_a, hi_a, lo_b, hi_b):
    """Return True when two closed frequency intervals overlap."""
    return not (hi_a < lo_b or lo_a > hi_b)


def _estimate_j_range_for_rotor_analytic(rotor_kind, A, B, C, fqmin, fqmax, j_cap):
    """
    Estimate the J range for one rigid analytic rotor model inside a frequency window.

    Inputs are rotor kind, rotational constants, normalized frequency bounds,
    and a scan cap. Output is a dictionary containing hit counts, estimated
    ``J_min/J_max``, and whether the scan touched the upper boundary.
    """
    rotor = str(rotor_kind).strip().lower()
    if rotor not in ("linear", "spherical", "prolate", "oblate"):
        raise ValueError(f"Unsupported analytic rotor kind: {rotor_kind}")

    gamma = 0.0
    if rotor == "prolate":
        gamma = float(A) - float(B)
    elif rotor == "oblate":
        gamma = float(C) - float(B)

    j_min = None
    j_max = None
    hit_count = 0
    boundary_hit = False

    def _register_hit(jl, ju, j_scan):
        nonlocal j_min, j_max, boundary_hit
        jl_i = int(max(0, jl))
        ju_i = int(max(jl_i, ju))
        if ju_i > j_cap:
            boundary_hit = True
            ju_i = int(j_cap)
        if j_scan >= j_cap:
            boundary_hit = True
        if j_min is None:
            j_min = jl_i
            j_max = ju_i
            return
        j_min = min(j_min, jl_i)
        j_max = max(j_max, ju_i)

    for J in range(0, int(j_cap) + 1):
        # Parallel branch (Delta J = +1, Delta K = 0): nu = 2 B (J+1)
        nu_par = abs(2.0 * float(B) * (J + 1))
        if fqmin <= nu_par <= fqmax:
            hit_count += 1
            _register_hit(J, J + 1, J)

        if rotor in ("prolate", "oblate"):
            # K-extremes envelope for Delta J = +1 and Delta K = +/-1:
            # evaluate only edge terms (K ~ +/-J), as requested for a fast
            # rigid-top estimate based on analytic bounds.
            base = 2.0 * float(B) * (J + 1)
            edge_1 = abs(base + gamma * (-2.0 * J + 1.0))
            edge_2 = abs(base + gamma * (2.0 * J + 1.0))
            env_min = min(edge_1, edge_2)
            env_max = max(edge_1, edge_2)
            if _window_overlaps(env_min, env_max, fqmin, fqmax):
                hit_count += 1
                _register_hit(J, J + 1, J)

    return {
        "rotor": rotor,
        "hits": int(hit_count),
        "J_min": None if j_min is None else int(j_min),
        "J_max": None if j_max is None else int(j_max),
        "boundary_hit": bool(boundary_hit),
    }


def estimate_j_range_from_frequency_window(
    T, A, B, C,
    DJ=0, DJK=0, DK=0, dJ=0, dK=0,
    HJ=0, HJK=0, HKJ=0, HK=0, h1=0, h2=0, h3=0,
    mu_a=1.0, mu_b=0.0, mu_c=0.0,
    rotorType='asymmetric',
    groupSymmetry='C1',
    a_checked=True, b_checked=True, c_checked=True,
    scan_J_start=None, scan_J_cap=1200, scan_growth=2,
    freq_min=None, freq_max=None, freq_unit=None
):
    """
    Estimate a safe J interval before running the final simulation.

    For asymmetric rotors the code merges prolate and oblate symmetric-top
    envelopes, then expands the range conservatively. The goal is to keep the
    final matrix calculation fast without clipping transitions in the selected
    frequency window.
    """
    approx_rotors = ['prolate', 'oblate'] if str(rotorType) == 'asymmetric' else [str(rotorType)]
    supported = {'linear', 'prolate', 'oblate', 'spherical', 'asymmetric'}
    if any(rt not in supported for rt in approx_rotors):
        raise ValueError(f"Unsupported rotor type for J-range estimate: {rotorType}")

    try:
        j_cap = max(1, int(scan_J_cap))
    except Exception as exc:
        raise ValueError(f"scan_J_cap must be an integer. Got {scan_J_cap}.") from exc

    snapshot_keys = [
        "LAST_SIM_CACHE",
        "LAST_DF_HF",
        "LAST_WAVEFUNC_CACHE",
        "LAST_SIM_ENGINE",
        "LAST_SIM_REUSED",
        "LAST_QROT_USED",
        "LAST_QROT_SOURCE",
        "FREQ_MIN",
        "FREQ_MAX",
        "FREQ_UNIT",
        "groupSymmetry",
    ]
    snapshot = {key: globals().get(key, None) for key in snapshot_keys}

    try:
        globals()["groupSymmetry"] = groupSymmetry
        if freq_min is not None:
            globals()["FREQ_MIN"] = float(freq_min)
        if freq_max is not None:
            globals()["FREQ_MAX"] = float(freq_max)
        if freq_unit is not None:
            globals()["FREQ_UNIT"] = str(freq_unit)

        fqmin, fqmax, unit = _normalized_freq_window_bounds(freq_min, freq_max, freq_unit)
        if fqmax <= 0.0:
            raise ValueError("Selected frequency window must have FREQ_MAX > 0.")

        details = []
        ranges = []
        boundary_hit = False
        for approx_rotor in approx_rotors:
            d = _estimate_j_range_for_rotor_analytic(
                approx_rotor,
                float(A), float(B), float(C),
                float(fqmin), float(fqmax),
                int(j_cap),
            )
            details.append({
                "rotor": d["rotor"],
                "hits": int(d["hits"]),
                "J_min": d["J_min"],
                "J_max": d["J_max"],
            })
            if d["J_min"] is not None and d["J_max"] is not None:
                ranges.append((int(d["J_min"]), int(d["J_max"])))
            boundary_hit = boundary_hit or bool(d["boundary_hit"])

        if not ranges:
            raise ValueError(
                f"No rigid symmetric-top transitions found in the selected window {fqmin:.6f}-{fqmax:.6f} MHz."
            )

        est_min = min(r[0] for r in ranges)
        est_max = max(r[1] for r in ranges)
        return {
            "J_min": int(est_min),
            "J_max": int(est_max),
            "scan_J_max": int(j_cap),
            "truncated": bool(boundary_hit),
            "rotor_types": approx_rotors,
            "details": details,
            "window_mhz": {
                "min": float(fqmin),
                "max": float(fqmax),
            },
            "freq_unit": unit,
            "method": "rigid_analytic_no_distortion_no_hfs",
        }
    finally:
        for key, value in snapshot.items():
            globals()[key] = value
