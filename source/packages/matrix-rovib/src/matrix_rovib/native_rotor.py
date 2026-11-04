"""Native NumPy rotational Hamiltonian used by MATRIX and TRINITY.

This module replaces the mutable-global and DataFrame boundary of the historical
browser rotor engine. It keeps the Pickett-compatible operator convention
needed by existing MATRIX inputs while exposing reusable matrices for coupled
large-amplitude calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Sequence

import numpy as np


MHZ_TO_KELVIN = 6.62607015e-34 * 1.0e6 / 1.380649e-23
_REPRESENTATIONS = {"ir", "iir", "iiir", "il", "iil", "iiil"}


@dataclass(frozen=True)
class NativeRotorParameters:
    A_MHz: float
    B_MHz: float
    C_MHz: float
    quartic_MHz: tuple[float, float, float, float, float] = (0.0,) * 5
    sextic_MHz: tuple[float, float, float, float, float, float, float] = (0.0,) * 7
    reduction: str = "S"
    representation: str = "Ir"
    octic_MHz: tuple[float, float, float, float, float, float, float, float, float] = (0.0,) * 9

    def __post_init__(self) -> None:
        values = np.asarray((self.A_MHz, self.B_MHz, self.C_MHz), dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("positive finite rotational constants are required")
        if (len(self.quartic_MHz) != 5 or len(self.sextic_MHz) != 7
                or len(self.octic_MHz) != 9):
            raise ValueError("five quartic, seven sextic, and nine octic constants are required")
        if not np.all(np.isfinite(self.quartic_MHz + self.sextic_MHz + self.octic_MHz)):
            raise ValueError("distortion constants must be finite")
        if self.reduction.upper() not in {"A", "S"}:
            raise ValueError("Watson reduction must be A or S")
        normalize_representation(self.representation)


@dataclass(frozen=True)
class NativeRotorBlock:
    J: int
    energies_MHz: np.ndarray
    vectors: np.ndarray
    assignments: tuple[tuple[int, int, str], ...]


def native_rotor_parameters_from_section(
    rotational: object,
    *,
    reduction: str | None = None,
    representation: str | None = None,
) -> NativeRotorParameters:
    """Promote a normalized rotational section to native LOCK parameters."""
    abc = tuple(getattr(rotational, name, None) for name in ("A_MHz", "B_MHz", "C_MHz"))
    if any(value is None for value in abc):
        raise ValueError("the rotational section must define A_MHz, B_MHz and C_MHz")
    quartic = tuple(float(value) for value in getattr(rotational, "quartic_distortion_MHz", ()))
    sextic = tuple(float(value) for value in getattr(rotational, "sextic_distortion_MHz", ()))
    octic = tuple(float(value) for value in getattr(rotational, "octic_distortion_MHz", ()))
    return NativeRotorParameters(
        A_MHz=float(abc[0]),
        B_MHz=float(abc[1]),
        C_MHz=float(abc[2]),
        quartic_MHz=quartic or (0.0,) * 5,
        sextic_MHz=sextic or (0.0,) * 7,
        octic_MHz=octic or (0.0,) * 9,
        reduction=(reduction or getattr(rotational, "watson_reduction", None) or "S"),
        representation=(representation or getattr(rotational, "representation", None) or "Ir"),
    )


def normalize_representation(value: str) -> str:
    token = str(value).strip().replace(" ", "")
    if token.lower() not in _REPRESENTATIONS:
        raise ValueError("representation must be Ir, IIr, IIIr, Il, IIl or IIIl")
    roman = token[:-1].upper().replace("III", "III").replace("II", "II").replace("I", "I")
    return roman + token[-1].lower()


@lru_cache(maxsize=128)
def angular_momentum_operators(
    J: int, representation: str = "Ir"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return body-fixed ``Ja,Jb,Jc`` matrices in the selected Pickett family."""
    if int(J) != J or J < 0:
        raise ValueError("J must be a nonnegative integer")
    J = int(J)
    rep = normalize_representation(representation)
    K = np.arange(-J, J + 1, dtype=float)
    raising = np.zeros((len(K), len(K)), dtype=complex)
    for index, value in enumerate(K[:-1]):
        raising[index + 1, index] = math.sqrt(J * (J + 1) - value * (value + 1))
    lowering = raising.T
    Jz = np.diag(K).astype(complex)
    Jx = 0.5 * (raising + lowering)
    # Body-fixed components obey the reverse angular-momentum commutator.
    Jy = (lowering - raising) / (2.0j)
    axes = ("c", "b", "a") if rep.endswith("l") else ("a", "b", "c")
    by_axis = {axes[0]: Jz, axes[1]: Jx, axes[2]: Jy}
    for value in by_axis.values():
        value.setflags(write=False)
    return by_axis["a"], by_axis["b"], by_axis["c"]


@lru_cache(maxsize=512)
def rotational_hamiltonian(parameters: NativeRotorParameters, J: int) -> np.ndarray:
    """Build a Watson A/S rotational block without mutable engine state."""
    Ja, Jb, Jc = angular_momentum_operators(J, parameters.representation)
    Ja2, Jb2, Jc2 = Ja @ Ja, Jb @ Jb, Jc @ Jc
    A, B, C = parameters.A_MHz, parameters.B_MHz, parameters.C_MHz
    H = A * Ja2 + B * Jb2 + C * Jc2
    distortion = centrifugal_distortion_operators(
        J, parameters.reduction, parameters.representation
    )
    for coefficient, operator in zip(
        parameters.quartic_MHz + parameters.sextic_MHz + parameters.octic_MHz,
        distortion, strict=True,
    ):
        H += coefficient * operator
    result = _hermitian(H)
    result.setflags(write=False)
    return result


def rotational_energy_levels(parameters: NativeRotorParameters, J: int) -> np.ndarray:
    """Exact energy-only rotor spectrum, using rigid K-parity tridiagonals.

    Distorted tops retain the resident full Watson Hamiltonian. A rigid top
    needs neither its eigenvectors nor high-order centrifugal operators.
    Eigenvalues are independent of the equivalent body-axis representation.
    """
    if isinstance(J, (bool, np.bool_)) or int(J) != J or J < 0:
        raise ValueError('J must be a nonnegative integer')
    J = int(J)
    if any(parameters.quartic_MHz + parameters.sextic_MHz + parameters.octic_MHz):
        return np.linalg.eigvalsh(rotational_hamiltonian(parameters, J)).real
    from scipy.linalg import eigh_tridiagonal

    A, B, C = parameters.A_MHz, parameters.B_MHz, parameters.C_MHz
    jj = J * (J + 1)
    blocks = []
    for parity in (0, 1):
        k = np.arange(-J + parity, J + 1, 2, dtype=float)
        if not len(k):
            continue
        diagonal = A * k * k + .5 * (B + C) * (jj - k * k)
        if len(k) == 1:
            blocks.append(diagonal)
            continue
        low = k[:-1]
        off = .25 * (B - C) * np.sqrt((jj - low * (low + 1)) *
                                     (jj - (low + 1) * (low + 2)))
        blocks.append(eigh_tridiagonal(diagonal, off, eigvals_only=True))
    return np.sort(np.concatenate(blocks))


@lru_cache(maxsize=512)
def centrifugal_distortion_operators(
    J: int, reduction: str = "S", representation: str = "Ir"
) -> tuple[np.ndarray, ...]:
    """Return the 5 quartic, 7 sextic, and 9 octic Watson operators."""
    reduction = str(reduction).upper()
    if reduction not in {"A", "S"}:
        raise ValueError("Watson reduction must be A or S")
    rep = normalize_representation(representation)
    Ja, Jb, Jc = angular_momentum_operators(J, rep)
    return _centrifugal_distortion_algebra((Ja,Jb,Jc), reduction, rep)


def _centrifugal_distortion_algebra(angular_operators, reduction, rep, *,
                                    power=np.linalg.matrix_power, freeze=True):
    """Shared Watson algebra for numerical matrices and ordered words."""
    Ja,Jb,Jc = angular_operators
    J2 = Ja @ Ja + Jb @ Jb + Jc @ Jc
    z_axis, x_axis, y_axis = ("c", "b", "a") if rep.endswith("l") else ("a", "b", "c")
    angular = {"a": Ja, "b": Jb, "c": Jc}
    K2 = angular[z_axis] @ angular[z_axis]
    Jx, Jy = angular[x_axis], angular[y_axis]
    asymmetry = Jy @ Jy - Jx @ Jx
    fourth = 0.5 * (
        power(Jx + 1.0j * Jy, 4) + power(Jx - 1.0j * Jy, 4)
    )
    sixth = -0.5 * (
        power(Jx + 1.0j * Jy, 6) + power(Jx - 1.0j * Jy, 6)
    )
    eighth = 0.5 * (
        power(Jx + 1.0j * Jy, 8) + power(Jx - 1.0j * Jy, 8)
    )
    quartic = (
        J2 @ J2,
        J2 @ K2,
        K2 @ K2,
        -2.0 * (J2 @ asymmetry),
        2.0 * fourth if reduction == "S" else -2.0 * _symmetrize(K2, asymmetry),
    )
    if reduction == "S":
        lower_order_sextic = (
            2.0 * (J2 @ fourth),
            2.0 * _symmetrize(K2, fourth),
            -2.0 * sixth,
        )
    else:
        lower_order_sextic = (
            -2.0 * (J2 @ J2 @ asymmetry),
            -2.0 * (J2 @ _symmetrize(K2, asymmetry)),
            -2.0 * _symmetrize(K2 @ K2, asymmetry),
        )
    sextic = (
        power(J2, 3),
        J2 @ J2 @ K2,
        J2 @ K2 @ K2,
        power(K2, 3),
        *lower_order_sextic,
    )
    if reduction == "S":
        lower_order_octic = (
            2.0 * (J2 @ J2 @ fourth),
            2.0 * (J2 @ _symmetrize(K2, fourth)),
            2.0 * _symmetrize(K2 @ K2, fourth),
            2.0 * eighth,
        )
    else:
        lower_order_octic = (
            -2.0 * (J2 @ J2 @ J2 @ asymmetry),
            -2.0 * (J2 @ J2 @ _symmetrize(K2, asymmetry)),
            -2.0 * (J2 @ _symmetrize(K2 @ K2, asymmetry)),
            -2.0 * _symmetrize(power(K2, 3), asymmetry),
        )
    octic = (
        power(J2, 4),
        power(J2, 3) @ K2,
        J2 @ J2 @ K2 @ K2,
        J2 @ power(K2, 3),
        power(K2, 4),
        *lower_order_octic,
    )
    result = tuple(_hermitian(operator) for operator in quartic + sextic + octic)
    if freeze:
        for operator in result:
            operator.setflags(write=False)
    return result


@lru_cache(maxsize=512)
def diagonalize_rotor(parameters: NativeRotorParameters, J: int) -> NativeRotorBlock:
    energies, vectors = np.linalg.eigh(rotational_hamiltonian(parameters, J))
    assignments = _adiabatic_asymmetric_top_assignments(parameters, J, vectors)
    return NativeRotorBlock(J, energies.real, vectors, assignments)


def _adiabatic_asymmetric_top_assignments(
    parameters: NativeRotorParameters, J: int, final_vectors: np.ndarray,
) -> tuple[tuple[int, int, str], ...]:
    """Track ``Ka,Kc`` labels from a slightly asymmetric prolate limit.

    Direct assignment from angular-momentum expectation values fails after
    high-J avoided crossings. This continuation preserves the spectroscopic
    IDIAG labels through both asymmetry and centrifugal distortion.
    """
    from scipy.optimize import linear_sum_assignment

    angular = angular_momentum_operators(J, parameters.representation)
    Ja, Jb, Jc = angular
    mean_bc = 0.5 * (parameters.B_MHz + parameters.C_MHz)
    epsilon = 1.0e-4

    def rigid(fraction: float) -> np.ndarray:
        B = mean_bc + fraction * (parameters.B_MHz - mean_bc)
        C = mean_bc + fraction * (parameters.C_MHz - mean_bc)
        return _hermitian(parameters.A_MHz * Ja @ Ja + B * Jb @ Jb + C * Jc @ Jc)

    _start_values, previous = np.linalg.eigh(rigid(epsilon))
    labels = _assign_asymmetric_top_block(previous, J, Ja @ Ja, Jc @ Jc)

    def continue_with(matrix: np.ndarray) -> None:
        nonlocal previous
        _values, raw = np.linalg.eigh(_hermitian(matrix))
        old, new = linear_sum_assignment(-(np.abs(previous.conj().T @ raw) ** 2))
        if not np.array_equal(old, np.arange(len(old))):
            raise RuntimeError("asymmetric-top continuation lost an assignment")
        previous = raw[:, new]

    for fraction in np.linspace(epsilon, 1.0, 25)[1:]:
        continue_with(rigid(float(fraction)))
    overlap = np.abs(final_vectors.conj().T @ previous) ** 2
    tracked_ka: list[int | None] = [None] * len(labels)
    symmetry_blocks: dict[tuple[int, int], list[int]] = {}
    for slot, (Ka, Kc, _parity) in enumerate(labels):
        symmetry_blocks.setdefault((Ka % 2, Kc % 2), []).append(slot)
    assigned_energies: set[int] = set()
    for slots in symmetry_blocks.values():
        character = np.sum(overlap[:, slots], axis=1)
        count = len(slots)
        energy_indices = [
            int(index) for index in np.argsort(-character)[:count]
            if int(index) not in assigned_energies
        ]
        if len(energy_indices) != count:
            raise RuntimeError("asymmetric-top IDIAG symmetry blocks overlap")
        energy_indices.sort()
        slots.sort(key=lambda slot: (labels[slot][0], labels[slot][1]))
        for energy, slot in zip(energy_indices, slots, strict=True):
            tracked_ka[energy] = labels[slot][0]
            assigned_energies.add(energy)
    assignments: list[tuple[int, int, str] | None] = [None] * len(labels)
    for Ka in range(J + 1):
        indices = [index for index, value in enumerate(tracked_ka) if value == Ka]
        if len(indices) != (1 if Ka == 0 else 2):
            raise RuntimeError("asymmetric-top IDIAG changed a Ka multiplicity")
        if Ka == 0:
            assignments[indices[0]] = (0, J, "e" if J % 2 == 0 else "o")
            continue
        for index, Kc in zip(indices, (J - Ka + 1, J - Ka), strict=True):
            assignments[index] = (Ka, Kc, "e" if Ka + Kc == J else "o")
    if any(value is None for value in assignments):
        raise RuntimeError("asymmetric-top continuation produced incomplete IDIAG labels")
    return tuple(value for value in assignments if value is not None)


@lru_cache(maxsize=512)
def dipole_transition_matrix(
    J_lower: int, J_upper: int, axis: str, representation: str = "Ir"
) -> np.ndarray:
    """Return reduced body-fixed dipole amplitudes in the raw K basis."""
    axis = str(axis).lower()
    if axis not in {"a", "b", "c"}:
        raise ValueError("dipole axis must be a, b or c")
    rep = normalize_representation(representation)
    roles = {"a": "z", "b": "x", "c": "y"}
    if rep.endswith("l"):
        roles = {"c": "z", "b": "x", "a": "y"}
    role = roles[axis]
    lower = np.arange(-J_lower, J_lower + 1, dtype=int)
    upper = np.arange(-J_upper, J_upper + 1, dtype=int)
    matrix = np.zeros((len(lower), len(upper)), dtype=complex)
    for i, Kl in enumerate(lower):
        for j, Ku in enumerate(upper):
            q = int(Kl - Ku)
            if abs(q) > 1:
                continue
            if role == "z":
                factor = 1.0 if q == 0 else 0.0
            elif role == "x":
                factor = -q / math.sqrt(2.0) if abs(q) == 1 else 0.0
            else:
                factor = 1.0j / math.sqrt(2.0) if abs(q) == 1 else 0.0
            if factor == 0.0:
                continue
            matrix[i, j] = (
                (-1.0) ** Kl
                * math.sqrt((2 * J_lower + 1) * (2 * J_upper + 1))
                * _wigner_3j(J_lower, 1, J_upper, -Kl, q, Ku)
                * factor
            )
    matrix.setflags(write=False)
    return matrix


def simulate_native_rotor(
    parameters: NativeRotorParameters,
    *,
    temperature_K: float,
    dipole_debye: Sequence[float],
    j_min: int = 0,
    j_max: int = 30,
    frequency_min_MHz: float = 0.0,
    frequency_max_MHz: float = float("inf"),
    intensity_cut: float = 0.0,
    active_axes: Sequence[bool] = (True, True, True),
) -> list[dict[str, object]]:
    """Generate a typed-list-compatible rigid-rotor line catalogue."""
    if temperature_K <= 0.0 or not 0 <= j_min <= j_max:
        raise ValueError("temperature and J range are invalid")
    dipole = np.asarray(dipole_debye, dtype=float)
    active = np.asarray(active_axes, dtype=bool)
    if dipole.shape != (3,) or active.shape != (3,):
        raise ValueError("dipole and active_axes must each have three components")
    blocks = {J: diagonalize_rotor(parameters, J) for J in range(j_min, j_max + 1)}
    ground = min(float(block.energies_MHz[0]) for block in blocks.values())
    beta = MHZ_TO_KELVIN / temperature_K
    partition = sum(
        (2 * J + 1) * float(np.sum(np.exp(-(block.energies_MHz - ground) * beta)))
        for J, block in blocks.items()
    )
    rows: list[dict[str, object]] = []
    for Jl in range(j_min, j_max):
        Ju = Jl + 1
        lower, upper = blocks[Jl], blocks[Ju]
        amplitude = np.zeros((len(lower.energies_MHz), len(upper.energies_MHz)), dtype=complex)
        for index, axis in enumerate(("a", "b", "c")):
            if active[index] and dipole[index] != 0.0:
                raw = dipole_transition_matrix(Jl, Ju, axis, parameters.representation)
                amplitude += dipole[index] * (lower.vectors.conj().T @ raw @ upper.vectors)
        for il, (energy_l, label_l) in enumerate(
            zip(lower.energies_MHz, lower.assignments, strict=True)
        ):
            for iu, (energy_u, label_u) in enumerate(
                zip(upper.energies_MHz, upper.assignments, strict=True)
            ):
                frequency = float(energy_u - energy_l)
                strength = float(abs(amplitude[il, iu]) ** 2)
                if not frequency_min_MHz <= frequency <= frequency_max_MHz or strength <= 1.0e-18:
                    continue
                intensity = (
                    strength
                    * frequency
                    * (-math.expm1(-frequency * beta))
                    * math.exp(-(float(energy_l) - ground) * beta)
                    / partition
                )
                if intensity < intensity_cut:
                    continue
                rows.append(
                    {
                        "Frequency (MHz)": frequency,
                        "Intensity": intensity,
                        "Relative intensity": 0.0,
                        "Branch": "R",
                        "Jl": Jl,
                        "Ka_l": label_l[0],
                        "Kc_l": label_l[1],
                        "Ju": Ju,
                        "Ka_u": label_u[0],
                        "Kc_u": label_u[1],
                    }
                )
    maximum = max((float(row["Intensity"]) for row in rows), default=1.0)
    for row in rows:
        row["Relative intensity"] = 100.0 * float(row["Intensity"]) / maximum
    return sorted(rows, key=lambda row: float(row["Frequency (MHz)"]))


def assign_asymmetric_top(
    vector: np.ndarray, J: int, Ja2: np.ndarray, Jc2: np.ndarray
) -> tuple[int, int, str, float, float]:
    EJa2 = max(0.0, float(np.real(np.vdot(vector, Ja2 @ vector))))
    EJc2 = max(0.0, float(np.real(np.vdot(vector, Jc2 @ vector))))
    best: tuple[float, int, int] | None = None
    for Ka in range(J + 1):
        for Kc in (J - Ka, J - Ka + 1):
            if 0 <= Kc <= J:
                error = (EJa2 - Ka * Ka) ** 2 + (EJc2 - Kc * Kc) ** 2
                if best is None or error < best[0]:
                    best = (error, Ka, Kc)
    assert best is not None
    _error, Ka, Kc = best
    return Ka, Kc, "e" if Ka + Kc == J else "o", EJa2, EJc2


def _assign_asymmetric_top_block(
    vectors: np.ndarray, J: int, Ja2: np.ndarray, Jc2: np.ndarray
) -> tuple[tuple[int, int, str], ...]:
    """Assign the complete asymmetric-top block with a one-to-one mapping.

    Independent nearest-label assignments can duplicate labels when centrifugal
    distortion or spin mixing moves expectation values away from integer
    limits.  A linear assignment to the exact ``K_a`` multiplicities is
    followed by the energy ordering of each asymmetry doublet, enforcing the
    complete set of ``2J+1`` asymmetric-top labels.
    """
    from scipy.optimize import linear_sum_assignment

    ka_slots = tuple(Ka for Ka in range(J + 1) for _ in range(1 if Ka == 0 else 2))
    expectations = np.asarray(
        [
            (
                max(0.0, float(np.real(np.vdot(vector, Ja2 @ vector)))),
                max(0.0, float(np.real(np.vdot(vector, Jc2 @ vector)))),
            )
            for vector in vectors.T
        ]
    )
    costs = np.asarray([[(EJa2 - Ka * Ka) ** 2 for Ka in ka_slots] for EJa2, _EJc2 in expectations])
    rows, columns = linear_sum_assignment(costs)
    selected_ka = {
        int(row): ka_slots[int(column)] for row, column in zip(rows, columns, strict=True)
    }
    selected = {}
    for Ka in range(J + 1):
        indices = sorted(index for index, value in selected_ka.items() if value == Ka)
        if Ka == 0:
            selected[indices[0]] = (0, J)
            continue
        # For A >= B >= C the lower member of a K_a asymmetry doublet
        # correlates with the larger K_c value.
        selected[indices[0]] = (Ka, J - Ka + 1)
        selected[indices[1]] = (Ka, J - Ka)
    return tuple(
        (Ka, Kc, "e" if Ka + Kc == J else "o")
        for Ka, Kc in (selected[index] for index in range(len(ka_slots)))
    )


@lru_cache(maxsize=None)
def _wigner_3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    from sympy.physics.wigner import wigner_3j

    return float(wigner_3j(j1, j2, j3, m1, m2, m3))


def _symmetrize(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 0.5 * (left @ right + right @ left)


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.conj().T)


__all__ = [
    "MHZ_TO_KELVIN",
    "NativeRotorBlock",
    "NativeRotorParameters",
    "angular_momentum_operators",
    "assign_asymmetric_top",
    "centrifugal_distortion_operators",
    "diagonalize_rotor",
    "dipole_transition_matrix",
    "native_rotor_parameters_from_section",
    "normalize_representation",
    "rotational_hamiltonian",
    "simulate_native_rotor",
]
