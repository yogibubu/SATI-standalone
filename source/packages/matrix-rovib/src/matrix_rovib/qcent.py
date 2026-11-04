from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from matrix_chem import Phy, get_physical_constants
from matrix_rovib.contracts import read_rotational_section

from .conversions import conversion_factor
from .vibin import VibinData, read_vibin


CM1_TO_MHZ = get_physical_constants()[Phy.C_LIGHT] / 1.0e6
WATSON_A_QUARTIC_LABELS = ("DeltaJ", "DeltaJK", "DeltaK", "deltaJ", "deltaK")
WATSON_S_QUARTIC_LABELS = ("DJ", "DJK", "DK", "d1", "d2")
REPRESENTATION_AXIS_PERMUTATIONS = {
    "Ir": (2, 0, 1), "IIr": (0, 1, 2), "IIIr": (1, 2, 0),
    "Il": (2, 1, 0), "IIl": (1, 0, 2), "IIIl": (0, 2, 1),
}


@dataclass(frozen=True)
class QCentResult:
    representation: str
    linear: bool
    tauP6_MHz: tuple[float, ...]
    tauP6_cm1: tuple[float, ...]
    QA_MHz: tuple[float, ...]
    QA_cm1: tuple[float, ...]
    QS_MHz: tuple[float, ...]
    QS_cm1: tuple[float, ...]
    nvib: int
    min_principal_moment_amuA2: float

    @property
    def DelJ_MHz(self) -> float:
        return self.QA_MHz[0]

    @property
    def DelJK_MHz(self) -> float:
        return self.QA_MHz[1]

    @property
    def DelK_MHz(self) -> float:
        return self.QA_MHz[2]

    @property
    def delJ_MHz(self) -> float:
        return self.QA_MHz[3]

    @property
    def delK_MHz(self) -> float:
        return self.QA_MHz[4]


def compute_qcent(
    vibin: Path | str | VibinData,
    *,
    representation: str | None = None,
    linear: bool | None = None,
) -> QCentResult:
    data = read_vibin(vibin) if not isinstance(vibin, VibinData) else vibin
    rep = representation or data.representation
    is_linear = data.linear if linear is None else bool(linear)
    inertia = inertia_tensor_amuA2(data.masses_amu, data.coords_A)
    pmom = np.array([inertia[0, 0], inertia[1, 1], inertia[2, 2]], dtype=float)
    ax_ok = pmom > 1.0e-14
    didq = didq_sym6_qcent(data.masses_amu, data.coords_A, data.modes_mw)
    tau = tau_wilson(data.freq_cm1, pmom, didq)
    tau_mhz = tau * CM1_TO_MHZ
    tauP6 = np.array(
        [
            tau_mhz[0, 0, 0, 0],
            tau_mhz[0, 0, 1, 1],
            tau_mhz[0, 0, 2, 2],
            tau_mhz[1, 1, 1, 1],
            tau_mhz[1, 1, 2, 2],
            tau_mhz[2, 2, 2, 2],
        ],
        dtype=float,
    )
    tau_permuted = _permute_tau(tau_mhz, _get_rep_perm(_rep_id(rep)))
    qa_mhz = _tau_to_qcent_ared_ir(tau_permuted, ax_ok)
    qs_mhz = np.asarray(convert_qcent_watson_reduction(qa_mhz, "A", "S"), dtype=float)
    return QCentResult(
        representation=rep,
        linear=is_linear,
        tauP6_MHz=tuple(float(value) for value in tauP6),
        tauP6_cm1=tuple(float(value) for value in tauP6 / CM1_TO_MHZ),
        QA_MHz=tuple(float(value) for value in qa_mhz),
        QA_cm1=tuple(float(value) for value in qa_mhz / CM1_TO_MHZ),
        QS_MHz=tuple(float(value) for value in qs_mhz),
        QS_cm1=tuple(float(value) for value in qs_mhz / CM1_TO_MHZ),
        nvib=int(data.freq_cm1.size),
        min_principal_moment_amuA2=float(np.min(pmom)),
    )


def compute_qcent_from_inertia_derivatives(
    masses_amu,
    coords_A,
    frequencies_cm1,
    inertia_derivatives_amuA2,
    *,
    representation: str = "Ir",
    linear: bool = False,
) -> QCentResult:
    """Watson quartic constants from principal-axis dI/dQ in six-index order.

    ``Q`` is a mass-weighted normal coordinate. This entry point also accepts
    modes obtained from a nonredundant internal-coordinate Wilson GF problem.
    """
    masses = np.asarray(masses_amu, dtype=float).reshape(-1)
    coordinates = np.asarray(coords_A, dtype=float)
    frequencies = np.asarray(frequencies_cm1, dtype=float).reshape(-1)
    didq = np.asarray(inertia_derivatives_amuA2, dtype=float)
    if (coordinates.shape != (masses.size, 3) or didq.shape != (6, frequencies.size)
            or np.any(masses <= 0) or np.any(frequencies <= 0)
            or not all(np.all(np.isfinite(value)) for value in (masses, coordinates, frequencies, didq))):
        raise ValueError("QCENT requires positive frequencies and matching finite inertia derivatives")
    if representation not in REPRESENTATION_AXIS_PERMUTATIONS:
        raise ValueError("unknown Watson axis representation")
    inertia = inertia_tensor_amuA2(masses, coordinates)
    pmom = np.diag(inertia).copy()
    if not np.allclose(inertia, np.diag(pmom), atol=1e-8):
        raise ValueError("QCENT inertia derivatives require principal-axis coordinates")
    rep = representation
    is_linear = bool(linear)
    ax_ok = pmom > 1.0e-14
    tau = tau_wilson(frequencies, pmom, didq)
    tau_mhz = tau * CM1_TO_MHZ
    tauP6 = np.array(
        [
            tau_mhz[0, 0, 0, 0],
            tau_mhz[0, 0, 1, 1],
            tau_mhz[0, 0, 2, 2],
            tau_mhz[1, 1, 1, 1],
            tau_mhz[1, 1, 2, 2],
            tau_mhz[2, 2, 2, 2],
        ],
        dtype=float,
    )
    tau_permuted = _permute_tau(tau_mhz, _get_rep_perm(_rep_id(rep)))
    qa_mhz = _tau_to_qcent_ared_ir(tau_permuted, ax_ok)
    qs_mhz = np.asarray(convert_qcent_watson_reduction(qa_mhz, "A", "S"), dtype=float)
    return QCentResult(
        representation=rep,
        linear=is_linear,
        tauP6_MHz=tuple(float(value) for value in tauP6),
        tauP6_cm1=tuple(float(value) for value in tauP6 / CM1_TO_MHZ),
        QA_MHz=tuple(float(value) for value in qa_mhz),
        QA_cm1=tuple(float(value) for value in qa_mhz / CM1_TO_MHZ),
        QS_MHz=tuple(float(value) for value in qs_mhz),
        QS_cm1=tuple(float(value) for value in qs_mhz / CM1_TO_MHZ),
        nvib=int(frequencies.size),
        min_principal_moment_amuA2=float(np.min(pmom)),
    )


def compute_qcent_from_xyzin(
    xyzin: Path | str,
    *,
    vibin: Path | str | None = None,
) -> QCentResult:
    target = Path(xyzin)
    rot = read_rotational_section(target)
    data_path = target.parent / "vibin" if vibin is None else Path(vibin)
    return compute_qcent(
        data_path,
        representation=rot.representation or None,
        linear=None,
    )


def qcent_report_lines(result: QCentResult) -> list[str]:
    lines = [
        "QCENT",
        f"representation = {result.representation}",
        f"linear = {int(result.linear)}",
        "TauP_MHz aaaa aabb aacc bbbb bbcc cccc",
        " ".join(f"{value:.10e}" for value in result.tauP6_MHz),
        "A-reduction_MHz DelJ DelJK DelK delJ delK",
        " ".join(f"{value:.10e}" for value in result.QA_MHz),
        "S-reduction_MHz DJ DJK DK d1 d2",
        " ".join(f"{value:.10e}" for value in result.QS_MHz),
    ]
    return lines


def append_qcent_to_vibin(path: Path | str, result: QCentResult) -> None:
    target = Path(path)
    with target.open("a", encoding="utf-8") as handle:
        handle.write("\nBEGIN_RESULTS\n")
        for line in qcent_report_lines(result):
            handle.write(line + "\n")
        handle.write("END_RESULTS\n")


def inertia_tensor_amuA2(masses_amu, coords_A) -> np.ndarray:
    masses = np.asarray(masses_amu, dtype=float)
    coords = np.asarray(coords_A, dtype=float)
    inertia = np.zeros((3, 3), dtype=float)
    for mass, (x, y, z) in zip(masses, coords):
        inertia[0, 0] += mass * (y * y + z * z)
        inertia[1, 1] += mass * (x * x + z * z)
        inertia[2, 2] += mass * (x * x + y * y)
        inertia[0, 1] -= mass * x * y
        inertia[0, 2] -= mass * x * z
        inertia[1, 2] -= mass * y * z
    inertia[1, 0] = inertia[0, 1]
    inertia[2, 0] = inertia[0, 2]
    inertia[2, 1] = inertia[1, 2]
    return inertia


def didq_sym6_qcent(masses_amu, coords_A, modes_mw) -> np.ndarray:
    masses = np.asarray(masses_amu, dtype=float)
    coords = np.asarray(coords_A, dtype=float)
    modes = np.asarray(modes_mw, dtype=float)
    nvib = modes.shape[0]
    didq = np.zeros((6, nvib), dtype=float)
    for mode_idx in range(nvib):
        for atom_idx, mass in enumerate(masses):
            x, y, z = coords[atom_idx]
            dx, dy, dz = modes[mode_idx, atom_idx]
            didq[0, mode_idx] += mass * (2.0 * y * dy + 2.0 * z * dz)
            didq[1, mode_idx] += mass * (2.0 * x * dx + 2.0 * z * dz)
            didq[2, mode_idx] += mass * (2.0 * x * dx + 2.0 * y * dy)
            didq[3, mode_idx] -= mass * (x * dy + y * dx)
            didq[4, mode_idx] -= mass * (x * dz + z * dx)
            didq[5, mode_idx] -= mass * (y * dz + z * dy)
    return didq


def tau_wilson(freq_cm1, principal_moments, didq) -> np.ndarray:
    freq = np.asarray(freq_cm1, dtype=float)
    pmom = np.asarray(principal_moments, dtype=float)
    didq = np.asarray(didq, dtype=float)
    factg = conversion_factor("FACTG")
    tau = np.zeros((3, 3, 3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    acc = 0.0
                    for mode_idx, omega in enumerate(freq):
                        if omega == 0.0:
                            continue
                        acc += (
                            didq[_sym6(i, j), mode_idx] * didq[_sym6(k, l), mode_idx] / (omega**2)
                        )
                    denom = factg**3 * pmom[i] * pmom[j] * pmom[k] * pmom[l]
                    tau[i, j, k, l] = 0.0 if denom == 0.0 else -0.5 * acc / denom
    return tau


def _sym6(i: int, j: int) -> int:
    return [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)].index(tuple(sorted((i, j))))


def _rep_id(rep: str) -> int:
    mapping = {"Ir": 1, "IIr": 2, "IIIr": 3, "Il": 4, "IIl": 5, "IIIl": 6}
    return mapping.get((rep or "").strip(), 2)


def _get_rep_perm(rep_id: int) -> list[int]:
    if rep_id == 1:
        return [2, 0, 1]
    if rep_id == 2:
        return [0, 1, 2]
    if rep_id == 3:
        return [1, 2, 0]
    if rep_id == 4:
        return [2, 1, 0]
    if rep_id == 5:
        return [1, 0, 2]
    if rep_id == 6:
        return [0, 2, 1]
    return [0, 1, 2]


def representation_axis_permutation(representation: str) -> tuple[int, int, int]:
    """Return the QCENT principal-axis permutation for all six conventions."""
    token = (representation or "").strip()
    if token not in REPRESENTATION_AXIS_PERMUTATIONS:
        raise ValueError("representation must be Ir, IIr, IIIr, Il, IIl or IIIl")
    return REPRESENTATION_AXIS_PERMUTATIONS[token]


def convert_qcent_watson_reduction(
    constants_MHz,
    reduction_from: str,
    reduction_to: str,
) -> tuple[float, float, float, float, float]:
    """Convert the Q-cent A/S five-constant convention.

    This is the exact reversible map used by ``geometry/qcent.py`` and is
    intentionally named as a Q-cent compatibility conversion.  General
    Hamiltonian contact transformations remain the responsibility of CeDiTT.
    """
    values = np.asarray(constants_MHz, dtype=float)
    if values.shape != (5,) or not np.all(np.isfinite(values)):
        raise ValueError("Watson quartic conversion requires five finite constants")
    source = reduction_from.strip().upper()
    target = reduction_to.strip().upper()
    if source not in {"A", "S"} or target not in {"A", "S"}:
        raise ValueError("Watson reduction must be A or S")
    if source == target:
        out = values
    elif source == "A":
        delta_j, delta_jk, delta_k, small_j, small_k = values
        out = np.array(
            [delta_j, delta_jk + delta_j, delta_k + delta_jk + delta_j, small_j, small_k]
        )
    else:
        d_j, d_jk, d_k, d1, d2 = values
        out = np.array([d_j, d_jk - d_j, d_k - d_jk, d1, d2])
    return tuple(float(value) for value in out)


def _permute_tau(tau_in: np.ndarray, perm: list[int]) -> np.ndarray:
    tau_out = np.zeros_like(tau_in)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    tau_out[i, j, k, l] = tau_in[perm[i], perm[j], perm[k], perm[l]]
    return tau_out


def _tau_to_qcent_ared_ir(tau: np.ndarray, ax_ok: np.ndarray) -> np.ndarray:
    if not (ax_ok[0] and ax_ok[1] and ax_ok[2]):
        return np.zeros(5, dtype=float)
    delta_j = 0.125 * (tau[1, 1, 1, 1] + tau[2, 2, 2, 2] + 2.0 * tau[1, 1, 2, 2])
    delta_k = 0.125 * tau[0, 0, 0, 0]
    delta_jk = -0.25 * (tau[0, 0, 1, 1] + tau[0, 0, 2, 2])
    small_j = 0.125 * (tau[1, 1, 1, 1] + tau[2, 2, 2, 2] - 2.0 * tau[1, 1, 2, 2])
    small_k = -0.25 * (tau[0, 1, 0, 1] + tau[0, 2, 0, 2])
    return np.array([delta_j, delta_jk, delta_k, small_j, small_k], dtype=float)


def _qcent_ared_to_sred_ir(q_a: np.ndarray) -> np.ndarray:
    return np.asarray(convert_qcent_watson_reduction(q_a, "A", "S"), dtype=float)
