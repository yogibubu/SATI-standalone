from __future__ import annotations

"""Validated one-mode rovibrational diagnostics for Gaussian log/FCHK pairs.

The implementation reuses the shared FCHK, semidiagonal-vibrot and normal-mode
adapters instead of carrying private Gaussian parsers.
"""

from dataclasses import asdict, dataclass
import itertools
import json
import math
from pathlib import Path

import numpy as np

from matrix_chem import Phy, get_physical_constants
from matrix_rovib import modes_from_hessian

from .fchk import BOHR_TO_ANGSTROM, read_gaussian_fchk
from .semidiagonal import read_gaussian_vibrot_diagnostics, semidiagonal_cubic_cm


_PHY = get_physical_constants()
_B_MHZ_FROM_I_AMU_A2 = _PHY[Phy.PLANCK] / (
    8.0 * math.pi**2 * _PHY[Phy.TO_KG] * 1.0e-20
) / 1.0e6
_COMPONENTS = ((0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2))


@dataclass(frozen=True)
class GaussianOneModeAlignment:
    axis_permutation: tuple[int, int, int]
    axis_signs: tuple[int, int, int]
    gaussian_to_matrix_modes: tuple[int, ...]
    mode_signs: tuple[int, ...]
    didq_rms: float
    didq_max_abs: float


@dataclass(frozen=True)
class GaussianOneModeResult:
    log_path: Path
    fchk_path: Path
    gaussian_mode: int
    matrix_mode: int
    frequency_cm1: float
    alignment: GaussianOneModeAlignment
    q_grid: tuple[float, ...]
    inertia_amuA2: tuple[tuple[float, ...], ...]
    inverse_inertia_amuA2_inv: tuple[tuple[float, ...], ...]
    rotational_constants_MHz: tuple[tuple[float, ...], ...]
    didq_gaussian: tuple[float, ...]
    didq_finite_difference: tuple[float, ...]
    cubic_cm1: float
    quartic_cm1: float
    potential_cm1: tuple[float, ...]
    axis: str | None = None
    harmonic_average_MHz: float | None = None
    perturbative_average_MHz: float | None = None
    variational_average_MHz: float | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["log_path"] = str(self.log_path)
        payload["fchk_path"] = str(self.fchk_path)
        return payload


def probe_gaussian_one_mode(
    log_path: Path | str,
    fchk_path: Path | str,
    *,
    mode: int,
    qmax: float = 1.0,
    nq: int = 101,
    axis: str | None = None,
    quartic_cm1: float = 0.0,
    basis_size: int = 32,
    polynomial_degree: int = 10,
) -> GaussianOneModeResult:
    """Build aligned ``dI/dQ``, ``I(Q)`` and ``1/I(Q)`` diagnostics.

    ``mode`` is one-based in Gaussian's printed/anharmonic ordering.  The
    selected semidiagonal cubic constant is reused when present; a diagonal
    quartic may be supplied explicitly until the shared Gaussian adapter owns
    the corresponding full-QFF block.
    """
    log = Path(log_path)
    fchk = Path(fchk_path)
    if qmax <= 0.0 or nq < 3:
        raise ValueError("one-mode diagnostics require qmax > 0 and nq >= 3")
    data = read_gaussian_vibrot_diagnostics(log)
    if mode < 1 or mode > data.nvib:
        raise ValueError(f"Gaussian mode must be between 1 and {data.nvib}")
    fdata = read_gaussian_fchk(fchk)
    coords = np.asarray(fdata.cartesian_coordinates_bohr, dtype=float) * BOHR_TO_ANGSTROM
    freq, modes = modes_from_hessian(
        fdata.masses_amu,
        fdata.cartesian_hessian_lower,
        coords,
        linear=data.linear,
        project_tr=True,
    )
    aligned_coords, aligned_modes, alignment, didq_matrix = _align_to_gaussian(
        np.asarray(fdata.masses_amu, dtype=float),
        coords,
        freq,
        modes,
        data,
    )
    gaussian_index = mode - 1
    matrix_index = alignment.gaussian_to_matrix_modes[gaussian_index] - 1
    mode_vector = aligned_modes[matrix_index] * alignment.mode_signs[gaussian_index]
    masses = np.asarray(fdata.masses_amu, dtype=float)
    displacement = mode_vector / np.sqrt(masses)[:, None]
    q_grid = np.linspace(-float(qmax), float(qmax), int(nq))
    inertia = np.stack(
        [_inertia_tensor(masses, aligned_coords + value * displacement) for value in q_grid]
    )
    moments = np.stack([inertia[:, 0, 0], inertia[:, 1, 1], inertia[:, 2, 2]], axis=1)
    inverse = np.divide(1.0, moments, out=np.full_like(moments, np.inf), where=moments != 0.0)
    constants = _B_MHZ_FROM_I_AMU_A2 * inverse
    h = min(1.0e-4, qmax / 100.0)
    didq_fd = _sym6(
        (_inertia_tensor(masses, aligned_coords + h * displacement)
         - _inertia_tensor(masses, aligned_coords - h * displacement)) / (2.0 * h)
    )
    didq_gaussian = _sym6(np.asarray(data.inertia_derivatives_amu_sqrt_ang)[:, :, gaussian_index])
    cubic = float(semidiagonal_cubic_cm(data).get((gaussian_index,) * 3, 0.0))
    omega = float(data.frequencies_cm1[gaussian_index])
    potential = (
        0.5 * omega * omega * q_grid**2
        + cubic * q_grid**3 / 6.0
        + float(quartic_cm1) * q_grid**4 / 24.0
    )
    averages = (None, None, None)
    axis_name = None if axis is None else axis.strip().upper()
    if axis_name is not None:
        if axis_name not in {"A", "B", "C"}:
            raise ValueError("axis must be A, B or C")
        averages = _variational_perturbative_average(
            omega,
            cubic,
            float(quartic_cm1),
            q_grid,
            constants[:, "ABC".index(axis_name)],
            basis_size=basis_size,
            polynomial_degree=polynomial_degree,
        )
    return GaussianOneModeResult(
        log_path=log,
        fchk_path=fchk,
        gaussian_mode=mode,
        matrix_mode=matrix_index + 1,
        frequency_cm1=omega,
        alignment=alignment,
        q_grid=tuple(float(value) for value in q_grid),
        inertia_amuA2=tuple(tuple(float(value) for value in row) for row in moments),
        inverse_inertia_amuA2_inv=tuple(tuple(float(value) for value in row) for row in inverse),
        rotational_constants_MHz=tuple(tuple(float(value) for value in row) for row in constants),
        didq_gaussian=tuple(float(value) for value in didq_gaussian),
        didq_finite_difference=tuple(float(value) for value in didq_fd),
        cubic_cm1=cubic,
        quartic_cm1=float(quartic_cm1),
        potential_cm1=tuple(float(value) for value in potential),
        axis=axis_name,
        harmonic_average_MHz=averages[0],
        perturbative_average_MHz=averages[1],
        variational_average_MHz=averages[2],
    )


def write_gaussian_one_mode_json(path: Path | str, result: GaussianOneModeResult) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _align_to_gaussian(masses, coords, frequencies, modes, data):
    center = np.average(coords, axis=0, weights=masses)
    shifted = coords - center
    _values, axes = np.linalg.eigh(_inertia_tensor(masses, shifted))
    base_coords = shifted @ axes
    base_modes = np.asarray(modes) @ axes
    gaussian_freq = np.asarray(data.frequencies_cm1, dtype=float)
    mapping = _frequency_assignment(gaussian_freq, np.asarray(frequencies, dtype=float))
    reference = np.asarray(data.inertia_derivatives_amu_sqrt_ang, dtype=float)
    best = None
    for permutation in itertools.permutations(range(3)):
        pcoords = base_coords[:, permutation]
        pmodes = base_modes[:, :, permutation]
        for signs in itertools.product((1, -1), repeat=3):
            scoords = pcoords * np.asarray(signs)[None, :]
            smodes = pmodes * np.asarray(signs)[None, None, :]
            calculated = _didq_modes(masses, scoords, smodes)
            mode_signs = []
            columns = []
            for gaussian_index, matrix_index in enumerate(mapping):
                candidate = calculated[:, matrix_index]
                ref = _sym6(reference[:, :, gaussian_index])
                sign = -1 if float(np.dot(candidate, ref)) < 0.0 else 1
                mode_signs.append(sign)
                columns.append(sign * candidate)
            matrix = np.column_stack(columns)
            target = np.column_stack([_sym6(reference[:, :, idx]) for idx in range(data.nvib)])
            residual = matrix - target
            score = float(np.sqrt(np.mean(residual**2)))
            candidate = (score, permutation, signs, tuple(mode_signs), scoords, smodes, matrix)
            if best is None or score < best[0]:
                best = candidate
    assert best is not None
    score, permutation, signs, mode_signs, out_coords, out_modes, didq = best
    target = np.column_stack([_sym6(reference[:, :, idx]) for idx in range(data.nvib)])
    alignment = GaussianOneModeAlignment(
        axis_permutation=tuple(int(value) for value in permutation),
        axis_signs=tuple(int(value) for value in signs),
        gaussian_to_matrix_modes=tuple(int(value) + 1 for value in mapping),
        mode_signs=mode_signs,
        didq_rms=score,
        didq_max_abs=float(np.max(np.abs(didq - target))),
    )
    return out_coords, out_modes, alignment, didq


def _frequency_assignment(reference: np.ndarray, calculated: np.ndarray) -> tuple[int, ...]:
    remaining = set(range(len(calculated)))
    mapping = []
    for value in reference:
        selected = min(remaining, key=lambda index: abs(calculated[index] - value))
        mapping.append(selected)
        remaining.remove(selected)
    return tuple(mapping)


def _inertia_tensor(masses: np.ndarray, coords: np.ndarray) -> np.ndarray:
    tensor = np.zeros((3, 3), dtype=float)
    for mass, vector in zip(masses, coords):
        tensor += mass * (np.dot(vector, vector) * np.eye(3) - np.outer(vector, vector))
    return tensor


def _sym6(tensor: np.ndarray) -> np.ndarray:
    return np.asarray([tensor[i, j] for i, j in _COMPONENTS], dtype=float)


def _didq_modes(masses, coords, modes) -> np.ndarray:
    columns = []
    for mode in modes:
        displacement = mode / np.sqrt(masses)[:, None]
        h = 1.0e-4
        columns.append(
            _sym6(
                (_inertia_tensor(masses, coords + h * displacement)
                 - _inertia_tensor(masses, coords - h * displacement)) / (2.0 * h)
            )
        )
    return np.column_stack(columns)


def _variational_perturbative_average(
    omega, cubic, quartic, q_grid, observable, *, basis_size, polynomial_degree
):
    if basis_size < 4:
        raise ValueError("one-mode variational basis_size must be at least 4")
    qop = np.zeros((basis_size, basis_size), dtype=float)
    scale = 1.0 / math.sqrt(2.0 * omega)
    for n in range(basis_size - 1):
        value = math.sqrt(n + 1.0) * scale
        qop[n, n + 1] = qop[n + 1, n] = value
    q2, q3 = qop @ qop, qop @ qop @ qop
    perturbation = cubic * q3 / 6.0 + quartic * (q2 @ q2) / 24.0
    h0 = np.diag(omega * (np.arange(basis_size) + 0.5))
    _energies, vectors = np.linalg.eigh(h0 + perturbation)
    degree = min(int(polynomial_degree), len(q_grid) - 1)
    coefficients = np.polynomial.polynomial.polyfit(q_grid, observable, degree)
    operator = np.zeros_like(qop)
    power = np.eye(basis_size)
    for coefficient in coefficients:
        operator += coefficient * power
        power = power @ qop
    harmonic = float(operator[0, 0])
    first_order = np.zeros(basis_size)
    for n in range(1, basis_size):
        first_order[n] = perturbation[n, 0] / (h0[0, 0] - h0[n, n])
    perturbative = harmonic + 2.0 * float(np.dot(operator[0, 1:], first_order[1:]))
    psi = vectors[:, 0]
    variational = float(np.real_if_close(psi.conj() @ operator @ psi))
    return harmonic, perturbative, variational
