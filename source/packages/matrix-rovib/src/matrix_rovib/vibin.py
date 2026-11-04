from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from matrix_chem import Phy, Structure, get_physical_constants, read_enriched_xyz
from matrix_chem.rotational import rotational_info
from matrix_core import has_section
from matrix_numerics import eigh_arrays
from matrix_rovib.contracts import VibrationalSection, write_vibrational_section


@dataclass(frozen=True)
class VibinData:
    symbols: tuple[str, ...]
    coords_A: np.ndarray
    masses_amu: np.ndarray
    freq_cm1: np.ndarray
    modes_mw: np.ndarray
    representation: str = "Ir"
    linear: bool = False
    project_TR: bool = True
    normalization: str = "mass-weighted"
    didq_sym6: np.ndarray | None = None

    @property
    def natoms(self) -> int:
        return len(self.symbols)

    @property
    def nvib(self) -> int:
        return int(self.freq_cm1.size)


@dataclass(frozen=True)
class VibinBuildResult:
    vibin: Path
    data: VibinData
    n_imag_like: int


def read_vibin(path: Path | str) -> VibinData:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError("vibin is too short")
    natoms = int(lines[0].strip())
    symbols: list[str] = []
    coords = np.zeros((natoms, 3), dtype=float)
    for idx, raw in enumerate(lines[2 : 2 + natoms]):
        parts = raw.split()
        if len(parts) < 4:
            raise ValueError(f"invalid vibin XYZ line: {raw}")
        symbols.append(parts[0])
        coords[idx] = [float(parts[1]), float(parts[2]), float(parts[3])]

    representation = "Ir"
    linear = False
    project_TR = True
    normalization = "mass-weighted"
    masses_amu = None
    freq_cm1 = None
    didq = None
    modes: list[np.ndarray] = []

    i = 2 + natoms
    while i < len(lines):
        raw = lines[i]
        text = raw.strip()
        if not text:
            i += 1
            continue
        if text.startswith("representation"):
            representation = text.split("=", 1)[1].strip()
            i += 1
            continue
        if text.startswith("linear"):
            linear = _parse_bool(text.split("=", 1)[1])
            i += 1
            continue
        if text.startswith("project_TR"):
            project_TR = _parse_bool(text.split("=", 1)[1])
            i += 1
            continue
        if text.startswith("normalization"):
            normalization = text.split("=", 1)[1].strip()
            i += 1
            continue
        if text.startswith("masses_amu") and "[" in text:
            masses_amu, i = _read_indexed_vector(lines, i + 1)
            continue
        if text.startswith("freq_cm1") and "[" in text:
            freq_cm1, i = _read_indexed_vector(lines, i + 1)
            continue
        if text.startswith("didq_sym6") and "[" in text:
            didq, i = _read_indexed_matrix(lines, i + 1, width=6)
            didq = didq.T
            continue
        if text.startswith("MODE"):
            mode = np.zeros((natoms, 3), dtype=float)
            i += 1
            while i < len(lines):
                row = lines[i].strip()
                if row.startswith("ENDMODE"):
                    break
                parts = row.split()
                if len(parts) >= 4:
                    atom_idx = int(parts[0]) - 1
                    mode[atom_idx] = [float(parts[1]), float(parts[2]), float(parts[3])]
                i += 1
            modes.append(mode)
            i += 1
            continue
        if text.startswith("BEGIN_RESULTS") or text.startswith("BEGIN_CORIOLIS"):
            break
        i += 1

    if masses_amu is None:
        raise ValueError("vibin masses_amu block not found")
    if freq_cm1 is None:
        raise ValueError("vibin freq_cm1 block not found")
    modes_mw = np.asarray(modes, dtype=float)
    if modes_mw.shape != (len(freq_cm1), natoms, 3):
        raise ValueError("vibin MODE blocks do not match frequencies/atoms")
    return VibinData(
        symbols=tuple(symbols),
        coords_A=coords,
        masses_amu=np.asarray(masses_amu, dtype=float),
        freq_cm1=np.asarray(freq_cm1, dtype=float),
        modes_mw=modes_mw,
        representation=representation,
        linear=linear,
        project_TR=project_TR,
        normalization=normalization,
        didq_sym6=didq,
    )


def write_vibin(path: Path | str, data: VibinData) -> Path:
    target = Path(path)
    lines = [str(data.natoms), "vibro-rotational input"]
    for symbol, row in zip(data.symbols, data.coords_A):
        lines.append(f"{symbol:2s} {row[0]:15.8f} {row[1]:15.8f} {row[2]:15.8f}")
    lines.extend(
        [
            "",
            f"representation = {data.representation}",
            f"linear = {bool(data.linear)}",
            f"project_TR = {bool(data.project_TR)}",
            f"normalization = {data.normalization}",
            "",
            "masses_amu [",
        ]
    )
    for idx, mass in enumerate(data.masses_amu, 1):
        lines.append(f" {idx:5d} {float(mass):15.8f}")
    lines.extend(["]", "", "freq_cm1 ["])
    for idx, freq in enumerate(data.freq_cm1, 1):
        lines.append(f" {idx:5d} {float(freq):15.8f}")
    lines.extend(["]", ""])
    if data.didq_sym6 is not None:
        lines.append("didq_sym6 [")
        didq = np.asarray(data.didq_sym6, dtype=float)
        for idx in range(data.nvib):
            row = didq[:, idx]
            lines.append(
                f" {idx + 1:5d} {row[0]: .10e} {row[1]: .10e} {row[2]: .10e}"
                f" {row[3]: .10e} {row[4]: .10e} {row[5]: .10e}"
            )
        lines.extend(["]", ""])
    for mode_idx in range(data.nvib):
        lines.append(f"MODE {mode_idx + 1}")
        for atom_idx in range(data.natoms):
            dx, dy, dz = data.modes_mw[mode_idx, atom_idx]
            lines.append(f"{atom_idx + 1:5d} {dx:15.8e} {dy:15.8e} {dz:15.8e}")
        lines.extend(["ENDMODE", ""])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return target


def vibin_from_xyzin_fchk(
    xyzin: Path | str,
    fchk: Path | str,
    *,
    workdir: Path | str | None = None,
    project_TR: bool = True,
    update_vibrational_section: bool = True,
) -> VibinBuildResult:
    from matrix_gaussian import read_gaussian_fchk

    xyzin_path = Path(xyzin)
    fchk_data = read_gaussian_fchk(Path(fchk))
    geometry = read_enriched_xyz(xyzin_path)
    coords = np.asarray(geometry.coordinates_angstrom, dtype=float)
    if len(fchk_data.masses_amu) != geometry.natoms:
        raise ValueError(
            "fchk atom count does not match xyzin "
            f"({len(fchk_data.masses_amu)} vs {geometry.natoms})"
        )
    structure = Structure(list(geometry.atoms), [tuple(row) for row in coords])
    rot_info = rotational_info(structure, isotopic=True)
    linear = rot_info["rotor_type"] == "linear"
    representation = "Ir"
    if has_section(xyzin_path, "ROTATIONAL"):
        from matrix_rovib.contracts import read_rotational_section

        representation = read_rotational_section(xyzin_path).representation or representation
    freq, modes = modes_from_hessian(
        masses_amu=fchk_data.masses_amu,
        hess_tri=fchk_data.cartesian_hessian_lower,
        coords_A=coords,
        linear=linear,
        project_tr=project_TR,
    )
    didq = didq_sym6(fchk_data.masses_amu, coords, modes)
    data = VibinData(
        symbols=tuple(geometry.atoms),
        coords_A=coords,
        masses_amu=fchk_data.masses_amu,
        freq_cm1=freq,
        modes_mw=modes,
        representation=representation,
        linear=linear,
        project_TR=project_TR,
        didq_sym6=didq,
    )
    output_dir = Path(workdir) if workdir is not None else xyzin_path.parent
    vibin_path = write_vibin(output_dir / "vibin", data)
    n_imag_like = int(np.sum(freq < 0.0))
    if update_vibrational_section:
        write_vibrational_section(
            xyzin_path,
            VibrationalSection(
                linear=linear,
                nvib=int(freq.size),
                n_imag_like=n_imag_like,
                frequencies_cm1=tuple(float(value) for value in freq),
            ),
        )
    return VibinBuildResult(vibin=vibin_path, data=data, n_imag_like=n_imag_like)


def modes_from_hessian(
    masses_amu,
    hess_tri,
    coords_A,
    *,
    linear: bool = False,
    project_tr: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    masses = np.asarray(masses_amu, dtype=float)
    coords = np.asarray(coords_A, dtype=float)
    natoms = len(masses)
    nd = 3 * natoms
    expected = nd * (nd + 1) // 2
    hess = np.asarray(hess_tri, dtype=float)
    if len(hess) != expected:
        raise ValueError(f"Hessian size mismatch: expected {expected}, got {len(hess)}")

    H = np.zeros((nd, nd), dtype=float)
    idx = 0
    for i in range(nd):
        for j in range(i + 1):
            H[i, j] = hess[idx]
            H[j, i] = hess[idx]
            idx += 1

    weights = 1.0 / np.sqrt(np.repeat(masses, 3))
    Hmw = H * (weights[:, None] * weights[None, :])
    coords_bohr = coords / _constants()[Phy.TO_ANG]
    basis = _tr_basis(coords_bohr, masses)
    Q = _orthonormal_columns(basis)
    if project_tr and Q.shape[1] > 0:
        P = np.eye(nd) - Q @ Q.T
        Hmw = P @ Hmw @ P

    eigvals, eigvecs = eigh_arrays(Hmw)
    nzero = Q.shape[1] if Q.shape[1] > 0 else (5 if linear else 6)
    if nzero >= len(eigvals):
        raise ValueError("not enough modes to drop translation/rotation")
    drop_idx = np.argsort(np.abs(eigvals))[:nzero]
    keep = np.ones(len(eigvals), dtype=bool)
    keep[drop_idx] = False
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return _hessian_eig_to_freq_cm1(eigvals), eigvecs.T.reshape((-1, natoms, 3))


def didq_sym6(masses_amu, coords_A, modes_mw) -> np.ndarray:
    masses = np.asarray(masses_amu, dtype=float)
    coords = np.asarray(coords_A, dtype=float)
    modes = np.asarray(modes_mw, dtype=float)
    nvib = modes.shape[0]
    out = np.zeros((6, nvib), dtype=float)
    for mode_idx in range(nvib):
        for atom_idx, mass in enumerate(masses):
            sqrm = np.sqrt(mass)
            x, y, z = coords[atom_idx]
            dx, dy, dz = modes[mode_idx, atom_idx] / sqrm
            rdotdr = x * dx + y * dy + z * dz
            out[0, mode_idx] += mass * (2.0 * rdotdr - 2.0 * x * dx)
            out[1, mode_idx] += mass * (2.0 * rdotdr - 2.0 * y * dy)
            out[2, mode_idx] += mass * (2.0 * rdotdr - 2.0 * z * dz)
            out[3, mode_idx] += mass * (-x * dy - y * dx)
            out[4, mode_idx] += mass * (-x * dz - z * dx)
            out[5, mode_idx] += mass * (-y * dz - z * dy)
    return out


def _hessian_eig_to_freq_cm1(eigvals) -> np.ndarray:
    phy = _constants()
    factor = (1.0 / (2.0 * np.pi * phy[Phy.C_LIGHT])) * np.sqrt(
        phy[Phy.HARTREE] / (phy[Phy.M_PER_B] * phy[Phy.M_PER_B] * phy[Phy.TO_KG])
    )
    eig = np.asarray(eigvals, dtype=float)
    return np.sign(eig) * np.sqrt(np.abs(eig)) * factor


def _tr_basis(coords_bohr: np.ndarray, masses_amu: np.ndarray) -> np.ndarray:
    natoms = len(masses_amu)
    nd = 3 * natoms
    sqrt_m = np.sqrt(masses_amu)
    com = np.average(coords_bohr, axis=0, weights=masses_amu)
    shifted = coords_bohr - com
    translations = []
    for axis in range(3):
        vector = np.zeros((natoms, 3), dtype=float)
        vector[:, axis] = sqrt_m
        translations.append(vector.reshape(nd))
    rotations = []
    for axis_vector in np.eye(3):
        rotations.append((np.cross(axis_vector, shifted) * sqrt_m[:, None]).reshape(nd))
    return np.column_stack(translations + rotations)


def _orthonormal_columns(matrix: np.ndarray, rtol: float = 1.0e-12) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    if s.size == 0:
        return matrix[:, :0]
    rank = int(np.sum(s > rtol * s[0]))
    return u[:, :rank]


def _read_indexed_vector(lines: list[str], start: int) -> tuple[np.ndarray, int]:
    values: list[float] = []
    idx = start
    while idx < len(lines):
        text = lines[idx].strip()
        if text.startswith("]"):
            return np.asarray(values, dtype=float), idx + 1
        if text:
            parts = text.split()
            if len(parts) >= 2:
                values.append(float(parts[1]))
        idx += 1
    raise ValueError("unterminated vibin vector block")


def _read_indexed_matrix(lines: list[str], start: int, *, width: int) -> tuple[np.ndarray, int]:
    rows: list[list[float]] = []
    idx = start
    while idx < len(lines):
        text = lines[idx].strip()
        if text.startswith("]"):
            return np.asarray(rows, dtype=float), idx + 1
        if text:
            parts = text.split()
            if len(parts) >= width + 1:
                rows.append([float(value) for value in parts[1 : width + 1]])
        idx += 1
    raise ValueError("unterminated vibin matrix block")


def _parse_bool(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "y"}


def _constants():
    return get_physical_constants()
