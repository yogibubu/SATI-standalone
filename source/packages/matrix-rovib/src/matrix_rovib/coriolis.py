from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from matrix_chem import Phy, get_physical_constants
from matrix_core import has_section
from matrix_rovib.contracts import (
    read_deltabvib_section,
    read_rotational_section,
    resolved_rotational_constants_MHz,
)

from .vibin import VibinData, read_vibin


_C_CM_S = get_physical_constants()[Phy.C_LIGHT]
CM1_TO_MHZ = _C_CM_S / 1.0e6
MHZ_TO_CM1 = 1.0 / CM1_TO_MHZ


@dataclass(frozen=True)
class CoriolisEntry:
    i: int
    j: int
    kneg: int
    zeta: float
    Geff_cm1: float
    Geff_MHz: float


@dataclass(frozen=True)
class CoriolisResult:
    entries: tuple[CoriolisEntry, ...]
    A_cm1: float
    B_cm1: float
    C_cm1: float
    threshold_cm1: float


def compute_coriolis_sparse_entries(
    masses_amu,
    modes_mw,
    freq_cm1,
    A_cm1: float,
    B_cm1: float,
    C_cm1: float,
    *,
    Geff_thr_cm1: float = 1.0,
    only_upper: bool = True,
) -> tuple[CoriolisEntry, ...]:
    masses = np.asarray(masses_amu, dtype=float)
    freq = np.asarray(freq_cm1, dtype=float)
    modes = np.asarray(modes_mw, dtype=float)
    nvib, natoms, _ = modes.shape
    if masses.shape[0] != natoms:
        raise ValueError("Coriolis masses do not match normal modes")
    axis_constants = {1: float(A_cm1), 2: float(B_cm1), 3: float(C_cm1)}
    entries: list[CoriolisEntry] = []

    for i in range(nvib):
        jstart = i + 1 if only_upper else 0
        for j in range(jstart, nvib):
            if i == j:
                continue
            wi = float(freq[i])
            wj = float(freq[j])
            if wi <= 0.0 or wj <= 0.0:
                continue
            acc = np.zeros(3, dtype=float)
            for atom_idx in range(natoms):
                acc += np.cross(modes[i, atom_idx, :], modes[j, atom_idx, :]) / masses[atom_idx]
            for k in (1, 2, 3):
                zeta = float(acc[k - 1])
                geff_cm1 = 2.0 * axis_constants[k] * zeta * (wi + wj) / np.sqrt(wi * wj)
                if abs(geff_cm1) >= float(Geff_thr_cm1):
                    entries.append(
                        CoriolisEntry(
                            i=i + 1,
                            j=j + 1,
                            kneg=-k,
                            zeta=zeta,
                            Geff_cm1=float(geff_cm1),
                            Geff_MHz=float(geff_cm1 * CM1_TO_MHZ),
                        )
                    )
    return tuple(sorted(entries, key=lambda item: abs(item.Geff_cm1), reverse=True))


def compute_coriolis_from_vibin(
    vibin: Path | str | VibinData,
    A: float,
    B: float,
    C: float,
    *,
    units: str = "MHz",
    Geff_thr_cm1: float = 1.0,
    only_upper: bool = True,
) -> CoriolisResult:
    data = read_vibin(vibin) if not isinstance(vibin, VibinData) else vibin
    if units.lower() in {"mhz", "mhz."}:
        A_cm1 = float(A) * MHZ_TO_CM1
        B_cm1 = float(B) * MHZ_TO_CM1
        C_cm1 = float(C) * MHZ_TO_CM1
    elif units.lower() in {"cm-1", "cm^-1", "cm1"}:
        A_cm1 = float(A)
        B_cm1 = float(B)
        C_cm1 = float(C)
    else:
        raise ValueError("units must be 'MHz' or 'cm-1'")
    entries = compute_coriolis_sparse_entries(
        data.masses_amu,
        data.modes_mw,
        data.freq_cm1,
        A_cm1,
        B_cm1,
        C_cm1,
        Geff_thr_cm1=Geff_thr_cm1,
        only_upper=only_upper,
    )
    return CoriolisResult(
        entries=entries, A_cm1=A_cm1, B_cm1=B_cm1, C_cm1=C_cm1, threshold_cm1=Geff_thr_cm1
    )


def compute_coriolis_from_xyzin(
    xyzin: Path | str,
    *,
    vibin: Path | str | None = None,
    Geff_thr_cm1: float = 1.0,
    only_upper: bool = True,
) -> CoriolisResult:
    target = Path(xyzin)
    rot = read_rotational_section(target)
    deltabvib = read_deltabvib_section(target) if has_section(target, "DELTABVIB") else None
    A, B, C = resolved_rotational_constants_MHz(
        rot, state="ground_state", deltabvib=deltabvib
    )
    if A is None or B is None or C is None:
        raise ValueError("Coriolis requires A_MHz, B_MHz and C_MHz in #ROTATIONAL")
    return compute_coriolis_from_vibin(
        target.parent / "vibin" if vibin is None else vibin,
        A,
        B,
        C,
        units="MHz",
        Geff_thr_cm1=Geff_thr_cm1,
        only_upper=only_upper,
    )


def coriolis_report_lines(result: CoriolisResult) -> list[str]:
    lines = [
        "CORIOLIS",
        f"A_cm1 = {result.A_cm1:.12g}",
        f"B_cm1 = {result.B_cm1:.12g}",
        f"C_cm1 = {result.C_cm1:.12g}",
        f"threshold_Geff_cm1 = {result.threshold_cm1:.8g}",
        "i j -k zeta Geff_cm1 Geff_MHz",
    ]
    for entry in result.entries:
        lines.append(
            f"{entry.i:d} {entry.j:d} {entry.kneg:d} "
            f"{entry.zeta:.10e} {entry.Geff_cm1:.10e} {entry.Geff_MHz:.10e}"
        )
    return lines


def append_coriolis_to_vibin(path: Path | str, result: CoriolisResult) -> None:
    target = Path(path)
    with target.open("a", encoding="utf-8") as handle:
        handle.write("\nBEGIN_CORIOLIS\n")
        for line in coriolis_report_lines(result):
            handle.write(line + "\n")
        handle.write("END_CORIOLIS\n")
