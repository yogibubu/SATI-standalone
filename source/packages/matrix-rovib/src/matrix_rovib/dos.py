from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from matrix_core import has_section, read_basic_section

from .contracts import (
    read_deltabvib_section,
    read_rotational_section,
    read_vibrational_section,
    resolved_rotational_constants_MHz,
)


HC_OVER_KB = 1.438776877
MHZ_TO_CM1 = 1.0e6 / 2.99792458e10


@dataclass(frozen=True)
class DOSResult:
    path: Path | None
    bins_logg: dict[int, float]
    emin_cm1: float
    bin_cm1: float
    Q_vib: float | None = None
    T_K: float | None = None
    q_path: Path | None = None
    number_path: Path | None = None
    is_transition_state: bool = False
    excluded_imaginary_modes: tuple[int, ...] = ()
    cache_hit: bool = False


@dataclass(frozen=True)
class RovibDOSResult:
    dos_vib: Path
    dos_rot: Path | None
    dos_rovib: Path
    q_path: Path | None
    Q_rovib: float
    T_K: float
    bin_cm1: float
    vib_emin_cm1: float
    vib_emax_cm1: float
    number_path: Path | None = None


def direct_sum_dos(
    omega_cm1: list[float],
    chi_cm1: list[list[float]],
    vmax: Iterable[int] | int,
    emax_cm1: float,
    bin_cm1: float,
    ncap: Iterable[float] | float | None = None,
) -> dict[int, float]:
    n = len(omega_cm1)
    if n == 0:
        return {}
    vmax_list = list(vmax) if not isinstance(vmax, (int, float)) else [int(vmax)] * n
    if ncap is None:
        ncap_list = [0.0] * n
    else:
        ncap_list = list(ncap) if not isinstance(ncap, (int, float)) else [float(ncap)] * n
    chi = chi_cm1 if chi_cm1 and len(chi_cm1) == n else [[0.0 for _ in range(n)] for _ in range(n)]

    def v_eff(v: int, cap: float) -> float:
        if cap is None or cap <= 0.0:
            return float(v)
        return cap * math.erf(float(v) / cap)

    def energy_cm1(vs: list[int]) -> float:
        ve = [v_eff(vs[i], ncap_list[i]) for i in range(n)]
        value = sum(omega_cm1[i] * ve[i] for i in range(n))
        for i in range(n):
            for j in range(i, n):
                value += chi[i][j] * ve[i] * ve[j]
        return value

    dos: dict[int, float] = {}

    def rec(i: int, vlist: list[int]) -> None:
        if i == n:
            e = energy_cm1(vlist)
            if 0.0 <= e <= emax_cm1:
                bin_idx = int(e // bin_cm1)
                dos[bin_idx] = dos.get(bin_idx, 0.0) + 1.0
            return
        for v in range(int(vmax_list[i]) + 1):
            vlist[i] = v
            # Do not prune from a partially assigned vector: later entries may
            # still contain values left by a previous recursion branch and
            # anharmonic cross terms are not necessarily monotone.
            rec(i + 1, vlist)

    rec(0, [0] * n)
    return dos


def direct_vibrational_dos_from_xyzin(
    xyzin: Path | str,
    *,
    vmax: int = 6,
    emax_cm1: float = 8000.0,
    bin_cm1: float = 50.0,
    ncap: float = 10.0,
    emin_cm1: float = 0.0,
    t_k: float | None = None,
    out: Path | str | None = None,
    q_out: Path | str | None = None,
    number_out: Path | str | None = None,
    cache_path: Path | str | None = None,
) -> DOSResult:
    target = Path(xyzin)
    vib = read_vibrational_section(target)
    positive = tuple(index for index, value in enumerate(vib.frequencies_cm1) if value > 0.0)
    excluded = tuple(index + 1 for index, value in enumerate(vib.frequencies_cm1) if value < 0.0)
    freqs = [float(vib.frequencies_cm1[index]) for index in positive]
    chi = _filtered_chi_matrix(vib.chi_cm1, positive)
    T = float(t_k if t_k is not None else read_basic_section(target).temperature_K)
    out_path = None if out is None else Path(out)
    q_path = (
        Path(q_out)
        if q_out is not None
        else (target.parent / "vib_qt.dat" if out_path is not None else None)
    )
    number_path = (
        Path(number_out)
        if number_out is not None
        else (target.parent / "n_vib_ts.dat" if excluded and out_path is not None else None)
    )
    settings = {
        "algorithm": "direct-sum-v2-complete-enumeration",
        "vmax": vmax,
        "emax_cm1": emax_cm1,
        "emin_cm1": emin_cm1,
        "bin_cm1": bin_cm1,
        "ncap": ncap,
        "T_K": T,
        "positive_modes": positive,
    }
    cache = Path(cache_path) if cache_path is not None else None
    cache_key = _dos_cache_key(target, settings)
    if cache is not None and out_path is not None:
        cached = _load_dos_cache(cache)
        requested_outputs_exist = (
            (q_path is None or q_path.is_file())
            and (number_path is None or number_path.is_file())
        )
        if cached.get("key") == cache_key and out_path.is_file() and requested_outputs_exist:
            bins, cached_emin, cached_bin = read_dos_binned(out_path)
            if bins:
                q_value = q_from_dos(_build_energy_logg_from_bins(bins, cached_bin, cached_emin), T)
                return DOSResult(
                    path=out_path,
                    bins_logg=bins,
                    emin_cm1=cached_emin,
                    bin_cm1=cached_bin,
                    Q_vib=q_value,
                    T_K=T,
                    q_path=q_path if q_path is not None and q_path.exists() else None,
                    number_path=number_path if number_path is not None and number_path.exists() else None,
                    is_transition_state=bool(excluded),
                    excluded_imaginary_modes=excluded,
                    cache_hit=True,
                )
    dos_counts = direct_sum_dos(freqs, chi, vmax, emax_cm1, bin_cm1, ncap)
    dos_logg = {
        bin_idx: math.log(count)
        for bin_idx, count in dos_counts.items()
        if count > 0.0 and emin_cm1 <= (bin_idx + 0.5) * bin_cm1 <= emax_cm1
    }
    if out_path is not None:
        write_dos(out_path, dos_logg, 0.0, bin_cm1)
    q_value = q_from_dos(_build_energy_logg_from_bins(dos_logg, bin_cm1, 0.0), T)
    if q_path is not None:
        q_path.parent.mkdir(parents=True, exist_ok=True)
        q_path.write_text("# T_K Q_vib\n" + f"{T:.6f} {q_value:.12e}\n", encoding="utf-8")
    if number_path is not None:
        write_dos(number_path, cumulative_number_of_states_log(dos_logg), 0.0, bin_cm1)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"key": cache_key, "settings": settings}, indent=2), encoding="utf-8")
    return DOSResult(
        path=out_path,
        bins_logg=dos_logg,
        emin_cm1=0.0,
        bin_cm1=bin_cm1,
        Q_vib=q_value,
        T_K=T,
        q_path=q_path,
        number_path=number_path,
        is_transition_state=bool(excluded),
        excluded_imaginary_modes=excluded,
    )


def rot_dos_logg(
    rotor_kind: str,
    A_MHz: float,
    B_MHz: float,
    C_MHz: float,
    sigma: int,
    emax_cm1: float,
    bin_cm1: float,
    *,
    jmax: int | None = None,
) -> dict[int, float]:
    A = float(A_MHz) * MHZ_TO_CM1
    B = float(B_MHz) * MHZ_TO_CM1
    C = float(C_MHz) * MHZ_TO_CM1
    sigma_eff = max(1, int(sigma)) if sigma else 1
    kind = _rotor_kind(rotor_kind) or str(rotor_kind).strip().lower()

    if kind == "linear":
        beff = max(B, C)
        if beff <= 0.0:
            return {}
        if jmax is None:
            jmax = int((math.sqrt(1.0 + 4.0 * emax_cm1 / beff) - 1.0) / 2.0) + 1
        counts: dict[int, float] = {}
        for J in range(jmax + 1):
            e = beff * J * (J + 1)
            if e > emax_cm1:
                break
            counts[int(e // bin_cm1)] = counts.get(int(e // bin_cm1), 0.0) + (2 * J + 1) / sigma_eff
        return {bin_idx: math.log(count) for bin_idx, count in counts.items() if count > 0.0}

    if kind == "spherical":
        beff = max(A, B, C)
        if beff <= 0.0:
            return {}
        if jmax is None:
            jmax = int((math.sqrt(1.0 + 4.0 * emax_cm1 / beff) - 1.0) / 2.0) + 1
        counts = {}
        for J in range(jmax + 1):
            e = beff * J * (J + 1)
            if e > emax_cm1:
                break
            counts[int(e // bin_cm1)] = (
                counts.get(int(e // bin_cm1), 0.0) + ((2 * J + 1) ** 2) / sigma_eff
            )
        return {bin_idx: math.log(count) for bin_idx, count in counts.items() if count > 0.0}

    Aeff = max(A, B, C)
    Beff = max(B, C)
    if Beff <= 0.0:
        return {}
    if jmax is None:
        jmax = int((math.sqrt(1.0 + 4.0 * emax_cm1 / Beff) - 1.0) / 2.0) + 1
    counts = {}
    for J in range(jmax + 1):
        base = Beff * J * (J + 1)
        if base > emax_cm1:
            break
        for K in range(J + 1):
            e = base + (Aeff - Beff) * K * K
            if e > emax_cm1:
                continue
            g = (2 * J + 1) * (1.0 if K == 0 else 2.0) / sigma_eff
            counts[int(e // bin_cm1)] = counts.get(int(e // bin_cm1), 0.0) + g
    return {bin_idx: math.log(count) for bin_idx, count in counts.items() if count > 0.0}


def rovib_pipeline(
    xyzin: Path | str,
    *,
    vib_dos: Path | str | None = None,
    out: Path | str | None = None,
    rot_out: Path | str | None = None,
    q_out: Path | str | None = None,
    t_k: float | None = None,
    emax_rot: float | None = None,
    jmax: int | None = None,
    sigma: int | None = None,
    number_out: Path | str | None = None,
) -> RovibDOSResult:
    target = Path(xyzin)
    workdir = target.parent
    vib_dos_path = workdir / "dos_vib.dat" if vib_dos is None else Path(vib_dos)
    out_path = workdir / "dos_rovib.dat" if out is None else Path(out)
    q_path = workdir / "rovib_qt.dat" if q_out is None else Path(q_out)
    vib_section = read_vibrational_section(target)
    is_transition_state = bool(vib_section.n_imag_like or 0) or any(
        frequency < 0.0 for frequency in vib_section.frequencies_cm1
    )
    number_path = (
        Path(number_out)
        if number_out is not None
        else (workdir / "n_rovib_ts.dat" if is_transition_state else None)
    )

    vib_bins, vib_emin, vib_bin = read_dos_binned(vib_dos_path)
    if not vib_bins:
        raise ValueError("rovib_pipeline: vibrational DOS is empty")
    energies = sorted(vib_bins.keys())
    vib_emax = vib_emin + (energies[-1] + 0.5) * vib_bin
    rot = read_rotational_section(target)
    deltabvib = read_deltabvib_section(target) if has_section(target, "DELTABVIB") else None
    A, B, C = resolved_rotational_constants_MHz(
        rot, state="ground_state", deltabvib=deltabvib
    )
    if A is None or B is None or C is None:
        raise ValueError("rovib_pipeline requires A_MHz, B_MHz and C_MHz in #ROTATIONAL")
    rotor_type = rot.rotor_type
    if not rotor_type:
        raise ValueError("rovib_pipeline requires rotor_type in #ROTATIONAL")
    sigma_eff = sigma if sigma is not None else (rot.symmetry_number or 1)
    rot_logg = rot_dos_logg(
        rotor_type,
        A,
        B,
        C,
        sigma_eff,
        vib_emax if emax_rot is None and jmax is None else float(emax_rot or vib_emax),
        vib_bin,
        jmax=jmax,
    )
    rot_path = None if rot_out is None else Path(rot_out)
    if rot_path is not None:
        write_dos(rot_path, rot_logg, 0.0, vib_bin)
    rovib_logg = convolve_log_dos(vib_bins, rot_logg)
    write_dos(out_path, rovib_logg, vib_emin, vib_bin)
    T = float(t_k if t_k is not None else read_basic_section(target).temperature_K)
    q_rovib = q_from_dos(_build_energy_logg_from_bins(rovib_logg, vib_bin, vib_emin), T)
    if q_path is not None:
        q_path.parent.mkdir(parents=True, exist_ok=True)
        q_path.write_text("# T_K Q_rovib\n" + f"{T:.6f} {q_rovib:.12e}\n", encoding="utf-8")
    if number_path is not None:
        write_dos(number_path, cumulative_number_of_states_log(rovib_logg), vib_emin, vib_bin)
    return RovibDOSResult(
        dos_vib=vib_dos_path,
        dos_rot=rot_path,
        dos_rovib=out_path,
        q_path=q_path,
        Q_rovib=q_rovib,
        T_K=T,
        bin_cm1=vib_bin,
        vib_emin_cm1=vib_emin,
        vib_emax_cm1=vib_emax,
        number_path=number_path,
    )


def write_dos(
    path: Path | str, dos_logg: dict[int, float], emin_cm1: float, bin_cm1: float
) -> Path:
    target = Path(path)
    lines = ["# format: E_cm1 log_g"]
    for bin_idx in sorted(dos_logg):
        e = emin_cm1 + (bin_idx + 0.5) * bin_cm1
        lines.append(f"{e:.6f}  {dos_logg[bin_idx]:.12f}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return target


def read_dos_binned(path: Path | str) -> tuple[dict[int, float], float, float]:
    target = Path(path)
    if not target.exists():
        return {}, 0.0, 1.0
    data: list[tuple[float, float]] = []
    for raw in target.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        if len(parts) < 2:
            continue
        data.append((float(parts[0].replace("D", "E")), float(parts[1].replace("D", "E"))))
    if not data:
        return {}, 0.0, 1.0
    data.sort(key=lambda item: item[0])
    bin_cm1 = max(1.0e-12, data[1][0] - data[0][0]) if len(data) >= 2 else 1.0
    emin = data[0][0] - 0.5 * bin_cm1
    return {int(round((e - emin) / bin_cm1 - 0.5)): lg for e, lg in data}, emin, bin_cm1


def convolve_log_dos(dos1_logg: dict[int, float], dos2_logg: dict[int, float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for b1, lg1 in dos1_logg.items():
        for b2, lg2 in dos2_logg.items():
            key = b1 + b2
            out[key] = _logsumexp(out.get(key), lg1 + lg2)
    return out


def q_from_dos(dos_e_logg: dict[float, float], t_k: float) -> float:
    if t_k <= 0:
        return 0.0
    beta = HC_OVER_KB / t_k
    acc = None
    for energy, logg in dos_e_logg.items():
        acc = _logsumexp(acc, logg - beta * float(energy))
    return 0.0 if acc is None else math.exp(acc)


def cumulative_number_of_states_log(dos_logg: dict[int, float]) -> dict[int, float]:
    out: dict[int, float] = {}
    accumulator = None
    for bin_idx in sorted(dos_logg):
        accumulator = _logsumexp(accumulator, dos_logg[bin_idx])
        out[bin_idx] = accumulator
    return out


def _chi_matrix(n: int, chi_rows: tuple[tuple[int, int, float], ...]) -> list[list[float]]:
    chi = [[0.0 for _ in range(n)] for _ in range(n)]
    for i, j, value in chi_rows:
        if 1 <= i <= n and 1 <= j <= n:
            chi[i - 1][j - 1] = float(value)
            chi[j - 1][i - 1] = float(value)
    return chi


def _filtered_chi_matrix(
    chi_rows: tuple[tuple[int, int, float], ...],
    retained_zero_based: tuple[int, ...],
) -> list[list[float]]:
    remap = {old + 1: new for new, old in enumerate(retained_zero_based)}
    chi = [[0.0 for _ in retained_zero_based] for _ in retained_zero_based]
    for i, j, value in chi_rows:
        if i in remap and j in remap:
            ni = remap[i]
            nj = remap[j]
            chi[ni][nj] = float(value)
            chi[nj][ni] = float(value)
    return chi


def _dos_cache_key(path: Path, settings: dict[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    digest.update(b"\0")
    digest.update(json.dumps(settings, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def _load_dos_cache(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _rotor_kind(rotor_type: str | None) -> str | None:
    if not rotor_type:
        return None
    text = str(rotor_type).strip().lower()
    if "linear" in text:
        return "linear"
    if "spherical" in text:
        return "spherical"
    if "symmetric" in text:
        return "symmetric"
    if "asymmetric" in text:
        return "asymmetric"
    return None


def _logsumexp(a: float | None, b: float) -> float:
    if a is None:
        return b
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _build_energy_logg_from_bins(
    dos_logg: dict[int, float],
    bin_cm1: float,
    emin_cm1: float,
) -> dict[float, float]:
    return {emin_cm1 + (bin_idx + 0.5) * bin_cm1: logg for bin_idx, logg in dos_logg.items()}
