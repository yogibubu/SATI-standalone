from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen
import csv
import math
import re

import numpy as np

from .contracts import VibrationalSection, read_vibrational_section


NIST_WEBBOOK_BASE = "https://webbook.nist.gov"
NIST_USER_AGENT = "ORACLE-MATRIX/0.1"
VIBRATIONAL_OBSERVABLES = ("IR", "RAMAN", "VCD", "ROA")
VIBRATIONAL_SOURCES = ("harmonic", "anharmonic", "hybrid")


@dataclass(frozen=True)
class VibrationalPeak:
    mode: int
    frequency_cm1: float
    intensity: float
    observable: str
    source: str


@dataclass(frozen=True)
class VibrationalSpectrumOptions:
    fwhm_cm1: float = 10.0
    step_cm1: float = 1.0
    lineshape: str = "gaussian"
    normalize: bool = True
    x_min_cm1: float | None = None
    x_max_cm1: float | None = None


@dataclass(frozen=True)
class VibrationalSpectrum:
    observable: str
    source: str
    x_cm1: np.ndarray
    y: np.ndarray
    peaks: tuple[VibrationalPeak, ...]
    y_label: str


@dataclass(frozen=True)
class VibrationalSpectrumComparison:
    first: VibrationalSpectrum
    second: VibrationalSpectrum
    x_cm1: np.ndarray
    first_y: np.ndarray
    second_y: np.ndarray
    plotted_second_y: np.ndarray
    mirror_second: bool


@dataclass(frozen=True)
class NormalModeMatch:
    level1_mode: int
    level2_mode: int
    overlap: float
    level1_harmonic_cm1: float
    level2_harmonic_cm1: float
    level2_anharmonic_cm1: float
    correction_cm1: float
    hybrid_cm1: float


@dataclass(frozen=True)
class HybridVibrationalSpectrumResult:
    spectrum: VibrationalSpectrum
    matches: tuple[NormalModeMatch, ...]


@dataclass(frozen=True)
class DegenerateModeBlock:
    level1_modes: tuple[int, ...]
    level2_modes: tuple[int, ...]
    principal_overlaps: tuple[float, ...]
    procrustes_rotation: np.ndarray
    ambiguous: bool


@dataclass(frozen=True)
class RobustModeMatchResult:
    matches: tuple[tuple[int, int, float], ...]
    blocks: tuple[DegenerateModeBlock, ...]
    minimum_principal_overlap: float
    ambiguity_rejected: bool


@dataclass(frozen=True)
class NISTIRPoint:
    wavenumber_cm1: float
    value: float


@dataclass(frozen=True)
class NISTIRDownloadResult:
    status: str
    message: str
    identifier: str
    page_url: str
    jcamp_url: str = ""
    state: str = ""
    csv_path: Path | None = None
    points: tuple[NISTIRPoint, ...] = ()

    @property
    def needs_user_instruction(self) -> bool:
        return self.status in {"not_gas_phase", "not_found", "parse_error"}


def vibrational_peaks_from_section(
    section: VibrationalSection,
    *,
    observable: str = "IR",
    source: str = "harmonic",
) -> tuple[VibrationalPeak, ...]:
    obs = _normalize_observable(observable)
    src = _normalize_source(source)
    frequencies = _frequencies_for_source(section, src)
    intensities = _intensities_for_observable(section, obs, src)
    if not frequencies:
        raise ValueError(f"no {src} vibrational frequencies are available")
    if not intensities:
        raise ValueError(f"no {src} {obs} intensities are available")
    if len(intensities) != len(frequencies):
        raise ValueError(
            f"{src} {obs} intensity count ({len(intensities)}) does not match "
            f"frequency count ({len(frequencies)})"
        )
    return tuple(
        VibrationalPeak(
            mode=idx,
            frequency_cm1=float(freq),
            intensity=float(intensity),
            observable=obs,
            source=src,
        )
        for idx, (freq, intensity) in enumerate(zip(frequencies, intensities), start=1)
    )


def build_vibrational_spectrum(
    section: VibrationalSection,
    *,
    observable: str = "IR",
    source: str = "harmonic",
    options: VibrationalSpectrumOptions | None = None,
) -> VibrationalSpectrum:
    opts = options or VibrationalSpectrumOptions()
    peaks = vibrational_peaks_from_section(section, observable=observable, source=source)
    if opts.fwhm_cm1 <= 0.0:
        raise ValueError("fwhm_cm1 must be positive")
    if opts.step_cm1 <= 0.0:
        raise ValueError("step_cm1 must be positive")
    centers = np.asarray([peak.frequency_cm1 for peak in peaks], dtype=float)
    x_min = (
        float(opts.x_min_cm1)
        if opts.x_min_cm1 is not None
        else max(0.0, centers.min() - 5.0 * opts.fwhm_cm1)
    )
    x_max = (
        float(opts.x_max_cm1) if opts.x_max_cm1 is not None else centers.max() + 5.0 * opts.fwhm_cm1
    )
    if x_max <= x_min:
        raise ValueError("x_max_cm1 must be greater than x_min_cm1")
    x = np.arange(x_min, x_max + 0.5 * opts.step_cm1, opts.step_cm1, dtype=float)
    y = np.zeros_like(x)
    for peak in peaks:
        y += peak.intensity * _profile(x, peak.frequency_cm1, opts.fwhm_cm1, opts.lineshape)
    if opts.normalize:
        scale = float(np.max(np.abs(y))) if y.size else 0.0
        if scale > 0.0:
            y = y / scale
    return VibrationalSpectrum(
        observable=_normalize_observable(observable),
        source=_normalize_source(source),
        x_cm1=x,
        y=y,
        peaks=peaks,
        y_label="normalized intensity" if opts.normalize else _y_label(observable),
    )


def build_vibrational_spectrum_from_xyzin(
    xyzin: Path | str,
    *,
    observable: str = "IR",
    source: str = "harmonic",
    options: VibrationalSpectrumOptions | None = None,
) -> VibrationalSpectrum:
    return build_vibrational_spectrum(
        read_vibrational_section(Path(xyzin)),
        observable=observable,
        source=source,
        options=options,
    )


def write_vibrational_spectrum_csv(path: Path | str, spectrum: VibrationalSpectrum) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wavenumber_cm-1", spectrum.y_label, "observable", "source"])
        for x, y in zip(spectrum.x_cm1, spectrum.y):
            writer.writerow(
                [f"{float(x):.8f}", f"{float(y):.12g}", spectrum.observable, spectrum.source]
            )
    return target


def write_vibrational_peak_csv(path: Path | str, peaks: tuple[VibrationalPeak, ...]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "frequency_cm-1", "intensity", "observable", "source"])
        for peak in peaks:
            writer.writerow(
                [
                    peak.mode,
                    f"{peak.frequency_cm1:.8f}",
                    f"{peak.intensity:.12g}",
                    peak.observable,
                    peak.source,
                ]
            )
    return target


def write_vibrational_spectrum_plot(path: Path | str, spectrum: VibrationalSpectrum) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    ax.plot(spectrum.x_cm1, spectrum.y, color="#225ea8", linewidth=1.4)
    ax.axhline(0.0, color="#4d4d4d", linewidth=0.6)
    ax.set_xlabel("Wavenumber / cm$^{-1}$")
    ax.set_ylabel(spectrum.y_label)
    ax.set_title(f"{spectrum.source.capitalize()} {spectrum.observable} spectrum")
    ax.invert_xaxis()
    fig.savefig(target)
    plt.close(fig)
    return target


def write_vibrational_spectrum_outputs(
    xyzin: Path | str,
    *,
    csv_path: Path | str,
    plot_path: Path | str | None = None,
    peaks_path: Path | str | None = None,
    level2_xyzin: Path | str | None = None,
    mode_match_csv_path: Path | str | None = None,
    observable: str = "IR",
    source: str = "harmonic",
    min_mode_overlap: float = 0.70,
    options: VibrationalSpectrumOptions | None = None,
) -> VibrationalSpectrum:
    if _normalize_source(source) == "hybrid":
        if level2_xyzin is None:
            raise ValueError("hybrid vibrational spectra require level2_xyzin")
        result = build_hybrid_vibrational_spectrum_from_xyzin(
            xyzin,
            level2_xyzin,
            observable=observable,
            options=options,
            min_mode_overlap=min_mode_overlap,
        )
        spectrum = result.spectrum
        if mode_match_csv_path is not None:
            write_normal_mode_match_csv(mode_match_csv_path, result.matches)
    else:
        spectrum = build_vibrational_spectrum_from_xyzin(
            xyzin,
            observable=observable,
            source=source,
            options=options,
        )
    write_vibrational_spectrum_csv(csv_path, spectrum)
    if peaks_path is not None:
        write_vibrational_peak_csv(peaks_path, spectrum.peaks)
    if plot_path is not None:
        write_vibrational_spectrum_plot(plot_path, spectrum)
    return spectrum


def match_normal_modes(
    level1_modes: np.ndarray,
    level2_modes: np.ndarray,
    *,
    min_overlap: float = 0.70,
) -> tuple[tuple[int, int, float], ...]:
    first = _normalized_mode_rows(level1_modes)
    second = _normalized_mode_rows(level2_modes)
    if first.shape != second.shape:
        raise ValueError(
            "normal-mode matching requires equal mode and coordinate counts "
            f"(got {first.shape} and {second.shape})"
        )
    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError("min_overlap must be between 0 and 1")
    overlap = np.abs(first @ second.T)
    rows, cols = _linear_sum_assignment(-overlap)
    matches = tuple(
        sorted(
            (
                (int(row) + 1, int(col) + 1, float(overlap[row, col]))
                for row, col in zip(rows, cols)
            ),
            key=lambda item: item[0],
        )
    )
    weak = [item for item in matches if item[2] < min_overlap]
    if weak:
        text = ", ".join(
            f"L1 mode {level1}->L2 mode {level2} overlap={value:.3f}"
            for level1, level2, value in weak
        )
        raise ValueError(f"normal-mode correspondence below threshold: {text}")
    return matches


def match_normal_modes_robust(
    level1_modes: np.ndarray,
    level2_modes: np.ndarray,
    level1_frequencies_cm1: tuple[float, ...] | list[float] | np.ndarray,
    *,
    degeneracy_tolerance_cm1: float = 5.0,
    min_principal_overlap: float = 0.70,
) -> RobustModeMatchResult:
    """Match modes with Procrustes alignment inside quasi-degenerate subspaces."""

    first = _normalized_mode_rows(level1_modes)
    second = _normalized_mode_rows(level2_modes)
    frequencies = np.asarray(level1_frequencies_cm1, dtype=float)
    if first.shape != second.shape or frequencies.shape != (first.shape[0],):
        raise ValueError("robust mode matching requires equal mode shapes and one L1 frequency per mode")
    if degeneracy_tolerance_cm1 < 0.0 or not 0.0 <= min_principal_overlap <= 1.0:
        raise ValueError("invalid degeneracy or principal-overlap threshold")

    raw = np.abs(first @ second.T)
    rows, cols = _linear_sum_assignment(-raw)
    initial = {int(row): int(col) for row, col in zip(rows, cols)}
    blocks: list[DegenerateModeBlock] = []
    matches: list[tuple[int, int, float]] = []
    minimum = 1.0
    rejected = False
    for indices in _frequency_blocks(frequencies, float(degeneracy_tolerance_cm1)):
        mapped = tuple(initial[index] for index in indices)
        first_block = first[np.asarray(indices)]
        second_block = second[np.asarray(mapped)]
        cross = first_block @ second_block.T
        left, singular, right_t = np.linalg.svd(cross, full_matrices=False)
        rotation = left @ right_t
        rotated = rotation @ second_block
        aligned_overlap = np.abs(first_block @ rotated.T)
        local_rows, local_cols = _linear_sum_assignment(-aligned_overlap)
        block_minimum = float(np.min(singular)) if singular.size else 0.0
        minimum = min(minimum, block_minimum)
        ambiguous = block_minimum < min_principal_overlap
        rejected = rejected or ambiguous
        for local_row, local_col in zip(local_rows, local_cols):
            matches.append(
                (
                    indices[int(local_row)] + 1,
                    mapped[int(local_col)] + 1,
                    float(aligned_overlap[local_row, local_col]),
                )
            )
        blocks.append(
            DegenerateModeBlock(
                level1_modes=tuple(index + 1 for index in indices),
                level2_modes=tuple(index + 1 for index in mapped),
                principal_overlaps=tuple(float(value) for value in singular),
                procrustes_rotation=rotation,
                ambiguous=ambiguous,
            )
        )
    if rejected:
        bad = [
            "/".join(str(value) for value in block.level1_modes)
            for block in blocks
            if block.ambiguous
        ]
        raise ValueError(
            "normal-mode subspace correspondence below threshold for L1 block(s): "
            + ", ".join(bad)
        )
    return RobustModeMatchResult(
        matches=tuple(sorted(matches)),
        blocks=tuple(blocks),
        minimum_principal_overlap=minimum,
        ambiguity_rejected=False,
    )


def _frequency_blocks(frequencies: np.ndarray, tolerance: float) -> tuple[tuple[int, ...], ...]:
    order = np.argsort(frequencies)
    blocks: list[list[int]] = []
    for raw_index in order:
        index = int(raw_index)
        if not blocks:
            blocks.append([index])
            continue
        previous = blocks[-1][-1]
        if abs(float(frequencies[index] - frequencies[previous])) <= tolerance:
            blocks[-1].append(index)
        else:
            blocks.append([index])
    return tuple(tuple(sorted(block)) for block in blocks)


def hybrid_vibrational_section_from_sections(
    level1: VibrationalSection,
    level2: VibrationalSection,
    matches: tuple[tuple[int, int, float], ...],
) -> tuple[VibrationalSection, tuple[NormalModeMatch, ...]]:
    if not level1.frequencies_cm1:
        raise ValueError("level1 #VIBRATIONAL lacks harmonic frequencies")
    if not level2.frequencies_cm1:
        raise ValueError("level2 #VIBRATIONAL lacks harmonic frequencies")
    if not level2.anharmonic_frequencies_cm1:
        raise ValueError("level2 #VIBRATIONAL lacks anharmonic frequencies")
    mode_count = len(level1.frequencies_cm1)
    if len(matches) != mode_count:
        raise ValueError("normal-mode match count does not match level1 vibrational mode count")
    if (
        len(level2.frequencies_cm1) < mode_count
        or len(level2.anharmonic_frequencies_cm1) < mode_count
    ):
        raise ValueError("level2 harmonic/anharmonic frequency counts are incomplete")

    hybrid_freqs = [0.0] * mode_count
    detailed: list[NormalModeMatch] = []
    for level1_mode, level2_mode, overlap in matches:
        i = level1_mode - 1
        j = level2_mode - 1
        level1_harm = float(level1.frequencies_cm1[i])
        level2_harm = float(level2.frequencies_cm1[j])
        level2_anh = float(level2.anharmonic_frequencies_cm1[j])
        correction = level2_anh - level2_harm
        hybrid = level1_harm + correction
        hybrid_freqs[i] = hybrid
        detailed.append(
            NormalModeMatch(
                level1_mode=level1_mode,
                level2_mode=level2_mode,
                overlap=float(overlap),
                level1_harmonic_cm1=level1_harm,
                level2_harmonic_cm1=level2_harm,
                level2_anharmonic_cm1=level2_anh,
                correction_cm1=correction,
                hybrid_cm1=hybrid,
            )
        )
    section = VibrationalSection(
        linear=level1.linear,
        nvib=level1.nvib,
        n_imag_like=level1.n_imag_like,
        symmetry_group=level1.symmetry_group,
        frequencies_cm1=tuple(hybrid_freqs),
        ir_intensities_km_mol=level1.ir_intensities_km_mol,
        raman_activities_A4_amu=level1.raman_activities_A4_amu,
        vcd_rot_strengths=level1.vcd_rot_strengths,
        roa_intensities=level1.roa_intensities,
    )
    return section, tuple(detailed)


def build_hybrid_vibrational_spectrum_from_xyzin(
    level1_xyzin: Path | str,
    level2_xyzin: Path | str,
    *,
    observable: str = "IR",
    options: VibrationalSpectrumOptions | None = None,
    min_mode_overlap: float = 0.70,
) -> HybridVibrationalSpectrumResult:
    from matrix_qm import read_normal_modes_section

    level1_path = Path(level1_xyzin)
    level2_path = Path(level2_xyzin)
    matches = match_normal_modes(
        read_normal_modes_section(level1_path).modes,
        read_normal_modes_section(level2_path).modes,
        min_overlap=min_mode_overlap,
    )
    section, detailed = hybrid_vibrational_section_from_sections(
        read_vibrational_section(level1_path),
        read_vibrational_section(level2_path),
        matches,
    )
    spectrum = build_vibrational_spectrum(
        section,
        observable=observable,
        source="hybrid",
        options=options,
    )
    return HybridVibrationalSpectrumResult(spectrum=spectrum, matches=detailed)


def write_normal_mode_match_csv(
    path: Path | str,
    matches: tuple[NormalModeMatch, ...],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "level1_mode",
                "level2_mode",
                "overlap_abs",
                "level1_harmonic_cm-1",
                "level2_harmonic_cm-1",
                "level2_anharmonic_cm-1",
                "correction_cm-1",
                "hybrid_cm-1",
            ]
        )
        for match in matches:
            writer.writerow(
                [
                    match.level1_mode,
                    match.level2_mode,
                    f"{match.overlap:.12g}",
                    f"{match.level1_harmonic_cm1:.8f}",
                    f"{match.level2_harmonic_cm1:.8f}",
                    f"{match.level2_anharmonic_cm1:.8f}",
                    f"{match.correction_cm1:.8f}",
                    f"{match.hybrid_cm1:.8f}",
                ]
            )
    return target


def compare_vibrational_spectra(
    first: VibrationalSpectrum,
    second: VibrationalSpectrum,
    *,
    mirror_second: bool | None = None,
) -> VibrationalSpectrumComparison:
    if first.x_cm1.size == 0 or second.x_cm1.size == 0:
        raise ValueError("cannot compare empty spectra")
    auto_mirror = _should_mirror_second(first.observable, second.observable)
    mirror = auto_mirror if mirror_second is None else bool(mirror_second)
    x_min = max(float(np.min(first.x_cm1)), float(np.min(second.x_cm1)))
    x_max = min(float(np.max(first.x_cm1)), float(np.max(second.x_cm1)))
    if x_max <= x_min:
        raise ValueError("spectra do not overlap in wavenumber")
    step = min(_grid_step(first.x_cm1), _grid_step(second.x_cm1))
    x = np.arange(x_min, x_max + 0.5 * step, step, dtype=float)
    first_y = np.interp(x, first.x_cm1, first.y)
    second_y = np.interp(x, second.x_cm1, second.y)
    plotted_second = -second_y if mirror else second_y
    return VibrationalSpectrumComparison(
        first=first,
        second=second,
        x_cm1=x,
        first_y=first_y,
        second_y=second_y,
        plotted_second_y=plotted_second,
        mirror_second=mirror,
    )


def write_vibrational_spectrum_comparison_csv(
    path: Path | str,
    comparison: VibrationalSpectrumComparison,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "wavenumber_cm-1",
                "first_y",
                "second_y",
                "plotted_second_y",
                "first_observable",
                "first_source",
                "second_observable",
                "second_source",
                "mirror_second",
            ]
        )
        for x, first_y, second_y, plotted_second_y in zip(
            comparison.x_cm1,
            comparison.first_y,
            comparison.second_y,
            comparison.plotted_second_y,
        ):
            writer.writerow(
                [
                    f"{float(x):.8f}",
                    f"{float(first_y):.12g}",
                    f"{float(second_y):.12g}",
                    f"{float(plotted_second_y):.12g}",
                    comparison.first.observable,
                    comparison.first.source,
                    comparison.second.observable,
                    comparison.second.source,
                    int(comparison.mirror_second),
                ]
            )
    return target


def write_vibrational_spectrum_comparison_plot(
    path: Path | str,
    comparison: VibrationalSpectrumComparison,
    *,
    first_label: str | None = None,
    second_label: str | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    first_text = first_label or f"{comparison.first.source} {comparison.first.observable}"
    second_text = second_label or f"{comparison.second.source} {comparison.second.observable}"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.plot(comparison.x_cm1, comparison.first_y, color="#225ea8", linewidth=1.4, label=first_text)
    ax.plot(
        comparison.x_cm1,
        comparison.plotted_second_y,
        color="#c44e52",
        linewidth=1.4,
        label=second_text,
    )
    ax.axhline(0.0, color="#4d4d4d", linewidth=0.7)
    ax.set_xlabel("Wavenumber / cm$^{-1}$")
    ax.set_ylabel("normalized intensity")
    ax.set_title("Vibrational spectrum comparison")
    ax.legend(frameon=False)
    ax.invert_xaxis()
    fig.savefig(target)
    plt.close(fig)
    return target


def write_vibrational_spectrum_comparison_outputs(
    xyzin: Path | str,
    *,
    csv_path: Path | str,
    plot_path: Path | str | None = None,
    second_xyzin: Path | str | None = None,
    observable: str = "IR",
    first_source: str = "harmonic",
    second_source: str = "anharmonic",
    options: VibrationalSpectrumOptions | None = None,
    mirror_second: bool | None = None,
    min_mode_overlap: float = 0.70,
    mode_match_csv_path: Path | str | None = None,
) -> VibrationalSpectrumComparison:
    second_target = xyzin if second_xyzin is None else second_xyzin
    first = _comparison_spectrum(
        xyzin,
        second_target,
        observable=observable,
        source=first_source,
        options=options,
        min_mode_overlap=min_mode_overlap,
        mode_match_csv_path=mode_match_csv_path,
    )
    second = _comparison_spectrum(
        second_target,
        second_target,
        level1_xyzin=xyzin,
        observable=observable,
        source=second_source,
        options=options,
        min_mode_overlap=min_mode_overlap,
        mode_match_csv_path=mode_match_csv_path,
    )
    comparison = compare_vibrational_spectra(first, second, mirror_second=mirror_second)
    write_vibrational_spectrum_comparison_csv(csv_path, comparison)
    if plot_path is not None:
        write_vibrational_spectrum_comparison_plot(plot_path, comparison)
    return comparison


def nist_ir_points_to_spectrum(
    points: tuple[NISTIRPoint, ...],
    *,
    source: str = "nist-gas-experiment",
    normalize: bool = True,
) -> VibrationalSpectrum:
    if not points:
        raise ValueError("NIST IR point list is empty")
    x = np.asarray([point.wavenumber_cm1 for point in points], dtype=float)
    transmittance = np.asarray([point.value for point in points], dtype=float)
    y = 1.0 - transmittance
    if normalize:
        scale = float(np.max(np.abs(y))) if y.size else 0.0
        if scale > 0.0:
            y = y / scale
    return VibrationalSpectrum(
        observable="IR",
        source=source,
        x_cm1=x,
        y=y,
        peaks=(),
        y_label="normalized absorbance" if normalize else "absorbance proxy",
    )


def fetch_nist_ir_gas_phase_csv(
    identifier: str,
    csv_path: Path | str,
    *,
    index: int = 1,
    timeout: float = 20.0,
) -> NISTIRDownloadResult:
    page_url = _nist_ir_page_url(identifier, index=index)
    page = _fetch_text(page_url, timeout=timeout)
    jcamp_url = _find_jcamp_url(page, page_url)
    if not jcamp_url:
        return NISTIRDownloadResult(
            status="not_found",
            message="NIST IR spectrum not found; user instruction is required",
            identifier=identifier,
            page_url=page_url,
        )
    jcamp = _fetch_text(jcamp_url, timeout=timeout, encoding="latin1")
    metadata = _jcamp_metadata(jcamp)
    state = metadata.get("STATE", "").strip()
    if not state.upper().startswith("GAS"):
        return NISTIRDownloadResult(
            status="not_gas_phase",
            message=f"NIST IR spectrum is not gas phase (state={state or 'unknown'})",
            identifier=identifier,
            page_url=page_url,
            jcamp_url=jcamp_url,
            state=state,
        )
    points = parse_nist_jcamp_ir_points(jcamp)
    if not points:
        return NISTIRDownloadResult(
            status="parse_error",
            message="NIST JCAMP spectrum could not be converted to numeric CSV",
            identifier=identifier,
            page_url=page_url,
            jcamp_url=jcamp_url,
            state=state,
        )
    target = write_nist_ir_csv(csv_path, points, state=state, source_url=jcamp_url)
    return NISTIRDownloadResult(
        status="downloaded",
        message=f"Downloaded NIST gas-phase IR spectrum ({len(points)} points)",
        identifier=identifier,
        page_url=page_url,
        jcamp_url=jcamp_url,
        state=state,
        csv_path=target,
        points=points,
    )


def parse_nist_jcamp_ir_points(text: str) -> tuple[NISTIRPoint, ...]:
    metadata = _jcamp_metadata(text)
    xfactor = float(metadata.get("XFACTOR", "1") or "1")
    yfactor = float(metadata.get("YFACTOR", "1") or "1")
    deltax = float(metadata.get("DELTAX", "0") or "0") * xfactor
    points: list[NISTIRPoint] = []
    in_xy = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("##"):
            key = line[2:].split("=", 1)[0].strip().upper()
            in_xy = key == "XYDATA"
            if key == "END":
                break
            continue
        if not in_xy:
            continue
        numbers = [float(item) for item in _NUMBER_RE.findall(line)]
        if len(numbers) < 2:
            continue
        x0 = numbers[0] * xfactor
        for offset, y_raw in enumerate(numbers[1:]):
            x = x0 + offset * deltax if deltax else x0
            points.append(NISTIRPoint(wavenumber_cm1=x, value=y_raw * yfactor))
    return tuple(points)


def write_nist_ir_csv(
    path: Path | str,
    points: tuple[NISTIRPoint, ...],
    *,
    state: str,
    source_url: str,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["# source", source_url])
        writer.writerow(["# state", state])
        writer.writerow(["wavenumber_cm-1", "transmittance"])
        for point in points:
            writer.writerow([f"{point.wavenumber_cm1:.8f}", f"{point.value:.12g}"])
    return target


def _profile(x: np.ndarray, center: float, fwhm: float, lineshape: str) -> np.ndarray:
    kind = lineshape.strip().lower()
    if kind == "gaussian":
        sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        return np.exp(-0.5 * ((x - center) / sigma) ** 2)
    if kind == "lorentzian":
        gamma = 0.5 * fwhm
        return (gamma * gamma) / ((x - center) ** 2 + gamma * gamma)
    raise ValueError("lineshape must be 'gaussian' or 'lorentzian'")


def _should_mirror_second(first_observable: str, second_observable: str) -> bool:
    signed = {"VCD", "ROA"}
    return (
        _normalize_observable(first_observable) not in signed
        and _normalize_observable(second_observable) not in signed
    )


def _grid_step(x: np.ndarray) -> float:
    if x.size < 2:
        return 1.0
    diffs = np.diff(np.sort(x))
    positive = diffs[diffs > 0.0]
    if positive.size == 0:
        return 1.0
    return float(np.min(positive))


def _comparison_spectrum(
    xyzin: Path | str,
    level2_xyzin: Path | str,
    *,
    level1_xyzin: Path | str | None = None,
    observable: str,
    source: str,
    options: VibrationalSpectrumOptions | None,
    min_mode_overlap: float,
    mode_match_csv_path: Path | str | None,
) -> VibrationalSpectrum:
    src = _normalize_source(source)
    if src == "hybrid":
        result = build_hybrid_vibrational_spectrum_from_xyzin(
            xyzin if level1_xyzin is None else level1_xyzin,
            level2_xyzin,
            observable=observable,
            options=options,
            min_mode_overlap=min_mode_overlap,
        )
        if mode_match_csv_path is not None:
            write_normal_mode_match_csv(mode_match_csv_path, result.matches)
        return result.spectrum
    return build_vibrational_spectrum_from_xyzin(
        xyzin,
        observable=observable,
        source=source,
        options=options,
    )


def _frequencies_for_source(section: VibrationalSection, source: str) -> tuple[float, ...]:
    if source == "anharmonic":
        return section.anharmonic_frequencies_cm1
    return section.frequencies_cm1


def _intensities_for_observable(
    section: VibrationalSection,
    observable: str,
    source: str,
) -> tuple[float, ...]:
    if source == "anharmonic":
        anharmonic = {
            "IR": section.anharmonic_ir_intensities_km_mol,
            "RAMAN": section.anharmonic_raman_activities_A4_amu,
            "VCD": section.anharmonic_vcd_rot_strengths,
            "ROA": section.anharmonic_roa_intensities,
        }[observable]
        if anharmonic:
            return anharmonic
    return {
        "IR": section.ir_intensities_km_mol,
        "RAMAN": section.raman_activities_A4_amu,
        "VCD": section.vcd_rot_strengths,
        "ROA": section.roa_intensities,
    }[observable]


def _normalize_observable(value: str) -> str:
    observable = value.strip().upper()
    if observable not in VIBRATIONAL_OBSERVABLES:
        raise ValueError(f"unsupported vibrational observable: {value}")
    return observable


def _normalize_source(value: str) -> str:
    source = value.strip().lower()
    if source not in VIBRATIONAL_SOURCES:
        raise ValueError("source must be 'harmonic', 'anharmonic' or 'hybrid'")
    return source


def _normalized_mode_rows(modes: np.ndarray) -> np.ndarray:
    matrix = np.asarray(modes, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("normal-mode matrix must be two-dimensional")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("normal-mode matrix contains zero-norm rows")
    return matrix / norms[:, None]


def _linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)
    except Exception:
        return _linear_sum_assignment_greedy(cost)


def _linear_sum_assignment_greedy(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if cost.shape[0] != cost.shape[1]:
        raise ValueError("greedy normal-mode assignment fallback requires a square matrix")
    remaining_rows = set(range(cost.shape[0]))
    remaining_cols = set(range(cost.shape[1]))
    pairs: list[tuple[int, int]] = []
    while remaining_rows:
        row, col = min(
            ((row, col) for row in remaining_rows for col in remaining_cols),
            key=lambda item: float(cost[item[0], item[1]]),
        )
        pairs.append((row, col))
        remaining_rows.remove(row)
        remaining_cols.remove(col)
    rows, cols = zip(*pairs, strict=True)
    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


def _y_label(observable: str) -> str:
    return {
        "IR": "IR intensity / km mol$^{-1}$",
        "RAMAN": "Raman activity / A$^4$ amu$^{-1}$",
        "VCD": "VCD rotational strength",
        "ROA": "ROA intensity",
    }[_normalize_observable(observable)]


def _nist_ir_page_url(identifier: str, *, index: int) -> str:
    ident = identifier.strip()
    if re.fullmatch(r"C?\d+", ident, flags=re.IGNORECASE):
        nist_id = ident.upper() if ident.upper().startswith("C") else f"C{ident}"
        query = f"ID={quote(nist_id)}"
    elif re.fullmatch(r"\d{2,7}-\d{2}-\d", ident):
        query = f"ID={quote(ident)}"
    else:
        query = f"Name={quote(ident)}"
    return f"{NIST_WEBBOOK_BASE}/cgi/cbook.cgi?{query}&Index={int(index)}&Type=IR-SPEC"


def _fetch_text(url: str, *, timeout: float, encoding: str = "utf-8") -> str:
    request = Request(url, headers={"User-Agent": NIST_USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode(encoding, errors="replace")


def _find_jcamp_url(page: str, page_url: str) -> str:
    match = re.search(r'href="([^"]*JCAMP[^"]*)"', page)
    if not match:
        return ""
    return urljoin(page_url, match.group(1).replace("&amp;", "&"))


def _jcamp_metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("##") or "=" not in line:
            continue
        key, value = line[2:].split("=", 1)
        metadata[key.strip().upper()] = value.strip()
    return metadata


_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")
