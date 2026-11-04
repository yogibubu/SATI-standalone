from __future__ import annotations

from dataclasses import dataclass
import csv
from io import StringIO
from pathlib import Path

import numpy as np
from matrix_qm import quartic_force_field_from_qff_section, read_qff_section

from .gaussian_qff import anharmonic_input_from_gaussian_fchk, read_indexed_qff_text
from .validation import validate_force_field
from .vci import QuarticForceField, VCIOptions, force_field_from_anharmonic_input
from .vpt2 import VPT2VCIComparison, compare_vpt2_vci


@dataclass(frozen=True)
class VPT2VCIReport:
    force_field: QuarticForceField
    comparison: VPT2VCIComparison
    text: str


def load_force_field(
    fchk_path: Path | None = None,
    qff_path: Path | None = None,
    xyzin_path: Path | None = None,
) -> QuarticForceField:
    """Load a canonical MATRIX QFF using optional Gaussian/FCHK frequencies."""
    if xyzin_path is not None:
        return quartic_force_field_from_qff_section(read_qff_section(Path(xyzin_path)))
    frequencies = None
    anharmonic_input = None
    if fchk_path is not None:
        anharmonic_input = anharmonic_input_from_gaussian_fchk(Path(fchk_path))
        frequencies = (
            anharmonic_input.anharmonic_frequencies_cm
            if anharmonic_input.anharmonic_frequencies_cm.size
            else anharmonic_input.harmonic_frequencies_cm
        )
    if qff_path is not None:
        return read_indexed_qff_text(Path(qff_path), frequencies)
    if anharmonic_input is not None:
        return force_field_from_anharmonic_input(anharmonic_input)
    raise FileNotFoundError("Provide an existing FCHK file or indexed QFF text file")


def run_vpt2_vci_report(
    force_field: QuarticForceField,
    *,
    max_quanta: int,
    roots: int,
    options: VCIOptions | None = None,
    vci_method: str = "dense",
) -> VPT2VCIReport:
    """Run VPT2/VCI on a canonical force field and return a formatted report."""
    validate_force_field(force_field)
    comparison = compare_vpt2_vci(
        force_field,
        max_quanta=max_quanta,
        n_roots=roots,
        options=options,
        vci_method=vci_method,
    )
    return VPT2VCIReport(force_field, comparison, format_vpt2_vci_report(force_field, comparison))


def format_vpt2_vci_report(qff: QuarticForceField, comparison: VPT2VCIComparison) -> str:
    lines = [
        "VPT2/VCI comparison on canonical MATRIX QFF",
        f"Modes used in input force field: {len(qff.harmonic_frequencies_cm)}",
        f"Cubic terms: {len(qff.cubic_cm)}",
        f"Quartic terms: {len(qff.quartic_cm)}",
        f"VCI basis size: {len(comparison.vci.basis)}",
        "Input harmonic frequencies (cm-1): "
        + ", ".join(f"{value:.3f}" for value in qff.harmonic_frequencies_cm),
        "",
        "Root     VPT2 abs      VCI abs        d_abs     VPT2 exc      VCI exc        d_exc",
    ]
    n = min(
        len(comparison.vpt2.energies_cm),
        len(comparison.vci.energies_cm),
        len(comparison.energy_differences_cm),
    )
    for idx in range(n):
        lines.append(
            f"{idx + 1:4d} "
            f"{comparison.vpt2.energies_cm[idx]:12.4f} "
            f"{comparison.vci.energies_cm[idx]:12.4f} "
            f"{comparison.energy_differences_cm[idx]:10.4f} "
            f"{comparison.vpt2.excitation_energies_cm[idx]:12.4f} "
            f"{comparison.vci.excitation_energies_cm[idx]:12.4f} "
            f"{comparison.excitation_differences_cm[idx]:10.4f}"
        )

    if comparison.vci.blocks:
        lines.extend(["", "Symmetry blocks:"])
        for block in comparison.vci.blocks:
            lines.append(
                f"  {block.label}: states={len(block.basis_indices)} roots={block.n_roots}"
            )

    if comparison.vci.state_contributions:
        lines.extend(["", "Dominant VCI contributions:"])
        for root, contribution in enumerate(comparison.vci.state_contributions[:n], start=1):
            pieces = [
                f"{state}:{coeff:+.3f}" for state, coeff in contribution.dominant_basis_states[:4]
            ]
            lines.append(
                f"  root {root:3d}: <n>={np.array2string(contribution.mode_quanta, precision=3)} "
                + ", ".join(pieces)
            )
    return "\n".join(lines)


def vpt2_vci_csv_tables(report: VPT2VCIReport) -> dict[str, str]:
    """Return CSV tables for VPT2/VCI comparison and dominant contributions."""
    qff = report.force_field
    comparison = report.comparison
    freq_rows = [["mode", "harmonic_frequency_cm-1"]]
    freq_rows.extend(
        [[idx, f"{freq:.10g}"] for idx, freq in enumerate(qff.harmonic_frequencies_cm, start=1)]
    )

    n = min(
        len(comparison.vpt2.energies_cm),
        len(comparison.vci.energies_cm),
        len(comparison.energy_differences_cm),
    )
    comparison_rows = [
        [
            "root",
            "vpt2_abs_cm-1",
            "vci_abs_cm-1",
            "delta_abs_cm-1",
            "vpt2_exc_cm-1",
            "vci_exc_cm-1",
            "delta_exc_cm-1",
        ]
    ]
    for idx in range(n):
        comparison_rows.append(
            [
                idx + 1,
                f"{comparison.vpt2.energies_cm[idx]:.10g}",
                f"{comparison.vci.energies_cm[idx]:.10g}",
                f"{comparison.energy_differences_cm[idx]:.10g}",
                f"{comparison.vpt2.excitation_energies_cm[idx]:.10g}",
                f"{comparison.vci.excitation_energies_cm[idx]:.10g}",
                f"{comparison.excitation_differences_cm[idx]:.10g}",
            ]
        )

    contribution_rows = [["root", "mode", "expected_quanta"]]
    for root, contribution in enumerate(comparison.vci.state_contributions[:n], start=1):
        for mode, quanta in enumerate(contribution.mode_quanta, start=1):
            contribution_rows.append([root, mode, f"{quanta:.10g}"])

    return {
        "frequencies.csv": _csv_text(freq_rows),
        "comparison.csv": _csv_text(comparison_rows),
        "mode_contributions.csv": _csv_text(contribution_rows),
    }


def write_csv_tables(
    report: VPT2VCIReport, outdir: Path, *, prefix: str = "vpt2_vci"
) -> dict[str, Path]:
    """Write structured CSV outputs for a VPT2/VCI report."""
    target_dir = Path(outdir)
    target_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, text in vpt2_vci_csv_tables(report).items():
        path = target_dir / f"{prefix}_{name}"
        path.write_text(text, encoding="utf-8")
        written[name] = path
    return written


def _csv_text(rows: list[list[object]]) -> str:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerows(rows)
    return stream.getvalue()
