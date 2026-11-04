from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .workflow import VPT2VCISection, read_vpt2_vci_section, write_vpt2_vci_section


@dataclass(frozen=True)
class VPT2VCIComparisonRow:
    root: int
    vpt2_abs_cm: float
    vci_abs_cm: float
    delta_abs_cm: float
    vpt2_exc_cm: float
    vci_exc_cm: float
    delta_exc_cm: float


@dataclass(frozen=True)
class VPT2VCIFrequencyRow:
    mode: int
    harmonic_frequency_cm: float


@dataclass(frozen=True)
class VPT2VCIModeContribution:
    root: int
    mode: int
    expected_quanta: float


@dataclass(frozen=True)
class VPT2VCIOutputSnapshot:
    section: VPT2VCISection
    status: str
    outputs: dict[str, Path]
    missing_outputs: tuple[str, ...]
    comparison: tuple[VPT2VCIComparisonRow, ...] = ()
    frequencies: tuple[VPT2VCIFrequencyRow, ...] = ()
    mode_contributions: tuple[VPT2VCIModeContribution, ...] = ()
    report_text: str = ""


PRIMARY_OUTPUTS = {
    "report": "vpt2_vci.report",
    "comparison": "vpt2_vci_comparison.csv",
    "frequencies": "vpt2_vci_frequencies.csv",
    "mode_contributions": "vpt2_vci_mode_contributions.csv",
}


def read_vpt2_vci_comparison(path: Path | str) -> tuple[VPT2VCIComparisonRow, ...]:
    rows = _read_csv_rows(Path(path))
    return tuple(
        VPT2VCIComparisonRow(
            root=int(_float_field(row, "root")),
            vpt2_abs_cm=_float_field(row, "vpt2_abs_cm-1", "vpt2_abs_cm1"),
            vci_abs_cm=_float_field(row, "vci_abs_cm-1", "vci_abs_cm1"),
            delta_abs_cm=_float_field(row, "delta_abs_cm-1", "delta_abs_cm1"),
            vpt2_exc_cm=_float_field(row, "vpt2_exc_cm-1", "vpt2_exc_cm1"),
            vci_exc_cm=_float_field(row, "vci_exc_cm-1", "vci_exc_cm1"),
            delta_exc_cm=_float_field(row, "delta_exc_cm-1", "delta_exc_cm1"),
        )
        for row in rows
    )


def read_vpt2_vci_frequencies(path: Path | str) -> tuple[VPT2VCIFrequencyRow, ...]:
    rows = _read_csv_rows(Path(path))
    return tuple(
        VPT2VCIFrequencyRow(
            mode=int(_float_field(row, "mode")),
            harmonic_frequency_cm=_float_field(
                row,
                "harmonic_frequency_cm-1",
                "harmonic_frequency_cm1",
                "harmonic_frequency_cm",
            ),
        )
        for row in rows
    )


def read_vpt2_vci_mode_contributions(path: Path | str) -> tuple[VPT2VCIModeContribution, ...]:
    rows = _read_csv_rows(Path(path))
    return tuple(
        VPT2VCIModeContribution(
            root=int(_float_field(row, "root")),
            mode=int(_float_field(row, "mode")),
            expected_quanta=_float_field(row, "expected_quanta"),
        )
        for row in rows
    )


def collect_vpt2_vci_outputs(section: VPT2VCISection) -> VPT2VCIOutputSnapshot:
    outputs = detected_vpt2_vci_outputs(section)
    comparison = read_vpt2_vci_comparison(outputs["comparison"]) if "comparison" in outputs else ()
    frequencies = (
        read_vpt2_vci_frequencies(outputs["frequencies"]) if "frequencies" in outputs else ()
    )
    mode_contributions = (
        read_vpt2_vci_mode_contributions(outputs["mode_contributions"])
        if "mode_contributions" in outputs
        else ()
    )
    report_text = outputs["report"].read_text(encoding="utf-8") if "report" in outputs else ""
    missing = tuple(name for name in PRIMARY_OUTPUTS if name not in outputs)
    status = "complete" if comparison and not missing else "partial" if outputs else "prepared"
    return VPT2VCIOutputSnapshot(
        section=section,
        status=status,
        outputs=outputs,
        missing_outputs=missing,
        comparison=comparison,
        frequencies=frequencies,
        mode_contributions=mode_contributions,
        report_text=report_text,
    )


def collect_vpt2_vci_outputs_from_xyzin(path: Path | str) -> VPT2VCIOutputSnapshot:
    return collect_vpt2_vci_outputs(read_vpt2_vci_section(Path(path)))


def refresh_vpt2_vci_section(path: Path | str) -> VPT2VCIOutputSnapshot:
    target = Path(path)
    snapshot = collect_vpt2_vci_outputs_from_xyzin(target)
    section = snapshot.section.with_outputs(snapshot.outputs, status=snapshot.status)
    write_vpt2_vci_section(target, section)
    return collect_vpt2_vci_outputs(section)


def vpt2_vci_output_summary_lines(snapshot: VPT2VCIOutputSnapshot) -> list[str]:
    lines = [
        f"status: {snapshot.status}",
        f"source_kind: {snapshot.section.source_kind}",
        f"run_dir: {snapshot.section.run_dir}",
        f"outputs: {len(snapshot.outputs)}",
        f"roots: {len(snapshot.comparison)}",
        f"frequencies: {len(snapshot.frequencies)}",
        f"mode_contributions: {len(snapshot.mode_contributions)}",
    ]
    if snapshot.missing_outputs:
        lines.append("missing_primary: " + ", ".join(snapshot.missing_outputs))
    for name, path in sorted(snapshot.outputs.items()):
        lines.append(f"output:{name}: {path}")
    return lines


def detected_vpt2_vci_outputs(section: VPT2VCISection) -> dict[str, Path]:
    outputs = {name: path for name, path in section.outputs.items() if path.exists()}
    for name, path in _candidate_output_paths(section).items():
        if path.exists():
            outputs.setdefault(name, path)
    return outputs


def _candidate_output_paths(section: VPT2VCISection) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if section.report_path is not None:
        paths["report"] = section.report_path
    if section.csv_dir is not None:
        for name, filename in PRIMARY_OUTPUTS.items():
            if name != "report":
                paths[name] = section.csv_dir / filename
    if section.run_dir is not None:
        paths.setdefault("report", section.run_dir / PRIMARY_OUTPUTS["report"])
        for name, filename in PRIMARY_OUTPUTS.items():
            if name != "report":
                paths.setdefault(name, section.run_dir / filename)
        paths.setdefault("manifest", section.run_dir / "vpt2_vci_manifest.json")
    if section.manifest_path is not None:
        paths["manifest"] = section.manifest_path
    return paths


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"empty VPT2/VCI CSV: {path}")
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append(
                {
                    key.strip(): (value or "").strip()
                    for key, value in row.items()
                    if key is not None
                }
            )
        return rows


def _float_field(row: dict[str, str], *candidates: str) -> float:
    by_norm = {_normalized(key): key for key in row}
    for candidate in candidates:
        key = by_norm.get(_normalized(candidate))
        if key is not None and row[key] != "":
            return float(row[key].replace("D", "E").replace("d", "e"))
    raise ValueError(f"missing numeric VPT2/VCI CSV field; candidates={candidates}")


def _normalized(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
