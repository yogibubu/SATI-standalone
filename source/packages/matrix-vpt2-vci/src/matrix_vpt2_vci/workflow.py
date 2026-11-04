from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from matrix_core import (
    build_run_manifest,
    key_value_section_lines,
    parse_key_value_section,
    read_sectioned_lines,
    replace_section,
    section_content,
)
from matrix_gf import GFResult, mass_weighted_cartesian_hessian, solve_wilson_gf

from .gaussian_qff import (
    FCHKData,
    anharmonic_input_from_gaussian_fchk,
    hessian_input_from_gaussian_fchk,
    read_gaussian_fchk_qff,
    read_indexed_qff_text,
)
from .vci import VCIResult, solve_vci, zero_anharmonic_force_field


ORACLE_XYZ_VPT2_VCI_SCHEMA = "oracle.xyz.vpt2_vci.v1"


@dataclass(frozen=True)
class VPT2VCIRun:
    gaussian_data: FCHKData
    gf: GFResult
    vci: VCIResult


@dataclass(frozen=True)
class VPT2VCISection:
    source_kind: str = "xyzin"
    source_path: Path | None = None
    run_dir: Path | None = None
    report_path: Path | None = None
    csv_dir: Path | None = None
    manifest_path: Path | None = None
    max_quanta: int = 2
    roots: int = 10
    vci_method: str = "dense"
    status: str = "prepared"
    outputs: Mapping[str, Path] | None = None
    schema: str = ORACLE_XYZ_VPT2_VCI_SCHEMA

    def __post_init__(self) -> None:
        for attr in ("source_path", "run_dir", "report_path", "csv_dir", "manifest_path"):
            value = getattr(self, attr)
            if value is not None:
                object.__setattr__(self, attr, Path(value))
        outputs = {
            _normalize_output_key(name): Path(path)
            for name, path in dict(self.outputs or {}).items()
            if path is not None
        }
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "max_quanta", int(self.max_quanta))
        object.__setattr__(self, "roots", int(self.roots))

    def with_outputs(
        self,
        outputs: Mapping[str, Path],
        *,
        status: str | None = None,
    ) -> "VPT2VCISection":
        return VPT2VCISection(
            source_kind=self.source_kind,
            source_path=self.source_path,
            run_dir=self.run_dir,
            report_path=outputs.get("report", self.report_path),
            csv_dir=self.csv_dir,
            manifest_path=outputs.get("manifest", self.manifest_path),
            max_quanta=self.max_quanta,
            roots=self.roots,
            vci_method=self.vci_method,
            status=self.status if status is None else status,
            outputs=outputs,
            schema=self.schema,
        )


def run_python_vci_from_gaussian_fchk(
    path: Path,
    *,
    qff_path: Path | None = None,
    max_quanta: int = 2,
    n_roots: int = 10,
) -> VPT2VCIRun:
    """Run the independent Python GF/VCI path from a Gaussian FCHK file.

    If `qff_path` is provided, indexed cubic/quartic normal-coordinate terms
    are read from that file and used by VCI. Otherwise a harmonic VCI reference
    is run from the Gaussian frequencies.
    """
    data = read_gaussian_fchk_qff(Path(path))
    hessian_input = hessian_input_from_gaussian_fchk(Path(path))
    anharmonic_input = anharmonic_input_from_gaussian_fchk(Path(path))
    mw_hessian = mass_weighted_cartesian_hessian(
        hessian_input.cartesian_hessian, hessian_input.masses_amu
    )
    gf = solve_wilson_gf(mw_hessian, np.eye(mw_hessian.shape[0]))
    frequencies = (
        anharmonic_input.anharmonic_frequencies_cm
        if anharmonic_input.anharmonic_frequencies_cm.size
        else anharmonic_input.harmonic_frequencies_cm
    )
    qff = (
        read_indexed_qff_text(qff_path, frequencies)
        if qff_path is not None
        else zero_anharmonic_force_field(frequencies)
    )
    vci = solve_vci(qff, max_quanta=max_quanta, n_roots=n_roots)
    return VPT2VCIRun(gaussian_data=data, gf=gf, vci=vci)


def vpt2_vci_section_from_run(
    *,
    source_kind: str,
    source_path: Path | None,
    run_dir: Path | None,
    report_path: Path | None,
    csv_dir: Path | None,
    manifest_path: Path | None,
    max_quanta: int,
    roots: int,
    vci_method: str,
    outputs: Mapping[str, Path] | None = None,
    status: str = "complete",
) -> VPT2VCISection:
    return VPT2VCISection(
        source_kind=source_kind,
        source_path=source_path,
        run_dir=run_dir,
        report_path=report_path,
        csv_dir=csv_dir,
        manifest_path=manifest_path,
        max_quanta=max_quanta,
        roots=roots,
        vci_method=vci_method,
        status=status,
        outputs=outputs,
    )


def write_vpt2_vci_manifest(
    *,
    run_dir: Path,
    inputs: Mapping[str, Path],
    outputs: Mapping[str, Path],
    max_quanta: int,
    roots: int,
    vci_method: str,
    source_kind: str,
    status: str = "complete",
) -> Path:
    manifest = build_run_manifest(
        workflow="vpt2_vci",
        status=status,
        run_dir=Path(run_dir),
        inputs=inputs,
        outputs={name: path for name, path in outputs.items() if Path(path).exists()},
        parameters={
            "max_quanta": int(max_quanta),
            "roots": int(roots),
            "vci_method": vci_method,
            "source_kind": source_kind,
        },
        backend={"solver": "matrix_vpt2_vci", "vci_method": vci_method},
    )
    return manifest.write(Path(run_dir) / "vpt2_vci_manifest.json")


def vpt2_vci_section_lines(section: VPT2VCISection) -> list[str]:
    values = {
        "SOURCE_KIND": section.source_kind,
        "SOURCE_PATH": section.source_path,
        "RUN_DIR": section.run_dir,
        "REPORT": section.report_path,
        "CSV_DIR": section.csv_dir,
        "MANIFEST": section.manifest_path,
        "MAX_QUANTA": section.max_quanta,
        "ROOTS": section.roots,
        "VCI_METHOD": section.vci_method,
        "STATUS": section.status,
    }
    values.update({f"OUTPUT_{name.upper()}": path for name, path in section.outputs.items()})
    return key_value_section_lines(
        ORACLE_XYZ_VPT2_VCI_SCHEMA,
        values,
        key_order=(
            "SOURCE_KIND",
            "SOURCE_PATH",
            "RUN_DIR",
            "REPORT",
            "CSV_DIR",
            "MANIFEST",
            "MAX_QUANTA",
            "ROOTS",
            "VCI_METHOD",
            "STATUS",
        ),
    )


def parse_vpt2_vci_section(lines: list[str] | tuple[str, ...]) -> VPT2VCISection:
    values = parse_key_value_section(lines)
    schema = values.get("SCHEMA", ORACLE_XYZ_VPT2_VCI_SCHEMA)
    if schema != ORACLE_XYZ_VPT2_VCI_SCHEMA:
        raise ValueError(f"unsupported VPT2_VCI schema: {schema}")
    outputs = {
        _normalize_output_key(key[len("OUTPUT_") :]): Path(raw)
        for key, raw in values.items()
        if key.startswith("OUTPUT_") and raw.strip()
    }
    return VPT2VCISection(
        source_kind=values.get("SOURCE_KIND", "xyzin"),
        source_path=_optional_path(values.get("SOURCE_PATH")),
        run_dir=_optional_path(values.get("RUN_DIR")),
        report_path=_optional_path(values.get("REPORT")),
        csv_dir=_optional_path(values.get("CSV_DIR")),
        manifest_path=_optional_path(values.get("MANIFEST")),
        max_quanta=_int_value(values, "MAX_QUANTA", 2),
        roots=_int_value(values, "ROOTS", 10),
        vci_method=values.get("VCI_METHOD", "dense"),
        status=values.get("STATUS", "prepared"),
        outputs=outputs,
        schema=schema,
    )


def read_vpt2_vci_section(path: Path | str) -> VPT2VCISection:
    content = section_content(read_sectioned_lines(Path(path)), "VPT2_VCI")
    if not content:
        raise ValueError("missing #VPT2_VCI section")
    return parse_vpt2_vci_section(content)


def write_vpt2_vci_section(path: Path | str, section: VPT2VCISection) -> None:
    replace_section(Path(path), "VPT2_VCI", vpt2_vci_section_lines(section))


def _optional_path(raw: str | None) -> Path | None:
    if raw is None or not raw.strip():
        return None
    return Path(raw)


def _int_value(values: Mapping[str, str], key: str, default: int) -> int:
    raw = values.get(key)
    if raw is None:
        return default
    return int(float(raw.replace("D", "E").replace("d", "e")))


def _normalize_output_key(value: str) -> str:
    return "_".join(part for part in str(value).lower().replace("-", "_").split("_") if part)
