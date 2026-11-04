from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .workflow import DVRSection, read_dvr_section, write_dvr_section


@dataclass(frozen=True)
class DVRLevel:
    state: int
    energy_cm: float
    energy_above_ground_cm: float


@dataclass(frozen=True)
class DVRGridPoint:
    point: int
    coordinate_au: float | None
    coordinate_sqrtamu_angstrom: float | None
    potential_cm: float
    values: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DVRExpectation:
    state: int
    energy_cm: float
    energy_above_ground_cm: float
    values: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DVRVectorPoint:
    point: int
    state: int
    amplitude: float
    q_au: float | None = None
    q1_au: float | None = None
    q2_au: float | None = None
    potential_cm: float | None = None


@dataclass(frozen=True)
class DVROutputSnapshot:
    section: DVRSection
    status: str
    outputs: dict[str, Path]
    missing_outputs: tuple[str, ...]
    levels: tuple[DVRLevel, ...] = ()
    grid: tuple[DVRGridPoint, ...] = ()
    expectations: tuple[DVRExpectation, ...] = ()
    fortran_vectors: tuple[DVRVectorPoint, ...] = ()
    summary_text: str = ""

    @property
    def ground_cm(self) -> float | None:
        return self.levels[0].energy_cm if self.levels else None


PRIMARY_OUTPUTS = {
    "summary": "{prefix}_summary.txt",
    "levels": "{prefix}_levels.csv",
    "grid_csv": "{prefix}_grid.csv",
}

OPTIONAL_OUTPUTS = {
    "profile": "{prefix}_profile.csv",
    "model_profile": "{prefix}_model_profile.csv",
    "expectations": "{prefix}_expectations.csv",
    "thermal_expectations": "{prefix}_thermal_expectations.csv",
    "property_comparison": "{prefix}_property_comparison.csv",
    "vpt2_property_correction": "{prefix}_vpt2_property_correction.csv",
    "anharmonic_summary": "{prefix}_anharmonic_summary.txt",
    "anharmonic_levels": "{prefix}_anharmonic_levels.csv",
    "anharmonic_grid": "{prefix}_anharmonic_grid.csv",
    "anharmonic_vpt2_comparison": "{prefix}_anharmonic_vpt2_comparison.csv",
    "two_d_summary": "{prefix}_2d_summary.txt",
    "two_d_levels": "{prefix}_2d_levels.csv",
    "two_d_grid": "{prefix}_2d_grid.csv",
    "two_d_expectations": "{prefix}_2d_expectations.csv",
    "two_d_convergence_summary": "{prefix}_2d_convergence_summary.txt",
    "fortran_dvrout": "{prefix}_dvrout",
    "fortran_levels": "{prefix}_dvr_levels.csv",
    "fortran_vectors": "{prefix}_dvr_vectors.csv",
}

FIGURE_OUTPUTS = {
    "figure_potential_levels_pdf": "{prefix}_potential_levels.pdf",
    "figure_potential_levels_png": "{prefix}_potential_levels.png",
    "figure_potential_profile_pdf": "{prefix}_potential_profile.pdf",
    "figure_potential_profile_png": "{prefix}_potential_profile.png",
    "figure_circular_potential_pdf": "{prefix}_circular_potential.pdf",
    "figure_circular_potential_png": "{prefix}_circular_potential.png",
    "figure_anharmonic_potential_levels_pdf": "{prefix}_anharmonic_potential_levels.pdf",
    "figure_anharmonic_potential_levels_png": "{prefix}_anharmonic_potential_levels.png",
    "figure_2d_potential_pdf": "{prefix}_2d_potential.pdf",
    "figure_2d_potential_png": "{prefix}_2d_potential.png",
}


def read_dvr_levels(path: Path | str) -> tuple[DVRLevel, ...]:
    rows = _read_csv_rows(Path(path))
    levels: list[DVRLevel] = []
    for row in rows:
        levels.append(
            DVRLevel(
                state=int(_float_field(row, "state")),
                energy_cm=_float_field(row, "energy_cm-1", "energy_cm1", "energy_cm"),
                energy_above_ground_cm=_float_field(
                    row,
                    "energy_above_ground_cm-1",
                    "energy_above_ground_cm1",
                    "energy_above_ground_cm",
                ),
            )
        )
    return tuple(levels)


def read_dvr_grid(path: Path | str) -> tuple[DVRGridPoint, ...]:
    rows = _read_csv_rows(Path(path))
    grid: list[DVRGridPoint] = []
    for idx, row in enumerate(rows):
        point = int(_float_field(row, "point", "grid", default=idx))
        coordinate_au = _optional_float_field(row, "s_au", "q_au", "q1", "q1_au", "Q_dimensionless")
        coordinate_sqrt = _optional_float_field(row, "s_sqrtamu_angstrom")
        potential = _float_field(
            row,
            "V_cm-1",
            "V_cm1",
            "relative_energy_cm-1",
            "relative_energy_cm1",
            "potential_cm-1",
            "potential_cm1",
        )
        reserved = {
            "point",
            "grid",
            "s_au",
            "q_au",
            "q1",
            "q1_au",
            "Q_dimensionless",
            "s_sqrtamu_angstrom",
            "V_cm-1",
            "V_cm1",
            "V_model_cm-1",
            "V_model_cm1",
            "V_taylor_cm-1",
            "V_taylor_cm1",
            "relative_energy_cm-1",
            "relative_energy_cm1",
            "potential_cm-1",
            "potential_cm1",
        }
        values = {
            key: value
            for key, value in _numeric_values(row).items()
            if key not in reserved
            and not key.startswith("psi_")
            and not key.startswith("prob_density_")
        }
        grid.append(
            DVRGridPoint(
                point=point,
                coordinate_au=coordinate_au,
                coordinate_sqrtamu_angstrom=coordinate_sqrt,
                potential_cm=potential,
                values=values,
            )
        )
    return tuple(grid)


def read_dvr_expectations(path: Path | str) -> tuple[DVRExpectation, ...]:
    rows = _read_csv_rows(Path(path))
    expectations: list[DVRExpectation] = []
    reserved = {
        "state",
        "energy_cm-1",
        "energy_cm1",
        "energy_cm",
        "energy_above_ground_cm-1",
        "energy_above_ground_cm1",
        "energy_above_ground_cm",
    }
    for row in rows:
        expectations.append(
            DVRExpectation(
                state=int(_float_field(row, "state")),
                energy_cm=_float_field(row, "energy_cm-1", "energy_cm1", "energy_cm"),
                energy_above_ground_cm=_float_field(
                    row,
                    "energy_above_ground_cm-1",
                    "energy_above_ground_cm1",
                    "energy_above_ground_cm",
                ),
                values={
                    key: value for key, value in _numeric_values(row).items() if key not in reserved
                },
            )
        )
    return tuple(expectations)


def read_fortran_vectors(path: Path | str) -> tuple[DVRVectorPoint, ...]:
    rows = _read_csv_rows(Path(path))
    vectors: list[DVRVectorPoint] = []
    for row in rows:
        vectors.append(
            DVRVectorPoint(
                point=int(_float_field(row, "point")),
                state=int(_float_field(row, "state")),
                amplitude=_float_field(row, "amplitude"),
                q_au=_optional_float_field(row, "q_au"),
                q1_au=_optional_float_field(row, "q1_au"),
                q2_au=_optional_float_field(row, "q2_au"),
                potential_cm=_optional_float_field(row, "potential_cm-1", "potential_cm1"),
            )
        )
    return tuple(vectors)


def collect_dvr_outputs(section: DVRSection) -> DVROutputSnapshot:
    outputs = detected_dvr_outputs(section)
    level_path = (
        outputs.get("levels")
        or outputs.get("anharmonic_levels")
        or outputs.get("two_d_levels")
        or outputs.get("fortran_levels")
    )
    grid_path = (
        outputs.get("grid_csv") or outputs.get("anharmonic_grid") or outputs.get("two_d_grid")
    )
    expectation_path = outputs.get("expectations") or outputs.get("two_d_expectations")
    levels = read_dvr_levels(level_path) if level_path is not None else ()
    grid = read_dvr_grid(grid_path) if grid_path is not None else ()
    expectations = read_dvr_expectations(expectation_path) if expectation_path is not None else ()
    vectors = (
        read_fortran_vectors(outputs["fortran_vectors"]) if "fortran_vectors" in outputs else ()
    )
    summary_path = (
        outputs.get("summary") or outputs.get("anharmonic_summary") or outputs.get("two_d_summary")
    )
    summary_text = summary_path.read_text(encoding="utf-8") if summary_path is not None else ""
    missing = tuple(
        name for name in PRIMARY_OUTPUTS if not _has_primary_or_alternate(name, outputs)
    )
    status = "complete" if levels and not missing else "partial" if outputs else "prepared"
    return DVROutputSnapshot(
        section=section,
        status=status,
        outputs=outputs,
        missing_outputs=missing,
        levels=levels,
        grid=grid,
        expectations=expectations,
        fortran_vectors=vectors,
        summary_text=summary_text,
    )


def collect_dvr_outputs_from_xyzin(path: Path | str) -> DVROutputSnapshot:
    return collect_dvr_outputs(read_dvr_section(Path(path)))


def refresh_dvr_section(path: Path | str) -> DVROutputSnapshot:
    target = Path(path)
    snapshot = collect_dvr_outputs_from_xyzin(target)
    section = snapshot.section.with_outputs(snapshot.outputs, status=snapshot.status)
    write_dvr_section(target, section)
    return collect_dvr_outputs(section)


def dvr_output_summary_lines(snapshot: DVROutputSnapshot) -> list[str]:
    lines = [
        f"status: {snapshot.status}",
        f"prefix: {snapshot.section.prefix}",
        f"outdir: {snapshot.section.outdir}",
        f"outputs: {len(snapshot.outputs)}",
        f"levels: {len(snapshot.levels)}",
        f"grid_points: {len(snapshot.grid)}",
        f"expectation_rows: {len(snapshot.expectations)}",
    ]
    if snapshot.ground_cm is not None:
        lines.append(f"ground_cm-1: {snapshot.ground_cm:.10g}")
    if snapshot.missing_outputs:
        lines.append("missing_primary: " + ", ".join(snapshot.missing_outputs))
    for name, path in sorted(snapshot.outputs.items()):
        lines.append(f"output:{name}: {path}")
    return lines


def detected_dvr_outputs(section: DVRSection) -> dict[str, Path]:
    outputs = {name: path for name, path in section.outputs.items() if path.exists()}
    for name, path in _candidate_output_paths(section).items():
        if path.exists():
            outputs.setdefault(name, path)
    return outputs


def _has_primary_or_alternate(name: str, outputs: dict[str, Path]) -> bool:
    if name in outputs:
        return True
    alternates = {
        "summary": ("anharmonic_summary", "two_d_summary", "fortran_dvrout"),
        "levels": ("anharmonic_levels", "two_d_levels", "fortran_levels"),
        "grid_csv": ("anharmonic_grid", "two_d_grid"),
    }
    return any(candidate in outputs for candidate in alternates.get(name, ()))


def _candidate_output_paths(section: DVRSection) -> dict[str, Path]:
    prefix = section.prefix
    paths: dict[str, Path] = {}
    primary = {
        "summary": section.summary,
        "levels": section.levels,
        "grid_csv": section.grid_csv,
    }
    for name, path in primary.items():
        if path is not None:
            paths[name] = path
        else:
            paths[name] = section.outdir / PRIMARY_OUTPUTS[name].format(prefix=prefix)
    for name, template in OPTIONAL_OUTPUTS.items():
        paths.setdefault(name, section.outdir / template.format(prefix=prefix))
    for name, template in FIGURE_OUTPUTS.items():
        paths.setdefault(name, section.figdir / template.format(prefix=prefix))
    return paths


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"empty DVR CSV: {path}")
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


def _float_field(row: dict[str, str], *candidates: str, default: float | None = None) -> float:
    value = _optional_float_field(row, *candidates)
    if value is not None:
        return value
    if default is not None:
        return float(default)
    raise ValueError(f"missing numeric DVR CSV field; candidates={candidates}")


def _optional_float_field(row: dict[str, str], *candidates: str) -> float | None:
    by_norm = {_normalized(key): key for key in row}
    for candidate in candidates:
        key = by_norm.get(_normalized(candidate))
        if key is not None and row[key] != "":
            return float(row[key].replace("D", "E").replace("d", "e"))
    return None


def _numeric_values(row: dict[str, str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, raw in row.items():
        if raw == "":
            continue
        try:
            values[key] = float(raw.replace("D", "E").replace("d", "e"))
        except ValueError:
            continue
    return values


def _normalized(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
