"""Executable harness for the vendored GICForge Fortran reference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from collections.abc import Sequence

from .gicforge import gicforge_fortran_layout


_ATOMIC_NUMBERS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
}


@dataclass(frozen=True)
class GICForgeReferenceRun:
    workdir: Path
    provout_path: Path
    bmatrix_path: Path
    provout: str
    gic_labels: tuple[str, ...]
    redundant_counts: tuple[int, int, int, int, int, int]
    final_counts: tuple[int, int, int, int, int, int]
    b_matrix_rows: tuple[tuple[float, ...], ...]
    eckart_coordinates_angstrom: tuple[tuple[float, float, float], ...] | None = None


def run_gicforge_reference(
    workdir: Path,
    *,
    atoms: Sequence[str | int],
    coordinates_angstrom: Sequence[Sequence[float]],
    point_group: str = "C1",
    title: str = "MATRIX GICForge reference harness",
    keywords: Sequence[str] = ("GNIC", "BMAT"),
    charge: int = 0,
    multiplicity: int = 1,
    repo_root: Path | None = None,
    executable: Path | None = None,
) -> GICForgeReferenceRun:
    """Run the vendored GICForge reference on a normalized XYZ input."""
    target = Path(workdir)
    target.mkdir(parents=True, exist_ok=True)
    executable_path = (
        Path(executable) if executable is not None else gicforge_reference_executable(repo_root)
    )
    _write_provin(
        target / "provin",
        keywords=keywords,
        title=title,
        charge=charge,
        multiplicity=multiplicity,
    )
    _write_xyzin(
        target / "xyzin",
        atoms=atoms,
        coordinates_angstrom=coordinates_angstrom,
        point_group=point_group,
    )
    subprocess.run(
        [str(executable_path)],
        cwd=target,
        check=True,
        capture_output=True,
        text=True,
    )
    return read_gicforge_reference_run(target)


def read_gicforge_reference_run(workdir: Path) -> GICForgeReferenceRun:
    target = Path(workdir)
    provout_path = target / "provout"
    bmatrix_path = target / "bmat.out"
    provout = provout_path.read_text(encoding="utf-8")
    return GICForgeReferenceRun(
        workdir=target,
        provout_path=provout_path,
        bmatrix_path=bmatrix_path,
        provout=provout,
        gic_labels=_parse_gic_labels(provout),
        redundant_counts=_parse_count_line(provout, "Redundant"),
        final_counts=_parse_count_line(provout, "Final Non Redund."),
        b_matrix_rows=_parse_reference_b_matrix(bmatrix_path),
        eckart_coordinates_angstrom=_parse_eckart_coordinates(provout),
    )


def gicforge_reference_executable(repo_root: Path | None = None) -> Path:
    """Return GICForge, rebuilding when any tracked Fortran input is newer."""

    layout = gicforge_fortran_layout(repo_root)
    build_inputs = (
        [layout.reference_compile_script, layout.matrix_kernel]
        + sorted(layout.root.rglob("*.f"))
        + sorted(layout.root.rglob("*.inc"))
    )
    executable_mtime = (
        layout.reference_executable.stat().st_mtime
        if layout.reference_executable.is_file()
        else float("-inf")
    )
    if any(
        path.is_file() and path.stat().st_mtime > executable_mtime
        for path in build_inputs
    ):
        subprocess.run(
            [str(layout.reference_compile_script)],
            check=True,
            cwd=layout.root,
            capture_output=True,
            text=True,
        )
    return layout.reference_executable


def _write_provin(
    path: Path,
    *,
    keywords: Sequence[str],
    title: str,
    charge: int,
    multiplicity: int,
) -> None:
    keyword_line = "# " + " ".join(keywords)
    path.write_text(
        f"{keyword_line}\n\n{title}\n\n{charge} {multiplicity}\n\n",
        encoding="utf-8",
    )


def _write_xyzin(
    path: Path,
    *,
    atoms: Sequence[str | int],
    coordinates_angstrom: Sequence[Sequence[float]],
    point_group: str,
) -> None:
    if len(atoms) != len(coordinates_angstrom):
        raise ValueError("atoms and coordinates must have the same length")
    lines = [str(len(atoms)), point_group or "C1"]
    for atom, coords in zip(atoms, coordinates_angstrom):
        x, y, z = (float(value) for value in coords)
        lines.append(f"{_reference_atom_token(atom)} {x:.8f} {y:.8f} {z:.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _reference_atom_token(atom: str | int) -> str:
    if isinstance(atom, int):
        return str(atom)
    text = str(atom).strip()
    if text.isdigit():
        return text
    symbol = text[:1].upper() + text[1:].lower()
    atomic_number = _ATOMIC_NUMBERS.get(symbol)
    if atomic_number is None:
        raise ValueError(f"unsupported atomic symbol for reference GICForge: {atom!r}")
    return str(atomic_number)


def _parse_count_line(
    provout: str,
    label: str,
) -> tuple[int, int, int, int, int, int]:
    for line in provout.splitlines():
        if line.lstrip().startswith(label):
            values = tuple(int(value) for value in re.findall(r"\d+", line))
            if len(values) == 6:
                return values
    raise ValueError(f"missing reference GICForge count line: {label}")


def _parse_gic_labels(provout: str) -> tuple[str, ...]:
    labels: list[str] = []
    pattern = re.compile(r"^\s*([A-Za-z]{4}\d{4})\b")
    in_summary = False
    for line in provout.splitlines():
        if "Final GIC summary" in line or "Final symmetrized GIC summary" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        match = pattern.match(line)
        if match:
            labels.append(match.group(1))
    return tuple(labels)


def _parse_reference_b_matrix(path: Path) -> tuple[tuple[float, ...], ...]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not lines:
        raise ValueError(f"empty reference B matrix: {path}")
    row_count, column_count = (int(value) for value in lines[0].split()[:2])
    rows = [[0.0 for _column in range(column_count)] for _row in range(row_count)]
    for line in lines[1:]:
        row_text, column_text, value_text = line.split()[:3]
        row = int(row_text) - 1
        column = int(column_text) - 1
        rows[row][column] = float(value_text.replace("D", "E"))
    return tuple(tuple(row) for row in rows)


def _parse_eckart_coordinates(
    provout: str,
) -> tuple[tuple[float, float, float], ...] | None:
    lines = provout.splitlines()
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+-]?\d+)?"
    row_pattern = re.compile(
        rf"^\s*\d+\s+[A-Za-z]{{1,2}}\s+{number}\s+{number}\s+"
        rf"({number})\s+({number})\s+({number})\s*$"
    )
    for index, line in enumerate(lines):
        if "Cartesian Coords.in Eckart Orientation" not in line:
            continue
        coordinates: list[tuple[float, float, float]] = []
        for raw in lines[index + 1 :]:
            if not raw.strip():
                if coordinates:
                    return tuple(coordinates)
                continue
            match = row_pattern.match(raw)
            if match is None:
                if coordinates:
                    return tuple(coordinates)
                continue
            coordinates.append(
                tuple(float(value.replace("D", "E").replace("d", "E")) for value in match.groups())
            )
        if coordinates:
            return tuple(coordinates)
    return None
