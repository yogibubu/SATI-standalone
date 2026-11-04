from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Callable

import numpy as np

from matrix_chem import (
    CoordinateComponent,
    validate_coordinate_component_transform,
)
from matrix_chem.topology.covalent_radii import covalent_radius
from matrix_chem.topology.elements import atomic_number, atomic_symbol
from matrix_chem.topology.pipeline import build_topology_objects
from matrix_core import sha256_file
from matrix_core.paths import repo_root
from matrix_smith.bmatrix import SparseBMatrix, SparseBRow
from matrix_smith.coordinate_kernel.pipeline import b_matrix_analytic
from matrix_smith.coordinate_kernel.primitives import Primitive, eval_primitives

from .gicforge_service import GICForgeResult, run_gicforge


GIC_DEFINITION_SCHEMA = "oracle.gic.definition.v1"
SUPPORTED_GIC_DEFINITION_SCHEMAS = (GIC_DEFINITION_SCHEMA,)


class GICDefinitionError(RuntimeError):
    """Raised when a frozen GIC definition cannot be created or evaluated."""


@dataclass(frozen=True)
class GICDefinition:
    """Frozen generalized-internal-coordinate definition.

    The definition is the output of the expensive GIC construction stage.  It is
    independent of the geometry used later to evaluate values or Wilson B rows,
    provided the atom ordering and atom count remain compatible.
    """

    atom_symbols: tuple[str, ...]
    atomic_numbers: tuple[int, ...]
    reference_coordinates_angstrom: tuple[tuple[float, float, float], ...]
    primitives: tuple[Primitive, ...]
    u_matrix: np.ndarray
    labels: tuple[str, ...]
    names: tuple[str, ...]
    irreps: tuple[str, ...]
    point_group: str = "UNKNOWN"
    symmetrized: bool = True
    symmetry_source: str = "gicforge-postcheck"
    gaussian_input: str = ""
    source: str = "gicforge"
    generation_workdir: str = ""
    provenance: dict[str, str] = field(default_factory=dict)

    def model(self) -> tuple[list[Primitive], np.ndarray, tuple[str, ...]]:
        """Return the legacy `(primitives, U, labels)` representation."""
        return (
            list(self.primitives),
            np.array(self.u_matrix, dtype=float, copy=True),
            tuple(self.labels),
        )

    def to_dict(self) -> dict:
        return {
            "schema": GIC_DEFINITION_SCHEMA,
            "source": self.source,
            "point_group": self.point_group,
            "symmetrized": self.symmetrized,
            "symmetry_source": self.symmetry_source,
            "generation_workdir": self.generation_workdir,
            "atom_symbols": list(self.atom_symbols),
            "atomic_numbers": list(self.atomic_numbers),
            "reference_coordinates_angstrom": [
                list(row) for row in self.reference_coordinates_angstrom
            ],
            "primitives": [_primitive_to_dict(primitive) for primitive in self.primitives],
            "u_matrix": np.asarray(self.u_matrix, dtype=float).tolist(),
            "labels": list(self.labels),
            "names": list(self.names),
            "irreps": list(self.irreps),
            "gaussian_input": self.gaussian_input,
            "provenance": dict(sorted((str(k), str(v)) for k, v in self.provenance.items())),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GICDefinition":
        schema = data.get("schema")
        if schema not in SUPPORTED_GIC_DEFINITION_SCHEMAS:
            raise GICDefinitionError(
                "GIC definition schema must be one of "
                f"{SUPPORTED_GIC_DEFINITION_SCHEMAS!r}, got {schema!r}"
            )
        primitives = tuple(_primitive_from_dict(item) for item in data.get("primitives", ()))
        u_matrix = np.asarray(data.get("u_matrix", ()), dtype=float)
        if u_matrix.ndim != 2:
            raise GICDefinitionError("GIC definition u_matrix must be two-dimensional")
        if u_matrix.shape[0] != len(primitives):
            raise GICDefinitionError("GIC definition primitive count does not match u_matrix rows")
        labels = tuple(str(item) for item in data.get("labels", ()))
        if u_matrix.shape[1] != len(labels):
            raise GICDefinitionError("GIC definition label count does not match u_matrix columns")
        atom_symbols = tuple(str(item) for item in data.get("atom_symbols", ()))
        atomic_numbers = tuple(int(item) for item in data.get("atomic_numbers", ()))
        if not atomic_numbers and atom_symbols:
            atomic_numbers = tuple(atomic_number(symbol) for symbol in atom_symbols)
        if not atom_symbols and atomic_numbers:
            atom_symbols = tuple(atomic_symbol(number) for number in atomic_numbers)
        irreps = tuple(str(item) for item in data.get("irreps", ()))
        symmetrized = data.get("symmetrized")
        if symmetrized is None:
            symmetrized = any(irrep != "UNK" for irrep in irreps)
        definition = cls(
            atom_symbols=atom_symbols,
            atomic_numbers=atomic_numbers,
            reference_coordinates_angstrom=tuple(
                tuple(float(value) for value in row)
                for row in data.get("reference_coordinates_angstrom", ())
            ),
            primitives=primitives,
            u_matrix=u_matrix,
            labels=labels,
            names=tuple(str(item) for item in data.get("names", ())),
            irreps=irreps,
            point_group=str(data.get("point_group", "UNKNOWN")),
            symmetrized=bool(symmetrized),
            symmetry_source=str(data.get("symmetry_source", "legacy" if symmetrized else "none")),
            gaussian_input=str(data.get("gaussian_input", "")),
            source=str(data.get("source", "gicforge")),
            generation_workdir=str(data.get("generation_workdir", "")),
            provenance={
                str(key): str(value) for key, value in dict(data.get("provenance") or {}).items()
            },
        )
        validate_gic_definition(definition)
        return definition

    def write(self, path: Path) -> Path:
        validate_gic_definition(self)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return target

    @classmethod
    def read(cls, path: Path) -> "GICDefinition":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class GICEvaluation:
    values: np.ndarray
    primitive_b_matrix: np.ndarray
    b_matrix: np.ndarray
    labels: tuple[str, ...]
    names: tuple[str, ...]
    irreps: tuple[str, ...]
    point_group: str
    symmetrized: bool
    primitive_sparse_b_matrix: SparseBMatrix | None = None
    sparse_b_matrix: SparseBMatrix | None = None


@dataclass(frozen=True)
class GICForgeComputation:
    definition: GICDefinition
    workdir: Path
    files: dict[str, Path]
    manifest: Path | None = None
    sycart_coordinates_angstrom: np.ndarray | None = None


class GICForge:
    """Public GICForge API for non-redundant GIC and SYCART generation."""

    def __init__(
        self, *, executable: Path | None = None, runner: RunGICForge | None = None
    ) -> None:
        self.executable = executable
        self.runner = runner

    def compute(
        self,
        atom_symbols: tuple[str, ...] | list[str],
        coordinates_angstrom: np.ndarray,
        *,
        workdir: Path | None = None,
        mode: str = "gicsym",
        extra_keywords: tuple[str, ...] = (),
        symmetry_backend: str | None = None,
    ) -> GICForgeComputation:
        normalized = mode.strip().lower().replace("-", "_")
        if normalized in {"gic", "raw"}:
            symmetrize = False
            symmetrize_cartesians = False
        elif normalized in {"gicsym", "symmetrized_gic", "sym"}:
            symmetrize = True
            symmetrize_cartesians = False
        elif normalized in {"sycart", "cartesian", "symmetrized_cartesian"}:
            symmetrize = False
            symmetrize_cartesians = True
        elif normalized in {"gicsym_sycart", "sycart_gicsym"}:
            symmetrize = True
            symmetrize_cartesians = True
        else:
            raise GICDefinitionError(f"Unsupported GICForge mode {mode!r}")
        definition = define_gics_from_cartesian(
            atom_symbols,
            coordinates_angstrom,
            workdir=workdir,
            executable=self.executable,
            runner=self.runner,
            symmetrize=symmetrize,
            symmetrize_cartesians=symmetrize_cartesians,
            symmetry_backend=symmetry_backend,
            extra_keywords=extra_keywords,
        )
        run_dir = Path(definition.generation_workdir)
        files = {
            name: run_dir / name for name in _known_gicforge_outputs() if (run_dir / name).exists()
        }
        sycart = (
            _read_sycart_coordinates(run_dir / "sycart.xyz", tuple(definition.atom_symbols))
            if (run_dir / "sycart.xyz").exists()
            else None
        )
        manifest = run_dir / "gicforge_manifest.json"
        return GICForgeComputation(
            definition=definition,
            workdir=run_dir,
            files=files,
            manifest=manifest if manifest.exists() else None,
            sycart_coordinates_angstrom=sycart,
        )


@dataclass(frozen=True)
class GICBMatrixComparison:
    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    python_shape: tuple[int, int]
    fortran_shape: tuple[int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
            "python_shape": list(self.python_shape),
            "fortran_shape": list(self.fortran_shape),
        }


@dataclass(frozen=True)
class GICForgePythonFortranContract:
    passed: bool
    raw_b_matrix: GICBMatrixComparison
    raw_gic_count: int
    sym_gic_count: int
    sym_primitive_count: int
    point_group: str
    irreps: tuple[str, ...]
    raw_workdir: str
    sym_workdir: str
    contract_errors: tuple[str, ...] = ()
    raw_point_group: str = "UNKNOWN"
    raw_names: tuple[str, ...] = ()
    sym_names: tuple[str, ...] = ()
    raw_labels: tuple[str, ...] = ()
    sym_labels: tuple[str, ...] = ()
    raw_primitive_signatures: tuple[str, ...] = ()
    sym_primitive_signatures: tuple[str, ...] = ()
    raw_coordinate_kind_counts: dict[str, int] = field(default_factory=dict)
    sym_coordinate_kind_counts: dict[str, int] = field(default_factory=dict)
    totally_symmetric_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "contract_errors": list(self.contract_errors),
            "raw_b_matrix": self.raw_b_matrix.to_dict(),
            "raw_gic_count": self.raw_gic_count,
            "sym_gic_count": self.sym_gic_count,
            "sym_primitive_count": self.sym_primitive_count,
            "raw_point_group": self.raw_point_group,
            "point_group": self.point_group,
            "irreps": list(self.irreps),
            "raw_names": list(self.raw_names),
            "sym_names": list(self.sym_names),
            "raw_labels": list(self.raw_labels),
            "sym_labels": list(self.sym_labels),
            "raw_primitive_signatures": list(self.raw_primitive_signatures),
            "sym_primitive_signatures": list(self.sym_primitive_signatures),
            "raw_coordinate_kind_counts": dict(sorted(self.raw_coordinate_kind_counts.items())),
            "sym_coordinate_kind_counts": dict(sorted(self.sym_coordinate_kind_counts.items())),
            "totally_symmetric_count": self.totally_symmetric_count,
            "raw_workdir": self.raw_workdir,
            "sym_workdir": self.sym_workdir,
        }


RunGICForge = Callable[..., GICForgeResult]


def define_gics_from_cartesian(
    atom_symbols: tuple[str, ...] | list[str],
    coordinates_angstrom: np.ndarray,
    *,
    workdir: Path | None = None,
    executable: Path | None = None,
    runner: RunGICForge | None = None,
    symmetrize: bool = True,
    symmetrize_cartesians: bool = False,
    symmetry_backend: str | None = None,
    extra_keywords: tuple[str, ...] = (),
) -> GICDefinition:
    """Construct and freeze a GIC definition from Cartesian geometry.

    This is the first GIC utility: it runs the expensive topology/GIC/symmetry
    construction exactly once and serializes the resulting coordinate model.
    """
    atoms = tuple(str(atom).strip() for atom in atom_symbols)
    coords = _validated_coordinates(coordinates_angstrom, len(atoms))
    if not atoms:
        raise GICDefinitionError("GIC definition needs at least one atom")
    _validate_gicforge_input_topology(atoms, coords)
    run_dir = (
        Path(workdir)
        if workdir is not None
        else Path(tempfile.mkdtemp(prefix="matrix_gic_define_"))
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_gicforge_inputs(
        run_dir,
        atoms,
        coords,
        symmetrize=symmetrize,
        symmetrize_cartesians=symmetrize_cartesians,
        extra_keywords=extra_keywords,
    )
    run = runner or run_gicforge
    if executable is not None and runner is None:
        result = run_gicforge(
            run_dir, executable=executable, symmetrize=symmetrize, symmetry_backend=symmetry_backend
        )
    elif runner is None:
        result = run_gicforge(run_dir, symmetrize=symmetrize, symmetry_backend=symmetry_backend)
    else:
        try:
            result = run(run_dir, symmetrize=symmetrize, symmetry_backend=symmetry_backend)
        except TypeError:
            result = run(run_dir)
    gauin = (result.files.get("gauin.symm") if symmetrize else None) or result.files.get("gauin")
    if gauin is None:
        raise GICDefinitionError(f"GICForge did not produce gauin/gauin.symm in {run_dir}")
    was_symmetrized = bool(symmetrize and gauin.name == "gauin.symm")
    definition = read_gic_definition_from_gauin(
        gauin,
        atoms,
        coords,
        point_group=_gicforge_point_group(run_dir / "provout"),
        symmetrized=was_symmetrized,
        symmetry_source="gicforge-postcheck" if was_symmetrized else "none",
        generation_workdir=run_dir,
        provenance=_gic_definition_provenance(run_dir, result, gauin),
    )
    definition.write(run_dir / "gic_definition.json")
    return definition


def _validate_gicforge_input_topology(atoms: tuple[str, ...], coords: np.ndarray) -> None:
    z_numbers: list[int] = []
    for atom in atoms:
        z = atomic_number(atom)
        if z is None:
            raise GICDefinitionError(f"Unknown element symbol {atom}")
        z_numbers.append(int(z))
    try:
        _continuous, graph, _ringset, _synthons, _aromaticity = build_topology_objects(
            coords, np.asarray(z_numbers, dtype=int)
        )
    except Exception as exc:
        raise GICDefinitionError(f"GICForge input topology validation failed: {exc}") from exc
    bonded = {tuple(sorted((int(i), int(j)))) for i, j in graph.bonds}
    contacts: list[str] = []
    for i, zi in enumerate(z_numbers):
        if zi != 1:
            continue
        ri = covalent_radius(zi)
        if ri is None:
            continue
        for j in range(i + 1, len(z_numbers)):
            if z_numbers[j] != 1 or (i, j) in bonded:
                continue
            rj = covalent_radius(z_numbers[j])
            if rj is None:
                continue
            distance = float(np.linalg.norm(coords[i] - coords[j]))
            if distance <= 1.25 * (float(ri) + float(rj)):
                contacts.append(f"{i + 1}-{j + 1} ({distance:.3f} A)")
    if contacts:
        preview = ", ".join(contacts[:8])
        extra = f"; {len(contacts) - 8} additional H-H contacts" if len(contacts) > 8 else ""
        raise GICDefinitionError(
            f"GICForge input topology validation failed: spurious nonbonded H-H contact {preview}{extra}"
        )


def read_gic_definition_from_gauin(
    gauin: Path,
    atom_symbols: tuple[str, ...] | list[str],
    coordinates_angstrom: np.ndarray,
    *,
    point_group: str = "UNKNOWN",
    symmetrized: bool = True,
    symmetry_source: str = "gicforge-postcheck",
    generation_workdir: Path | str = "",
    provenance: dict[str, str] | None = None,
) -> GICDefinition:
    """Read a frozen GIC definition from a Gaussian-style GIC block."""
    atoms = tuple(str(atom).strip() for atom in atom_symbols)
    coords = _validated_coordinates(coordinates_angstrom, len(atoms))
    irreps_by_name = _read_gicforge_irreps(Path(gauin).with_name("gicsym"))
    prims: list[Primitive] = []
    prim_index: dict[Primitive, int] = {}
    columns: list[np.ndarray] = []
    labels: list[str] = []
    names: list[str] = []
    irreps: list[str] = []
    text = Path(gauin).read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        parsed = parse_gicforge_line(raw)
        if parsed is None:
            continue
        name, terms, expression = parsed
        column = np.zeros(len(prims), dtype=float)
        for coeff, primitive in terms:
            if primitive not in prim_index:
                prim_index[primitive] = len(prims)
                prims.append(primitive)
                column = np.pad(column, (0, 1))
                for idx, existing in enumerate(columns):
                    columns[idx] = np.pad(existing, (0, 1))
            column[prim_index[primitive]] += coeff
        irrep = irreps_by_name.get(name, "UNK")
        label_index = len(labels) + 1
        labels.append(
            f"GIC{label_index:03d} GICForge {name} irrep={irrep} {expression} {_gic_aliases(terms)}"
        )
        names.append(name)
        irreps.append(irrep)
        columns.append(column)
    if not columns:
        raise GICDefinitionError(f"No linear GICForge coordinates found in {gauin}")
    definition = GICDefinition(
        atom_symbols=atoms,
        atomic_numbers=tuple(atomic_number(atom) for atom in atoms),
        reference_coordinates_angstrom=tuple(
            tuple(float(value) for value in row) for row in coords
        ),
        primitives=tuple(prims),
        u_matrix=np.column_stack(columns),
        labels=tuple(labels),
        names=tuple(names),
        irreps=tuple(irreps),
        point_group=point_group,
        symmetrized=bool(symmetrized),
        symmetry_source=symmetry_source,
        gaussian_input=text,
        generation_workdir=str(generation_workdir),
        provenance=dict(provenance or {}),
    )
    validate_gic_definition(definition)
    return definition


def evaluate_gic_definition(
    definition: GICDefinition,
    coordinates_angstrom: np.ndarray,
    *,
    atomic_numbers: tuple[int, ...] | list[int] | np.ndarray | None = None,
) -> GICEvaluation:
    """Evaluate frozen GIC values and the corresponding Wilson B matrix.

    This is the second GIC utility.  It never performs topology perception,
    redundancy removal, or symmetry assignment; it only evaluates the frozen
    primitive combinations on the supplied Cartesian geometry.
    """
    validate_gic_definition(definition)
    coords = _validated_coordinates(coordinates_angstrom, len(definition.atom_symbols))
    if atomic_numbers is not None and len(tuple(atomic_numbers)) != len(definition.atom_symbols):
        raise GICDefinitionError(
            "B-matrix evaluation atomic-number count differs from GIC definition atom count"
        )
    prims = list(definition.primitives)
    u_matrix = np.asarray(definition.u_matrix, dtype=float)
    primitive_values = eval_primitives(prims, coords)
    primitive_b = b_matrix_analytic(prims, coords)
    primitive_sparse_b = SparseBMatrix.from_dense(
        primitive_b,
        row_labels=tuple(f"P{index:03d}" for index in range(1, len(prims) + 1)),
        backend="oracle-runtime-primitive-sparse-bmatrix.v1",
    )
    sparse_rows: list[SparseBRow] = []
    for column in range(u_matrix.shape[1]):
        weighted_rows = tuple(
            (float(coefficient), primitive_sparse_b.rows[row_index])
            for row_index, coefficient in enumerate(u_matrix[:, column])
            if abs(float(coefficient)) > 1.0e-14
        )
        sparse_rows.append(
            SparseBRow.combine(*weighted_rows)
            if weighted_rows
            else SparseBRow.zeros(int(primitive_b.shape[1]))
        )
    return GICEvaluation(
        values=u_matrix.T @ primitive_values,
        primitive_b_matrix=primitive_b,
        b_matrix=u_matrix.T @ primitive_b,
        labels=definition.labels,
        names=definition.names,
        irreps=definition.irreps,
        point_group=definition.point_group,
        symmetrized=definition.symmetrized,
        primitive_sparse_b_matrix=primitive_sparse_b,
        sparse_b_matrix=SparseBMatrix(
            rows=tuple(sparse_rows),
            column_count=int(primitive_b.shape[1]),
            row_labels=definition.labels,
            backend="oracle-runtime-sparse-bmatrix.v1",
        ),
    )


def validate_gic_definition(definition: GICDefinition) -> None:
    """Validate a frozen GIC schema before reuse.

    The checks are deliberately structural and deterministic.  They prevent
    silent reuse of malformed or partially parsed GIC definitions.
    """
    natoms = len(definition.atom_symbols)
    if natoms <= 0:
        raise GICDefinitionError("GIC definition has no atoms")
    if definition.atomic_numbers and len(definition.atomic_numbers) != natoms:
        raise GICDefinitionError("GIC definition atom_symbols/atomic_numbers length mismatch")
    coords = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    if coords.shape != (natoms, 3):
        raise GICDefinitionError(
            f"GIC definition reference_coordinates_angstrom must have shape ({natoms}, 3), got {coords.shape}"
        )
    if not np.isfinite(coords).all():
        raise GICDefinitionError("GIC definition reference coordinates contain non-finite values")
    if not definition.primitives:
        raise GICDefinitionError("GIC definition contains no primitives")
    u_matrix = np.asarray(definition.u_matrix, dtype=float)
    if u_matrix.ndim != 2:
        raise GICDefinitionError("GIC definition u_matrix must be two-dimensional")
    if u_matrix.shape[0] != len(definition.primitives):
        raise GICDefinitionError("GIC definition primitive count does not match u_matrix rows")
    if u_matrix.shape[1] <= 0:
        raise GICDefinitionError("GIC definition contains no GIC columns")
    if not np.isfinite(u_matrix).all():
        raise GICDefinitionError("GIC definition u_matrix contains non-finite values")
    zero_columns = np.where(np.linalg.norm(u_matrix, axis=0) <= 1.0e-14)[0]
    if zero_columns.size:
        first = int(zero_columns[0]) + 1
        raise GICDefinitionError(f"GIC definition has a zero-norm GIC column at index {first}")
    if len(definition.labels) != u_matrix.shape[1]:
        raise GICDefinitionError("GIC definition label count does not match u_matrix columns")
    if len(definition.names) != u_matrix.shape[1]:
        raise GICDefinitionError("GIC definition name count does not match u_matrix columns")
    if len(definition.irreps) != u_matrix.shape[1]:
        raise GICDefinitionError("GIC definition irrep count does not match u_matrix columns")
    if len(set(definition.labels)) != len(definition.labels):
        raise GICDefinitionError("GIC definition labels are not unique")
    if len(set(definition.names)) != len(definition.names):
        raise GICDefinitionError("GIC definition names are not unique")
    for index, primitive in enumerate(definition.primitives, start=1):
        _validate_primitive(primitive, natoms, index)
    try:
        validate_coordinate_component_transform(
            tuple(
                CoordinateComponent(
                    operator=primitive.function,
                    atoms=primitive.atoms,
                    mode=primitive.mode,
                    ref_atoms=primitive.ref,
                    context=(primitive.kind,),
                )
                for primitive in definition.primitives
            ),
            u_matrix,
        )
    except ValueError as exc:
        raise GICDefinitionError(
            f"GIC definition component invariant failed: {exc}"
        ) from exc
    if definition.symmetrized and any(
        not irrep.strip() or irrep == "UNK" for irrep in definition.irreps
    ):
        raise GICDefinitionError("Symmetrized GIC definition contains missing/UNK irreps")


def read_gicforge_b_matrix(path: Path) -> np.ndarray:
    """Read the machine-readable Fortran `bmat.out` triplet format.

    The returned matrix is shaped `(n_gic, 3 * n_atoms)`, matching
    `evaluate_gic_definition(...).b_matrix`.
    """
    target = Path(path)
    lines = [
        line.strip()
        for line in target.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise GICDefinitionError(f"Empty Fortran B-matrix file: {target}")
    try:
        n_gic, n_cart = (int(item) for item in lines[0].replace(",", " ").split()[:2])
    except Exception as exc:
        raise GICDefinitionError(f"Malformed Fortran B-matrix header in {target}") from exc
    if n_gic <= 0 or n_cart <= 0:
        raise GICDefinitionError(f"Invalid Fortran B-matrix shape {n_gic}x{n_cart} in {target}")
    matrix = np.zeros((n_gic, n_cart), dtype=float)
    seen: set[tuple[int, int]] = set()
    for raw in lines[1:]:
        parts = raw.replace(",", " ").split()
        if len(parts) < 3:
            raise GICDefinitionError(f"Malformed Fortran B-matrix row in {target}: {raw!r}")
        row = int(parts[0]) - 1
        col = int(parts[1]) - 1
        if row < 0 or row >= n_gic or col < 0 or col >= n_cart:
            raise GICDefinitionError(
                f"Fortran B-matrix index outside declared shape in {target}: {raw!r}"
            )
        key = (row, col)
        if key in seen:
            raise GICDefinitionError(f"Duplicate Fortran B-matrix element in {target}: {raw!r}")
        seen.add(key)
        matrix[row, col] = float(parts[2].replace("D", "E").replace("d", "e"))
    expected = n_gic * n_cart
    if len(seen) != expected:
        raise GICDefinitionError(f"Fortran B-matrix has {len(seen)} elements, expected {expected}")
    return matrix


def compare_gic_b_matrix_to_fortran(
    definition: GICDefinition,
    coordinates_angstrom: np.ndarray,
    fortran_bmat: Path,
    *,
    atol: float = 1.0e-8,
    rtol: float = 1.0e-7,
) -> GICBMatrixComparison:
    """Compare Python analytic B evaluation with a Fortran `bmat.out` file."""
    python_b = evaluate_gic_definition(definition, coordinates_angstrom).b_matrix
    fortran_b = read_gicforge_b_matrix(fortran_bmat)
    if python_b.shape != fortran_b.shape:
        return GICBMatrixComparison(
            passed=False,
            max_abs_diff=float("inf"),
            max_rel_diff=float("inf"),
            python_shape=tuple(int(item) for item in python_b.shape),
            fortran_shape=tuple(int(item) for item in fortran_b.shape),
        )
    diff = np.abs(python_b - fortran_b)
    denom = np.maximum(np.abs(fortran_b), 1.0)
    max_abs = float(np.max(diff)) if diff.size else 0.0
    max_rel = float(np.max(diff / denom)) if diff.size else 0.0
    return GICBMatrixComparison(
        passed=bool(np.allclose(python_b, fortran_b, atol=atol, rtol=rtol)),
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        python_shape=tuple(int(item) for item in python_b.shape),
        fortran_shape=tuple(int(item) for item in fortran_b.shape),
    )


def run_gicforge_python_fortran_contract(
    atom_symbols: tuple[str, ...] | list[str],
    coordinates_angstrom: np.ndarray,
    *,
    workdir: Path,
    executable: Path | None = None,
    atol: float = 1.0e-7,
    rtol: float = 1.0e-7,
) -> GICForgePythonFortranContract:
    """Run the reusable Python/Fortran GICForge consistency contract.

    The raw run compares the Fortran `bmat.out` with Python analytic B rows in
    the same oriented Cartesian frame written by GICForge.  The symmetrized run
    verifies deterministic point-group and irrep assignment in the frozen
    schema.  `bmat.out` is intentionally not used for the symmetrized numerical
    comparison because it is produced before the Python post-symmetry block is
    written to `gauin.symm`.
    """
    root = Path(workdir)
    raw_dir = root / "raw"
    sym_dir = root / "sym"
    raw_definition = define_gics_from_cartesian(
        tuple(atom_symbols),
        coordinates_angstrom,
        workdir=raw_dir,
        executable=executable,
        symmetrize=False,
    )
    raw_coords = _gicforge_cartesian_from_gauin(raw_dir / "gauin", len(raw_definition.atom_symbols))
    raw_comparison = compare_gic_b_matrix_to_fortran(
        raw_definition,
        raw_coords,
        raw_dir / "bmat.out",
        atol=atol,
        rtol=rtol,
    )
    sym_definition = define_gics_from_cartesian(
        tuple(atom_symbols),
        coordinates_angstrom,
        workdir=sym_dir,
        executable=executable,
        symmetrize=True,
    )
    totally_symmetric_count = sum(
        1 for irrep in sym_definition.irreps if irrep in {"A1", "A'", "Ag", "A"}
    )
    raw_signatures = tuple(
        _primitive_signature(primitive) for primitive in raw_definition.primitives
    )
    sym_signatures = tuple(
        _primitive_signature(primitive) for primitive in sym_definition.primitives
    )
    raw_kind_counts = _definition_coordinate_kind_counts(raw_definition)
    sym_kind_counts = _definition_coordinate_kind_counts(sym_definition)
    errors: list[str] = []
    if not raw_comparison.passed:
        errors.append(
            "Python analytic B matrix differs from Fortran bmat.out "
            f"(max_abs={raw_comparison.max_abs_diff:.3e}, max_rel={raw_comparison.max_rel_diff:.3e})"
        )
    if raw_definition.point_group != sym_definition.point_group:
        errors.append(
            f"Raw/sym point group mismatch: {raw_definition.point_group} != {sym_definition.point_group}"
        )
    if raw_definition.point_group == "UNKNOWN" or sym_definition.point_group == "UNKNOWN":
        errors.append("GICForge point group is UNKNOWN")
    if len(raw_definition.names) != len(raw_definition.labels):
        errors.append("Raw definition name/label count mismatch")
    if len(sym_definition.names) != len(sym_definition.labels):
        errors.append("Sym definition name/label count mismatch")
    if len(set(raw_definition.names)) != len(raw_definition.names):
        errors.append("Raw definition names are not unique")
    if len(set(sym_definition.names)) != len(sym_definition.names):
        errors.append("Sym definition names are not unique")
    if len(raw_definition.labels) != len(sym_definition.labels):
        errors.append(
            f"Raw/sym GIC count mismatch: {len(raw_definition.labels)} != {len(sym_definition.labels)}"
        )
    if sorted(raw_signatures) != sorted(sym_signatures):
        errors.append("Raw/sym primitive signature sets differ")
    if raw_kind_counts != sym_kind_counts:
        errors.append(
            f"Raw/sym coordinate kind counts differ: {raw_kind_counts} != {sym_kind_counts}"
        )
    if raw_comparison.python_shape[0] != len(raw_definition.labels):
        errors.append("Fortran B row count does not match raw GIC count")
    if raw_comparison.python_shape[1] != 3 * len(raw_definition.atom_symbols):
        errors.append("Fortran B column count does not match 3N")
    if not sym_definition.symmetrized:
        errors.append("Symmetrized definition is not marked symmetrized")
    if len(sym_definition.irreps) != len(sym_definition.labels):
        errors.append("Sym definition irrep/label count mismatch")
    if any(not irrep or irrep == "UNK" for irrep in sym_definition.irreps):
        errors.append("Sym definition contains missing irreducible representations")
    if totally_symmetric_count <= 0:
        errors.append("Sym definition has no totally symmetric coordinates")
    return GICForgePythonFortranContract(
        passed=not errors,
        raw_b_matrix=raw_comparison,
        raw_gic_count=len(raw_definition.labels),
        sym_gic_count=len(sym_definition.labels),
        sym_primitive_count=len(sym_definition.primitives),
        point_group=sym_definition.point_group,
        irreps=sym_definition.irreps,
        raw_workdir=str(raw_dir),
        sym_workdir=str(sym_dir),
        contract_errors=tuple(errors),
        raw_point_group=raw_definition.point_group,
        raw_names=raw_definition.names,
        sym_names=sym_definition.names,
        raw_labels=raw_definition.labels,
        sym_labels=sym_definition.labels,
        raw_primitive_signatures=raw_signatures,
        sym_primitive_signatures=sym_signatures,
        raw_coordinate_kind_counts=raw_kind_counts,
        sym_coordinate_kind_counts=sym_kind_counts,
        totally_symmetric_count=totally_symmetric_count,
    )


def write_gaussian_gic_input(definition: GICDefinition, path: Path) -> Path:
    """Write the Gaussian-readable GIC block stored in a definition."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = definition.gaussian_input or _gaussian_input_from_definition(definition)
    target.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return target


def _primitive_signature(primitive: Primitive) -> str:
    atoms = ",".join(str(int(idx) + 1) for idx in primitive.atoms)
    ref = ",".join(str(int(idx) + 1) for idx in getattr(primitive, "ref", ()))
    suffix = f":ref={ref}" if ref else ""
    return f"{primitive.kind}:mode={int(getattr(primitive, 'mode', 0))}:atoms={atoms}{suffix}"




def _coordinate_kind(name: str, label: str) -> str:
    text = f"{name} {label}".lower()
    if any(marker in text for marker in ("str", "stre", "r(")):
        return "bond"
    if any(marker in text for marker in ("lin", "l(")):
        return "linear_bend"
    if any(marker in text for marker in ("oop", "out", "impd", "u(", "h(")):
        return "out_of_plane"
    if any(marker in text for marker in ("tor", "pck", "phi", "d(")):
        return "dihedral"
    if any(marker in text for marker in ("ang", "bend", "rock", "symd", "rdef", "a(")):
        return "angle"
    return "unknown"


def _definition_coordinate_kind_counts(definition: GICDefinition) -> dict[str, int]:
    counts: dict[str, int] = {}
    u_matrix = np.asarray(definition.u_matrix, dtype=float)
    for column in range(u_matrix.shape[1]):
        rows = np.flatnonzero(np.abs(u_matrix[:, column]) > 1.0e-12)
        kinds = {definition.primitives[int(row)].kind for row in rows}
        if len(kinds) == 1:
            kind = next(iter(kinds))
        elif not kinds:
            kind = "unknown"
        else:
            kind = "mixed:" + "+".join(sorted(kinds))
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def parse_gicforge_line(line: str) -> tuple[str, list[tuple[float, Primitive]], str] | None:
    stripped = line.strip()
    if not stripped or "=" not in stripped:
        return None
    name = stripped.split("=", 1)[0].replace("(Inactive)", "").strip()
    rhs = stripped.split("=", 1)[1].strip()
    terms: list[tuple[float, Primitive]] = []
    number = r"[+-]?\s*(?:\d+(?:\.\d*)?|\.\d+)(?:[EDed][+-]?\d+)?"
    for match in re.finditer(rf"({number})\s*\*\s*([RADLUFH])\(([^)]*)\)", rhs):
        coeff = float(match.group(1).replace(" ", "").replace("D", "E").replace("d", "e"))
        terms.append((coeff, _gicforge_primitive(match.group(2), match.group(3))))
    if not terms:
        simple = re.search(r"\b([RADLUFH])\(([^)]*)\)", rhs)
        if simple:
            terms.append((1.0, _gicforge_primitive(simple.group(1), simple.group(2))))
    if not terms:
        return None
    return name, terms, rhs


def _write_gicforge_inputs(
    workdir: Path,
    atoms: tuple[str, ...],
    coords: np.ndarray,
    *,
    symmetrize: bool,
    symmetrize_cartesians: bool = False,
    extra_keywords: tuple[str, ...] = (),
) -> None:
    base_keywords = ["GNIC", "BMAT", "ECKART", "GDV", "CLEAN"]
    if symmetrize:
        base_keywords.insert(1, "GICSYM")
    if symmetrize_cartesians:
        base_keywords.insert(1, "SYCART")
    keywords = "# " + " ".join((*base_keywords, *extra_keywords))
    (workdir / "provin").write_text(
        f"{keywords}\n\nORACLE GIC definition utility\n\n0 1\n",
        encoding="utf-8",
    )
    lines = [str(len(atoms)), ""]
    for atom, (x, y, z) in zip(atoms, coords):
        lines.append(f"{atom:>4s} {x: 16.8E} {y: 16.8E} {z: 16.8E}")
    (workdir / "xyzin").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _known_gicforge_outputs() -> tuple[str, ...]:
    return (
        "gicforge.out",
        "provout",
        "gauin",
        "gauin.raw",
        "gauin.symm",
        "gicsym",
        "gic_symmetry_diagnostics.json",
        "sycart.xyz",
        "symmetrized.xyz",
        "msrin",
        "VPT2in",
        "bmat.out",
        "gic_definition.json",
    )


def _read_sycart_coordinates(path: Path, atoms: tuple[str, ...]) -> np.ndarray:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < len(atoms) + 2:
        raise GICDefinitionError(f"Cannot read SYCART coordinates from {path}")
    coords = []
    read_atoms = []
    for raw in lines[2 : 2 + len(atoms)]:
        fields = raw.split()
        if len(fields) < 4:
            raise GICDefinitionError(f"Malformed SYCART coordinate line in {path}: {raw!r}")
        read_atoms.append(atomic_symbol(atomic_number(fields[0])))
        coords.append((float(fields[1]), float(fields[2]), float(fields[3])))
    if tuple(read_atoms) != tuple(atomic_symbol(atomic_number(atom)) for atom in atoms):
        raise GICDefinitionError(f"SYCART atom order changed in {path}")
    return np.asarray(coords, dtype=float)


def _validated_coordinates(coordinates: np.ndarray, natoms: int) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=float)
    if coords.shape != (natoms, 3):
        raise GICDefinitionError(
            f"Expected coordinates with shape ({natoms}, 3), got {coords.shape}"
        )
    return coords


def _gicforge_cartesian_from_gauin(gauin: Path, natoms: int) -> np.ndarray:
    lines = Path(gauin).read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, raw in enumerate(lines):
        parts = raw.split()
        if len(parts) == 2 and all(part.lstrip("+-").isdigit() for part in parts):
            coords: list[tuple[float, float, float]] = []
            for coord_line in lines[idx + 1 :]:
                fields = coord_line.split()
                if not fields:
                    break
                if len(fields) < 4 or not fields[0].lstrip("+-").isdigit():
                    break
                coords.append((float(fields[1]), float(fields[2]), float(fields[3])))
                if len(coords) == natoms:
                    break
            array = np.asarray(coords, dtype=float)
            if array.shape == (natoms, 3):
                return array
    raise GICDefinitionError(f"Cannot read oriented Cartesian block from {gauin}")


def _read_gicforge_irreps(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    irreps: dict[str, str] = {}
    for idx, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if idx == 0 and raw.lower().startswith("name,"):
            continue
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) >= 2 and parts[0]:
            irreps[parts[0]] = parts[1]
    return irreps


def _gicforge_point_group(provout: Path) -> str:
    text = provout.read_text(encoding="utf-8", errors="replace") if provout.exists() else ""
    match = re.search(r"Point Group from symm\.f:\s*([A-Za-z0-9]+)", text)
    return match.group(1) if match else "UNKNOWN"


def _gicforge_primitive(kind: str, atoms_text: str) -> Primitive:
    values = tuple(int(item.strip()) for item in atoms_text.split(",") if item.strip())
    atoms = tuple(value - 1 for value in values)
    if kind == "R" and len(atoms) == 2:
        return Primitive("bond", atoms)
    if kind == "A" and len(atoms) == 3:
        return Primitive("angle", atoms)
    if kind in {"D", "F"} and len(atoms) == 4:
        return Primitive("dihedral", atoms)
    if kind == "U" and len(atoms) == 4:
        return Primitive("out_of_plane", atoms)
    if kind == "H" and len(atoms) == 4:
        return Primitive("out_of_plane_height", atoms)
    if kind == "L" and len(values) == 5:
        mode_token = values[4]
        mode = mode_token if mode_token in {-1, -2} else -1
        ref = (values[3] - 1,) if values[3] > 0 else ()
        return Primitive("linear_bend", atoms[:3], mode=mode, ref=ref)
    raise GICDefinitionError(f"Unsupported GICForge primitive {kind}({atoms_text})")


def _gic_aliases(terms: list[tuple[float, Primitive]]) -> str:
    return " ".join(sorted({_primitive_alias(primitive) for _coeff, primitive in terms}))


def _primitive_alias(primitive: Primitive) -> str:
    atoms = ",".join(str(atom + 1) for atom in primitive.atoms)
    if primitive.kind == "bond":
        return f"R({atoms})"
    if primitive.kind == "angle":
        return f"A({atoms})"
    if primitive.kind == "dihedral":
        return f"D({atoms})"
    if primitive.kind == "out_of_plane":
        return f"U({atoms})"
    if primitive.kind == "out_of_plane_height":
        return f"H({atoms})"
    if primitive.kind == "linear_bend":
        reference = primitive.ref[0] + 1 if len(primitive.ref) == 1 else 0
        return f"L({atoms},{reference},{primitive.mode})"
    return f"{primitive.kind}({atoms})"


def _primitive_to_dict(primitive: Primitive) -> dict:
    return {
        "kind": primitive.kind,
        "atoms": [int(atom) for atom in primitive.atoms],
        "mode": int(primitive.mode),
        "ref": [int(atom) for atom in primitive.ref],
    }


def _primitive_from_dict(data: dict) -> Primitive:
    return Primitive(
        str(data["kind"]),
        tuple(int(atom) for atom in data.get("atoms", ())),
        mode=int(data.get("mode", 0)),
        ref=tuple(int(atom) for atom in data.get("ref", ())),
    )


def _validate_primitive(primitive: Primitive, natoms: int, index: int) -> None:
    expected = {
        "bond": 2,
        "inverse_bond": 2,
        "angle": 3,
        "dihedral": 4,
        "out_of_plane": 4,
        "out_of_plane_height": 4,
        "linear_bend": 3,
    }.get(primitive.kind)
    if expected is None:
        raise GICDefinitionError(f"Primitive {index} has unsupported kind {primitive.kind!r}")
    if len(primitive.atoms) != expected:
        raise GICDefinitionError(
            f"Primitive {index} kind {primitive.kind!r} expects {expected} atoms"
        )
    if any(atom < 0 or atom >= natoms for atom in primitive.atoms):
        raise GICDefinitionError(f"Primitive {index} contains atom index outside 0..{natoms - 1}")
    if primitive.kind == "linear_bend" and primitive.mode not in {-1, -2}:
        raise GICDefinitionError(f"Primitive {index} linear bend has invalid mode {primitive.mode}")


def _gic_definition_provenance(
    run_dir: Path, result: GICForgeResult, gauin: Path
) -> dict[str, str]:
    executable = getattr(result, "executable", None)
    manifest = getattr(result, "manifest", None)
    provenance: dict[str, str] = {
        "backend": "gicforge",
        "gic_source_file": str(gauin),
    }
    if executable is not None:
        provenance["backend_executable"] = str(executable)
    for name in ("provin", "xyzin"):
        path = run_dir / name
        if path.exists():
            provenance[f"{name}_sha256"] = sha256_file(path)
    for name in (
        "gauin",
        "gauin.raw",
        "gauin.symm",
        "gicsym",
        "gic_symmetry_diagnostics.json",
        "sycart.xyz",
        "symmetrized.xyz",
        "bmat.out",
    ):
        path = run_dir / name
        if path.exists():
            provenance[f"{name}_sha256"] = sha256_file(path)
    if executable is not None:
        exe_path = Path(executable)
        if exe_path.exists() and exe_path.is_file():
            provenance["backend_executable_sha256"] = sha256_file(exe_path)
    if manifest is not None:
        manifest_path = Path(manifest)
        if manifest_path.exists():
            provenance["gicforge_manifest_sha256"] = sha256_file(manifest_path)
    commit = _git_commit()
    if commit:
        provenance["git_commit"] = commit
    dirty = _git_dirty()
    if dirty is not None:
        provenance["git_dirty"] = "true" if dirty else "false"
    return provenance


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def _git_dirty() -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        return None
    return bool(completed.stdout.strip())


def _gaussian_input_from_definition(definition: GICDefinition) -> str:
    lines = []
    for name, column in zip(definition.names, np.asarray(definition.u_matrix, dtype=float).T):
        terms = []
        for coeff, primitive in zip(column, definition.primitives):
            if abs(float(coeff)) <= 1.0e-12:
                continue
            terms.append(f"{float(coeff): .8f}*{_primitive_expression(primitive)}")
        if terms:
            lines.append(f"{name}=[{'+'.join(terms)}]")
    return "\n".join(lines) + "\n"


def _primitive_expression(primitive: Primitive) -> str:
    atoms = tuple(atom + 1 for atom in primitive.atoms)
    if primitive.kind == "bond":
        return f"R({atoms[0]:3d},{atoms[1]:3d})"
    if primitive.kind == "angle":
        return f"A({atoms[0]:3d},{atoms[1]:3d},{atoms[2]:3d})"
    if primitive.kind == "dihedral":
        return f"D({atoms[0]:3d},{atoms[1]:3d},{atoms[2]:3d},{atoms[3]:3d})"
    if primitive.kind == "out_of_plane":
        return f"U({atoms[0]:3d},{atoms[1]:3d},{atoms[2]:3d},{atoms[3]:3d})"
    if primitive.kind == "out_of_plane_height":
        return f"H({atoms[0]:3d},{atoms[1]:3d},{atoms[2]:3d},{atoms[3]:3d})"
    if primitive.kind == "linear_bend":
        reference = primitive.ref[0] + 1 if len(primitive.ref) == 1 else 0
        return (
            f"L({atoms[0]:3d},{atoms[1]:3d},{atoms[2]:3d},"
            f"{reference:3d},{primitive.mode:3d})"
        )
    raise GICDefinitionError(
        f"Unsupported primitive kind for Gaussian GIC output: {primitive.kind}"
    )
