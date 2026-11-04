from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Iterable

import numpy as np

from matrix_chem import preprocess_to_enriched_xyz, read_enriched_xyz, write_validation_section
from matrix_engines import gicforge_fortran_layout, run_gicforge_reference

from .corpus import default_gic_corpus_root
from .definition import build_gic_definition_from_xyzin, write_gicforge_build_sections
from .evaluation import build_gic_b_matrix
from .models import GICDefinition
from .policy import SPECIAL_REDUCTION_CLASS, primitive_reduction_class


DEFAULT_FORTRAN_AUDIT_MOLECULES = (
    "pyrrole.inp",
    "benzene.inp",
    "pyridine.inp",
    "pyrimidine.inp",
    "naphtalene.inp",
    "phenantrene.inp",
    "anthracene.inp",
    "pyrene.inp",
    "fluorene.inp",
    "azulene.inp",
    "norbornane.inp",
    "norbornene.inp",
    "norbornadiene.inp",
    "norcamphor.inp",
    "spiro.inp",
    "c2h2.inp",
    "c4s.inp",
    "thujone.inp",
    "ribose.inp",
    "cubane.inp",
    "sf6_octahedral.inp",
    "ferrocene.inp",
    "ferrocene_staggered.inp",
    "progly1_nohbond.inp",
    "cyclottane.inp",
)


@dataclass(frozen=True)
class GICForgeFortranAuditResult:
    molecule: str
    source: Path
    status: str
    point_group: str = ""
    oracle_rank: int = 0
    fortran_rank: int = 0
    oracle_shape: tuple[int, int] = (0, 0)
    fortran_shape: tuple[int, int] = (0, 0)
    oracle_row_rank: int = 0
    fortran_row_rank: int = 0
    row_space_residual: float | None = None
    oracle_ring_pucker_components: int = 0
    fortran_label_prefixes: tuple[str, ...] = ()
    projector_status: str = "UNKNOWN"
    symmetry_group_count: int = 0
    special_symmetry_group_count: int = 0
    mixed_symmetry_group_count: int = 0
    total_symmetric_gic_count: int = 0
    salc_coefficient_gic_count: int = 0
    salc_coefficient_max_norm_error: float | None = None
    message: str = ""
    workdir: Path | None = None

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_record(self, *, root: Path | None = None) -> dict[str, object]:
        data = asdict(self)
        data["source"] = _display_path(self.source, root=root)
        data["workdir"] = None if self.workdir is None else str(self.workdir)
        return data


@dataclass(frozen=True)
class GICForgeFortranAudit:
    root: Path
    workdir: Path | None
    tolerance: float
    results: tuple[GICForgeFortranAuditResult, ...]

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.status == "PASS")

    @property
    def failed(self) -> int:
        return sum(1 for result in self.results if result.status == "FAIL")

    @property
    def skipped(self) -> int:
        return sum(1 for result in self.results if result.status == "SKIP")

    @property
    def errored(self) -> int:
        return sum(1 for result in self.results if result.status == "ERROR")

    @property
    def max_row_space_residual(self) -> float | None:
        values = [
            float(result.row_space_residual)
            for result in self.results
            if result.row_space_residual is not None and np.isfinite(result.row_space_residual)
        ]
        return max(values) if values else None

    @property
    def mixed_symmetry_groups(self) -> int:
        return sum(result.mixed_symmetry_group_count for result in self.results)

    @property
    def salc_coefficient_gics(self) -> int:
        return sum(result.salc_coefficient_gic_count for result in self.results)

    @property
    def max_salc_coefficient_norm_error(self) -> float | None:
        values = [
            float(result.salc_coefficient_max_norm_error)
            for result in self.results
            if result.salc_coefficient_max_norm_error is not None
            and np.isfinite(result.salc_coefficient_max_norm_error)
        ]
        return max(values) if values else None


def audit_gicforge_fortran_corpus(
    *,
    root: Path | None = None,
    molecules: Iterable[str | Path] | None = None,
    workdir: Path | None = None,
    repo_root: Path | None = None,
    limit: int | None = None,
    tolerance: float = 2.0e-8,
) -> GICForgeFortranAudit:
    corpus_root = Path(root) if root is not None else default_gic_corpus_root(repo_root)
    selected = tuple(molecules or DEFAULT_FORTRAN_AUDIT_MOLECULES)
    if limit is not None:
        if limit < 0:
            raise ValueError("fortran audit limit cannot be negative")
        selected = selected[:limit]

    layout = gicforge_fortran_layout(repo_root)
    if shutil.which("gfortran") is None and not layout.reference_executable.is_file():
        results = tuple(
            GICForgeFortranAuditResult(
                molecule=str(item),
                source=_resolve_corpus_source(corpus_root, item),
                status="SKIP",
                message="gfortran is not available and the GICForge reference is not built",
            )
            for item in selected
        )
        return GICForgeFortranAudit(
            root=corpus_root,
            workdir=Path(workdir) if workdir is not None else None,
            tolerance=float(tolerance),
            results=results,
        )

    if workdir is None:
        with TemporaryDirectory(prefix="matrix_smith_fortran_audit_") as tmp:
            results = _audit_many(
                corpus_root,
                selected,
                workdir=Path(tmp),
                repo_root=repo_root,
                tolerance=float(tolerance),
            )
        return GICForgeFortranAudit(corpus_root, None, float(tolerance), results)

    target = Path(workdir)
    target.mkdir(parents=True, exist_ok=True)
    results = _audit_many(
        corpus_root,
        selected,
        workdir=target,
        repo_root=repo_root,
        tolerance=float(tolerance),
    )
    return GICForgeFortranAudit(corpus_root, target, float(tolerance), results)


def gicforge_fortran_audit_records(
    audit: GICForgeFortranAudit,
) -> list[dict[str, object]]:
    return [result.to_record(root=audit.root) for result in audit.results]


def format_gicforge_fortran_audit_summary(audit: GICForgeFortranAudit) -> list[str]:
    max_residual = audit.max_row_space_residual
    lines = [
        f"ROOT {audit.root}",
        f"WORKDIR {audit.workdir or 'TEMPORARY'}",
        f"TOLERANCE {audit.tolerance:.12g}",
        f"CASES {len(audit.results)}",
        f"PASS {audit.passed}",
        f"FAIL {audit.failed}",
        f"ERROR {audit.errored}",
        f"SKIP {audit.skipped}",
        "MAX_ROW_SPACE_RESIDUAL " + ("NONE" if max_residual is None else f"{max_residual:.12g}"),
        f"MIXED_SYMMETRY_GROUPS {audit.mixed_symmetry_groups}",
        f"SALC_COEFFICIENT_GICS {audit.salc_coefficient_gics}",
        "MAX_SALC_COEFFICIENT_NORM_ERROR "
        + (
            "NONE"
            if audit.max_salc_coefficient_norm_error is None
            else f"{audit.max_salc_coefficient_norm_error:.12g}"
        ),
    ]
    lines.extend(format_gicforge_fortran_audit_cases(audit))
    return lines


def format_gicforge_fortran_audit_cases(
    audit: GICForgeFortranAudit,
    *,
    status: str = "all",
) -> list[str]:
    normalized = status.strip().upper()
    if normalized not in {"ALL", "PASS", "FAIL", "ERROR", "SKIP"}:
        raise ValueError(f"unsupported fortran audit status filter: {status}")
    lines: list[str] = []
    for result in audit.results:
        if normalized != "ALL" and result.status != normalized:
            continue
        residual = (
            "NONE" if result.row_space_residual is None else f"{result.row_space_residual:.12g}"
        )
        lines.append(
            "CASE "
            f"{result.status} "
            f"{_display_path(result.source, root=audit.root)} "
            f"point_group={result.point_group or 'UNKNOWN'} "
            f"oracle_rank={result.oracle_rank} "
            f"fortran_rank={result.fortran_rank} "
            f"oracle_row_rank={result.oracle_row_rank} "
            f"fortran_row_rank={result.fortran_row_rank} "
            f"row_space_residual={residual} "
            f"projector_status={result.projector_status} "
            f"symmetry_groups={result.symmetry_group_count} "
            f"special_symmetry_groups={result.special_symmetry_group_count} "
            f"mixed_symmetry_groups={result.mixed_symmetry_group_count} "
            f"total_symmetric_gics={result.total_symmetric_gic_count}"
            f" salc_coefficient_gics={result.salc_coefficient_gic_count}"
            " salc_norm_error="
            + (
                "NONE"
                if result.salc_coefficient_max_norm_error is None
                else f"{result.salc_coefficient_max_norm_error:.12g}"
            )
            + (f" message={result.message}" if result.message else "")
        )
    return lines


def _audit_many(
    corpus_root: Path,
    selected: tuple[str | Path, ...],
    *,
    workdir: Path,
    repo_root: Path | None,
    tolerance: float,
) -> tuple[GICForgeFortranAuditResult, ...]:
    return tuple(
        _audit_one(
            _resolve_corpus_source(corpus_root, item),
            workdir=workdir / _audit_slug(item),
            repo_root=repo_root,
            tolerance=tolerance,
        )
        for item in selected
    )


def _audit_one(
    source: Path,
    *,
    workdir: Path,
    repo_root: Path | None,
    tolerance: float,
) -> GICForgeFortranAuditResult:
    molecule = source.name
    try:
        case_dir = Path(workdir)
        case_dir.mkdir(parents=True, exist_ok=True)
        xyzin = case_dir / f"{source.stem}.xyzin"
        preprocess_to_enriched_xyz(source, xyzin)
        write_validation_section(xyzin)
        definition = write_gicforge_build_sections(xyzin)
        symmetrized = build_gic_definition_from_xyzin(xyzin, symmetrize=True)
        projector = _projector_audit(definition, symmetrized)
        projector_required = _requires_point_group_projector(definition.point_group)
        geometry = read_enriched_xyz(xyzin)
        reference_keywords = _reference_keywords_for_definition(definition)
        reference = run_gicforge_reference(
            case_dir / "reference",
            atoms=geometry.atoms,
            coordinates_angstrom=geometry.coordinates_angstrom,
            point_group="C1",
            title=source.stem,
            keywords=reference_keywords,
            repo_root=repo_root,
        )
        b_coordinates = (
            reference.eckart_coordinates_angstrom
            if "ECKART" in reference_keywords
            and reference.eckart_coordinates_angstrom is not None
            else geometry.coordinates_angstrom
        )
        oracle_b = np.asarray(
            build_gic_b_matrix(definition, coordinates_angstrom=b_coordinates).rows,
            dtype=float,
        )
        fortran_b = np.asarray(reference.b_matrix_rows, dtype=float)
        oracle_row_rank = _row_space_rank(oracle_b)
        fortran_row_rank = _row_space_rank(fortran_b)
        residual = _row_space_residual(oracle_b, fortran_b)
        oracle_rank = int(definition.rank)
        fortran_rank = int(reference.final_counts[-1])
        salc_norm_error = float(projector["salc_coefficient_max_norm_error"])
        passed = (
            oracle_rank == fortran_rank
            and oracle_row_rank == fortran_row_rank == oracle_rank
            and oracle_b.shape == fortran_b.shape
            and residual <= tolerance
            and (not projector_required or projector["projector_status"] == "POINT_GROUP_PROJECTOR")
            and projector["mixed_reduction_class_group_count"] == 0
            and salc_norm_error <= tolerance
        )
        return GICForgeFortranAuditResult(
            molecule=molecule,
            source=source,
            status="PASS" if passed else "FAIL",
            point_group=definition.point_group,
            oracle_rank=oracle_rank,
            fortran_rank=fortran_rank,
            oracle_shape=tuple(int(value) for value in oracle_b.shape),
            fortran_shape=tuple(int(value) for value in fortran_b.shape),
            oracle_row_rank=oracle_row_rank,
            fortran_row_rank=fortran_row_rank,
            row_space_residual=float(residual),
            oracle_ring_pucker_components=sum(
                1 for gic in definition.gics if gic.family == "RING_PUCKER_COMPONENT"
            ),
            fortran_label_prefixes=tuple(sorted({label[:4] for label in reference.gic_labels})),
            projector_status=str(projector["projector_status"]),
            symmetry_group_count=int(projector["symmetry_group_count"]),
            special_symmetry_group_count=int(projector["special_symmetry_group_count"]),
            mixed_symmetry_group_count=int(projector["mixed_symmetry_group_count"]),
            total_symmetric_gic_count=int(projector["total_symmetric_gic_count"]),
            salc_coefficient_gic_count=int(projector["salc_coefficient_gic_count"]),
            salc_coefficient_max_norm_error=salc_norm_error,
            message=""
            if passed
            else _failure_message(
                residual,
                tolerance,
                projector,
                projector_required=projector_required,
            ),
            workdir=case_dir,
        )
    except Exception as exc:
        return GICForgeFortranAuditResult(
            molecule=molecule,
            source=source,
            status="ERROR",
            message=f"{type(exc).__name__}: {exc}",
            workdir=workdir,
        )


def _projector_audit(
    source_definition: GICDefinition,
    symmetrized_definition: GICDefinition,
) -> dict[str, object]:
    diagnostics = symmetrized_definition.symmetry_diagnostics
    if diagnostics is None:
        return {
            "projector_status": "MISSING",
            "symmetry_group_count": 0,
            "special_symmetry_group_count": 0,
            "mixed_symmetry_group_count": 0,
            "mixed_reduction_class_group_count": 0,
            "total_symmetric_gic_count": 0,
            "salc_coefficient_gic_count": 0,
            "salc_coefficient_max_norm_error": 0.0,
        }
    source_family_by_name = {gic.name: gic.family for gic in source_definition.gics}
    mixed_groups = 0
    mixed_reduction_class_groups = 0
    special_groups = 0
    for group in diagnostics.groups:
        source_families = {
            source_family_by_name[name]
            for name in group.source_gics
            if name in source_family_by_name
        }
        if source_families and source_families != {group.family}:
            mixed_groups += 1
        reduction_classes = {
            primitive_reduction_class(family) for family in source_families
        }
        if len(reduction_classes) > 1:
            mixed_reduction_class_groups += 1
        if primitive_reduction_class(group.family) == SPECIAL_REDUCTION_CLASS:
            special_groups += 1
    return {
        "projector_status": diagnostics.method,
        "symmetry_group_count": len(diagnostics.groups),
        "special_symmetry_group_count": special_groups,
        "mixed_symmetry_group_count": mixed_groups,
        "mixed_reduction_class_group_count": mixed_reduction_class_groups,
        "total_symmetric_gic_count": len(diagnostics.total_symmetric_gics),
        "salc_coefficient_gic_count": _salc_coefficient_gic_count(symmetrized_definition),
        "salc_coefficient_max_norm_error": _salc_coefficient_max_norm_error(symmetrized_definition),
    }


def _salc_coefficient_gic_count(definition: GICDefinition) -> int:
    return sum(1 for gic in definition.gics if len(gic.coefficients) > 1)


def _salc_coefficient_max_norm_error(definition: GICDefinition) -> float:
    errors = []
    for gic in definition.gics:
        if len(gic.coefficients) <= 1:
            continue
        norm2 = sum(float(coefficient) ** 2 for _primitive_id, coefficient in gic.coefficients)
        errors.append(abs(norm2 - 1.0))
    return max(errors) if errors else 0.0


def _failure_message(
    residual: float,
    tolerance: float,
    projector: dict[str, object],
    *,
    projector_required: bool,
) -> str:
    reasons: list[str] = []
    if residual > tolerance:
        reasons.append("row-space residual exceeds tolerance")
    if projector_required and projector["projector_status"] != "POINT_GROUP_PROJECTOR":
        reasons.append("symmetry projector was not applied")
    if int(projector["mixed_reduction_class_group_count"]):
        reasons.append("symmetry projector mixed intra- and inter-fragment coordinates")
    salc_norm_error = float(projector["salc_coefficient_max_norm_error"])
    if salc_norm_error > tolerance:
        reasons.append("SALC coefficient normalization exceeds tolerance")
    if not reasons:
        reasons.append("rank, shape or row-space residual mismatch")
    return "; ".join(reasons)


def _requires_point_group_projector(point_group: str) -> bool:
    normalized = _normalized_point_group(point_group)
    return normalized not in {"", "C1", "CINFV", "DINFH"}


def _reference_keywords_for_definition(
    definition: GICDefinition,
) -> tuple[str, ...]:
    # The reference Linear flag is set in the Eckart orientation path.
    if _normalized_point_group(definition.point_group) in {"CINFV", "DINFH"}:
        return ("ECKART", "GNIC", "BMAT")
    return ("GNIC", "BMAT")


def _normalized_point_group(point_group: str) -> str:
    return point_group.strip().upper().replace("∞", "INF")


def _row_space_basis(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    if matrix.size == 0:
        return np.zeros((0, matrix.shape[1] if matrix.ndim == 2 else 0)), 0
    _u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    if not singular_values.size:
        return np.zeros((0, matrix.shape[1])), 0
    tolerance = max(matrix.shape) * float(singular_values[0]) * np.finfo(float).eps * 10.0
    rank = int(np.sum(singular_values > tolerance))
    return vh[:rank], rank


def _row_space_rank(matrix: np.ndarray) -> int:
    return _row_space_basis(matrix)[1]


def _row_space_residual(left: np.ndarray, right: np.ndarray) -> float:
    left_basis, left_rank = _row_space_basis(left)
    right_basis, right_rank = _row_space_basis(right)
    if left_rank != right_rank:
        return float("inf")
    if left_rank == 0:
        return 0.0
    projected = left_basis @ right_basis.T @ right_basis
    return float(np.linalg.norm(left_basis - projected) / np.sqrt(float(left_rank)))


def _resolve_corpus_source(root: Path, item: str | Path) -> Path:
    path = Path(item)
    if path.is_absolute() or path.exists():
        return path
    return Path(root) / path


def _audit_slug(item: str | Path) -> str:
    stem = Path(item).stem if Path(item).suffix else str(item)
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)


def _display_path(path: Path, *, root: Path | None) -> str:
    target = Path(path)
    if root is not None:
        try:
            return str(target.relative_to(root))
        except ValueError:
            pass
    return str(target)
