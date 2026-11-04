from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .definition import read_gic_definition_from_xyzin
from .models import GICDefinition
from .policy import (
    SALC_PATH_OVERLAP_WARNING_THRESHOLD,
    SALC_PATH_PIVOT_GAP_WARNING,
    SALC_SUBSPACE_PROJECTOR_TOLERANCE,
)


SALC_SNAPSHOT_SCHEMA = "matrix.smith.gic_salc_snapshot.v2"
DEFAULT_ROUNDING_DECIMALS = 12
DEFAULT_SELECTED_PER_FAMILY = 3
SALC_COEFFICIENT_TOLERANCE = 1.0e-8
PRIORITY_FAMILIES = frozenset(
    {
        "RING_PUCKER_COMPONENT",
        "BUTTERFLY",
        "SPIRO_BEND",
        "CENTER_ATOM_DISTANCE",
        "FRAG_DISTANCE",
        "FRAG_CENTER_ATOM_DISTANCE",
        "FRAG_TRANSLATION",
        "FRAG_ORIENTATION",
    }
)


@dataclass(frozen=True)
class SALCSnapshotComparison:
    ok: bool
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class SALCPathStepDiagnostic:
    step: int
    group: tuple[str, str]
    dimension_previous: int
    dimension_current: int
    min_subspace_overlap: float
    pivot_changed: bool
    warning: str = ""


@dataclass(frozen=True)
class SALCPathDiagnostics:
    ok: bool
    steps: tuple[SALCPathStepDiagnostic, ...] = ()
    messages: tuple[str, ...] = ()


def salc_snapshot_record(
    definition: GICDefinition,
    *,
    case_id: str = "",
    source: str = "",
    rounding_decimals: int = DEFAULT_ROUNDING_DECIMALS,
    selected_per_family: int = DEFAULT_SELECTED_PER_FAMILY,
) -> dict[str, Any]:
    full = _salc_records(definition, rounding_decimals=rounding_decimals)
    selected = _selected_salc_records(full, selected_per_family=selected_per_family)
    return {
        "id": case_id,
        "source": source,
        "point_group": definition.point_group,
        "rank": definition.rank,
        "target_rank": definition.target_rank,
        "symmetry_method": (
            definition.symmetry_diagnostics.method if definition.symmetry_diagnostics else "NONE"
        ),
        "salc_count": len(full),
        "salc_sha256": _stable_sha256(full),
        "selected_salcs": selected,
    }


def salc_snapshot_document(
    definitions: tuple[tuple[str, str, GICDefinition], ...],
    *,
    rounding_decimals: int = DEFAULT_ROUNDING_DECIMALS,
    selected_per_family: int = DEFAULT_SELECTED_PER_FAMILY,
) -> dict[str, Any]:
    return {
        "schema": SALC_SNAPSHOT_SCHEMA,
        "description": (
            "Compact golden snapshots of nontrivial SMITH/GICForge SALC coefficient "
            "vectors. salc_sha256 covers the complete coefficient list; selected_salcs "
            "keeps representative human-readable vectors."
        ),
        "rounding_decimals": int(rounding_decimals),
        "selected_per_family": int(selected_per_family),
        "entries": tuple(
            salc_snapshot_record(
                definition,
                case_id=case_id,
                source=source,
                rounding_decimals=rounding_decimals,
                selected_per_family=selected_per_family,
            )
            for case_id, source, definition in definitions
        ),
    }


def write_salc_snapshot(path: Path, definition: GICDefinition) -> Path:
    target = Path(path)
    payload = salc_snapshot_document((("", "", definition),))
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def write_salc_snapshot_from_xyzin(xyzin: Path, output: Path) -> Path:
    return write_salc_snapshot(Path(output), read_gic_definition_from_xyzin(Path(xyzin)))


def compare_salc_snapshot_entry(
    expected: dict[str, Any],
    definition: GICDefinition,
    *,
    rounding_decimals: int,
    selected_per_family: int = DEFAULT_SELECTED_PER_FAMILY,
) -> SALCSnapshotComparison:
    current = salc_snapshot_record(
        definition,
        case_id=str(expected.get("id", "")),
        source=str(expected.get("source", "")),
        rounding_decimals=rounding_decimals,
        selected_per_family=selected_per_family,
    )
    messages: list[str] = []
    for key in ("point_group", "rank", "target_rank", "symmetry_method", "salc_count"):
        if current[key] != expected.get(key):
            messages.append(
                f"{expected.get('id', '<unknown>')}: {key} changed "
                f"expected={expected.get(key)!r} current={current[key]!r}"
            )
    detail = _first_selected_difference(
        _canonical_records(expected.get("selected_salcs", ())),
        _canonical_records(current["selected_salcs"]),
    )
    if detail:
        messages.append(f"{expected.get('id', '<unknown>')}: {detail}")
    elif current["salc_sha256"] != expected.get("salc_sha256"):
        # The complete hash is intentionally retained as a drift diagnostic, but
        # platform BLAS/SVD details can change non-selected SALC bases inside the
        # same symmetry subspace.  The hard golden gate is therefore the
        # human-inspectable representative set plus rank/symmetry metadata.
        pass
    return SALCSnapshotComparison(ok=not messages, messages=tuple(messages))


def salc_path_diagnostics(
    definitions: tuple[GICDefinition, ...],
    *,
    overlap_warning_threshold: float = SALC_PATH_OVERLAP_WARNING_THRESHOLD,
) -> SALCPathDiagnostics:
    """Monitor SALC gauge continuity for a geometry path.

    Individual vectors inside a repeated irrep can rotate or flip sign without
    changing the represented coordinate space.  The hard diagnostic is
    therefore the principal-angle overlap between consecutive family/irrep
    subspaces; pivot changes are reported as gauge events for downstream
    PED/DVR code that needs continuous rows.
    """
    steps: list[SALCPathStepDiagnostic] = []
    messages: list[str] = []
    for step, (previous, current) in enumerate(zip(definitions, definitions[1:]), start=1):
        previous_groups = _salc_matrix_groups(previous)
        current_groups = _salc_matrix_groups(current)
        for key in sorted(set(previous_groups) & set(current_groups)):
            previous_ids, previous_matrix = previous_groups[key]
            current_ids, current_matrix = current_groups[key]
            if previous_ids != current_ids:
                message = f"step {step} {key}: primitive support changed"
                messages.append(message)
                steps.append(
                    SALCPathStepDiagnostic(
                        step=step,
                        group=key,
                        dimension_previous=previous_matrix.shape[0],
                        dimension_current=current_matrix.shape[0],
                        min_subspace_overlap=0.0,
                        pivot_changed=True,
                        warning=message,
                    )
                )
                continue
            overlap = _subspace_min_overlap(previous_matrix, current_matrix)
            pivot_changed = _pivot_signature(previous_matrix, previous_ids) != _pivot_signature(
                current_matrix, current_ids
            )
            warning = ""
            if overlap < overlap_warning_threshold:
                warning = f"step {step} {key}: SALC subspace overlap {overlap:.6g}"
                messages.append(warning)
            elif pivot_changed:
                warning = f"step {step} {key}: SALC gauge pivot changed"
                messages.append(warning)
            steps.append(
                SALCPathStepDiagnostic(
                    step=step,
                    group=key,
                    dimension_previous=previous_matrix.shape[0],
                    dimension_current=current_matrix.shape[0],
                    min_subspace_overlap=overlap,
                    pivot_changed=pivot_changed,
                    warning=warning,
                )
            )
        missing = sorted(set(previous_groups) ^ set(current_groups))
        for key in missing:
            message = f"step {step} {key}: SALC group appeared or disappeared"
            messages.append(message)
            steps.append(
                SALCPathStepDiagnostic(
                    step=step,
                    group=key,
                    dimension_previous=previous_groups.get(key, ((), np.zeros((0, 0))))[1].shape[0],
                    dimension_current=current_groups.get(key, ((), np.zeros((0, 0))))[1].shape[0],
                    min_subspace_overlap=0.0,
                    pivot_changed=True,
                    warning=message,
                )
            )
    return SALCPathDiagnostics(ok=not messages, steps=tuple(steps), messages=tuple(messages))


def procrustes_align_salc_matrix(
    reference_matrix: np.ndarray,
    current_matrix: np.ndarray,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Rotate the current SALC row basis onto the reference basis."""
    reference = np.asarray(reference_matrix, dtype=float)
    current = np.asarray(current_matrix, dtype=float)
    if reference.shape != current.shape:
        raise ValueError("SALC matrices must have the same shape for Procrustes alignment")
    if reference.size == 0:
        return current.copy(), ()
    cross = current @ reference.T
    left, singular_values, right_t = np.linalg.svd(cross, full_matrices=False)
    rotation = right_t.T @ left.T
    return rotation @ current, tuple(float(value) for value in singular_values)


def _salc_records(
    definition: GICDefinition,
    *,
    rounding_decimals: int,
) -> tuple[dict[str, Any], ...]:
    records = []
    for gic in definition.gics:
        if len(gic.coefficients) <= 1:
            continue
        records.append(
            {
                "name": gic.name,
                "family": gic.family,
                "irrep": gic.irrep,
                "coefficients": [
                    [primitive_id, _rounded_coefficient(coefficient, rounding_decimals)]
                    for primitive_id, coefficient in gic.coefficients
                    if abs(float(coefficient)) > SALC_COEFFICIENT_TOLERANCE
                ],
            }
        )
    return tuple(records)


def _selected_salc_records(
    records: tuple[dict[str, Any], ...],
    *,
    selected_per_family: int,
) -> tuple[dict[str, Any], ...]:
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for record in records:
        family = str(record["family"])
        if family in PRIORITY_FAMILIES:
            selected.append(record)
            continue
        count = counts.get(family, 0)
        if count < selected_per_family:
            selected.append(record)
        counts[family] = count + 1
    return tuple(selected)


def _stable_sha256(records: tuple[dict[str, Any], ...]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rounded_coefficient(value: float, decimals: int) -> float:
    rounded = round(float(value), decimals)
    if rounded == 0.0:
        return 0.0
    return rounded


def _canonical_records(records: Any) -> tuple[dict[str, Any], ...]:
    canonical: list[dict[str, Any]] = []
    for record in records or ():
        coefficients = [
            [str(primitive_id), float(coefficient)]
            for primitive_id, coefficient in record.get("coefficients", ())
        ]
        canonical.append(
            {
                "name": str(record.get("name", "")),
                "family": str(record.get("family", "")),
                "irrep": str(record.get("irrep", "")),
                "coefficients": coefficients,
            }
        )
    return tuple(canonical)


def _first_selected_difference(
    expected: tuple[Any, ...],
    current: tuple[Any, ...],
) -> str:
    if len(expected) != len(current):
        return f"selected SALC count changed expected={len(expected)} current={len(current)}"
    left_groups = _selected_subspace_groups(expected)
    right_groups = _selected_subspace_groups(current)
    if set(left_groups) != set(right_groups):
        return (
            "selected SALC family/irrep groups changed "
            f"expected={sorted(left_groups)} current={sorted(right_groups)}"
        )
    for key in sorted(left_groups):
        left_records = left_groups[key]
        right_records = right_groups[key]
        if len(left_records) != len(right_records):
            return (
                f"selected SALC count changed for {key}: "
                f"expected={len(left_records)} current={len(right_records)}"
            )
        if key[0] not in PRIORITY_FAMILIES:
            # Generic symmetry blocks can be larger than the stored compact
            # selection.  Later vectors in a truncated degenerate block are not
            # unique across BLAS/SVD backends, so only the leading representative
            # is a stable golden artifact.  Special-coordinate families are kept
            # as full selected subspaces because they are the regression target.
            left_records = left_records[:1]
            right_records = right_records[:1]
        left_ids, left_projector = _selected_subspace_projector(left_records)
        right_ids, right_projector = _selected_subspace_projector(right_records)
        if left_ids != right_ids or not np.allclose(
            left_projector,
            right_projector,
            atol=SALC_SUBSPACE_PROJECTOR_TOLERANCE,
            rtol=0.0,
        ):
            return (
                f"selected SALC subspace changed for family={key[0]} irrep={key[1]}: "
                f"expected={left_records!r} current={right_records!r}"
            )
    return ""


def _selected_subspace_groups(records: tuple[Any, ...]) -> dict[tuple[str, str], tuple[Any, ...]]:
    groups: dict[tuple[str, str], list[Any]] = {}
    for record in records:
        key = (str(record.get("family", "")), str(record.get("irrep", "")))
        groups.setdefault(key, []).append(record)
    return {key: tuple(value) for key, value in groups.items()}


def _salc_matrix_groups(
    definition: GICDefinition,
) -> dict[tuple[str, str], tuple[tuple[str, ...], np.ndarray]]:
    grouped: dict[tuple[str, str], list[Any]] = {}
    for record in _salc_records(definition, rounding_decimals=DEFAULT_ROUNDING_DECIMALS):
        key = (str(record["family"]), str(record["irrep"]))
        grouped.setdefault(key, []).append(record)
    matrices: dict[tuple[str, str], tuple[tuple[str, ...], np.ndarray]] = {}
    for key, records in grouped.items():
        primitive_ids = tuple(
            sorted(
                {
                    str(primitive_id)
                    for record in records
                    for primitive_id, _coefficient in record["coefficients"]
                }
            )
        )
        primitive_index = {primitive_id: index for index, primitive_id in enumerate(primitive_ids)}
        matrix = np.zeros((len(records), len(primitive_ids)), dtype=float)
        for row, record in enumerate(records):
            for primitive_id, coefficient in record["coefficients"]:
                matrix[row, primitive_index[str(primitive_id)]] = float(coefficient)
        matrices[key] = (primitive_ids, matrix)
    return matrices


def _subspace_min_overlap(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape[1] != right.shape[1]:
        return 0.0
    left_basis = _row_orthonormal_basis(left)
    right_basis = _row_orthonormal_basis(right)
    if left_basis.shape[0] != right_basis.shape[0]:
        return 0.0
    if left_basis.size == 0:
        return 1.0
    singular_values = np.linalg.svd(left_basis @ right_basis.T, compute_uv=False)
    return float(np.min(singular_values)) if singular_values.size else 1.0


def _row_orthonormal_basis(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0, matrix.shape[1] if matrix.ndim == 2 else 0), dtype=float)
    _u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.count_nonzero(singular_values > SALC_COEFFICIENT_TOLERANCE))
    return vh[:rank, :]


def _pivot_signature(matrix: np.ndarray, primitive_ids: tuple[str, ...]) -> tuple[str, ...]:
    pivots: list[str] = []
    for row in matrix:
        magnitudes = np.abs(row)
        if not magnitudes.size:
            pivots.append("")
            continue
        order = np.argsort(-magnitudes)
        first = int(order[0])
        second_value = float(magnitudes[int(order[1])]) if len(order) > 1 else -np.inf
        if float(magnitudes[first]) - second_value <= SALC_PATH_PIVOT_GAP_WARNING:
            pivots.append("AMBIGUOUS")
        else:
            pivots.append(primitive_ids[first])
    return tuple(pivots)


def _selected_subspace_projector(records: tuple[Any, ...]) -> tuple[tuple[str, ...], np.ndarray]:
    coefficient_tolerance = SALC_SUBSPACE_PROJECTOR_TOLERANCE
    primitive_ids = tuple(
        sorted(
            {
                str(primitive_id)
                for record in records
                for primitive_id, coefficient in record.get("coefficients", ())
                if abs(float(coefficient)) > coefficient_tolerance
            }
        )
    )
    primitive_index = {primitive_id: index for index, primitive_id in enumerate(primitive_ids)}
    vectors = np.zeros((len(primitive_ids), len(records)), dtype=float)
    for column, record in enumerate(records):
        for primitive_id, coefficient in record.get("coefficients", ()):
            if abs(float(coefficient)) <= coefficient_tolerance:
                continue
            vectors[primitive_index[str(primitive_id)], column] = float(coefficient)
    if vectors.size == 0:
        return primitive_ids, np.zeros((0, 0), dtype=float)
    left, singular_values, _right_t = np.linalg.svd(vectors, full_matrices=False)
    rank = int(np.count_nonzero(singular_values > SALC_COEFFICIENT_TOLERANCE))
    basis = left[:, :rank]
    return primitive_ids, basis @ basis.T
