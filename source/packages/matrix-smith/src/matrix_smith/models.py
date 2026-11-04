"""Stable data contracts for frozen SMITH/SONIC definitions."""

from __future__ import annotations

from dataclasses import dataclass

from .bmatrix import SparseBMatrix, SparseBRow
from .contracts import ORACLE_XYZ_GIC_SCHEMA
from .periodic_estimates import PeriodicCoordinateEstimate
from .policy import (
    FRAGMENT_MODE_NONE,
    SALC_PATH_OVERLAP_WARNING_THRESHOLD,
    SYMMETRY_OPERATION_TOLERANCE_ANGSTROM,
    XH_STRETCH_POLICY_SYMMETRIZE,
    primitive_reduction_class,
)
from .semantic import AUTO_PROVENANCE


def _improper_dihedral_atoms(atoms: tuple[int, ...]) -> tuple[int, ...]:
    if len(atoms) != 4:
        return atoms
    center, first, second, third = atoms
    return (first, center, third, second)


@dataclass(frozen=True)
class GICPrimitive:
    identifier: str
    name: str
    family: str
    function: str
    atoms: tuple[int, ...]
    mode: int = 0
    ref_atoms: tuple[int, ...] = ()
    refs: tuple[str, ...] = ()
    frame_atoms: tuple[int, ...] = ()
    ref_frame_atoms: tuple[int, ...] = ()
    provenance: str = AUTO_PROVENANCE
    semantic_id: str = ""
    semantic_type: str = ""
    chart: str = "PRINCIPAL"
    chart_reference_radian: float | None = None

    def gaussian_expression(self) -> str:
        if self.function == "IMPD":
            atoms = ",".join(str(atom) for atom in _improper_dihedral_atoms(self.atoms))
            return f"D({atoms})"
        if self.function == "D" and self.chart == "PERIODIC_CONTINUATION":
            if self.chart_reference_radian is None:
                raise ValueError(
                    f"periodic continuation {self.identifier} has no reference value"
                )
            atoms = ",".join(str(atom) for atom in self.atoms)
            reference = f"{float(self.chart_reference_radian):.17g}"
            # Gaussian's native D chart is shifted by pi relative to the
            # ORACLE plane-normal convention used by the compiled primitive
            # kernel.  Apply the constant phase before reference centering;
            # derivatives are unchanged and values now agree at the seam.
            delta = f"D({atoms})+(3.1415926535897931)-({reference})"
            return f"ATAN2(SIN({delta}),COS({delta}))"
        if not self.is_gaussian_native:
            return "NONE"
        atoms = ",".join(str(atom) for atom in self.atoms)
        if self.function == "L":
            reference = self.ref_atoms[0] if len(self.ref_atoms) == 1 else 0
            return f"L({atoms},{reference},{self.mode})"
        return f"{self.function}({atoms})"

    @property
    def is_gaussian_native(self) -> bool:
        return self.function in {"R", "A", "L", "D", "U", "H"}

    @property
    def reduction_class(self) -> str:
        return primitive_reduction_class(self.family)


@dataclass(frozen=True)
class FrozenGIC:
    identifier: str
    name: str
    family: str
    irrep: str
    primitive_id: str
    gaussian_expression: str
    coefficients: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class GICReductionDiagnostics:
    rank_method: str
    reduction_policy: str
    selected: tuple[str, ...] = ()
    skipped_singular: tuple[str, ...] = ()
    skipped_dependent: tuple[str, ...] = ()
    selected_by_family: tuple[str, ...] = ()
    skipped_singular_details: tuple[str, ...] = ()
    skipped_dependent_details: tuple[str, ...] = ()
    conditioning_decisions: tuple[str, ...] = ()


@dataclass(frozen=True)
class GICSymmetrizedGroup:
    block: str
    family: str
    signature: str
    source_gics: tuple[str, ...]
    output_gics: tuple[str, ...]


@dataclass(frozen=True)
class GICSymmetrizationDiagnostics:
    method: str
    policy: str
    status: str
    point_group: str
    symmetry_group: str
    total_symmetric_irrep: str
    total_symmetric_gics: tuple[str, ...] = ()
    groups: tuple[GICSymmetrizedGroup, ...] = ()
    sign_gauge_policy: str = "largest_abs_coefficient_pivot"
    path_gauge_policy: str = "subspace_overlap_procrustes"
    path_overlap_warning_threshold: float = SALC_PATH_OVERLAP_WARNING_THRESHOLD
    operation_tolerance_angstrom: float = SYMMETRY_OPERATION_TOLERANCE_ANGSTROM
    max_operation_residual_angstrom: float = 0.0
    min_operation_margin_angstrom: float = 0.0
    near_threshold_operations: tuple[str, ...] = ()
    fallback_events: tuple["FallbackEvent", ...] = ()


@dataclass(frozen=True)
class GICPointGroupOperation:
    label: str
    rotation: tuple[tuple[float, float, float], ...]
    permutation: tuple[int, ...]


@dataclass(frozen=True)
class FallbackEvent:
    """One typed, auditable fallback or completion decision."""

    event_id: str
    stage: str
    algorithm_id: str
    trigger: str
    domain: str = "GLOBAL"
    macrofamily: str = "UNSPECIFIED"
    rank_before: int | None = None
    rank_after: int | None = None
    condition_before: float | None = None
    condition_after: float | None = None
    source: str = ""


@dataclass(frozen=True)
class GICDefinition:
    backend: str
    point_group: str
    symmetrize: bool
    target_rank: int
    rank: int
    candidate_count: int
    reference_coordinates_angstrom: tuple[tuple[float, float, float], ...]
    primitives: tuple[GICPrimitive, ...]
    gics: tuple[FrozenGIC, ...]
    reduction_diagnostics: GICReductionDiagnostics | None = None
    symmetry_diagnostics: GICSymmetrizationDiagnostics | None = None
    fragment_mode: str = FRAGMENT_MODE_NONE
    pseudo_bonds: tuple[tuple[int, int], ...] = ()
    pseudo_bond_kinds: tuple[str, ...] = ()
    aromatic_atoms: tuple[int, ...] = ()
    xh_stretch_policy: str = XH_STRETCH_POLICY_SYMMETRIZE
    local_xh_bonds: tuple[tuple[int, int], ...] = ()
    local_xh_classes: tuple[str, ...] = ()
    ring_puckering_diagnostics: tuple[str, ...] = ()
    periodic_coordinate_estimates: tuple[PeriodicCoordinateEstimate, ...] = ()
    contract_schema_version: str = ORACLE_XYZ_GIC_SCHEMA
    semantic_grammar_version: str = ""
    semantic_diagnostics: tuple[str, ...] = ()
    fallback_diagnostics: tuple[str, ...] = ()
    fallback_events: tuple[FallbackEvent, ...] = ()
    primitive_source: str = "EXPLICIT_DEFINITION"
    primitive_source_schema: str = ""
    primitive_b_matrix_sha256: str = ""
    wilson_tangent_rank: int = 0
    wilson_tangent_singular_min: float = 0.0
    wilson_tangent_singular_max: float = 0.0


@dataclass(frozen=True)
class GICBMatrix:
    backend: str
    coordinate_labels: tuple[str, ...]
    coordinate_names: tuple[str, ...]
    irreps: tuple[str, ...]
    cartesian_columns: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]
    sparse_rows: tuple[SparseBRow, ...] = ()

    def sparse_matrix(self) -> SparseBMatrix:
        sparse_rows = self.sparse_rows or tuple(SparseBRow.from_dense(row) for row in self.rows)
        return SparseBMatrix(
            rows=sparse_rows,
            column_count=len(self.cartesian_columns),
            row_labels=self.coordinate_labels,
            backend=self.backend,
        )


@dataclass(frozen=True)
class SYCartDefinition:
    backend: str
    point_group: str
    target_rank: int
    vectors: tuple[tuple[float, ...], ...]
    irreps: tuple[str, ...] = ()
    external_mode_count: int = 0
    linearity: str = ""
    gauge_policy: str = ""
