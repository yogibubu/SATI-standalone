from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matrix_smith import GICDefinition

from .commands import (
    OracleGuiCommand,
    gicforge_bmatrix_command,
    gicforge_build_command,
    gicforge_gaussian_input_command,
    gicforge_report_command,
)


@dataclass(frozen=True)
class GICForgeTable:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class GICForgeSummary:
    backend: str = ""
    point_group: str = ""
    symmetry_group: str = ""
    total_symmetric_irrep: str = ""
    total_symmetric_gics: tuple[str, ...] = ()
    symmetrize: bool = False
    target_rank: int = 0
    rank: int = 0
    candidate_count: int = 0
    primitive_count: int = 0
    gic_count: int = 0
    skipped_singular_count: int = 0
    skipped_dependent_count: int = 0
    symmetry_status: str = ""
    symmetry_method: str = ""
    reduction_policy: str = ""
    rank_method: str = ""


@dataclass(frozen=True)
class GICForgeGuiState:
    xyzin: Path
    exists: bool
    ready: bool
    summary: GICForgeSummary
    primitives: GICForgeTable
    frozen_gics: GICForgeTable
    symmetry_groups: GICForgeTable
    diagnostics: GICForgeTable
    messages: tuple[str, ...] = ()


class OracleGICForgeController:
    def __init__(self, xyzin: Path | str | None = None) -> None:
        self.xyzin = None if xyzin is None else Path(xyzin)

    def set_xyzin(self, xyzin: Path | str | None) -> GICForgeGuiState | None:
        self.xyzin = None if xyzin is None else Path(xyzin)
        if self.xyzin is None:
            return None
        return self.state()

    def state(self) -> GICForgeGuiState:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return load_gicforge_gui_state(self.xyzin)

    def build_command(
        self,
        *,
        symmetrize: bool = True,
        sycart: bool = True,
    ) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return gicforge_build_command(
            self.xyzin,
            symmetrize=symmetrize,
            sycart=sycart,
        )

    def report_command(self, output: Path | str | None = None) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return gicforge_report_command(self.xyzin, output)

    def bmatrix_command(self, output: Path | str | None = None) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return gicforge_bmatrix_command(self.xyzin, output)

    def gaussian_input_command(
        self,
        output: Path | str,
        *,
        route: str = "#p hf/sto-3g opt=readallgic",
        title: str | None = None,
        charge: int | None = None,
        multiplicity: int | None = None,
    ) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return gicforge_gaussian_input_command(
            self.xyzin,
            output,
            route=route,
            title=title,
            charge=charge,
            multiplicity=multiplicity,
        )


def load_gicforge_gui_state(path: Path | str) -> GICForgeGuiState:
    from matrix_smith import GICForgeContractError, read_gic_definition_from_xyzin

    target = Path(path)
    empty = _empty_state(target, exists=target.exists())
    if not target.exists():
        return _replace_messages(empty, (f"Missing file: {target}",))
    try:
        definition = read_gic_definition_from_xyzin(target)
    except (GICForgeContractError, OSError, ValueError) as exc:
        return _replace_messages(empty, (str(exc),))
    return GICForgeGuiState(
        xyzin=target,
        exists=True,
        ready=True,
        summary=_summary(definition),
        primitives=_primitive_table(definition),
        frozen_gics=_frozen_gic_table(definition),
        symmetry_groups=_symmetry_group_table(definition),
        diagnostics=_diagnostics_table(definition),
        messages=(),
    )


def gicforge_gui_state_lines(state: GICForgeGuiState) -> list[str]:
    if not state.ready:
        return [
            f"xyzin: {state.xyzin}",
            f"ready: {int(state.ready)}",
            *state.messages,
        ]
    summary = state.summary
    return [
        f"xyzin: {state.xyzin}",
        f"ready: {int(state.ready)}",
        f"backend: {summary.backend}",
        f"point group: {summary.point_group}",
        f"symmetry group: {summary.symmetry_group or summary.point_group}",
        f"total symmetric irrep: {summary.total_symmetric_irrep}",
        "total symmetric GICs: " + _csv_or_none(summary.total_symmetric_gics),
        f"symmetrized: {summary.symmetrize}",
        f"target rank: {summary.target_rank}",
        f"rank: {summary.rank}",
        f"candidate count: {summary.candidate_count}",
        f"primitive count: {summary.primitive_count}",
        f"GIC count: {summary.gic_count}",
        f"symmetry status: {summary.symmetry_status or 'UNKNOWN'}",
        f"symmetry method: {summary.symmetry_method or 'UNKNOWN'}",
        f"rank method: {summary.rank_method or 'UNKNOWN'}",
        f"reduction policy: {summary.reduction_policy or 'UNKNOWN'}",
        f"skipped singular: {summary.skipped_singular_count}",
        f"skipped dependent: {summary.skipped_dependent_count}",
    ]


def default_gicforge_report_output(xyzin: Path | str) -> Path:
    target = Path(xyzin)
    return target.with_name(f"{target.stem}.gicforge_report.txt")


def default_gicforge_bmatrix_output(xyzin: Path | str) -> Path:
    target = Path(xyzin)
    return target.with_name(f"{target.stem}.gic_bmatrix.txt")


def default_gicforge_gaussian_output(xyzin: Path | str) -> Path:
    target = Path(xyzin)
    return target.with_name(f"{target.stem}.gic.gjf")


def _summary(definition: GICDefinition) -> GICForgeSummary:
    from matrix_smith import total_symmetric_gic_names

    reduction = definition.reduction_diagnostics
    symmetry = definition.symmetry_diagnostics
    return GICForgeSummary(
        backend=definition.backend,
        point_group=definition.point_group,
        symmetry_group=symmetry.symmetry_group if symmetry else definition.point_group,
        total_symmetric_irrep=symmetry.total_symmetric_irrep if symmetry else "",
        total_symmetric_gics=(
            symmetry.total_symmetric_gics if symmetry else total_symmetric_gic_names(definition)
        ),
        symmetrize=definition.symmetrize,
        target_rank=definition.target_rank,
        rank=definition.rank,
        candidate_count=definition.candidate_count,
        primitive_count=len(definition.primitives),
        gic_count=len(definition.gics),
        skipped_singular_count=len(reduction.skipped_singular) if reduction else 0,
        skipped_dependent_count=len(reduction.skipped_dependent) if reduction else 0,
        symmetry_status=symmetry.status if symmetry else "",
        symmetry_method=symmetry.method if symmetry else "",
        reduction_policy=reduction.reduction_policy if reduction else "",
        rank_method=reduction.rank_method if reduction else "",
    )


def _primitive_table(definition: GICDefinition) -> GICForgeTable:
    rows = tuple(
        (
            primitive.identifier,
            primitive.name,
            primitive.family,
            primitive.reduction_class,
            primitive.function,
            _join_ints(primitive.atoms),
            _join_ints(primitive.ref_atoms),
            ",".join(primitive.refs),
        )
        for primitive in definition.primitives
    )
    return GICForgeTable(
        "Primitives",
        ("ID", "Name", "Family", "Class", "Function", "Atoms", "Ref atoms", "Refs"),
        rows,
    )


def _frozen_gic_table(definition: GICDefinition) -> GICForgeTable:
    rows = tuple(
        (
            gic.identifier,
            gic.name,
            gic.family,
            gic.irrep,
            gic.primitive_id,
            _coefficients(gic.coefficients),
            gic.gaussian_expression,
        )
        for gic in definition.gics
    )
    return GICForgeTable(
        "Frozen GICs",
        ("ID", "Name", "Family", "Irrep", "Primitive", "Coefficients", "Gaussian"),
        rows,
    )


def _symmetry_group_table(definition: GICDefinition) -> GICForgeTable:
    groups = definition.symmetry_diagnostics.groups if definition.symmetry_diagnostics else ()
    rows = tuple(
        (
            group.block,
            group.family,
            group.signature,
            ",".join(group.source_gics),
            ",".join(group.output_gics),
        )
        for group in groups
    )
    return GICForgeTable(
        "Symmetry Groups",
        ("Block", "Family", "Signature", "Sources", "Outputs"),
        rows,
    )


def _diagnostics_table(definition: GICDefinition) -> GICForgeTable:
    reduction = definition.reduction_diagnostics
    symmetry = definition.symmetry_diagnostics
    rows: list[tuple[str, str]] = []
    rows.extend(
        (
            ("Backend", definition.backend),
            ("Point group", definition.point_group),
            ("Symmetrized", str(definition.symmetrize)),
            ("Target rank", str(definition.target_rank)),
            ("Rank", str(definition.rank)),
            ("Candidate count", str(definition.candidate_count)),
        )
    )
    if reduction is not None:
        rows.extend(
            (
                ("Rank method", reduction.rank_method),
                ("Reduction policy", reduction.reduction_policy),
                ("Selected", _csv_or_none(reduction.selected)),
                ("Skipped singular", _csv_or_none(reduction.skipped_singular)),
                ("Skipped dependent", _csv_or_none(reduction.skipped_dependent)),
            )
        )
    if symmetry is not None:
        rows.extend(
            (
                ("Symmetry method", symmetry.method),
                ("Symmetry policy", symmetry.policy),
                ("Symmetry status", symmetry.status),
                ("Symmetry group", symmetry.symmetry_group),
                ("Total symmetric irrep", symmetry.total_symmetric_irrep),
                ("Total symmetric GICs", _csv_or_none(symmetry.total_symmetric_gics)),
            )
        )
    return GICForgeTable("Diagnostics", ("Field", "Value"), tuple(rows))


def _empty_state(target: Path, *, exists: bool) -> GICForgeGuiState:
    return GICForgeGuiState(
        xyzin=target,
        exists=exists,
        ready=False,
        summary=GICForgeSummary(),
        primitives=GICForgeTable(
            "Primitives",
            ("ID", "Name", "Family", "Class", "Function", "Atoms", "Ref atoms", "Refs"),
            (),
        ),
        frozen_gics=GICForgeTable(
            "Frozen GICs",
            ("ID", "Name", "Family", "Irrep", "Primitive", "Coefficients", "Gaussian"),
            (),
        ),
        symmetry_groups=GICForgeTable(
            "Symmetry Groups",
            ("Block", "Family", "Signature", "Sources", "Outputs"),
            (),
        ),
        diagnostics=GICForgeTable("Diagnostics", ("Field", "Value"), ()),
        messages=(),
    )


def _replace_messages(
    state: GICForgeGuiState,
    messages: tuple[str, ...],
) -> GICForgeGuiState:
    return GICForgeGuiState(
        xyzin=state.xyzin,
        exists=state.exists,
        ready=state.ready,
        summary=state.summary,
        primitives=state.primitives,
        frozen_gics=state.frozen_gics,
        symmetry_groups=state.symmetry_groups,
        diagnostics=state.diagnostics,
        messages=messages,
    )


def _join_ints(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def _coefficients(values: tuple[tuple[str, float], ...]) -> str:
    return ",".join(f"{name}:{coefficient:.6g}" for name, coefficient in values)


def _csv_or_none(values: tuple[str, ...]) -> str:
    return ",".join(values) if values else "NONE"
