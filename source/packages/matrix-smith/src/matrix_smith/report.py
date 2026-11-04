"""Human-readable GICForge reports."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import degrees
from pathlib import Path

from .definition import read_gic_definition_from_xyzin, total_symmetric_gic_names
from .evaluation import evaluate_gic_values
from .models import GICDefinition
from .policy import (
    FRAGMENT_MODE_PSEUDO_BONDS,
    FRAGMENT_MODE_SPECIAL_COORDINATES,
    REDUCTION_POLICY,
    SPECIAL_REDUCTION_CLASS,
    primitive_reduction_class,
)
from .symmetry import gic_symmetry_source_blocks


@dataclass(frozen=True)
class GICClosureSummary:
    closed: bool
    rank_complete: bool
    rank: int
    target_rank: int
    symmetry_method: str
    symmetry_status: str
    total_symmetric_irrep: str
    total_symmetric_count: int
    protected_special_count: int
    protected_special_families: tuple[str, ...]
    ring_diagnostic_count: int
    salc_coefficient_count: int
    max_salc_norm_error: float
    skipped_singular_count: int
    skipped_dependent_count: int


def gic_report_lines(definition: GICDefinition) -> list[str]:
    family_counts = Counter(primitive.family for primitive in definition.primitives)
    protected_count = sum(
        1
        for primitive in definition.primitives
        if primitive.reduction_class == SPECIAL_REDUCTION_CLASS
    )
    diagnostics = definition.reduction_diagnostics
    skipped_singular = diagnostics.skipped_singular if diagnostics else ()
    skipped_dependent = diagnostics.skipped_dependent if diagnostics else ()
    rank_method = diagnostics.rank_method if diagnostics else "UNKNOWN"
    reduction_policy = diagnostics.reduction_policy if diagnostics else REDUCTION_POLICY
    selected = (
        diagnostics.selected
        if diagnostics
        else tuple(primitive.identifier for primitive in definition.primitives)
    )
    selected_by_family = (
        diagnostics.selected_by_family
        if diagnostics and diagnostics.selected_by_family
        else _family_count_tokens(family_counts)
    )
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}

    lines = [
        "MATRIX SMITH/SONIC Report",
        "=============================",
        "",
        f"Backend: {definition.backend}",
        f"Point group: {definition.point_group}",
        f"Symmetry group: {definition.point_group}",
        f"Totally symmetric GICs: {_list_or_none(total_symmetric_gic_names(definition))}",
        f"Symmetrize requested: {definition.symmetrize}",
        f"Target rank: {definition.target_rank}",
        f"Target rank rationale: {_target_rank_rationale(definition)}",
        f"Final rank: {definition.rank}",
        f"Candidate count: {definition.candidate_count}",
        f"Selected primitive count: {len(definition.primitives)}",
        f"Frozen GIC count: {len(definition.gics)}",
        f"Protected selected count: {protected_count}",
        f"Rank method: {rank_method}",
        f"Reduction policy: {reduction_policy}",
        f"Skipped singular/zero rows: {len(skipped_singular)}",
        f"Skipped dependent rows: {len(skipped_dependent)}",
        "",
        "Closure Summary",
        "---------------",
        *_closure_summary_lines(gic_closure_summary(definition)),
        "",
        "Fragment Mode Policy",
        "--------------------",
        *_fragment_policy_lines(definition),
        "",
        "Selected Families",
        "-----------------",
    ]
    if family_counts:
        for family in sorted(family_counts):
            lines.append(f"{family}: {family_counts[family]} ({primitive_reduction_class(family)})")
    else:
        lines.append("NONE")

    lines.extend(["", "Protected Coordinates", "---------------------"])
    protected = [
        primitive
        for primitive in definition.primitives
        if primitive.reduction_class == SPECIAL_REDUCTION_CLASS
    ]
    if protected:
        for primitive in protected:
            lines.append(_primitive_summary(primitive))
    else:
        lines.append("NONE")

    lines.extend(["", "Symmetry Source Blocks", "----------------------"])
    blocks = gic_symmetry_source_blocks(definition)
    if blocks:
        for block in blocks:
            lines.append(
                f"{block.block}: family={block.family} "
                f"class={block.reduction_class} count={len(block.gic_names)}"
            )
    else:
        lines.append("NONE")

    lines.extend(["", "Local Equivalence Diagnostics", "-----------------------------"])
    local_equivalence = _local_equivalence_lines(definition)
    lines.extend(local_equivalence or ["NONE"])

    lines.extend(["", "Ring Puckering Diagnostics", "--------------------------"])
    lines.extend(definition.ring_puckering_diagnostics or ("NONE",))

    symmetry = definition.symmetry_diagnostics
    lines.extend(["", "Symmetrization Diagnostics", "---------------------------"])
    if symmetry:
        lines.append(f"Method: {symmetry.method}")
        lines.append(f"Policy: {symmetry.policy}")
        lines.append(f"Status: {symmetry.status}")
        lines.append(f"Symmetry group: {symmetry.symmetry_group}")
        lines.append(f"Total irrep: {symmetry.total_symmetric_irrep}")
        lines.append("Total GICs: " + _list_or_none(symmetry.total_symmetric_gics))
        lines.append(f"Sign gauge policy: {symmetry.sign_gauge_policy}")
        lines.append(f"Path gauge policy: {symmetry.path_gauge_policy}")
        lines.append(
            "Operation residual max (Angstrom): "
            f"{symmetry.max_operation_residual_angstrom:.6g}"
        )
        lines.append(
            "Operation margin min (Angstrom): "
            f"{symmetry.min_operation_margin_angstrom:.6g}"
        )
        lines.append(
            "Near-threshold operations: " + _list_or_none(symmetry.near_threshold_operations)
        )
        lines.append(f"Groups: {len(symmetry.groups)}")
        for group in symmetry.groups:
            lines.append(
                f"{group.block} {group.signature}: "
                f"{','.join(group.source_gics)} -> {','.join(group.output_gics)}"
            )
    else:
        lines.append("NONE")

    lines.extend(["", "SALC Coefficients", "-----------------"])
    coefficient_lines = _salc_coefficient_lines(definition)
    lines.extend(coefficient_lines or ["NONE"])

    lines.extend(["", "Reduction Diagnostics", "---------------------"])
    lines.append("Selected: " + _list_or_none(selected))
    lines.append("Selected by family: " + _list_or_none(selected_by_family))
    lines.append("Skipped singular: " + _list_or_none(skipped_singular))
    lines.append("Skipped dependent: " + _list_or_none(skipped_dependent))
    lines.append(
        "Skipped singular details: "
        + _list_or_none(diagnostics.skipped_singular_details if diagnostics else ())
    )
    lines.append(
        "Skipped dependent details: "
        + _list_or_none(diagnostics.skipped_dependent_details if diagnostics else ())
    )

    lines.extend(["", "Coordinate Definitions", "----------------------"])
    if definition.gics:
        values = evaluate_gic_values(definition)
        total_names = set(total_symmetric_gic_names(definition))
        for index, (gic, value) in enumerate(zip(definition.gics, values), start=1):
            primitive = primitive_by_id.get(gic.primitive_id)
            reduction_class = primitive.reduction_class if primitive else "UNKNOWN"
            state = "ACTIVE" if not total_names or gic.name in total_names else "FROZEN"
            components = gic.coefficients or ((gic.primitive_id, 1.0),)
            component_text = ",".join(
                f"{primitive_id}:{coefficient:+.8g}"
                for primitive_id, coefficient in components
            )
            lines.append(
                f"{index:4d} {gic.identifier} {gic.name} family={gic.family} "
                f"class={reduction_class} irrep={gic.irrep} state={state} "
                f"value={float(value):.10g} unit={_coordinate_unit(gic.family)} "
                f"components={component_text}"
            )
            lines.extend(_human_coordinate_definition_lines(gic, primitive_by_id))
    else:
        lines.append("NONE")
    lines.extend(["", "Periodicity and Barrier Seeds", "-----------------------------"])
    if definition.periodic_coordinate_estimates:
        for record in definition.periodic_coordinate_estimates:
            bonds = ",".join(f"{left}-{right}" for left, right in record.central_bonds) or "NONE"
            ring = ",".join(str(atom) for atom in record.ring_atoms) or "NONE"
            sources = ",".join(record.source_coordinates) or "NONE"
            reference = (
                "UNDEFINED"
                if record.reference_value_radian is None
                else f"{degrees(record.reference_value_radian):.10g} deg"
            )
            lines.append(
                f"{record.coordinate_identifier} {record.coordinate_name} family={record.family} "
                f"definition={record.coordinate_definition} "
                f"periodicity={record.periodicity} barrier={record.barrier_kcal_mol:.6g} "
                f"kcal/mol ({record.barrier_cm1:.6g} cm-1) target={record.target} "
                f"central_bonds={bonds} ring_atoms={ring} sources={sources} "
                f"reference_value={reference} value_status={record.reference_value_status} "
                f"domain={record.coordinate_domain} symmetry_number={record.symmetry_number} "
                f"priority_atom={record.priority_atom if record.priority_atom is not None else 'NONE'} "
                f"periodicity_source={record.periodicity_source} "
                f"barrier_source={record.barrier_source} status={record.status}"
            )
    else:
        lines.append("NONE")
    return lines


def _coordinate_unit(family: str) -> str:
    name = str(family).strip().upper()
    if name in {
        "STRETCH",
        "FRAGMENT_CENTER_DISTANCE",
        "FRAGMENT_CENTER_ATOM_DISTANCE",
        "CENTER_ATOM_DISTANCE",
        "FRAG_TRANSLATION",
    }:
        return "angstrom"
    if name in {
        "BEND",
        "LINEAR_BEND",
        "TORSION",
        "OUT_OF_PLANE",
        "IMPROPER_DIHEDRAL",
        "CYCLIC_BEND",
        "RING_DEFORMATION",
        "RING_PUCKERING",
        "RING_PUCKER_COMPONENT",
        "PSEUDO_CYCLE_BEND",
        "PSEUDO_CYCLE_TORSION",
        "BUTTERFLY",
        "FRAG_ORIENTATION",
    }:
        return "radian"
    return "native"


def _human_coordinate_definition_lines(gic, primitive_by_id: dict[str, object]) -> list[str]:
    components = gic.coefficients or ((gic.primitive_id, 1.0),)
    formula = " ".join(
        f"{float(coefficient):+.8g}*{primitive_id}"
        for primitive_id, coefficient in components
    )
    lines = [f"     Human formula: {gic.name} = {formula}"]
    for primitive_id, coefficient in components:
        primitive = primitive_by_id.get(primitive_id)
        if primitive is None:
            lines.append(f"       {float(coefficient):+.8g} * {primitive_id}: UNKNOWN")
            continue
        lines.extend(_human_primitive_lines(primitive, float(coefficient)))
    return lines


def _human_primitive_lines(primitive, sonic_coefficient: float) -> list[str]:
    prefix = f"       {sonic_coefficient:+.8g} * {primitive.identifier}"
    atoms = ",".join(str(atom) for atom in primitive.atoms)
    function = str(primitive.function).upper()
    family = str(primitive.family).upper()
    if family in {"RING_PUCKER_COMPONENT", "RING_PUCKERING"} and function == "D":
        ring_atoms = _ring_atoms_from_refs(primitive.refs)
        ring_text = ",".join(str(atom) for atom in ring_atoms) or atoms
        return [
            f"{prefix}: signed triangular-flap plane incidence F({atoms}) "
            f"in ordered ring ({ring_text}); analytic kernel D"
        ]
    if family in {"RING_PUCKER_COMPONENT", "RING_PUCKERING"} and function == "U":
        ring_atoms = _ring_atoms_from_refs(primitive.refs)
        ring_text = ",".join(str(atom) for atom in ring_atoms) or atoms
        return [
            f"{prefix}: ring-puckering source out-of-plane U({atoms}) "
            f"in ordered ring ({ring_text})"
        ]
    if function == "R":
        return [f"{prefix}: distance R({atoms})"]
    if function == "A":
        return [f"{prefix}: bond angle A({atoms})"]
    if function == "L":
        return [f"{prefix}: linear-bend component L({atoms}, mode={primitive.mode})"]
    if function in {"D", "IMPD"}:
        label = "improper torsion" if function == "IMPD" else "torsion"
        return [f"{prefix}: {label} D({atoms})"]
    if function == "U":
        return [f"{prefix}: out-of-plane coordinate U({atoms})"]
    if function == "H":
        return [f"{prefix}: signed out-of-plane height H({atoms})"]
    if function == "SOOP":
        return [f"{prefix}: selective out-of-plane SOOP({atoms}) in frozen local frame"]
    if function == "RPCK":
        lines = [
            f"{prefix}: normalized ring-puckering component on ordered ring atoms ({atoms})",
            "         Primitive expansion (radians):",
        ]
        lines.extend(
            f"           {coefficient:+.8g} * torsion D({','.join(str(atom) for atom in term_atoms)})"
            for coefficient, term_atoms in _encoded_primitive_terms(primitive.refs, 4)
        )
        return lines
    if function == "RPCB":
        lines = [
            f"{prefix}: normalized in-plane ring-deformation component on ordered ring atoms ({atoms})",
            "         Primitive expansion (radians):",
        ]
        lines.extend(
            f"           {coefficient:+.8g} * angle A({','.join(str(atom) for atom in term_atoms)})"
            for coefficient, term_atoms in _encoded_primitive_terms(primitive.refs, 3)
        )
        return lines
    if function == "FTRANS":
        return [
            f"{prefix}: relative fragment translation component {primitive.mode}; "
            f"fragment atoms ({atoms}), reference atoms ({','.join(map(str, primitive.ref_atoms))})"
        ]
    if function == "FROT":
        return [
            f"{prefix}: relative fragment exponential-map rotation component {primitive.mode}; "
            f"fragment atoms ({atoms}), reference atoms ({','.join(map(str, primitive.ref_atoms))})"
        ]
    expression = primitive.gaussian_expression()
    detail = expression if expression != "NONE" else f"function={primitive.function} atoms=({atoms})"
    return [f"{prefix}: {primitive.family.lower().replace('_', ' ')}; {detail}"]


def _encoded_primitive_terms(
    refs: tuple[str, ...],
    atom_count: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    terms: list[tuple[float, tuple[int, ...]]] = []
    for ref in refs:
        if ":" not in ref:
            continue
        coefficient_text, atoms_text = ref.split(":", 1)
        try:
            coefficient = float(coefficient_text)
            atoms = tuple(int(atom) for atom in atoms_text.split("-") if atom)
        except ValueError:
            continue
        if len(atoms) == atom_count:
            terms.append((coefficient, atoms))
    return tuple(terms)


def _ring_atoms_from_refs(refs: tuple[str, ...]) -> tuple[int, ...]:
    for ref in refs:
        if not ref.upper().startswith("RING:"):
            continue
        try:
            return tuple(int(atom) for atom in ref.split(":", 1)[1].split("-") if atom)
        except ValueError:
            return ()
    return ()


def _local_equivalence_lines(definition: GICDefinition) -> list[str]:
    diagnostics = definition.reduction_diagnostics
    if diagnostics is None:
        return []
    return [
        item.removeprefix("LOCAL_EQUIVALENCE ").removeprefix("LOCAL_SALC ")
        for item in diagnostics.skipped_dependent_details
        if item.startswith(("LOCAL_EQUIVALENCE ", "LOCAL_SALC "))
    ]


def gic_report_from_xyzin(path: Path) -> list[str]:
    return gic_report_lines(read_gic_definition_from_xyzin(Path(path)))


def write_gic_report(path: Path, output: Path) -> Path:
    target = Path(output)
    target.write_text("\n".join(gic_report_from_xyzin(Path(path))) + "\n", encoding="utf-8")
    return target


def _list_or_none(values: tuple[str, ...]) -> str:
    return ", ".join(values) if values else "NONE"


def _family_count_tokens(family_counts: Counter[str]) -> tuple[str, ...]:
    return tuple(f"{family}:{family_counts[family]}" for family in sorted(family_counts))


def _target_rank_rationale(definition: GICDefinition) -> str:
    natoms = len(definition.reference_coordinates_angstrom)
    nonlinear = 3 * natoms - 6
    linear = 3 * natoms - 5
    if definition.target_rank == nonlinear:
        return f"3N-6 non-linear vibrational rank for N={natoms}"
    if definition.target_rank == linear:
        return f"3N-5 linear vibrational rank for N={natoms}"
    return f"contract-specified rank {definition.target_rank} for N={natoms}"


def gic_closure_summary(definition: GICDefinition) -> GICClosureSummary:
    symmetry = definition.symmetry_diagnostics
    salc_count = sum(1 for gic in definition.gics if len(gic.coefficients) > 1)
    salc_norm_error = _max_salc_norm_error(definition)
    special_count = sum(
        1
        for primitive in definition.primitives
        if primitive.reduction_class == SPECIAL_REDUCTION_CLASS
    )
    protected_families = tuple(
        sorted(
            {
                primitive.family
                for primitive in definition.primitives
                if primitive.reduction_class == SPECIAL_REDUCTION_CLASS
            }
        )
    )
    diagnostics = definition.reduction_diagnostics
    skipped_singular = diagnostics.skipped_singular if diagnostics else ()
    skipped_dependent = diagnostics.skipped_dependent if diagnostics else ()
    rank_closed = definition.rank == definition.target_rank
    symmetry_status = symmetry.status if symmetry is not None else "NONE"
    symmetry_method = symmetry.method if symmetry is not None else "NONE"
    total_irrep = symmetry.total_symmetric_irrep if symmetry is not None else "NONE"
    total_count = len(symmetry.total_symmetric_gics) if symmetry is not None else 0
    closed = (
        rank_closed
        and not skipped_singular
        and (symmetry is None or symmetry.status in {"APPLIED", "NOT_REQUESTED", "NO_ELIGIBLE_GROUPS"})
        and salc_norm_error <= 1.0e-10
    )
    return GICClosureSummary(
        closed=closed,
        rank_complete=rank_closed,
        rank=definition.rank,
        target_rank=definition.target_rank,
        symmetry_method=symmetry_method,
        symmetry_status=symmetry_status,
        total_symmetric_irrep=total_irrep,
        total_symmetric_count=total_count,
        protected_special_count=special_count,
        protected_special_families=protected_families,
        ring_diagnostic_count=len(definition.ring_puckering_diagnostics),
        salc_coefficient_count=salc_count,
        max_salc_norm_error=salc_norm_error,
        skipped_singular_count=len(skipped_singular),
        skipped_dependent_count=len(skipped_dependent),
    )


def _closure_summary_lines(summary: GICClosureSummary) -> list[str]:
    return [
        f"Closed: {'YES' if summary.closed else 'NO'}",
        f"Rank complete: {'YES' if summary.rank_complete else 'NO'} "
        f"({summary.rank}/{summary.target_rank})",
        f"Symmetry method: {summary.symmetry_method}",
        f"Symmetry status: {summary.symmetry_status}",
        f"Total-symmetric irrep/count: "
        f"{summary.total_symmetric_irrep}/{summary.total_symmetric_count}",
        f"Protected special coordinates: {summary.protected_special_count}",
        f"Protected special families: {_list_or_none(summary.protected_special_families)}",
        f"Ring diagnostics: {summary.ring_diagnostic_count}",
        f"SALC coefficient vectors: {summary.salc_coefficient_count}",
        f"Max SALC norm error: {summary.max_salc_norm_error:.12g}",
        f"Skipped singular rows: {summary.skipped_singular_count}",
        f"Skipped dependent rows: {summary.skipped_dependent_count}",
    ]


def _fragment_policy_lines(definition: GICDefinition) -> list[str]:
    mode = definition.fragment_mode
    lines = [f"Mode: {mode}"]
    if mode == FRAGMENT_MODE_SPECIAL_COORDINATES:
        lines.extend(
            [
                "Policy: frozen ORACLE atlas; keep fragments as protected bodies.",
                "Coordinates: fragment-center distances, center-atom distances, translations and orientations.",
                "Rationale: preserve inter-fragment rigid-body motion before ordinary valence pruning.",
            ]
        )
    elif mode == FRAGMENT_MODE_PSEUDO_BONDS:
        kinds = definition.pseudo_bond_kinds or (
            ("INTERFRAGMENT_CLOSEST",) * len(definition.pseudo_bonds)
        )
        contacts = tuple(
            f"{left}-{right}:{kind}" for (left, right), kind in zip(definition.pseudo_bonds, kinds)
        )
        lines.extend(
            [
                "Policy: augmented graph with provenance-preserving SONIC coordinates and no artificial-ring protection.",
                "Selection: BDPCS3 H-bonds first, then short closure contacts needed for the inter-fragment span.",
                "Pseudo-bonds: " + _list_or_none(contacts),
                "Rationale: H-bonds retain the protected HBOND_DISTANCE family while all pseudo-contacts augment connectivity; the resulting graph cycles are not treated as chemical rings.",
            ]
        )
    else:
        lines.append("Policy: no built fragments were consumed by this GIC definition.")
    if mode != FRAGMENT_MODE_PSEUDO_BONDS and definition.pseudo_bonds:
        kinds = definition.pseudo_bond_kinds or (
            ("UNCLASSIFIED",) * len(definition.pseudo_bonds)
        )
        contacts = tuple(
            f"{left}-{right}:{kind}"
            for (left, right), kind in zip(definition.pseudo_bonds, kinds)
        )
        lines.append("Pseudo-bonds: " + _list_or_none(contacts))
    return lines


def _primitive_summary(primitive) -> str:
    atoms = ",".join(str(atom) for atom in primitive.atoms) or "NONE"
    refs = ",".join(primitive.refs) or "NONE"
    return (
        f"{primitive.identifier} {primitive.name} family={primitive.family} "
        f"class={primitive.reduction_class} atoms={atoms} refs={refs}"
    )


def _salc_coefficient_lines(definition: GICDefinition) -> list[str]:
    lines: list[str] = []
    for gic in definition.gics:
        if len(gic.coefficients) <= 1:
            continue
        terms = ",".join(
            f"{primitive_id}:{coefficient:+.12g}" for primitive_id, coefficient in gic.coefficients
        )
        norm2 = sum(float(coefficient) ** 2 for _primitive_id, coefficient in gic.coefficients)
        lines.append(
            f"{gic.name} irrep={gic.irrep} family={gic.family} norm2={norm2:.12g} coeffs={terms}"
        )
    return lines


def _max_salc_norm_error(definition: GICDefinition) -> float:
    errors = []
    for gic in definition.gics:
        if len(gic.coefficients) <= 1:
            continue
        norm2 = sum(float(coefficient) ** 2 for _primitive_id, coefficient in gic.coefficients)
        errors.append(abs(norm2 - 1.0))
    return max(errors) if errors else 0.0
