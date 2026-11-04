"""Gaussian Generalized Internal Coordinate export for SMITH definitions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
from typing import TYPE_CHECKING

import numpy as np

from matrix_chem import BOHR_TO_ANGSTROM
from matrix_core import read_sectioned_lines, section_content

from .block_payload import payload_owned_atom_frame
from .contracts import GICForgeContractError
from .coordinate_registry import (
    FRAGMENT_BODY_PAIR_FUNCTIONS,
    FRAGMENT_BODY_SINGLE_FUNCTIONS,
    gaussian_primitive_lowering,
)
from .models import FrozenGIC, GICDefinition, GICPrimitive
from .numerics import (
    _angle_component_terms_from_refs,
    _axis_axis_stereographic_chart,
    _fragment_frame_anchor_atoms,
    _fragment_relative_frames,
    _linear_fragment_stereographic_axes,
    _ring_pucker_terms_from_refs,
)
from .symmetrization import _single_source_primitive
from .symmetry_labels import is_total_symmetric_irrep

if TYPE_CHECKING:
    from .onic_blocks import OnicCoordinateBlock
    from .typed_onic_artifact import TypedOnicArtifact


_GAUSSIAN_DEPENDENCY_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
_CARTESIAN_SOURCE_RE = re.compile(r"^ATOM(?P<atom>[1-9][0-9]*)\.(?P<axis>[XYZ])$")
_INVERSE_DISTANCE_SOURCE_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9_.:-]*\.Pair(?P<left>[0-9]{4})_(?P<right>[0-9]{4})$"
)
_TYPED_GAUSSIAN_PREFIX = {
    "SYMMETRY_ADAPTED_CARTESIAN": "C",
    "INVERSE_DISTANCE_PROJECTOR": "I",
    "NATURAL_INTERNAL": "N",
    "EXPONENTIAL_MAP": "E",
    "PSEUDO_BOND_CONTACT": "P",
}


def _subsection(section_lines: list[str], name: str) -> list[str]:
    wanted = name.strip().upper()
    content: list[str] = []
    active = False
    for line in section_lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            active = stripped[1:-1].strip().upper() == wanted
            continue
        if active and stripped:
            content.append(stripped)
    return content


def _section_value(section_lines: list[str], key: str) -> str | None:
    wanted = key.strip().upper()
    for line in section_lines:
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].strip().upper() == wanted:
            return parts[1].strip()
    return None


def _parse_text_list(text: str) -> tuple[str, ...]:
    if not text or text.upper() == "NONE":
        return ()
    return tuple(item for item in text.split(",") if item)


def _key_values(parts: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key.upper()] = value
    return fields


def gaussian_gic_lines_from_xyzin(
    path: Path,
    *,
    total_symmetric_only: bool = False,
    freeze_non_total: bool | None = None,
) -> list[str]:
    from .typed_onic_artifact import read_typed_onic_artifact_from_xyzin

    typed_artifact = read_typed_onic_artifact_from_xyzin(Path(path), required=False)
    if typed_artifact is not None:
        return gaussian_gic_lines_from_typed_onic_artifact(
            typed_artifact,
            total_symmetric_only=total_symmetric_only,
            freeze_non_total=bool(freeze_non_total),
        )
    gic = section_content(read_sectioned_lines(Path(path)), "GIC")
    if freeze_non_total is None:
        # Transition-state charts must retain every physical relaxation
        # direction.  Minimum and exploration charts retain the established
        # symmetry activation policy at this translation boundary.
        freeze_non_total = _scientific_path(gic) != "TRANSITION_STATE"
    block = _subsection(gic, "GAUSSIAN_GIC")
    lines = [line for line in block if line.strip() and line.strip().upper() != "NONE"]
    total_names = set(_parse_text_list(_section_value(gic, "TOTAL_SYMMETRIC_GICS")))
    final_names = _frozen_gic_names(gic)
    if freeze_non_total:
        # Exploitation torsions remain active for minimum and transition-state
        # charts.  Exploration retains its established symmetry constraints.
        active_non_total_names = (
            _frozen_gic_names_by_family(gic, family="TORSION")
            if _scientific_path(gic) in {"MINIMUM", "TRANSITION_STATE"}
            else set()
        )
        lines = _freeze_non_total_gaussian_lines(
            lines,
            final_names=final_names,
            total_names=total_names,
            active_non_total_names=active_non_total_names,
        )
    if not total_symmetric_only:
        return lines
    if not total_names:
        return lines
    # The semantic names stored in FROZEN_GICS and the parser-safe Gaussian
    # labels can differ within one chart (for example GICnnn plus RPcknnnn in
    # an unsymmetrized C1 export).  If every final coordinate is explicitly
    # classified as totally symmetric, the complete Gaussian block is the
    # requested symmetry subspace irrespective of that label mapping.
    if final_names and final_names.issubset(total_names):
        return lines
    available_names = {
        label for line in lines if (label := _gaussian_definition_label(line)) is not None
    }
    # An unsymmetrized C1 chart retains legacy GICnnn export labels while its
    # semantic total-symmetry list carries the internal block names.  In that
    # case every coordinate is totally symmetric and filtering by disjoint
    # namespaces would incorrectly erase the complete chart.
    if total_names.isdisjoint(available_names):
        return lines
    return _gaussian_dependency_closed_lines(lines, total_names)


def gaussian_gic_lines_from_typed_onic_artifact(
    artifact: TypedOnicArtifact,
    *,
    total_symmetric_only: bool = False,
    freeze_non_total: bool = False,
) -> list[str]:
    """Serialize a complete typed-ONIC chart as Gaussian ``ReadAllGIC`` text.

    Explicit Cartesian and inverse-distance blocks are translated directly
    from their frozen coefficient operators.  Delegated natural, pseudo-bond
    and exponential-map blocks reuse the established SMITH Gaussian exporter;
    their atom numbering and user labels are remapped deterministically into
    the complete composite frame.
    """

    from .typed_onic_artifact import TypedOnicArtifact as TypedArtifact

    if not isinstance(artifact, TypedArtifact):
        raise TypeError("typed Gaussian export requires a TypedOnicArtifact")
    definition = artifact.definition
    atom_order = tuple(definition.atom_indices_one_based)
    if atom_order != tuple(range(1, len(atom_order) + 1)):
        raise GICForgeContractError(
            "Gaussian typed-ONIC export requires the canonical complete atom order"
        )
    payloads = artifact.payload_by_id
    lines: list[str] = []
    final_index = 0
    helper_index = 0
    for block in definition.blocks:
        prefix = _TYPED_GAUSSIAN_PREFIX[block.representation]
        final_labels = tuple(
            f"{prefix}{index:04d}"
            for index in range(final_index + 1, final_index + block.target_rank + 1)
        )
        final_index += block.target_rank
        retained = tuple(
            not total_symmetric_only or is_total_symmetric_irrep(block.exact_retained_group, irrep)
            for irrep in block.irrep_labels
        )
        options = tuple(
            _typed_gaussian_final_option(
                block,
                irrep,
                freeze_non_total=freeze_non_total,
            )
            for irrep in block.irrep_labels
        )
        if block.representation == "SYMMETRY_ADAPTED_CARTESIAN":
            lines.extend(
                _gaussian_cartesian_block_lines(
                    block,
                    final_labels=final_labels,
                    retained=retained,
                    options=options,
                )
            )
            continue
        if block.representation == "INVERSE_DISTANCE_PROJECTOR":
            lines.extend(
                _gaussian_inverse_distance_block_lines(
                    block,
                    final_labels=final_labels,
                    retained=retained,
                    options=options,
                )
            )
            continue
        payload = payloads.get(block.identifier)
        if payload is None:
            raise GICForgeContractError(
                f"typed Gaussian export lacks payload for block {block.identifier}"
            )
        payload_lines, helper_index = _gaussian_delegated_block_lines(
            block,
            payload,
            complete_reference=definition.reference_coordinates_angstrom,
            final_labels=final_labels,
            retained=retained,
            options=options,
            helper_index=helper_index,
        )
        lines.extend(payload_lines)
    return lines


def _typed_gaussian_final_option(
    block: OnicCoordinateBlock,
    irrep: str,
    *,
    freeze_non_total: bool,
) -> str:
    if not block.active:
        return "Frozen"
    if freeze_non_total and not is_total_symmetric_irrep(block.exact_retained_group, irrep):
        return "Frozen"
    return ""


def _gaussian_cartesian_block_lines(
    block: OnicCoordinateBlock,
    *,
    final_labels: tuple[str, ...],
    retained: tuple[bool, ...],
    options: tuple[str, ...],
) -> list[str]:
    coefficients = np.asarray(block.coefficient_operator.to_dense(), dtype=float)
    expected = (block.target_rank, 3 * len(block.atom_indices_one_based))
    if coefficients.shape != expected:
        raise GICForgeContractError(
            f"Cartesian block {block.identifier} has coefficient shape {coefficients.shape}; "
            f"expected {expected}"
        )
    source_functions: list[str] = []
    for source in block.source_order:
        match = _CARTESIAN_SOURCE_RE.fullmatch(source)
        if match is None:
            raise GICForgeContractError(
                f"Cartesian block {block.identifier} has invalid source {source!r}"
            )
        source_functions.append(f"{match.group('axis')}({int(match.group('atom'))})")
    reference = np.asarray(block.reference_coordinates_angstrom, dtype=float).reshape(-1)
    coefficients = _normalize_gaussian_export_rows(coefficients)
    reference_values = coefficients @ reference
    lines: list[str] = []
    for row, label, keep, option, reference_value in zip(
        coefficients,
        final_labels,
        retained,
        options,
        reference_values,
        strict=True,
    ):
        if not keep:
            continue
        scaled = row * BOHR_TO_ANGSTROM
        expression = _gaussian_centered_linear_expression(
            tuple(zip(scaled, source_functions, strict=True)),
            reference_value=float(reference_value),
        )
        lines.append(_gaussian_typed_final_line(label, expression, option=option))
    return lines


def _gaussian_inverse_distance_block_lines(
    block: OnicCoordinateBlock,
    *,
    final_labels: tuple[str, ...],
    retained: tuple[bool, ...],
    options: tuple[str, ...],
) -> list[str]:
    coefficients = np.asarray(block.coefficient_operator.to_dense(), dtype=float)
    expected = (block.target_rank, block.source_count)
    if coefficients.shape != expected:
        raise GICForgeContractError(
            f"inverse-distance block {block.identifier} has coefficient shape "
            f"{coefficients.shape}; expected {expected}"
        )
    pair_functions: list[str] = []
    local_lookup = {atom: index for index, atom in enumerate(block.atom_indices_one_based)}
    reference = np.asarray(block.reference_coordinates_angstrom, dtype=float)
    reference_sources: list[float] = []
    for source in block.source_order:
        match = _INVERSE_DISTANCE_SOURCE_RE.fullmatch(source)
        if match is None:
            raise GICForgeContractError(
                f"inverse-distance block {block.identifier} has invalid source {source!r}"
            )
        left = int(match.group("left"))
        right = int(match.group("right"))
        if left not in local_lookup or right not in local_lookup or left == right:
            raise GICForgeContractError(
                f"inverse-distance source {source!r} is outside its owned block"
            )
        distance = float(
            np.linalg.norm(reference[local_lookup[left]] - reference[local_lookup[right]])
        )
        if distance <= 0.0:
            raise GICForgeContractError(
                f"inverse-distance source {source!r} is singular at the reference geometry"
            )
        pair_functions.append(f"1/R({left},{right})")
        reference_sources.append(1.0 / distance)
    coefficients = _normalize_gaussian_export_rows(coefficients)
    reference_values = coefficients @ np.asarray(reference_sources, dtype=float)
    lines: list[str] = []
    for row, label, keep, option, reference_value in zip(
        coefficients,
        final_labels,
        retained,
        options,
        reference_values,
        strict=True,
    ):
        if not keep:
            continue
        # Gaussian distances are in bohr.  The frozen operator has units of
        # Angstrom^2, so multiplying by Angstrom/bohr preserves the typed
        # coordinate value in Angstrom.
        scaled = row / BOHR_TO_ANGSTROM
        expression = _gaussian_centered_linear_expression(
            tuple(zip(scaled, pair_functions, strict=True)),
            reference_value=float(reference_value),
        )
        lines.append(_gaussian_typed_final_line(label, expression, option=option))
    return lines


def _gaussian_centered_linear_expression(
    terms: tuple[tuple[float, str], ...],
    *,
    reference_value: float,
) -> str:
    addends: list[str] = []
    for coefficient, source in terms:
        if abs(float(coefficient)) <= 1.0e-14:
            continue
        addends.append(_gaussian_linear_term(coefficient, source, first=not addends))
    if not addends:
        raise GICForgeContractError("typed Gaussian coordinate has an empty coefficient row")
    if abs(reference_value) > 1.0e-14:
        addends.append(f"{-reference_value:+.12g}")
    return "".join(addends)


def _gaussian_typed_final_line(label: str, expression: str, *, option: str) -> str:
    label_text = f"{label}({option})" if option else label
    return f"{label_text} = {expression}"


def _gaussian_delegated_block_lines(
    block: OnicCoordinateBlock,
    payload: GICDefinition,
    *,
    complete_reference: tuple[tuple[float, float, float], ...],
    final_labels: tuple[str, ...],
    retained: tuple[bool, ...],
    options: tuple[str, ...],
    helper_index: int,
) -> tuple[list[str], int]:
    global_payload = _gaussian_globalized_payload_definition(
        block,
        payload,
        complete_reference=complete_reference,
    )
    gic_by_identifier = {gic.identifier: gic for gic in global_payload.gics}
    source_prefix = f"{block.identifier}."
    selected_labels: list[str] = []
    for source in block.source_order:
        if not source.startswith(source_prefix):
            raise GICForgeContractError(
                f"delegated block {block.identifier} has noncanonical source {source!r}"
            )
        identifier = source[len(source_prefix) :]
        gic = gic_by_identifier.get(identifier)
        if gic is None:
            raise GICForgeContractError(
                f"delegated block {block.identifier} references unknown GIC {identifier!r}"
            )
        selected_labels.append(_gaussian_label_for_gic(global_payload, gic))
    if len(set(selected_labels)) != len(selected_labels):
        raise GICForgeContractError(
            f"delegated block {block.identifier} has duplicate Gaussian coordinate labels"
        )

    kept_final = {
        old_label: (new_label, option)
        for old_label, new_label, keep, option in zip(
            selected_labels,
            final_labels,
            retained,
            options,
            strict=True,
        )
        if keep
    }
    if not kept_final:
        return [], helper_index
    raw_lines = _gaussian_gic_block_lines(global_payload)
    selected_lines = _gaussian_reverse_dependency_closed_lines(
        raw_lines,
        set(kept_final),
    )
    labels = tuple(
        label for line in selected_lines if (label := _gaussian_definition_label(line)) is not None
    )
    if not set(kept_final).issubset(labels):
        missing = sorted(set(kept_final) - set(labels))
        raise GICForgeContractError(
            f"delegated block {block.identifier} cannot export Gaussian rows {missing}"
        )
    rename: dict[str, str] = {
        old_label: new_label for old_label, (new_label, _option) in kept_final.items()
    }
    for label in labels:
        if label in rename:
            continue
        helper_index += 1
        rename[label] = f"V{helper_index:05d}"

    output: list[str] = []
    for line in selected_lines:
        if "=" not in line:
            continue
        old_label = _gaussian_definition_label(line)
        if old_label is None:
            continue
        _label_text, expression = _gaussian_assignment_parts(line)
        expression = _replace_gaussian_user_labels(expression.strip(), rename)
        if old_label in kept_final:
            new_label, option = kept_final[old_label]
            output.append(_gaussian_typed_final_line(new_label, expression, option=option))
            continue
        original_label, _expression = _gaussian_assignment_parts(line)
        suffix = original_label[len(old_label) :]
        output.append(f"{rename[old_label]}{suffix} = {expression}")
    return output, helper_index


def _gaussian_globalized_payload_definition(
    block: OnicCoordinateBlock,
    payload: GICDefinition,
    *,
    complete_reference: tuple[tuple[float, float, float], ...],
) -> GICDefinition:
    payload_atoms, payload_frame = payload_owned_atom_frame(
        block.atom_indices_one_based,
        payload_natoms=len(payload.reference_coordinates_angstrom),
        payload_name=f"typed Gaussian block {block.identifier}",
        explicit_local_order=block.kind == "RELATIVE_POSE",
    )
    if payload_frame == "LOCAL":
        atom_map = dict(zip(payload_atoms, block.atom_indices_one_based, strict=True))
    else:
        atom_map = {atom: atom for atom in range(1, len(complete_reference) + 1)}

    def mapped(atoms: tuple[int, ...]) -> tuple[int, ...]:
        try:
            return tuple(atom_map[atom] for atom in atoms)
        except KeyError as exc:
            raise GICForgeContractError(
                f"typed Gaussian block {block.identifier} contains an atom outside its frame"
            ) from exc

    primitives = tuple(
        replace(
            primitive,
            atoms=mapped(primitive.atoms),
            ref_atoms=mapped(primitive.ref_atoms),
            frame_atoms=mapped(primitive.frame_atoms),
            ref_frame_atoms=mapped(primitive.ref_frame_atoms),
        )
        for primitive in payload.primitives
    )
    return replace(
        payload,
        reference_coordinates_angstrom=complete_reference,
        primitives=primitives,
        pseudo_bonds=tuple(
            (atom_map[left], atom_map[right]) for left, right in payload.pseudo_bonds
        ),
        local_xh_bonds=tuple(
            (atom_map[left], atom_map[right]) for left, right in payload.local_xh_bonds
        ),
    )


def _gaussian_reverse_dependency_closed_lines(
    lines: list[str],
    wanted_names: set[str],
) -> list[str]:
    definitions: dict[str, str] = {}
    for line in lines:
        label = _gaussian_definition_label(line)
        if label is None:
            continue
        definitions[label] = line
    definition_names = set(definitions)
    selected: set[str] = set()
    stack = [name for name in wanted_names if name in definitions]
    while stack:
        name = stack.pop()
        if name in selected:
            continue
        selected.add(name)
        _label, expression = _gaussian_assignment_parts(definitions[name])
        stack.extend(_gaussian_user_dependencies(expression, definition_names))
    return [
        line
        for line in lines
        if (label := _gaussian_definition_label(line)) is not None and label in selected
    ]


def _replace_gaussian_user_labels(expression: str, rename: dict[str, str]) -> str:
    if not rename:
        return expression
    pattern = re.compile(
        r"(?<![A-Za-z0-9_.:])("
        + "|".join(re.escape(label) for label in sorted(rename, key=len, reverse=True))
        + r")(?![A-Za-z0-9_.:])"
    )
    return pattern.sub(lambda match: rename[match.group(1)], expression)


def _gaussian_user_dependencies(expression: str, labels: set[str]) -> tuple[str, ...]:
    """Find frozen helper labels, including typed labels containing dots."""

    if not labels:
        return ()
    pattern = re.compile(
        r"(?<![A-Za-z0-9_.:])("
        + "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
        + r")(?![A-Za-z0-9_.:])"
    )
    found: list[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(expression):
        label = match.group(1)
        if label not in seen:
            found.append(label)
            seen.add(label)
    return tuple(found)


def _freeze_non_total_gaussian_lines(
    lines: list[str],
    *,
    final_names: set[str],
    total_names: set[str],
    active_non_total_names: set[str] | None = None,
) -> list[str]:
    if not final_names or not total_names:
        return lines
    active_names = set(active_non_total_names or ())
    frozen = [
        _gaussian_line_with_frozen_label(line)
        if (
            (label := _gaussian_definition_label(line)) in final_names
            and label not in total_names
            and label not in active_names
        )
        else line
        for line in lines
    ]
    non_total_names = final_names - total_names - active_names
    output: list[str] = []
    for line in frozen:
        if "=" not in line:
            output.append(line)
            continue
        label, expression = _gaussian_assignment_parts(line)
        base = label.strip().split("(", 1)[0]
        dependencies = set(_GAUSSIAN_DEPENDENCY_RE.findall(expression))
        if base.startswith(("QPck", "PhiP")) and dependencies.intersection(non_total_names):
            output.append(_gaussian_line_with_frozen_label(line))
        else:
            output.append(line)
    return output


def _gaussian_line_with_frozen_label(line: str) -> str:
    if "=" not in line:
        return line
    label, expression = _gaussian_assignment_parts(line)
    base = label.strip().split("(", 1)[0]
    return f"{base}(Frozen) = {expression.strip()}"


def _frozen_gic_names(gic_section: list[str]) -> set[str]:
    names: set[str] = set()
    for line in _subsection(gic_section, "FROZEN_GICS"):
        text = line.strip()
        if not text or text.upper() == "NONE":
            continue
        fields = _key_values(text.split()[1:])
        name = fields.get("NAME")
        if name:
            names.add(name)
    return names


def _frozen_gic_names_by_family(
    gic_section: list[str],
    *,
    family: str,
) -> set[str]:
    wanted = str(family).strip().upper()
    names: set[str] = set()
    for line in _subsection(gic_section, "FROZEN_GICS"):
        text = line.strip()
        if not text or text.upper() == "NONE":
            continue
        fields = _key_values(text.split()[1:])
        name = fields.get("NAME")
        if name and fields.get("FAMILY", "").upper() == wanted:
            names.add(name)
    return names


def _scientific_path(gic_section: list[str]) -> str:
    for line in _subsection(gic_section, "SEMANTIC_DIAGNOSTICS"):
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].upper() == "SCIENTIFIC_PATH":
            return parts[1].strip().upper()
    return ""


def _gaussian_dependency_closed_lines(lines: list[str], wanted_names: set[str]) -> list[str]:
    definitions: dict[str, str] = {}
    dependencies: dict[str, tuple[str, ...]] = {}
    for line in lines:
        label = _gaussian_definition_label(line)
        if label is None:
            continue
        definitions[label] = line
        dependencies[label] = tuple(_gaussian_definition_dependencies(line))

    selected: set[str] = set()
    stack = [name for name in wanted_names if name in definitions]
    while stack:
        name = stack.pop()
        if name in selected:
            continue
        selected.add(name)
        stack.extend(
            dependency for dependency in dependencies.get(name, ()) if dependency in definitions
        )

    # QPck/PhiP are forward-derived from selected RPck auxiliaries. Include a
    # polar functional when its complete dependency set is already in the
    # requested symmetry-closed source space.
    changed = True
    while changed:
        changed = False
        for name, required in dependencies.items():
            if name in selected or not name.startswith(("QPck", "PhiP")):
                continue
            source_names = tuple(item for item in required if item in definitions)
            if source_names and set(source_names).issubset(selected):
                selected.add(name)
                changed = True

    return [
        line
        for line in lines
        if (label := _gaussian_definition_label(line)) is not None and label in selected
    ]


def _gaussian_definition_label(line: str) -> str | None:
    if _gaussian_assignment_index(line) is None:
        return None
    label, _expression = _gaussian_assignment_parts(line)
    if not label:
        return None
    return label.split("(", 1)[0].strip()


def _gaussian_definition_dependencies(line: str) -> tuple[str, ...]:
    if _gaussian_assignment_index(line) is None:
        return ()
    label = _gaussian_definition_label(line)
    _label, expression = _gaussian_assignment_parts(line)
    dependencies: list[str] = []
    seen: set[str] = set()
    for token in _GAUSSIAN_DEPENDENCY_RE.findall(expression):
        if token == label or token in seen:
            continue
        seen.add(token)
        dependencies.append(token)
    return tuple(dependencies)


def _gaussian_assignment_index(line: str) -> int | None:
    """Return the top-level assignment separator, ignoring ``Value=``."""

    depth = 0
    for index, character in enumerate(line):
        if character == "(":
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        elif character == "=" and depth == 0:
            return index
    return None


def _gaussian_assignment_parts(line: str) -> tuple[str, str]:
    index = _gaussian_assignment_index(line)
    if index is None:
        raise ValueError(f"Gaussian line has no top-level assignment: {line!r}")
    return line[:index].strip(), line[index + 1 :].strip()


def _gaussian_gic_block_lines(definition: GICDefinition) -> list[str]:
    coords = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    lines: list[str] = []
    virtual_centers = _gaussian_virtual_center_atoms(definition.primitives)
    if virtual_centers:
        for center_id, atoms in sorted(virtual_centers.items()):
            lines.extend(_gaussian_virtual_center_lines(center_id, atoms))
    fragment_atoms = _gaussian_fragment_atoms(definition.primitives)
    if fragment_atoms:
        for fragment_id, atoms in sorted(fragment_atoms.items()):
            if len(atoms) == 1:
                # Gaussian/GDV rejects Fragment(i) for a one-atom body with
                # ``RP2Crd: Wrong count of atoms in a fragment``.  SMITH's
                # singleton fragment is nevertheless a valid translational
                # body, so expose its center directly through the atom's
                # Cartesian coordinates.  No fragment primitive is needed.
                lines.extend(_gaussian_singleton_center_lines(fragment_id, atoms[0]))
                continue
            lines.append(f"{fragment_id}=Fragment({_atom_list(atoms)})")
            lines.extend(_gaussian_center_lines(fragment_id))
    component_atoms = {**virtual_centers, **fragment_atoms}
    for context in _gaussian_axial_pose_contexts(definition.primitives):
        lines.extend(
            _gaussian_axial_pose_lines(
                *context,
                component_atoms=component_atoms,
                coords=coords,
            )
        )
    frame_contexts = _gaussian_frame_contexts(
        definition.primitives,
        component_atoms=component_atoms,
        coords=coords,
    )
    emitted_primary: set[tuple[str, str]] = set()
    for (fragment_id, frame_key), suffixes in sorted(frame_contexts.items()):
        if len(frame_key) == 1:
            continue
        primary_suffix, secondary_suffix = suffixes
        lines.extend(
            _gaussian_frame_lines(
                fragment_id,
                component_atoms[fragment_id],
                coords=coords,
                frame_atoms=frame_key,
                primary_suffix=primary_suffix,
                secondary_suffix=secondary_suffix,
                include_primary=(fragment_id, primary_suffix) not in emitted_primary,
            )
        )
        emitted_primary.add((fragment_id, primary_suffix))
    quaternion_contexts = _gaussian_quaternion_contexts(
        definition.primitives,
        component_atoms=component_atoms,
        coords=coords,
    )
    linear_references = _gaussian_linear_frame_references(definition.primitives)
    for (fragment_id, frame_key), (ref_id, ref_frame) in sorted(linear_references.items()):
        primary_suffix, secondary_suffix = frame_contexts[(fragment_id, frame_key)]
        reference_suffixes = frame_contexts[(ref_id, ref_frame)]
        lines.extend(
            _gaussian_linear_frame_lines(
                fragment_id,
                component_atoms[fragment_id],
                anchor_atom=frame_key[0],
                reference_id=ref_id,
                reference_atoms=component_atoms[ref_id],
                reference_frame_atoms=ref_frame,
                coords=coords,
                primary_suffix=primary_suffix,
                secondary_suffix=secondary_suffix,
                reference_suffixes=reference_suffixes,
                rotation_suffix=quaternion_contexts[(fragment_id, ref_id, frame_key, ref_frame)],
            )
        )
    for (frag_id, ref_id, frag_frame, ref_frame), suffix in sorted(quaternion_contexts.items()):
        if len(frag_frame) == 1:
            continue
        moving_reference_frame, fixed_reference_frame = _fragment_relative_frames(
            coords,
            component_atoms[frag_id],
            component_atoms[ref_id],
            frame_atoms=frag_frame,
            ref_frame_atoms=ref_frame,
            gauge_reference_coords=coords,
        )
        lines.extend(
            _gaussian_quaternion_lines(
                frag_id,
                ref_id,
                suffix=suffix,
                moving_suffixes=frame_contexts[(frag_id, frag_frame)],
                reference_suffixes=frame_contexts[(ref_id, ref_frame)],
                reference_rotation=(moving_reference_frame.T @ fixed_reference_frame),
            )
        )

    polar_lines = _gaussian_ring_puckering_function_lines(definition)
    polar_components = {
        token
        for line in polar_lines
        for token in re.findall(r"\b(?:[A-Za-z0-9]+)?RPck[0-9]+\b", line)
    }

    for gic in definition.gics:
        expression = _gaussian_expression_for_gic(
            definition,
            gic,
            frame_contexts=frame_contexts,
            quaternion_contexts=quaternion_contexts,
        )
        if expression:
            label = _gaussian_label_for_gic(definition, gic)
            if gic.family == "RING_PUCKER_COMPONENT" and label in polar_components:
                label = f"{label}(Inactive)"
            lines.append(f"{label} = {expression}")
    lines.extend(polar_lines)
    return lines


def _gaussian_label_for_gic(definition: GICDefinition, gic: FrozenGIC) -> str:
    if gic.family == "RING_PUCKER_COMPONENT":
        return gic.name
    return gic.name if definition.symmetrize else gic.identifier


def _gaussian_ring_puckering_function_lines(definition: GICDefinition) -> list[str]:
    """Serialize the ring-polar chart already frozen by SMITH."""

    lines: list[str] = []
    pair_index = 0
    for record in definition.semantic_diagnostics:
        if not record.startswith("RING_PHASE_CHART ") or " CHART=RING_POLAR " not in record:
            continue
        component_match = re.search(r"\bCOMPONENTS=([^ ]+)", record)
        mode_match = re.search(r"\bMODE=([^ ]+)", record)
        if component_match is None or mode_match is None:
            raise GICForgeContractError(f"incomplete frozen ring-polar chart: {record}")
        components = tuple(name for name in component_match.group(1).split(",") if name)
        mode = mode_match.group(1)
        pair_index += 1
        if mode == "SINGLE_COMPONENT" and len(components) == 1:
            label = components[0]
            lines.append(f"QPck{pair_index:04d} = SQRT({label}*{label})")
            lines.append(f"PhiP{pair_index:04d} = {label}")
        elif mode in {"ATAN2", "ZERO_GAUGE"} and len(components) == 2:
            left, right = components
            lines.append(f"QPck{pair_index:04d} = SQRT({left}*{left}+{right}*{right})")
            phase = f"ATAN2({right},{left})" if mode == "ATAN2" else f"0.0*{left}"
            lines.append(f"PhiP{pair_index:04d} = {phase}")
        else:
            raise GICForgeContractError(f"invalid frozen ring-polar chart: {record}")
    return lines


def _ring_pucker_group_key_for_gic(
    gic: FrozenGIC,
    primitive_by_id: dict[str, GICPrimitive],
) -> tuple[int, ...] | None:
    primitive = _single_source_primitive(gic, primitive_by_id)
    if primitive is not None and primitive.function == "RPCK":
        return primitive.atoms
    atoms: set[int] = set()
    for primitive_id, _coefficient in gic.coefficients or ():
        primitive = primitive_by_id.get(primitive_id)
        if primitive is None:
            return None
        if primitive.family != "RING_PUCKER_COMPONENT" or primitive.function not in {"D", "U"}:
            return None
        atoms.update(primitive.atoms)
    if len(atoms) < 4:
        return None
    return tuple(sorted(atoms))


def _ring_ref_text(ring: tuple[int, ...]) -> str:
    return "RING:" + "-".join(str(atom) for atom in ring)


def _ring_ref_atoms(ref: str) -> tuple[int, ...] | None:
    if not ref.startswith("RING:"):
        return None
    try:
        atoms = tuple(int(atom) for atom in ref[5:].split("-") if atom)
    except ValueError:
        return None
    if len(atoms) < 4:
        return None
    return atoms


def _ring_refs_from_primitive(primitive: GICPrimitive) -> tuple[tuple[int, ...], ...]:
    rings: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for ref in primitive.refs:
        ring = _ring_ref_atoms(ref)
        if ring is None or ring in seen:
            continue
        seen.add(ring)
        rings.append(ring)
    return tuple(rings)


def _ring_pucker_source_ring_keys_for_gic(
    gic: FrozenGIC,
    primitive_by_id: dict[str, GICPrimitive],
) -> tuple[tuple[int, ...], ...]:
    primitive = _single_source_primitive(gic, primitive_by_id)
    if (
        primitive is not None
        and primitive.family == "RING_PUCKER_COMPONENT"
        and primitive.function == "RPCK"
    ):
        return (primitive.atoms,)

    rings: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for primitive_id, _coefficient in gic.coefficients or ():
        primitive = primitive_by_id.get(primitive_id)
        if primitive is None or primitive.family != "RING_PUCKER_COMPONENT":
            continue
        primitive_rings = _ring_refs_from_primitive(primitive)
        if not primitive_rings and primitive.function == "RPCK":
            primitive_rings = (primitive.atoms,)
        for ring in primitive_rings:
            if ring in seen:
                continue
            seen.add(ring)
            rings.append(ring)
    if rings:
        return tuple(rings)

    group_key = _ring_pucker_group_key_for_gic(gic, primitive_by_id)
    return (group_key,) if group_key is not None else ()


def _ring_edges(ring: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    return tuple(
        tuple(sorted((atom, ring[(index + 1) % len(ring)]))) for index, atom in enumerate(ring)
    )


def _all_ring_pucker_source_ring_keys(
    definition: GICDefinition,
    primitive_by_id: dict[str, GICPrimitive],
) -> tuple[tuple[int, ...], ...]:
    rings: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for primitive in definition.primitives:
        if primitive.family != "RING_PUCKER_COMPONENT":
            continue
        primitive_rings = _ring_refs_from_primitive(primitive)
        if not primitive_rings and primitive.function == "RPCK":
            primitive_rings = (primitive.atoms,)
        for ring in primitive_rings:
            if ring in seen:
                continue
            seen.add(ring)
            rings.append(ring)
    if rings:
        return tuple(rings)

    for gic in definition.gics:
        if gic.family != "RING_PUCKER_COMPONENT":
            continue
        for ring in _ring_pucker_source_ring_keys_for_gic(gic, primitive_by_id):
            if ring in seen:
                continue
            seen.add(ring)
            rings.append(ring)
    return tuple(rings)


def _condensed_ring_pucker_keys(
    definition: GICDefinition,
    primitive_by_id: dict[str, GICPrimitive],
) -> set[tuple[int, ...]]:
    rings = _all_ring_pucker_source_ring_keys(definition, primitive_by_id)
    edge_to_rings: dict[tuple[int, int], list[tuple[int, ...]]] = {}
    for ring in rings:
        for edge in _ring_edges(ring):
            edge_to_rings.setdefault(edge, []).append(ring)
    condensed: set[tuple[int, ...]] = set()
    for shared_rings in edge_to_rings.values():
        if len(shared_rings) > 1:
            condensed.update(shared_rings)
    return condensed


def _gaussian_expression_for_gic(
    definition: GICDefinition,
    gic: FrozenGIC,
    *,
    frame_contexts: dict[tuple[str, tuple[int, ...]], tuple[str, str]] | None = None,
    quaternion_contexts: dict[tuple[str, str, tuple[int, ...], tuple[int, ...]], str] | None = None,
) -> str | None:
    primitive_by_id = {primitive.identifier: primitive for primitive in definition.primitives}
    coefficients = _normalized_gaussian_gic_coefficients(gic)
    if len(coefficients) == 1 and abs(coefficients[0][1] - 1.0) <= 1.0e-12:
        primitive = primitive_by_id.get(coefficients[0][0])
        if primitive is None:
            raise GICForgeContractError(
                f"unknown primitive {coefficients[0][0]!r} in frozen GIC {gic.identifier}"
            )
        return _gaussian_expression_for_primitive(
            primitive,
            frame_contexts=frame_contexts,
            quaternion_contexts=quaternion_contexts,
        )

    terms: list[str] = []
    for primitive_id, coefficient in coefficients:
        primitive = primitive_by_id.get(primitive_id)
        if primitive is None:
            raise GICForgeContractError(
                f"unknown primitive {primitive_id!r} in frozen GIC {gic.identifier}"
            )
        expression = _gaussian_expression_for_primitive(
            primitive,
            frame_contexts=frame_contexts,
            quaternion_contexts=quaternion_contexts,
        )
        if expression is None:
            return None
        terms.append(_gaussian_linear_term(coefficient, expression, first=not terms))
    return "".join(terms) if terms else None


def _normalized_gaussian_gic_coefficients(
    gic: FrozenGIC,
) -> tuple[tuple[str, float], ...]:
    """Return a unit-norm primitive-space row for Gaussian export.

    SONIC construction and rank/symmetry analysis deliberately retain their
    native internal scaling.  Gaussian, however, uses the coefficient scale
    of the exported primitive combination when initializing its force
    constants.  Normalize only at this serialization boundary so both
    contracts remain independent and reproducible.
    """

    coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
    # Keep Gaussian primitive expressions in their native coordinate units;
    # angular values are represented in degrees by ReadAllGIC.
    native = np.asarray([float(value) for _identifier, value in coefficients], dtype=float)
    norm = float(np.linalg.norm(native))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise GICForgeContractError(
            f"Gaussian GIC {gic.identifier} has a zero or non-finite primitive-space norm"
        )
    return tuple(
        (identifier, float(native[index] / norm))
        for index, (identifier, _value) in enumerate(coefficients)
    )


def _normalize_gaussian_export_rows(rows: np.ndarray) -> np.ndarray:
    """Normalize every final typed-ONIC row in its source primitive space."""

    values = np.asarray(rows, dtype=float)
    norms = np.linalg.norm(values, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-14):
        raise GICForgeContractError(
            "Gaussian typed-ONIC export contains a zero or non-finite coefficient row"
        )
    return values / norms[:, None]


def _gaussian_linear_term(coefficient: float, expression: str, *, first: bool) -> str:
    sign = "-" if coefficient < 0.0 else "+"
    magnitude = abs(float(coefficient))
    coefficient_text = f"{magnitude:.12g}"
    term = f"{coefficient_text}*({_strip_gaussian_outer_parentheses(expression)})"
    if first:
        return f"-{term}" if sign == "-" else term
    return f"{sign}{term}"


def _strip_gaussian_outer_parentheses(expression: str) -> str:
    text = expression.strip()
    if text.startswith("(") and text.endswith(")"):
        return text[1:-1]
    return text


def _gaussian_expression_for_primitive(
    primitive: GICPrimitive,
    *,
    frame_contexts: dict[tuple[str, tuple[int, ...]], tuple[str, str]] | None = None,
    quaternion_contexts: dict[tuple[str, str, tuple[int, ...], tuple[int, ...]], str] | None = None,
) -> str | None:
    if primitive.is_gaussian_native:
        return primitive.gaussian_expression()
    if primitive.function == "SOOP":
        # SOOP is a frozen Cartesian covector q = b.x - q0.  Gaussian's X/Y/Z
        # expression primitives are evaluated in bohr, whereas the frozen
        # ORACLE/SMITH frame is stored in Angstrom.  Scale every Cartesian
        # coefficient at this serialization boundary so the exported scalar
        # and its Wilson row describe exactly the resident SOOP coordinate.
        offset = 0.0
        components: list[tuple[int, float]] = []
        for ref in primitive.refs:
            if ref.startswith("OFFSET:"):
                offset = float(ref.split(":", 1)[1])
            elif ref.startswith("B:"):
                _tag, index_text, value_text = ref.split(":", 2)
                components.append((int(index_text), float(value_text)))
        if not components:
            raise GICForgeContractError(
                f"SOOP primitive {primitive.identifier} has no frozen Cartesian support"
            )
        terms: list[str] = []
        axes = ("X", "Y", "Z")
        for cartesian_index, coefficient in components:
            atom = cartesian_index // 3 + 1
            axis = axes[cartesian_index % 3]
            terms.append(
                _gaussian_linear_term(
                    coefficient * BOHR_TO_ANGSTROM,
                    f"{axis}({atom})",
                    first=not terms,
                )
            )
        if abs(offset) > 1.0e-15:
            sign = "-" if offset >= 0.0 else "+"
            terms.append(f"{sign}{abs(offset):.12g}")
        return "".join(terms)
    if primitive.function == "FC_DIST":
        frag_id, ref_id = primitive.refs
        return (
            f"SQRT((Cx{frag_id}-Cx{ref_id})**2+"
            f"(Cy{frag_id}-Cy{ref_id})**2+"
            f"(Cz{frag_id}-Cz{ref_id})**2)"
        )
    if primitive.function == "FCA_DIST":
        if not primitive.refs or len(primitive.ref_atoms) != 1:
            raise GICForgeContractError(
                f"FCA_DIST primitive {primitive.identifier} requires one typed center "
                "and one reference atom"
            )
        frag_id = primitive.refs[0]
        atom = primitive.ref_atoms[0]
        return (
            f"SQRT((Cx{frag_id}-X({atom}))**2+"
            f"(Cy{frag_id}-Y({atom}))**2+"
            f"(Cz{frag_id}-Z({atom}))**2)"
        )
    if primitive.function == "CENTER_ATOM_DIST":
        if not primitive.refs or len(primitive.ref_atoms) != 1:
            raise GICForgeContractError(
                f"CENTER_ATOM_DIST primitive {primitive.identifier} requires one typed "
                "center and one reference atom"
            )
        center_id = primitive.refs[0]
        atom = primitive.ref_atoms[0]
        return (
            f"SQRT((Cx{center_id}-X({atom}))**2+"
            f"(Cy{center_id}-Y({atom}))**2+"
            f"(Cz{center_id}-Z({atom}))**2)"
        )
    if primitive.function == "FTRANS":
        frag_id, ref_id = primitive.refs
        if not primitive.ref_frame_atoms:
            axis = ("x", "y", "z")[primitive.mode]
            return f"C{axis}{frag_id}-C{axis}{ref_id}"
        axis = ("P", "Q", "S")[primitive.mode]
        frame_key = (ref_id, tuple(primitive.ref_frame_atoms))
        primary_suffix, secondary_suffix = (frame_contexts or {}).get(frame_key, ("", ""))
        suffix = primary_suffix if axis == "P" else secondary_suffix
        return (
            f"(Cx{frag_id}-Cx{ref_id})*{axis}x{ref_id}{suffix}+"
            f"(Cy{frag_id}-Cy{ref_id})*{axis}y{ref_id}{suffix}+"
            f"(Cz{frag_id}-Cz{ref_id})*{axis}z{ref_id}{suffix}"
        )
    if primitive.function == "FLIN_TRANS":
        frag_id, ref_id = primitive.refs
        return f"Jt{primitive.mode + 1}{frag_id}{ref_id}"
    if primitive.function == "FAXIS":
        frag_id, ref_id = primitive.refs
        return f"Ja{primitive.mode + 1}{frag_id}{ref_id}"
    if primitive.function == "FROT":
        frag_id, ref_id = primitive.refs
        frame_key = (
            frag_id,
            ref_id,
            tuple(primitive.frame_atoms),
            tuple(primitive.ref_frame_atoms),
        )
        suffix = (quaternion_contexts or {}).get(frame_key, "")
        if len(primitive.frame_atoms) == 1:
            if primitive.mode not in {0, 1}:
                raise GICForgeContractError(
                    f"linear FROT primitive {primitive.identifier} has invalid mode "
                    f"{primitive.mode}"
                )
            return f"Ls{primitive.mode + 1}{frag_id}{ref_id}{suffix}"
        axis = ("x", "y", "z")[primitive.mode]
        return f"E{axis}{frag_id}{ref_id}{suffix}"
    if primitive.function == "RPCB":
        terms: list[str] = []
        for coefficient, atoms in _angle_component_terms_from_refs(primitive):
            expression = "A(" + ",".join(str(atom) for atom in atoms) + ")"
            terms.append(_gaussian_linear_term(coefficient, expression, first=not terms))
        return "".join(terms) if terms else None
    if primitive.function == "RPCK":
        terms: list[str] = []
        for coefficient, atoms in _ring_pucker_terms_from_refs(primitive):
            expression = "D(" + ",".join(str(atom) for atom in atoms) + ")"
            terms.append(_gaussian_linear_term(coefficient, expression, first=not terms))
        return "".join(terms) if terms else None
    if primitive.function == "IMPD":
        return primitive.gaussian_expression()
    return None


def _gaussian_fragment_atoms(
    primitives: tuple[GICPrimitive, ...],
) -> dict[str, tuple[int, ...]]:
    """Return bodies consumed by Gaussian fragment-coordinate expressions.

    ``refs`` is also used for provenance tags on native atom coordinates.  In
    particular, pseudobond contact primitives retain their ORACLE fragment IDs
    there without consuming Gaussian ``Fragment`` objects.  Fragment helpers
    must therefore be derived from the typed primitive function, never from a
    string prefix found in provenance metadata.
    """

    fragments: dict[str, tuple[int, ...]] = {}
    for primitive in primitives:
        if primitive.function in FRAGMENT_BODY_SINGLE_FUNCTIONS:
            if not primitive.refs:
                raise GICForgeContractError(
                    f"FCA_DIST primitive {primitive.identifier} has no fragment ref"
                )
            _register_gaussian_fragment(
                fragments,
                primitive.refs[0],
                primitive.atoms,
                primitive=primitive,
            )
            continue
        if primitive.function not in FRAGMENT_BODY_PAIR_FUNCTIONS:
            continue
        if len(primitive.refs) < 2:
            raise GICForgeContractError(
                f"{primitive.function} primitive {primitive.identifier} requires two fragment refs"
            )
        _register_gaussian_fragment(
            fragments,
            primitive.refs[0],
            primitive.atoms,
            primitive=primitive,
        )
        _register_gaussian_fragment(
            fragments,
            primitive.refs[1],
            primitive.ref_atoms,
            primitive=primitive,
        )
    return fragments


def _register_gaussian_fragment(
    fragments: dict[str, tuple[int, ...]],
    fragment_id: str,
    atoms: tuple[int, ...],
    *,
    primitive: GICPrimitive,
) -> None:
    members = tuple(int(atom) for atom in atoms)
    if not fragment_id or not members:
        raise GICForgeContractError(
            f"{primitive.function} primitive {primitive.identifier} has invalid "
            f"Gaussian fragment {fragment_id!r} with atoms {members!r}"
        )
    if not fragment_id.startswith("F"):
        return
    previous = fragments.setdefault(fragment_id, members)
    if set(previous) != set(members):
        raise GICForgeContractError(
            f"Gaussian fragment {fragment_id} has inconsistent atom membership"
        )


def _gaussian_axial_pose_contexts(
    primitives: tuple[GICPrimitive, ...],
) -> tuple[tuple[str, str, int, int, bool], ...]:
    """Collect each atlas-prescribed linear-reference pose exactly once."""

    contexts: dict[tuple[str, str], tuple[int, int, bool]] = {}
    for primitive in primitives:
        if primitive.function not in {"FLIN_TRANS", "FAXIS"}:
            continue
        if len(primitive.refs) < 2 or len(primitive.ref_frame_atoms) != 1:
            raise GICForgeContractError(f"invalid axial-pose primitive {primitive.identifier}")
        frag_id, ref_id = primitive.refs[:2]
        frame_atom = primitive.frame_atoms[0] if primitive.frame_atoms else 0
        key = (frag_id, ref_id)
        previous = contexts.get(key)
        current = (
            frame_atom or (previous[0] if previous else 0),
            primitive.ref_frame_atoms[0],
            primitive.function == "FAXIS" or (previous[2] if previous else False),
        )
        if previous is not None and (
            previous[1] != current[1] or (previous[0] and current[0] and previous[0] != current[0])
        ):
            raise GICForgeContractError(f"inconsistent axial-pose context {key}")
        contexts[key] = current
    return tuple(
        (frag_id, ref_id, frame_atom, ref_frame_atom, has_axis)
        for (frag_id, ref_id), (frame_atom, ref_frame_atom, has_axis) in sorted(contexts.items())
    )


def _gaussian_axial_pose_lines(
    frag_id: str,
    ref_id: str,
    frame_atom: int,
    ref_frame_atom: int,
    has_axis: bool,
    *,
    component_atoms: dict[str, tuple[int, ...]],
    coords: np.ndarray,
) -> list[str]:
    """Lower axial Jacobi and axis--axis primitives to ReadAllGIC algebra."""

    pair = f"{frag_id}{ref_id}"
    lines = [
        f"Jrp{pair}(Inactive)=SQRT((X({ref_frame_atom})-Cx{ref_id})**2+"
        f"(Y({ref_frame_atom})-Cy{ref_id})**2+(Z({ref_frame_atom})-Cz{ref_id})**2)",
        f"Jpx{pair}(Inactive)=[X({ref_frame_atom})-Cx{ref_id}]/Jrp{pair}",
        f"Jpy{pair}(Inactive)=[Y({ref_frame_atom})-Cy{ref_id}]/Jrp{pair}",
        f"Jpz{pair}(Inactive)=[Z({ref_frame_atom})-Cz{ref_id}]/Jrp{pair}",
        f"Jdx{pair}(Inactive)=Cx{frag_id}-Cx{ref_id}",
        f"Jdy{pair}(Inactive)=Cy{frag_id}-Cy{ref_id}",
        f"Jdz{pair}(Inactive)=Cz{frag_id}-Cz{ref_id}",
        f"Jt1{pair}(Inactive)=Jdx{pair}*Jpx{pair}+Jdy{pair}*Jpy{pair}+Jdz{pair}*Jpz{pair}",
        f"Jqx0{pair}(Inactive)=Jdx{pair}-Jt1{pair}*Jpx{pair}",
        f"Jqy0{pair}(Inactive)=Jdy{pair}-Jt1{pair}*Jpy{pair}",
        f"Jqz0{pair}(Inactive)=Jdz{pair}-Jt1{pair}*Jpz{pair}",
        f"Jt2{pair}(Inactive)=SQRT(Jqx0{pair}**2+Jqy0{pair}**2+Jqz0{pair}**2)",
        f"Jqx{pair}(Inactive)=Jqx0{pair}/Jt2{pair}",
        f"Jqy{pair}(Inactive)=Jqy0{pair}/Jt2{pair}",
        f"Jqz{pair}(Inactive)=Jqz0{pair}/Jt2{pair}",
        f"Jsx{pair}(Inactive)=Jpy{pair}*Jqz{pair}-Jpz{pair}*Jqy{pair}",
        f"Jsy{pair}(Inactive)=Jpz{pair}*Jqx{pair}-Jpx{pair}*Jqz{pair}",
        f"Jsz{pair}(Inactive)=Jpx{pair}*Jqy{pair}-Jpy{pair}*Jqx{pair}",
    ]
    if not has_axis:
        return lines
    if frame_atom <= 0:
        raise GICForgeContractError(f"axis--axis context {pair} has no moving anchor")
    moving_atoms = component_atoms[frag_id]
    reference_atoms = component_atoms[ref_id]
    pole_index, pole_sign, transverse = _axis_axis_stereographic_chart(
        coords,
        moving_atoms,
        reference_atoms,
        frame_atom=frame_atom,
        ref_frame_atom=ref_frame_atom,
    )
    lines.extend(
        [
            f"Jrm{pair}(Inactive)=SQRT((X({frame_atom})-Cx{frag_id})**2+"
            f"(Y({frame_atom})-Cy{frag_id})**2+(Z({frame_atom})-Cz{frag_id})**2)",
            f"Jmx{pair}(Inactive)=[X({frame_atom})-Cx{frag_id}]/Jrm{pair}",
            f"Jmy{pair}(Inactive)=[Y({frame_atom})-Cy{frag_id}]/Jrm{pair}",
            f"Jmz{pair}(Inactive)=[Z({frame_atom})-Cz{frag_id}]/Jrm{pair}",
            f"Jc1{pair}(Inactive)=Jmx{pair}*Jpx{pair}+Jmy{pair}*Jpy{pair}+Jmz{pair}*Jpz{pair}",
            f"Jc2{pair}(Inactive)=Jmx{pair}*Jqx{pair}+Jmy{pair}*Jqy{pair}+Jmz{pair}*Jqz{pair}",
            f"Jc3{pair}(Inactive)=Jmx{pair}*Jsx{pair}+Jmy{pair}*Jsy{pair}+Jmz{pair}*Jsz{pair}",
        ]
    )
    sign = "+" if pole_sign > 0.0 else "-"
    lines.extend(
        [
            f"Jad{pair}(Inactive)=1{sign}Jc{pole_index + 1}{pair}",
            f"Ja1{pair}(Inactive)=Jc{transverse[0] + 1}{pair}/Jad{pair}",
            f"Ja2{pair}(Inactive)=Jc{transverse[1] + 1}{pair}/Jad{pair}",
        ]
    )
    return lines


def _gaussian_virtual_center_atoms(
    primitives: tuple[GICPrimitive, ...],
) -> dict[str, tuple[int, ...]]:
    centers: dict[str, tuple[int, ...]] = {}
    for primitive in primitives:
        if primitive.function == "CENTER_ATOM_DIST" and primitive.refs:
            center_id = primitive.refs[0]
            if center_id.startswith("C"):
                centers.setdefault(center_id, primitive.atoms)
        if primitive.function not in {"FC_DIST", "FTRANS", "FROT"}:
            continue
        if len(primitive.refs) < 2:
            continue
        for center_id, atoms in (
            (primitive.refs[0], primitive.atoms),
            (primitive.refs[1], primitive.ref_atoms),
        ):
            if not center_id.startswith("F"):
                centers.setdefault(center_id, atoms)
    return centers




def _gaussian_frame_contexts(
    primitives: tuple[GICPrimitive, ...],
    *,
    component_atoms: dict[str, tuple[int, ...]],
    coords: np.ndarray,
) -> dict[tuple[str, tuple[int, ...]], tuple[str, str]]:
    """Return distinct local-frame gauges required by fragment primitives.

    A fragment can legitimately occur with more than one frame anchor pair.
    Gaussian helper names must distinguish those gauges; otherwise separate
    fragment translations/rotations collapse into duplicate or cancelling
    expressions.  The first context retains the historical unsuffixed names;
    later contexts receive deterministic ``K2``, ``K3``, ... suffixes.
    """
    contexts: dict[str, list[tuple[int, ...]]] = {}

    def add(fragment_id: str, frame_atoms: tuple[int, ...]) -> None:
        if not frame_atoms or fragment_id not in component_atoms:
            return
        items = contexts.setdefault(fragment_id, [])
        if frame_atoms not in items:
            items.append(frame_atoms)

    for primitive in primitives:
        if primitive.function not in {"FTRANS", "FROT"} or len(primitive.refs) < 2:
            continue
        frag_id, ref_id = primitive.refs[:2]
        if primitive.function == "FTRANS":
            if primitive.ref_frame_atoms:
                add(ref_id, tuple(primitive.ref_frame_atoms))
            continue
        moving_frame = tuple(primitive.frame_atoms) or tuple(
            _fragment_frame_anchor_atoms(component_atoms[frag_id], coords=coords)
        )
        reference_frame = tuple(primitive.ref_frame_atoms) or tuple(
            _fragment_frame_anchor_atoms(component_atoms[ref_id], coords=coords)
        )
        add(frag_id, moving_frame)
        add(ref_id, reference_frame)

    result: dict[tuple[str, tuple[int, ...]], tuple[str, str]] = {}
    for fragment_id, frame_list in contexts.items():
        primary_suffixes: dict[int, str] = {}
        for index, frame_atoms in enumerate(frame_list, start=1):
            p_atom = frame_atoms[0]
            if p_atom not in primary_suffixes:
                primary_index = len(primary_suffixes) + 1
                primary_suffixes[p_atom] = "" if primary_index == 1 else f"K{primary_index}"
            result[(fragment_id, frame_atoms)] = (
                primary_suffixes[p_atom],
                "" if index == 1 else f"K{index}",
            )
    return result


def _gaussian_quaternion_contexts(
    primitives: tuple[GICPrimitive, ...],
    *,
    component_atoms: dict[str, tuple[int, ...]],
    coords: np.ndarray,
) -> dict[tuple[str, str, tuple[int, ...], tuple[int, ...]], str]:
    """Return distinct moving/reference frame contexts for every FROT pair."""
    contexts: dict[tuple[str, str], list[tuple[tuple[int, ...], tuple[int, ...]]]] = {}
    for primitive in primitives:
        if primitive.function != "FROT" or len(primitive.refs) < 2:
            continue
        frag_id, ref_id = primitive.refs[:2]
        moving_frame = tuple(primitive.frame_atoms) or tuple(
            _fragment_frame_anchor_atoms(component_atoms[frag_id], coords=coords)
        )
        reference_frame = tuple(primitive.ref_frame_atoms) or tuple(
            _fragment_frame_anchor_atoms(component_atoms[ref_id], coords=coords)
        )
        pair_contexts = contexts.setdefault((frag_id, ref_id), [])
        context = (moving_frame, reference_frame)
        if context not in pair_contexts:
            pair_contexts.append(context)

    result: dict[tuple[str, str, tuple[int, ...], tuple[int, ...]], str] = {}
    for (frag_id, ref_id), pair_contexts in contexts.items():
        for index, (moving_frame, reference_frame) in enumerate(pair_contexts, start=1):
            suffix = "" if index == 1 else f"K{index}"
            result[(frag_id, ref_id, moving_frame, reference_frame)] = suffix
    return result




def _gaussian_linear_frame_references(
    primitives: tuple[GICPrimitive, ...],
) -> dict[tuple[str, tuple[int, ...]], tuple[str, tuple[int, ...]]]:
    """Return the frozen nonlinear reference frame for each linear FROT body."""

    references: dict[tuple[str, tuple[int, ...]], tuple[str, tuple[int, ...]]] = {}
    for primitive in primitives:
        if (
            primitive.function != "FROT"
            or len(primitive.refs) < 2
            or len(primitive.frame_atoms) != 1
            or len(primitive.ref_frame_atoms) != 2
        ):
            continue
        key = (primitive.refs[0], tuple(primitive.frame_atoms))
        value = (primitive.refs[1], tuple(primitive.ref_frame_atoms))
        previous = references.setdefault(key, value)
        if previous != value:
            raise GICForgeContractError(
                f"linear fragment frame {key[0]} has inconsistent nonlinear references"
            )
    return references


def _gaussian_center_lines(fragment_id: str) -> list[str]:
    return [
        f"Cx{fragment_id}(Inactive)=XCntr({fragment_id})",
        f"Cy{fragment_id}(Inactive)=YCntr({fragment_id})",
        f"Cz{fragment_id}(Inactive)=ZCntr({fragment_id})",
    ]


def _gaussian_singleton_center_lines(fragment_id: str, atom: int) -> list[str]:
    """Expose a monatomic SMITH fragment without Gaussian's invalid Fragment(i)."""

    return [
        f"Cx{fragment_id}(Inactive)=X({atom})",
        f"Cy{fragment_id}(Inactive)=Y({atom})",
        f"Cz{fragment_id}(Inactive)=Z({atom})",
    ]


def _gaussian_virtual_center_lines(center_id: str, atoms: tuple[int, ...]) -> list[str]:
    if not atoms:
        return []
    denominator = len(atoms)
    return [
        f"Cx{center_id}(Inactive)=({_gaussian_axis_sum('X', atoms)})/{denominator}",
        f"Cy{center_id}(Inactive)=({_gaussian_axis_sum('Y', atoms)})/{denominator}",
        f"Cz{center_id}(Inactive)=({_gaussian_axis_sum('Z', atoms)})/{denominator}",
    ]


def _gaussian_axis_sum(axis: str, atoms: tuple[int, ...]) -> str:
    return "+".join(f"{axis}({atom})" for atom in atoms)


def _gaussian_frame_lines(
    fragment_id: str,
    atoms: tuple[int, ...],
    *,
    coords: np.ndarray,
    frame_atoms: tuple[int, ...] = (),
    primary_suffix: str = "",
    secondary_suffix: str = "",
    include_primary: bool = True,
) -> list[str]:
    p_atom, q_atom = (
        tuple(frame_atoms) if frame_atoms else _fragment_frame_anchor_atoms(atoms, coords=coords)
    )
    primary_tag = f"{fragment_id}{primary_suffix}"
    secondary_tag = f"{fragment_id}{secondary_suffix}"
    lines: list[str] = []
    if include_primary:
        lines.extend(
            [
                f"RP{primary_tag}(Inactive)=SQRT((X({p_atom})-Cx{fragment_id})**2+"
                f"(Y({p_atom})-Cy{fragment_id})**2+(Z({p_atom})-Cz{fragment_id})**2)",
                f"Px{primary_tag}(Inactive)=[X({p_atom})-Cx{fragment_id}]/RP{primary_tag}",
                f"Py{primary_tag}(Inactive)=[Y({p_atom})-Cy{fragment_id}]/RP{primary_tag}",
                f"Pz{primary_tag}(Inactive)=[Z({p_atom})-Cz{fragment_id}]/RP{primary_tag}",
            ]
        )
    lines.extend(
        [
            f"QQx{secondary_tag}(Inactive)=Py{primary_tag}*[Z({q_atom})-Cz{fragment_id}]-Pz{primary_tag}*[Y({q_atom})-Cy{fragment_id}]",
            f"QQy{secondary_tag}(Inactive)=Pz{primary_tag}*[X({q_atom})-Cx{fragment_id}]-Px{primary_tag}*[Z({q_atom})-Cz{fragment_id}]",
            f"QQz{secondary_tag}(Inactive)=Px{primary_tag}*[Y({q_atom})-Cy{fragment_id}]-Py{primary_tag}*[X({q_atom})-Cx{fragment_id}]",
            f"RQ{secondary_tag}(Inactive)=SQRT(QQx{secondary_tag}**2+QQy{secondary_tag}**2+QQz{secondary_tag}**2)",
            f"Qx{secondary_tag}(Inactive)=QQx{secondary_tag}/RQ{secondary_tag}",
            f"Qy{secondary_tag}(Inactive)=QQy{secondary_tag}/RQ{secondary_tag}",
            f"Qz{secondary_tag}(Inactive)=QQz{secondary_tag}/RQ{secondary_tag}",
            f"Sx{secondary_tag}(Inactive)=Py{primary_tag}*Qz{secondary_tag}-Pz{primary_tag}*Qy{secondary_tag}",
            f"Sy{secondary_tag}(Inactive)=Pz{primary_tag}*Qx{secondary_tag}-Px{primary_tag}*Qz{secondary_tag}",
            f"Sz{secondary_tag}(Inactive)=Px{primary_tag}*Qy{secondary_tag}-Py{primary_tag}*Qx{secondary_tag}",
        ]
    )
    return lines


def _gaussian_linear_frame_lines(
    fragment_id: str,
    atoms: tuple[int, ...],
    *,
    anchor_atom: int,
    reference_id: str,
    reference_atoms: tuple[int, ...],
    reference_frame_atoms: tuple[int, ...],
    coords: np.ndarray,
    primary_suffix: str = "",
    secondary_suffix: str = "",
    reference_suffixes: tuple[str, str] = ("", ""),
    rotation_suffix: str = "",
) -> list[str]:
    """Define a nonsingular two-coordinate chart for a linear-body direction."""

    pole_index, pole_sign, transverse = _linear_fragment_stereographic_axes(
        coords,
        atoms,
        anchor_atom,
        reference_atoms,
        reference_frame_atoms,
    )
    reference_primary, reference_secondary = reference_suffixes
    primary_tag = f"{fragment_id}{primary_suffix}"
    pair = f"{fragment_id}{reference_id}{rotation_suffix}"

    def reference_axis(index: int) -> tuple[str, str]:
        prefix = ("P", "Q", "S")[index]
        suffix = reference_primary if prefix == "P" else reference_secondary
        return prefix, f"{reference_id}{suffix}"

    def dot_line(name: str, index: int) -> str:
        prefix, tag = reference_axis(index)
        return (
            f"{name}{pair}(Inactive)="
            f"Px{primary_tag}*{prefix}x{tag}+"
            f"Py{primary_tag}*{prefix}y{tag}+"
            f"Pz{primary_tag}*{prefix}z{tag}"
        )

    pole_name = "Lp"
    first_name = "Lt1"
    second_name = "Lt2"
    sign_text = "+" if pole_sign > 0.0 else "-"
    return [
        f"RP{primary_tag}(Inactive)=SQRT((X({anchor_atom})-Cx{fragment_id})**2+"
        f"(Y({anchor_atom})-Cy{fragment_id})**2+(Z({anchor_atom})-Cz{fragment_id})**2)",
        f"Px{primary_tag}(Inactive)=[X({anchor_atom})-Cx{fragment_id}]/RP{primary_tag}",
        f"Py{primary_tag}(Inactive)=[Y({anchor_atom})-Cy{fragment_id}]/RP{primary_tag}",
        f"Pz{primary_tag}(Inactive)=[Z({anchor_atom})-Cz{fragment_id}]/RP{primary_tag}",
        dot_line(pole_name, pole_index),
        dot_line(first_name, transverse[0]),
        dot_line(second_name, transverse[1]),
        f"Ld{pair}(Inactive)=1{sign_text}{pole_name}{pair}",
        f"Ls1{pair}(Inactive)={first_name}{pair}/Ld{pair}",
        f"Ls2{pair}(Inactive)={second_name}{pair}/Ld{pair}",
    ]


def _gaussian_quaternion_lines(
    frag_id: str,
    ref_id: str,
    *,
    suffix: str = "",
    moving_suffixes: tuple[str, str] = ("", ""),
    reference_suffixes: tuple[str, str] = ("", ""),
    reference_rotation: np.ndarray,
) -> list[str]:
    """Lower frozen-reference FROT to its registered local ReadAllGIC chart.

    SMITH defines FROT as the local increment ``log(R R0**T)``, where ``R0``
    is the relative fragment rotation at the frozen Cartesian reference.  The
    Gaussian realization applies that same constant right transport and emits
    the stereographic quaternion chart ``4 K / (1 + Kw)``.  It has the same
    value, Jacobian and Hessian as the exponential map at the frozen reference,
    avoids the undefined ``theta K / |K|`` quotient at the chart origin, and
    retains a nonsingular radial derivative at a rotation of pi.  The frozen
    nonnegative scalar quaternion branch keeps its denominator in ``[1, 2]``.
    """
    lowering = gaussian_primitive_lowering("FROT")
    if lowering != "REFERENCE_RELATIVE_QUATERNION_STEREOGRAPHIC_4K_OVER_1_PLUS_KW":
        raise GICForgeContractError(f"unsupported Gaussian FROT lowering: {lowering}")
    pair = f"{frag_id}{ref_id}{suffix}"
    moving_primary, moving_secondary = moving_suffixes
    reference_primary, reference_secondary = reference_suffixes
    frozen = np.asarray(reference_rotation, dtype=float)
    if frozen.shape != (3, 3) or not np.all(np.isfinite(frozen)):
        raise GICForgeContractError("FROT reference rotation must be a finite 3x3 matrix")
    if not np.allclose(frozen @ frozen.T, np.eye(3), rtol=0.0, atol=1.0e-10):
        raise GICForgeContractError("FROT reference rotation must be orthogonal")
    if not np.isclose(np.linalg.det(frozen), 1.0, rtol=0.0, atol=1.0e-10):
        raise GICForgeContractError("FROT reference rotation must be proper")

    current_rows = []
    for left_axis, left_prefix in (("1", "P"), ("2", "Q"), ("3", "S")):
        for right_axis, right_prefix in (("1", "P"), ("2", "Q"), ("3", "S")):
            left_suffix = moving_primary if left_prefix == "P" else moving_secondary
            right_suffix = reference_primary if right_prefix == "P" else reference_secondary
            current_rows.append(
                f"C{left_axis}{right_axis}{pair}(Inactive)="
                f"{left_prefix}x{frag_id}{left_suffix}*{right_prefix}x{ref_id}{right_suffix}+"
                f"{left_prefix}y{frag_id}{left_suffix}*{right_prefix}y{ref_id}{right_suffix}+"
                f"{left_prefix}z{frag_id}{left_suffix}*{right_prefix}z{ref_id}{right_suffix}"
            )
    transported_rows = []
    for left in range(3):
        for right in range(3):
            terms: list[str] = []
            for axis in range(3):
                coefficient = float(frozen[right, axis])
                if abs(coefficient) <= 1.0e-14:
                    continue
                terms.append(
                    _gaussian_linear_term(
                        coefficient,
                        f"C{left + 1}{axis + 1}{pair}",
                        first=not terms,
                    )
                )
            transported_rows.append(f"R{left + 1}{right + 1}{pair}(Inactive)={''.join(terms)}")
    return [
        *current_rows,
        *transported_rows,
        f"Kw{pair}(Inactive)=0.5*SQRT(R11{pair}+R22{pair}+R33{pair}+1)",
        f"Kx{pair}(Inactive)=(R23{pair}-R32{pair})/(4*Kw{pair})",
        f"Ky{pair}(Inactive)=(R31{pair}-R13{pair})/(4*Kw{pair})",
        f"Kz{pair}(Inactive)=(R12{pair}-R21{pair})/(4*Kw{pair})",
        f"Ex{pair}(Inactive)=4*Kx{pair}/(1+Kw{pair})",
        f"Ey{pair}(Inactive)=4*Ky{pair}/(1+Kw{pair})",
        f"Ez{pair}(Inactive)=4*Kz{pair}/(1+Kw{pair})",
    ]




def _atom_list(atoms: tuple[int, ...]) -> str:
    return ",".join(str(atom) for atom in sorted(set(atoms)))
