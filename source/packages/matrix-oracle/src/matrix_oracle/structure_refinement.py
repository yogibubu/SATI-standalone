"""Keymaker-facing LCB26 structure refinement workflow.

The scientific operation remains owned by ORACLE: the frozen initial-structure
protocol performs the local LCB26 transfer and the final Cartesian/internal
closure.  This adapter adds auditable fragment provenance and spectroscopic
rotational constants for the structure handed to downstream tools.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from matrix_chem import Structure, build_topology_objects, principal_moments, read_xyz, rotational_constants_MHz
from matrix_chem.topology.elements import atomic_number
from matrix_chem.topology.torsion_classification import classify_torsion_ring_type
from matrix_switch import parse_smiles

from .initial_structure import (
    INITIAL_STRUCTURE_PROTOCOL_REVISION,
    InitialStructurePreparation,
    prepare_initial_structure,
)
from .lcb26 import query_lcb26


REFINE_STRUCTURE_SCHEMA = "matrix.oracle.refine_structure.v1"


class RefinedStructureError(ValueError):
    """Raised when a refined structure cannot enter downstream workflows."""


@dataclass(frozen=True)
class RefinedStructure:
    schema: str
    source: str
    source_kind: str
    output_xyz: str
    output_xyzin: str
    report: str
    correction_mode: str
    fragments: tuple[dict[str, Any], ...]
    rotational_constants_MHz: tuple[float, float, float]
    initial_structure: dict[str, Any]
    quality: dict[str, Any]
    validation: dict[str, Any]
    visualization_2d: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def refine_structure(
    source: Path | str,
    output: Path | str,
    *,
    lcb26_root: Path | str,
    declared_level: str = "AUTO",
    source_kind: str = "auto",
    fragment_limit: int = 5,
    allow_invalid: bool = False,
    strict: bool = False,
) -> RefinedStructure:
    """Refine a SMILES or XYZ using LCB26 donors and report provenance.

    Fragment records are selected by the same atom inventory and local
    chemistry available to the initial-structure protocol.  They are reported
    as ranked donors; the geometry itself is produced only by ORACLE's frozen
    protocol, so Keymaker never performs chemistry or geometry optimisation.
    """

    lcb26_root = os.path.realpath(Path(lcb26_root).expanduser())
    cache_key = _cache_key(source, lcb26_root, declared_level, source_kind, fragment_limit)
    cache_path = Path(output).expanduser().resolve().with_suffix(".refine_structure.json")
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("protocol", {}).get("cache_key") == cache_key:
                return _result_from_payload(cached)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    # The immutable SWITCH constitution and atom ordering must reach the
    # initial-structure protocol unchanged.  Overlap assembly is a separate
    # FRAGMENTS operation; using its Cartesian result here silently converted
    # a SMILES request into an unowned XYZ and disabled the constitutional
    # topology gate.
    assembly = None
    preparation: InitialStructurePreparation = prepare_initial_structure(
        source,
        output,
        lcb26_root=lcb26_root,
        declared_level=declared_level,
        source_kind=source_kind,
    )
    xyz_path = Path(preparation.output_xyz)
    geometry = read_xyz(xyz_path)
    validation = _validate_source(source, geometry, source_kind)
    validation.update({"charge": int(geometry.charge or 0), "multiplicity": int(geometry.multiplicity or 1), "open_shell": int(geometry.multiplicity or 1) != 1, "electronic_state_source": "geometry_metadata_or_default"})
    if not validation.get("valid", False) and not allow_invalid:
        raise RefinedStructureError("refined structure failed SMILES/XYZ validation; downstream use blocked")
    structure = Structure(
        list(geometry.atoms),
        [tuple(float(value) for value in row) for row in geometry.coordinates_angstrom],
    )
    constants = tuple(float(value) for value in rotational_constants_MHz(structure))
    moments = tuple(float(value) for value in principal_moments(structure))
    audit = json.loads(Path(preparation.report).read_text(encoding="utf-8"))
    donor_audit = audit.get("donor_audit", {})
    fragments = _trace_fragments(lcb26_root, donor_audit.get("donor_trace", ()), limit=fragment_limit)
    if not fragments:
        fragments = _rank_fragments(lcb26_root, geometry.atoms, limit=fragment_limit)
    validation["topology"] = {
        "bond_count": int(donor_audit.get("bond_count", 0)),
        "angle_count": int(donor_audit.get("angle_count", 0)),
        "perception_status": "PASS" if validation.get("valid") else "CHECK",
    }
    quality = {
        "donor_coverage": float(donor_audit.get("corrected_bond_count", 0) + donor_audit.get("corrected_angle_count", 0)) / max(1, int(donor_audit.get("bond_count", 0)) + int(donor_audit.get("angle_count", 0))),
        "closure_rms_residual": float(donor_audit.get("closure_rms_residual", 0.0)),
        "bond_residual_max": float(donor_audit.get("bond_residual_max", 0.0)),
        "angle_residual_max": float(donor_audit.get("angle_residual_max", 0.0)),
        "closure_converged": bool(donor_audit.get("closure_converged", False)),
        "overlap_assembly": bool(assembly is not None),
        "overlap_assembly_steps": int(len(assembly.steps)) if assembly is not None else 0,
        "overlap_assembly_max_rmsd_angstrom": float(assembly.max_overlap_rmsd_angstrom) if assembly is not None else None,
        **_soft_torsion_protocol(geometry),
        "fallback_status": "LOCAL_COMPATIBLE" if fragments and float(fragments[0].get("selection_score", 1.0)) == 0.0 else ("EXTRAPOLATION" if fragments else "NO_LCB26_DONOR"),
    }
    if strict and (not fragments or float(fragments[0].get("selection_score", 1.0)) != 0.0):
        raise RefinedStructureError("strict refinement requires at least one compatible local LCB26 donor")
    image_path = xyz_path.with_suffix(".refine_structure.svg")
    render_refined_structure_svg(geometry, image_path)
    report_path = xyz_path.with_suffix(".refine_structure.json")
    result = RefinedStructure(
        schema=REFINE_STRUCTURE_SCHEMA,
        source=str(source),
        source_kind=preparation.source_kind,
        output_xyz=preparation.output_xyz,
        output_xyzin=preparation.output_xyzin,
        report=str(report_path),
        correction_mode=preparation.correction_mode,
        fragments=tuple(fragments),
        rotational_constants_MHz=constants,
        initial_structure=preparation.to_dict(),
        quality={**quality, "cache_key": cache_key, "principal_moments_amu_angstrom2": moments, "rotor_type": _rotor_type(constants)},
        validation=validation,
        visualization_2d=str(image_path),
    )
    payload = result.to_dict()
    payload["protocol"] = {
        "geometry_owner": "ORACLE",
        "fragment_library": "LCB26",
        "fragment_selection": "audited_initial_structure_local_donor_trace",
        "conformation_protocol": "classify_soft_torsions_by_central_Mayer_order_then_relax_with_ZAFF-fast",
        "rotational_constants": "final_refined_cartesian_geometry",
        "cache_key": cache_key,
        "fallback_policy": "explicit_report_only_when_no_compatible_local_donor_exists",
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _result_from_payload(payload: dict[str, Any]) -> RefinedStructure:
    return RefinedStructure(
        schema=str(payload["schema"]), source=str(payload["source"]), source_kind=str(payload["source_kind"]),
        output_xyz=str(payload["output_xyz"]), output_xyzin=str(payload["output_xyzin"]), report=str(payload["report"]),
        correction_mode=str(payload.get("correction_mode", "")), fragments=tuple(payload.get("fragments", ())),
        rotational_constants_MHz=tuple(float(value) for value in payload.get("rotational_constants_MHz", ())),
        initial_structure=dict(payload.get("initial_structure", {})), quality=dict(payload.get("quality", {})),
        validation=dict(payload.get("validation", {})), visualization_2d=str(payload.get("visualization_2d", "")),
    )


def complete_refined_structure(
    result: RefinedStructure,
    initial_zaff0: Any,
    *,
    charge: int = 0,
    multiplicity: int = 1,
) -> RefinedStructure:
    """Refresh ORACLE's presentation record after ARCHITECT completes ZAFF0.

    The scientific artifacts are accepted from the owner workflow as an
    immutable manifest.  ORACLE only recomputes geometry-dependent validation,
    rotational constants and the audit view for the final Cartesian structure.
    """

    build = (
        initial_zaff0.to_dict()
        if hasattr(initial_zaff0, "to_dict")
        else dict(initial_zaff0)
    )
    if not bool(build.get("complete", False)):
        raise RefinedStructureError("ARCHITECT did not return a complete ZAFF0 build")
    geometry_path = Path(result.output_xyz).expanduser().resolve()
    if Path(str(build.get("geometry_xyz", ""))).expanduser().resolve() != geometry_path:
        raise RefinedStructureError("ARCHITECT ZAFF0 geometry does not match the refined structure")
    force_field = Path(str(build.get("force_field", ""))).expanduser().resolve()
    manifest_path = Path(str(build.get("manifest", ""))).expanduser().resolve()
    if not geometry_path.is_file() or not force_field.is_file() or not manifest_path.is_file():
        raise RefinedStructureError("the completed ZAFF0 artifact set is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    geometry = read_xyz(geometry_path)
    validation = _validate_source(result.source, geometry, result.source_kind)
    validation.update(
        {
            "charge": int(charge),
            "multiplicity": int(multiplicity),
            "open_shell": int(multiplicity) != 1,
            "electronic_state_source": "explicit_complete_refinement_request",
        }
    )
    numbers = tuple(int(atomic_number(atom)) for atom in geometry.atoms)
    _continuous, graph, _rings, _synthons, _aromaticity = build_topology_objects(
        geometry.coordinates_angstrom, numbers
    )
    validation["topology"] = {
        "bond_count": len(graph.bonds),
        "perception_status": "PASS" if validation.get("valid") else "CHECK",
        "geometry_stage": "FINAL_POST_ZAFF_FAST",
    }
    structure = Structure(
        list(geometry.atoms),
        [tuple(float(value) for value in row) for row in geometry.coordinates_angstrom],
    )
    constants = tuple(float(value) for value in rotational_constants_MHz(structure))
    moments = tuple(float(value) for value in principal_moments(structure))
    relaxation = dict(manifest.get("zaff_fast_relaxation", {}))
    quality = {
        **result.quality,
        "principal_moments_amu_angstrom2": moments,
        "rotor_type": _rotor_type(constants),
        "initial_zaff0": build,
        "initial_zaff0_complete": True,
        "population_mode": str(build.get("population_mode", "")),
        "zaff0_force_field": str(force_field),
        "zaff0_manifest": str(manifest_path),
        "zaff_fast_relaxation": relaxation,
        "soft_torsion_count": int(relaxation.get("active_torsion_count", 0)),
        "soft_torsions": list(relaxation.get("active_torsions", ())),
        "soft_torsion_order_source": "ORACLE_L0_MAYER_WITH_SMITH_SONIC_FAMILY",
        "final_geometry_sha256": str(manifest.get("geometry_sha256", "")),
        "whole_molecule_heavy_atom_limit": int(
            manifest.get("protocol", {}).get("whole_molecule_heavy_atom_limit", 60)
        ),
    }
    image_path = geometry_path.with_suffix(".refine_structure.svg")
    render_refined_structure_svg(geometry, image_path)
    completed = replace(
        result,
        rotational_constants_MHz=constants,
        quality=quality,
        validation=validation,
        visualization_2d=str(image_path),
    )
    report_path = Path(completed.report).expanduser().resolve()
    try:
        old_payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        old_payload = {}
    payload = completed.to_dict()
    payload["protocol"] = {
        **dict(old_payload.get("protocol", {})),
        "complete_workflow": "ORACLE_LCB26_TO_ARCHITECT_ZAFF0",
        "population_owner": "ORACLE",
        "sonic_owner": "SMITH",
        "torsional_optimizer_owner": "LINK",
        "force_field_owner": "ARCHITECT_ZAFF",
        "keymaker_role": "ORCHESTRATION_ONLY",
        "whole_molecule_policy": "L0_UP_TO_60_HEAVY_ATOMS_ON_EVERY_ARCHITECTURE",
    }
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return completed


def _cache_key(source, root, level, kind, limit):
    source_path = Path(str(source)).expanduser()
    source_token = source_path.read_bytes() if source_path.is_file() else str(source).encode()
    index = Path(root).expanduser() / "enriched" / "index.json"
    index_token = index.read_bytes() if index.is_file() else b"MISSING_LCB26_INDEX"
    return hashlib.sha256(
        source_token
        + index_token
        + f"|{level}|{kind}|{limit}|{INITIAL_STRUCTURE_PROTOCOL_REVISION}".encode()
    ).hexdigest()


def _validate_source(source, geometry, source_kind):
    coords = geometry.coordinates_angstrom
    valid = bool(geometry.atoms) and all(math.isfinite(float(value)) for row in coords for value in row)
    payload = {"valid": valid, "atom_count": len(geometry.atoms), "source_kind": source_kind, "unique_symbols": sorted(set(geometry.atoms))}
    path = Path(str(source)).expanduser()
    if source_kind in {"xyz", "geometry", "enriched_xyz"} or (source_kind == "auto" and path.is_file()):
        original = read_xyz(path)
        payload["atom_count_matches_input"] = len(original.atoms) == len(geometry.atoms)
        payload["atom_order_matches_input"] = tuple(original.atoms) == tuple(geometry.atoms)
        payload["valid"] = payload["valid"] and payload["atom_count_matches_input"] and payload["atom_order_matches_input"]
    else:
        parse_smiles(str(source))
        payload["smiles_parse"] = "PASS"
    return payload


def _rotor_type(constants):
    a, b, c = constants
    if c <= 1.0e-6:
        return "linear"
    if abs(a - b) / max(a, 1.0) < 1.0e-3 and abs(b - c) / max(b, 1.0) < 1.0e-3:
        return "spherical"
    if abs(b - c) / max(b, 1.0) < 1.0e-3:
        return "symmetric_prolate"
    if abs(a - b) / max(a, 1.0) < 1.0e-3:
        return "symmetric_oblate"
    return "asymmetric_prolate" if (2.0 * b - a - c) / max(a - c, 1.0e-12) < 0 else "asymmetric_oblate"


def _soft_torsion_protocol(geometry, *, central_bond_order_threshold: float = 1.25) -> dict[str, Any]:
    """Describe the primitive torsions that ZAFF-fast must relax first.

    The central-bond order is Mayer when supplied by an electronic record; for
    an uncharged SMILES/XYZ seed the topology fallback is explicitly marked so
    that a later L0/LCB26 population pass can replace it without ambiguity.
    """
    numbers = tuple(int(atomic_number(atom)) for atom in geometry.atoms)
    _continuous, graph, rings, _synthons, _aromaticity = build_topology_objects(
        geometry.coordinates_angstrom, numbers
    )
    adjacency = {index: set() for index in range(len(numbers))}
    for left, right in graph.bonds:
        adjacency[left].add(right); adjacency[right].add(left)
    torsions = []
    seen = set()
    for center_left, center_right in graph.bonds:
        order = 1.0  # replaced by Mayer order when the population record exists
        if order >= central_bond_order_threshold or tuple(sorted((center_left, center_right))) in rings.bond_to_rings:
            continue
        for left in sorted(adjacency[center_left] - {center_right}):
            for right in sorted(adjacency[center_right] - {center_left}):
                key = min((left, center_left, center_right, right), (right, center_right, center_left, left))
                if key in seen or len({left, center_left, center_right, right}) < 4:
                    continue
                seen.add(key)
                torsions.append({"atoms": list(key), "central_bond": [center_left, center_right], "central_bond_order": order, "ring_type": classify_torsion_ring_type(key, rings.bond_to_rings), "soft": True})
    return {
        "soft_torsion_count": len(torsions),
        "soft_torsions": torsions,
        "soft_torsion_central_bond_order_threshold": float(central_bond_order_threshold),
        "soft_torsion_order_source": "MAYER_WHEN_AVAILABLE_GEOMETRY_TOPOLOGY_FALLBACK",
        "soft_torsion_relaxation": "ZAFF-fast",
        "conformational_starting_point": "LCB26_ASSEMBLY_THEN_ZAFF_FAST_SOFT_TORSIONS",
    }


def render_refined_structure_svg(geometry, output: Path | str) -> Path:
    """Render a dependency-free 2-D audit view of the refined structure."""
    target = Path(output).expanduser().resolve(); target.parent.mkdir(parents=True, exist_ok=True)
    coords = geometry.coordinates_angstrom
    scale = 55.0; margin = 45.0
    xs = [float(row[0]) for row in coords]; ys = [float(row[1]) for row in coords]
    xmin, xmax = min(xs, default=0.0), max(xs, default=1.0); ymin, ymax = min(ys, default=0.0), max(ys, default=1.0)
    width = max(320.0, (xmax - xmin) * scale + 2 * margin); height = max(260.0, (ymax - ymin) * scale + 2 * margin)
    numbers = tuple(int(atomic_number(atom)) for atom in geometry.atoms)
    _continuous, graph, _rings, _synthons, _aromaticity = build_topology_objects(coords, numbers)
    lines = []
    for left, right in graph.bonds:
        x1, y1 = margin + (xs[left] - xmin) * scale, height - margin - (ys[left] - ymin) * scale
        x2, y2 = margin + (xs[right] - xmin) * scale, height - margin - (ys[right] - ymin) * scale
        lines.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#8794a8" stroke-width="2"/>')
    for index, atom in enumerate(geometry.atoms):
        x, y = margin + (xs[index] - xmin) * scale, height - margin - (ys[index] - ymin) * scale
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="14" fill="#16243a"/><text x="{x:.2f}" y="{y + 5:.2f}" text-anchor="middle" fill="white" font-size="12">{atom}</text>')
    target.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}"><rect width="100%" height="100%" fill="#f7f9fc"/>{"".join(lines)}</svg>\n', encoding="utf-8")
    return target


def _rank_fragments(root: Path | str, atoms: tuple[str, ...], *, limit: int) -> list[dict[str, Any]]:
    target: dict[str, int] = {}
    for atom in atoms:
        target[str(atom)] = target.get(str(atom), 0) + 1
    rows = query_lcb26(Path(root), limit=None)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        counts = {str(k): int(v) for k, v in row.get("element_counts", {}).items()}
        missing = sum(max(0, value - counts.get(key, 0)) for key, value in target.items())
        excess = sum(max(0, value - target.get(key, 0)) for key, value in counts.items())
        inventory = missing + excess
        score = float(missing * 10 + excess + inventory * 0.01)
        ranked.append((score, row))
    ranked.sort(key=lambda item: (item[0], str(item[1].get("identifier", ""))))
    selected = []
    for score, row in ranked[: max(0, int(limit))]:
        selected.append({
            "identifier": row.get("identifier"),
            "name": row.get("name"),
            "canonical_smiles": row.get("canonical_smiles"),
            "dataset": row.get("dataset"),
            "geometry_path": row.get("geometry_path"),
            "element_counts": row.get("element_counts", {}),
            "selection_score": score,
            "selection_status": "LOCAL_COMPATIBLE" if score == 0.0 else "EXTRAPOLATION",
        })
    return selected






def _trace_fragments(root: Path | str, trace: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    usage: dict[str, dict[str, Any]] = {}
    for item in trace:
        role = str(item.get("role", ""))
        for donor in item.get("donors", ()):
            identifier = (
                donor.get("identifier") if isinstance(donor, dict) else donor
            )
            if not identifier:
                continue
            entry = usage.setdefault(str(identifier), {"roles": set(), "uses": 0})
            entry["roles"].add(role); entry["uses"] += 1
    if not usage:
        return []
    rows = {str(row.get("identifier")): row for row in query_lcb26(Path(root), limit=None)}
    selected = []
    for identifier, metadata in sorted(usage.items(), key=lambda item: (-int(item[1]["uses"]), item[0]))[: max(0, int(limit))]:
        row = rows.get(identifier)
        if row is None:
            continue
        selected.append({
            "identifier": identifier, "name": row.get("name"), "canonical_smiles": row.get("canonical_smiles"),
            "dataset": row.get("dataset"), "geometry_path": row.get("geometry_path"),
            "element_counts": row.get("element_counts", {}), "uses": int(metadata["uses"]),
            "roles": sorted(metadata["roles"]), "selection_status": "ACTUAL_LOCAL_DONOR", "selection_score": 0.0,
        })
    return selected


__all__ = [
    "REFINE_STRUCTURE_SCHEMA",
    "RefinedStructure",
    "RefinedStructureError",
    "complete_refined_structure",
    "refine_structure",
    "render_refined_structure_svg",
]
