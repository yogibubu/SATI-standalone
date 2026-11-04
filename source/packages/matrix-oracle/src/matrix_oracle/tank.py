"""TANK: LCB26 geometry search and overlap realization.

TANK is the public name of the fragment-search capability.  The legacy
``matrix_fragments`` implementation remains an internal compatibility
backend for the existing XYZ container contract; it is not exposed as the
scientific owner of this workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matrix_chem import build_topology_objects, read_xyz, write_xyz
from matrix_chem.topology.elements import atomic_number
from matrix_fragments import OverlapAssemblyError, assemble_overlapping_fragments
from matrix_switch import (
    graph_from_topology,
    maximum_common_connected_subgraphs,
    parse_smiles,
    perceive_aromaticity,
)

from .lcb26 import query_lcb26, query_lcb26_l1_geometry


TANK_GEOMETRY_SCHEMA = "matrix.tank.geometry_proposal.v1"
TANK_GEOMETRY_POLICY = "L2_THEN_L1"
_L2_DATASETS = frozenset({"L2", "PL2", "PCS2"})


def propose_lcb26_geometry(
    smiles: str,
    output: Path | str,
    *,
    lcb26_root: Path | str,
    fragment_limit: int = 8,
) -> dict[str, Any]:
    """Propose and, when possible, realize a geometry from LCB26 donors.

    L2-family geometries are searched first.  L1 geometry-only records are
    considered only when the L2 candidate pool cannot realize a complete
    overlap assembly.  L1 records never contribute CM5 or Mayer values.
    """

    target = perceive_aromaticity(parse_smiles(str(smiles)))
    root = Path(lcb26_root).expanduser().resolve()
    l2_rows = [
        row for row in query_lcb26(root, limit=None)
        if str(row.get("dataset", "")).upper() in _L2_DATASETS
        and row.get("geometry_path")
    ]
    l2_ranked = _rank_l2_rows(root, target, l2_rows)
    l2_paths = [root / str(item["row"]["geometry_path"]) for item in l2_ranked]
    l1_ranked: list[dict[str, Any]] = []

    assembly = None
    selected_level = None
    selected_paths: list[Path] = []
    l2_error = ""
    if l2_paths:
        try:
            assembly = assemble_overlapping_fragments(
                str(smiles),
                l2_paths,
                max_fragment_atoms=60,
                min_overlap_atoms=3,
                relax=False,
                beam_width=64,
                extract_common_subgraphs=True,
            )
            selected_level = "L2"
            selected_paths = l2_paths
        except OverlapAssemblyError as exc:
            l2_error = str(exc)
    if assembly is None:
        l1_rows = list(query_lcb26_l1_geometry(root, limit=None))
        l1_ranked = _rank_l1_rows(root, target, l1_rows)
        combined_paths = l2_paths + [root / "l1_geometries" / str(item["row"]["file"]) for item in l1_ranked]
        if combined_paths:
            try:
                assembly = assemble_overlapping_fragments(
                    str(smiles),
                    combined_paths,
                    max_fragment_atoms=60,
                    min_overlap_atoms=3,
                    relax=False,
                    beam_width=64,
                    extract_common_subgraphs=True,
                )
                selected_level = "L2_PLUS_L1" if l2_paths else "L1"
                selected_paths = combined_paths
            except OverlapAssemblyError as exc:
                if not l2_error:
                    l2_error = str(exc)

    output_path = Path(output).expanduser().resolve()
    realized = None
    if assembly is not None:
        write_xyz(
            output_path,
            assembly.atoms,
            assembly.coordinates_angstrom,
            comment=f"TANK {selected_level} LCB26 geometry proposal",
        )
        realized = {
            "path": str(output_path),
            "level": selected_level,
            "covered_heavy_atoms": int(assembly.covered_heavy_atoms),
            "target_heavy_atoms": int(assembly.target_heavy_atoms),
            "max_overlap_rmsd_angstrom": float(assembly.max_overlap_rmsd_angstrom),
            "quality_warnings": list(assembly.quality_warnings),
            "selected_steps": [
                {
                    "source_path": str(step.source_path),
                    "target_atoms": list(step.target_atoms),
                    "overlap_atoms": list(step.overlap_atoms),
                    "new_atoms": list(step.new_atoms),
                    "overlap_rmsd_angstrom": float(step.overlap_rmsd_angstrom),
                }
                for step in assembly.steps
            ],
        }
    return {
        "schema": TANK_GEOMETRY_SCHEMA,
        "tool": "TANK",
        "owner": "ORACLE",
        "status": "PASS" if realized is not None else "NO_COMPLETE_ASSEMBLY",
        "policy": TANK_GEOMETRY_POLICY,
        "source_smiles": str(smiles),
        "l2_candidates": [_candidate_record(item, level="L2") for item in l2_ranked[:fragment_limit]],
        "l1_candidates": [_candidate_record(item, level="L1") for item in l1_ranked[:fragment_limit]],
        "l2_attempt": {
            "candidate_count": len(l2_ranked),
            "assembly_succeeded": selected_level == "L2",
            "failure": l2_error or None,
        },
        "l1_fallback": {
            "considered": selected_level in {"L2_PLUS_L1", "L1"},
            "candidate_count": len(l1_ranked),
            "electronic_properties_available": False,
        },
        "realized_geometry": realized,
        "selected_geometry_paths": [str(path) for path in selected_paths],
        "provenance": {
            "geometry_owner": "TANK/ORACLE",
            "l2_geometry_source": "LCB26 enriched geometry records",
            "l1_geometry_source": "LCB26 L1 geometry archive",
            "l1_electronic_data": "NOT_IMPORTED",
            "promotion_rule": "QM_L1_OR_PL1_REQUIRED_FOR_FINAL_GEOMETRY",
        },
    }


def _rank_l2_rows(root: Path, target: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for row in rows:
        try:
            query = perceive_aromaticity(parse_smiles(str(row.get("canonical_smiles", ""))))
            matches = maximum_common_connected_subgraphs(
                query, target, minimum_atoms=3, timeout_seconds=0.25, max_matches=8
            )
        except Exception:
            continue
        if matches:
            match = max(matches, key=lambda item: item.atom_count)
            ranked.append({"row": row, "match": match, "score": _match_score(query, target, match, row)})
    return sorted(ranked, key=lambda item: (-item["score"], str(item["row"].get("identifier", ""))))


def _rank_l1_rows(root: Path, target: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for row in rows:
        try:
            geometry = read_xyz(root / "l1_geometries" / str(row["file"]))
            numbers = tuple(int(atomic_number(atom) or 0) for atom in geometry.atoms)
            _continuous, graph, _rings, _synthons, _aromaticity = build_topology_objects(
                geometry.coordinates_angstrom, numbers
            )
            query = graph_from_topology(
                geometry.atoms,
                tuple(tuple(int(value) for value in bond.key) for bond in graph.bonds),
                source_smiles=str(row.get("identifier", "")),
            )
            query = perceive_aromaticity(query)
            matches = maximum_common_connected_subgraphs(
                query, target, minimum_atoms=3, timeout_seconds=0.15, max_matches=4
            )
        except Exception:
            continue
        if matches:
            match = max(matches, key=lambda item: item.atom_count)
            ranked.append({"row": row, "match": match, "score": _match_score(query, target, match, row)})
    return sorted(ranked, key=lambda item: (-item["score"], str(item["row"].get("identifier", ""))))


def _match_score(query: Any, target: Any, match: Any, row: dict[str, Any]) -> float:
    hetero = sum(query.atoms[index].symbol not in {"C", "H"} for index in match.source_atoms)
    aromatic = sum(
        bool(query.atoms[source].aromatic and target.atoms[dest].aromatic)
        for source, dest in zip(match.source_atoms, match.target_atoms, strict=True)
    )
    return 10.0 * match.atom_count + 7.0 * hetero + 3.0 * aromatic + 0.02 * float(row.get("atom_count", 0))


def _candidate_record(item: dict[str, Any], *, level: str) -> dict[str, Any]:
    row = item["row"]
    record = {
        "identifier": row.get("identifier"),
        "level": level,
        "dataset": row.get("dataset", "L1_GEOMETRY" if level == "L1" else None),
        "geometry_path": row.get("geometry_path", row.get("file")),
        "method": row.get("method"),
        "basis_family": row.get("basis_family"),
        "dispersion": row.get("dispersion"),
        "atom_count": row.get("atom_count"),
        "selection_score": float(item["score"]),
        "matched_atom_count": int(item["match"].atom_count),
        "matched_target_atoms": list(item["match"].target_atoms),
        "electronic_properties_available": level == "L2",
    }
    return record


__all__ = ["TANK_GEOMETRY_POLICY", "TANK_GEOMETRY_SCHEMA", "propose_lcb26_geometry"]
