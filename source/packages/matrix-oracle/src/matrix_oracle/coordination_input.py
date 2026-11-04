"""Coordinate-free input contract for transition-metal assembly."""

from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
import json
from pathlib import Path
from typing import Any, Mapping

from matrix_chem import DEFAULT_SPECIAL_EDGE_EFFECTIVE_ORDER
from matrix_fragments import HapticInteractionRequest


ORACLE_COORDINATION_INPUT_SCHEMA = "matrix.oracle.coordination-input.v1"
_REPRESENTATIONS = {"smiles", "cxsmiles", "inchi", "template", "fragment"}
_COMPONENT_KINDS = {"metal", "ligand", "counterion", "solvent"}
_MODES = {"sigma", "pi", "kappa", "eta", "mu", "metal-metal"}
_GEOMETRIES = {
    "unspecified",
    "linear",
    "trigonal-planar",
    "tetrahedral",
    "square-planar",
    "trigonal-bipyramidal",
    "square-pyramidal",
    "octahedral",
    "pentagonal-bipyramidal",
    "square-antiprismatic",
}


def coordination_input_json_schema() -> dict[str, Any]:
    """Return the packaged JSON Schema for the coordinate-free contract."""

    source = files("matrix_oracle").joinpath("data/coordination_input.schema.json")
    return json.loads(source.read_text(encoding="utf-8"))


def load_coordination_input(path: Path | str) -> dict[str, Any]:
    """Load and validate one coordinate-free coordination specification."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_coordination_input(payload)


def validate_coordination_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate references and chemical invariants not expressible in JSON Schema."""

    data = deepcopy(dict(payload))
    if data.get("schema") != ORACLE_COORDINATION_INPUT_SCHEMA:
        raise ValueError(
            f"coordination input requires schema={ORACLE_COORDINATION_INPUT_SCHEMA}"
        )
    if not str(data.get("system_id", "")).strip():
        raise ValueError("coordination input requires a nonempty system_id")
    multiplicities = data.get("multiplicity_hypotheses")
    if not isinstance(multiplicities, list) or not multiplicities:
        raise ValueError("multiplicity_hypotheses must be a nonempty list")
    if any(int(value) < 1 for value in multiplicities):
        raise ValueError("multiplicity hypotheses must be positive integers")
    if len({int(value) for value in multiplicities}) != len(multiplicities):
        raise ValueError("multiplicity hypotheses must be unique")

    components = data.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("coordination input requires at least one component")
    component_by_id: dict[str, dict[str, Any]] = {}
    for component in components:
        identifier = str(component.get("component_id", "")).strip()
        if not identifier or identifier in component_by_id:
            raise ValueError("component_id values must be nonempty and unique")
        kind = str(component.get("kind", ""))
        if kind not in _COMPONENT_KINDS:
            raise ValueError(f"unsupported component kind for {identifier}: {kind}")
        representation = component.get("representation", {})
        representation_format = str(representation.get("format", ""))
        if representation_format not in _REPRESENTATIONS:
            raise ValueError(
                f"unsupported representation for {identifier}: "
                f"{representation_format}"
            )
        if not str(representation.get("value", "")).strip():
            raise ValueError(f"component {identifier} has an empty representation")
        component_by_id[identifier] = component

    centers = data.get("metal_centers")
    if not isinstance(centers, list) or not centers:
        raise ValueError("coordination input requires at least one metal center")
    center_by_id: dict[str, dict[str, Any]] = {}
    for center in centers:
        identifier = str(center.get("center_id", "")).strip()
        if not identifier or identifier in center_by_id:
            raise ValueError("center_id values must be nonempty and unique")
        component_id = str(center.get("component_id", ""))
        component = component_by_id.get(component_id)
        if component is None or component["kind"] != "metal":
            raise ValueError(
                f"metal center {identifier} must reference a metal component"
            )
        geometries = center.get("coordination_geometry_hypotheses", [])
        if not geometries or any(str(value) not in _GEOMETRIES for value in geometries):
            raise ValueError(
                f"metal center {identifier} has invalid geometry hypotheses"
            )
        oxidation = center.get("oxidation_state_hypotheses", [])
        if not oxidation or any(not isinstance(value, int) for value in oxidation):
            raise ValueError(
                f"metal center {identifier} requires integer oxidation hypotheses"
            )
        center_by_id[identifier] = center

    interactions = data.get("coordination_interactions")
    if not isinstance(interactions, list) or not interactions:
        raise ValueError("at least one coordination interaction is required")
    interaction_ids: set[str] = set()
    for interaction in interactions:
        identifier = str(interaction.get("interaction_id", "")).strip()
        if not identifier or identifier in interaction_ids:
            raise ValueError("interaction_id values must be nonempty and unique")
        interaction_ids.add(identifier)
        centers_used = [str(value) for value in interaction.get("center_ids", [])]
        if not centers_used or any(value not in center_by_id for value in centers_used):
            raise ValueError(f"interaction {identifier} references an unknown center")
        mode = str(interaction.get("mode", ""))
        if mode not in _MODES:
            raise ValueError(f"interaction {identifier} has unsupported mode {mode}")
        ligand_id = interaction.get("ligand_component_id")
        donor_atoms = [int(value) for value in interaction.get("donor_atoms", [])]
        if mode == "metal-metal":
            if len(centers_used) != 2 or ligand_id is not None or donor_atoms:
                raise ValueError(
                    f"metal-metal interaction {identifier} requires two centers only"
                )
            continue
        ligand = component_by_id.get(str(ligand_id))
        if ligand is None or ligand["kind"] not in {"ligand", "counterion"}:
            raise ValueError(
                f"interaction {identifier} must reference a ligand or counterion"
            )
        if not donor_atoms or min(donor_atoms) < 1 or len(set(donor_atoms)) != len(
            donor_atoms
        ):
            raise ValueError(
                f"interaction {identifier} requires unique one-based donor atoms"
            )
        denticity = int(interaction.get("denticity", len(donor_atoms)))
        hapticity = int(interaction.get("hapticity", 1))
        if mode == "eta":
            if hapticity != len(donor_atoms) or hapticity < 2:
                raise ValueError(
                    f"eta interaction {identifier} requires hapticity=donor count>=2"
                )
        elif denticity != len(donor_atoms):
            raise ValueError(
                f"interaction {identifier} requires denticity=donor count"
            )

    assembly = data.get("assembly", {})
    if not isinstance(assembly, Mapping):
        raise ValueError("assembly must be an object")
    if int(assembly.get("maximum_candidates", 1)) < 1:
        raise ValueError("assembly.maximum_candidates must be positive")
    return data


def materialize_haptic_interaction_requests(
    payload: Mapping[str, Any],
    *,
    center_atom_by_id: Mapping[str, int],
    component_atom_maps: Mapping[str, Mapping[int, int]],
) -> tuple[HapticInteractionRequest, ...]:
    """Map validated eta interactions to global atom-index center requests.

    ORACLE owns the declared coordination chemistry and atom mapping;
    matrix-fragments owns center materialization.  This bridge supports any
    hapticity, including eta3 and eta5, without generating metal--donor bonds.
    """

    data = validate_coordination_input(payload)
    requests: list[HapticInteractionRequest] = []
    for interaction in data["coordination_interactions"]:
        if str(interaction.get("mode")) != "eta":
            continue
        component_id = str(interaction["ligand_component_id"])
        atom_map = component_atom_maps.get(component_id)
        if atom_map is None:
            raise ValueError(f"missing assembled atom map for component {component_id}")
        try:
            donors = tuple(int(atom_map[int(atom)]) for atom in interaction["donor_atoms"])
        except KeyError as exc:
            raise ValueError(
                f"eta interaction {interaction['interaction_id']} has an unmapped donor atom"
            ) from exc
        order = float(
            interaction.get("bond_order_hint", DEFAULT_SPECIAL_EDGE_EFFECTIVE_ORDER)
        )
        if order <= 0.0:
            order = DEFAULT_SPECIAL_EDGE_EFFECTIVE_ORDER
        for center_id in interaction["center_ids"]:
            try:
                metal_atom = int(center_atom_by_id[str(center_id)])
            except KeyError as exc:
                raise ValueError(f"missing assembled atom for metal center {center_id}") from exc
            requests.append(
                HapticInteractionRequest(
                    metal_atom=metal_atom,
                    donor_atoms=donors,
                    effective_order=order,
                    source=f"ORACLE_COORDINATION_INPUT:{interaction['interaction_id']}",
                )
            )
    return tuple(requests)
