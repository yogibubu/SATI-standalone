"""Declarative catalogue of endpoint-defined reaction-coordinate builders."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
from typing import Mapping


REACTION_ATLAS_SCHEMA = "matrix.reaction_atlas.v1"
TS_EXPLORATION_CAMPAIGN_SCHEMA = "matrix.ts_exploration_campaign.v1"
TS_CAMPAIGN_OUTCOME_TRANSITION_STATE = "TRANSITION_STATE"
TS_CAMPAIGN_OUTCOME_NO_TRANSITION_STATE = "NO_TRANSITION_STATE"
TS_CAMPAIGN_OUTCOMES = frozenset(
    {TS_CAMPAIGN_OUTCOME_TRANSITION_STATE, TS_CAMPAIGN_OUTCOME_NO_TRANSITION_STATE}
)


@dataclass(frozen=True)
class ReactionAtlasEntry:
    class_id: str
    title: str
    reverse_class_id: str
    mobile_min_atoms: int
    mobile_max_atoms: int
    broken_bonds_min: int
    broken_bonds_max: int
    formed_bonds_min: int
    formed_bonds_max: int
    cycle_rank_deltas: tuple[int, ...]
    coordinate_builder: str
    reactive_rank_policy: str
    stationary_policy: str
    symmetry_policy: str
    executable: bool


@lru_cache(maxsize=1)
def reaction_atlas() -> tuple[ReactionAtlasEntry, ...]:
    payload = json.loads(
        files("matrix_chem").joinpath("data", "reaction_atlas_v1.json").read_text(encoding="utf-8")
    )
    if payload.get("schema") != REACTION_ATLAS_SCHEMA:
        raise RuntimeError("invalid reaction-atlas schema")
    entries = tuple(
        ReactionAtlasEntry(**{**record, "cycle_rank_deltas": tuple(record["cycle_rank_deltas"])})
        for record in payload.get("entries", ())
    )
    identifiers = tuple(entry.class_id for entry in entries)
    if not entries or len(identifiers) != len(set(identifiers)):
        raise RuntimeError("reaction-atlas identifiers must be nonempty and unique")
    known = set(identifiers)
    by_identifier = {entry.class_id: entry for entry in entries}
    for entry in entries:
        if entry.reverse_class_id not in known:
            raise RuntimeError(f"unknown reverse reaction class: {entry.reverse_class_id}")
        if by_identifier[entry.reverse_class_id].reverse_class_id != entry.class_id:
            raise RuntimeError(f"nonreciprocal reverse reaction class: {entry.class_id}")
        if entry.mobile_min_atoms < 0 or (
            entry.mobile_max_atoms >= 0 and entry.mobile_max_atoms < entry.mobile_min_atoms
        ):
            raise RuntimeError(f"invalid mobile-fragment bounds for {entry.class_id}")
    return entries


def reaction_atlas_entry(class_id: str) -> ReactionAtlasEntry:
    normalized = str(class_id).strip().upper()
    for entry in reaction_atlas():
        if entry.class_id == normalized:
            return entry
    raise KeyError(f"unknown reaction class: {class_id}")


def validate_transition_state_exploration_campaign(
    payload: Mapping[str, object],
) -> None:
    """Validate generic campaign identity, outcomes, and reciprocal links."""

    if payload.get("schema") != TS_EXPLORATION_CAMPAIGN_SCHEMA:
        raise ValueError("unsupported transition-state exploration campaign schema")
    if not str(payload.get("campaign_id", "")).strip():
        raise ValueError("transition-state exploration campaign lacks an identifier")
    default_outcome = str(payload.get("default_expected_outcome", ""))
    if default_outcome not in TS_CAMPAIGN_OUTCOMES:
        raise ValueError("transition-state campaign lacks a valid default outcome")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("transition-state exploration campaign contains no cases")
    identifiers: list[str] = []
    reciprocal: dict[str, str] = {}
    known_classes = {entry.class_id for entry in reaction_atlas()}
    for raw in cases:
        if not isinstance(raw, Mapping):
            raise ValueError("invalid transition-state campaign case")
        case_id = str(raw.get("case_id", "")).strip()
        reaction_class = str(raw.get("reaction_class_id", "")).strip().upper()
        outcome = str(raw.get("expected_outcome", default_outcome))
        if not case_id:
            raise ValueError("transition-state campaign case lacks an identifier")
        if reaction_class not in known_classes:
            raise ValueError(f"unknown reaction class for campaign case {case_id}")
        if outcome not in TS_CAMPAIGN_OUTCOMES:
            raise ValueError(f"invalid expected outcome for campaign case {case_id}")
        identifiers.append(case_id)
        reverse = str(raw.get("reciprocal_case_id", "")).strip()
        if reverse:
            reciprocal[case_id] = reverse
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("transition-state campaign case identifiers must be unique")
    known_ids = set(identifiers)
    for case_id, reverse in reciprocal.items():
        if reverse not in known_ids:
            raise ValueError(f"campaign case {case_id} has an unknown reciprocal case")
        if reciprocal.get(reverse) != case_id:
            raise ValueError(f"campaign reciprocal link is not mutual for {case_id}")


__all__ = [
    "REACTION_ATLAS_SCHEMA",
    "TS_CAMPAIGN_OUTCOME_NO_TRANSITION_STATE",
    "TS_CAMPAIGN_OUTCOME_TRANSITION_STATE",
    "TS_CAMPAIGN_OUTCOMES",
    "TS_EXPLORATION_CAMPAIGN_SCHEMA",
    "ReactionAtlasEntry",
    "reaction_atlas",
    "reaction_atlas_entry",
    "validate_transition_state_exploration_campaign",
]
