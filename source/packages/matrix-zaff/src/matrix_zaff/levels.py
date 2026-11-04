"""Permanent, read-only definition of the ZAFF0/ZAFF1/ZAFF2/ZAFF3 hierarchy."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
import json
from typing import Any

from .compatibility import normalize_legacy_zaff_payload


ZAFF_LEVEL_MANIFEST_SCHEMA = "matrix.zaff.level_manifest.v1"


@lru_cache(maxsize=1)
def zaff_level_manifest() -> dict[str, Any]:
    path = files("matrix_zaff").joinpath("data/zaff_levels.json")
    payload = normalize_legacy_zaff_payload(
        json.loads(path.read_text(encoding="utf-8"))
    )
    if payload.get("schema") != ZAFF_LEVEL_MANIFEST_SCHEMA:
        raise ValueError("unsupported ZAFF level manifest")
    levels = payload.get("levels")
    if not isinstance(levels, list) or [item.get("id") for item in levels] != [
        "ZAFF0", "ZAFF1", "ZAFF2", "ZAFF3"
    ]:
        raise ValueError("ZAFF hierarchy must define ZAFF0, ZAFF1, ZAFF2 and ZAFF3 in order")
    if [item.get("extends") for item in levels] != [None, "ZAFF0", "ZAFF1", "ZAFF2"]:
        raise ValueError("ZAFF level inheritance is inconsistent")
    typing = payload.get("synthon_typing", {})
    fields = tuple(typing.get("descriptor_fields", ()))
    thresholds = tuple(float(value) for value in typing.get("component_thresholds", ()))
    if len(fields) != 11 or len(thresholds) != len(fields) or any(value <= 0.0 for value in thresholds):
        raise ValueError("ZAFF synthon thresholds are incomplete")
    return payload


def zaff_level(identifier: str) -> dict[str, Any]:
    requested = str(identifier).strip().casefold()
    matches = [
        item
        for item in zaff_level_manifest()["levels"]
        if requested
        in {
            str(item["id"]).casefold(),
            *(str(alias).casefold() for alias in item.get("aliases", ())),
        }
    ]
    if len(matches) != 1:
        raise KeyError(f"unknown or ambiguous ZAFF level: {identifier!r}")
    return dict(matches[0])


def zaff_synthon_type_thresholds() -> tuple[float, ...]:
    return tuple(
        float(value)
        for value in zaff_level_manifest()["synthon_typing"]["component_thresholds"]
    )


__all__ = [
    "ZAFF_LEVEL_MANIFEST_SCHEMA",
    "zaff_level",
    "zaff_level_manifest",
    "zaff_synthon_type_thresholds",
]
