"""Structured fallback provenance for frozen SONIC definitions."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from .models import FallbackEvent, GICDefinition


FALLBACK_LEDGER_SCHEMA = "matrix.smith.fallback_ledger.v1"


def build_fallback_ledger(definition: GICDefinition) -> tuple[FallbackEvent, ...]:
    """Collect only typed events emitted by scientific decision sites."""

    symmetry = definition.symmetry_diagnostics
    events = merge_fallback_events(
        definition.fallback_events,
        symmetry.fallback_events if symmetry is not None else (),
    )
    declared_sources = set(definition.fallback_diagnostics)
    typed_sources = {event.source for event in events}
    if declared_sources - typed_sources:
        raise ValueError("untyped fallback diagnostics are forbidden in frozen SONIC builds")
    return events


def make_fallback_event(
    *,
    stage: str,
    algorithm_id: str,
    trigger: str,
    domain: str = "GLOBAL",
    macrofamily: str = "UNSPECIFIED",
    rank_before: int | None = None,
    rank_after: int | None = None,
    condition_before: float | None = None,
    condition_after: float | None = None,
    source: str = "",
) -> FallbackEvent:
    """Create one deterministic event at the scientific decision site."""

    canonical = {
        "stage": str(stage).strip(),
        "algorithm_id": str(algorithm_id).strip().upper(),
        "trigger": " ".join(str(trigger).split()),
        "domain": str(domain).strip() or "GLOBAL",
        "macrofamily": str(macrofamily).strip() or "UNSPECIFIED",
        "rank_before": rank_before,
        "rank_after": rank_after,
        "condition_before": condition_before,
        "condition_after": condition_after,
        "source": " ".join(str(source).split()),
    }
    if not canonical["source"]:
        canonical["source"] = (
            f"{canonical['algorithm_id']} STAGE={canonical['stage']} "
            f"DOMAIN={canonical['domain']} FAMILY={canonical['macrofamily']} "
            f"TRIGGER={canonical['trigger']}"
        )
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    event = FallbackEvent(event_id=f"FB-{digest}", **canonical)
    _validate_fallback_events((event,))
    return event


def merge_fallback_events(
    *groups: tuple[FallbackEvent, ...],
) -> tuple[FallbackEvent, ...]:
    """Merge typed events deterministically and reject identifier collisions."""

    events: list[FallbackEvent] = []
    by_id: dict[str, FallbackEvent] = {}
    for event in (item for group in groups for item in group):
        existing = by_id.get(event.event_id)
        if existing is not None and existing != event:
            raise ValueError("fallback ledger identifier collision")
        if existing is None:
            by_id[event.event_id] = event
            events.append(event)
    result = tuple(events)
    _validate_fallback_events(result)
    return result


def fallback_event_to_dict(event: FallbackEvent) -> dict[str, object]:
    return {"schema": FALLBACK_LEDGER_SCHEMA, **asdict(event)}


def fallback_event_from_dict(record: object) -> FallbackEvent:
    if not isinstance(record, dict):
        raise ValueError("fallback ledger records must be objects")
    payload = dict(record)
    if payload.pop("schema", None) != FALLBACK_LEDGER_SCHEMA:
        raise ValueError("unsupported fallback ledger schema")
    if set(payload) != set(FallbackEvent.__dataclass_fields__):
        raise ValueError("fallback ledger fields do not match the typed contract")
    event = FallbackEvent(**payload)
    _validate_fallback_events((event,))
    return event


def fallback_ledger_section_lines(
    events: tuple[FallbackEvent, ...],
) -> tuple[str, ...]:
    """Serialize both the compatibility view and typed ledger subsections."""

    _validate_fallback_events(events)
    diagnostics = tuple(event.source for event in events) or ("NONE",)
    records = tuple(
        json.dumps(
            fallback_event_to_dict(event),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for event in events
    ) or ("NONE",)
    return (
        "[FALLBACK_DIAGNOSTICS]",
        *diagnostics,
        "[FALLBACK_LEDGER]",
        *records,
    )


def fallback_provenance_from_lines(
    diagnostic_lines: tuple[str, ...],
    ledger_lines: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[FallbackEvent, ...]]:
    """Read new typed ledgers and deterministic legacy diagnostic records."""

    diagnostics = tuple(
        line for line in diagnostic_lines if line.strip().upper() != "NONE"
    )
    try:
        events = tuple(
            fallback_event_from_dict(json.loads(line))
            for line in ledger_lines
            if line.strip().upper() != "NONE"
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("malformed fallback ledger") from exc
    if not events:
        from .fallback_legacy import fallback_event_from_legacy_diagnostic

        events = tuple(
            fallback_event_from_legacy_diagnostic(record)
            for record in dict.fromkeys(diagnostics)
        )
    _validate_fallback_events(events)
    derived = tuple(event.source for event in events)
    if diagnostics and diagnostics != derived:
        raise ValueError("fallback diagnostics disagree with the typed ledger")
    return derived, events


def _validate_fallback_events(events: tuple[FallbackEvent, ...]) -> None:
    identifiers = tuple(event.event_id for event in events)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("fallback ledger event identifiers must be unique")
    for event in events:
        if not event.event_id.startswith("FB-") or not event.algorithm_id or not event.trigger:
            raise ValueError("fallback ledger event is incomplete")


__all__ = [
    "FALLBACK_LEDGER_SCHEMA",
    "build_fallback_ledger",
    "fallback_event_from_dict",
    "fallback_event_to_dict",
    "fallback_ledger_section_lines",
    "fallback_provenance_from_lines",
    "make_fallback_event",
    "merge_fallback_events",
]
