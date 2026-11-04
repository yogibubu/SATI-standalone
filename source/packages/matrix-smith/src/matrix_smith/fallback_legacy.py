"""Read-only migration of pre-ledger fallback diagnostics."""

from __future__ import annotations

import re

from .fallback_ledger import make_fallback_event
from .models import FallbackEvent


def fallback_event_from_legacy_diagnostic(source: str) -> FallbackEvent:
    """Translate one legacy text record; never used by new construction."""

    text = " ".join(str(source).split())
    algorithm = text.split(maxsplit=1)[0].upper() if text else "UNKNOWN"
    if algorithm == "DECLARED_FALLBACK":
        match = re.search(r"\bFALLBACK=([^ ]+)", text, flags=re.IGNORECASE)
        algorithm = match.group(1).upper() if match is not None else algorithm
    rank_match = re.search(r"\bRANK=(\d+)/(\d+)", text, flags=re.IGNORECASE)
    return make_fallback_event(
        stage="LEGACY_IMPORT",
        algorithm_id=algorithm,
        trigger=text or "LEGACY_DECLARED",
        domain=_tag(text, "DOMAIN") or _tag(text, "BLOCK") or "GLOBAL",
        macrofamily=_tag(text, "FAMILY") or "UNSPECIFIED",
        rank_before=int(rank_match.group(1)) if rank_match is not None else None,
        rank_after=int(rank_match.group(2)) if rank_match is not None else None,
        condition_before=_float_tag(text, "DIRECT"),
        condition_after=_float_tag(text, "SALC"),
        source=text,
    )


def _tag(text: str, name: str) -> str:
    match = re.search(rf"\b{name}=([^ ]+)", text, flags=re.IGNORECASE)
    return "" if match is None else match.group(1)


def _float_tag(text: str, name: str) -> float | None:
    value = _tag(text, name)
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if parsed == parsed and abs(parsed) != float("inf") else None


__all__ = ["fallback_event_from_legacy_diagnostic"]
