"""Runtime contracts shared by KEYMAKER and scientific backends."""

from __future__ import annotations

from collections.abc import Sequence


FROZEN_KEYMAKER_STAGES = (
    "SENTINEL-basin-definition",
    "ZAFF-fast-GA",
    "CYPHER-reduction-and-medoid-selection",
    "exact-equivalent-geometry-deduplication",
    "xTB-medoid-optimization",
    "ORACLE-GDV32-L0-single-point",
    "ORACLE-GDV32-L0-optimization-and-Hessian",
)


def validate_frozen_keymaker_stage_sequence(
    completed_stages: Sequence[str], *, allow_prefix: bool = True
) -> tuple[str, ...]:
    """Validate the ordered, non-skippable exploration/exploitation stages."""

    observed = tuple(str(stage).strip() for stage in completed_stages)
    if any(not stage for stage in observed):
        raise ValueError("KEYMAKER protocol stages must be non-empty")
    expected = FROZEN_KEYMAKER_STAGES
    if allow_prefix and observed == expected[: len(observed)]:
        return observed
    if observed == expected:
        return observed
    raise ValueError(
        "invalid frozen KEYMAKER stage sequence: "
        f"expected prefix of {expected!r}, received {observed!r}"
    )


def frozen_keymaker_stage_labels() -> dict[str, dict[str, str]]:
    """Stable bilingual labels for GUI/report consumers."""

    return {
        "SENTINEL-basin-definition": {"it": "Definizione bacini SENTINEL", "en": "SENTINEL basin definition"},
        "ZAFF-fast-GA": {"it": "Esplorazione GA ZAFF-fast", "en": "ZAFF-fast GA exploration"},
        "CYPHER-reduction-and-medoid-selection": {"it": "Riduzione e medoidi CYPHER", "en": "CYPHER reduction and medoid selection"},
        "exact-equivalent-geometry-deduplication": {"it": "Deduplicazione geometrica esatta", "en": "Exact-equivalent geometry deduplication"},
        "xTB-medoid-optimization": {"it": "Ottimizzazione xTB dei medoidi", "en": "xTB medoid optimization"},
        "ORACLE-GDV32-L0-single-point": {"it": "Single point L0 ORACLE/GDV32", "en": "ORACLE/GDV32 L0 single point"},
        "ORACLE-GDV32-L0-optimization-and-Hessian": {"it": "Ottimizzazione e Hessiano L0 ORACLE/GDV32", "en": "ORACLE/GDV32 L0 optimization and Hessian"},
    }
