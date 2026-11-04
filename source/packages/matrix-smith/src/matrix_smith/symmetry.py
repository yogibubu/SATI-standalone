"""Symmetry source-block helpers for frozen ORACLE GIC definitions."""

from __future__ import annotations

from dataclasses import dataclass

from .models import GICDefinition
from .policy import SPECIAL_REDUCTION_CLASS, primitive_reduction_class, primitive_symmetry_block


@dataclass(frozen=True)
class GICSymmetrySourceBlock:
    block: str
    family: str
    reduction_class: str
    gic_names: tuple[str, ...]

    @property
    def is_special(self) -> bool:
        return self.reduction_class == SPECIAL_REDUCTION_CLASS


def gic_symmetry_source_blocks(
    definition: GICDefinition,
) -> tuple[GICSymmetrySourceBlock, ...]:
    """Group final frozen GICs into homogeneous symmetry source blocks.

    Symmetry projection is allowed only inside these blocks. This prevents
    ordinary valence coordinates and protected fragment/center coordinates from
    being mixed just because they share an irrep.
    """
    grouped: dict[tuple[str, str], list[str]] = {}
    for gic in definition.gics:
        key = (primitive_symmetry_block(gic.family), gic.family)
        grouped.setdefault(key, []).append(gic.name)
    return tuple(
        GICSymmetrySourceBlock(
            block=block,
            family=family,
            reduction_class=primitive_reduction_class(family),
            gic_names=tuple(names),
        )
        for (block, family), names in grouped.items()
    )


def special_symmetry_source_blocks(
    definition: GICDefinition,
) -> tuple[GICSymmetrySourceBlock, ...]:
    return tuple(block for block in gic_symmetry_source_blocks(definition) if block.is_special)
