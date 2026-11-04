"""Shared coordinate-component invariants for frozen SMITH charts."""

from __future__ import annotations

import numpy as np

from matrix_chem import CoordinateComponent, validate_coordinate_component_transform

from .contracts import GICForgeContractError
from .models import GICDefinition


def validate_indivisible_gic_components(definition: GICDefinition) -> None:
    """Require every active multicomponent primitive to retain its complete span."""

    primitive_index = {
        primitive.identifier: index
        for index, primitive in enumerate(definition.primitives)
    }
    transform = np.zeros((len(definition.primitives), len(definition.gics)))
    for column, gic in enumerate(definition.gics):
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        for primitive_id, coefficient in coefficients:
            try:
                row = primitive_index[primitive_id]
            except KeyError as exc:
                raise GICForgeContractError(
                    f"frozen GIC {gic.identifier} references unknown primitive {primitive_id}"
                ) from exc
            transform[row, column] += float(coefficient)
    components = tuple(
        CoordinateComponent(
            operator=primitive.function,
            atoms=primitive.atoms,
            mode=primitive.mode,
            ref_atoms=primitive.ref_atoms,
            context=(primitive.family, *primitive.refs),
        )
        for primitive in definition.primitives
    )
    try:
        validate_coordinate_component_transform(components, transform)
    except ValueError as exc:
        raise GICForgeContractError(
            f"frozen SONIC component invariant failed: {exc}"
        ) from exc


__all__ = ["validate_indivisible_gic_components"]
