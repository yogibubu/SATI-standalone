"""Deterministic reconstruction of the compiled ZAFF SONIC coordinate chart."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from matrix_smith import (
    augment_shared_proton_coordinates,
    build_gic_b_matrix,
    build_gic_definition_from_xyzin,
)


def build_zaff_runtime_sonic_definition(
    xyzin: Path | str,
    *,
    minimum_relative_singular_value: float = 1.0e-7,
    ring_puckering_model: str = "charm",
) -> tuple[Any, Mapping[str, Any]]:
    """Rebuild the frozen local chart with ARCHITECT's deterministic policy."""

    path = Path(xyzin)
    definition = build_gic_definition_from_xyzin(
        path,
        symmetrize=False,
        symmetry_group="C1",
        local_salc=True,
        ring_puckering_model=ring_puckering_model,
    )
    primary_b = np.asarray(build_gic_b_matrix(definition).rows, dtype=float)
    primary_singular = np.linalg.svd(primary_b, compute_uv=False)
    primary_relative = (
        0.0
        if not primary_singular.size or primary_singular[0] <= 0.0
        else float(primary_singular[-1] / primary_singular[0])
    )
    primary_rank = int(np.linalg.matrix_rank(primary_b, tol=1.0e-7))
    fallback = primary_rank < int(definition.target_rank) or primary_relative < float(
        minimum_relative_singular_value
    )
    common = {
        "policy": "LOCAL_SALC_WITH_RANK_STABLE_SVD_FALLBACK_V1",
        "local_salc_requested": True,
        "target_rank": int(definition.target_rank),
        "effective_point_group": "C1",
        "global_symmetrization": False,
        "ring_puckering_model": str(ring_puckering_model).upper(),
    }
    if not fallback:
        redundant = augment_shared_proton_coordinates(definition)
        return redundant, {
            **common,
            "local_salc_realized": True,
            "fallback_used": False,
            "rank": primary_rank,
            "relative_smallest_singular_value": primary_relative,
            "field_coordinate_count": len(redundant.gics),
            "field_redundancy": len(redundant.gics) - primary_rank,
            "normal_mode_chart": "DEFERRED_NONREDUNDANT_REDUCTION",
        }
    stable = build_gic_definition_from_xyzin(
        path,
        symmetrize=False,
        symmetry_group="C1",
        local_salc=False,
        ring_puckering_model=ring_puckering_model,
    )
    stable_b = np.asarray(build_gic_b_matrix(stable).rows, dtype=float)
    stable_singular = np.linalg.svd(stable_b, compute_uv=False)
    stable_relative = (
        0.0
        if not stable_singular.size or stable_singular[0] <= 0.0
        else float(stable_singular[-1] / stable_singular[0])
    )
    stable_rank = int(np.linalg.matrix_rank(stable_b, tol=1.0e-7))
    if stable_rank < int(stable.target_rank) or stable_relative < float(
        minimum_relative_singular_value
    ):
        raise ValueError(
            "local SONIC chart remains rank deficient after the rank-revealing fallback"
        )
    redundant = augment_shared_proton_coordinates(stable)
    return redundant, {
        **common,
        "local_salc_realized": False,
        "fallback_used": True,
        "fallback_reason": "LOCAL_SALC_B_MATRIX_ILL_CONDITIONED",
        "primary_rank": primary_rank,
        "rank": stable_rank,
        "primary_relative_smallest_singular_value": primary_relative,
        "relative_smallest_singular_value": stable_relative,
        "field_coordinate_count": len(redundant.gics),
        "field_redundancy": len(redundant.gics) - stable_rank,
        "normal_mode_chart": "DEFERRED_NONREDUNDANT_REDUCTION",
    }


__all__ = ["build_zaff_runtime_sonic_definition"]
