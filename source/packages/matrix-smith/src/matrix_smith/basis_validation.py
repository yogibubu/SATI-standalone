from __future__ import annotations

import numpy as np

from .contracts import GICForgeContractError
from .coordinate_diagnostics import normalized_sonic_condition_diagnostics
from .evaluation import build_gic_b_matrix
from .models import GICDefinition
from .policy import (
    FRAGMENT_MODE_NONE,
    MAX_NORMALIZED_SONIC_CONDITION,
    MAX_UNPARTITIONED_SONIC_CONDITION,
)
from .symmetry_labels import is_total_symmetric_irrep


def validate_frozen_sonic_basis(
    definition: GICDefinition,
    *,
    rank_tolerance: float,
) -> dict[str, float | int]:
    """Reject a frozen SONIC chart that is rank-deficient or unusably conditioned."""

    matrix = build_gic_b_matrix(
        definition,
        coordinates_angstrom=definition.reference_coordinates_angstrom,
    )
    diagnostics = normalized_sonic_condition_diagnostics(
        np.asarray(matrix.rows, dtype=float),
        tolerance=rank_tolerance,
        maximum_condition_number=MAX_NORMALIZED_SONIC_CONDITION,
    )
    evaluated_rank = int(diagnostics["rank"])
    if evaluated_rank < definition.target_rank:
        raise GICForgeContractError(
            "frozen SONIC Jacobian is rank deficient at its reference geometry: "
            f"need {definition.target_rank}, evaluated rank {evaluated_rank}, "
            f"status {diagnostics['status']}"
        )
    condition_rows = np.asarray(matrix.rows, dtype=float)
    symmetry_preserving_transition_state = (
        "SCIENTIFIC_PATH TRANSITION_STATE" in definition.semantic_diagnostics
        and definition.symmetrize
        and definition.point_group.strip().upper() not in {"C1", "UNKNOWN"}
    )
    if symmetry_preserving_transition_state:
        active_indices = tuple(
            index
            for index, gic in enumerate(definition.gics)
            if is_total_symmetric_irrep(definition.point_group, gic.irrep)
        )
        if not active_indices:
            raise GICForgeContractError(
                "symmetry-preserving transition-state chart has no totally symmetric SONICs"
            )
        active_diagnostics = normalized_sonic_condition_diagnostics(
            condition_rows[np.asarray(active_indices, dtype=int)],
            tolerance=rank_tolerance,
            maximum_condition_number=MAX_NORMALIZED_SONIC_CONDITION,
        )
        if int(active_diagnostics["rank"]) != len(active_indices):
            raise GICForgeContractError(
                "totally symmetric transition-state SONIC block is rank deficient: "
                f"need {len(active_indices)}, evaluated rank "
                f"{int(active_diagnostics['rank'])}"
            )
        condition = float(active_diagnostics["condition_number"])
    else:
        condition = float(diagnostics["condition_number"])
    strict_condition_gate = (
        definition.fragment_mode != FRAGMENT_MODE_NONE
        or "SCIENTIFIC_PATH TRANSITION_STATE" in definition.semantic_diagnostics
    )
    maximum_condition = (
        MAX_NORMALIZED_SONIC_CONDITION
        if strict_condition_gate
        else MAX_UNPARTITIONED_SONIC_CONDITION
    )
    if not np.isfinite(condition) or condition > maximum_condition:
        raise GICForgeContractError(
            "frozen SONIC Jacobian exceeds the normalized condition gate at its "
            f"reference geometry: condition {condition:.12g}, maximum "
            f"{maximum_condition:.12g}"
        )
    return {"rank": evaluated_rank, "condition_number": condition}
