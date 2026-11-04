"""Explicit transition-state chart conditioning and dense reference audits.

Transition-state chart selection may require global rank/conditioning checks;
it is deliberately isolated from the minimum-chart sparse construction API.
The implementation remains in the compatibility module while callers use
this boundary, so the final move can preserve existing private integrations.
"""

from ..transition_state_conditioning import (
    _condition_transition_state_salc_alternatives,
    _normalized_gic_chart_condition,
    _select_conserved_reactive_exact_chart,
    _select_rank_aware_chart,
    _wilson_tangent_diagnostics,
    _with_prescribed_distance_only_primitives,
    _with_transition_state_reaction_distance_family,
)

__all__ = [
    "_condition_transition_state_salc_alternatives",
    "_normalized_gic_chart_condition",
    "_select_conserved_reactive_exact_chart",
    "_select_rank_aware_chart",
    "_wilson_tangent_diagnostics",
    "_with_prescribed_distance_only_primitives",
    "_with_transition_state_reaction_distance_family",
]
