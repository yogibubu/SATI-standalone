"""LINK objective joining direct bonded SONIC and Cartesian nonbonded ZAFF.

This opt-in objective consumes already typed local terms and a complete
SMITH chart. It does not guess force-field topology or change the default
Cartesian calculator protocol used by other LINK methods.
"""

from dataclasses import dataclass
from copy import deepcopy

import numpy as np


@dataclass(frozen=True)
class ZaffSonicValue:
    energy_hartree: float
    gradient_sonic: np.ndarray
    hessian_vector_sonic: np.ndarray | None
    coordinates_bohr: np.ndarray
    execution: dict


class ZaffSonicObjective:
    """Energy/gradient and analytic matrix-free Hv in active natural SONIC.

For an independent local pool, bonded derivatives stay entirely in SONIC.
For a redundant pool, compatible primitive changes are pulled through the
implicit geometry; all operations remain sparse/operator actions.
"""

    def __init__(self, geometry_map, bonded, nonbonded, *, construction_contract=None):
        if bonded.variables is not geometry_map.variables:
            raise ValueError("bonded runtime and inverse geometry must share the same variable map")
        self.geometry_map, self.bonded, self.nonbonded = geometry_map, bonded, nonbonded
        self._state, self._coordinates = None, None
        self.construction_contract = deepcopy(construction_contract)

    def evaluate(self, coordinates, *, direction=None):
        z = np.asarray(coordinates, dtype=float)
        if z.shape != (len(self.geometry_map.active),) or not np.all(np.isfinite(z)):
            raise ValueError("invalid active SONIC coordinates")
        if direction is not None:
            direction = np.asarray(direction, dtype=float)
            if direction.shape != z.shape or not np.all(np.isfinite(direction)):
                raise ValueError("invalid active SONIC direction")
        # Exact-point reuse only: no derivative evaluated on a stale geometry.
        if self._state is None or not np.array_equal(z, self._coordinates):
            state = self.geometry_map.realize(z)
            self._state, self._coordinates = state, z.copy()
        state = self._state
        embedding = self.geometry_map.embedding
        tangent = None if direction is None else state.push(direction)
        independent = embedding.analysis.shape[0] == embedding.analysis.shape[1]
        local_direction = (None if direction is None else
                           embedding.push(direction) if independent else state.natural_b @ tangent)
        bonded = self.bonded.evaluate(state.natural_values, direction=local_direction)
        nonbonded = self.nonbonded.evaluate(
            state.coordinates_bohr,
            direction_bohr=None if tangent is None else tangent.reshape(state.coordinates_bohr.shape))
        cart_gradient = nonbonded.gradient_hartree_per_bohr.copy()
        if not independent:
            cart_gradient += state.natural_b.T @ bonded.gradient_natural
        multipliers = state.multipliers(cart_gradient)
        gradient = multipliers[self.geometry_map.active].copy()
        if independent:
            gradient += embedding.pull(bonded.gradient_natural)
        product = None
        if direction is not None:
            weights = -(embedding.analysis.T @ multipliers[:self.geometry_map.row_count])
            cart_product = nonbonded.hessian_vector_hartree_per_bohr.copy()
            if not independent:
                weights += bonded.gradient_natural
                cart_product += state.natural_b.T @ bonded.hessian_vector_natural
            cart_product += state.weighted_curvature(weights, tangent)
            product = state.pull(cart_product)
            if independent:
                product += embedding.pull(bonded.hessian_vector_natural)
        return ZaffSonicValue(
            bonded.energy_hartree + nonbonded.energy_hartree, gradient, product,
            state.coordinates_bohr.copy(), {
                "coordinate_space": "ACTIVE_NATURAL_SONIC",
                "bonded_pullback": "DIRECT_SONIC" if independent else "GEOMETRICALLY_COMPATIBLE_SPARSE",
                "inverse_iterations": state.iterations, "inverse_residual": state.residual,
                "inverse_condition_estimate": state.condition_estimate,
                "inverse_map_curvature": direction is not None,
                "owned_global_dense_matrices": 0,
                "linear_scaling_certified": False,
                "bonded": bonded.execution, "nonbonded": nonbonded.execution,
                "architect_construction": (None if self.construction_contract is None else {
                    key: self.construction_contract[key] for key in
                    ("owner", "family", "route", "topology_sha256", "epoch")}),
            })

    def energy_gradient(self, coordinates):
        """Optimizer callable (energy, gradient), e.g. with jac=True."""
        value = self.evaluate(coordinates)
        return value.energy_hartree, value.gradient_sonic

    def hessian_vector_product(self, coordinates, direction):
        return self.evaluate(coordinates, direction=direction).hessian_vector_sonic
