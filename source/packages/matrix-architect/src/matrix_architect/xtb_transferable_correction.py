"""Hierarchical, self-consistent corrections for LINK-xTB primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


def _roles(family: str, count: int) -> tuple[str, ...]:
    if family == "R" and count == 2:
        return ("endpoint", "endpoint")
    if family == "A" and count == 3:
        return ("terminal", "center", "terminal")
    if family == "L" and count == 3:
        return ("terminal", "center", "terminal")
    if family == "SOOP" and count >= 4:
        return ("center", "moving", *("plane" for _ in range(count - 2)))
    if family == "D" and count == 4:
        return ("terminal", "axis", "axis", "terminal")
    return tuple("atom" for _ in range(count))


@dataclass(frozen=True)
class HierarchicalPrimitiveParameters:
    """SH-like family baselines with sparse chemical modulation.

    Parameters live in log-factor space for Hessian scales and stretch/SOOP
    geometry scales.  Angle-geometry parameters are in radians.  An absent
    atom-type/role key contributes exactly zero, so unseen chemistry falls
    back to the family baseline instead of becoming undefined.
    """

    family: Mapping[str, float]
    atom_type_role: Mapping[str, float] = field(default_factory=dict)
    bond_order_slope: Mapping[str, float] = field(default_factory=dict)
    scc_charge_slope: Mapping[str, float] = field(default_factory=dict)
    # Geometry corrections are independent of the raw xTB distance.  The
    # environment key is ``X:hyb_X|Y:hyb_Y`` in canonical endpoint order and
    # the complete correction is multiplied by the final Mayer bond order.
    # Consequently it vanishes continuously on dissociation.  C--H is an
    # immutable identity branch, irrespective of populated parameters.
    geometry_hybridization_environment: Mapping[str, float] = field(default_factory=dict)
    geometry_charge_difference: float = 0.0
    geometry_charge_sum: float = 0.0
    geometry_charge_product: float = 0.0
    geometry_angle_center: Mapping[str, float] = field(default_factory=dict)
    geometry_angle_order_sum: float = 0.0
    geometry_angle_order_geom: float = 0.0

    def linear_predictor(self, feature: Mapping[str, Any]) -> float:
        family = str(feature["family"])
        types = tuple(str(value) for value in feature["atom_types"])
        roles = _roles(family, len(types))
        modifiers = [
            float(self.atom_type_role.get(f"{family}:{role}:{atom_type}", 0.0))
            for role, atom_type in zip(roles, types, strict=True)
        ]
        atom_term = float(np.mean(modifiers)) if modifiers else 0.0
        order = max(float(feature["geometric_mean_mayer_order"]), 1.0e-6)
        charge = float(feature.get("mean_abs_scc_charge", 0.0))
        return (
            float(self.family.get(family, 0.0))
            + atom_term
            + float(self.bond_order_slope.get(family, 0.0)) * math.log(order)
            + float(self.scc_charge_slope.get(family, 0.0)) * charge
        )


def hierarchical_hessian_scale_factors(
    features: Sequence[Mapping[str, Any]],
    parameters: HierarchicalPrimitiveParameters,
) -> np.ndarray:
    """Evaluate positive diagonal factors; couplings use geometric means."""
    return np.exp(
        np.asarray([parameters.linear_predictor(feature) for feature in features])
    )


def lcb26_link_xtb_geometry_parameters() -> HierarchicalPrimitiveParameters:
    """Load the frozen 26-molecule LCB26 geometry correction."""

    payload = json.loads(
        files("matrix_architect")
        .joinpath("data/link_xtb_geometry_correction.json")
        .read_text(encoding="utf-8")
    )
    if payload.get("schema") != "matrix.architect.link-xtb-geometry-correction.v2":
        raise ValueError("unsupported LINK-xTB geometry-correction parameter schema")
    if payload.get("distance_descriptor_used") is not False:
        raise ValueError("LINK-xTB v2 geometry correction must be distance-independent")
    return HierarchicalPrimitiveParameters(
        family={},
        geometry_hybridization_environment={
            str(key): float(value)
            for key, value in dict(payload["hybridization_environment"]).items()
        },
        geometry_charge_difference=float(payload["charge_difference"]),
        geometry_charge_sum=float(payload["charge_sum"]),
        geometry_charge_product=float(payload["charge_product"]),
    )


def correct_primitive_geometry_values(
    values: Sequence[float],
    features: Sequence[Mapping[str, Any]],
    parameters: HierarchicalPrimitiveParameters,
) -> np.ndarray:
    """Correct equilibrium primitive parameters without breaking periodicity.

    Stretches are scaled multiplicatively.  Valence angles receive bounded
    additive corrections in radians.  SOOP values are corrected by scaling
    their tangent, which is periodic and leaves an exactly planar coordinate
    exactly zero.  This is safer than multiplying an angle itself.
    """
    raw = np.asarray(values, dtype=float).reshape(-1)
    if raw.shape != (len(features),):
        raise ValueError("primitive values and xTB features have inconsistent sizes")
    corrected = np.empty_like(raw)
    for index, (value, feature) in enumerate(zip(raw, features, strict=True)):
        family = str(feature["family"])
        if family == "R":
            pair = str(feature.get("element_pair", ""))
            normalized_pair = ":".join(sorted(part for part in pair.split(":") if part))
            if normalized_pair == "C:H":
                corrected[index] = value
            elif parameters.geometry_hybridization_environment:
                order = max(float(feature.get("mayer_bond_order", 0.0)), 0.0)
                charge_difference = abs(float(feature.get("charge_difference", 0.0)))
                charge_sum = float(feature.get("charge_sum", 0.0))
                charge_product = float(feature.get("charge_product", 0.0))
                environment = str(feature.get("hybridization_environment", ""))
                corrected[index] = value + order * (
                    float(parameters.geometry_hybridization_environment.get(environment, 0.0))
                    + float(parameters.geometry_charge_difference) * charge_difference
                    + float(parameters.geometry_charge_sum) * charge_sum
                    + float(parameters.geometry_charge_product) * charge_product
                )
            else:
                correction = parameters.linear_predictor(feature)
                corrected[index] = value * math.exp(correction)
        elif family == "A":
            correction = parameters.linear_predictor(feature)
            if parameters.geometry_angle_center:
                center = str(feature.get("center_element", ""))
                left = max(float(feature.get("mayer_bond_order_left", 0.0)), 0.0)
                right = max(float(feature.get("mayer_bond_order_right", 0.0)), 0.0)
                correction = (
                    float(parameters.geometry_angle_center.get(center, 0.0))
                    + float(parameters.geometry_angle_order_sum) * 0.5 * (left + right)
                    + float(parameters.geometry_angle_order_geom) * math.sqrt(left * right)
                )
            corrected[index] = math.acos(float(np.clip(math.cos(value) + correction, -1.0, 1.0)))
        elif family == "SOOP":
            correction = parameters.linear_predictor(feature)
            corrected[index] = math.atan2(math.exp(correction) * math.sin(value), math.cos(value))
        else:
            corrected[index] = value
    return corrected


__all__ = [
    "HierarchicalPrimitiveParameters",
    "correct_primitive_geometry_values",
    "hierarchical_hessian_scale_factors",
    "lcb26_link_xtb_geometry_parameters",
]
