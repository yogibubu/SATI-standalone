"""Construction and validation of ellipsoidal EPOC boundary requests."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

from .contracts import EPOC_BOUNDARY_SCHEMA, LEGACY_KEYMAKER_BOUNDARY_SCHEMA


ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
READABLE_BOUNDARY_SCHEMAS = frozenset(
    {EPOC_BOUNDARY_SCHEMA, LEGACY_KEYMAKER_BOUNDARY_SCHEMA}
)


def is_boundary_record(boundary: Mapping[str, Any]) -> bool:
    """Return whether a mapping is a readable frozen boundary record."""

    return str(boundary.get("schema", "")) in READABLE_BOUNDARY_SCHEMAS


def build_boundary_request(
    boundary: Mapping[str, Any],
    coordinates_angstrom: np.ndarray | None,
    sampler: str,
    *,
    geometry_source: str,
    packing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze boundary intent and coordinates into a restart-safe EPOC request."""

    kind = str(boundary.get("kind", "none")).strip().lower()
    if kind == "none":
        return {
            "schema": EPOC_BOUNDARY_SCHEMA,
            "kind": "none",
            "model": "isolated",
            "enforcement": "none",
        }
    if kind not in {"sphere", "ellipsoid"}:
        raise ValueError("boundary kind must be none, sphere, or ellipsoid")
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1:] != (3,) or not len(coordinates):
        raise ValueError("EPOC boundary coordinates must have shape (n, 3)")
    if np.any(~np.isfinite(coordinates)):
        raise ValueError("EPOC boundary coordinates must be finite")
    padding = float(boundary.get("padding_angstrom", 4.0))
    if not np.isfinite(padding) or padding <= 0.0:
        raise ValueError("NPBC padding must be finite and positive")
    minimum = np.min(coordinates, axis=0)
    maximum = np.max(coordinates, axis=0)
    center = 0.5 * (minimum + maximum)
    rotation = np.eye(3)
    explicit_axes = boundary.get("semiaxes_angstrom")
    if explicit_axes is not None:
        semiaxes = np.asarray(explicit_axes, dtype=float).reshape(3)
        center = np.asarray(boundary.get("center_angstrom", np.zeros(3)), dtype=float).reshape(3)
        rotation = np.asarray(boundary.get("rotation", np.eye(3)), dtype=float).reshape(3, 3)
        if (
            np.any(~np.isfinite(semiaxes))
            or np.any(semiaxes <= 0.0)
            or np.any(~np.isfinite(center))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-12)
            or (kind == "sphere" and not np.allclose(semiaxes, semiaxes[0]))
        ):
            raise ValueError("explicit NPBC centre, axes or rotation are invalid")
    elif kind == "ellipsoid":
        center, rotation = _deterministic_principal_frame(coordinates)
        local = (coordinates - center) @ rotation
        semiaxes = np.max(np.abs(local), axis=0) + padding
    else:
        semiaxes = 0.5 * (maximum - minimum) + padding
        semiaxes[:] = float(np.max(semiaxes))
    wall = _validated_wall(boundary)
    enforcement = (
        "event-driven-specular-reflection"
        if sampler == "MOLECULAR_DYNAMICS"
        else "hard-constraint-dominance"
    )
    packing_record = None if packing is None else dict(packing)
    physical = {
        "kind": kind,
        "center_bohr": (center * ANGSTROM_TO_BOHR).tolist(),
        "semiaxes_bohr": (semiaxes * ANGSTROM_TO_BOHR).tolist(),
        "rotation": rotation.tolist(),
        "wall": wall,
    }
    if packing_record is not None and packing_record.get("sha256"):
        physical["packing_sha256"] = str(packing_record["sha256"])
    signature = _physical_signature(physical)
    request = {
        "schema": EPOC_BOUNDARY_SCHEMA,
        "kind": kind,
        "model": "CPCM",
        "auto_dimensions": explicit_axes is None,
        "geometry_source": str(geometry_source),
        "orientation_policy": (
            "fixed-solute-principal-frame-v1"
            if explicit_axes is not None
            else "deterministic-principal-envelope-v1" if kind == "ellipsoid"
            else "rotation-invariant-sphere"
        ),
        "padding_angstrom": padding,
        "solvent": str(boundary.get("solvent", "water")),
        "dielectric": boundary.get("dielectric", "from-solvent-library"),
        "center_angstrom": center.tolist(),
        "semiaxes_angstrom": semiaxes.tolist(),
        "center_bohr": physical["center_bohr"],
        "semiaxes_bohr": physical["semiaxes_bohr"],
        "rotation": physical["rotation"],
        "enforcement": enforcement,
        "constraint_evaluator": (
            "matrix_epoc.ellipsoidal_molecular_constraint_violation"
        ),
        "mean_field_potential": {
            "factory": "matrix_epoc.ellipsoidal_repulsive_site_boundary",
            "profile": "strictly-repulsive-exponential-times-quintic-C2",
            "wall_height_kcal_per_mol": wall["wall_height_kcal_per_mol"],
            "decay_per_angstrom": wall["decay_per_angstrom"],
            "layer_depth_angstrom": wall["layer_depth_angstrom"],
        },
        "wall": wall,
        "checkpoint_policy": {
            "boundary_signature": signature,
            "legacy_upgrade_requires_explicit_authorization": True,
            "legacy_upgrade_behavior": (
                "preserve-pose-and-velocities-reset-forces-clock-and-samples"
            ),
            "allow_legacy_upgrade": bool(
                boundary.get(
                    "allow_reflecting_boundary_upgrade",
                    boundary.get("allow_legacy_checkpoint_upgrade", False),
                )
            ),
        },
    }
    if packing_record is not None:
        request["packing"] = packing_record
    return request


def boundary_signature(boundary: Mapping[str, Any]) -> str:
    """Return the frozen physical-boundary signature."""

    policy = boundary.get("checkpoint_policy", {})
    return str(policy.get("boundary_signature", ""))


def variable_cell_boundary_request(
    boundary: Mapping[str, Any],
    *,
    center_bohr: np.ndarray,
    semiaxes_bohr: np.ndarray,
    rotation: np.ndarray,
) -> dict[str, Any]:
    """Return one signed variable-cell revision of a readable EPOC boundary."""

    if not is_boundary_record(boundary) or boundary.get("kind") == "none":
        raise ValueError("variable-cell dynamics requires an explicit NPBC contract")
    center = np.asarray(center_bohr, dtype=float)
    axes = np.asarray(semiaxes_bohr, dtype=float)
    frame = np.asarray(rotation, dtype=float)
    if (
        center.shape != (3,)
        or axes.shape != (3,)
        or frame.shape != (3, 3)
        or np.any(~np.isfinite(center))
        or np.any(~np.isfinite(axes))
        or np.any(axes <= 0.0)
        or not np.allclose(frame.T @ frame, np.eye(3), atol=1.0e-12)
        or not np.isclose(np.linalg.det(frame), 1.0, atol=1.0e-12)
    ):
        raise ValueError("variable-cell NPBC geometry is invalid")
    updated = dict(boundary)
    updated["schema"] = EPOC_BOUNDARY_SCHEMA
    updated["kind"] = "ellipsoid"
    updated["auto_dimensions"] = False
    updated["orientation_policy"] = "variable-cell-fixed-principal-frame-v1"
    updated["center_bohr"] = center.tolist()
    updated["semiaxes_bohr"] = axes.tolist()
    updated["rotation"] = frame.tolist()
    updated["center_angstrom"] = (center / ANGSTROM_TO_BOHR).tolist()
    updated["semiaxes_angstrom"] = (axes / ANGSTROM_TO_BOHR).tolist()
    physical = {
        "kind": "ellipsoid",
        "center_bohr": updated["center_bohr"],
        "semiaxes_bohr": updated["semiaxes_bohr"],
        "rotation": updated["rotation"],
        "wall": dict(updated.get("wall", {})),
    }
    policy = dict(updated.get("checkpoint_policy", {}))
    policy["boundary_signature"] = _physical_signature(physical)
    updated["checkpoint_policy"] = policy
    return updated


def _validated_wall(boundary: Mapping[str, Any]) -> dict[str, float | int]:
    wall: dict[str, float | int] = {
        "safety_clearance_bohr": float(boundary.get("safety_clearance_bohr", 1.0e-6)),
        "constraint_safety_fraction": float(
            boundary.get("constraint_safety_fraction", 5.0e-4)
        ),
        "restitution": float(boundary.get("restitution", 1.0)),
        "maximum_collisions_per_drift": int(
            boundary.get("maximum_collisions_per_drift", 8)
        ),
        "wall_height_kcal_per_mol": float(
            boundary.get("wall_height_kcal_per_mol", 25.0)
        ),
        "decay_per_angstrom": float(boundary.get("decay_per_angstrom", 3.0)),
        "layer_depth_angstrom": float(boundary.get("layer_depth_angstrom", 1.5)),
    }
    if (
        wall["safety_clearance_bohr"] <= 0.0
        or not 0.0 <= wall["constraint_safety_fraction"] < 1.0
        or not 0.0 <= wall["restitution"] <= 1.0
        or wall["maximum_collisions_per_drift"] < 1
        or wall["wall_height_kcal_per_mol"] <= 0.0
        or wall["decay_per_angstrom"] <= 0.0
        or wall["layer_depth_angstrom"] <= 0.0
    ):
        raise ValueError("NPBC wall parameters are invalid")
    return wall


def _physical_signature(physical: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(physical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _deterministic_principal_frame(
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(np.asarray(coordinates, dtype=float), axis=0)
    centered = np.asarray(coordinates, dtype=float) - center
    covariance = centered.T @ centered
    eigenvalues, axes = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    axes = axes[:, order]
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if np.any(np.abs(np.diff(eigenvalues)) <= 1.0e-12 * scale):
        return center, np.eye(3)
    oriented = centered @ axes
    for axis in range(3):
        pivot = int(np.argmax(np.abs(oriented[:, axis])))
        if oriented[pivot, axis] < 0.0:
            axes[:, axis] *= -1.0
            oriented[:, axis] *= -1.0
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    return center, axes


__all__ = [
    "ANGSTROM_TO_BOHR",
    "EPOC_BOUNDARY_SCHEMA",
    "READABLE_BOUNDARY_SCHEMAS",
    "boundary_signature",
    "build_boundary_request",
    "is_boundary_record",
    "variable_cell_boundary_request",
]
