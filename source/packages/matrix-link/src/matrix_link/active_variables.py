from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .internal_coordinates import (
    cartesian_from_internal_jacobian,
    internal_from_cartesian_jacobian,
)

from .optimizer import OptimizerCoordinateModel
from .scan import (
    coordinate_direction_from_cartesian_vector,
    coordinate_direction_from_normal_mode,
)


LINK_ACTIVE_VARIABLES_SCHEMA = "matrix.link.active_variables.v1"


@dataclass(frozen=True)
class ActiveVariableContract:
    """Frozen map from user/SENTINEL variables into the SONIC tangent space."""

    source_path: Path
    model: OptimizerCoordinateModel
    variables: tuple[dict[str, Any], ...]
    projection_residuals: tuple[float, ...]

    def protocol_payload(self) -> dict[str, object]:
        transform = self.model.sonic_from_coordinates
        if transform is None:
            transform = np.eye(len(self.model.labels), dtype=float)
        return {
            "schema": LINK_ACTIVE_VARIABLES_SCHEMA,
            "source": str(self.source_path),
            "variable_labels": list(self.model.labels),
            "reference_values": np.asarray(self.model.reference_values, dtype=float).tolist(),
            "variables": [dict(item) for item in self.variables],
            "sonic_labels": list(self.model.sonic_labels),
            "sonic_from_variable_displacements": np.asarray(transform, dtype=float).tolist(),
            "mapping": "delta_q_SONIC = sonic_from_variable_displacements @ delta_variables",
            "projection_residuals": list(self.projection_residuals),
            "frozen_policy": "all SONIC directions outside the mapped active subspace",
        }


def active_variable_contract_from_file(
    xyzin_path: Path | str,
    path: Path | str,
    *,
    retained_group: str = "C1",
    pes_exploration: bool = False,
) -> ActiveVariableContract:
    """Read and freeze a partial LINK/SENTINEL variable specification."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != LINK_ACTIVE_VARIABLES_SCHEMA:
        raise ValueError(f"active-variable input must use schema {LINK_ACTIVE_VARIABLES_SCHEMA}")
    raw_variables = payload.get("variables")
    if not isinstance(raw_variables, list) or not raw_variables:
        raise ValueError("active-variable input needs a non-empty variables list")
    rcond = float(payload.get("projection_rcond", 1.0e-8))
    tolerance = float(payload.get("projection_tolerance", 1.0e-5))
    if rcond <= 0.0 or tolerance < 0.0:
        raise ValueError("projection_rcond must be positive and projection_tolerance non-negative")

    from matrix_smith import (
        build_gic_b_matrix,
        build_pes_exploration_gic_definition_from_xyzin,
        evaluate_gic_values,
        read_gic_definition_from_xyzin,
    )

    target = Path(xyzin_path)
    definition = (
        build_pes_exploration_gic_definition_from_xyzin(
            target, retained_group=retained_group
        )
        if pes_exploration
        else read_gic_definition_from_xyzin(target)
    )
    coordinates = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    b_matrix = np.asarray(
        build_gic_b_matrix(definition, coordinates_angstrom=coordinates).rows,
        dtype=float,
    )
    from .fragment_backtransform import direct_fragment_rigid_tangent

    cartesian_from_sonic = cartesian_from_internal_jacobian(b_matrix, rcond=rcond)
    fragment_tangent = direct_fragment_rigid_tangent(definition, coordinates, b_matrix)
    for handled_index in fragment_tangent.handled_indices:
        cartesian_from_sonic[:, handled_index] = fragment_tangent.cartesian_from_q[
            :, handled_index
        ]
    sonic_reference = np.asarray(
        evaluate_gic_values(definition, coordinates_angstrom=coordinates), dtype=float
    )
    identifiers = tuple(gic.identifier for gic in definition.gics)
    names = tuple(gic.name for gic in definition.gics)
    columns: list[np.ndarray] = []
    references: list[float] = []
    normalized_variables: list[dict[str, Any]] = []
    residuals: list[float] = []
    seen: set[str] = set()

    for index, raw in enumerate(raw_variables):
        if not isinstance(raw, dict):
            raise ValueError("each active variable must be a JSON object")
        item = dict(raw)
        name = str(item.get("name", f"v{index + 1}")).strip()
        if not name or name in seen:
            raise ValueError(f"invalid or duplicate active-variable name: {name!r}")
        seen.add(name)
        kind = str(item.get("kind", "sonic")).strip().lower().replace("-", "_")
        if str(item.get("role", "active")).strip().lower() != "active":
            raise ValueError("the variables list defines active variables; omit frozen entries")

        residual = 0.0
        if kind == "sonic":
            coordinate = item.get("coordinate", item.get("label"))
            sonic_index = _coordinate_index(coordinate, identifiers, names)
            column = np.zeros(len(identifiers), dtype=float)
            column[sonic_index] = float(item.get("coefficient", 1.0))
            reference = float(item.get("reference_value", sonic_reference[sonic_index]))
            item["coordinate"] = identifiers[sonic_index]
            item.setdefault("units", "SONIC-unit")
            item.update(gic_metadata_for_contract(definition, identifiers[sonic_index]))
        elif kind in {"sonic_linear_combination", "linear_sonic"}:
            terms = item.get("terms")
            if not isinstance(terms, list) or not terms:
                raise ValueError(f"active variable {name} needs a non-empty terms list")
            column = np.zeros(len(identifiers), dtype=float)
            normalized_terms = []
            for term in terms:
                if not isinstance(term, dict):
                    raise ValueError(f"active variable {name} has an invalid SONIC term")
                sonic_index = _coordinate_index(
                    term.get("coordinate", term.get("label")), identifiers, names
                )
                coefficient = float(term.get("coefficient", 1.0))
                column[sonic_index] += coefficient
                normalized_terms.append(
                    {
                        "coordinate": identifiers[sonic_index],
                        "coefficient": coefficient,
                        **gic_metadata_for_contract(definition, identifiers[sonic_index]),
                    }
                )
            reference = float(item.get("reference_value", column @ sonic_reference))
            item["terms"] = normalized_terms
            kind = "sonic_linear_combination"
        else:
            if kind == "normal_mode":
                direction = coordinate_direction_from_normal_mode(target, int(item["mode"]))
                item.setdefault("units", "angstrom")
            elif kind == "cartesian":
                direction = coordinate_direction_from_cartesian_vector(
                    item.get("vector", ()), label=name
                )
                item.setdefault("units", "user-unit")
            else:
                raise ValueError(
                    f"unsupported active-variable kind {kind!r}; use sonic, "
                    "sonic_linear_combination, normal_mode or cartesian"
                )
            vector = np.asarray(direction.vector_angstrom, dtype=float).reshape(-1)
            if vector.shape != (cartesian_from_sonic.shape[0],):
                raise ValueError(
                    f"active variable {name} has {vector.size} Cartesian components; "
                    f"expected {cartesian_from_sonic.shape[0]}"
                )
            column = internal_from_cartesian_jacobian(
                cartesian_from_sonic, rcond=rcond
            ) @ vector
            represented = cartesian_from_sonic @ column
            residual = float(np.linalg.norm(represented - vector) / max(np.linalg.norm(vector), 1.0e-15))
            if residual > tolerance:
                raise ValueError(
                    f"active variable {name} is not representable in the frozen SONIC space: "
                    f"relative residual {residual:.6g} > {tolerance:.6g}"
                )
            reference = float(item.get("reference_value", 0.0))

        if not np.all(np.isfinite(column)) or float(np.linalg.norm(column)) <= 0.0:
            raise ValueError(f"active variable {name} has a null/non-finite SONIC projection")
        item["name"] = name
        item["kind"] = kind
        item["reference_value"] = reference
        columns.append(column)
        references.append(reference)
        residuals.append(residual)
        normalized_variables.append(item)

    full_projection = np.column_stack(columns)
    active_rows = np.any(np.abs(full_projection) > 1.0e-12, axis=1)
    projection = full_projection[active_rows, :]
    if np.linalg.matrix_rank(projection) < len(columns):
        raise ValueError("active-variable definitions are linearly dependent in SONIC space")
    sonic_labels = tuple(identifier for identifier, active in zip(identifiers, active_rows) if active)
    directions = cartesian_from_sonic[:, active_rows] @ projection
    model = OptimizerCoordinateModel(
        kind="sonic",
        labels=tuple(item["name"] for item in normalized_variables),
        directions_angstrom=directions.T,
        metric_diagonal=np.maximum(np.sum(directions * directions, axis=0), 1.0e-12),
        sonic_labels=sonic_labels,
        sonic_from_coordinates=projection,
        reference_values=np.asarray(references, dtype=float),
        sonic_definition=definition,
        pes_exploration=bool(pes_exploration),
        retained_group=(
            str(retained_group).strip().upper() or "C1" if pes_exploration else ""
        ),
    )
    return ActiveVariableContract(
        source_path=source.resolve(),
        model=model,
        variables=tuple(normalized_variables),
        projection_residuals=tuple(residuals),
    )


def _coordinate_index(
    coordinate: object,
    identifiers: tuple[str, ...],
    names: tuple[str, ...],
) -> int:
    text = str(coordinate or "").strip()
    if text in identifiers:
        return identifiers.index(text)
    if text in names:
        return names.index(text)
    try:
        index = int(text) - 1
    except ValueError as exc:
        raise ValueError(f"unknown SONIC coordinate: {coordinate!r}") from exc
    if index < 0 or index >= len(identifiers):
        raise ValueError(f"SONIC coordinate index outside 1..{len(identifiers)}: {coordinate!r}")
    return index


def gic_metadata_for_contract(
    definition: Any, identifier: str
) -> dict[str, object]:
    """Expose stable SMITH semantics to downstream samplers."""

    gic = next(
        (
            item
            for item in definition.gics
            if identifier in {item.identifier, item.name}
        ),
        None,
    )
    if gic is None:
        return {}
    primitive = next(
        (
            item
            for item in definition.primitives
            if item.identifier == gic.primitive_id
        ),
        None,
    )
    payload: dict[str, object] = {
        "family": str(gic.family),
        "irrep": str(gic.irrep),
    }
    if primitive is not None:
        payload.update(
            {
                "function": str(primitive.function),
                "semantic_type": str(primitive.semantic_type),
                "atoms": list(primitive.atoms),
                "mode": int(primitive.mode),
                "block": (
                    f"{primitive.function}:"
                    f"{','.join(str(value) for value in primitive.atoms)}:"
                    f"{','.join(str(value) for value in primitive.ref_atoms)}"
                ),
            }
        )
        if primitive.function in {"D", "IMPD", "RPCK"}:
            payload["periodic"] = True
            payload["period"] = float(2.0 * np.pi)
    return payload
