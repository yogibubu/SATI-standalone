#!/usr/bin/env python3
"""Protocol adapter template between LINK and an existing SENTINEL GA module.

The collaborator supplies a Python module with:
  load_or_initialize(state_dir, contract, cycle) -> state
  ingest(state, realized_points) -> None
  evaluate_driver_owned(state, realized_points) -> list[dict]
  converged(state) -> bool
  evolve(state) -> iterable[sequence[float]]
  save(state, state_dir) -> None

This file owns protocol validation, serialization and atomic response writing;
the supplied module owns every genetic and fitness decision.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
from collections.abc import Sequence

from matrix_core import atomic_json_write


REQUEST_SCHEMA = "matrix.link.sentinel.request.v1"
RESPONSE_SCHEMA = "matrix.link.sentinel.response.v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    parser.add_argument("--ga-module", required=True)
    parser.add_argument("--evaluation-owner", choices=("link", "driver"), default="link")
    parser.add_argument(
        "--property",
        choices=("energy", "gradient", "hessian"),
        action="append",
        default=[],
    )
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    _validate_request(request)
    cycle = int(request["cycle"])
    contract = request["coordinate_contract"]["active_variables"]
    labels = tuple(str(item) for item in contract["variable_labels"])
    points = request["points"]
    state_dir = args.request.parent.parent / "sentinel_state"
    state_dir.mkdir(parents=True, exist_ok=True)

    ga = importlib.import_module(args.ga_module)
    state = ga.load_or_initialize(state_dir, contract, cycle)
    evaluations = list(ga.evaluate_driver_owned(state, points))
    ga.ingest(state, points)

    if ga.converged(state):
        status = "complete"
        next_points: list[dict[str, object]] = []
    else:
        status = "continue"
        properties = args.property or ["energy"]
        chromosomes = list(ga.evolve(state))
        if not chromosomes:
            raise ValueError("SENTINEL evolve() returned an empty population")
        next_points = [
            _next_point(
                chromosome,
                labels=labels,
                cycle=cycle + 1,
                index=index,
                evaluation_owner=args.evaluation_owner,
                properties=properties,
            )
            for index, chromosome in enumerate(chromosomes)
        ]

    ga.save(state, state_dir)
    response = {
        "schema": RESPONSE_SCHEMA,
        "protocol_version": "1.0",
        "run_id": request["run_id"],
        "transaction_id": request["transaction_id"],
        "cycle": cycle,
        "status": status,
        "evaluations": evaluations,
        "next_points": next_points,
    }
    _write_atomic(args.response, response)
    return 0


def _validate_request(payload: object) -> None:
    if not isinstance(payload, dict) or payload.get("schema") != REQUEST_SCHEMA:
        raise ValueError(f"request must use schema {REQUEST_SCHEMA}")
    if payload.get("sender") != "LINK" or payload.get("receiver") != "SENTINEL":
        raise ValueError("invalid LINK/SENTINEL sender or receiver")
    if not isinstance(payload.get("points"), list) or not payload["points"]:
        raise ValueError("LINK request must contain at least one point")
    contract = payload.get("coordinate_contract", {}).get("active_variables", {})
    labels = contract.get("variable_labels")
    references = contract.get("reference_values")
    if not isinstance(labels, list) or not labels or len(labels) != len(references or ()):
        raise ValueError("invalid active-variable labels/reference values")


def _next_point(
    chromosome: Sequence[float],
    *,
    labels: tuple[str, ...],
    cycle: int,
    index: int,
    evaluation_owner: str,
    properties: Sequence[str],
) -> dict[str, object]:
    values = [float(item) for item in chromosome]
    if len(values) != len(labels) or not all(math.isfinite(item) for item in values):
        raise ValueError("SENTINEL chromosome has invalid length or non-finite genes")
    return {
        "point_id": f"generation-{cycle}-individual-{index}",
        "labels": list(labels),
        "variable_values": values,
        "evaluation_owner": evaluation_owner,
        "requested_properties": list(properties),
    }


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    atomic_json_write(path, payload, sort_keys=False)


if __name__ == "__main__":
    raise SystemExit(main())
