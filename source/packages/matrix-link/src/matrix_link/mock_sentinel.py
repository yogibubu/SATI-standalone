"""Installed deterministic SENTINEL mock used by LINK GUI and conformance tests.

This module deliberately contains no genetic algorithm.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from matrix_core import atomic_json_write

from .sentinel_protocol import validate_request, validate_response


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic non-genetic SENTINEL mock")
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    parser.add_argument(
        "--mode",
        choices=("scan-1d", "scan-2d", "partial", "batch"),
        default="scan-1d",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--driver-owned", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = validate_request(json.loads(args.request.read_text(encoding="utf-8")))
    evaluations = _evaluations(request["points"]) if args.driver_owned else []
    if int(request["cycle"]) == 0:
        response = _envelope(request, "continue")
        response["evaluations"] = evaluations
        response["next_points"] = _next_points(request, args.mode, args.batch_size, args.driver_owned)
    else:
        response = _envelope(request, "complete")
        response["evaluations"] = evaluations
        response["next_points"] = []
    validate_response(response, request=request)
    _write_atomic(args.response, response)
    return 0


def _next_points(request: dict, mode: str, batch_size: int, driver_owned: bool) -> list[dict]:
    active = request["coordinate_contract"]["active_variables"]
    labels = list(active["variable_labels"])
    reference = [float(value) for value in active["reference_values"]]
    vectors = _vectors(mode, reference, batch_size)
    owner = "driver" if driver_owned else "link"
    return [
        {
            "point_id": f"gui-{mode}-{index:04d}",
            "labels": labels,
            "variable_values": vector,
            "evaluation_owner": owner,
            "calculator_id": "sentinel-harmonic" if driver_owned else "link-default",
            "requested_properties": ["energy"],
        }
        for index, vector in enumerate(vectors)
    ]


def _vectors(mode: str, reference: list[float], batch_size: int) -> list[list[float]]:
    if mode == "scan-1d":
        return [_offset(reference, {0: value}) for value in (-0.02, 0.0, 0.02)]
    if mode == "scan-2d":
        if len(reference) < 2:
            raise ValueError("scan-2d needs at least two active variables")
        return [
            _offset(reference, {0: first, 1: second})
            for first, second in itertools.product((-0.01, 0.01), repeat=2)
        ]
    if mode == "partial":
        return [_offset(reference, {0: 0.015})]
    if batch_size < 2:
        raise ValueError("batch-size must be at least two")
    center = 0.5 * (batch_size - 1)
    return [_offset(reference, {0: 0.005 * (index - center)}) for index in range(batch_size)]


def _offset(reference: list[float], changes: dict[int, float]) -> list[float]:
    values = list(reference)
    for index, offset in changes.items():
        values[index] += offset
    return values


def _evaluations(points: list[dict]) -> list[dict]:
    results = []
    force = 0.02
    angstrom_to_bohr = 1.0 / 0.529177210903
    for point in points:
        if point["evaluation_owner"] != "driver":
            continue
        requested = set(point["requested_properties"])
        coordinates = [
            float(value) * angstrom_to_bohr
            for xyz in point["cartesian"]["coordinates_angstrom"]
            for value in xyz
        ]
        ncart = len(coordinates)
        result = {
            "point_id": point["point_id"],
            "status": "completed",
            "energy_hartree": (
                -75.0 + 0.5 * force * sum(value * value for value in coordinates)
                if "energy" in requested
                else None
            ),
            "cartesian_gradient_hartree_per_bohr": (
                [force * value for value in coordinates] if "gradient" in requested else None
            ),
            "cartesian_hessian_hartree_per_bohr2": (
                [
                    [force if row == column else 0.0 for column in range(ncart)]
                    for row in range(ncart)
                ]
                if "hessian" in requested
                else None
            ),
        }
        results.append(result)
    return results


def _envelope(request: dict, status: str) -> dict:
    return {
        "schema": "matrix.link.sentinel.response.v1",
        "protocol_version": "1.0",
        "run_id": request["run_id"],
        "transaction_id": request["transaction_id"],
        "cycle": request["cycle"],
        "status": status,
    }


def _write_atomic(path: Path, payload: dict) -> None:
    atomic_json_write(path, payload, allow_nan=False)


if __name__ == "__main__":
    raise SystemExit(main())
