#!/usr/bin/env python3
"""Minimal SENTINEL-style driver requesting a batch in one active variable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RESPONSE_SCHEMA = "matrix.link.sentinel.response.v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    parser.add_argument("--coordinate", required=True)
    parser.add_argument("--offset", type=float, action="append", required=True)
    parser.add_argument("--evaluation-owner", choices=("link", "driver"), default="link")
    parser.add_argument(
        "--property",
        choices=("energy", "gradient", "hessian"),
        action="append",
        default=[],
    )
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    labels = tuple(
        request["coordinate_contract"]["active_variables"]["variable_labels"]
    )
    try:
        coordinate_index = labels.index(args.coordinate)
    except ValueError as exc:
        raise SystemExit(f"unknown active variable: {args.coordinate}") from exc

    if int(request["cycle"]) == 0:
        reference = list(request["points"][0]["active_variables"]["values"])
        properties = args.property or ["energy"]
        next_points = []
        for index, offset in enumerate(args.offset):
            values = list(reference)
            values[coordinate_index] += float(offset)
            next_points.append(
                {
                    "point_id": f"scan-{index:04d}",
                    "labels": list(labels),
                    "variable_values": values,
                    "evaluation_owner": args.evaluation_owner,
                    "requested_properties": properties,
                }
            )
        response = {
            "schema": RESPONSE_SCHEMA,
            "protocol_version": "1.0",
            "run_id": request["run_id"],
            "transaction_id": request["transaction_id"],
            "cycle": request["cycle"],
            "status": "continue",
            "next_points": next_points,
        }
    else:
        response = {
            "schema": RESPONSE_SCHEMA,
            "protocol_version": "1.0",
            "run_id": request["run_id"],
            "transaction_id": request["transaction_id"],
            "cycle": request["cycle"],
            "status": "complete",
            "next_points": [],
        }
    args.response.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
