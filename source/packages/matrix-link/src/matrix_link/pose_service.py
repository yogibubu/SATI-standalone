"""Persistent JSON-lines service for B-free rigid-complex realization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, TextIO

import numpy as np

from .rigid_pose import RigidComplexModel, RigidFragmentPose

if TYPE_CHECKING:
    from matrix_smith import GICDefinition


PROTOCOL_SCHEMA = "matrix.link.rigid_pose.v1"


def load_rigid_complex_model(path: Path | str) -> RigidComplexModel:
    from matrix_smith import read_gic_definition_from_xyzin

    definition = read_gic_definition_from_xyzin(Path(path))
    return RigidComplexModel.from_definition(definition)


def serve_pose_requests(
    model: RigidComplexModel,
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    definition: "GICDefinition | None" = None,
    parallel_workers: int = 1,
) -> None:
    """Serve independent JSON requests until EOF.

    Errors are returned per request, so one malformed candidate does not
    terminate a long-running Monte Carlo/GA worker.
    """

    for raw_line in input_stream:
        line = raw_line.strip()
        if not line:
            continue
        request_id = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            request_id = request.get("id")
            response = _handle_request(
                model,
                request,
                definition=definition,
                parallel_workers=parallel_workers,
            )
            payload = {
                "schema": PROTOCOL_SCHEMA,
                "id": request_id,
                "ok": True,
                **response,
            }
        except Exception as exc:  # service boundary: report and continue
            payload = {
                "schema": PROTOCOL_SCHEMA,
                "id": request_id,
                "ok": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        output_stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
        output_stream.flush()


def _handle_request(
    model: RigidComplexModel,
    request: dict[str, object],
    *,
    definition: "GICDefinition | None" = None,
    parallel_workers: int = 1,
) -> dict[str, object]:
    operation = str(request.get("op", "")).strip().lower()
    workers = int(request.get("workers", parallel_workers))
    if workers <= 0:
        raise ValueError("workers must be positive")
    if operation == "describe":
        return {
            "coordinate_count": model.coordinate_count,
            "fragment_count": len(model.blocks),
            "coordinate_indices": list(model.coordinate_indices),
            "atom_count": int(model.reference_coordinates_angstrom.shape[0]),
            "parallel_workers": workers,
        }
    if operation == "realize_sonic":
        coordinates = model.realize_sonic(request["values"])
        return {"coordinates_angstrom": coordinates.tolist()}
    if operation == "realize_sonic_batch":
        coordinates = model.realize_sonic_batch(request["values"], workers=workers)
        return {"coordinates_angstrom": coordinates.tolist()}
    if operation == "realize_pose":
        coordinates = model.realize(_poses_from_payload(request["poses"]))
        return {"coordinates_angstrom": coordinates.tolist()}
    if operation == "realize_pose_batch":
        batches = tuple(_poses_from_payload(candidate) for candidate in request["poses"])
        return {"coordinates_angstrom": model.realize_batch(batches, workers=workers).tolist()}
    if operation == "extract_pose":
        poses = model.extract_poses(np.asarray(request["coordinates_angstrom"], dtype=float))
        return {"poses": _poses_to_payload(poses)}
    if operation == "mutate_pose":
        poses = model.mutate_pose(
            _poses_from_payload(request["poses"]),
            int(request["fragment_index"]),
            translation_increment_angstrom=request.get(
                "translation_increment_angstrom", (0.0, 0.0, 0.0)
            ),
            rotation_increment_radian=request.get("rotation_increment_radian", (0.0, 0.0, 0.0)),
        )
        return {
            "poses": _poses_to_payload(poses),
            "coordinates_angstrom": model.realize(poses).tolist(),
        }
    if operation == "realize_constraints":
        if definition is None:
            raise ValueError("constraint realization requires a loaded GIC definition")
        initial = None if request.get("poses") is None else _poses_from_payload(request["poses"])
        result = model.realize_sonic_constraints(
            definition,
            request["coordinate_indices"],
            request["target_values"],
            initial_poses=initial,
            max_iterations=int(request.get("max_iterations", 20)),
            tolerance=float(request.get("tolerance", 1.0e-9)),
            parallel_workers=workers,
        )
        return {
            "poses": _poses_to_payload(result.poses),
            "coordinates_angstrom": result.coordinates_angstrom.tolist(),
            "values": result.values.tolist(),
            "residual": result.residual.tolist(),
            "iterations": result.iterations,
            "converged": result.converged,
        }
    if operation == "realize_constraints_batch":
        if definition is None:
            raise ValueError("constraint realization requires a loaded GIC definition")
        initial = request.get("poses")
        initial_batch = (
            None
            if initial is None
            else tuple(_poses_from_payload(candidate) for candidate in initial)
        )
        results = model.realize_sonic_constraints_batch(
            definition,
            request["coordinate_indices"],
            request["target_values"],
            initial_pose_batch=initial_batch,
            workers=workers,
            max_iterations=int(request.get("max_iterations", 20)),
            tolerance=float(request.get("tolerance", 1.0e-9)),
        )
        return {"results": [_constraint_result_to_payload(result) for result in results]}
    raise ValueError(f"unsupported pose-service operation: {operation!r}")


def _poses_from_payload(payload: object) -> tuple[RigidFragmentPose, ...]:
    if not isinstance(payload, list):
        raise ValueError("poses must be a JSON list")
    poses = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each pose must be a JSON object")
        poses.append(
            RigidFragmentPose(
                item["translation_angstrom"],
                item["quaternion_wxyz"],
            )
        )
    return tuple(poses)


def _poses_to_payload(
    poses: tuple[RigidFragmentPose, ...],
) -> list[dict[str, list[float]]]:
    return [
        {
            "translation_angstrom": pose.translation_angstrom.tolist(),
            "quaternion_wxyz": pose.quaternion_wxyz.tolist(),
        }
        for pose in poses
    ]


def _constraint_result_to_payload(result) -> dict[str, object]:
    return {
        "poses": _poses_to_payload(result.poses),
        "coordinates_angstrom": result.coordinates_angstrom.tolist(),
        "values": result.values.tolist(),
        "residual": result.residual.tolist(),
        "iterations": result.iterations,
        "converged": result.converged,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Persistent MATRIX LINK rigid-pose geometry service"
    )
    parser.add_argument("xyzin", type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="worker threads for independent candidates (default: 1)",
    )
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be positive")
    from matrix_smith import read_gic_definition_from_xyzin

    definition = read_gic_definition_from_xyzin(args.xyzin)
    model = RigidComplexModel.from_definition(definition)
    serve_pose_requests(
        model,
        sys.stdin,
        sys.stdout,
        definition=definition,
        parallel_workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROTOCOL_SCHEMA",
    "load_rigid_complex_model",
    "main",
    "serve_pose_requests",
]
