from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from matrix_core import atomic_json_write, require_authorized_descendant_calculation
from .optimizer import (
    GeometryEvaluationService,
    OptimizerCoordinateModel,
    OptimizerSettings,
    coordinate_model_from_xyzin,
)
from .scan import (
    ANGSTROM_TO_BOHR,
    PESExplorationPolicy,
    PointEvaluationResult,
    QMScanBackend,
    prepare_pes_exploration_geometry,
)
from .active_variables import (
    ActiveVariableContract,
    LINK_ACTIVE_VARIABLES_SCHEMA,
    gic_metadata_for_contract,
)
from .sentinel_protocol import (
    CHECKPOINT_SCHEMA,
    ERROR_SCHEMA,
    PROTOCOL_VERSION,
    contract_digest,
    validate_request,
    validate_response,
)


LINK_SENTINEL_REQUEST_SCHEMA = "matrix.link.sentinel.request.v1"
LINK_SENTINEL_RESPONSE_SCHEMA = "matrix.link.sentinel.response.v1"
LINK_SENTINEL_SUMMARY_SCHEMA = "matrix.link.sentinel.summary.v1"
LINK_SENTINEL_TRACE_SCHEMA = "matrix.link.sentinel.trace.v1"
LINK_EXTERNAL_DRIVER_REQUEST_SCHEMA = LINK_SENTINEL_REQUEST_SCHEMA
LINK_EXTERNAL_DRIVER_RESPONSE_SCHEMA = LINK_SENTINEL_RESPONSE_SCHEMA
LINK_EXTERNAL_DRIVER_SUMMARY_SCHEMA = LINK_SENTINEL_SUMMARY_SCHEMA
LINK_EXTERNAL_DRIVER_TRACE_SCHEMA = LINK_SENTINEL_TRACE_SCHEMA
_PROPERTIES = frozenset({"energy", "gradient", "hessian"})


@dataclass(frozen=True)
class ExternalDriverPoint:
    point_id: str
    q: np.ndarray
    evaluation_owner: str = "link"
    requested_properties: tuple[str, ...] = ("energy",)
    calculator_id: str = "link-default"

    def __post_init__(self) -> None:
        q = np.asarray(self.q, dtype=float).reshape(-1)
        if not np.all(np.isfinite(q)):
            raise ValueError("external-driver SONIC values contain non-finite entries")
        owner = str(self.evaluation_owner).strip().lower()
        if owner not in {"link", "driver"}:
            raise ValueError("evaluation_owner must be 'link' or 'driver'")
        properties = _normalize_properties(self.requested_properties)
        object.__setattr__(self, "point_id", str(self.point_id))
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "evaluation_owner", owner)
        object.__setattr__(self, "requested_properties", properties)
        calculator_id = str(self.calculator_id).strip()
        if not calculator_id:
            raise ValueError("calculator_id must be nonempty")
        object.__setattr__(self, "calculator_id", calculator_id)


@dataclass(frozen=True)
class ExternalDriverResult:
    status: str
    cycles: int
    point_count: int
    completed_point_count: int
    summary_path: Path
    trace_path: Path
    final_response_path: Path


def run_external_driver_loop(
    xyzin_path: Path | str,
    *,
    run_dir: Path | str,
    driver_command: str,
    coordinate_model: OptimizerCoordinateModel | None = None,
    coordinates: Sequence[str] = (),
    engine_command: str = "",
    backend: QMScanBackend | None = None,
    timeout: float | None = None,
    max_cycles: int = 100,
    batch_workers: int = 1,
    initial_evaluation_owner: str = "link",
    initial_properties: Sequence[str] = ("energy",),
    active_variable_contract: ActiveVariableContract | None = None,
    calculator_backends: Mapping[str, QMScanBackend] | None = None,
    calculator_engine_commands: Mapping[str, str] | None = None,
    resume: bool = False,
    run_id: str | None = None,
    retained_group: str = "C1",
    independent_candidates: bool = False,
    geometry_filter: Mapping[str, object] | None = None,
) -> ExternalDriverResult:
    """Run LINK as a SONIC/Cartesian service controlled by an external driver.

    LINK never proposes a step in this workflow.  On every cycle the external
    driver receives SONIC values, their realized Cartesian geometry and any
    available energy/gradient/Hessian data.  It returns one or more SONIC points
    for the next cycle.  Electronic properties may be evaluated either by LINK
    or by the driver itself on a point-by-point basis.
    """

    if not str(driver_command).strip():
        raise ValueError("external-driver workflow needs a driver command")
    if int(max_cycles) <= 0:
        raise ValueError("max_cycles must be positive")
    if int(batch_workers) <= 0:
        raise ValueError("batch_workers must be positive")
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "link_sentinel_checkpoint.json"
    if checkpoint_path.exists() and not resume:
        raise FileExistsError(
            f"LINK-SENTINEL run already exists at {root}; use resume=True or a new run directory"
        )
    if independent_candidates and str(retained_group).strip().upper() != "C1":
        raise ValueError("independent-candidate policy requires retained_group C1")
    exploration_policy = (
        PESExplorationPolicy.monte_carlo()
        if independent_candidates
        else PESExplorationPolicy(retained_group=retained_group)
    )
    model = coordinate_model
    if model is None:
        model = coordinate_model_from_xyzin(
            xyzin_path,
            kind="sonic",
            coordinates=tuple(coordinates),
            pes_exploration=True,
            retained_group=exploration_policy.retained_group,
        )
    elif not model.pes_exploration and model.sonic_from_coordinates is None:
        model = coordinate_model_from_xyzin(
            xyzin_path,
            kind="sonic",
            coordinates=model.labels,
            metric_diagonal=model.metric_diagonal,
            pes_exploration=True,
            retained_group=exploration_policy.retained_group,
        )
    elif not model.pes_exploration:
        raise ValueError(
            "SENTINEL active variables must be constructed in PES-exploration SONIC space"
        )
    if model.retained_group != exploration_policy.retained_group:
        raise ValueError(
            "SENTINEL coordinate model retained_group does not match the run policy"
        )
    if model.kind != "sonic":
        raise ValueError("external-driver workflow requires a frozen SONIC coordinate model")
    if active_variable_contract is not None and active_variable_contract.model is not model:
        raise ValueError("active-variable contract and coordinate model must be the same object")
    service = GeometryEvaluationService(
        xyzin_path=xyzin_path,
        run_dir=root / "link_evaluations",
        coordinate_model=model,
        engine_command=engine_command,
        backend=backend,
        timeout=timeout,
        settings=OptimizerSettings(resume=False),
        pes_exploration_policy=exploration_policy,
    )
    q0 = np.zeros(len(model.labels), dtype=float)
    service.initialize_coordinate_projector(q0, service.reference_coordinates)
    services = {"link-default": service}
    profile_ids = set(calculator_backends or {}).union(calculator_engine_commands or {})
    for calculator_id in profile_ids:
        normalized_id = str(calculator_id).strip()
        if not normalized_id or normalized_id == "link-default":
            raise ValueError("calculator profile IDs must be nonempty and not 'link-default'")
        calculator_backend = (calculator_backends or {}).get(calculator_id)
        calculator_command = (calculator_engine_commands or {}).get(calculator_id, "")
        if calculator_backend is not None and calculator_command:
            raise ValueError(f"calculator profile {normalized_id} has two execution modes")
        calculator_service = GeometryEvaluationService(
            xyzin_path=xyzin_path,
            run_dir=root / "link_evaluations" / normalized_id,
            coordinate_model=model,
            engine_command=calculator_command,
            backend=calculator_backend,
            timeout=timeout,
            settings=OptimizerSettings(resume=False),
            pes_exploration_policy=exploration_policy,
        )
        calculator_service.initialize_coordinate_projector(
            q0, calculator_service.reference_coordinates
        )
        services[normalized_id] = calculator_service
    reference_values = service.coordinate_reference_values()
    sonic_reference_values = service.sonic_reference_values()
    sonic_labels = service.sonic_contract_labels()
    variable_contract_payload = (
        active_variable_contract.protocol_payload()
        if active_variable_contract is not None
        else _direct_variable_contract_payload(model, sonic_labels, reference_values)
    )
    coordinate_contract = {
        "pes_exploration": exploration_policy.protocol_payload(),
        "active_variables": variable_contract_payload,
        "sonic": {
            "labels": list(sonic_labels),
            "reference_values": sonic_reference_values.tolist(),
            "value_semantics": "absolute SONIC values",
            "displacement_semantics": "relative to the frozen reference contract",
        },
    }
    coordinate_digest = contract_digest(coordinate_contract)
    trace_path = root / "external_driver_trace.jsonl"
    if resume:
        checkpoint = _read_checkpoint(checkpoint_path, coordinate_digest)
        pending = tuple(_point_from_checkpoint(item) for item in checkpoint["pending_points"])
        start_cycle = int(checkpoint["next_cycle"])
        point_count = int(checkpoint["point_count"])
        completed_count = int(checkpoint["completed_point_count"])
        seen_point_ids = set(str(item) for item in checkpoint["seen_point_ids"])
        current_run_id = str(checkpoint["run_id"])
        if checkpoint.get("status") == "complete":
            raise ValueError("LINK-SENTINEL run is already complete")
    else:
        current_run_id = str(run_id or uuid.uuid4())
        pending = (
            ExternalDriverPoint(
                point_id="reference",
                q=q0,
                evaluation_owner=initial_evaluation_owner,
                requested_properties=tuple(initial_properties),
                calculator_id=(
                    "sentinel" if initial_evaluation_owner == "driver" else "link-default"
                ),
            ),
        )
        start_cycle = 0
        point_count = 0
        completed_count = 0
        seen_point_ids: set[str] = set()
        trace_path.write_text("", encoding="utf-8")
        _write_checkpoint(
            checkpoint_path,
            run_id=current_run_id,
            coordinate_digest=coordinate_digest,
            next_cycle=0,
            pending=pending,
            point_count=0,
            completed_count=0,
            seen_point_ids=seen_point_ids,
            status="running",
        )
    final_response_path = root / "cycle_0000_response.json"
    final_status = "max_cycles"

    for cycle in range(start_cycle, int(max_cycles)):
        duplicate = seen_point_ids.intersection(point.point_id for point in pending)
        if duplicate:
            raise ValueError(f"SENTINEL reused point_id across cycles: {sorted(duplicate)[0]}")
        seen_point_ids.update(point.point_id for point in pending)
        cycle_dir = root / f"cycle_{cycle:04d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        records = _realize_and_evaluate_points(
            pending,
            service=service,
            services=services,
            reference_values=reference_values,
            cycle=cycle,
            batch_workers=int(batch_workers),
            geometry_filter=geometry_filter,
        )
        point_count += len(records)
        request_path = cycle_dir / "request.json"
        response_path = cycle_dir / "response.json"
        request_payload = {
            "schema": LINK_EXTERNAL_DRIVER_REQUEST_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "sender": "LINK",
            "receiver": "SENTINEL",
            "run_id": current_run_id,
            "transaction_id": f"{current_run_id}:{cycle:04d}",
            "cycle": cycle,
            "coordinate_contract_sha256": coordinate_digest,
            "coordinate_contract": coordinate_contract,
            "restart": {"resumed": bool(resume), "attempt": 1},
            "capabilities": {
                "batch": True,
                "partial_optimization": True,
                "scans": ["one-dimensional", "multidimensional"],
                "link_calculators": sorted(services),
                "driver_owned_calculators": True,
                "pointwise_oracle_symmetry": True,
            },
            "units": {
                "cartesian_coordinates": "angstrom",
                "energy": "hartree",
                "cartesian_gradient": "hartree/bohr",
                "cartesian_hessian": "hartree/bohr^2",
                "sonic_gradient": "hartree/SONIC-unit",
                "sonic_hessian": "hartree/SONIC-unit^2",
            },
            "points": records,
        }
        validate_request(request_payload)
        _write_json(request_path, request_payload)
        try:
            _invoke_driver(
                driver_command,
                request_path=request_path,
                response_path=response_path,
                run_dir=root,
                cycle=cycle,
                timeout=timeout,
            )
            response = _read_driver_response(response_path, request=request_payload)
            records = _merge_driver_evaluations(
                records, response.get("evaluations", ()), service
            )
        except Exception as exc:
            _write_failure(root, current_run_id, cycle, exc)
            raise
        completed_count += sum(
            record.get("properties", {}).get("energy_hartree") is not None
            for record in records
        )
        completed_path = cycle_dir / "completed_exchange.json"
        _write_json(
            completed_path,
            {
                "schema": LINK_EXTERNAL_DRIVER_TRACE_SCHEMA,
                "cycle": cycle,
                "points": records,
                "driver_response": response,
            },
        )
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "schema": LINK_EXTERNAL_DRIVER_TRACE_SCHEMA,
                        "cycle": cycle,
                        "request": str(request_path),
                        "response": str(response_path),
                        "completed_exchange": str(completed_path),
                        "point_ids": [record["point_id"] for record in records],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        final_response_path = response_path
        final_status = str(response.get("status", "continue")).strip().lower()
        next_payload = response.get("next_points", ())
        if final_status == "error":
            errors = response.get("errors", ())
            exc = RuntimeError(f"SENTINEL returned an error response: {errors}")
            _write_failure(root, current_run_id, cycle, exc)
            raise exc
        if final_status in {"complete", "completed", "stop"}:
            _write_checkpoint(
                checkpoint_path,
                run_id=current_run_id,
                coordinate_digest=coordinate_digest,
                next_cycle=cycle + 1,
                pending=(),
                point_count=point_count,
                completed_count=completed_count,
                seen_point_ids=seen_point_ids,
                status="complete",
            )
            break
        pending = _parse_next_points(
            next_payload,
            variable_labels=model.labels,
            variable_reference_values=reference_values,
            sonic_labels=sonic_labels,
            sonic_reference_values=sonic_reference_values,
            sonic_from_variables=model.sonic_from_coordinates,
            allowed_link_calculators=frozenset(services),
        )
        if not pending:
            raise ValueError("driver returned no next_points without completing the workflow")
        reused = seen_point_ids.intersection(point.point_id for point in pending)
        if reused:
            raise ValueError(f"SENTINEL reused point_id across cycles: {sorted(reused)[0]}")
        _write_checkpoint(
            checkpoint_path,
            run_id=current_run_id,
            coordinate_digest=coordinate_digest,
            next_cycle=cycle + 1,
            pending=pending,
            point_count=point_count,
            completed_count=completed_count,
            seen_point_ids=seen_point_ids,
            status="running",
        )
    else:
        cycle = int(max_cycles) - 1

    summary_path = root / "external_driver_summary.json"
    _write_json(
        summary_path,
        {
            "schema": LINK_EXTERNAL_DRIVER_SUMMARY_SCHEMA,
            "status": final_status,
            "run_id": current_run_id,
            "coordinate_contract_sha256": coordinate_digest,
            "cycles": cycle + 1,
            "point_count": point_count,
            "completed_point_count": completed_count,
            "coordinate_labels": list(model.labels),
            "pes_exploration": exploration_policy.protocol_payload(),
            "trace_path": str(trace_path),
            "final_response_path": str(final_response_path),
        },
    )
    return ExternalDriverResult(
        status=final_status,
        cycles=cycle + 1,
        point_count=point_count,
        completed_point_count=completed_count,
        summary_path=summary_path,
        trace_path=trace_path,
        final_response_path=final_response_path,
    )


def _realize_and_evaluate_points(
    points: Sequence[ExternalDriverPoint],
    *,
    service: GeometryEvaluationService,
    services: Mapping[str, GeometryEvaluationService],
    reference_values: np.ndarray,
    cycle: int,
    batch_workers: int,
    geometry_filter: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    calculator_ids = {point.calculator_id for point in points}
    property_sets = {point.requested_properties for point in points}
    if (
        len(points) > 1
        and all(point.evaluation_owner == "link" for point in points)
        and len(calculator_ids) == 1
        and len(property_sets) == 1
        and geometry_filter is None
    ):
        calculator_id = next(iter(calculator_ids))
        evaluation_service = services.get(calculator_id)
        if evaluation_service is None:
            raise ValueError(f"unknown LINK calculator_id: {calculator_id}")
        if evaluation_service.supports_resident_potential_batch():
            for point in points:
                if point.q.shape != reference_values.shape:
                    raise ValueError(
                        "driver SONIC vector length does not match the frozen contract"
                    )
            properties = next(iter(property_sets))
            evaluations = evaluation_service.evaluate_resident_potential_batch(
                tuple(point.q for point in points),
                tags=tuple(
                    f"external-driver-{cycle}-{point.point_id}" for point in points
                ),
                requested_properties=properties,
                persist_cache=False,
            )
            records: list[dict[str, object]] = []
            for point, evaluation in zip(points, evaluations, strict=True):
                _require_properties(evaluation.result, properties, point.point_id)
                records.append(
                    _point_record(
                        point,
                        q=evaluation.q,
                        reference_values=reference_values,
                        coordinates=evaluation.coordinates_angstrom,
                        atoms=service.atoms,
                        result=evaluation.result,
                        service=evaluation_service,
                    )
                )
            return records

    def realize(point: ExternalDriverPoint) -> dict[str, object]:
        if point.q.shape != reference_values.shape:
            raise ValueError("driver SONIC vector length does not match the frozen contract")
        if point.evaluation_owner == "link":
            evaluation_service = services.get(point.calculator_id)
            if evaluation_service is None:
                raise ValueError(f"unknown LINK calculator_id: {point.calculator_id}")
            if evaluation_service.backend is None and not evaluation_service.engine_command.strip():
                raise ValueError("LINK-owned evaluation needs --backend or --engine-command")
            coordinates = evaluation_service.coordinates_from_q(point.q)
            rejection = geometry_filter_rejection(
                evaluation_service,
                coordinates,
                geometry_filter,
            )
            if rejection is None:
                evaluation = evaluation_service.evaluate(
                    point.q,
                    tag=f"external-driver-{cycle}-{point.point_id}",
                    use_cache=True,
                    requested_properties=point.requested_properties,
                    realized_coordinates_angstrom=coordinates,
                )
                coordinates = evaluation.coordinates_angstrom
                realized_q = evaluation.q
                result = evaluation.result
            else:
                if set(point.requested_properties) - {"energy"}:
                    raise ValueError(
                        "geometry prefilters support energy-only candidate evaluation"
                    )
                realized_q = point.q.copy()
                penalty = float(
                    (geometry_filter or {}).get("penalty_energy_hartree", 1.0e3)
                )
                result = PointEvaluationResult(
                    point_index=-1,
                    displacement=0.0,
                    energy_hartree=penalty,
                    backend_coordinates_angstrom=coordinates,
                    status="completed",
                    source="LINK geometry prefilter",
                    message=rejection,
                    execution={
                        "energy_evaluations": 0,
                        "gradient_evaluations": 0,
                        "geometry_filter_rejection": rejection,
                    },
                )
            _require_properties(result, point.requested_properties, point.point_id)
            record_service = evaluation_service
        else:
            coordinates = service.coordinates_from_q(point.q)
            realized_q = service.actual_q(coordinates)
            result = None
            record_service = service
        return _point_record(
            point,
            q=realized_q,
            reference_values=reference_values,
            coordinates=coordinates,
            atoms=service.atoms,
            result=result,
            service=record_service,
        )

    if batch_workers == 1 or len(points) <= 1:
        return [realize(point) for point in points]
    with ThreadPoolExecutor(max_workers=batch_workers) as pool:
        return list(pool.map(realize, points))


def geometry_filter_rejection(
    service: GeometryEvaluationService,
    coordinates_angstrom: np.ndarray,
    config: Mapping[str, object] | None,
) -> str | None:
    """Return the native MATRIX connectivity/clash rejection, if any.

    The service argument is intentionally structural: owner bridges may reuse
    this exact LINK filter for rigid-fragment candidates without duplicating
    collision criteria.
    """
    if config is None:
        return None
    from matrix_chem import topology_bonds_from_xyzin
    from matrix_chem.topology.covalent_radii import covalent_radius
    from matrix_chem.topology.elements import atomic_number

    coords = np.asarray(coordinates_angstrom, dtype=float)
    reference = np.asarray(service.reference_coordinates, dtype=float)
    bonds = tuple(
        (left - 1, right - 1)
        for left, right in topology_bonds_from_xyzin(service.xyzin_path)
    )
    if bool(config.get("preserve_connectivity", True)):
        minimum_ratio = float(config.get("minimum_bond_ratio", 0.65))
        maximum_ratio = float(config.get("maximum_bond_ratio", 1.45))
        for left, right in bonds:
            reference_distance = float(np.linalg.norm(reference[left] - reference[right]))
            distance = float(np.linalg.norm(coords[left] - coords[right]))
            ratio = distance / max(reference_distance, 1.0e-12)
            if not minimum_ratio <= ratio <= maximum_ratio:
                return (
                    f"bond {left + 1}-{right + 1} changed to "
                    f"{ratio:.3f} times its reference length"
                )
    bonded = {tuple(sorted(pair)) for pair in bonds}
    adjacency = [set() for _ in service.atoms]
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    excluded = set(bonded)
    for center, neighbors in enumerate(adjacency):
        for left in neighbors:
            for right in neighbors:
                if left < right:
                    excluded.add((left, right))
    scale = float(config.get("minimum_nonbonded_covalent_scale", 0.65))
    numbers = tuple(int(atomic_number(atom) or 0) for atom in service.atoms)
    for left in range(len(coords)):
        for right in range(left + 1, len(coords)):
            if (left, right) in excluded:
                continue
            radius_left = float(covalent_radius(numbers[left]) or 0.75)
            radius_right = float(covalent_radius(numbers[right]) or 0.75)
            minimum = scale * (radius_left + radius_right)
            distance = float(np.linalg.norm(coords[left] - coords[right]))
            if distance < minimum:
                return (
                    f"nonbonded collision {left + 1}-{right + 1}: "
                    f"{distance:.3f} < {minimum:.3f} angstrom"
                )
    return None


# Compatibility for callers that used the former module-private spelling.
_geometry_filter_rejection = geometry_filter_rejection


def _point_record(
    point: ExternalDriverPoint,
    *,
    q: np.ndarray,
    reference_values: np.ndarray,
    coordinates: np.ndarray,
    atoms: Sequence[str],
    result: PointEvaluationResult | None,
    service: GeometryEvaluationService,
) -> dict[str, object]:
    coords = np.asarray(coordinates, dtype=float)
    properties = _properties_payload(result, coordinates=coords, service=service)
    _projected, symmetry = prepare_pes_exploration_geometry(
        service.xyzin_path,
        atoms,
        coords,
        policy=service.pes_exploration_policy or PESExplorationPolicy(),
    )
    return {
        "point_id": point.point_id,
        "status": "completed" if result is not None else "geometry_ready",
        "evaluation_owner": point.evaluation_owner,
        "calculator_id": point.calculator_id,
        "requested_properties": list(point.requested_properties),
        "execution": {} if result is None else dict(result.execution),
        "active_variables": {
            "labels": list(service.coordinate_model.labels),
            "values": (reference_values + np.asarray(q, dtype=float)).tolist(),
            "displacements": np.asarray(q, dtype=float).tolist(),
        },
        "sonic": {
            "labels": list(service.sonic_contract_labels()),
            "values": service.sonic_values_from_q(q).tolist(),
            "displacements": (
                service.sonic_values_from_q(q) - service.sonic_reference_values()
            ).tolist(),
        },
        "cartesian": {
            "atoms": list(atoms),
            "coordinates_angstrom": coords.tolist(),
        },
        "point_symmetry": {
            "point_group": symmetry.point_group,
            "operation_count": symmetry.operation_count,
            "projection_status": symmetry.projection_status,
            "projection_max_displacement_angstrom": (
                symmetry.projection_max_displacement_angstrom
            ),
            "projection_rms_displacement_angstrom": (
                symmetry.projection_rms_displacement_angstrom
            ),
        },
        "properties": properties,
    }


def _properties_payload(
    result: PointEvaluationResult | None,
    *,
    coordinates: np.ndarray,
    service: GeometryEvaluationService,
) -> dict[str, object]:
    if result is None:
        return {
            "energy_hartree": None,
            "cartesian_gradient_hartree_per_bohr": None,
            "cartesian_hessian_hartree_per_bohr2": None,
            "sonic_gradient_hartree": None,
            "sonic_hessian_hartree": None,
            "active_variable_gradient_hartree": None,
            "active_variable_hessian_hartree": None,
            "sonic_hessian_semantics": "linearized congruence; gradient-curvature term omitted",
        }
    gradient = result.gradient_hartree_per_bohr
    hessian = result.hessian_hartree_per_bohr2
    variable_directions_bohr = service.coordinate_directions(coordinates) * ANGSTROM_TO_BOHR
    sonic_directions_bohr = service.sonic_coordinate_directions(coordinates) * ANGSTROM_TO_BOHR
    sonic_gradient = None
    sonic_hessian = None
    variable_gradient = None
    variable_hessian = None
    if gradient is not None:
        cartesian_gradient = np.asarray(gradient, dtype=float).reshape(-1)
        sonic_gradient = sonic_directions_bohr @ cartesian_gradient
        variable_gradient = variable_directions_bohr @ cartesian_gradient
    if hessian is not None:
        matrix = np.asarray(hessian, dtype=float)
        sonic_hessian = sonic_directions_bohr @ matrix @ sonic_directions_bohr.T
        sonic_hessian = 0.5 * (sonic_hessian + sonic_hessian.T)
        variable_hessian = variable_directions_bohr @ matrix @ variable_directions_bohr.T
        variable_hessian = 0.5 * (variable_hessian + variable_hessian.T)
    return {
        "energy_hartree": result.energy_hartree,
        "cartesian_gradient_hartree_per_bohr": (
            None if gradient is None else np.asarray(gradient, dtype=float).reshape(-1).tolist()
        ),
        "cartesian_hessian_hartree_per_bohr2": (
            None if hessian is None else np.asarray(hessian, dtype=float).tolist()
        ),
        "sonic_gradient_hartree": (
            None if sonic_gradient is None else np.asarray(sonic_gradient).tolist()
        ),
        "sonic_hessian_hartree": (
            None if sonic_hessian is None else np.asarray(sonic_hessian).tolist()
        ),
        "active_variable_gradient_hartree": (
            None if variable_gradient is None else np.asarray(variable_gradient).tolist()
        ),
        "active_variable_hessian_hartree": (
            None if variable_hessian is None else np.asarray(variable_hessian).tolist()
        ),
        "sonic_hessian_semantics": "linearized congruence; gradient-curvature term omitted",
    }


def _merge_driver_evaluations(
    records: Sequence[dict[str, object]],
    evaluations: object,
    service: GeometryEvaluationService,
) -> list[dict[str, object]]:
    if evaluations is None:
        evaluations = ()
    if not isinstance(evaluations, (list, tuple)):
        raise ValueError("driver evaluations must be a list")
    by_id: dict[str, dict[str, object]] = {}
    record_ids = {str(record["point_id"]) for record in records}
    for raw in evaluations:
        if not isinstance(raw, dict):
            raise ValueError("each driver evaluation must be an object")
        point_id = str(raw.get("point_id", ""))
        if not point_id:
            raise ValueError("driver evaluation is missing point_id")
        if point_id in by_id:
            raise ValueError(f"duplicate driver evaluation for point_id: {point_id}")
        if point_id not in record_ids:
            raise ValueError(f"driver evaluation refers to unknown point_id: {point_id}")
        by_id[point_id] = raw
    merged: list[dict[str, object]] = []
    for original in records:
        record = dict(original)
        if record.get("evaluation_owner") == "driver":
            payload = by_id.get(str(record["point_id"]))
            if payload is None:
                raise ValueError(f"driver did not return evaluation for {record['point_id']}")
            cartesian = record["cartesian"]
            assert isinstance(cartesian, dict)
            coordinates = np.asarray(cartesian["coordinates_angstrom"], dtype=float)
            result = _driver_point_result(
                payload,
                point_id=str(record["point_id"]),
                ncart=int(coordinates.size),
            )
            _require_properties(
                result,
                tuple(str(item) for item in record.get("requested_properties", ())),
                str(record["point_id"]),
            )
            record["properties"] = _properties_payload(
                result,
                coordinates=coordinates,
                service=service,
            )
            record["status"] = "completed"
        elif str(record["point_id"]) in by_id:
            raise ValueError(
                f"driver returned an evaluation for LINK-owned point {record['point_id']}"
            )
        merged.append(record)
    return merged


def _driver_point_result(
    payload: dict[str, object],
    *,
    point_id: str,
    ncart: int,
) -> PointEvaluationResult:
    result = PointEvaluationResult(
        point_index=0,
        displacement=0.0,
        energy_hartree=_optional_float(payload.get("energy_hartree")),
        gradient_hartree_per_bohr=_optional_array(
            payload.get("cartesian_gradient_hartree_per_bohr")
        ),
        cm5_charges=_optional_array(payload.get("cm5_charges")),
        hessian_hartree_per_bohr2=_optional_array(
            payload.get("cartesian_hessian_hartree_per_bohr2")
        ),
        status=str(payload.get("status", "completed")),
        message=str(payload.get("message", "")),
        source=f"external-driver:{point_id}",
    )
    if result.energy_hartree is not None and not np.isfinite(result.energy_hartree):
        raise ValueError(f"driver energy for {point_id} is not finite")
    if result.gradient_hartree_per_bohr is not None:
        gradient = np.asarray(result.gradient_hartree_per_bohr, dtype=float).reshape(-1)
        if gradient.shape != (ncart,):
            raise ValueError(
                f"driver Cartesian gradient for {point_id} must have length {ncart}"
            )
    if result.hessian_hartree_per_bohr2 is not None:
        hessian = np.asarray(result.hessian_hartree_per_bohr2, dtype=float)
        if hessian.shape != (ncart, ncart):
            raise ValueError(
                f"driver Cartesian Hessian for {point_id} must have shape {(ncart, ncart)}"
            )
    return result


def _parse_next_points(
    payload: object,
    *,
    variable_labels: Sequence[str],
    variable_reference_values: np.ndarray,
    sonic_labels: Sequence[str],
    sonic_reference_values: np.ndarray,
    sonic_from_variables: np.ndarray | None,
    allowed_link_calculators: frozenset[str] = frozenset({"link-default"}),
) -> tuple[ExternalDriverPoint, ...]:
    if not isinstance(payload, (list, tuple)):
        raise ValueError("driver next_points must be a list")
    points: list[ExternalDriverPoint] = []
    identifiers: set[str] = set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError("each next point must be an object")
        point_id = str(raw.get("point_id", f"point-{index}"))
        if point_id in identifiers:
            raise ValueError(f"duplicate driver point_id: {point_id}")
        identifiers.add(point_id)
        evaluation_owner = str(raw.get("evaluation_owner", "link"))
        calculator_id = str(
            raw.get(
                "calculator_id",
                "sentinel" if evaluation_owner == "driver" else "link-default",
            )
        )
        if evaluation_owner == "link" and calculator_id not in allowed_link_calculators:
            raise ValueError(
                f"SENTINEL requested unadvertised LINK calculator_id: {calculator_id}"
            )
        supplied_labels = raw.get("labels")
        if supplied_labels is not None and tuple(str(item) for item in supplied_labels) != tuple(
            variable_labels
        ):
            raise ValueError("SENTINEL variable labels do not match the frozen LINK contract")
        if "variable_values" in raw:
            q = (
                np.asarray(raw["variable_values"], dtype=float).reshape(-1)
                - variable_reference_values
            )
        elif "variable_displacements" in raw:
            q = np.asarray(raw["variable_displacements"], dtype=float).reshape(-1)
        elif "sonic_values" in raw:
            sonic_displacement = (
                np.asarray(raw["sonic_values"], dtype=float).reshape(-1)
                - sonic_reference_values
            )
            q = _variables_from_sonic_displacement(sonic_displacement, sonic_from_variables)
        elif "sonic_displacements" in raw:
            q = _variables_from_sonic_displacement(
                np.asarray(raw["sonic_displacements"], dtype=float).reshape(-1),
                sonic_from_variables,
            )
        else:
            variables = raw.get("active_variables")
            sonic = raw.get("sonic")
            if isinstance(variables, dict) and "values" in variables:
                q = np.asarray(variables["values"], dtype=float).reshape(-1) - variable_reference_values
            elif isinstance(variables, dict) and "displacements" in variables:
                q = np.asarray(variables["displacements"], dtype=float).reshape(-1)
            elif isinstance(sonic, dict) and "values" in sonic:
                if tuple(str(item) for item in sonic.get("labels", sonic_labels)) != tuple(
                    sonic_labels
                ):
                    raise ValueError("SENTINEL SONIC labels do not match the LINK contract")
                q = _variables_from_sonic_displacement(
                    np.asarray(sonic["values"], dtype=float).reshape(-1)
                    - sonic_reference_values,
                    sonic_from_variables,
                )
            elif isinstance(sonic, dict) and "displacements" in sonic:
                q = _variables_from_sonic_displacement(
                    np.asarray(sonic["displacements"], dtype=float).reshape(-1),
                    sonic_from_variables,
                )
            else:
                raise ValueError(
                    "next point needs variable_values/variable_displacements or a SONIC vector"
                )
        points.append(
            ExternalDriverPoint(
                point_id=point_id,
                q=q,
                evaluation_owner=evaluation_owner,
                requested_properties=tuple(raw.get("requested_properties", ("energy",))),
                calculator_id=calculator_id,
            )
        )
    return tuple(points)


def _variables_from_sonic_displacement(
    displacement: np.ndarray,
    transform: np.ndarray | None,
) -> np.ndarray:
    sonic = np.asarray(displacement, dtype=float).reshape(-1)
    if transform is None:
        return sonic
    matrix = np.asarray(transform, dtype=float)
    if sonic.shape != (matrix.shape[0],):
        raise ValueError("SENTINEL SONIC vector length does not match the frozen contract")
    variables = np.linalg.pinv(matrix, rcond=1.0e-10) @ sonic
    residual = float(np.linalg.norm(matrix @ variables - sonic))
    if residual > 1.0e-8 * max(1.0, float(np.linalg.norm(sonic))):
        raise ValueError("SENTINEL SONIC point lies outside the active-variable subspace")
    return variables


def _invoke_driver(
    command_template: str,
    *,
    request_path: Path,
    response_path: Path,
    run_dir: Path,
    cycle: int,
    timeout: float | None,
) -> None:
    mapping = {
        "request": str(request_path),
        "response": str(response_path),
        "run_dir": str(run_dir),
        "cycle": str(cycle),
    }
    command = [part.format(**mapping) for part in shlex.split(command_template)]
    require_authorized_descendant_calculation(
        backend="LINK/external-driver",
        input_path=request_path,
        command=command,
        workdir=request_path.parent,
    )
    completed = subprocess.run(
        command,
        cwd=request_path.parent,
        check=False,
        timeout=timeout,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"external LINK driver failed in cycle {cycle}: {completed.stdout.strip()}"
        )
    if not response_path.is_file():
        raise RuntimeError(f"external LINK driver did not write {response_path}")


def _read_driver_response(
    path: Path, *, request: dict[str, object] | None = None
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("driver response must be a JSON object")
    if payload.get("schema") != LINK_EXTERNAL_DRIVER_RESPONSE_SCHEMA:
        raise ValueError(f"unsupported external-driver response schema: {payload.get('schema')}")
    validate_response(payload, request=request)
    return payload


def _direct_variable_contract_payload(
    model: OptimizerCoordinateModel,
    sonic_labels: Sequence[str],
    reference_values: np.ndarray,
) -> dict[str, object]:
    definition = model.sonic_definition
    return {
        "schema": LINK_ACTIVE_VARIABLES_SCHEMA,
        "source": "LINK command line",
        "variable_labels": list(model.labels),
        "reference_values": np.asarray(reference_values, dtype=float).tolist(),
        "variables": [
            {
                "name": name,
                "kind": "sonic",
                "coordinate": sonic,
                "reference_value": float(reference),
                "units": "SONIC-unit",
                **(
                    gic_metadata_for_contract(definition, sonic)
                    if definition is not None
                    else {}
                ),
            }
            for name, sonic, reference in zip(model.labels, sonic_labels, reference_values)
        ],
        "sonic_labels": list(sonic_labels),
        "sonic_from_variable_displacements": np.eye(len(model.labels), dtype=float).tolist(),
        "mapping": "delta_q_SONIC = sonic_from_variable_displacements @ delta_variables",
        "projection_residuals": [0.0] * len(model.labels),
        "frozen_policy": "all SONIC directions outside the mapped active subspace",
    }


def _require_properties(
    result: PointEvaluationResult,
    properties: Sequence[str],
    point_id: str,
) -> None:
    missing = []
    for name in _normalize_properties(properties):
        value = {
            "energy": result.energy_hartree,
            "gradient": result.gradient_hartree_per_bohr,
            "hessian": result.hessian_hartree_per_bohr2,
        }[name]
        if value is None:
            missing.append(name)
    if missing:
        raise ValueError(f"point {point_id} is missing requested properties: {', '.join(missing)}")


def _normalize_properties(properties: Sequence[str]) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(str(item).strip().lower() for item in properties))
    if not result:
        result = ("energy",)
    unknown = set(result) - _PROPERTIES
    if unknown:
        raise ValueError(f"unsupported requested properties: {', '.join(sorted(unknown))}")
    return result


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_array(value: object) -> np.ndarray | None:
    return None if value is None else np.asarray(value, dtype=float)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    atomic_json_write(path, payload, allow_nan=False)
    return path


def _point_checkpoint_payload(point: ExternalDriverPoint) -> dict[str, object]:
    return {
        "point_id": point.point_id,
        "q": point.q.tolist(),
        "evaluation_owner": point.evaluation_owner,
        "requested_properties": list(point.requested_properties),
        "calculator_id": point.calculator_id,
    }


def _point_from_checkpoint(payload: object) -> ExternalDriverPoint:
    if not isinstance(payload, dict):
        raise ValueError("invalid pending point in LINK-SENTINEL checkpoint")
    return ExternalDriverPoint(
        point_id=str(payload["point_id"]),
        q=np.asarray(payload["q"], dtype=float),
        evaluation_owner=str(payload["evaluation_owner"]),
        requested_properties=tuple(str(item) for item in payload["requested_properties"]),
        calculator_id=str(payload.get("calculator_id", "link-default")),
    )


def _write_checkpoint(
    path: Path,
    *,
    run_id: str,
    coordinate_digest: str,
    next_cycle: int,
    pending: Sequence[ExternalDriverPoint],
    point_count: int,
    completed_count: int,
    seen_point_ids: set[str],
    status: str,
) -> None:
    _write_json(
        path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "run_id": run_id,
            "coordinate_contract_sha256": coordinate_digest,
            "next_cycle": next_cycle,
            "pending_points": [_point_checkpoint_payload(point) for point in pending],
            "point_count": point_count,
            "completed_point_count": completed_count,
            "seen_point_ids": sorted(seen_point_ids),
            "status": status,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def _read_checkpoint(path: Path, coordinate_digest: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"no LINK-SENTINEL checkpoint found at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported LINK-SENTINEL checkpoint")
    if payload.get("coordinate_contract_sha256") != coordinate_digest:
        raise ValueError("checkpoint belongs to a different coordinate contract")
    return payload


def _write_failure(root: Path, run_id: str, cycle: int, exc: Exception) -> None:
    _write_json(
        root / "last_error.json",
        {
            "schema": ERROR_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "run_id": run_id,
            "cycle": cycle,
            "status": "error",
            "retryable": isinstance(exc, (subprocess.TimeoutExpired, TimeoutError)),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
