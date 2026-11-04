"""Canonical multilevel geometry/Hessian initialization for LINK.

This module composes the existing ORACLE, SMITH and LINK owner APIs.  It does
not contain an alternative optimizer, Hessian transform or backend runner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

from matrix_chem import read_xyzin_geometry, write_xyz
from .initialization_protocol import InitializationProtocol, load_initialization_protocol
from .optimizer import (
    OptimizerResult,
    OptimizerSettings,
    build_optimizer_hessian_seed,
    coordinate_model_from_xyzin,
    optimize_geometry,
    optimizer_hessian_from_force_field_cartesian,
    write_optimizer_hessian,
)
from .scan import QMScanBackend, read_point_result


INITIALIZATION_WORKFLOW_REQUEST_SCHEMA = "matrix.link.initialization_workflow_request.v1"
INITIALIZATION_WORKFLOW_RESULT_SCHEMA = "matrix.link.initialization_workflow_result.v1"


@dataclass(frozen=True)
class InitializationWorkflowRequest:
    final_backend: QMScanBackend
    method_class: str
    final_engine_command: str = ""
    source_kind: str = "auto"
    geometry_quality_override: str = "auto"
    xtb_available: bool = True
    xtb_executable: str | None = None
    hf_sto3g_backend: str = "auto"
    coordinate_kind: str = "sonic"
    chart_lifecycle: bool = False
    coordinates: tuple[str, ...] = ()
    variables: Path | None = None
    optimizer_settings: OptimizerSettings = field(default_factory=OptimizerSettings)
    timeout: float | None = None
    schema: str = INITIALIZATION_WORKFLOW_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != INITIALIZATION_WORKFLOW_REQUEST_SCHEMA:
            raise ValueError(f"unsupported initialization request schema: {self.schema}")
        if self.method_class not in {"low_cost", "higher_level"}:
            raise ValueError("initialization method_class must be low_cost or higher_level")
        if self.coordinate_kind != "sonic":
            raise ValueError(
                "the frozen initialization protocol requires the shared SMITH SONIC "
                "coordinate contract"
            )
        if self.chart_lifecycle and (self.coordinates or self.variables is not None):
            raise ValueError("chart lifecycle requires the complete SONIC coordinate set")
        object.__setattr__(self, "coordinates", tuple(str(item) for item in self.coordinates))

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(
            {
                "schema": self.schema,
                "method_class": self.method_class,
                "final_engine_command": self.final_engine_command,
                "source_kind": self.source_kind,
                "geometry_quality_override": self.geometry_quality_override,
                "xtb_available": self.xtb_available,
                "xtb_executable": self.xtb_executable,
                "hf_sto3g_backend": self.hf_sto3g_backend,
                "coordinate_kind": self.coordinate_kind,
                "chart_lifecycle": self.chart_lifecycle,
                "coordinates": self.coordinates,
                "variables": self.variables,
                "timeout": self.timeout,
                "final_backend": asdict(self.final_backend),
                "optimizer_settings": asdict(self.optimizer_settings),
            }
        )


@dataclass(frozen=True)
class InitializationStageResult:
    index: int
    provider: str
    action: str
    input_xyzin: Path
    output_xyzin: Path
    run_dir: Path
    geometry_hash_before: str
    geometry_hash_after: str
    execution_backend: str = ""
    hessian_path: Path | None = None
    raw_hessian_path: Path | None = None
    optimizer_result: OptimizerResult | None = None

    def to_dict(self) -> dict[str, Any]:
        result = self.optimizer_result
        return {
            "index": self.index,
            "provider": self.provider,
            "action": self.action,
            "input_xyzin": str(self.input_xyzin),
            "output_xyzin": str(self.output_xyzin),
            "run_dir": str(self.run_dir),
            "geometry_hash_before": self.geometry_hash_before,
            "geometry_hash_after": self.geometry_hash_after,
            "execution_backend": self.execution_backend,
            "hessian_path": None if self.hessian_path is None else str(self.hessian_path),
            "raw_hessian_path": (
                None if self.raw_hessian_path is None else str(self.raw_hessian_path)
            ),
            "converged": None if result is None else bool(result.converged),
            "optimization_steps": None if result is None else len(result.iterations),
            "qm_evaluations": None if result is None else result.qm_evaluations,
            "optimizer_summary": None if result is None else str(result.summary_path),
        }


@dataclass(frozen=True)
class InitializationWorkflowResult:
    initial_xyzin: Path
    final_xyzin: Path
    manifest_path: Path
    geometry_quality: Mapping[str, Any]
    stages: tuple[InitializationStageResult, ...]
    converged: bool
    schema: str = INITIALIZATION_WORKFLOW_RESULT_SCHEMA


def initialization_request_from_dict(
    payload: Mapping[str, Any],
) -> InitializationWorkflowRequest:
    data = dict(payload)
    if data.get("schema") != INITIALIZATION_WORKFLOW_REQUEST_SCHEMA:
        raise ValueError("initialization request has an unsupported schema")
    backend_data = dict(data.get("final_backend", {}))
    if not backend_data.get("name"):
        raise ValueError("initialization request requires final_backend.name")
    for key in ("basis_file", "force_field", "zaff_library", "zaff_gaff_parameters"):
        if backend_data.get(key):
            backend_data[key] = Path(str(backend_data[key]))
        elif key in backend_data:
            backend_data[key] = None
    for key in ("extra_args", "oniom_high_atoms", "oniom_atom_types", "properties"):
        if key in backend_data:
            backend_data[key] = tuple(backend_data[key])
    for key in ("topology_bonds",):
        if key in backend_data:
            backend_data[key] = tuple(backend_data[key])
    if backend_data.get("gaussian_connectivity_bonds") is not None:
        backend_data["gaussian_connectivity_bonds"] = tuple(
            tuple(item) for item in backend_data["gaussian_connectivity_bonds"]
        )
    backend = QMScanBackend(**backend_data)
    settings_data = dict(data.get("optimizer_settings", {}))
    allowed_settings = {item.name for item in fields(OptimizerSettings)}
    unknown = sorted(set(settings_data) - allowed_settings)
    if unknown:
        raise ValueError("unknown optimizer settings: " + ", ".join(unknown))
    for key in ("fixed_atoms", "rigid_reference_groups"):
        if key in settings_data:
            settings_data[key] = tuple(settings_data[key])
    settings = OptimizerSettings(**settings_data)
    return InitializationWorkflowRequest(
        final_backend=backend,
        method_class=str(data["method_class"]),
        final_engine_command=str(data.get("final_engine_command", "")),
        source_kind=str(data.get("source_kind", "auto")),
        geometry_quality_override=str(data.get("geometry_quality_override", "auto")),
        xtb_available=_resolved_xtb_capability(data.get("xtb_available", "auto"))[0],
        xtb_executable=_resolved_xtb_executable(data),
        hf_sto3g_backend=str(data.get("hf_sto3g_backend", "auto")),
        coordinate_kind=str(data.get("coordinate_kind", "sonic")),
        chart_lifecycle=bool(data.get("chart_lifecycle", False)),
        coordinates=tuple(str(item) for item in data.get("coordinates", ())),
        variables=None if not data.get("variables") else Path(str(data["variables"])),
        optimizer_settings=settings,
        timeout=None if data.get("timeout") is None else float(data["timeout"]),
    )


def plan_initialization_stages(
    *,
    geometry_status: str,
    method_class: str,
    xtb_available: bool,
    protocol: InitializationProtocol | None = None,
) -> tuple[dict[str, str], ...]:
    """Return the exact stage sequence from the frozen package manifest."""

    selected = protocol or load_initialization_protocol()
    return selected.route(
        geometry_status=geometry_status,
        method_class=method_class,
        xtb_available=xtb_available,
    )


def run_initialization_workflow(
    xyzin_path: Path | str,
    *,
    run_dir: Path | str,
    request: InitializationWorkflowRequest,
    optimized_xyzin: Path | str | None = None,
) -> InitializationWorkflowResult:
    """Execute the frozen initialization chain without mutating the source."""

    from matrix_oracle import assess_initial_geometry_quality
    from matrix_qm import default_initial_hessian_level

    source = Path(xyzin_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"initialization source does not exist: {source}")
    root = Path(run_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    protocol = load_initialization_protocol()
    initial_xyzin = root / "00_input.xyzin"
    shutil.copy2(source, initial_xyzin)
    geometry = read_xyzin_geometry(initial_xyzin)
    deferred_quality = (
        str(request.geometry_quality_override).strip().lower().replace("-", "_")
        == "requires_preoptimization"
        or str(request.source_kind).strip().lower().replace("-", "_")
        in {"smiles", "cxsmiles", "inchi", "switch_smiles", "generated"}
    )
    quality = None if deferred_quality else assess_initial_geometry_quality(
        geometry, source_kind=request.source_kind, override=request.geometry_quality_override
    )
    quality_path = root / "initial_geometry_quality.json"
    if quality is not None:
        quality_path.write_text(
            json.dumps(quality.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if quality is not None and quality.status == "INVALID":
        raise ValueError(
            "ORACLE rejected the initial geometry: " + "; ".join(quality.reasons)
        )
    route = plan_initialization_stages(
        geometry_status="PREOPTIMIZE" if deferred_quality else quality.status,
        method_class=request.method_class,
        xtb_available=request.xtb_available,
        protocol=protocol,
    )
    # A generic external evaluator promises energies/gradients only.  Do not
    # reinterpret its method label as an engine capable of the canonical Li-1
    # Hessian job: in that case LINK must start from its internal model seed.
    hessian_guess = (
        None
        if request.final_backend.name.strip().casefold() == "external"
        else default_initial_hessian_level(
            request.final_backend.method or request.final_backend.route,
            request.final_backend.basis,
        )
    )
    if hessian_guess is not None:
        prefix: tuple[dict[str, str], ...] = ()
        if deferred_quality or quality.requires_preoptimization:
            first = dict(route[0])
            if first["action"] == "optimize_and_hessian":
                first["action"] = "optimize_only"
            prefix = (first,)
        route = (
            *prefix,
            {"provider": "LI-1", "action": "hessian_at_unchanged_geometry"},
            {"provider": "FINAL", "action": "optimize_with_previous_hessian"},
        )
    if deferred_quality or quality.requires_preoptimization:
        first_provider = str(route[0]["provider"])
        provider_owner = str(
            protocol.payload["providers"][first_provider].get("owner", "")
        )
        valid_owner = (
            provider_owner == "MATRIX_GFN2_resident_calculator"
            if first_provider == "GFN-xTB"
            else provider_owner.startswith("ARCHITECT")
        )
        if not valid_owner:
            raise RuntimeError(
                "initial Cartesian preoptimization has an invalid provider owner"
            )
    current_xyzin = initial_xyzin
    previous_hessian = None
    previous_hessian_source = ""
    previous_hessian_is_qm_linear = False
    stage_results: list[InitializationStageResult] = []
    chain_converged = True
    for index, stage in enumerate(route, start=1):
        provider = stage["provider"]
        action = stage["action"]
        stage_dir = root / f"{index:02d}_{_safe_name(provider)}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        input_xyzin = current_xyzin
        before_hash = _geometry_hash(input_xyzin)
        # A geometry-quality PREOPTIMIZE decision is deliberately handled in
        # Cartesian coordinates.  SONIC is constructed only after the first
        # low-cost refinement has produced a chemically credible geometry.
        cartesian_preoptimization = (deferred_quality or quality.requires_preoptimization) and index == 1
        model = _coordinate_model(
            input_xyzin,
            request,
            coordinate_kind="cartesian" if cartesian_preoptimization else None,
        )
        optimizer_result = None
        hessian_path = None
        raw_hessian_path = None
        execution_backend = ""
        if action == "hessian_at_unchanged_geometry":
            previous_hessian, raw_hessian_path, execution_backend = _fresh_provider_hessian(
                provider,
                input_xyzin,
                model,
                run_dir=stage_dir / "hessian",
                request=request,
            )
            hessian_path = write_optimizer_hessian(
                stage_dir / "optimizer_hessian.json",
                previous_hessian,
                model,
                source=(
                    f"{provider} fresh full Hessian; pseudo-bond source SONIC then "
                    "active-target congruence without B-prime"
                    if provider in {"UFF", "GFN-FF"}
                    else f"{provider} fresh full Hessian; direct active-SONIC "
                    "congruence without B-prime"
                ),
            )
            previous_hessian_source = str(hessian_path)
            previous_hessian_is_qm_linear = provider not in {
                "UFF",
                "GFN-FF",
                "GFN-xTB",
            }
            output_xyzin = input_xyzin
        elif action in {
            "optimize_only",
            "optimize_and_hessian",
            "optimize_with_previous_hessian",
        }:
            backend = _provider_backend(provider, request)
            engine_command, execution_backend = _provider_engine_command(
                provider,
                input_xyzin,
                stage_dir=stage_dir,
                request=request,
            )
            if backend is not None:
                execution_backend = str(backend.name).strip().casefold()
            if action == "optimize_with_previous_hessian" and previous_hessian is None:
                raise RuntimeError("final initialization stage has no preceding exact Hessian")
            stage_settings = request.optimizer_settings
            if cartesian_preoptimization:
                # This stage only brings an unreliable seed into the correct
                # basin; production thresholds apply to the later refinement.
                stage_settings = replace(
                    stage_settings,
                    convergence_profile="geometry_seed",
                )
            chart_lifecycle_controller = None
            if request.chart_lifecycle and not cartesian_preoptimization:
                from .chart_lifecycle import chart_lifecycle_controller_from_xyzin

                chart_lifecycle_controller = chart_lifecycle_controller_from_xyzin(
                    input_xyzin,
                    run_dir=stage_dir / "optimization",
                    coordinate_model=model,
                    stationary_point=request.optimizer_settings.stationary_point,
                    smith_refresh_interval=(
                        request.optimizer_settings.smith_refresh_interval
                    ),
                )
            optimizer_result = optimize_geometry(
                input_xyzin,
                run_dir=stage_dir / "optimization",
                coordinate_model=model,
                allow_non_sonic_coordinates=cartesian_preoptimization,
                engine_command=engine_command,
                backend=backend,
                settings=stage_settings,
                timeout=request.timeout,
                initial_hessian=previous_hessian,
                initial_hessian_source=(
                    previous_hessian_source or "frozen_LINK_internal_chemical_seed"
                ),
                refine_initial_qm_hessian_with_b_prime=(
                    previous_hessian is not None and previous_hessian_is_qm_linear
                ),
                require_frozen_symmetry_contract=True,
                chart_lifecycle_controller=chart_lifecycle_controller,
            )
            output_xyzin = stage_dir / "optimized.xyzin"
            _write_optimized_xyzin(
                input_xyzin,
                output_xyzin,
                optimizer_result,
                coordinate_kind=(
                    "cartesian" if cartesian_preoptimization else request.coordinate_kind
                ),
                provider=provider,
                chart_lifecycle=request.chart_lifecycle,
                stationary_point=request.optimizer_settings.stationary_point,
            )
            current_xyzin = output_xyzin
            if cartesian_preoptimization and optimizer_result.converged:
                quality = assess_initial_geometry_quality(
                    read_xyzin_geometry(output_xyzin),
                    source_kind="xyzin",
                    override="auto",
                )
                if quality.status == "INVALID":
                    raise ValueError(
                        "ORACLE rejected the post-preoptimization geometry: "
                        + "; ".join(quality.reasons)
                    )
                if quality.requires_preoptimization:
                    raise RuntimeError(
                        "Cartesian preoptimization did not produce a geometry "
                        "suitable for ORACLE perception and SONIC construction"
                    )
                quality_path.write_text(
                    json.dumps(quality.to_dict(), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                # Perception and coordinate construction deliberately occur
                # only after the Cartesian basin-finding stage.
                from .multilevel import _refresh_sonic_contract

                _refresh_sonic_contract(output_xyzin)
            if not optimizer_result.converged:
                chain_converged = False
                if cartesian_preoptimization:
                    raise RuntimeError(
                        "Cartesian basin-finding preoptimization did not converge; "
                        "ORACLE perception and SONIC construction were not started"
                    )
            if action == "optimize_and_hessian" and optimizer_result.converged:
                refreshed_model = _coordinate_model(output_xyzin, request)
                previous_hessian, raw_hessian_path, hessian_backend = _fresh_provider_hessian(
                    provider,
                    output_xyzin,
                    refreshed_model,
                    run_dir=stage_dir / "hessian",
                    request=request,
                )
                if hessian_backend != execution_backend and {
                    hessian_backend,
                    execution_backend,
                } != {"matrix_gfn2", "xtb"}:
                    raise RuntimeError(
                        f"provider {provider} optimization/Hessian backend drift: "
                        f"{execution_backend} != {hessian_backend}"
                    )
                hessian_path = write_optimizer_hessian(
                    stage_dir / "optimizer_hessian.json",
                    previous_hessian,
                    refreshed_model,
                    source=(
                        f"{provider} fresh full Hessian at converged stage geometry; "
                        + (
                            "pseudo-bond source SONIC then active-target congruence "
                            "without B-prime"
                            if provider in {"UFF", "GFN-FF"}
                            else "direct active-SONIC congruence without B-prime"
                        )
                    ),
                )
                previous_hessian_source = str(hessian_path)
                previous_hessian_is_qm_linear = provider not in {
                    "UFF",
                    "GFN-FF",
                    "GFN-xTB",
                }
        else:
            raise RuntimeError(f"unsupported frozen initialization action: {action}")
        after_hash = _geometry_hash(output_xyzin)
        stage_results.append(
            InitializationStageResult(
                index=index,
                provider=provider,
                action=action,
                input_xyzin=input_xyzin,
                output_xyzin=output_xyzin,
                run_dir=stage_dir,
                geometry_hash_before=before_hash,
                geometry_hash_after=after_hash,
                execution_backend=execution_backend,
                hessian_path=hessian_path,
                raw_hessian_path=raw_hessian_path,
                optimizer_result=optimizer_result,
            )
        )
        if optimizer_result is not None and not optimizer_result.converged:
            break
    target = (
        Path(optimized_xyzin).expanduser().resolve()
        if optimized_xyzin is not None
        else root / "optimized.xyzin"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(current_xyzin, target)
    final_optimizer_result = next(
        (
            item.optimizer_result
            for item in reversed(stage_results)
            if item.optimizer_result is not None
        ),
        None,
    )
    if final_optimizer_result is not None:
        shutil.copy2(final_optimizer_result.summary_path, root / "optimizer_summary.json")
    manifest_path = root / "initialization_workflow.json"
    manifest = {
        "schema": INITIALIZATION_WORKFLOW_RESULT_SCHEMA,
        "protocol": {
            "id": protocol.payload["protocol_id"],
            "version": protocol.payload["manifest_version"],
            "sha256": protocol.sha256,
            "source": protocol.source,
        },
        "request": request.to_dict(),
        "default_initial_hessian_level": (
            None if hessian_guess is None else hessian_guess.to_dict()
        ),
        "initial_geometry_quality": quality.to_dict(),
        "initial_geometry_quality_path": str(quality_path),
        "initial_xyzin": str(initial_xyzin),
        "final_xyzin": str(target),
        "converged": chain_converged and len(stage_results) == len(route),
        "stages": [item.to_dict() for item in stage_results],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return InitializationWorkflowResult(
        initial_xyzin=initial_xyzin,
        final_xyzin=target,
        manifest_path=manifest_path,
        geometry_quality=quality.to_dict(),
        stages=tuple(stage_results),
        converged=bool(manifest["converged"]),
    )


def _provider_backend(
    provider: str, request: InitializationWorkflowRequest
) -> QMScanBackend | None:
    final = request.final_backend
    if provider == "FINAL":
        if final.name.strip().casefold() == "external":
            if not request.final_engine_command.strip():
                raise ValueError("external final provider requires final_engine_command")
            return None
        return final
    if provider == "GFN-FF":
        selected = _architect_provider_descriptor(provider, request=request)
        return QMScanBackend(
            name=str(selected["runtime_backend"]),
            route=str(selected["route"]),
            method=str(selected["method"]),
            charge=final.charge,
            multiplicity=final.multiplicity,
            executable=_optional_string(selected.get("executable")),
            timeout=request.timeout,
            processors=final.processors,
            memory_gb=final.memory_gb,
            gradient_mode="analytic",
        )
    if provider == "GFN-xTB":
        from .matrix_gfn2_backend import matrix_gfn2_available

        if not matrix_gfn2_available():
            raise RuntimeError(
                "LINK semiempirical initialization requires autonomous MATRIX-GFN2; "
                "external semiempirical fallbacks are forbidden"
            )
        return QMScanBackend(
            name="matrix_gfn2",
            route="",
            method="MATRIX-GFN2",
            charge=final.charge,
            multiplicity=final.multiplicity,
            executable=None,
            timeout=request.timeout,
            processors=final.processors,
            memory_gb=final.memory_gb,
            gradient_mode="analytic",
        )
    if provider == "UFF":
        return None
    if provider == "HF/STO-3G":
        from matrix_core import load_runtime_environment
        from matrix_qm import resolve_open_hf_sto3g_provider

        selected = resolve_open_hf_sto3g_provider(
            load_runtime_environment(missing_ok=True),
            requested_backend=request.hf_sto3g_backend,
        )
        return QMScanBackend(
            name=selected.backend,
            method=selected.method,
            basis=selected.basis,
            charge=final.charge,
            multiplicity=final.multiplicity,
            executable=selected.executable,
            timeout=request.timeout,
            processors=final.processors,
            memory_gb=final.memory_gb,
            gradient_mode="analytic",
        )
    if provider == "LI-1":
        from matrix_qm import default_initial_hessian_level

        selected = default_initial_hessian_level(
            final.method or final.route,
            final.basis,
        )
        if selected is None:
            raise RuntimeError("the final method has no canonical Li-1 Hessian level")
        return QMScanBackend(
            # LI-1 is a scientific level, not a PySCF requirement.  Reuse the
            # selected final engine when it can supply the lower-level Hessian;
            # this keeps the initialization portable and avoids silently
            # introducing an extra backend dependency.
            name=final.name,
            method=selected.method,
            basis=selected.basis,
            charge=final.charge,
            multiplicity=final.multiplicity,
            executable=final.executable,
            timeout=request.timeout,
            processors=final.processors,
            memory_gb=final.memory_gb,
            gradient_mode="analytic",
        )
    raise RuntimeError(
        f"provider {provider} is not registered in the canonical initialization workflow"
    )


def _provider_engine_command(
    provider: str,
    xyzin_path: Path,
    *,
    stage_dir: Path,
    request: InitializationWorkflowRequest,
) -> tuple[str, str]:
    if provider == "FINAL":
        return request.final_engine_command, (
            "external"
            if _provider_backend(provider, request) is None
            else str(request.final_backend.name).strip().casefold()
        )
    if provider != "UFF":
        return "", ""
    model_path = stage_dir / "uff_model.json"
    _run_architect(
        "uff-compile",
        str(xyzin_path),
        "--output",
        str(model_path),
    )
    selected = _architect_provider_descriptor(
        provider,
        request=request,
        uff_model=model_path,
    )
    return str(selected["engine_command"]), str(selected["runtime_backend"])


def _coordinate_model(
    path: Path,
    request: InitializationWorkflowRequest,
    *,
    coordinate_kind: str | None = None,
):
    if coordinate_kind == "cartesian":
        return coordinate_model_from_xyzin(
            path,
            kind="cartesian",
            coordinates=(),
        )
    if request.variables is not None:
        if request.coordinates:
            raise ValueError("use either initialization variables or coordinate labels")
        from .active_variables import active_variable_contract_from_file

        return active_variable_contract_from_file(path, request.variables).model
    if request.chart_lifecycle:
        from matrix_smith import read_gic_definition_from_xyzin

        definition = read_gic_definition_from_xyzin(path)
        return coordinate_model_from_xyzin(
            path,
            kind="sonic",
            # Do not turn the frozen GIC list into an explicit user-selected
            # subset: that flag disables the compiled sparse SONIC runtime.
            # LINK will select the complete symmetry-allowed minimum chart
            # from the same ORACLE/SMITH definition below.
            coordinates=(),
            sonic_definition=definition,
            task_regime=(
                "TRANSITION_STATE"
                if request.optimizer_settings.stationary_point == "transition_state"
                else "MINIMUM"
                if request.optimizer_settings.stationary_point == "minimum"
                else None
            ),
            matrix_free=(
                str(request.optimizer_settings.stationary_point)
                .strip()
                .casefold()
                == "minimum"
            ),
        )
    return coordinate_model_from_xyzin(
        path,
        kind=coordinate_kind or request.coordinate_kind,
        coordinates=request.coordinates,
    )


def _resolved_xtb_capability(value: Any) -> tuple[bool, str | None]:
    if isinstance(value, str) and value.strip().casefold() == "auto":
        from matrix_core import load_runtime_environment

        environment = load_runtime_environment(missing_ok=True)
        program = environment.program("xtb")
        available = bool(program and program.enabled and program.available)
        executable = None if program is None else str(program.executable).strip() or None
        return available, executable
    return bool(value), None


def _resolved_xtb_executable(data: Mapping[str, Any]) -> str | None:
    explicit = data.get("xtb_executable")
    if explicit:
        return str(explicit)
    return _resolved_xtb_capability(data.get("xtb_available", "auto"))[1]


def _fresh_provider_hessian(
    provider: str,
    xyzin_path: Path,
    model,
    *,
    run_dir: Path,
    request: InitializationWorkflowRequest,
):
    if provider == "UFF":
        geometry = read_xyzin_geometry(xyzin_path)
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "uff_model.json"
        _run_architect(
            "uff-compile",
            str(xyzin_path),
            "--output",
            str(model_path),
        )
        point_dir = run_dir / "point_0000"
        point_dir.mkdir(parents=True, exist_ok=True)
        point_path = write_xyz(
            point_dir / "point.xyz",
            geometry.atoms,
            geometry.coordinates_angstrom,
            comment="ARCHITECT UFF exact-Hessian evaluation",
        )
        result_path = point_dir / "point.json"
        _run_architect(
            "uff-point",
            str(model_path),
            str(point_path),
            "--result",
            str(result_path),
            "--point-index",
            "0",
            "--properties",
            "energy,gradient,hessian",
        )
        uff_result = read_point_result(result_path)
        if uff_result.status != "completed":
            raise RuntimeError(
                "ARCHITECT UFF Hessian evaluation failed: "
                f"{uff_result.message or uff_result.status}"
            )
        cartesian = uff_result.hessian_hartree_per_bohr2
        if cartesian is None:
            raise RuntimeError("ARCHITECT UFF did not return its seed Hessian")
        raw_hessian_path = run_dir / "cartesian_hessian.json"
        raw_hessian_path.parent.mkdir(parents=True, exist_ok=True)
        raw_hessian_path.write_text(
            json.dumps(
                {
                    "schema": "matrix.link.initialization_cartesian_hessian.v1",
                    "provider": provider,
                    "execution_backend": "architect_uff",
                    "method": "Universal Force Field",
                    "basis": "",
                    "units": "hartree/bohr^2",
                    "hessian_contract": (
                        "FF_cartesian_to_pseudobond_source_SONIC_to_active_target_"
                        "without_B_prime"
                    ),
                    "atoms": list(geometry.atoms),
                    "coordinates_angstrom": geometry.coordinates_angstrom.tolist(),
                    "hessian": np.asarray(cartesian, dtype=float).tolist(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return (
            optimizer_hessian_from_force_field_cartesian(
                cartesian,
                model,
                xyzin_path=xyzin_path,
                coordinates_angstrom=geometry.coordinates_angstrom,
            ),
            raw_hessian_path,
            "architect_uff",
        )
    if provider in {"GFN-FF", "GFN-xTB"}:
        selected = _architect_provider_descriptor(provider, request=request)
        route = str(selected["route"])
    elif provider in {"HF/STO-3G", "LI-1"}:
        backend = _provider_backend(provider, request)
        assert backend is not None
        hessian, raw_hessian_path = build_optimizer_hessian_seed(
            xyzin_path,
            model,
            engine=backend.name,
            run_dir=run_dir,
            method=backend.method,
            basis=backend.basis,
            charge=backend.charge,
            multiplicity=backend.multiplicity,
            executable=backend.executable,
            timeout=request.timeout,
            processors=backend.processors,
            memory_gb=backend.memory_gb,
        )
        return hessian, raw_hessian_path, backend.name
    else:
        raise RuntimeError(
            f"fresh Hessian provider {provider} is not registered in the canonical workflow"
        )
    hessian, source = build_optimizer_hessian_seed(
        xyzin_path,
        model,
        engine="xtb",
        run_dir=run_dir,
        route=route,
        charge=request.final_backend.charge,
        multiplicity=request.final_backend.multiplicity,
        executable=request.xtb_executable,
        timeout=request.timeout,
        hessian_kind=("force_field" if provider == "GFN-FF" else "qm"),
    )
    return hessian, source, str(selected["runtime_backend"])


def _architect_provider_descriptor(
    provider: str,
    *,
    request: InitializationWorkflowRequest,
    uff_model: Path | None = None,
) -> dict[str, Any]:
    arguments = ["initial-provider", provider]
    if request.xtb_executable:
        arguments.extend(("--xtb-executable", request.xtb_executable))
    if uff_model is not None:
        arguments.extend(("--uff-model", str(uff_model)))
    completed = _run_architect(*arguments)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ARCHITECT returned an invalid provider descriptor") from exc
    if payload.get("schema") != "matrix.architect.initial_geometry_provider.v1":
        raise RuntimeError("ARCHITECT returned an unsupported provider descriptor")
    if payload.get("owner") != "ARCHITECT":
        raise RuntimeError("initial-geometry provider is not owned by ARCHITECT")
    if not payload.get("runtime_backend") or not payload.get("method"):
        raise RuntimeError("ARCHITECT returned an incomplete provider descriptor")
    if uff_model is not None and not payload.get("engine_command"):
        raise RuntimeError("ARCHITECT UFF provider did not return an engine command")
    return payload


def _run_architect(*arguments: str) -> subprocess.CompletedProcess[str]:
    command = (sys.executable, "-m", "matrix_architect.cli", *arguments)
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            "ARCHITECT process-boundary call failed"
            + (f": {detail}" if detail else "")
        )
    return completed


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _write_optimized_xyzin(
    source: Path,
    target: Path,
    result: OptimizerResult,
    *,
    coordinate_kind: str,
    provider: str,
    chart_lifecycle: bool = False,
    stationary_point: str = "minimum",
) -> None:
    if chart_lifecycle:
        from matrix_oracle import materialize_optimization_chart_artifact
        from matrix_smith import write_gicforge_build_sections

        regime = (
            "TRANSITION_STATE"
            if str(stationary_point).strip().casefold().replace("-", "_")
            == "transition_state"
            else "MINIMUM"
        )
        materialize_optimization_chart_artifact(
            source,
            target,
            result.atoms,
            result.final_coordinates_angstrom,
            task_regime=regime,
        )
        write_gicforge_build_sections(
            target,
            symmetrize=False,
            out_of_plane_model="soop",
            fragment_context=(
                "transition_state" if regime == "TRANSITION_STATE" else "minimum"
            ),
        )
        return
    from matrix_chem.xyzin_geometry import replace_xyzin_geometry

    shutil.copy2(source, target)
    replace_xyzin_geometry(
        target,
        result.atoms,
        result.final_coordinates_angstrom,
        comment=f"MATRIX frozen initialization protocol; provider={provider}",
    )
    if coordinate_kind == "sonic":
        from .multilevel import _refresh_sonic_contract

        _refresh_sonic_contract(target)


def _geometry_hash(path: Path) -> str:
    geometry = read_xyzin_geometry(path)
    token = json.dumps(
        {
            "atoms": list(geometry.atoms),
            "coordinates_angstrom": geometry.coordinates_angstrom.tolist(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(token).hexdigest()


def _safe_name(value: str) -> str:
    result = "".join(character.lower() if character.isalnum() else "_" for character in value)
    return result.strip("_") or "stage"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


__all__ = [
    "INITIALIZATION_WORKFLOW_REQUEST_SCHEMA",
    "INITIALIZATION_WORKFLOW_RESULT_SCHEMA",
    "InitializationStageResult",
    "InitializationWorkflowRequest",
    "InitializationWorkflowResult",
    "initialization_request_from_dict",
    "plan_initialization_stages",
    "run_initialization_workflow",
]
