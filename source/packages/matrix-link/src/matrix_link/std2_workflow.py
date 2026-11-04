"""Prepared KEYMAKER std2 optimization and gradient-difference Hessian workflows."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from matrix_core import (
    CalculationResources, load_keymaker_protocol_library,
    write_calculation_authorization_bundle,
)
from matrix_core.calculation_launch import (
    build_calculation_launch_plan, require_calculation_launch_authorization,
    CALCULATION_AUTHORIZATION_BUNDLE_ENV,
)
from .scan import QMScanBackend
from .std2 import STD2Settings


def prepare_std2_geometry(
    xyzin_path, run_dir, *, settings: STD2Settings, reference: dict,
    target_state_id: int, resources: CalculationResources, operation="optimization",
    coordinate_kind="sonic", maximum_steps=50, charge=0, multiplicity=1,
    gradient_step_bohr=0.01, hessian_step_bohr=0.01,
    minimum_overlap=0.70, ambiguity_margin=0.05, executable=None,
):
    """Freeze input, reference provider and resource limits without evaluating."""
    from .optimizer import coordinate_model_from_xyzin

    if operation not in {"optimization", "hessian"}:
        raise ValueError("std2 geometry operation must be optimization or hessian")
    if coordinate_kind not in {"sonic", "sycart"}:
        raise ValueError("std2 geometry workflows require SONIC or SYCART")
    if resources.process_count != 1 or resources.concurrent_jobs != 1:
        raise ValueError("std2 geometry workflow requires one sequential process")
    if target_state_id < 1 or maximum_steps < 1:
        raise ValueError("std2 target root and maximum steps must be positive")
    if not np.all(np.isfinite([gradient_step_bohr, hessian_step_bohr])) or min(gradient_step_bohr, hessian_step_bohr) <= 0:
        raise ValueError("std2 derivative steps must be positive finite numbers")
    if not 0 < minimum_overlap <= 1 or not 0 <= ambiguity_margin < 1:
        raise ValueError("invalid APOC state-following thresholds")
    settings.arguments()
    if reference.get("backend") not in {"pyscf", "command"}:
        raise ValueError("std2 ground-state provider must be pyscf or command")
    source = Path(xyzin_path).expanduser().resolve(strict=True)
    # Validate the coordinate contract before presenting a runnable plan.
    coordinate_model_from_xyzin(source, kind=coordinate_kind)
    root = Path(run_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    request_path = root / "std2-geometry-request.json"
    payload = {
        "schema": "matrix.link.std2.geometry_request.v1", "xyzin": str(source),
        "xyzin_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "settings": asdict(settings), "reference": reference,
        "operation": operation, "target_state_id": target_state_id,
        "coordinate_kind": coordinate_kind, "maximum_steps": maximum_steps,
        "charge": charge, "multiplicity": multiplicity,
        "gradient_step_bohr": gradient_step_bohr, "hessian_step_bohr": hessian_step_bohr,
        "minimum_overlap": minimum_overlap, "ambiguity_margin": ambiguity_margin,
        "executable": executable,
    }
    if any(path.name != request_path.name for path in root.iterdir()):
        raise FileExistsError("std2 geometry workflow requires a new run directory")
    request_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return build_calculation_launch_plan(
        backend="LINK/std2-geometry", input_path=request_path, workdir=root,
        command=(sys.executable, "-m", __name__, str(request_path)), resources=resources,
    )


@contextmanager
def _authorized_workflow(plan, authorization):
    require_calculation_launch_authorization(plan, authorization)
    bundle = write_calculation_authorization_bundle(
        Path(plan.workdir) / "launch-authorization.json", plan, authorization,
    )
    previous = os.environ.get(CALCULATION_AUTHORIZATION_BUNDLE_ENV)
    os.environ[CALCULATION_AUTHORIZATION_BUNDLE_ENV] = str(bundle)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(CALCULATION_AUTHORIZATION_BUNDLE_ENV, None)
        else:
            os.environ[CALCULATION_AUTHORIZATION_BUNDLE_ENV] = previous


def run_std2_geometry(plan, *, launch_authorization):
    """Use the existing LINK optimizer/Hessian acquisition and APOC decisions."""
    from .optimizer import (
        GeometryEvaluationService, OptimizerSettings, coordinate_model_from_xyzin,
        optimize_geometry, optimizer_result_to_json, _evaluate_final_cartesian_hessian,
    )

    require_calculation_launch_authorization(plan, launch_authorization)
    if plan.backend != "LINK/std2-geometry":
        raise ValueError("incorrect std2 geometry launch backend")
    request = json.loads(Path(plan.input_path).read_text())
    if request.get("schema") != "matrix.link.std2.geometry_request.v1":
        raise ValueError("unsupported std2 geometry request")
    source = Path(request["xyzin"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != request["xyzin_sha256"]:
        raise ValueError("std2 geometry source changed after preparation")
    protocol_id = "link-std2-optimization.v1" if request["operation"] == "optimization" else "link-std2-hessian.v1"
    library = load_keymaker_protocol_library()
    protocol = library.resolve(protocol_id)
    if protocol.status != "approved" or (protocol.owner_api, protocol.action) != (__name__, "run_std2_geometry"):
        raise ValueError("std2 geometry protocol owner binding changed")
    backend = QMScanBackend(
        name="std2", method="std2/" + request["settings"]["method"],
        std2_contract={"settings": request["settings"], "reference": request["reference"]},
        electronic_state=int(request["target_state_id"]), state_tracking="apoc",
        state_tracking_minimum_overlap=request["minimum_overlap"],
        state_tracking_small_gap_minimum_overlap=min(0.5, request["minimum_overlap"]),
        state_tracking_ambiguity_margin=request["ambiguity_margin"],
        gradient_mode="hybrid", numerical_gradient_step_bohr=request["gradient_step_bohr"],
        std2_hessian_step_bohr=request["hessian_step_bohr"],
        charge=int(request["charge"]), multiplicity=int(request["multiplicity"]),
        processors=plan.resources.threads_per_process,
        memory_gb=plan.resources.memory_per_job_gb, executable=request["executable"],
        properties=("energy", "gradient"),
    )
    model = coordinate_model_from_xyzin(source, kind=request["coordinate_kind"])
    root = Path(plan.workdir)
    optimizer_settings = OptimizerSettings(
        stationary_point="minimum", max_steps=int(request["maximum_steps"]),
        compute_final_hessian=request["operation"] == "optimization",
        excited_state_gap_control=True, coordinate_parallel_workers=1,
    )
    with _authorized_workflow(plan, launch_authorization):
        if request["operation"] == "optimization":
            optimized = optimize_geometry(
                source, run_dir=root / "link", coordinate_model=model, backend=backend,
                settings=optimizer_settings, allow_non_sonic_coordinates=model.kind == "sycart",
                allow_native_sycart=False,
            )
            output = optimizer_result_to_json(optimized)
        else:
            service = GeometryEvaluationService(
                xyzin_path=source, run_dir=root / "link", coordinate_model=model,
                backend=backend, settings=optimizer_settings,
            )
            q = service.actual_q(service.reference_coordinates)
            center = service.evaluate(q, tag="std2-hessian-center", requested_properties=("energy", "gradient"))
            service.accept_electronic_state(center)
            evaluated = _evaluate_final_cartesian_hessian(service, center, q, tag="std2-hessian")
            output = {
                "schema": "matrix.link.std2.hessian.v1",
                "energy_hartree": evaluated.energy_hartree,
                "gradient_hartree_per_bohr": evaluated.gradient_hartree_per_bohr.tolist(),
                "hessian_hartree_per_bohr2": evaluated.result.hessian_hartree_per_bohr2.tolist(),
                "coordinates_angstrom": evaluated.coordinates_angstrom.tolist(),
                "execution": evaluated.result.execution,
            }
    output["keymaker_provenance"] = {
        "schema": "matrix.keymaker.protocol_binding.v1", "protocol_id": protocol_id,
        "sha256": protocol.sha256, "library_sha256": library.sha256,
    }
    output["derivative_contract"] = {
        "ground_state_gradient": "reference_backend_analytic",
        "excitation_gradient": "APOC_followed_central_energy_differences",
        "hessian": "central_differences_of_complete_gradients",
        "gradient_step_bohr": request["gradient_step_bohr"],
        "hessian_step_bohr": request["hessian_step_bohr"],
        "std2_eigenvalue_storage": "upstream_real4",
    }
    if request["operation"] == "optimization":
        output["final_gradient_verification"] = {
            "performed": False, "confirmed": False,
            "reason": "hybrid_gradient_requires_independent_numerical_qualification",
        }
    (root / "std2-geometry-result.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


__all__ = ["prepare_std2_geometry", "run_std2_geometry"]


if __name__ == "__main__":
    from matrix_core import read_calculation_authorization_bundle
    from matrix_link.std2_workflow import run_std2_geometry as run_prepared

    bundle = os.environ.get(CALCULATION_AUTHORIZATION_BUNDLE_ENV)
    if not bundle or len(sys.argv) != 2:
        raise SystemExit("run through KEYMAKER or provide the exact authorized workflow bundle")
    prepared_plan, authorization = read_calculation_authorization_bundle(bundle)
    if Path(sys.argv[1]).resolve() != Path(prepared_plan.input_path).resolve():
        raise SystemExit("request differs from the authorized std2 workflow")
    print(json.dumps(run_prepared(prepared_plan, launch_authorization=authorization), indent=2))
