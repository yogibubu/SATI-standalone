from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from collections.abc import Sequence

from matrix_chem.xyzin_geometry import replace_xyzin_geometry

from .optimizer import (
    QMScanBackend,
    OptimizerResult,
    OptimizerSettings,
    build_optimizer_hessian_seed,
    coordinate_model_from_xyzin,
    optimize_geometry,
    read_optimizer_hessian,
)


MULTILEVEL_OPTIMIZATION_SCHEMA = "matrix.link.multilevel_optimization.v1"
OPTIMIZATION_REPORT_SCHEMA = "matrix.link.optimization_report.v1"


@dataclass(frozen=True)
class OptimizationLevel:
    name: str
    backend: str = ""
    route: str = ""
    method: str = ""
    basis: str = ""
    executable: str | None = None
    engine_command: str = ""
    coordinate_kind: str = "cartesian"
    coordinates: tuple[str, ...] = ()
    max_steps: int = 50
    trust_radius: float = 0.2
    max_trust_radius: float = 0.3
    gradient_tolerance: float = 4.5e-4
    step_tolerance: float = 1.8e-3
    timeout: float | None = None
    use_previous_hessian: bool = True
    convergence: str = "normal"
    initial_hessian_model: str = "lindh_swart_special"
    enable_gdiis: bool = True
    processors: int = 1
    memory_gb: int | None = None


@dataclass(frozen=True)
class OptimizationChainResult:
    smiles: str
    initial_xyzin: Path
    final_xyzin: Path
    manifest_path: Path
    report_text_path: Path
    report_json_path: Path
    results: tuple[OptimizerResult, ...]
    stopped_reason: str = ""


def optimize_from_smiles_multilevel(
    smiles: str = "",
    *,
    run_dir: Path | str,
    levels: Sequence[OptimizationLevel],
    source_xyz: Path | str | None = None,
    title: str = "",
    charge: int | None = None,
    multiplicity: int | None = None,
    random_seed: int = 61453,
    lcb26_root: Path | str | None = None,
) -> OptimizationChainResult:
    """Run a sequential MATRIX optimizer chain from an unrelaxed SWITCH seed."""

    if not levels:
        raise ValueError("at least one optimization level is required")
    if bool(str(smiles).strip()) == (source_xyz is not None):
        raise ValueError("provide exactly one of smiles or source_xyz")
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    needs_sonic = any(
        str(level.coordinate_kind).replace("-", "_") == "sonic" for level in levels
    )
    initialization_route = "provided_xyz"
    lcb26_seed = (
        _lcb26_seed_for_smiles(smiles, lcb26_root=lcb26_root)
        if source_xyz is None and needs_sonic
        else None
    )
    if lcb26_seed is not None:
        initial_xyzin = _write_sonic_xyzin(
            lcb26_seed,
            root / "00_oracle_sonic.xyzin",
            source_kind="xyz",
        )
        initial_geometry_source = f"LCB26 geometry {lcb26_seed}"
        initialization_route = "lcb26"
    elif source_xyz is None:
        initial_xyzin = _write_smiles_xyzin(
            smiles,
            root / "00_switch_seed.xyzin",
            title=title,
            charge=charge,
            multiplicity=multiplicity,
            random_seed=random_seed,
        )
        initial_geometry_source = "SMILES -> MATRIX SWITCH deterministic seed"
        initialization_route = "switch_matrix_gfn2"
    else:
        source = Path(source_xyz).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"source XYZ does not exist: {source}")
        initial_xyzin = _write_sonic_xyzin(
            source,
            root / "00_oracle_sonic.xyzin",
            source_kind="xyz",
            build_sonic=needs_sonic,
        )
        initial_geometry_source = f"XYZ file {source}"
    if initialization_route == "switch_matrix_gfn2" and needs_sonic:
        from .matrix_gfn2_backend import matrix_gfn2_available

        if not matrix_gfn2_available():
            raise RuntimeError(
                "automatic Keymaker initialization requires the qualified "
                "autonomous MATRIX-GFN2 calculator"
            )
        preopt_dir = root / "00_matrix_gfn2_cartesian_preoptimization"
        preopt_model = coordinate_model_from_xyzin(
            initial_xyzin,
            kind="cartesian",
            coordinates=(),
        )
        preopt_backend = QMScanBackend(
            name="matrix_gfn2",
            method="MATRIX-GFN2",
            charge=0 if charge is None else int(charge),
            multiplicity=1 if multiplicity is None else int(multiplicity),
            gradient_mode="analytic",
        )
        preopt_result = optimize_geometry(
            initial_xyzin,
            run_dir=preopt_dir,
            coordinate_model=preopt_model,
            allow_non_sonic_coordinates=True,
            backend=preopt_backend,
            settings=OptimizerSettings(
                max_steps=30,
                convergence_profile="geometry_seed",
                compute_final_hessian=False,
                require_exact_final_hessian=False,
            ),
        )
        if not preopt_result.converged:
            raise RuntimeError(
                "Keymaker Cartesian MATRIX-GFN2 preoptimization did not converge"
            )
        preoptimized = root / "00_cartesian_preoptimized.xyzin"
        preoptimized.write_text(initial_xyzin.read_text(encoding="utf-8"), encoding="utf-8")
        replace_xyzin_geometry(
            preoptimized,
            preopt_result.atoms,
            preopt_result.final_coordinates_angstrom,
            comment="KEYMAKER resident MATRIX-GFN2 Cartesian preoptimization",
        )
        initial_xyzin = _write_sonic_xyzin(
            preoptimized,
            root / "00_oracle_sonic.xyzin",
            source_kind="enriched_xyz",
        )
        initial_geometry_source += " -> resident MATRIX-GFN2 Cartesian preoptimization"
    current_xyzin = initial_xyzin
    previous_hessian: Path | None = None
    results: list[OptimizerResult] = []
    level_records: list[dict[str, object]] = []
    stopped_reason = ""
    for index, level in enumerate(levels, start=1):
        level_dir = root / f"{index:02d}_{_safe_name(level.name)}"
        level_dir.mkdir(parents=True, exist_ok=True)
        input_xyzin = current_xyzin
        model = coordinate_model_from_xyzin(
            input_xyzin,
            kind=level.coordinate_kind,
            coordinates=level.coordinates,
        )
        initial_hessian = None
        initial_hessian_source = "chemical-valence"
        if previous_hessian is not None and level.use_previous_hessian:
            initial_hessian = read_optimizer_hessian(previous_hessian, expected_labels=model.labels)
            initial_hessian_source = f"previous-level {previous_hessian}"
        elif index == 1 and level.use_previous_hessian:
            from matrix_qm import default_initial_hessian_level

            guess_level = default_initial_hessian_level(level.method or level.route, level.basis)
            if guess_level is not None:
                initial_hessian, artifact = build_optimizer_hessian_seed(
                    input_xyzin,
                    model,
                    engine="pyscf",
                    run_dir=level_dir / "initial_hessian_Li_minus_1",
                    method=guess_level.method,
                    basis=guess_level.basis,
                    charge=0 if charge is None else int(charge),
                    multiplicity=1 if multiplicity is None else int(multiplicity),
                    timeout=level.timeout,
                )
                initial_hessian_source = (
                    f"canonical {guess_level.level} {guess_level.method}/"
                    f"{guess_level.basis} {artifact}"
                )
        backend = None
        if level.backend:
            backend = QMScanBackend(
                name=level.backend,
                route=level.route,
                method=level.method,
                basis=level.basis,
                charge=0 if charge is None else int(charge),
                multiplicity=1 if multiplicity is None else int(multiplicity),
                executable=level.executable,
                timeout=level.timeout,
                processors=level.processors,
                memory_gb=level.memory_gb,
            )
        if backend is None and not level.engine_command.strip():
            raise ValueError(f"optimization level {level.name!r} needs backend or engine_command")
        convergence = str(level.convergence).strip().lower()
        if convergence not in {"normal", "tight"}:
            raise ValueError(f"optimization level {level.name!r} has invalid convergence preset")
        tolerances = (
            (1.0e-6, 4.5e-4, 3.0e-4, 1.8e-3, 1.2e-3)
            if convergence == "normal"
            else (1.0e-8, 1.5e-5, 1.0e-5, 6.0e-5, 4.0e-5)
        )
        result = optimize_geometry(
            input_xyzin,
            run_dir=level_dir,
            coordinate_model=model,
            allow_non_sonic_coordinates=model.kind not in {"sonic", "typed_onic"},
            engine_command=level.engine_command,
            backend=backend,
            settings=OptimizerSettings(
                max_steps=level.max_steps,
                trust_radius=level.trust_radius,
                max_trust_radius=level.max_trust_radius,
                gradient_tolerance=level.gradient_tolerance,
                step_tolerance=level.step_tolerance,
                energy_tolerance=tolerances[0],
                max_force_tolerance=tolerances[1],
                rms_force_tolerance=tolerances[2],
                max_displacement_tolerance=tolerances[3],
                rms_displacement_tolerance=tolerances[4],
                initial_hessian_model=level.initial_hessian_model,
                enable_gdiis=level.enable_gdiis,
                # An intermediate Hessian is an explicit dependency of the
                # next level when that level requests reuse.  The terminal
                # level never receives a final Hessian implicitly.
                compute_final_hessian=(
                    index < len(levels) and levels[index].use_previous_hessian
                ),
                # This Hessian is a quasi-Newton continuation seed, not a
                # stationary-point certificate for an intermediate level.
                require_exact_final_hessian=False,
            ),
            timeout=level.timeout,
            initial_hessian=initial_hessian,
            initial_hessian_source=initial_hessian_source,
        )
        next_xyzin = level_dir / "optimized.xyzin"
        next_xyzin.write_text(Path(input_xyzin).read_text(encoding="utf-8"), encoding="utf-8")
        replace_xyzin_geometry(
            next_xyzin,
            result.atoms,
            result.final_coordinates_angstrom,
            comment=f"MATRIX multilevel optimized; level={level.name}",
        )
        if needs_sonic:
            _refresh_sonic_contract(next_xyzin)
        results.append(result)
        previous_hessian = result.final_hessian_path
        current_xyzin = next_xyzin
        level_records.append(
            {
                "name": level.name,
                "backend": level.backend,
                "route": level.route,
                "program": _level_program(level),
                "engine_command": level.engine_command,
                "coordinate_kind": level.coordinate_kind,
                "run_dir": str(level_dir),
                "input_xyzin": str(input_xyzin),
                "optimized_xyzin": str(next_xyzin),
                "optimizer_summary": str(result.summary_path),
                "optimizer_hessian": str(result.final_hessian_path),
                "geometry_and_sonic_contract_refreshed": bool(needs_sonic),
                "converged": result.converged,
                "status": result.status,
                "optimization_steps": len(result.iterations),
                "qm_evaluations": result.qm_evaluations,
                "gradient_evaluations": result.gradient_evaluations,
                "fd_displacements": result.fd_displacements,
            }
        )
        if not result.converged:
            stopped_reason = f"level {index} ({level.name}) did not converge"
            break
    manifest_path = root / "multilevel_optimization.json"
    manifest = {
        "schema": MULTILEVEL_OPTIMIZATION_SCHEMA,
        "smiles": smiles,
        "source_xyz": str(Path(source_xyz).expanduser().resolve()) if source_xyz else "",
        "switch_used": initialization_route == "switch_matrix_gfn2",
        "lcb26_used": initialization_route == "lcb26",
        "matrix_gfn2_preoptimization_used": initialization_route == "switch_matrix_gfn2",
        "provided_good_geometry_used": initialization_route == "provided_xyz",
        "initialization_route": initialization_route,
        "initial_xyzin": str(initial_xyzin),
        "final_xyzin": str(current_xyzin),
        "chain_status": "stopped" if stopped_reason else "completed",
        "stopped_reason": stopped_reason,
        "levels": level_records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_text_path = root / "optimization_report.txt"
    report_json_path = root / "optimization_report.json"
    write_optimization_report(
        manifest,
        text_path=report_text_path,
        json_path=report_json_path,
        title=title or smiles,
        charge=charge,
        multiplicity=multiplicity,
        initial_geometry_source=initial_geometry_source,
    )
    return OptimizationChainResult(
        smiles=smiles,
        initial_xyzin=initial_xyzin,
        final_xyzin=current_xyzin,
        manifest_path=manifest_path,
        report_text_path=report_text_path,
        report_json_path=report_json_path,
        results=tuple(results),
        stopped_reason=stopped_reason,
    )


def write_optimization_report(
    manifest: Path | str | dict[str, object],
    *,
    text_path: Path | str,
    json_path: Path | str,
    title: str = "",
    charge: int | None = None,
    multiplicity: int | None = None,
    initial_geometry_source: str = "",
) -> tuple[Path, Path]:
    """Write human-readable and structured reports for a multilevel optimization."""

    payload = _load_manifest(manifest)
    report = _optimization_report_payload(
        payload,
        title=title,
        charge=charge,
        multiplicity=multiplicity,
        initial_geometry_source=initial_geometry_source,
    )
    target_json = Path(json_path)
    target_text = Path(text_path)
    target_json.parent.mkdir(parents=True, exist_ok=True)
    target_text.parent.mkdir(parents=True, exist_ok=True)
    target_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    target_text.write_text(_optimization_report_text(report), encoding="utf-8")
    return target_text, target_json


def optimization_level_from_json(payload: str | dict[str, object]) -> OptimizationLevel:
    data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    coordinates = data.get("coordinates", ())
    if isinstance(coordinates, str):
        coordinates = tuple(item.strip() for item in coordinates.split(",") if item.strip())
    return OptimizationLevel(
        name=str(data["name"]),
        backend=str(data.get("backend", "")),
        route=str(data.get("route", "")),
        method=str(data.get("method", "")),
        basis=str(data.get("basis", "")),
        executable=None if data.get("executable") is None else str(data.get("executable")),
        engine_command=str(data.get("engine_command", "")),
        coordinate_kind=str(data.get("coordinate_kind", "cartesian")),
        coordinates=tuple(str(item) for item in coordinates),
        max_steps=int(data.get("max_steps", 50)),
        trust_radius=float(data.get("trust_radius", 0.2)),
        max_trust_radius=float(data.get("max_trust_radius", 0.3)),
        gradient_tolerance=float(data.get("gradient_tolerance", 4.5e-4)),
        step_tolerance=float(data.get("step_tolerance", 1.8e-3)),
        timeout=None if data.get("timeout") is None else float(data.get("timeout")),
        use_previous_hessian=bool(data.get("use_previous_hessian", True)),
        convergence=str(data.get("convergence", "normal")),
        initial_hessian_model=str(
            data.get("initial_hessian_model", "lindh_swart_special")
        ),
        enable_gdiis=bool(data.get("enable_gdiis", True)),
        processors=int(data.get("processors", 1)),
        memory_gb=None if data.get("memory_gb") is None else int(data.get("memory_gb")),
    )


def _load_manifest(manifest: Path | str | dict[str, object]) -> dict[str, object]:
    if isinstance(manifest, dict):
        return dict(manifest)
    target = Path(manifest)
    return json.loads(target.read_text(encoding="utf-8"))


def _optimization_report_payload(
    manifest: dict[str, object],
    *,
    title: str,
    charge: int | None,
    multiplicity: int | None,
    initial_geometry_source: str,
) -> dict[str, object]:
    levels = []
    warnings: list[str] = []
    for item in manifest.get("levels", []):
        if not isinstance(item, dict):
            continue
        summary_path = Path(str(item.get("optimizer_summary", item.get("summary", ""))))
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        diagnostics = summary.get("optimizer_diagnostics", {}) if isinstance(summary, dict) else {}
        geometry_statuses = [
            step.get("geometry_status", "")
            for step in summary.get("iterations", [])
            if isinstance(step, dict) and step.get("geometry_status")
        ]
        level_warnings = []
        if any(str(status).startswith("short_contact") for status in geometry_statuses):
            level_warnings.append("short-contact warning encountered")
        rejected_trials = int(diagnostics.get("rejected_trial_count", 0)) if isinstance(diagnostics, dict) else 0
        if rejected_trials:
            level_warnings.append(f"{rejected_trials} rejected trial point(s)")
        if not bool(summary.get("converged", item.get("converged", False))):
            level_warnings.append("not converged")
        warnings.extend(f"{item.get('name', summary_path.parent.name)}: {warning}" for warning in level_warnings)
        levels.append(
            {
                "name": str(item.get("name", summary_path.parent.name)),
                "backend": str(item.get("backend", "")),
                "route": str(item.get("route", "")),
                "program": str(item.get("program", item.get("backend", ""))),
                "status": str(summary.get("status", item.get("status", "unknown"))),
                "converged": bool(summary.get("converged", item.get("converged", False))),
                "optimization_steps": int(summary.get("optimization_steps", item.get("optimization_steps", item.get("steps", 0)))),
                "final_energy_hartree": summary.get("final_energy_hartree"),
                "final_energy_change_hartree": summary.get("final_energy_change_hartree"),
                "final_gradient_inf_norm": summary.get("final_gradient_inf_norm"),
                "final_gradient_rms_norm": summary.get("final_gradient_rms_norm"),
                "final_displacement_inf_norm": summary.get("final_displacement_inf_norm"),
                "final_displacement_rms_norm": summary.get("final_displacement_rms_norm"),
                "qm_evaluations": int(summary.get("qm_evaluations", item.get("qm_evaluations", 0))),
                "energy_evaluations": int(summary.get("energy_evaluations", item.get("energy_evaluations", 0))),
                "gradient_evaluations": int(summary.get("gradient_evaluations", item.get("gradient_evaluations", 0))),
                "hessian_evaluations": int(summary.get("hessian_evaluations", item.get("hessian_evaluations", 0))),
                "fd_displacements": int(summary.get("fd_displacements", item.get("fd_displacements", 0))),
                "initial_hessian_source": str(summary.get("initial_hessian_source", "")),
                "optimizer_summary": str(summary_path),
                "optimizer_hessian": str(item.get("optimizer_hessian", "")),
                "optimized_xyzin": str(item.get("optimized_xyzin", "")),
                "warnings": level_warnings,
            }
        )
    return {
        "schema": OPTIMIZATION_REPORT_SCHEMA,
        "molecule": {
            "title": title,
            "smiles": manifest.get("smiles", ""),
            "initial_geometry_source": initial_geometry_source,
            "initial_xyzin": manifest.get("initial_xyzin", ""),
            "final_xyzin": manifest.get("final_xyzin", ""),
            "charge": charge,
            "multiplicity": multiplicity,
        },
        "chain_status": manifest.get("chain_status", "completed"),
        "stopped_reason": manifest.get("stopped_reason", ""),
        "levels": levels,
        "warnings": warnings,
    }


def _optimization_report_text(report: dict[str, object]) -> str:
    molecule = report.get("molecule", {})
    assert isinstance(molecule, dict)
    levels = [item for item in report.get("levels", []) if isinstance(item, dict)]
    lines = [
        "MATRIX Geometry Optimization Report",
        "",
        "Molecule",
        f"  Title: {molecule.get('title') or 'n/a'}",
        f"  Source: SMILES {molecule.get('smiles') or 'n/a'}",
        f"  Initial geometry: {molecule.get('initial_geometry_source') or 'n/a'}",
        f"  Charge/multiplicity: {_format_charge_multiplicity(molecule.get('charge'), molecule.get('multiplicity'))}",
        f"  Initial xyzin: {molecule.get('initial_xyzin') or 'n/a'}",
        f"  Final xyzin: {molecule.get('final_xyzin') or 'n/a'}",
    ]
    lines.append(f"  Chain status: {report.get('chain_status', 'completed')}")
    if report.get("stopped_reason"):
        lines.append(f"  Stopped reason: {report.get('stopped_reason')}")
    lines.extend(["", "Level Summary"])
    if not levels:
        lines.append("  No completed levels found.")
    else:
        lines.append(
            "  #  Level                         Program              Status        Steps  QM evals  Grad evals  Energy (Eh)"
        )
        for index, level in enumerate(levels, start=1):
            lines.append(
                "  "
                f"{index:<2d} {str(level.get('name', 'level'))[:28]:<28} "
                f"{str(level.get('program', 'n/a'))[:20]:<20} "
                f"{str(level.get('status', 'unknown'))[:12]:<12} "
                f"{int(level.get('optimization_steps', 0)):>5d} "
                f"{int(level.get('qm_evaluations', 0)):>9d} "
                f"{int(level.get('gradient_evaluations', 0)):>10d} "
                f"{_format_float(level.get('final_energy_hartree'), 12):>14}"
            )
    for index, level in enumerate(levels, start=1):
        lines.extend(
            [
                "",
                f"Level {index}: {level.get('name', 'level')}",
                f"  Backend: {_level_backend(level)}",
                f"  Status: {level.get('status')} (converged={int(bool(level.get('converged')))})",
                f"  Macro-steps: {level.get('optimization_steps')}",
                f"  QM evaluations: {level.get('qm_evaluations')} "
                f"(energy={level.get('energy_evaluations')}, gradient={level.get('gradient_evaluations')}, "
                f"hessian={level.get('hessian_evaluations')}, fd={level.get('fd_displacements')})",
                f"  Final energy: {_format_float(level.get('final_energy_hartree'), 14)} Eh",
                f"  Final max/RMS force: {_format_float(level.get('final_gradient_inf_norm'), 6)} / "
                f"{_format_float(level.get('final_gradient_rms_norm'), 6)}",
                f"  Final max/RMS displacement: {_format_float(level.get('final_displacement_inf_norm'), 6)} / "
                f"{_format_float(level.get('final_displacement_rms_norm'), 6)}",
                f"  Hessian seed: {level.get('initial_hessian_source') or 'n/a'}",
                f"  Summary: {level.get('optimizer_summary')}",
            ]
        )
        warnings = level.get("warnings", [])
        if warnings:
            lines.append("  Warnings: " + "; ".join(str(item) for item in warnings))
    warnings = report.get("warnings", [])
    lines.extend(["", "Diagnostics"])
    if warnings:
        for warning in warnings:
            lines.append(f"  WARNING: {warning}")
    else:
        lines.append("  No warnings recorded.")
    return "\n".join(lines) + "\n"


def _level_backend(level: dict[str, object]) -> str:
    backend = str(level.get("program", level.get("backend", ""))).strip()
    route = str(level.get("route", "")).strip()
    return " ".join(item for item in (backend, route) if item) or "external command"


def _level_program(level: OptimizationLevel) -> str:
    backend = str(level.backend).strip()
    executable = str(level.executable or "").strip()
    if backend:
        return f"{backend} ({Path(executable).name})" if executable else backend
    if level.engine_command.strip():
        return "external command"
    return "unknown"


def _format_charge_multiplicity(charge: object, multiplicity: object) -> str:
    left = "n/a" if charge is None else str(charge)
    right = "n/a" if multiplicity is None else str(multiplicity)
    return f"{left}/{right}"


def _format_float(value: object, precision: int) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{precision}g}"
    except (TypeError, ValueError):
        return str(value)


def _lcb26_seed_for_smiles(
    smiles: str,
    *,
    lcb26_root: Path | str | None,
) -> Path | None:
    from matrix_oracle import query_lcb26

    root = (
        Path(lcb26_root).expanduser().resolve()
        if lcb26_root is not None
        else Path(__file__).resolve().parents[4] / "data" / "lcb26"
    )
    if not (root / "enriched" / "index.json").is_file():
        return None
    matches = query_lcb26(root, smiles=smiles, limit=1)
    if not matches:
        return None
    relative = str(matches[0].get("geometry_path", "")).strip()
    if not relative:
        return None
    candidate = (root / "l1_geometries" / relative).resolve()
    if root not in candidate.parents or not candidate.is_file():
        return None
    return candidate


def _write_smiles_xyzin(
    smiles: str,
    path: Path,
    *,
    title: str,
    charge: int | None,
    multiplicity: int | None,
    random_seed: int,
) -> Path:
    from .smiles import smiles_to_geometry

    geometry = smiles_to_geometry(
        smiles,
        title=title or smiles,
        charge=charge,
        multiplicity=multiplicity,
        random_seed=random_seed,
        relaxation_method="NONE",
    )
    lines = geometry.xyz_lines()
    lines.extend(
        [
            "#SMILES",
            f"VALUE {smiles}",
            f"SWITCH_COMPATIBILITY_SEED {random_seed}",
            "SWITCH_ROLE INITIAL_CARTESIAN_EMBEDDING_ONLY",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_sonic_xyzin(
    source: Path,
    target: Path,
    *,
    source_kind: str = "enriched_xyz",
    build_sonic: bool = True,
) -> Path:
    from matrix_chem import preprocess_to_enriched_xyz, write_validation_section
    from matrix_smith import write_gicforge_build_sections

    preprocess_to_enriched_xyz(source, target, source_kind=source_kind)
    write_validation_section(target)
    if build_sonic:
        write_gicforge_build_sections(
            target, symmetrize=True, out_of_plane_model="soop"
        )
    return target


def _refresh_sonic_contract(path: Path) -> None:
    """Rebuild the complete ORACLE state, then freeze a new SONIC definition."""
    from matrix_oracle import analyze_structure
    from matrix_smith import write_gicforge_build_sections

    target = Path(path)
    with TemporaryDirectory(prefix="matrix_oracle_refresh_", dir=target.parent) as directory:
        source = Path(directory) / target.name
        shutil.copy2(target, source)
        analyze_structure(
            source,
            target,
            source_kind="enriched_xyz",
            validate=True,
        )
    write_gicforge_build_sections(
        target, symmetrize=True, out_of_plane_model="soop"
    )


def _safe_name(name: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name).strip())
    return safe.strip("_") or "level"
