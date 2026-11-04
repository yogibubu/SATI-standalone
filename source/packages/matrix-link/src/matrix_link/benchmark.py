from __future__ import annotations

from dataclasses import dataclass
import json
import sys
from pathlib import Path

import numpy as np

from .optimizer import OptimizerSettings, coordinate_model_from_xyzin, optimize_geometry


@dataclass(frozen=True)
class OptimizerBenchmarkCase:
    name: str
    atoms: tuple[str, ...]
    start_coordinates_angstrom: np.ndarray
    target_coordinates_angstrom: np.ndarray


@dataclass(frozen=True)
class OptimizerBenchmarkRun:
    case: str
    coordinate_kind: str
    policy: str
    converged: bool
    status: str
    steps: int
    qm_evaluations: int
    energy_evaluations: int
    gradient_evaluations: int
    hessian_evaluations: int
    fd_displacements: int
    gaussian_equivalent_steps: int
    final_gradient_inf_norm: float
    final_displacement_rms_angstrom: float
    summary_path: Path


@dataclass(frozen=True)
class OptimizerBenchmarkReport:
    runs: tuple[OptimizerBenchmarkRun, ...]
    json_path: Path
    markdown_path: Path


def default_optimizer_benchmark_cases() -> tuple[OptimizerBenchmarkCase, ...]:
    return (
        _case(
            "h2o",
            ("O", "H", "H"),
            ((0.000, 0.000, 0.000), (0.000, 0.000, 1.030), (0.970, 0.000, -0.255)),
            ((0.000, 0.000, 0.000), (0.000, 0.000, 0.958), (0.926, 0.000, -0.239)),
        ),
        _case(
            "nh3",
            ("N", "H", "H", "H"),
            ((0.000, 0.000, 0.070), (0.955, 0.000, -0.310), (-0.478, 0.827, -0.310), (-0.478, -0.827, -0.310)),
            ((0.000, 0.000, 0.050), (0.935, 0.000, -0.310), (-0.468, 0.811, -0.310), (-0.468, -0.811, -0.310)),
        ),
        _case(
            "h2co",
            ("C", "O", "H", "H"),
            ((0.000, 0.000, 0.000), (1.260, 0.000, 0.000), (-0.610, 0.940, 0.000), (-0.610, -0.940, 0.000)),
            ((0.000, 0.000, 0.000), (1.205, 0.000, 0.000), (-0.585, 0.935, 0.000), (-0.585, -0.935, 0.000)),
        ),
    )


def run_optimizer_validation_benchmark(
    run_dir: Path | str,
    *,
    cases: tuple[OptimizerBenchmarkCase, ...] | None = None,
    max_steps: int = 20,
    include_sonic: bool = False,
) -> OptimizerBenchmarkReport:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    selected = cases or default_optimizer_benchmark_cases()
    runs: list[OptimizerBenchmarkRun] = []
    for case in selected:
        case_dir = root / case.name
        for coordinate_kind in (("cartesian", "sonic") if include_sonic else ("cartesian",)):
            xyzin, coordinates = _write_case_optimizer_input(case_dir, case, coordinate_kind=coordinate_kind)
            framed_case = _case_in_optimizer_frame(
                case,
                xyzin=xyzin,
                coordinate_kind=coordinate_kind,
            )
            evaluator = _write_quadratic_evaluator(
                case_dir / coordinate_kind / "quadratic_point.py",
                framed_case,
            )
            model = coordinate_model_from_xyzin(xyzin, kind=coordinate_kind, coordinates=coordinates)
            initial_hessian = np.eye(len(model.labels), dtype=float) * 2.0
            for policy, settings in (
                (
                    "fd_full",
                    OptimizerSettings(
                        max_steps=max_steps,
                        trust_radius=0.2,
                        gradient_tolerance=1.0e-6,
                        fd_step=0.01,
                        cache_tolerance=1.0e-12,
                        prefer_analytic_gradient=False,
                    ),
                ),
                (
                    "fd_selective",
                    OptimizerSettings(
                        max_steps=max_steps,
                        trust_radius=0.2,
                        gradient_tolerance=1.0e-6,
                        fd_step=0.01,
                        cache_tolerance=1.0e-12,
                        prefer_analytic_gradient=False,
                        selective_fd_refresh=True,
                        selective_min_refresh_fraction=0.34,
                        fd_refresh_interval=4,
                    ),
                ),
            ):
                result = optimize_geometry(
                    xyzin,
                    run_dir=case_dir / coordinate_kind / policy,
                    coordinate_model=model,
                    allow_non_sonic_coordinates=coordinate_kind == "cartesian",
                    engine_command=f"{sys.executable} {evaluator} " + "{xyz} {result} {index}",
                    settings=settings,
                    initial_hessian=initial_hessian,
                    initial_hessian_source="quadratic-reference",
                )
                rms = float(
                    np.sqrt(
                        np.mean(
                            (
                                np.asarray(result.final_coordinates_angstrom, dtype=float)
                                - np.asarray(framed_case.target_coordinates_angstrom, dtype=float)
                            )
                            ** 2
                        )
                    )
                )
                runs.append(
                    OptimizerBenchmarkRun(
                        case=case.name,
                        coordinate_kind=coordinate_kind,
                        policy=policy,
                        converged=bool(result.converged),
                        status=result.status,
                        steps=len(result.iterations),
                        qm_evaluations=result.qm_evaluations,
                        energy_evaluations=result.energy_evaluations,
                        gradient_evaluations=result.gradient_evaluations,
                        hessian_evaluations=result.hessian_evaluations,
                        fd_displacements=result.fd_displacements,
                        gaussian_equivalent_steps=len(result.iterations),
                        final_gradient_inf_norm=float(np.max(np.abs(result.final_gradient))),
                        final_displacement_rms_angstrom=rms,
                        summary_path=result.summary_path,
                    )
                )
    json_path = _write_benchmark_json(root / "optimizer_validation_benchmark.json", runs)
    markdown_path = _write_benchmark_markdown(root / "optimizer_validation_benchmark.md", runs)
    return OptimizerBenchmarkReport(tuple(runs), json_path=json_path, markdown_path=markdown_path)


def _case(
    name: str,
    atoms: tuple[str, ...],
    start: tuple[tuple[float, float, float], ...],
    target: tuple[tuple[float, float, float], ...],
) -> OptimizerBenchmarkCase:
    return OptimizerBenchmarkCase(
        name=name,
        atoms=atoms,
        start_coordinates_angstrom=np.asarray(start, dtype=float),
        target_coordinates_angstrom=np.asarray(target, dtype=float),
    )


def _write_case_xyzin(path: Path, case: OptimizerBenchmarkCase) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(case.atoms)), f"{case.name} optimizer benchmark"]
    for atom, xyz in zip(case.atoms, case.start_coordinates_angstrom, strict=True):
        lines.append(f"{atom:<2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_case_optimizer_input(
    case_dir: Path,
    case: OptimizerBenchmarkCase,
    *,
    coordinate_kind: str,
) -> tuple[Path, tuple[str, ...]]:
    if coordinate_kind == "cartesian":
        return _write_case_xyzin(case_dir / f"{case.name}.xyzin", case), ()
    xyz = _write_case_xyzin(case_dir / f"{case.name}.xyz", case)
    xyzin = case_dir / f"{case.name}.sonic.xyzin"
    try:
        from matrix_chem import preprocess_to_enriched_xyz, write_validation_section
        from matrix_smith import write_gicforge_build_sections
    except ImportError as exc:
        raise RuntimeError("SONIC optimizer benchmark needs matrix_chem and matrix_smith") from exc
    preprocess_to_enriched_xyz(xyz, xyzin)
    write_validation_section(xyzin)
    definition = write_gicforge_build_sections(
        xyzin, symmetrize=False, out_of_plane_model="soop"
    )
    names = tuple(str(gic.name) for gic in definition.gics)
    labels = tuple(str(gic.identifier) for gic in definition.gics)
    return xyzin, names or labels


def _case_in_optimizer_frame(
    case: OptimizerBenchmarkCase,
    *,
    xyzin: Path,
    coordinate_kind: str,
) -> OptimizerBenchmarkCase:
    """Express the benchmark target in the Cartesian frame persisted by ORACLE."""

    if coordinate_kind == "cartesian":
        return case
    from matrix_chem import read_enriched_xyz

    persisted_start = np.asarray(read_enriched_xyz(xyzin).coordinates_angstrom, dtype=float)
    source_start = np.asarray(case.start_coordinates_angstrom, dtype=float)
    source_center = np.mean(source_start, axis=0)
    persisted_center = np.mean(persisted_start, axis=0)
    covariance = (source_start - source_center).T @ (persisted_start - persisted_center)
    left, _singular, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    transformed_target = (
        np.asarray(case.target_coordinates_angstrom, dtype=float) - source_center
    ) @ rotation + persisted_center
    return OptimizerBenchmarkCase(
        name=case.name,
        atoms=case.atoms,
        start_coordinates_angstrom=persisted_start,
        target_coordinates_angstrom=transformed_target,
    )


def _write_quadratic_evaluator(path: Path, case: OptimizerBenchmarkCase) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    target_path = path.with_name("target_coordinates.json")
    target_path.write_text(
        json.dumps(np.asarray(case.target_coordinates_angstrom, dtype=float).tolist()) + "\n",
        encoding="utf-8",
    )
    script = [
        "import json, sys",
        "from pathlib import Path",
        f"target = json.loads(Path({str(target_path)!r}).read_text(encoding='utf-8'))",
        "xyz = Path(sys.argv[1])",
        "out = Path(sys.argv[2])",
        "coords = []",
        "for raw in xyz.read_text(encoding='utf-8').splitlines()[2:]:",
        "    parts = raw.split()",
        "    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])",
        "energy = 0.0",
        "for row, ref in zip(coords, target):",
        "    for x, x0 in zip(row, ref):",
        "        energy += (x - x0) ** 2",
        "payload = {'schema': 'oracle.link.point_result.v1', 'point_index': int(sys.argv[3]), 'displacement': 0.0, 'energy_hartree': energy}",
        "out.write_text(json.dumps(payload) + '\\n', encoding='utf-8')",
    ]
    path.write_text("\n".join(script) + "\n", encoding="utf-8")
    return path


def _write_benchmark_json(path: Path, runs: list[OptimizerBenchmarkRun]) -> Path:
    payload = {
        "schema": "matrix.link.optimizer_validation_benchmark.v1",
        "runs": [
            {
                "case": item.case,
                "coordinate_kind": item.coordinate_kind,
                "policy": item.policy,
                "converged": item.converged,
                "status": item.status,
                "steps": item.steps,
                "gaussian_equivalent_steps": item.gaussian_equivalent_steps,
                "qm_evaluations": item.qm_evaluations,
                "energy_evaluations": item.energy_evaluations,
                "gradient_evaluations": item.gradient_evaluations,
                "hessian_evaluations": item.hessian_evaluations,
                "fd_displacements": item.fd_displacements,
                "final_gradient_inf_norm": item.final_gradient_inf_norm,
                "final_displacement_rms_angstrom": item.final_displacement_rms_angstrom,
                "summary_path": str(item.summary_path),
            }
            for item in runs
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_benchmark_markdown(path: Path, runs: list[OptimizerBenchmarkRun]) -> Path:
    lines = [
        "# Optimizer Validation Benchmark",
        "",
        "| case | coordinates | policy | converged | Gaussian-equivalent steps | QM evals | energies | gradients | FD displacements | RMS geometry error (Angstrom) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in runs:
        lines.append(
            f"| {item.case} | {item.coordinate_kind} | {item.policy} | {int(item.converged)} | "
            f"{item.gaussian_equivalent_steps} | {item.qm_evaluations} | "
            f"{item.energy_evaluations} | {item.gradient_evaluations} | "
            f"{item.fd_displacements} | {item.final_displacement_rms_angstrom:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
