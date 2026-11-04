"""Adaptive pseudo-arclength continuation for nonperiodic tunneling paths."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .qim_optimizer import ConstrainedQimPoint, _optimize_point, mass_orthogonal_basis


LINK_CURVILINEAR_PATH_SCHEMA = "matrix.link.curvilinear_tunneling_optimization.v1"


@dataclass(frozen=True)
class CurvilinearContinuationStep:
    index: int
    step_sqrt_amu_angstrom: float
    energy_hartree: float
    transverse_gradient_norm: float
    iterations: int
    function_evaluations: int
    turning_angle_degrees: float
    retries: int


@dataclass(frozen=True)
class CurvilinearOptimizationResult:
    path: object
    steps: tuple[CurvilinearContinuationStep, ...]
    reached_outer_wall: bool
    electronic_calls: int = 0
    cache_hits: int = 0
    schema: str = LINK_CURVILINEAR_PATH_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "path": self.path.to_dict(),
            "reached_outer_wall": self.reached_outer_wall,
            "electronic_calls": self.electronic_calls,
            "cache_hits": self.cache_hits,
            "steps": [step.__dict__ for step in self.steps],
        }


def _cartesian_masses(contract: object, atom_count: int) -> np.ndarray:
    values = np.asarray(getattr(contract, "cartesian_masses"), dtype=float).reshape(-1)
    if values.shape != (3 * atom_count,):
        raise ValueError("the seed path and SMITH Cartesian masses are inconsistent")
    return values


def _normalized_secant(first: np.ndarray, second: np.ndarray, masses: np.ndarray) -> np.ndarray:
    vector = np.asarray(second, dtype=float).reshape(-1) - np.asarray(first, dtype=float).reshape(-1)
    norm = float(np.sqrt(np.dot(masses * vector, vector)))
    if norm <= 1.0e-12:
        raise ValueError("adaptive continuation requires distinct seed images")
    return vector / norm


def optimize_curvilinear_tunneling_path(
    session: object,
    contract: object,
    seed_points: Sequence[ConstrainedQimPoint],
    *,
    reversing_action: np.ndarray,
    preconditioner_hessian_hartree_per_bohr2: np.ndarray | None = None,
    initial_step: float | None = None,
    minimum_step: float = 0.03,
    maximum_step: float = 0.30,
    maximum_images: int = 24,
    maximum_turn_angle_degrees: float = 35.0,
    max_iterations: int = 160,
    gradient_tolerance: float = 2.0e-5,
    trust_radius: float = 0.25,
    outer_wall_hartree: float = 0.012,
    checkpoint_path: Path | str | None = None,
    label: str = "s",
) -> CurvilinearOptimizationResult:
    """Continue a relaxed seed through its minimum using tangent hyperplanes.

    Each corrector minimizes every degree of freedom perpendicular to the
    current mass-normalized tangent.  Failed correctors reduce the predictor
    step and retry from the last accepted image; accepted images are never
    recomputed after restart.
    """

    if len(seed_points) < 3 or maximum_images < len(seed_points):
        raise ValueError("adaptive continuation needs at least three seed images")
    if (
        not 0.0 < minimum_step <= maximum_step
        or outer_wall_hartree <= 0.0
        or not 0.0 < maximum_turn_angle_degrees < 90.0
    ):
        raise ValueError("adaptive step limits and outer-wall height must be positive")
    ordered = tuple(sorted(seed_points, key=lambda point: point.q_im))
    center_candidates = [index for index, point in enumerate(ordered) if abs(point.q_im) < 1.0e-12]
    if len(center_candidates) != 1:
        raise ValueError("seed images must contain exactly one saddle at q=0")
    center_seed = center_candidates[0]
    positive_seed = list(ordered[center_seed:])
    if len(positive_seed) < 3 or not all(point.converged for point in positive_seed):
        raise ValueError("the positive seed branch needs three converged images")
    atom_count = np.asarray(positive_seed[0].coordinates_angstrom).shape[0]
    masses = _cartesian_masses(contract, atom_count)
    hessian = None
    if preconditioner_hessian_hartree_per_bohr2 is not None:
        hessian = np.asarray(preconditioner_hessian_hartree_per_bohr2, dtype=float)
        if hessian.shape != (3 * atom_count, 3 * atom_count) or not np.all(np.isfinite(hessian)):
            raise ValueError("the Cartesian preconditioner Hessian has the wrong dimension")
    atom_masses = masses.reshape((-1, 3))[:, 0]
    from matrix_smith import build_curvilinear_tunneling_path, mass_eckart_align

    reference = np.asarray(positive_seed[0].coordinates_angstrom, dtype=float)
    geometries = [
        mass_eckart_align(point.coordinates_angstrom, reference, atom_masses)
        for point in positive_seed
    ]
    energies = [float(point.energy_hartree) for point in positive_seed]
    restored = _read_checkpoint(checkpoint_path, atom_count)
    diagnostics: list[CurvilinearContinuationStep] = []
    if restored:
        restored_count = len(restored)
        retained_rows = []
        for row in restored:
            diagnostic = row.get("diagnostics")
            if (
                diagnostic is not None
                and float(diagnostic["turning_angle_degrees"]) > maximum_turn_angle_degrees
            ):
                break
            retained_rows.append(row)
        restored = retained_rows
        geometries = [np.asarray(row["coordinates_angstrom"], dtype=float) for row in restored]
        energies = [float(row["energy_hartree"]) for row in restored]
        diagnostics = [
            CurvilinearContinuationStep(**row["diagnostics"])
            for row in restored
            if row.get("diagnostics") is not None
        ]
        if len(restored) != restored_count:
            _write_checkpoint(checkpoint_path, geometries, energies, diagnostics)
    step = float(initial_step) if initial_step is not None else float(
        np.sqrt(np.dot(masses * (geometries[-1].reshape(-1) - geometries[-2].reshape(-1)),
                       geometries[-1].reshape(-1) - geometries[-2].reshape(-1)))
    )
    step = float(np.clip(step, minimum_step, maximum_step))
    if diagnostics:
        previous = diagnostics[-1]
        if previous.turning_angle_degrees > 12.0 or previous.iterations > 35:
            step = max(minimum_step, 0.65 * step)
        elif previous.turning_angle_degrees < 7.5 and previous.iterations < 20:
            step = min(maximum_step, 1.20 * step)
    reached_wall = False
    minimum_index = int(np.argmin(energies))
    if (
        minimum_index < len(energies) - 1
        and energies[-1] - energies[minimum_index] >= outer_wall_hartree
    ):
        reached_wall = True
    while len(geometries) < maximum_images:
        if reached_wall:
            break
        old_tangent = _normalized_secant(geometries[-2], geometries[-1], masses)
        retries = 0
        while True:
            predictor = geometries[-1].reshape(-1) + step * old_tangent
            basis = mass_orthogonal_basis(
                old_tangent,
                masses,
                reference_coordinates_angstrom=predictor.reshape((-1, 3)),
            )
            if hessian is None:
                coordinate_scales = np.full(len(basis), 0.10)
            else:
                curvatures = (
                    np.einsum("ia,ab,ib->i", basis, hessian, basis)
                    * 1.8897261246257702**2
                )
                coordinate_scales = np.clip(
                    0.05 / np.sqrt(np.maximum(np.abs(curvatures), 1.0e-8)),
                    0.02,
                    0.50,
                )
            point = _optimize_point(
                session,
                predictor,
                old_tangent,
                basis,
                0.0,
                np.zeros(len(basis)),
                max_iterations=max_iterations,
                gradient_tolerance=gradient_tolerance,
                trust_radius=trust_radius,
                sample_dipole=False,
                dipole_evaluator=None,
                coordinate_scales=coordinate_scales,
                mobile_trust_region=False,
            )
            if point.converged:
                accepted = mass_eckart_align(point.coordinates_angstrom, reference, atom_masses)
                new_tangent = _normalized_secant(geometries[-1], accepted, masses)
                cosine = float(
                    np.clip(np.dot(masses * old_tangent, new_tangent), -1.0, 1.0)
                )
                angle = float(np.degrees(np.arccos(cosine)))
                if angle <= maximum_turn_angle_degrees:
                    break
            retries += 1
            step *= 0.5
            if step < minimum_step:
                reason = (
                    f"turning angle={angle:.2f} degrees"
                    if point.converged
                    else f"transverse gradient={point.orthogonal_gradient_norm:.3e}"
                )
                raise RuntimeError(
                    "curvilinear corrector failed at the minimum adaptive step; " + reason
                )
        geometries.append(accepted)
        energies.append(float(point.energy_hartree))
        diagnostics.append(
            CurvilinearContinuationStep(
                index=len(geometries) - 1,
                step_sqrt_amu_angstrom=step,
                energy_hartree=float(point.energy_hartree),
                transverse_gradient_norm=point.orthogonal_gradient_norm,
                iterations=point.iterations,
                function_evaluations=point.function_evaluations,
                turning_angle_degrees=angle,
                retries=retries,
            )
        )
        _write_checkpoint(checkpoint_path, geometries, energies, diagnostics)
        minimum_index = int(np.argmin(energies))
        if (
            minimum_index < len(energies) - 1
            and energies[-1] - energies[minimum_index] >= outer_wall_hartree
        ):
            reached_wall = True
            break
        if angle > 12.0 or point.iterations > 35:
            step *= 0.65
        elif angle < 7.5 and point.iterations < 20:
            step *= 1.20
        step = float(np.clip(step, minimum_step, maximum_step))

    action = np.asarray(reversing_action, dtype=float)
    if action.shape != (reference.size, reference.size):
        raise ValueError("the ORACLE reversing action has the wrong Cartesian dimension")
    negative_geometries = [
        (action @ geometry.reshape(-1)).reshape((-1, 3)) for geometry in geometries[1:][::-1]
    ]
    full_geometries = [*negative_geometries, *geometries]
    full_energies = [*energies[1:][::-1], *energies]
    center_index = len(negative_geometries)
    path = build_curvilinear_tunneling_path(
        full_geometries,
        full_energies,
        atom_masses,
        center_index=center_index,
        label=label,
        metadata={
            "producer": "LINK adaptive pseudo-arclength continuation",
            "gradient_tolerance_hartree_per_sqrt_amu_bohr": gradient_tolerance,
            "outer_wall_hartree": outer_wall_hartree,
            "reached_outer_wall": reached_wall,
        },
    )
    return CurvilinearOptimizationResult(
        path=path,
        steps=tuple(diagnostics),
        reached_outer_wall=reached_wall,
        electronic_calls=int(getattr(session, "electronic_calls", 0)),
        cache_hits=int(getattr(session, "cache_hits", 0)),
    )


def _read_checkpoint(path: Path | str | None, atom_count: int) -> list[Mapping[str, object]]:
    if path is None or not Path(path).is_file():
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != LINK_CURVILINEAR_PATH_SCHEMA:
        raise ValueError("unsupported curvilinear continuation checkpoint")
    rows = list(payload.get("positive_branch", ()))
    if any(np.asarray(row["coordinates_angstrom"]).shape != (atom_count, 3) for row in rows):
        raise ValueError("curvilinear checkpoint atom count changed")
    return rows


def _write_checkpoint(
    path: Path | str | None,
    geometries: Sequence[np.ndarray],
    energies: Sequence[float],
    diagnostics: Sequence[CurvilinearContinuationStep],
) -> None:
    if path is None:
        return
    seed_count = len(geometries) - len(diagnostics)
    rows = []
    for index, (geometry, energy) in enumerate(zip(geometries, energies, strict=True)):
        diagnostic = None if index < seed_count else diagnostics[index - seed_count].__dict__
        rows.append(
            {
                "coordinates_angstrom": np.asarray(geometry).tolist(),
                "energy_hartree": float(energy),
                "diagnostics": diagnostic,
            }
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"schema": LINK_CURVILINEAR_PATH_SCHEMA, "positive_branch": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def write_curvilinear_optimization_result(
    path: Path | str, result: CurvilinearOptimizationResult
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target


__all__ = [
    "CurvilinearContinuationStep",
    "CurvilinearOptimizationResult",
    "LINK_CURVILINEAR_PATH_SCHEMA",
    "optimize_curvilinear_tunneling_path",
    "write_curvilinear_optimization_result",
]
