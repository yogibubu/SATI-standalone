"""ORACLE geometry-refinement layers expressed in redundant primitives.

The naming follows the valence accuracy ladder used by JCP_IS1.  Core--valence
is transversal to the valence rung; BL1 and PL1 are empirical refinements of an
L1 valence baseline.  The historical BDPCS labels are not part of this API.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

import numpy as np

from .primitive_coordinates import Primitive, eval_primitives, primitive_b_matrix
from .structural_corrections import (
    cv_radial_bond_delta_angstrom,
    perceive_hydrogen_bonds,
    pl1_hbond_delta_angstrom,
)
from .pl1_model import PL1GaussianModel


class ValenceLevel(str, Enum):
    L0 = "L0"
    L1 = "L1"
    PL2 = "PL2"
    L2 = "L2"
    L3 = "L3"


class RefinementLayer(str, Enum):
    CORE_VALENCE = "CORE_VALENCE"
    BL1_CONJUGATION = "BL1_CONJUGATION"
    PL1_SELECTED_PAIR = "PL1_SELECTED_PAIR"


# Continuous pi character at and above this value belongs to the BL1
# conjugated-bond domain.  The value deliberately includes aromatic C--C
# bonds, whose geometry-only Pauling pi components are about 0.4--0.5, while
# excluding ordinary single bonds.
BL1_CONJUGATION_PI_THRESHOLD = 0.40
BL1_CONJUGATED_ELEMENT_PAIRS = frozenset({(6, 6), (6, 16)})

# Provisional class-balanced residual inferred from the independent saccharin
# PL2 structure.  The original BL1 reference set did not represent polar
# S--O/S--N environments adequately.  One shared contraction avoids fitting
# two independent one-example parameters; external sulfur compounds are
# required before this value can be promoted to a transferable calibration.
BL1_POLAR_SULFUR_PAIRS = frozenset({(7, 16), (8, 16)})
BL1_POLAR_SULFUR_RESIDUAL_ANGSTROM = -0.003310

# The current CV posterior is valence-method independent.  For an L1 DPCS3
# baseline, retain the small C--H enhancement of the original DPCS3-optimized
# implementation as a separate, explicit BL1 residual rather than folding it
# into the transversal CV layer.
BL1_DPCS3_CH_RESIDUAL_ANGSTROM = -0.000870


@dataclass(frozen=True)
class PrimitiveTarget:
    primitive_index: int
    delta: float
    layer: RefinementLayer
    source: str = ""


@dataclass(frozen=True)
class AccuracyLadderPlan:
    valence_level: ValenceLevel
    targets: tuple[PrimitiveTarget, ...]

    def __post_init__(self) -> None:
        for target in self.targets:
            if target.layer is not RefinementLayer.CORE_VALENCE and self.valence_level is not ValenceLevel.L1:
                raise ValueError(
                    f"{target.layer.value} is an L1 refinement and cannot be attached to "
                    f"the {self.valence_level.value} valence rung"
                )


@dataclass(frozen=True)
class BackTransformationResult:
    coordinates_angstrom: np.ndarray
    converged: bool
    iterations: int
    maximum_residual: float


@dataclass(frozen=True)
class PrimitiveForceFieldResult:
    energy: float
    cartesian_gradient: np.ndarray
    cartesian_hessian: np.ndarray
    primitive_residuals: np.ndarray


def core_valence_bond_shift(
    z_left: int,
    z_right: int,
    distance_angstrom: float | None = None,
    *,
    sigma_scale: float = 1.3,
    weight_threshold: float = 0.0,
) -> float:
    """Return the CV_radial a posteriori bond contraction in angstrom.

    If no distance is supplied, the correction is evaluated at the sum of the
    calibrated covalent radii, where the Gaussian locality factor is one.
    Unsupported elements return zero rather than silently extrapolating the
    second/third-period calibration.
    """
    from .structural_corrections import CV_RADIAL_COVALENT_RADII_ANGSTROM

    radius_left = CV_RADIAL_COVALENT_RADII_ANGSTROM.get(int(z_left))
    radius_right = CV_RADIAL_COVALENT_RADII_ANGSTROM.get(int(z_right))
    if radius_left is None or radius_right is None:
        return 0.0
    distance = (
        radius_left + radius_right if distance_angstrom is None else float(distance_angstrom)
    )
    result = cv_radial_bond_delta_angstrom(
        z_left,
        z_right,
        distance,
        sigma_scale=sigma_scale,
        weight_threshold=weight_threshold,
    )
    return 0.0 if result is None else float(result)


def build_accuracy_ladder_plan(
    primitives: Sequence[Primitive],
    atomic_numbers: Sequence[int],
    *,
    valence_level: ValenceLevel | str,
    include_core_valence: bool = True,
    coordinates_angstrom: np.ndarray | None = None,
    synthons=None,
    include_bl1_conjugation: bool = False,
    include_pl1_hydrogen_bonds: bool = False,
    cv_weight_threshold: float = 0.9,
    l1_targets: Iterable[PrimitiveTarget] = (),
    pl1_model: PL1GaussianModel | None = None,
) -> AccuracyLadderPlan:
    """Build an ORACLE correction plan without mixing CV and valence layers."""
    level = valence_level if isinstance(valence_level, ValenceLevel) else ValenceLevel(str(valence_level).upper())
    targets: list[PrimitiveTarget] = []
    if include_core_valence:
        for index, primitive in enumerate(primitives):
            if primitive.kind != "bond":
                continue
            left, right = primitive.atoms
            distance = None
            if coordinates_angstrom is not None:
                distance = float(
                    np.linalg.norm(
                        np.asarray(coordinates_angstrom, dtype=float)[left]
                        - np.asarray(coordinates_angstrom, dtype=float)[right]
                    )
                )
            delta = core_valence_bond_shift(
                atomic_numbers[left],
                atomic_numbers[right],
                distance,
                weight_threshold=cv_weight_threshold if distance is not None else 0.0,
            )
            if delta == 0.0:
                continue
            targets.append(
                PrimitiveTarget(
                    primitive_index=index,
                    delta=delta,
                    layer=RefinementLayer.CORE_VALENCE,
                    source="CV_RADIAL_GAUSSIAN_POSTERIOR",
                )
            )
    if include_bl1_conjugation or include_pl1_hydrogen_bonds:
        if level is not ValenceLevel.L1:
            raise ValueError("BL1/PL1 automatic corrections require the L1 valence rung")
        if coordinates_angstrom is None or synthons is None:
            raise ValueError("BL1/PL1 automatic corrections require coordinates and synthons")
        targets.extend(
            build_l1_refinement_targets(
                primitives,
                atomic_numbers,
                np.asarray(coordinates_angstrom, dtype=float),
                synthons,
                include_conjugation=include_bl1_conjugation,
                include_hydrogen_bonds=include_pl1_hydrogen_bonds,
                pl1_model=pl1_model,
            )
        )
    targets.extend(tuple(l1_targets))
    return AccuracyLadderPlan(level, tuple(targets))


def build_l1_refinement_targets(
    primitives: Sequence[Primitive],
    atomic_numbers: Sequence[int],
    coordinates_angstrom: np.ndarray,
    synthons,
    *,
    include_conjugation: bool = True,
    include_hydrogen_bonds: bool = True,
    pl1_model: PL1GaussianModel | None = None,
) -> tuple[PrimitiveTarget, ...]:
    """Build the JCP_IS1 BL1/PL1 targets on ORACLE primitives."""
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    numbers = tuple(int(value) for value in atomic_numbers)
    primitive_by_pair = {
        tuple(sorted(primitive.atoms)): index
        for index, primitive in enumerate(primitives)
        if primitive.kind in {"bond", "hbond_dist", "pseudo_bond"}
    }
    targets: list[PrimitiveTarget] = []
    if include_conjugation:
        for index, primitive in enumerate(primitives):
            if primitive.kind != "bond":
                continue
            left, right = primitive.atoms
            distance = float(np.linalg.norm(xyz[left] - xyz[right]))
            if pl1_model is not None:
                bond_order = max(1.0, min(3.0, float(synthons.bond_order(left, right))))
                delta = pl1_model.covalent_delta(numbers[left], numbers[right], bond_order, distance)
                if delta == 0.0:
                    continue
                targets.append(PrimitiveTarget(index, delta, RefinementLayer.BL1_CONJUGATION, "PL1_FITTED_COVALENT_GAUSSIAN"))
                continue
            if tuple(sorted((numbers[left], numbers[right]))) not in BL1_CONJUGATED_ELEMENT_PAIRS:
                continue
            if float(synthons.bond_order_total_pi(left, right)) < BL1_CONJUGATION_PI_THRESHOLD:
                continue
            from .topology.pykko_radii import bond_order_reference_radii
            rl = bond_order_reference_radii(numbers[left], fallback_radius=0.0)
            rr = bond_order_reference_radii(numbers[right], fallback_radius=0.0)
            double_ref = rl[1] + rr[1]
            delta_cv = core_valence_bond_shift(
                numbers[left], numbers[right], double_ref, weight_threshold=0.0
            )
            delta = _conjugation_delta_angstrom(
                numbers[left], numbers[right], distance, delta_cv
            )
            if delta != 0.0:
                targets.append(
                    PrimitiveTarget(
                        index,
                        delta,
                        RefinementLayer.BL1_CONJUGATION,
                        "JCP_IS1_COVALENT_GAUSSIAN_CONJUGATION",
                    )
                )
        if pl1_model is None:
            for index, primitive in enumerate(primitives):
                if primitive.kind != "bond":
                    continue
                left, right = primitive.atoms
                if tuple(sorted((numbers[left], numbers[right]))) not in BL1_POLAR_SULFUR_PAIRS:
                    continue
                targets.append(PrimitiveTarget(index, BL1_POLAR_SULFUR_RESIDUAL_ANGSTROM, RefinementLayer.BL1_CONJUGATION, "ORACLE_PROVISIONAL_POLAR_SULFUR_RESIDUAL"))
            for index, primitive in enumerate(primitives):
                if primitive.kind != "bond":
                    continue
                left, right = primitive.atoms
                if tuple(sorted((numbers[left], numbers[right]))) != (1, 6):
                    continue
                targets.append(PrimitiveTarget(index, BL1_DPCS3_CH_RESIDUAL_ANGSTROM, RefinementLayer.BL1_CONJUGATION, "ORACLE_DPCS3_CH_COMPENSATION"))
    if include_hydrogen_bonds:
        bonded_pairs = tuple(
            primitive.atoms for primitive in primitives if primitive.kind == "bond"
        )
        for contact in perceive_hydrogen_bonds(numbers, xyz, bonded_pairs):
            if pl1_model is None and not contact.pl1_calibrated:
                continue
            index = primitive_by_pair.get(tuple(sorted((contact.hydrogen, contact.acceptor))))
            if index is None:
                continue
            targets.append(
                PrimitiveTarget(
                    index,
                    (pl1_model.hbond_delta(
                        numbers[contact.donor], numbers[contact.acceptor],
                        contact.distance_angstrom, contact.angle_radians
                    ) if pl1_model is not None else pl1_hbond_delta_angstrom(
                        contact.distance_angstrom, contact.angle_radians
                    )),
                    RefinementLayer.PL1_SELECTED_PAIR,
                    "JCP_IS1_PL1_OH_OH_GAUSSIAN",
                )
            )
    return tuple(targets)


def target_values_from_plan(
    plan: AccuracyLadderPlan,
    primitives: Sequence[Primitive],
    coordinates_angstrom: np.ndarray,
) -> dict[int, float]:
    reference = eval_primitives(primitives, coordinates_angstrom)
    targets: dict[int, float] = {}
    for target in plan.targets:
        if target.primitive_index < 0 or target.primitive_index >= len(primitives):
            raise IndexError(f"primitive index outside ORACLE contract: {target.primitive_index}")
        targets[target.primitive_index] = (
            targets.get(target.primitive_index, float(reference[target.primitive_index])) + target.delta
        )
    return targets


def backtransform_primitive_targets(
    primitives: Sequence[Primitive],
    coordinates_angstrom: np.ndarray,
    targets: Mapping[int, float],
    *,
    primitive_weights: Mapping[int, float] | None = None,
    deformation_weights: Mapping[str, float] | None = None,
    cartesian_metric: np.ndarray | None = None,
    tolerance: float = 1.0e-8,
    max_iterations: int = 50,
    damping: float = 1.0e-10,
    maximum_cartesian_step: float = 0.15,
    allow_least_squares_projection: bool = False,
    cartesian_step_tolerance: float = 1.0e-10,
    objective_tolerance: float = 1.0e-12,
) -> BackTransformationResult:
    """Iteratively reconstruct Cartesians from selected primitive targets.

    ``allow_least_squares_projection`` is reserved for redundant target sets
    which need not be mutually realizable. In that mode a backtracked weighted
    Wilson-B step may converge to the stationary least-squares projection;
    the non-zero target residual remains explicitly reported.
    """
    xyz = np.asarray(coordinates_angstrom, dtype=float).copy()
    selected = np.asarray(sorted(int(index) for index in targets), dtype=int)
    if selected.size == 0:
        return BackTransformationResult(xyz, True, 0, 0.0)
    if selected.min() < 0 or selected.max() >= len(primitives):
        raise IndexError("primitive target lies outside the ORACLE contract")
    weights = np.asarray(
        [1.0 if primitive_weights is None else float(primitive_weights.get(int(index), 1.0)) for index in selected],
        dtype=float,
    )
    if np.any(weights <= 0.0):
        raise ValueError("primitive back-transformation weights must be positive")
    root_weights = np.sqrt(weights)
    maximum_residual = float("inf")
    for iteration in range(1, max_iterations + 1):
        values = eval_primitives(primitives, xyz)
        residual = np.asarray([float(targets[int(index)]) - values[index] for index in selected])
        for row, index in enumerate(selected):
            if primitives[int(index)].kind == "dihedral":
                residual[row] = _periodic_difference(residual[row])
        maximum_residual = float(np.max(np.abs(residual)))
        if maximum_residual <= tolerance:
            return BackTransformationResult(xyz, True, iteration - 1, maximum_residual)
        b_full = primitive_b_matrix(primitives, xyz)
        b_selected = b_full[selected, :]
        weighted_b = root_weights[:, None] * b_selected
        weighted_residual = root_weights * residual
        metric = None
        if cartesian_metric is not None:
            metric = np.asarray(cartesian_metric, dtype=float)
            expected = (xyz.size, xyz.size)
            if metric.shape != expected:
                raise ValueError(f"Cartesian deformation metric must have shape {expected}")
        elif deformation_weights is not None:
            class_weights = np.asarray(
                [float(deformation_weights.get(primitive.kind, 1.0)) for primitive in primitives]
            )
            if np.any(class_weights <= 0.0):
                raise ValueError("primitive deformation weights must be positive")
            metric = b_full.T @ (class_weights[:, None] * b_full)
        if metric is None:
            inverse_metric = np.eye(xyz.size)
        else:
            metric = 0.5 * (metric + metric.T) + damping * np.eye(xyz.size)
            inverse_metric = np.linalg.pinv(metric, rcond=1.0e-12)
        response = inverse_metric @ weighted_b.T
        normal = weighted_b @ response + damping * np.eye(selected.size)
        step = response @ np.linalg.solve(normal, weighted_residual)
        step_norm = float(np.linalg.norm(step))
        if step_norm > maximum_cartesian_step:
            step *= maximum_cartesian_step / step_norm
        if not allow_least_squares_projection:
            xyz += step.reshape(xyz.shape)
            continue
        if step_norm <= cartesian_step_tolerance:
            return BackTransformationResult(xyz, True, iteration - 1, maximum_residual)
        objective = 0.5 * float(np.dot(weighted_residual, weighted_residual))
        accepted = False
        scale = 1.0
        candidate = xyz
        candidate_objective = objective
        while scale >= 2.0**-20:
            trial = xyz + scale * step.reshape(xyz.shape)
            trial_values = eval_primitives(primitives, trial)
            trial_residual = np.asarray(
                [float(targets[int(index)]) - trial_values[index] for index in selected]
            )
            for row, index in enumerate(selected):
                if primitives[int(index)].kind == "dihedral":
                    trial_residual[row] = _periodic_difference(trial_residual[row])
            trial_weighted = root_weights * trial_residual
            trial_objective = 0.5 * float(np.dot(trial_weighted, trial_weighted))
            if trial_objective < objective:
                candidate = trial
                candidate_objective = trial_objective
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            return BackTransformationResult(xyz, True, iteration - 1, maximum_residual)
        xyz = candidate
        relative_improvement = (objective - candidate_objective) / max(1.0, objective)
        if relative_improvement <= objective_tolerance:
            values = eval_primitives(primitives, xyz)
            residual = np.asarray(
                [float(targets[int(index)]) - values[index] for index in selected]
            )
            for row, index in enumerate(selected):
                if primitives[int(index)].kind == "dihedral":
                    residual[row] = _periodic_difference(residual[row])
            return BackTransformationResult(
                xyz, True, iteration, float(np.max(np.abs(residual)))
            )
    return BackTransformationResult(xyz, False, max_iterations, maximum_residual)


def apply_accuracy_ladder_plan(
    plan: AccuracyLadderPlan,
    primitives: Sequence[Primitive],
    coordinates_angstrom: np.ndarray,
    *,
    cartesian_metric: np.ndarray | None = None,
    tolerance: float = 1.0e-8,
    max_iterations: int = 50,
) -> BackTransformationResult:
    """Apply an ORACLE L1->BL1/PL1 plan with the JCP_IS1 deformation ranking."""
    targets = target_values_from_plan(plan, primitives, coordinates_angstrom)
    has_pair_local = any(
        target.layer is RefinementLayer.PL1_SELECTED_PAIR for target in plan.targets
    )
    if has_pair_local:
        reference = eval_primitives(primitives, coordinates_angstrom)
        # PL1 fixes the already corrected covalent pattern during pair-local
        # iterations, including bonds outside the current CV calibration.
        for index, primitive in enumerate(primitives):
            if primitive.kind == "bond":
                targets.setdefault(index, float(reference[index]))
    class_weights = None
    if has_pair_local and cartesian_metric is None:
        class_weights = {
            "bond": 1000.0,
            "hbond_dist": 100.0,
            "pseudo_bond": 100.0,
            "angle": 100.0,
            "linear_bend": 100.0,
            "dihedral": 1.0,
            "out_of_plane": 10.0,
        }
    return backtransform_primitive_targets(
        primitives,
        coordinates_angstrom,
        targets,
        deformation_weights=class_weights,
        cartesian_metric=cartesian_metric,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def evaluate_primitive_force_field(
    primitives: Sequence[Primitive],
    coordinates_angstrom: np.ndarray,
    equilibrium_values: Sequence[float],
    force_constants: Sequence[float],
) -> PrimitiveForceFieldResult:
    """Evaluate a diagonal harmonic field in redundant ORACLE primitives."""
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    equilibrium = np.asarray(equilibrium_values, dtype=float)
    force = np.asarray(force_constants, dtype=float)
    if equilibrium.shape != (len(primitives),) or force.shape != (len(primitives),):
        raise ValueError("equilibrium values and force constants must match the primitive count")
    if np.any(force < 0.0):
        raise ValueError("primitive force constants cannot be negative")
    residual = eval_primitives(primitives, xyz) - equilibrium
    for index, primitive in enumerate(primitives):
        if primitive.kind == "dihedral":
            residual[index] = _periodic_difference(residual[index])
    b_matrix = primitive_b_matrix(primitives, xyz)
    weighted = force * residual
    gradient = (b_matrix.T @ weighted).reshape(xyz.shape)
    hessian = b_matrix.T @ (force[:, None] * b_matrix)
    return PrimitiveForceFieldResult(
        energy=0.5 * float(np.dot(residual, weighted)),
        cartesian_gradient=gradient,
        cartesian_hessian=hessian,
        primitive_residuals=residual,
    )


def _conjugation_delta_angstrom(
    z_left: int,
    z_right: int,
    distance_angstrom: float,
    delta_cv_angstrom: float,
) -> float:
    """JCP_IS1 C=C/C=S Gaussian that cancels CV at the double-bond radius."""
    from .topology.pykko_radii import bond_order_reference_radii

    try:
        left = bond_order_reference_radii(int(z_left), fallback_radius=0.0)
        right = bond_order_reference_radii(int(z_right), fallback_radius=0.0)
    except ValueError:
        return 0.0
    single = left[0] + right[0]
    double = left[1] + right[1]
    triple = left[2] + right[2]
    spacings = [abs(double - single), abs(triple - double)]
    # Some established coordination references (notably C--S) have a
    # degenerate double/triple sum.  Retain the published 1/1.5 width rule
    # using the remaining non-zero spacing instead of silently disabling Conj.
    positive_spacings = [value for value in spacings if value > 1.0e-12]
    if not positive_spacings:
        return 0.0
    width = min(positive_spacings) / 1.5
    return float(
        -delta_cv_angstrom * np.exp(-((float(distance_angstrom) - double) / width) ** 2)
    )


def _periodic_difference(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)
