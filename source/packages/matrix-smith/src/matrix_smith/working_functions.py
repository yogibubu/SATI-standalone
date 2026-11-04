"""Small opt-in library of parameter-free working functions for SONIC.

Uses the existing derived-coordinate algebra and half-angle implementation.
Native distances are angstrom and angles radians; Gaussian R expressions are
converted from bohr. Reference geometries are fixed origins, not fit parameters.
This module does not change coordinate selection or optimizer defaults.
"""

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping
import re

import numpy as np

from matrix_chem import BOHR_TO_ANGSTROM

from .bmatrix import SparseBRow
from .coordinate_functions import (
    cosine, evaluate_coordinate_function, inverse_distance, proton_transfer, sum_coordinates,
)
from .coordinate_registry import CoordinateSignature, primitive_signature
from .models import FrozenGIC, GICDefinition
from .trigonometric import HalfAngleCoordinate


WORKING_FUNCTION_CATALOG = MappingProxyType({
    "inverse_stretch": "1/r",
    "cosine_bend": "cos(theta), optionally centered at cos(theta0)",
    "half_angle_flap": "2*sin((theta-theta0)/2)",
    "proton_transfer": "r(D,H)-r(H,A), with r(D,H)+r(H,A) as optional companion",
})

_BICOORDINATED_HYDROGEN_PROVIDERS = frozenset({
    "ORACLE_SHARED_PROTON_LONE_PAIR_CENTERS",
    "ORACLE_TRANSFERABLE_HYDROGEN_CENTERS",
})


@dataclass(frozen=True)
class WorkingFunctionEvaluation:
    value: float
    b_row: SparseBRow
    b_prime: np.ndarray | None


@dataclass(frozen=True)
class WorkingCoordinateFunction:
    family: str
    sources: tuple[CoordinateSignature, ...]
    reference_radian: float | None = None

    def __post_init__(self):
        expected = {
            "inverse_stretch": ("R",), "cosine_bend": ("A",),
            "half_angle_flap": ("D",), "proton_transfer": ("R", "R"),
            "proton_transfer_sum": ("R", "R"),
        }
        if self.family not in expected:
            raise ValueError(f"unknown working-function family: {self.family}")
        if tuple(s.operator for s in self.sources) != expected[self.family]:
            raise ValueError("working function requires the indicated primitive types")
        for source in self.sources:
            size = {"R": 2, "A": 3, "D": 4}[source.operator]
            if (
                source.kind != "PRIMITIVE" or len(source.atoms) != size
                or len(set(source.atoms)) != size or min(source.atoms) < 1
                or source.mode or source.ref_atoms
            ):
                raise ValueError("invalid primitive support")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("working-function sources must be distinct")
        if self.family == "half_angle_flap" and self.reference_radian is None:
            raise ValueError("half-angle flap requires a fixed reference")
        if self.reference_radian is not None:
            if self.family not in {"half_angle_flap", "cosine_bend"}:
                raise ValueError("only angular functions accept a reference angle")
            if not np.isfinite(self.reference_radian):
                raise ValueError("reference angle must be finite")
            if self.family == "cosine_bend" and not 0 < self.reference_radian < np.pi:
                raise ValueError("cosine bend reference must lie between zero and pi")

    @property
    def units(self):
        if self.family == "inverse_stretch":
            return "angstrom^-1"
        if self.family in {"proton_transfer", "proton_transfer_sum"}:
            return "angstrom"
        return "dimensionless"

    def evaluate(
        self,
        values: Mapping[CoordinateSignature, float],
        b_rows: Mapping[CoordinateSignature, SparseBRow],
        b_primes: Mapping[CoordinateSignature, np.ndarray] | None = None,
    ) -> WorkingFunctionEvaluation:
        """Evaluate f, B and optionally B' from primitive values and derivatives."""
        args = np.array([values[s] for s in self.sources], dtype=float)
        rows = tuple(b_rows[s] for s in self.sources)
        if not np.all(np.isfinite(args)):
            raise ValueError("primitive values must be finite")
        for source, value in zip(self.sources, args, strict=True):
            if source.operator == "R" and value <= 1e-12:
                raise ValueError("distance must be strictly positive and nonzero")

        hessian = np.zeros((len(args), len(args)))
        if self.family == "inverse_stretch":
            definition = inverse_distance(self.sources[0])
            gradient = np.array([-1 / args[0]**2])
            hessian[0, 0] = 2 / args[0]**3
        elif self.family == "cosine_bend":
            if not 0 < args[0] < np.pi or abs(np.sin(args[0])) < 1e-6:
                raise ValueError("cosine bending coordinate loses rank near linearity")
            definition = cosine(self.sources[0])
            gradient = np.array([-np.sin(args[0])])
            hessian[0, 0] = -np.cos(args[0])
        elif self.family == "half_angle_flap":
            value, slope, curvature = HalfAngleCoordinate(self.reference_radian).evaluate_periodic(
                args[0]
            )
            gradient = np.array([float(slope)])
            hessian[0, 0] = curvature
            definition = None
            row = SparseBRow.combine((float(slope), rows[0]))
        else:
            if self.family == "proton_transfer":
                definition = proton_transfer(*self.sources)
                gradient = np.array([1., -1.])
            else:
                definition = sum_coordinates(*self.sources)
                gradient = np.array([1., 1.])
        if definition is not None:
            evaluated = evaluate_coordinate_function(definition, values, b_rows)
            value, row = evaluated.value, evaluated.b_row
            if self.family == "cosine_bend" and self.reference_radian is not None:
                value -= np.cos(self.reference_radian)
        second = None
        if b_primes is not None:
            # B' is explicitly a Cartesian Hessian product and is therefore
            # an audit/second-derivative request.  First-order production
            # evaluation keeps the Wilson rows sparse throughout.
            dense = np.stack([row.to_dense() for row in rows])
            if not np.all(np.isfinite(dense)):
                raise ValueError("primitive B rows must be finite")
            second = dense.T @ hessian @ dense
            for source, weight in zip(self.sources, gradient, strict=True):
                primitive_second = np.asarray(b_primes[source], dtype=float)
                if primitive_second.shape != second.shape or not np.all(np.isfinite(primitive_second)):
                    raise ValueError("primitive B-prime must be finite and match the Cartesian size")
                second += weight * primitive_second
        return WorkingFunctionEvaluation(float(value), row, second)

    def gaussian_expression(self):
        """Return a ReadAllGIC expression with the same native numerical units."""
        expressions = []
        for source in self.sources:
            expression = f"{source.operator}(" + ",".join(map(str, source.atoms)) + ")"
            if source.operator == "R":
                expression = f"({BOHR_TO_ANGSTROM:.17g}*{expression})"
            expressions.append(expression)
        if self.family == "inverse_stretch":
            return f"1/({expressions[0]})"
        if self.family == "cosine_bend":
            expression = f"COS({expressions[0]})"
            if self.reference_radian is not None:
                expression += f"-({np.cos(self.reference_radian):.17g})"
            return expression
        if self.family == "half_angle_flap":
            return HalfAngleCoordinate(self.reference_radian).gaussian_expression(expressions[0])
        sign = "-" if self.family == "proton_transfer" else "+"
        return expressions[0] + sign + expressions[1]


def inverse_stretch(first: int, second: int):
    return WorkingCoordinateFunction("inverse_stretch", (primitive_signature("R", (first, second)),))


def cosine_bend(first: int, center: int, last: int, *, reference_radian=None):
    return WorkingCoordinateFunction(
        "cosine_bend", (primitive_signature("A", (first, center, last)),), reference_radian
    )


def half_angle_flap(first: int, hinge_first: int, hinge_last: int, last: int, *, reference_radian):
    return WorkingCoordinateFunction(
        "half_angle_flap",
        (primitive_signature("D", (first, hinge_first, hinge_last, last)),),
        reference_radian,
    )


def proton_transfer_coordinate(donor: int, hydrogen: int, acceptor: int):
    if len({donor, hydrogen, acceptor}) != 3:
        raise ValueError("donor, hydrogen and acceptor must be distinct atoms")
    return WorkingCoordinateFunction("proton_transfer", (
        primitive_signature("R", (donor, hydrogen)),
        primitive_signature("R", (hydrogen, acceptor)),
    ))


def proton_transfer_coordinates(donor: int, hydrogen: int, acceptor: int):
    """Return a nonredundant difference/sum chart for later analysis."""
    difference = proton_transfer_coordinate(donor, hydrogen, acceptor)
    return difference, WorkingCoordinateFunction("proton_transfer_sum", difference.sources)


def augment_shared_proton_coordinates(definition: GICDefinition) -> GICDefinition:
    """Add a redundant proton-transfer row for every shared hydrogen.

    Both donor--H and acceptor--H distances already present in the field are
    retained.  The added difference ``r(D,H)-r(H,A)`` is a physical ZAFF term,
    not a replacement coordinate.  Consequently the number of rows may exceed
    the Wilson tangent rank; reduction to an independent chart belongs to the
    later normal-mode or displacement construction.
    """

    groups: dict[str, list[object]] = {}
    for primitive in definition.primitives:
        if (
            primitive.family == "MULTICENTER_DISTANCE"
            and any(
                provider in primitive.provenance
                for provider in _BICOORDINATED_HYDROGEN_PROVIDERS
            )
        ):
            groups.setdefault(primitive.semantic_id, []).append(primitive)
    pairs = []
    for domain_id, primitives in sorted(groups.items()):
        if len(primitives) != 2:
            raise ValueError(
                f"shared-proton domain {domain_id!r} must contain exactly two distances"
            )
        first, second = sorted(primitives, key=lambda item: (item.atoms, item.identifier))
        common = set(first.atoms) & set(second.atoms)
        if len(common) != 1:
            raise ValueError("shared-proton distances must have one common hydrogen")
        pairs.append((domain_id, first, second))
    if not pairs:
        return definition

    existing_names = {str(gic.name) for gic in definition.gics}
    added: list[FrozenGIC] = []
    for ordinal, (domain_id, first, second) in enumerate(pairs, start=1):
        label = f"PT{ordinal:03d}"
        name = f"PROTON_TRANSFER_{domain_id}"
        if name in existing_names:
            continue
        coefficients = ((first.identifier, 1.0), (second.identifier, -1.0))
        added.append(
            FrozenGIC(
                identifier=label,
                name=name,
                family="PROTON_TRANSFER",
                irrep="A",
                primitive_id=first.identifier,
                gaussian_expression="NONE",
                coefficients=coefficients,
            )
        )
    if not added:
        return definition
    diagnostics = (
        *definition.semantic_diagnostics,
        "ZAFF_REDUNDANT_FIELD ENABLED",
        f"PROTON_TRANSFER_COORDINATES AUTOMATIC_SHARED_HYDROGEN_DOMAINS={len(added)}",
        "PROTON_TRANSFER_POLICY RETAIN_BOTH_DISTANCES_AND_ADD_DISTANCE_DIFFERENCE",
    )
    return replace(
        definition,
        gics=(*definition.gics, *added),
        candidate_count=int(definition.candidate_count) + len(added),
        semantic_diagnostics=diagnostics,
    )


def promote_shared_proton_coordinates(definition: GICDefinition) -> GICDefinition:
    """Compatibility alias for the redundant augmentation policy."""

    return augment_shared_proton_coordinates(definition)


def independent_zaff_analysis_definition(
    definition: GICDefinition,
    *,
    rank_tolerance: float = 1.0e-9,
) -> GICDefinition:
    """Select an independent chart from a redundant ZAFF field.

    Proton-transfer rows are considered first so a shared-hydrogen direction
    survives the reduction.  The redundant field itself is not modified.
    """

    from .evaluation import build_gic_b_matrix

    matrix = np.asarray(build_gic_b_matrix(definition).rows, dtype=float)
    target = int(definition.target_rank)
    proton_transfer = [
        index
        for index, gic in enumerate(definition.gics)
        if str(gic.family).upper() == "PROTON_TRANSFER"
    ]
    remaining = [index for index in range(len(definition.gics)) if index not in proton_transfer]
    selected: list[int] = []
    rank = 0
    for index in (*proton_transfer, *remaining):
        trial = (*selected, index)
        trial_rank = int(np.linalg.matrix_rank(matrix[np.asarray(trial)], tol=rank_tolerance))
        if trial_rank <= rank:
            continue
        selected.append(index)
        rank = trial_rank
        if rank == target:
            break
    if rank != target:
        raise ValueError(f"ZAFF analysis chart does not span the Wilson tangent space: {rank}/{target}")
    diagnostics = (
        *definition.semantic_diagnostics,
        f"ZAFF_ANALYSIS_CHART NONREDUNDANT_ROWS={len(selected)} SOURCE_ROWS={len(definition.gics)}",
    )
    return replace(
        definition,
        gics=tuple(definition.gics[index] for index in selected),
        rank=rank,
        semantic_diagnostics=diagnostics,
    )


def evaluate_working_functions(functions, values, b_rows, b_primes=None, *, require_independent=True):
    """Evaluate a selected set and optionally reject dependent transformed rows."""
    functions = tuple(functions)
    if not functions:
        raise ValueError("working-function set must not be empty")
    results = tuple(f.evaluate(values, b_rows, b_primes) for f in functions)
    if require_independent:
        matrix = np.stack([result.b_row.to_dense() for result in results])
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms == 0) or np.linalg.matrix_rank(matrix / norms[:, None]) != len(results):
            raise ValueError("transformed working-coordinate set is rank deficient")
    return results


def replace_working_gic_lines(lines, replacements):
    """Replace explicitly selected final rows without adding redundant helpers.

    Preserve existing active/frozen options. Reject labels used by other rows,
    since redefining a helper would silently change their meaning. This checks
    export structure, not geometric rank; evaluate the complete set separately.
    """
    lines = list(lines)
    replacements = dict(replacements)
    found = set()
    result = []
    for line in lines:
        left, separator, right = line.partition("=")
        label = left.strip().split("(", 1)[0]
        if separator and set(re.findall(r"\b[A-Za-z][A-Za-z0-9]*\b", right)) & replacements.keys():
            raise ValueError("cannot replace a label used as a dependency by another GIC")
        if label not in replacements:
            result.append(line)
            continue
        if not separator or label in found:
            raise ValueError("selected GIC label is missing or duplicated")
        found.add(label)
        result.append(f"{left.strip()} = {replacements[label].gaussian_expression()}")
    if found != replacements.keys():
        raise ValueError("not all selected GIC labels were found")
    return result
