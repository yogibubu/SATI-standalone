from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from matrix_chem.structural_corrections import (
    CV_RADIAL_COVALENT_RADII_ANGSTROM as ORACLE_CV_RADIAL_COVALENT_RADII_ANGSTROM,
    CV_RADIAL_PERIOD as ORACLE_CV_RADIAL_PERIOD,
    CV_RADIAL_RADIUS_AWARE_PERIOD_LINE as ORACLE_CV_RADIAL_RADIUS_AWARE_PERIOD_LINE,
    CV_RADIAL_SIGMA_SCALE as ORACLE_CV_RADIAL_SIGMA_SCALE,
    CV_RADIAL_VALENCE_ELECTRONS as ORACLE_CV_RADIAL_VALENCE_ELECTRONS,
    CV_RADIAL_WEIGHT_THRESHOLD as ORACLE_CV_RADIAL_WEIGHT_THRESHOLD,
    cv_radial_bond_delta_angstrom as oracle_cv_radial_bond_delta_angstrom,
    cv_radial_posterior_amplitude_milliangstrom as oracle_cv_radial_posterior_amplitude_milliangstrom,
    cv_radial_radius_aware_alpha_milliangstrom as oracle_cv_radial_radius_aware_alpha_milliangstrom,
)

from .models import GICDefinition, GICPrimitive
from .coordinate_kernel.primitives import Primitive, eval_primitives
from .coordinate_kernel.transforms import internal_to_cart_coords


CV_RADIAL_ALPHA_MILLIANGSTROM = {
    1: 0.000000,
    5: -2.193086,
    6: -1.581176,
    7: -1.100622,
    8: -0.673101,
    9: -0.279031,
    13: -5.653147,
    14: -4.369122,
    15: -3.424289,
    16: -2.587613,
    17: -1.763088,
}

CV_RADIAL_BOND_RESPONSE_DELTA_MILLIANGSTROM = {
    "Al--H": -6.871317608815,
    "B--F": -2.327187781383,
    "B--H": -2.165126648579,
    "C--C": -3.508818467441,
    "C--Cl": -2.419805002705,
    "C--F": -1.655708194180,
    "C--H(sp2)": -1.482436654430,
    "C--H(sp3)": -1.466579131717,
    "C--N": -2.751919386265,
    "C--O": -2.520519622043,
    "C--S": -4.557989706312,
    "C--Si": -7.178550464679,
    "C=C": -3.018579270253,
    "C=N": -2.432846605787,
    "C=O": -2.145235188323,
    "C=S": -3.929104323399,
    "F--Cl": -2.122312074230,
    "F--F": -1.332131715865,
    "H--Cl": -1.670327316453,
    "H--F": -0.525037695449,
    "N--F": -1.898005448295,
    "N--H": -1.004311715301,
    "N--N": -1.862013407125,
    "N--O": -2.373795781913,
    "N=N": -2.364528079117,
    "N=O": -2.300790190376,
    "O--F": -1.696646466936,
    "O--H": -0.754987414707,
    "O--O": -1.457550479292,
    "O--P": -3.610227366396,
    "O--S": -2.723949014147,
    "O--Si": -5.147037215130,
    "O=P": -3.244938011937,
    "O=S": -2.629466069990,
    "P--Cl": -5.484005810206,
    "P--F": -2.785527816448,
    "P--H": -2.862360873435,
    "P--P": -7.440613200018,
    "P=P": -7.320491317649,
    "S--Cl": -5.113002009152,
    "S--F": -2.175529555420,
    "S--H": -2.229583695017,
    "S--S": -6.582616546303,
    "Si--Cl": -4.183548872242,
    "Si--F": -3.919518857943,
    "Si--H": -4.301612621336,
}

# Backward-compatible name for the calibration table.  These values are not
# used as a posteriori geometry-correction amplitudes because the underlying
# bond-response model requires MP2-FC local information.
CV_RADIAL_BOND_CLASS_DELTA_MILLIANGSTROM = CV_RADIAL_BOND_RESPONSE_DELTA_MILLIANGSTROM

# SMITH re-exports the unique ORACLE calibration used by the standalone GF path.
CV_RADIAL_SIGMA_SCALE = ORACLE_CV_RADIAL_SIGMA_SCALE
CV_RADIAL_WEIGHT_THRESHOLD = ORACLE_CV_RADIAL_WEIGHT_THRESHOLD
CV_RADIAL_COVALENT_RADII_ANGSTROM = ORACLE_CV_RADIAL_COVALENT_RADII_ANGSTROM
CV_RADIAL_PERIOD = ORACLE_CV_RADIAL_PERIOD
CV_RADIAL_VALENCE_ELECTRONS = ORACLE_CV_RADIAL_VALENCE_ELECTRONS
CV_RADIAL_RADIUS_AWARE_PERIOD_LINE = ORACLE_CV_RADIAL_RADIUS_AWARE_PERIOD_LINE

@dataclass(frozen=True)
class CVRadialGeometryCorrection:
    """Summary of the CV-radial geometry correction used for a Wilson G matrix."""

    enabled: bool
    sigma_scale: float = CV_RADIAL_SIGMA_SCALE
    weight_threshold: float = CV_RADIAL_WEIGHT_THRESHOLD
    corrected_bond_count: int = 0
    max_abs_delta_angstrom: float = 0.0
    max_abs_residual_angstrom: float = 0.0
    parameter_source: str = "radius-aware-period-line"

    @property
    def label(self) -> str:
        if not self.enabled:
            return "NONE"
        return (
            "CV_RADIAL_GAUSSIAN"
            f"(model={self.parameter_source},sigma_scale={self.sigma_scale:g},"
            f"threshold={self.weight_threshold:g},"
            f"bonds={self.corrected_bond_count})"
        )


def apply_cv_radial_geometry_correction(
    definition: GICDefinition,
    atomic_numbers: np.ndarray,
    coordinates_angstrom: np.ndarray,
    *,
    sigma_scale: float = CV_RADIAL_SIGMA_SCALE,
    weight_threshold: float = CV_RADIAL_WEIGHT_THRESHOLD,
    max_iter: int = 50,
    tolerance: float = 1.0e-8,
) -> tuple[np.ndarray, CVRadialGeometryCorrection]:
    """Apply the CV-radial bond-length correction before evaluating Wilson B/G.

    The frozen SONIC contract is left unchanged.  Bond-length targets are built
    at the primitive level, projected through the stored symmetry-adapted U
    matrix, and back-transformed with the existing SMITH coordinate-kernel Cartesian
    back-transform.  This preserves coefficients of symmetry-adapted coordinates
    instead of assuming one primitive bond maps to one final coordinate.
    """

    coords = np.asarray(coordinates_angstrom, dtype=float)
    numbers = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    if coords.ndim != 2 or coords.shape != (numbers.size, 3):
        raise ValueError("CV-radial correction needs coordinates with shape (natoms, 3)")
    if float(sigma_scale) <= 0.0:
        raise ValueError("CV-radial sigma scale must be positive")
    threshold = float(weight_threshold)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("CV-radial weight threshold must be between 0 and 1")

    primitive_basis = tuple(_coordinate_kernel_primitive_from_gic_primitive(primitive) for primitive in definition.primitives)
    primitive_values = eval_primitives(primitive_basis, coords)
    target_values = np.array(primitive_values, dtype=float, copy=True)

    corrected: list[tuple[int, float]] = []
    for index, primitive in enumerate(definition.primitives):
        if primitive.function != "R" or len(primitive.atoms) != 2:
            continue
        atom_i, atom_j = (int(atom) - 1 for atom in primitive.atoms)
        z_i = int(numbers[atom_i])
        z_j = int(numbers[atom_j])
        delta = cv_radial_bond_delta_angstrom(
            z_i,
            z_j,
            float(primitive_values[index]),
            sigma_scale=float(sigma_scale),
            weight_threshold=threshold,
        )
        if delta is None:
            continue
        target_values[index] = float(primitive_values[index]) + delta
        corrected.append((index, delta))

    if not corrected:
        return coords.copy(), CVRadialGeometryCorrection(
            enabled=True,
            sigma_scale=float(sigma_scale),
            weight_threshold=threshold,
        )

    u_matrix = _u_matrix_from_definition(definition)
    q_target = u_matrix.T @ target_values
    corrected_coords = internal_to_cart_coords(
        q_target,
        coords,
        primitive_basis,
        U=u_matrix,
        max_iter=int(max_iter),
        tol=float(tolerance),
    )
    final_values = eval_primitives(primitive_basis, corrected_coords)
    residuals = [float(final_values[index] - target_values[index]) for index, _delta in corrected]
    return corrected_coords, CVRadialGeometryCorrection(
        enabled=True,
        sigma_scale=float(sigma_scale),
        weight_threshold=threshold,
        corrected_bond_count=len(corrected),
        max_abs_delta_angstrom=max(abs(delta) for _index, delta in corrected),
        max_abs_residual_angstrom=max(abs(value) for value in residuals),
        parameter_source="radius-aware-period-line",
    )


def cv_radial_bond_delta_angstrom(
    atomic_number_a: int,
    atomic_number_b: int,
    distance_angstrom: float,
    *,
    bond_class: str | None = None,
    sigma_scale: float = CV_RADIAL_SIGMA_SCALE,
    weight_threshold: float = CV_RADIAL_WEIGHT_THRESHOLD,
) -> float | None:
    return oracle_cv_radial_bond_delta_angstrom(
        atomic_number_a,
        atomic_number_b,
        distance_angstrom,
        sigma_scale=sigma_scale,
        weight_threshold=weight_threshold,
    )


def cv_radial_posterior_amplitude_milliangstrom(
    atomic_number_a: int,
    atomic_number_b: int,
) -> float | None:
    """Return a CV-radial a posteriori amplitude independent of MP2-FC data."""
    return oracle_cv_radial_posterior_amplitude_milliangstrom(
        atomic_number_a, atomic_number_b
    )


def cv_radial_radius_aware_alpha_milliangstrom(atomic_number: int) -> float | None:
    """Radius-aware period-line atomic alpha used by the final Gaussian model."""
    return oracle_cv_radial_radius_aware_alpha_milliangstrom(atomic_number)


def cv_radial_bond_class_delta_milliangstrom(bond_class: str | None) -> float | None:
    if bond_class is None:
        return None
    canonical = _BOND_CLASS_ALIASES.get(_normalize_bond_class_key(bond_class))
    if canonical is None:
        return None
    return CV_RADIAL_BOND_RESPONSE_DELTA_MILLIANGSTROM[canonical]


def _normalize_bond_class_key(label: str) -> str:
    text = str(label).strip()
    text = text.replace("\\", "")
    text = text.replace("$", "")
    text = text.replace("{", "").replace("}", "")
    text = text.replace("^", "")
    text = text.replace("(", "").replace(")", "")
    text = text.replace(" ", "")
    text = text.replace("-", "")
    text = text.replace("=", "=")
    return text.lower()


_BOND_CLASS_ALIASES = {
    _normalize_bond_class_key(label): label
    for label in CV_RADIAL_BOND_RESPONSE_DELTA_MILLIANGSTROM
}


def _u_matrix_from_definition(definition: GICDefinition) -> np.ndarray:
    primitive_index = {
        primitive.identifier: index for index, primitive in enumerate(definition.primitives)
    }
    matrix = np.zeros((len(definition.primitives), len(definition.gics)), dtype=float)
    for col, gic in enumerate(definition.gics):
        coefficients = gic.coefficients or ((gic.primitive_id, 1.0),)
        for primitive_id, coefficient in coefficients:
            row = primitive_index.get(primitive_id)
            if row is not None:
                matrix[row, col] += float(coefficient)
    return matrix


def _coordinate_kernel_primitive_from_gic_primitive(primitive: GICPrimitive) -> Primitive:
    atoms = tuple(int(atom) - 1 for atom in primitive.atoms)
    if primitive.function == "R":
        return Primitive("bond", atoms)
    if primitive.function == "A":
        return Primitive("angle", atoms)
    if primitive.function == "L":
        return Primitive(
            "linear_bend",
            atoms,
            mode=int(primitive.mode),
            ref=tuple(int(atom) - 1 for atom in primitive.ref_atoms),
        )
    if primitive.function == "D":
        return Primitive("dihedral", atoms)
    if primitive.function == "U":
        return Primitive("out_of_plane", atoms)
    if primitive.function == "H":
        return Primitive("out_of_plane_height", atoms)
    raise ValueError(f"unsupported primitive function for CV-radial correction: {primitive.function}")
