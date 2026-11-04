"""ARCHITECT compiler boundary for the internal GFN-FF calculator.

GFN-FF topology perception and parameter assignment belong to ARCHITECT;
energy and derivatives belong to the single resident MATRIX runtime.  Every
compiled layer records its coverage explicitly: partial records are audit
artifacts and can never be selected as production calculators.  ``libxtb`` is
an independent parity oracle, not a fallback implementation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from matrix_zaff.nonbonded import BOHR_TO_ANGSTROM
from matrix_zaff.gfnff_nonbonded import (
    gfnff_charge_scale,
    load_gfnff_parameter_bundle,
)
from matrix_zaff.seed_runtime import ZAFF_SEED_MODEL_SCHEMA


GFNFF_MIGRATION_SCHEMA = "matrix.architect.gfnff_migration.v1"
ZAFF_GFNFF_COMPONENTS_SCHEMA = "matrix.zaff.gfnff_components.v1"
GFNFF_REFERENCE_VERSION = "xTB-6.7.1-GFN-FF"
GFNFF_REQUIRED_TERM_FAMILIES = (
    "coordination",
    "eeq",
    "bond",
    "bonded_repulsion",
    "angle",
    "torsion",
    "special_torsion",
    "nonbonded_repulsion",
    "d3_two_body",
    "hydrogen_bond",
    "halogen_bond",
    "bonded_atm",
)


@dataclass(frozen=True)
class GfnffPreparedCoordinationLayer:
    """Resolved per-atom data for the non-periodic GFN-FF log-CN."""

    covalent_radii_bohr: np.ndarray
    cutoff_bohr: float = 10.0
    cn_max: float = 4.4
    steepness: float = 7.5

    def __post_init__(self) -> None:
        radii = np.asarray(self.covalent_radii_bohr, dtype=float).reshape(-1)
        if not len(radii) or np.any(~np.isfinite(radii)) or np.any(radii <= 0.0):
            raise ValueError("GFN-FF covalent radii must be finite and positive")
        for name in ("cutoff_bohr", "cn_max", "steepness"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"GFN-FF {name} must be finite and positive")
            object.__setattr__(self, name, value)
        radii.setflags(write=False)
        object.__setattr__(self, "covalent_radii_bohr", radii)

    def record(self, natoms: int) -> dict[str, Any]:
        if len(self.covalent_radii_bohr) != int(natoms):
            raise ValueError("GFN-FF coordination radii do not cover all atoms")
        return {
            "kind": "GFNFF_LOG_ERF",
            "covalent_radii_angstrom": (self.covalent_radii_bohr * BOHR_TO_ANGSTROM).tolist(),
            "cutoff_angstrom": self.cutoff_bohr * BOHR_TO_ANGSTROM,
            "cn_max": self.cn_max,
            "steepness": self.steepness,
            "pair_storage": "SPARSE_VERLET",
            "derivative_storage": "MATRIX_FREE",
        }


@dataclass(frozen=True)
class GfnffPreparedEEQLayer:
    """Resolved molecular GFN-FF EEQ parameters in native xTB units."""

    electronegativities_hartree: np.ndarray
    cn_coefficients_hartree: np.ndarray
    gaussian_alpha_bohr2: np.ndarray
    hardness_hartree: np.ndarray
    fragment_ids: np.ndarray
    fragment_charges: np.ndarray
    tolerance: float = 1.0e-11
    maximum_iterations: int = 500
    fmm_precision: float = 1.0e-10
    fmm_minimum_atoms: int = 256

    def __post_init__(self) -> None:
        arrays = {
            "electronegativities_hartree": np.asarray(
                self.electronegativities_hartree, dtype=float
            ).reshape(-1),
            "cn_coefficients_hartree": np.asarray(
                self.cn_coefficients_hartree, dtype=float
            ).reshape(-1),
            "gaussian_alpha_bohr2": np.asarray(self.gaussian_alpha_bohr2, dtype=float).reshape(-1),
            "hardness_hartree": np.asarray(self.hardness_hartree, dtype=float).reshape(-1),
            "fragment_ids": np.asarray(self.fragment_ids, dtype=np.intp).reshape(-1),
            "fragment_charges": np.asarray(self.fragment_charges, dtype=float).reshape(-1),
        }
        natoms = len(arrays["electronegativities_hartree"])
        if any(
            len(arrays[name]) != natoms
            for name in (
                "cn_coefficients_hartree",
                "gaussian_alpha_bohr2",
                "hardness_hartree",
                "fragment_ids",
            )
        ):
            raise ValueError("GFN-FF EEQ atom arrays have inconsistent dimensions")
        if (
            np.any(~np.isfinite(arrays["electronegativities_hartree"]))
            or np.any(~np.isfinite(arrays["cn_coefficients_hartree"]))
            or np.any(~np.isfinite(arrays["gaussian_alpha_bohr2"]))
            or np.any(arrays["gaussian_alpha_bohr2"] <= 0.0)
            or np.any(~np.isfinite(arrays["hardness_hartree"]))
            or np.any(~np.isfinite(arrays["fragment_charges"]))
        ):
            raise ValueError("GFN-FF EEQ parameters are invalid")
        eeq_diagonal = (
            np.sqrt(2.0 / np.pi) / np.sqrt(arrays["gaussian_alpha_bohr2"])
            + arrays["hardness_hartree"]
        )
        if np.any(eeq_diagonal <= 0.0):
            raise ValueError("GFN-FF EEQ self-interaction diagonal must be positive")
        fragments = arrays["fragment_ids"]
        if np.any(fragments < 0) or set(fragments.tolist()) != set(
            range(len(arrays["fragment_charges"]))
        ):
            raise ValueError("GFN-FF EEQ fragment identifiers must be contiguous")
        if (
            not np.isfinite(self.tolerance)
            or self.tolerance <= 0.0
            or self.maximum_iterations < 1
            or not 0.0 < self.fmm_precision < 1.0
            or self.fmm_minimum_atoms < 1
        ):
            raise ValueError("GFN-FF EEQ solver controls are invalid")
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def record(self, natoms: int) -> dict[str, Any]:
        if len(self.electronegativities_hartree) != int(natoms):
            raise ValueError("GFN-FF EEQ parameters do not cover all atoms")
        return {
            "kind": "GFNFF_EEQ_GAUSSIAN",
            "electronegativities_hartree": self.electronegativities_hartree.tolist(),
            "cn_coefficients_hartree": self.cn_coefficients_hartree.tolist(),
            "gaussian_alpha_bohr2": self.gaussian_alpha_bohr2.tolist(),
            "hardness_hartree": self.hardness_hartree.tolist(),
            "fragment_ids": self.fragment_ids.tolist(),
            "fragment_charges": self.fragment_charges.tolist(),
            "tolerance": float(self.tolerance),
            "maximum_iterations": int(self.maximum_iterations),
            "backend": "auto",
            "fmm_precision": float(self.fmm_precision),
            "fmm_minimum_atoms": int(self.fmm_minimum_atoms),
            "matrix_storage": "MATRIX_FREE",
            "constraint_solver": "FRAGMENT_PROJECTED_PCG",
        }


@dataclass(frozen=True)
class GfnffPreparedRepulsionLayer:
    """Resolved exact GFN-FF repulsion coefficients in atomic units.

    Class codes are topology-time decisions: 0=other, 1=H, 2=C, 3=O,
    4=metal.  Sparse overrides carry the complete exponent multiplier for
    special topological pairs such as H--H 1,3 and 1,4 contacts.
    """

    bonded_pairs: tuple[tuple[int, int], ...]
    bonded_amplitudes_hartree_bohr: np.ndarray
    bonded_exponents_per_bohr3over2: np.ndarray
    nonbonded_amplitude_factors: np.ndarray
    nonbonded_exponent_factors: np.ndarray
    nonbonded_class_codes: np.ndarray
    scale_override_pairs: tuple[tuple[int, int], ...] = ()
    scale_override_values: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    nonbonded_cutoff_bohr: float = 20.0

    def __post_init__(self) -> None:
        bonds = tuple(tuple(int(value) for value in pair) for pair in self.bonded_pairs)
        overrides = tuple(tuple(int(value) for value in pair) for pair in self.scale_override_pairs)
        arrays = {
            "bonded_amplitudes_hartree_bohr": np.asarray(
                self.bonded_amplitudes_hartree_bohr, dtype=float
            ).reshape(-1),
            "bonded_exponents_per_bohr3over2": np.asarray(
                self.bonded_exponents_per_bohr3over2, dtype=float
            ).reshape(-1),
            "nonbonded_amplitude_factors": np.asarray(
                self.nonbonded_amplitude_factors, dtype=float
            ).reshape(-1),
            "nonbonded_exponent_factors": np.asarray(
                self.nonbonded_exponent_factors, dtype=float
            ).reshape(-1),
            "nonbonded_class_codes": np.asarray(self.nonbonded_class_codes, dtype=np.intp).reshape(
                -1
            ),
            "scale_override_values": np.asarray(self.scale_override_values, dtype=float).reshape(
                -1
            ),
        }
        natoms = len(arrays["nonbonded_amplitude_factors"])
        if natoms < 1 or any(
            len(arrays[name]) != natoms
            for name in ("nonbonded_exponent_factors", "nonbonded_class_codes")
        ):
            raise ValueError("GFN-FF repulsion atom arrays have inconsistent dimensions")
        if any(
            len(arrays[name]) != len(bonds)
            for name in (
                "bonded_amplitudes_hartree_bohr",
                "bonded_exponents_per_bohr3over2",
            )
        ) or len(arrays["scale_override_values"]) != len(overrides):
            raise ValueError("GFN-FF repulsion pair arrays have inconsistent dimensions")
        if any(len(pair) != 2 or min(pair) < 0 or pair[0] == pair[1] for pair in bonds + overrides):
            raise ValueError("GFN-FF repulsion pair index is invalid")
        if len({tuple(sorted(pair)) for pair in bonds}) != len(bonds) or len(
            {tuple(sorted(pair)) for pair in overrides}
        ) != len(overrides):
            raise ValueError("GFN-FF repulsion pairs must be unique")
        positive = (
            arrays["bonded_amplitudes_hartree_bohr"],
            arrays["bonded_exponents_per_bohr3over2"],
            arrays["nonbonded_amplitude_factors"],
            arrays["nonbonded_exponent_factors"],
            arrays["scale_override_values"],
        )
        if any(np.any(~np.isfinite(array)) or np.any(array <= 0.0) for array in positive):
            raise ValueError("GFN-FF repulsion coefficients must be finite and positive")
        if np.any((arrays["nonbonded_class_codes"] < 0) | (arrays["nonbonded_class_codes"] > 4)):
            raise ValueError("GFN-FF repulsion class code is invalid")
        if not np.isfinite(self.nonbonded_cutoff_bohr) or self.nonbonded_cutoff_bohr <= 0.0:
            raise ValueError("GFN-FF repulsion cutoff must be finite and positive")
        object.__setattr__(self, "bonded_pairs", bonds)
        object.__setattr__(self, "scale_override_pairs", overrides)
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def record(self, natoms: int) -> dict[str, Any]:
        if len(self.nonbonded_amplitude_factors) != int(natoms) or any(
            max(pair) >= int(natoms) for pair in self.bonded_pairs + self.scale_override_pairs
        ):
            raise ValueError("GFN-FF repulsion layer does not cover the atom table")
        return {
            "kind": "GFNFF_REPULSION_R32",
            "bonded_pairs": [list(pair) for pair in self.bonded_pairs],
            "bonded_amplitudes_hartree_bohr": (self.bonded_amplitudes_hartree_bohr.tolist()),
            "bonded_exponents_per_bohr3over2": (self.bonded_exponents_per_bohr3over2.tolist()),
            "nonbonded_amplitude_factors": self.nonbonded_amplitude_factors.tolist(),
            "nonbonded_exponent_factors": self.nonbonded_exponent_factors.tolist(),
            "nonbonded_class_codes": self.nonbonded_class_codes.tolist(),
            "scale_override_pairs": [list(pair) for pair in self.scale_override_pairs],
            "scale_override_values": self.scale_override_values.tolist(),
            "nonbonded_cutoff_bohr": float(self.nonbonded_cutoff_bohr),
            "pair_storage": "SPARSE_VERLET_PLUS_TOPOLOGY_BONDS",
            "derivative_storage": "MATRIX_FREE",
        }


@dataclass(frozen=True)
class GfnffPreparedD3Layer:
    """Resolved GFN-FF D3(BJ) data with compact element-reference tables."""

    atomic_numbers: np.ndarray
    charges_e: np.ndarray
    zeta_factors: np.ndarray
    sqrt_z_r4_over_r2: np.ndarray
    reference_cn_by_atomic_number: Mapping[int, np.ndarray]
    reference_c6_by_pair: Mapping[tuple[int, int], np.ndarray]
    cutoff_bohr: float = float(np.sqrt(1500.0))
    weighting_factor: float = 4.0
    dispersion_scale: float = 1.0
    damping_a1: float = 0.58
    damping_a2: float = 4.80

    @classmethod
    def from_reference_parameters(
        cls,
        atomic_numbers: np.ndarray,
        charges_e: np.ndarray,
    ) -> "GfnffPreparedD3Layer":
        """Resolve the resident xTB-6.7.1 tables once at compilation time."""

        numbers = np.asarray(atomic_numbers, dtype=np.intp).reshape(-1)
        charges = np.asarray(charges_e, dtype=float).reshape(-1)
        if len(numbers) != len(charges):
            raise ValueError("GFN-FF D3 atomic numbers and charges disagree")
        bundle = load_gfnff_parameter_bundle()
        if np.any(numbers < 1) or np.any(numbers > len(bundle.sqrt_z_r4_over_r2)):
            raise ValueError("GFN-FF D3 atomic number is outside the resident table")
        reference = bundle.reference_dispersion(numbers)
        zeta = np.asarray(
            [
                gfnff_charge_scale(
                    int(number),
                    float(charge),
                    bundle.effective_nuclear_charge,
                    bundle.hardness,
                )
                for number, charge in zip(numbers, charges, strict=True)
            ]
        )
        return cls(
            atomic_numbers=numbers,
            charges_e=charges,
            zeta_factors=zeta,
            sqrt_z_r4_over_r2=bundle.sqrt_z_r4_over_r2[numbers - 1],
            reference_cn_by_atomic_number={
                int(key): np.asarray(value, dtype=float)
                for key, value in reference.reference_cn_by_atomic_number.items()
            },
            reference_c6_by_pair={
                tuple(map(int, key)): np.asarray(value, dtype=float)
                for key, value in reference.reference_c6_by_pair.items()
            },
            weighting_factor=float(reference.weighting_factor),
        )

    def __post_init__(self) -> None:
        arrays = {
            "atomic_numbers": np.asarray(self.atomic_numbers, dtype=np.intp).reshape(-1),
            "charges_e": np.asarray(self.charges_e, dtype=float).reshape(-1),
            "zeta_factors": np.asarray(self.zeta_factors, dtype=float).reshape(-1),
            "sqrt_z_r4_over_r2": np.asarray(self.sqrt_z_r4_over_r2, dtype=float).reshape(-1),
        }
        natoms = len(arrays["atomic_numbers"])
        if natoms < 1 or any(len(array) != natoms for array in arrays.values()):
            raise ValueError("GFN-FF D3 atom arrays have inconsistent dimensions")
        if (
            np.any(arrays["atomic_numbers"] < 1)
            or np.any(~np.isfinite(arrays["charges_e"]))
            or np.any(~np.isfinite(arrays["zeta_factors"]))
            or np.any(arrays["zeta_factors"] < 0.0)
            or np.any(~np.isfinite(arrays["sqrt_z_r4_over_r2"]))
            or np.any(arrays["sqrt_z_r4_over_r2"] <= 0.0)
        ):
            raise ValueError("GFN-FF D3 atom parameters are invalid")
        references: dict[int, np.ndarray] = {}
        unique_numbers = sorted(set(map(int, arrays["atomic_numbers"])))
        for number in unique_numbers:
            if number not in self.reference_cn_by_atomic_number:
                raise ValueError("GFN-FF D3 reference CN table is incomplete")
            values = np.asarray(self.reference_cn_by_atomic_number[number], dtype=float).reshape(-1)
            if not len(values) or np.any(~np.isfinite(values)):
                raise ValueError("GFN-FF D3 reference CN table is invalid")
            values.setflags(write=False)
            references[number] = values
        c6_tables: dict[tuple[int, int], np.ndarray] = {}
        for left in unique_numbers:
            for right in unique_numbers:
                if right < left:
                    continue
                key = (left, right)
                if key not in self.reference_c6_by_pair:
                    raise ValueError("GFN-FF D3 reference C6 table is incomplete")
                table = np.asarray(self.reference_c6_by_pair[key], dtype=float)
                expected = (len(references[left]), len(references[right]))
                if table.shape != expected or np.any(~np.isfinite(table)) or np.any(table <= 0.0):
                    raise ValueError("GFN-FF D3 reference C6 table is invalid")
                table.setflags(write=False)
                c6_tables[key] = table
        controls = (
            self.cutoff_bohr,
            self.weighting_factor,
            self.dispersion_scale,
            self.damping_a1,
            self.damping_a2,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in controls):
            raise ValueError("GFN-FF D3 controls must be finite and positive")
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "reference_cn_by_atomic_number", references)
        object.__setattr__(self, "reference_c6_by_pair", c6_tables)

    def record(self, natoms: int) -> dict[str, Any]:
        if len(self.atomic_numbers) != int(natoms):
            raise ValueError("GFN-FF D3 parameters do not cover all atoms")
        return {
            "kind": "GFNFF_D3_BJ_TWO_BODY",
            "atomic_numbers": self.atomic_numbers.tolist(),
            "charges_e": self.charges_e.tolist(),
            "zeta_factors": self.zeta_factors.tolist(),
            "sqrt_z_r4_over_r2": self.sqrt_z_r4_over_r2.tolist(),
            "reference_cn_by_atomic_number": {
                str(key): value.tolist()
                for key, value in self.reference_cn_by_atomic_number.items()
            },
            "reference_c6_by_pair": {
                f"{key[0]}:{key[1]}": value.tolist()
                for key, value in self.reference_c6_by_pair.items()
            },
            "cutoff_bohr": float(self.cutoff_bohr),
            "weighting_factor": float(self.weighting_factor),
            "dispersion_scale": float(self.dispersion_scale),
            "damping_a1": float(self.damping_a1),
            "damping_a2": float(self.damping_a2),
            "pair_storage": "SPARSE_VERLET",
            "cn_response": "DYNAMIC_MATRIX_FREE",
            "derivative_storage": "MATRIX_FREE",
        }


@dataclass(frozen=True)
class GfnffPreparedValenceLayer:
    """Resolved exact GFN-FF angles, torsions, and special torsions."""

    angle_atoms: tuple[tuple[int, int, int], ...] = ()
    angle_parameters: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=float))
    torsion_atoms: tuple[tuple[int, int, int, int], ...] = ()
    torsion_parameters: np.ndarray = field(default_factory=lambda: np.empty((0, 7), dtype=float))
    special_torsion_atoms: tuple[tuple[int, int, int, int], ...] = ()
    special_torsion_amplitudes_hartree: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )

    def __post_init__(self) -> None:
        angles = tuple(tuple(map(int, atoms)) for atoms in self.angle_atoms)
        torsions = tuple(tuple(map(int, atoms)) for atoms in self.torsion_atoms)
        special = tuple(tuple(map(int, atoms)) for atoms in self.special_torsion_atoms)
        arrays = {
            "angle_parameters": np.asarray(self.angle_parameters, dtype=float).reshape(-1, 4),
            "torsion_parameters": np.asarray(self.torsion_parameters, dtype=float).reshape(-1, 7),
            "special_torsion_amplitudes_hartree": np.asarray(
                self.special_torsion_amplitudes_hartree, dtype=float
            ).reshape(-1),
        }
        if (
            len(angles) != len(arrays["angle_parameters"])
            or len(torsions) != len(arrays["torsion_parameters"])
            or len(special) != len(arrays["special_torsion_amplitudes_hartree"])
        ):
            raise ValueError("GFN-FF valence arrays have inconsistent dimensions")
        if any(
            len(set(atoms)) != len(atoms) or min(atoms, default=0) < 0
            for atoms in angles + torsions + special
        ):
            raise ValueError("GFN-FF valence atom tuple is invalid")
        if any(np.any(~np.isfinite(array)) for array in arrays.values()):
            raise ValueError("GFN-FF valence parameters must be finite")
        object.__setattr__(self, "angle_atoms", angles)
        object.__setattr__(self, "torsion_atoms", torsions)
        object.__setattr__(self, "special_torsion_atoms", special)
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def record(self, natoms: int) -> dict[str, Any]:
        all_atoms = self.angle_atoms + self.torsion_atoms + self.special_torsion_atoms
        if any(max(atoms) >= int(natoms) for atoms in all_atoms):
            raise ValueError("GFN-FF valence layer exceeds the atom table")
        return {
            "kind": "GFNFF_EXACT_LOCAL_VALENCE",
            "angle_atoms": [list(atoms) for atoms in self.angle_atoms],
            "angle_parameters": self.angle_parameters.tolist(),
            "torsion_atoms": [list(atoms) for atoms in self.torsion_atoms],
            "torsion_parameters": self.torsion_parameters.tolist(),
            "special_torsion_atoms": [list(atoms) for atoms in self.special_torsion_atoms],
            "special_torsion_amplitudes_hartree": (
                self.special_torsion_amplitudes_hartree.tolist()
            ),
            "angle_form": "GFNFF_COSINE_OR_LINEAR_WITH_RATIONAL_DAMPING",
            "torsion_form": "GFNFF_PROPER_OR_IMPROPER_WITH_RATIONAL_DAMPING",
            "special_torsion_form": "GFNFF_ACETYLENE_UNDAMPED_COSINE_2PHI",
            "derivative_storage": "MATRIX_FREE",
        }


@dataclass(frozen=True)
class GfnffPreparedNCILayer:
    """Resolved sparse GFN-FF H/X-bond and bonded-ATM interactions.

    Hydrogen-bond modes 1--4 are the exact terminal, oriented-neighbor,
    aromatic-lone-pair, and carbonyl-specific forms in xTB 6.7.1. ARCHITECT
    certifies family coverage only after its topology pass has compiled the
    complete interaction set.
    """

    hydrogen_bond_atoms: np.ndarray = field(default_factory=lambda: np.empty((0, 7), dtype=np.intp))
    hydrogen_bond_parameters: np.ndarray = field(
        default_factory=lambda: np.empty((0, 22), dtype=float)
    )
    halogen_bond_atoms: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.intp))
    halogen_bond_parameters: np.ndarray = field(
        default_factory=lambda: np.empty((0, 7), dtype=float)
    )
    bonded_atm_atoms: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.intp))
    bonded_atm_amplitudes_hartree_bohr9: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    coverage_certified: bool = False

    def __post_init__(self) -> None:
        arrays = {
            "hydrogen_bond_atoms": np.asarray(self.hydrogen_bond_atoms, dtype=np.intp).reshape(
                -1, 7
            ),
            "hydrogen_bond_parameters": np.asarray(
                self.hydrogen_bond_parameters, dtype=float
            ).reshape(-1, 22),
            "halogen_bond_atoms": np.asarray(self.halogen_bond_atoms, dtype=np.intp).reshape(-1, 3),
            "halogen_bond_parameters": np.asarray(
                self.halogen_bond_parameters, dtype=float
            ).reshape(-1, 7),
            "bonded_atm_atoms": np.asarray(self.bonded_atm_atoms, dtype=np.intp).reshape(-1, 3),
            "bonded_atm_amplitudes_hartree_bohr9": np.asarray(
                self.bonded_atm_amplitudes_hartree_bohr9, dtype=float
            ).reshape(-1),
        }
        if (
            len(arrays["hydrogen_bond_atoms"]) != len(arrays["hydrogen_bond_parameters"])
            or len(arrays["halogen_bond_atoms"]) != len(arrays["halogen_bond_parameters"])
            or len(arrays["bonded_atm_atoms"]) != len(arrays["bonded_atm_amplitudes_hartree_bohr9"])
        ):
            raise ValueError("GFN-FF NCI arrays have inconsistent dimensions")
        modes = arrays["hydrogen_bond_parameters"][:, 0]
        neighbors = arrays["hydrogen_bond_parameters"][:, 14]
        torsions = arrays["hydrogen_bond_parameters"][:, 21]
        if (
            np.any(~np.isfinite(modes))
            or np.any(np.abs(modes - np.rint(modes)) > 1.0e-12)
            or np.any(~np.isin(np.rint(modes).astype(int), (1, 2, 3, 4)))
            or np.any(~np.isfinite(neighbors))
            or np.any(np.abs(neighbors - np.rint(neighbors)) > 1.0e-12)
            or np.any((neighbors < 0.0) | (neighbors > 4.0))
            or np.any((np.rint(modes) == 1.0) & (neighbors != 0.0))
            or np.any((np.rint(modes) == 3.0) & (neighbors != 2.0))
            or np.any((np.rint(modes) == 4.0) & (neighbors != 1.0))
            or np.any(~np.isfinite(torsions))
            or np.any(np.abs(torsions - np.rint(torsions)) > 1.0e-12)
            or np.any((np.rint(modes) == 4.0) & ((torsions < 0.0) | (torsions > 3.0)))
        ):
            raise ValueError("unsupported GFN-FF hydrogen-bond variant")
        if any(np.any(~np.isfinite(array)) for array in arrays.values()):
            raise ValueError("GFN-FF NCI parameters must be finite")
        object.__setattr__(self, "coverage_certified", bool(self.coverage_certified))
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def record(self, natoms: int) -> dict[str, Any]:
        active = []
        for atoms in (
            self.hydrogen_bond_atoms,
            self.halogen_bond_atoms,
            self.bonded_atm_atoms,
        ):
            active.extend(int(value) for value in atoms.reshape(-1) if value >= 0)
        if active and max(active) >= int(natoms):
            raise ValueError("GFN-FF NCI layer exceeds the atom table")
        return {
            "kind": "GFNFF_EXACT_NCI",
            "hydrogen_bond_atoms": self.hydrogen_bond_atoms.tolist(),
            "hydrogen_bond_parameters": self.hydrogen_bond_parameters.tolist(),
            "halogen_bond_atoms": self.halogen_bond_atoms.tolist(),
            "halogen_bond_parameters": self.halogen_bond_parameters.tolist(),
            "bonded_atm_atoms": self.bonded_atm_atoms.tolist(),
            "bonded_atm_amplitudes_hartree_bohr9": (
                self.bonded_atm_amplitudes_hartree_bohr9.tolist()
            ),
            "hydrogen_bond_forms": [
                "GFNFF_TERMINAL_MODE_1",
                "GFNFF_ORIENTED_NEIGHBOR_MODE_2",
                "GFNFF_AROMATIC_LONE_PAIR_MODE_3",
                "GFNFF_CARBONYL_ANGLE_TORSION_MODE_4",
            ],
            "halogen_bond_form": "GFNFF_DAMPED_R_MINUS_3",
            "bonded_atm_form": "GFNFF_AXILROD_TELLER_MUTO_BONDED",
            "coverage_certified": self.coverage_certified,
            "interaction_storage": "SPARSE_TOPOLOGY_LISTS",
            "derivative_storage": "MATRIX_FREE",
        }


@dataclass(frozen=True)
class GfnffPreparedBondLayer:
    """Resolved GFN-FF Gaussian bonds in native xTB units.

    ``reference_distances_bohr`` are the already resolved values used by the
    Gaussian well at the compilation geometry.  Geometry-dependent CN
    response is tracked separately and is intentionally not claimed by this
    first native layer.
    """

    bonds: tuple[tuple[int, int], ...]
    amplitudes_hartree: np.ndarray
    exponents_per_bohr2: np.ndarray
    reference_distances_bohr: np.ndarray
    source: str = GFNFF_REFERENCE_VERSION
    reference_cn_coefficients_bohr: np.ndarray | None = None
    coordination_numbers: np.ndarray | None = None

    def __post_init__(self) -> None:
        bonds = tuple(tuple(int(value) for value in pair) for pair in self.bonds)
        if any(len(pair) != 2 or pair[0] == pair[1] or min(pair) < 0 for pair in bonds):
            raise ValueError("GFN-FF prepared bond indices are invalid")
        if len(set(tuple(sorted(pair)) for pair in bonds)) != len(bonds):
            raise ValueError("GFN-FF prepared bonds must be unique")
        arrays = (
            np.asarray(self.amplitudes_hartree, dtype=float).reshape(-1),
            np.asarray(self.exponents_per_bohr2, dtype=float).reshape(-1),
            np.asarray(self.reference_distances_bohr, dtype=float).reshape(-1),
        )
        if any(len(array) != len(bonds) for array in arrays):
            raise ValueError("GFN-FF prepared bond arrays have inconsistent dimensions")
        if any(np.any(~np.isfinite(array)) for array in arrays):
            raise ValueError("GFN-FF prepared bond arrays must be finite")
        if np.any(arrays[0] >= 0.0):
            raise ValueError("stable GFN-FF Gaussian bond amplitudes must be negative")
        if np.any(arrays[1] <= 0.0) or np.any(arrays[2] <= 0.0):
            raise ValueError("GFN-FF bond exponents and references must be positive")
        object.__setattr__(self, "bonds", bonds)
        object.__setattr__(self, "amplitudes_hartree", arrays[0])
        object.__setattr__(self, "exponents_per_bohr2", arrays[1])
        object.__setattr__(self, "reference_distances_bohr", arrays[2])

        coefficients = self.reference_cn_coefficients_bohr
        coordination = self.coordination_numbers
        if (coefficients is None) != (coordination is None):
            raise ValueError("GFN-FF CN coefficients and compilation CN must be supplied together")
        if coefficients is not None:
            coefficient_array = np.asarray(coefficients, dtype=float).reshape(-1, 2)
            coordination_array = np.asarray(coordination, dtype=float).reshape(-1)
            if len(coefficient_array) != len(bonds):
                raise ValueError("GFN-FF bond CN coefficients have wrong dimensions")
            if np.any(~np.isfinite(coefficient_array)):
                raise ValueError("GFN-FF bond CN coefficients must be finite")
            if np.any(~np.isfinite(coordination_array)) or np.any(coordination_array < 0.0):
                raise ValueError("GFN-FF compilation coordination is invalid")
            coefficient_array.setflags(write=False)
            coordination_array.setflags(write=False)
            object.__setattr__(self, "reference_cn_coefficients_bohr", coefficient_array)
            object.__setattr__(self, "coordination_numbers", coordination_array)

    @property
    def dynamic_coordination(self) -> bool:
        return self.reference_cn_coefficients_bohr is not None

    def records(self, natoms: int) -> tuple[dict[str, Any], ...]:
        """Compile the layer into the unit contract used by matrix-ZAFF."""

        if any(max(pair) >= int(natoms) for pair in self.bonds):
            raise ValueError("GFN-FF prepared bond exceeds the atom table")
        records = []
        for index, (pair, amplitude, exponent, reference) in enumerate(
            zip(
                self.bonds,
                self.amplitudes_hartree,
                self.exponents_per_bohr2,
                self.reference_distances_bohr,
                strict=True,
            )
        ):
            record = {
                "atoms": list(pair),
                "functional_form": "GFNFF_GAUSSIAN",
                "amplitude_hartree": float(amplitude),
                "exponent_per_angstrom2": float(exponent / BOHR_TO_ANGSTROM**2),
                "r0_angstrom": float(reference * BOHR_TO_ANGSTROM),
                "source": self.source,
                "reference_response": "FROZEN_PENDING_NATIVE_CN_RESPONSE",
            }
            if self.dynamic_coordination:
                assert self.reference_cn_coefficients_bohr is not None
                assert self.coordination_numbers is not None
                coefficients = self.reference_cn_coefficients_bohr[index]
                if max(pair) >= len(self.coordination_numbers):
                    raise ValueError("GFN-FF compilation CN does not cover all bonds")
                base = reference - sum(
                    coefficients[side] * self.coordination_numbers[atom]
                    for side, atom in enumerate(pair)
                )
                if base <= 0.0:
                    raise ValueError("GFN-FF CN-independent bond reference is not positive")
                record.update(
                    {
                        "reference_base_angstrom": float(base * BOHR_TO_ANGSTROM),
                        "reference_cn_coefficients_angstrom": (
                            coefficients * BOHR_TO_ANGSTROM
                        ).tolist(),
                        "reference_response": "DYNAMIC_GFNFF_LOG_CN",
                    }
                )
            records.append(record)
        return tuple(records)


def compile_gfnff_migration_model(
    seed_model: Mapping[str, Any],
    prepared_bonds: GfnffPreparedBondLayer,
    coordination: GfnffPreparedCoordinationLayer | None = None,
    eeq: GfnffPreparedEEQLayer | None = None,
    repulsion: GfnffPreparedRepulsionLayer | None = None,
    d3_two_body: GfnffPreparedD3Layer | None = None,
    valence: GfnffPreparedValenceLayer | None = None,
    nci: GfnffPreparedNCILayer | None = None,
    *,
    oracle_topology=None,
) -> dict[str, Any]:
    """Replace the covalent layer and attach a fail-closed coverage ledger.

    The returned model is executable for term-by-term parity tests and becomes
    calculator-ready only when every required family, including dynamic
    coordination response and certified complete NCI lists, is present.
    """

    if seed_model.get("schema") != ZAFF_SEED_MODEL_SCHEMA:
        raise ValueError("GFN-FF migration requires a compiled ZAFF seed model")
    model = deepcopy(dict(seed_model))
    topology = {
        tuple(sorted((int(pair[0]), int(pair[1])))) for pair in model.get("topology_bonds", ())
    }
    prepared = {tuple(sorted(pair)) for pair in prepared_bonds.bonds}
    if prepared != topology:
        raise ValueError("GFN-FF prepared bonds do not exactly cover the topology")
    model["bonds"] = list(prepared_bonds.records(len(model["atoms"])))
    dynamic = prepared_bonds.dynamic_coordination
    if dynamic != (coordination is not None):
        raise ValueError("dynamic GFN-FF bonds require exactly one prepared coordination layer")
    if eeq is not None and coordination is None:
        raise ValueError("GFN-FF EEQ requires the prepared coordination layer")
    if d3_two_body is not None and coordination is None:
        raise ValueError("GFN-FF D3 requires the prepared coordination layer")
    implemented_set = {"coordination", "bond"} if dynamic else {"bond"}
    if coordination is not None:
        model["gfnff_coordination"] = coordination.record(len(model["atoms"]))
    if eeq is not None:
        model["gfnff_eeq"] = eeq.record(len(model["atoms"]))
        implemented_set.add("eeq")
    if repulsion is not None:
        repulsion_topology = {tuple(sorted(pair)) for pair in repulsion.bonded_pairs}
        if repulsion_topology != topology:
            raise ValueError("GFN-FF bonded repulsion does not exactly cover the topology")
        model["gfnff_repulsion"] = repulsion.record(len(model["atoms"]))
        implemented_set.update(("bonded_repulsion", "nonbonded_repulsion"))
    if d3_two_body is not None:
        model["gfnff_d3_two_body"] = d3_two_body.record(len(model["atoms"]))
        implemented_set.add("d3_two_body")
    if valence is not None:
        model["gfnff_valence"] = valence.record(len(model["atoms"]))
        implemented_set.update(("angle", "torsion", "special_torsion"))
    if nci is not None:
        model["gfnff_nci"] = nci.record(len(model["atoms"]))
        if nci.coverage_certified:
            implemented_set.update(("hydrogen_bond", "halogen_bond", "bonded_atm"))
    implemented = [family for family in GFNFF_REQUIRED_TERM_FAMILIES if family in implemented_set]
    model["functional_profile"] = "GFNFF_NATIVE_MIGRATION"
    missing = [
        item for item in GFNFF_REQUIRED_TERM_FAMILIES if item not in implemented
    ]
    calculator_ready = not missing and dynamic
    model["gfnff_migration"] = {
        "schema": GFNFF_MIGRATION_SCHEMA,
        "reference": prepared_bonds.source,
        "implemented_term_families": implemented,
        "missing_term_families": missing,
        "dynamic_cn_bond_reference": dynamic,
        "calculator_ready": calculator_ready,
        "assembly": (
            "GFNFF_COMPLETE_RESIDENT"
            if calculator_ready
            else "TERM_PARITY_PARTIAL"
        ),
        "fallback": None,
        "validation_reference": "LIBXTB_GFNFF" if not calculator_ready else None,
    }
    if oracle_topology is not None:
        from .force_field_pipeline import prepare_gfnff_model

        return prepare_gfnff_model(oracle_topology, model)
    return model


def attach_gfnff_components_to_zaff(
    seed_model: Mapping[str, Any],
    *,
    components: tuple[str, ...] | list[str],
    coordination: GfnffPreparedCoordinationLayer | None = None,
    eeq: GfnffPreparedEEQLayer | None = None,
    valence: GfnffPreparedValenceLayer | None = None,
    nci: GfnffPreparedNCILayer | None = None,
) -> dict[str, Any]:
    """Attach independently selectable GFN-FF families to a ZAFF seed field.

    This is deliberately distinct from :func:`compile_gfnff_migration_model`:
    the covalent and Exp-PE layers remain ZAFF.  ``eeq`` (also accepted as
    ``qeq``) replaces ZAFF electrostatics/charge response, while special
    torsions and H/X-bonds are additive GFN-FF corrections.  Bonded ATM is
    always removed from the hybrid record.
    """

    if seed_model.get("schema") != ZAFF_SEED_MODEL_SCHEMA:
        raise ValueError("GFN-FF components require a compiled ZAFF seed model")
    aliases = {
        "qeq": "eeq",
        "gfnff_qeq": "eeq",
        "gfnff_eeq": "eeq",
        "special_torsions": "special_torsion",
        "h_bond": "hydrogen_bond",
        "hbond": "hydrogen_bond",
        "x_bond": "halogen_bond",
        "xbond": "halogen_bond",
    }
    selected = tuple(
        dict.fromkeys(
            aliases.get(str(value).strip().casefold(), str(value).strip().casefold())
            for value in components
        )
    )
    allowed = {"eeq", "special_torsion", "hydrogen_bond", "halogen_bond"}
    unknown = sorted(set(selected) - allowed)
    if unknown or not selected:
        suffix = ": " + ", ".join(unknown) if unknown else ""
        raise ValueError("unsupported or empty ZAFF/GFN-FF component selection" + suffix)
    natoms = len(seed_model["atoms"])
    model = deepcopy(dict(seed_model))
    for key in (
        "gfnff_coordination",
        "gfnff_eeq",
        "gfnff_valence",
        "gfnff_nci",
        "zaff_gfnff_components",
    ):
        model.pop(key, None)
    if "eeq" in selected:
        if coordination is None or eeq is None:
            raise ValueError("ZAFF/GFN-FF EEQ requires coordination and EEQ layers")
        model["gfnff_coordination"] = coordination.record(natoms)
        model["gfnff_eeq"] = eeq.record(natoms)
    if "special_torsion" in selected:
        if valence is None:
            raise ValueError("ZAFF/GFN-FF special torsions require a valence layer")
        record = valence.record(natoms)
        record.update(
            {
                "angle_atoms": [],
                "angle_parameters": [],
                "torsion_atoms": [],
                "torsion_parameters": [],
            }
        )
        model["gfnff_valence"] = record
    if {"hydrogen_bond", "halogen_bond"} & set(selected):
        if nci is None or not nci.coverage_certified:
            raise ValueError("ZAFF/GFN-FF H/X-bonds require a certified NCI layer")
        record = nci.record(natoms)
        if "hydrogen_bond" not in selected:
            record["hydrogen_bond_atoms"] = []
            record["hydrogen_bond_parameters"] = []
        if "halogen_bond" not in selected:
            record["halogen_bond_atoms"] = []
            record["halogen_bond_parameters"] = []
        record["bonded_atm_atoms"] = []
        record["bonded_atm_amplitudes_hartree_bohr9"] = []
        model["gfnff_nci"] = record
    model["zaff_gfnff_components"] = {
        "schema": ZAFF_GFNFF_COMPONENTS_SCHEMA,
        "enabled": list(selected),
        "charge_model": "GFNFF_EEQ" if "eeq" in selected else "ZAFF_QEQ_SQE",
        "base_force_field": "ZAFF",
        "bonded_atm": False,
        "double_counting_policy": (
            "EEQ_REPLACES_ZAFF_ELECTROSTATICS_AND_CHARGE_RESPONSE;"
            "SPECIAL_TORSION_AND_H_X_BONDS_ARE_ADDITIVE_CORRECTIONS"
        ),
    }
    return model


def assert_complete_gfnff_model(model: Mapping[str, Any]) -> None:
    """Reject partial migration artifacts at the calculator boundary."""

    migration = model.get("gfnff_migration")
    if not isinstance(migration, Mapping) or migration.get("schema") != GFNFF_MIGRATION_SCHEMA:
        raise ValueError("model is not an ARCHITECT GFN-FF migration artifact")
    if migration.get("calculator_ready") is not True:
        missing = ", ".join(str(value) for value in migration.get("missing_term_families", ()))
        raise ValueError(f"native GFN-FF model is incomplete: {missing}")


def load_complete_gfnff_model(path: Path | str) -> dict[str, Any]:
    """Load and certify the ARCHITECT model consumed by native GFN-FF.

    Both the compact compiler output and a serialized force-field artifact are
    accepted.  This is the sole file-format boundary used by KeyMaker; partial
    term-parity records never reach LINK.
    """

    source = Path(path).expanduser().resolve()
    record = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(record, Mapping):
        raise ValueError("ARCHITECT GFN-FF artifact must contain a JSON object")
    metadata = record.get("metadata", {})
    model = metadata.get("diagonal_seed_model", record) if isinstance(metadata, Mapping) else record
    if not isinstance(model, Mapping):
        raise ValueError("ARCHITECT artifact has no compiled GFN-FF model")
    assert_complete_gfnff_model(model)
    return deepcopy(dict(model))


__all__ = [
    "GFNFF_MIGRATION_SCHEMA",
    "GFNFF_REFERENCE_VERSION",
    "GFNFF_REQUIRED_TERM_FAMILIES",
    "GfnffPreparedCoordinationLayer",
    "GfnffPreparedEEQLayer",
    "GfnffPreparedRepulsionLayer",
    "GfnffPreparedD3Layer",
    "GfnffPreparedValenceLayer",
    "GfnffPreparedNCILayer",
    "GfnffPreparedBondLayer",
    "assert_complete_gfnff_model",
    "compile_gfnff_migration_model",
    "load_complete_gfnff_model",
]
