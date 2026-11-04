"""Resident TIP3P-FB water and translated NPBC UvdW parameters."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import json
import math
from typing import Any

import numpy as np

from .compatibility import normalize_legacy_zaff_payload

from matrix_epoc.confinement import (
    EllipsoidalVdwConfinement,
    ellipsoidal_molecular_constraint_violation,
    ellipsoidal_repulsive_site_boundary,
)


ANGSTROM_TO_BOHR = 1.8897261254578281
KJ_PER_MOL_TO_HARTREE = 1.0 / 2625.4996394799
KCAL_PER_MOL_TO_HARTREE = 1.0 / 627.5094740631
TIP3P_FB_SCHEMA = "matrix.zaff.water_model.v1"
TIP3P_FB_UVDW_TYPE = "TIP3P-FB-O"
TIP3P_FB_RIGID_SITE_BARRIER_TYPE = "TIP3P-FB-RIGID-SITE"


@dataclass(frozen=True)
class Tip3pFbParameters:
    """Rigid three-site water parameters in their conventional units."""

    oh_distance_angstrom: float
    hoh_angle_degree: float
    oxygen_charge_e: float
    hydrogen_charge_e: float
    oxygen_sigma_angstrom: float
    oxygen_epsilon_kcal_per_mol: float
    oxygen_mass_dalton: float
    hydrogen_mass_dalton: float
    provenance: dict[str, str]

    @property
    def charges_e(self) -> np.ndarray:
        return np.asarray(
            [self.oxygen_charge_e, self.hydrogen_charge_e, self.hydrogen_charge_e]
        )

    def reference_geometry_bohr(self) -> np.ndarray:
        """Return O, H, H in the xy plane with the bisector on +x."""

        half_angle = math.radians(self.hoh_angle_degree) / 2.0
        distance = self.oh_distance_angstrom * ANGSTROM_TO_BOHR
        return np.asarray(
            [
                [0.0, 0.0, 0.0],
                [
                    distance * math.cos(half_angle),
                    distance * math.sin(half_angle),
                    0.0,
                ],
                [
                    distance * math.cos(half_angle),
                    -distance * math.sin(half_angle),
                    0.0,
                ],
            ]
        )


def _record() -> dict[str, Any]:
    path = files("matrix_zaff").joinpath("data/tip3p_fb.json")
    record = normalize_legacy_zaff_payload(
        json.loads(path.read_text(encoding="utf-8"))
    )
    if record.get("schema") != TIP3P_FB_SCHEMA:
        raise ValueError("unsupported resident TIP3P-FB record")
    return record


def tip3p_fb_parameters() -> Tip3pFbParameters:
    """Load the resident TIP3P-FB molecular parameters."""

    record = _record()
    geometry = record["geometry"]
    oxygen = record["sites"]["O"]
    hydrogen = record["sites"]["H"]
    return Tip3pFbParameters(
        oh_distance_angstrom=float(geometry["oh_distance_angstrom"]),
        hoh_angle_degree=float(geometry["hoh_angle_degree"]),
        oxygen_charge_e=float(oxygen["charge_e"]),
        hydrogen_charge_e=float(hydrogen["charge_e"]),
        oxygen_sigma_angstrom=float(oxygen["sigma_angstrom"]),
        oxygen_epsilon_kcal_per_mol=float(
            oxygen["epsilon_kcal_per_mol"]
        ),
        oxygen_mass_dalton=float(oxygen["mass_dalton"]),
        hydrogen_mass_dalton=float(hydrogen["mass_dalton"]),
        provenance={
            str(key): str(value)
            for key, value in record["provenance"].items()
        },
    )


def tip3p_fb_source_uvdw_kj_per_mol(
    distance_from_wall_angstrom: np.ndarray | float,
    *,
    compact: bool = True,
) -> np.ndarray:
    """Evaluate the published polynomial, optionally with ZAFF's C2 cutoff."""

    record = _record()["uvdw"]
    distance = np.asarray(distance_from_wall_angstrom, dtype=float)
    if np.any(distance < 0.0):
        raise ValueError("distance from the wall cannot be negative")
    polynomial = np.polynomial.polynomial.polyval(
        distance,
        np.asarray(record["source_polynomial_coefficients_ascending"]),
    )
    if not compact:
        return polynomial
    cutoff = float(record["compact_cutoff_angstrom"])
    x = np.minimum(distance / cutoff, 1.0)
    switch = 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5
    return np.where(distance < cutoff, switch * polynomial, 0.0)


def tip3p_fb_uvdw_confinement(
    center_bohr: np.ndarray,
    semiaxes_bohr: np.ndarray,
    rotation: np.ndarray,
) -> EllipsoidalVdwConfinement:
    """Instantiate the resident Morse-Gaussian translation on an ellipsoid."""

    record = _record()["uvdw"]
    return EllipsoidalVdwConfinement(
        center_bohr=np.asarray(center_bohr, dtype=float),
        semiaxes_bohr=np.asarray(semiaxes_bohr, dtype=float),
        rotation=np.asarray(rotation, dtype=float),
        morse_parameters={
            TIP3P_FB_UVDW_TYPE: tuple(
                float(value)
                for value in record["morse_parameters_atomic_units"]
            )
        },
        gaussian_terms={
            TIP3P_FB_UVDW_TYPE: tuple(
                tuple(float(value) for value in term)
                for term in record["gaussian_terms_atomic_units"]
            )
        },
        layer_depth_bohr=(
            float(record["compact_cutoff_angstrom"]) * ANGSTROM_TO_BOHR
        ),
    )


def tip3p_fb_rigid_site_confinement(
    center_bohr: np.ndarray,
    semiaxes_bohr: np.ndarray,
    rotation: np.ndarray,
) -> EllipsoidalVdwConfinement:
    """Return the common all-site repulsive wall used by MD, MC, and GA."""

    return ellipsoidal_repulsive_site_boundary(
        center_bohr,
        semiaxes_bohr,
        rotation,
        (TIP3P_FB_RIGID_SITE_BARRIER_TYPE,),
        wall_height_hartree=25.0 * KCAL_PER_MOL_TO_HARTREE,
        decay_per_bohr=3.0 / ANGSTROM_TO_BOHR,
        layer_depth_bohr=1.5 * ANGSTROM_TO_BOHR,
    )


__all__ = [
    "ANGSTROM_TO_BOHR",
    "KCAL_PER_MOL_TO_HARTREE",
    "KJ_PER_MOL_TO_HARTREE",
    "TIP3P_FB_SCHEMA",
    "TIP3P_FB_RIGID_SITE_BARRIER_TYPE",
    "TIP3P_FB_UVDW_TYPE",
    "Tip3pFbParameters",
    "ellipsoidal_molecular_constraint_violation",
    "ellipsoidal_repulsive_site_boundary",
    "tip3p_fb_parameters",
    "tip3p_fb_rigid_site_confinement",
    "tip3p_fb_source_uvdw_kj_per_mol",
    "tip3p_fb_uvdw_confinement",
]
