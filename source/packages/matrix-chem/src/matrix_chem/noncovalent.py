"""Common noncovalent-interaction taxonomy and fitting contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

NONCOVALENT_CONTRACT_SCHEMA = "matrix.noncovalent.contract.v1"
NONCOVALENT_INTERACTION_KINDS = (
    "hydrogen-bond", "alkali-bond", "alkaline-earth-bond", "regium-bond",
    "spodium-bond", "triel-bond", "tetrel-bond", "pnictogen-bond",
    "chalcogen-bond", "halogen-bond", "aerogen-bond", "cation-pi",
    "anion-pi", "lone-pair-pi", "pi-pi", "orthogonal",
)
InteractionFamily = Literal["sigma-hole", "pi-hole", "pi-surface", "hydrogen"]

@dataclass(frozen=True)
class NoncovalentInteractionType:
    kind: str
    family: InteractionFamily
    donor_role: str
    acceptor_role: str
    requires_axis: bool
    requires_plane: bool
    cm5_response: str = "pairwise_charge_conserving_response"
    zaff_term: str = "exp_pe_radial_times_directional_angular_kernel"

def _type(kind: str, family: InteractionFamily, donor: str, acceptor: str, *, axis: bool = False, plane: bool = False) -> NoncovalentInteractionType:
    return NoncovalentInteractionType(kind, family, donor, acceptor, axis, plane)

NONCOVALENT_TYPES = {
    "hydrogen-bond": _type("hydrogen-bond", "hydrogen", "D-H", "lone-pair"),
    "alkali-bond": _type("alkali-bond", "sigma-hole", "alkali Lewis acid", "Lewis base", axis=True),
    "alkaline-earth-bond": _type("alkaline-earth-bond", "sigma-hole", "group-2 Lewis acid", "Lewis base", axis=True),
    "regium-bond": _type("regium-bond", "sigma-hole", "group-11 center", "Lewis base", axis=True),
    "spodium-bond": _type("spodium-bond", "sigma-hole", "group-12 center", "Lewis base", axis=True),
    "triel-bond": _type("triel-bond", "pi-hole", "group-13 center", "Lewis base", axis=True),
    "tetrel-bond": _type("tetrel-bond", "pi-hole", "group-14 center", "Lewis base", axis=True),
    "pnictogen-bond": _type("pnictogen-bond", "sigma-hole", "group-15 center", "Lewis base", axis=True),
    "chalcogen-bond": _type("chalcogen-bond", "sigma-hole", "group-16 center", "Lewis base", axis=True),
    "halogen-bond": _type("halogen-bond", "sigma-hole", "halogen", "Lewis base", axis=True),
    "aerogen-bond": _type("aerogen-bond", "sigma-hole", "noble-gas center", "Lewis base", axis=True),
    "cation-pi": _type("cation-pi", "pi-surface", "cation", "pi surface", plane=True),
    "anion-pi": _type("anion-pi", "pi-surface", "anion", "electron-poor pi surface", plane=True),
    "lone-pair-pi": _type("lone-pair-pi", "pi-surface", "lone pair", "pi surface", plane=True),
    "pi-pi": _type("pi-pi", "pi-surface", "pi surface", "pi surface", plane=True),
    "orthogonal": _type("orthogonal", "pi-surface", "pi surface", "orthogonal pi surface", plane=True),
}

@dataclass(frozen=True)
class NoncovalentFittingPlan:
    interaction_kind: str
    monomer_scan: str
    dimer_scan: str
    electronic_levels: tuple[str, ...]
    observables: tuple[str, ...]
    fit_targets: tuple[str, ...]
    validation: tuple[str, ...]
    schema: str = NONCOVALENT_CONTRACT_SCHEMA

def noncovalent_fitting_plan(kind: str) -> NoncovalentFittingPlan:
    normalized = str(kind).strip().lower().replace("_", "-")
    if normalized not in NONCOVALENT_TYPES:
        raise ValueError(f"unsupported noncovalent interaction kind: {kind}")
    return NoncovalentFittingPlan(
        normalized,
        "isolated_monomer_reference_and_orientation_scan",
        "counterpoise_dimer_r_theta_orientation_scan",
        ("PBE0-D4/def2-TZVPP", "DLPNO-CCSD(T)/def2-TZVPP"),
        ("CM5_MONOMER_CHARGES", "CM5_DIMER_CHARGES", "ESP_ON_MERZ_KOLLMAN_SURFACE", "COUNTERPOISE_INTERACTION_ENERGY", "CARTESIAN_GRADIENT"),
        ("CM5_RESPONSE_DELTA_CHARGES", "CHARGE_TRANSFER_AND_POLARIZATION_SEPARATION", "ZAFF_RADIAL_EXP_PE_PARAMETERS", "ZAFF_DIRECTIONAL_KERNEL_PARAMETERS", "ZAFF_FORCE_AND_ENERGY_RESIDUALS"),
        ("held_out_substituents_and_orientations", "monomer_charge_conservation", "dimer_energy_force_consistency", "cross_interaction_competition_with_hbond_and_xb"),
    )

def all_noncovalent_fitting_plans(kinds: Sequence[str] | None = None) -> tuple[NoncovalentFittingPlan, ...]:
    selected = NONCOVALENT_INTERACTION_KINDS if kinds is None else tuple(kinds)
    return tuple(noncovalent_fitting_plan(kind) for kind in selected)


def build_noncovalent_job_manifest(
    *,
    system_id: str,
    kinds: Sequence[str] | None = None,
    radial_grid_angstrom: Sequence[float] = (2.4, 2.8, 3.2, 3.6, 4.2, 5.0),
    polar_grid_degrees: Sequence[float] = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0),
    azimuth_grid_degrees: Sequence[float] = (0.0, 90.0, 180.0, 270.0),
    high_level_stride: int = 8,
) -> dict[str, object]:
    """Create a deterministic QM-job manifest for an NCI fitting campaign.

    Geometry files and molecular identities are supplied by the caller; this
    function only defines the jobs and observables, so it is safe to audit or
    submit on any backend (Gaussian, ORCA, or Psi4).
    """
    if not str(system_id).strip():
        raise ValueError("system_id cannot be empty")
    if int(high_level_stride) < 1:
        raise ValueError("high_level_stride must be positive")
    selected = NONCOVALENT_INTERACTION_KINDS if kinds is None else tuple(kinds)
    plans = [noncovalent_fitting_plan(kind) for kind in selected]
    radial = tuple(float(value) for value in radial_grid_angstrom)
    polar = tuple(float(value) for value in polar_grid_degrees)
    azimuth = tuple(float(value) for value in azimuth_grid_degrees)
    if not radial or any(value <= 0.0 for value in radial):
        raise ValueError("radial_grid_angstrom must contain positive values")
    jobs: list[dict[str, object]] = []
    for plan in plans:
        prefix = plan.interaction_kind.replace("-", "_")
        jobs.append({"id": f"{prefix}_monomer_A_cm5", "kind": plan.interaction_kind, "task": "monomer_population", "method": plan.electronic_levels[0], "observables": ["CM5_CHARGES", "ESP"]})
        jobs.append({"id": f"{prefix}_monomer_B_cm5", "kind": plan.interaction_kind, "task": "monomer_population", "method": plan.electronic_levels[0], "observables": ["CM5_CHARGES", "ESP"]})
        point = 0
        for distance in radial:
            for polar_angle in polar:
                for azimuth_angle in azimuth:
                    jobs.append({
                        "id": f"{prefix}_dimer_{point:05d}",
                        "kind": plan.interaction_kind,
                        "task": "counterpoise_single_point",
                        "method": plan.electronic_levels[0] if point % int(high_level_stride) else plan.electronic_levels[1],
                        "distance_angstrom": distance,
                        "polar_angle_degrees": polar_angle,
                        "azimuth_degrees": azimuth_angle,
                        "observables": ["CM5_CHARGES", "ESP", "CP_INTERACTION_ENERGY", "CARTESIAN_GRADIENT"],
                    })
                    point += 1
    return {
        "schema": "matrix.noncovalent.job_manifest.v1",
        "system_id": str(system_id),
        "plans": [plan.__dict__ for plan in plans],
        "job_count": len(jobs),
        "jobs": jobs,
        "fit": {"charge_response": "CM5_DIMER_MINUS_MONOMERS", "energy_target": "CP_RESIDUAL_AFTER_ORDINARY_EXPPE", "force_weighting": "JOINT_ENERGY_FORCE_CV"},
    }

__all__ = ["NONCOVALENT_CONTRACT_SCHEMA", "NONCOVALENT_INTERACTION_KINDS", "NONCOVALENT_TYPES", "NoncovalentFittingPlan", "NoncovalentInteractionType", "all_noncovalent_fitting_plans", "build_noncovalent_job_manifest", "noncovalent_fitting_plan"]
