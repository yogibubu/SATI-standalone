"""Resident execution of transferable diagonal ZAFF seed fields."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .compatibility import normalize_legacy_zaff_payload

from matrix_chem import (
    HydrogenBondPairList,
    Primitive,
    eval_primitive,
    prepare_hydrogen_bond_recognition,
    primitive_b_matrix,
)
from matrix_chem.topology.covalent_radii import covalent_radius
from matrix_chem.topology.descriptor_parameters import CNA_SCALE
from matrix_chem.topology.elements import atomic_number
from matrix_chem.spatial_regions import bounded_topological_distances
from matrix_chem.continuous_nonbonded import (
    continuous_topology_pair_corrections,
    continuous_topology_pair_scales,
)

from .basic_primitive_derivative import analytic_basic_primitive_hessian_vector
from .bonded import (
    BondOrderDampedAnglePotential,
    BondOrderDampedOutOfPlanePotential,
    BondOrderDampedTorsionPotential,
    BondOrderRadialFactor,
    GFNFFDynamicGaussianBondModel,
    GaussianBondPotential,
    UFFFourierTerm,
    SinOutOfPlanePolynomial,
)
from .charge_response import (
    ChargeResponseSite,
    PersistentAllPairSplitChargeResponse,
    PersistentLocalPairSplitChargeResponse,
    PersistentSplitChargeResponse,
    SplitChargeChannel,
)
from matrix_epoc.confinement import EllipsoidalVdwConfinement, vdw_confinement_from_record
from matrix_epoc.cpcm import CPCMReactionField, cpcm_confinement_from_record
from .interfacial_pcm import (
    InterfacialPCMReactionField,
    interfacial_pcm_reaction_field_from_record,
)
from .gfnff_nci import GFNFFNCIModel
from .gfnff_eeq import GFNFFEEQSession
from .gfnff_valence import GFNFFValenceModel
from .local_charge_electrostatics import evaluate_local_charge_electrostatics
from .native_kernels import (
    exppe_energy as _native_exppe_energy,
    exppe_energy_gradient as _native_exppe_energy_gradient,
    exppe_hessian_vector as _native_exppe_hessian_vector,
    dynamic_gaussian_bond_energy_gradient as _native_dynamic_gaussian_bond_energy_gradient,
    dynamic_gaussian_bond_hessian_vector as _native_dynamic_gaussian_bond_hessian_vector,
    gaussian_bond_energy as _native_gaussian_bond_energy,
    gaussian_bond_energy_gradient as _native_gaussian_bond_energy_gradient,
    gaussian_bond_hessian_vector as _native_gaussian_bond_hessian_vector,
    local_valence_energy as _native_local_valence_energy,
    local_valence_energy_gradient as _native_local_valence_energy_gradient,
    local_valence_hessian_vector as _native_local_valence_hessian_vector,
    morse_bond_energy as _native_morse_bond_energy,
    morse_bond_energy_gradient as _native_morse_bond_energy_gradient,
    morse_bond_hessian_vector as _native_morse_bond_hessian_vector,
    native_zaff_backend,
)
from .nonbonded import (
    BOHR_TO_ANGSTROM,
    MMRuntimePolicy,
    PersistentGaussianElectrostaticOperator,
    PersistentVerletNeighborList,
    build_nonbonded_neighbor_list,
    electrostatic_energy_gradient,
    electrostatic_hessian_vector_product,
    select_mm_runtime_policy,
)
from .radial import (
    InversePowerStretchPotential,
    SPFStretchPotential,
    exppe_from_minimum,
    mie_minimum_derivatives,
)

ZAFF_SEED_MODEL_SCHEMA = "matrix.zaff.diagonal_seed.v2"
ZAFF_SEED_PRIOR_SCHEMA = "matrix.zaff.full_fit_priors.v1"
ZAFF_FAST_RIGID_TORSIONAL_ZOOM = "FAST_RIGID_TORSIONAL"
KCAL_MOL_TO_HARTREE = 1.0 / 627.5094740631
WaterChargeResponseEvaluator = Callable[..., Any]
ZAFF_GFNFF_COMPONENTS_SCHEMA = "matrix.zaff.gfnff_components.v1"
ZAFF_CONTINUOUS_EXPPE_POLICY = (
    "CONTINUOUS_ERF_PATH_WEIGHTS_FOR_EXPPE_SEPARATE_FROM_COVALENT"
)
ZAFF_LEGACY_COVALENT_EXPPE_PARTITION_POLICY = (
    "CONTINUOUS_COVALENT_EXPPE_PARTITION_ERF_PATHS_FULL_1_4"
)


def _zaff_gfnff_components(model: Mapping[str, Any]) -> frozenset[str]:
    record = model.get("zaff_gfnff_components")
    if record is None:
        return frozenset()
    if not isinstance(record, Mapping) or record.get("schema") != ZAFF_GFNFF_COMPONENTS_SCHEMA:
        raise ValueError("unsupported ZAFF/GFN-FF component contract")
    enabled = frozenset(str(value).strip().casefold() for value in record.get("enabled", ()))
    allowed = {"eeq", "special_torsion", "hydrogen_bond", "halogen_bond"}
    if not enabled.issubset(allowed):
        raise ValueError("ZAFF/GFN-FF component contract contains an unknown family")
    return enabled


def _selected_charge_response_model(
    model: Mapping[str, Any], requested: str | None
) -> str:
    eeq_enabled = "eeq" in _zaff_gfnff_components(model)
    if requested is None:
        return "gfnff_eeq" if eeq_enabled else "variational_qeq_sqe"
    selected = str(requested).strip().casefold()
    if (selected == "gfnff_eeq") != eeq_enabled:
        raise ValueError(
            "the ZAFF/GFN-FF EEQ runtime selection must match the compiled artifact"
        )
    return selected


def _gfnff_special_torsions_from_model(
    model: Mapping[str, Any],
) -> GFNFFValenceModel | None:
    if "special_torsion" not in _zaff_gfnff_components(model):
        return None
    record = model.get("gfnff_valence")
    if not isinstance(record, Mapping):
        raise ValueError("ZAFF requests GFN-FF special torsions but lacks their record")
    valence = GFNFFValenceModel.from_record(record)
    if len(valence.angle_atoms) or len(valence.torsion_atoms):
        raise ValueError("ZAFF/GFN-FF special-torsion record contains ordinary valence terms")
    return valence


def _gfnff_nci_from_model(model: Mapping[str, Any]) -> GFNFFNCIModel | None:
    enabled = _zaff_gfnff_components(model)
    if not ({"hydrogen_bond", "halogen_bond"} & enabled):
        return None
    record = model.get("gfnff_nci")
    if not isinstance(record, Mapping):
        raise ValueError("ZAFF requests GFN-FF H/X-bonds but lacks their record")
    nci = GFNFFNCIModel.from_record(record)
    if len(nci.bonded_atm_atoms):
        raise ValueError("bonded ATM is not permitted in the ZAFF/GFN-FF hybrid")
    if "hydrogen_bond" not in enabled and len(nci.hydrogen_bond_atoms):
        raise ValueError("ZAFF/GFN-FF NCI record contains an unselected H-bond family")
    if "halogen_bond" not in enabled and len(nci.halogen_bond_atoms):
        raise ValueError("ZAFF/GFN-FF NCI record contains an unselected X-bond family")
    return nci


class ZaffSeedCoverageError(ValueError):
    """The transferable library cannot parameterize the requested molecule."""


@dataclass(frozen=True)
class ZaffSeedEvaluation:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray | None
    execution: Mapping[str, Any]


class ZaffSeedDynamicsRuntime:
    """Persistent E+G runtime for repeated ZAFF seed evaluations."""

    def __init__(
        self,
        model: Mapping[str, Any],
        *,
        zoom_level: str | None = None,
        neighbor_skin_angstrom: float = 1.0,
        charge_response_model: str | None = None,
        water_charge_response_evaluator: WaterChargeResponseEvaluator | None = None,
        exclude_electrostatics_12_13: bool = False,
    ) -> None:
        self.model = model
        self.fast = _fast_zoom_selected(zoom_level)
        self.zoom_level = zoom_level
        response_model = _selected_charge_response_model(model, charge_response_model)
        from .electrostatic_exclusions import exclusions_12_13, ExcludedGaussianElectrostaticOperator

        self.electrostatic_excluded_pairs = (
            exclusions_12_13(len(model["atoms"]), (item["atoms"] for item in model["bonds"]))
            if exclude_electrostatics_12_13 else ()
        )
        if exclude_electrostatics_12_13 and response_model != "variational_qeq_sqe":
            raise ValueError("diagnostic electrostatic exclusions require the variational QEq/SQE model")
        if response_model not in {
            "variational_qeq_sqe",
            "water_hbond_surrogate",
            "gfnff_eeq",
        }:
            raise ValueError("unsupported ZAFF charge-response runtime model")
        if self.fast and response_model != "variational_qeq_sqe":
            raise ValueError(
                "rigid-torsional zoom cannot be combined with an alternative "
                "charge-response contract"
            )
        self.charge_response_model = response_model
        self.water_charge_response_evaluator = water_charge_response_evaluator
        self.evaluation_count = 0
        nonbonded = dict(model["nonbonded"])
        self.policy = select_mm_runtime_policy(
            np.zeros((len(model["atoms"]), 3)),
            cutoff_angstrom=float(nonbonded["short_range_cutoff_angstrom"]),
            neighbor_list_minimum_atoms=int(
                nonbonded["neighbor_list_minimum_atoms"]
            ),
            fmm_minimum_atoms=int(nonbonded["fmm_minimum_atoms"]),
            fmm_precision=float(nonbonded["fmm_precision"]),
            materialize_neighbor_pairs=False,
            require_linear_scaling=True,
        )
        self.neighbor_list = PersistentVerletNeighborList(
            cutoff_bohr=(
                float(nonbonded["short_range_cutoff_angstrom"])
                / BOHR_TO_ANGSTROM
            ),
            skin_bohr=float(neighbor_skin_angstrom) / BOHR_TO_ANGSTROM,
        )
        atom_parameters = tuple(model["nonbonded_atoms"])
        self.electrostatics = PersistentGaussianElectrostaticOperator(
            np.asarray(
                [float(item["atomic_charge"]) for item in atom_parameters]
            ),
            tuple(int(atomic_number(symbol) or 0) for symbol in model["atoms"]),
            backend=self.policy.electrostatic_backend,
            fmm_precision=float(nonbonded["fmm_precision"]),
            fmm_minimum_atoms=int(nonbonded["fmm_minimum_atoms"]),
        )
        self.charge_response = _persistent_charge_response_from_model(
            model, backend=self.policy.electrostatic_backend, reaction_field=_model_reaction_field(model),
            excluded_atom_pairs=self.electrostatic_excluded_pairs,
        ) if response_model == "variational_qeq_sqe" else None
        if self.electrostatic_excluded_pairs:
            self.electrostatics = ExcludedGaussianElectrostaticOperator(
                self.electrostatics, self.electrostatic_excluded_pairs)
        self.gfnff_eeq = (
            GFNFFEEQSession.from_model(model)
            if response_model == "gfnff_eeq"
            else None
        )
        self.hbond_pair_list = None
        if response_model == "water_hbond_surrogate":
            numbers = tuple(
                int(atomic_number(symbol) or 0) for symbol in model["atoms"]
            )
            bonds = tuple(
                tuple(int(value) for value in item["atoms"])
                for item in model["bonds"]
            )
            self.hbond_pair_list = prepare_hydrogen_bond_recognition(
                numbers,
                bonds,
                selector_threshold=1.0e-8,
                minimum_angle_degrees=90.0,
            ).new_pair_list(skin_angstrom=float(neighbor_skin_angstrom))
        self.morse_bond_arrays = _seed_morse_bond_arrays(model)
        self.gaussian_bond_arrays = _seed_gaussian_bond_arrays(model)
        self.local_valence_arrays = _seed_local_valence_arrays(model)
        self.shared_valence = _gfnff_special_torsions_from_model(model)
        self.shared_nci = _gfnff_nci_from_model(model)

    def evaluate(self, coordinates_angstrom: np.ndarray) -> ZaffSeedEvaluation:
        energy, gradient, execution = _seed_energy_gradient(
            self.model,
            coordinates_angstrom,
            fast=self.fast,
            charge_response_model=self.charge_response_model,
            water_charge_response_evaluator=self.water_charge_response_evaluator,
            hbond_pair_list=self.hbond_pair_list,
            neighbor_list=self.neighbor_list,
            electrostatic_operator=self.electrostatics,
            charge_response_runtime=self.charge_response,
            runtime_policy=self.policy,
            morse_bond_arrays=self.morse_bond_arrays,
            gaussian_bond_arrays=self.gaussian_bond_arrays,
            local_valence_arrays=self.local_valence_arrays,
            shared_valence_model=self.shared_valence,
            shared_nci_model=self.shared_nci,
            gfnff_eeq_session=self.gfnff_eeq,
        )
        self.evaluation_count += 1
        execution = {
            **execution,
            "persistent_neighbor_list": True,
            "electrostatic_excluded_pairs": self.electrostatic_excluded_pairs,
            "neighbor_list_rebuild_count": self.neighbor_list.rebuild_count,
            "neighbor_list_reuse_count": self.neighbor_list.reuse_count,
            "persistent_electrostatic_operator": self.gfnff_eeq is None,
            "persistent_gfnff_eeq": self.gfnff_eeq is not None,
            "persistent_fmm": (
                self.electrostatics.persistent_fmm
                if self.gfnff_eeq is None
                else False
            ),
            "hbond_pair_list_rebuild_count": (
                0
                if self.hbond_pair_list is None
                else self.hbond_pair_list.rebuild_count
            ),
            "electrostatic_evaluation_count": self.electrostatics.evaluation_count,
            "gfnff_eeq_evaluation_count": (
                0 if self.gfnff_eeq is None else self.gfnff_eeq.evaluation_count
            ),
        }
        return ZaffSeedEvaluation(energy, gradient, None, execution)

    def hessian_vector_product(
        self,
        coordinates_angstrom: np.ndarray,
        vector_bohr: np.ndarray,
    ) -> np.ndarray:
        """Apply the analytic Hessian with all reusable runtime state resident."""

        return _seed_hessian_vector_product(
            self.model,
            coordinates_angstrom,
            vector_bohr,
            fast=self.fast,
            neighbor_list=self.neighbor_list,
            electrostatic_operator=self.electrostatics,
            charge_response_runtime=self.charge_response,
            runtime_policy=self.policy,
            morse_bond_arrays=self.morse_bond_arrays,
            gaussian_bond_arrays=self.gaussian_bond_arrays,
            local_valence_arrays=self.local_valence_arrays,
            charge_response_model=self.charge_response_model,
            shared_valence_model=self.shared_valence,
            shared_nci_model=self.shared_nci,
            gfnff_eeq_session=self.gfnff_eeq,
        )


class ZaffSecondOrderAnchoredSeedRuntime:
    """Expose only the beyond-harmonic part of a resident seed model.

    The compiled force field owns the reference energy, gradient and Hessian.
    Physical kernels may refine the potential away from that point, but must
    not alter its zero-, first- or second-order jet.  This wrapper subtracts
    that complete Cartesian jet once and reuses it during dynamics.
    """

    def __init__(
        self,
        model: Mapping[str, Any],
        reference_coordinates_angstrom: np.ndarray,
        *,
        zoom_level: str | None = None,
        neighbor_skin_angstrom: float = 1.0,
        charge_response_model: str | None = None,
    ) -> None:
        self.reference_coordinates_angstrom = np.asarray(
            reference_coordinates_angstrom, dtype=float
        ).copy()
        self.runtime = ZaffSeedDynamicsRuntime(
            model,
            zoom_level=zoom_level,
            neighbor_skin_angstrom=neighbor_skin_angstrom,
            charge_response_model=charge_response_model,
        )
        reference = self.runtime.evaluate(self.reference_coordinates_angstrom)
        self.reference_energy_hartree = float(reference.energy_hartree)
        self.reference_gradient_hartree_per_bohr = np.asarray(
            reference.gradient_hartree_per_bohr, dtype=float
        ).reshape(-1)
        self.reference_hessian_hartree_per_bohr2 = self._raw_hessian(
            self.reference_coordinates_angstrom
        )
        self._reset_runtime_counters()

    def _reset_runtime_counters(self) -> None:
        """Exclude reference-jet compilation from resident runtime telemetry."""

        runtime = self.runtime
        runtime.evaluation_count = 0
        for state in (
            runtime.neighbor_list,
            runtime.electrostatics,
            getattr(runtime.electrostatics, "penetration_pairs", None),
            runtime.charge_response,
            runtime.gfnff_eeq,
        ):
            if state is not None and hasattr(state, "evaluation_count"):
                state.evaluation_count = 0
            if state is not None and hasattr(state, "rebuild_count"):
                state.rebuild_count = 0
            if state is not None and hasattr(state, "reuse_count"):
                state.reuse_count = 0
            if isinstance(state, PersistentVerletNeighborList):
                state.reference_coordinates_bohr = None

    def _raw_hessian(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        size = 3 * len(self.reference_coordinates_angstrom)
        identity = np.eye(size, dtype=float)
        columns = tuple(
            np.asarray(
                self.runtime.hessian_vector_product(
                    coordinates_angstrom, identity[:, column]
                ),
                dtype=float,
            ).reshape(-1)
            for column in range(size)
        )
        hessian = np.column_stack(columns)
        return 0.5 * (hessian + hessian.T)

    def evaluate(self, coordinates_angstrom: np.ndarray) -> ZaffSeedEvaluation:
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        raw = self.runtime.evaluate(coordinates)
        displacement_bohr = (
            coordinates - self.reference_coordinates_angstrom
        ).reshape(-1) / BOHR_TO_ANGSTROM
        harmonic_gradient = (
            self.reference_gradient_hartree_per_bohr
            + self.reference_hessian_hartree_per_bohr2 @ displacement_bohr
        )
        harmonic_energy = (
            self.reference_energy_hartree
            + float(self.reference_gradient_hartree_per_bohr @ displacement_bohr)
            + 0.5
            * float(
                displacement_bohr
                @ self.reference_hessian_hartree_per_bohr2
                @ displacement_bohr
            )
        )
        return ZaffSeedEvaluation(
            float(raw.energy_hartree - harmonic_energy),
            np.asarray(raw.gradient_hartree_per_bohr, dtype=float).reshape(-1)
            - harmonic_gradient,
            None,
            {
                **dict(raw.execution),
                "second_order_anchor": "REFERENCE_CARTESIAN_JET_SUBTRACTED",
            },
        )

    def hessian(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        """Return the seed Hessian after subtraction of its reference Hessian."""

        return (
            self._raw_hessian(np.asarray(coordinates_angstrom, dtype=float))
            - self.reference_hessian_hartree_per_bohr2
        )

    def hessian_vector_product(
        self,
        coordinates_angstrom: np.ndarray,
        vector_bohr: np.ndarray,
    ) -> np.ndarray:
        """Apply the beyond-harmonic Hessian without materializing it."""

        vector = np.asarray(vector_bohr, dtype=float).reshape(-1)
        return (
            np.asarray(
                self.runtime.hessian_vector_product(coordinates_angstrom, vector),
                dtype=float,
            ).reshape(-1)
            - self.reference_hessian_hartree_per_bohr2 @ vector
        )


@lru_cache(maxsize=16)
def _compile_cpcm_record(payload: str) -> CPCMReactionField:
    return CPCMReactionField.compile(
        cpcm_confinement_from_record(normalize_legacy_zaff_payload(json.loads(payload)))
    )


def _model_cpcm_field(model: Mapping[str, Any]) -> CPCMReactionField | None:
    raw = model.get("cpcm_reaction_field")
    if raw is None:
        return None
    return _compile_cpcm_record(
        json.dumps(dict(raw), sort_keys=True, separators=(",", ":"))
    )


@lru_cache(maxsize=16)
def _compile_interfacial_pcm_record(
    payload: str,
) -> InterfacialPCMReactionField:
    return interfacial_pcm_reaction_field_from_record(
        normalize_legacy_zaff_payload(json.loads(payload))
    )


def _model_interfacial_pcm_field(
    model: Mapping[str, Any],
) -> InterfacialPCMReactionField | None:
    raw = model.get("interfacial_pcm_reaction_field")
    if raw is None:
        return None
    return _compile_interfacial_pcm_record(
        json.dumps(dict(raw), sort_keys=True, separators=(",", ":"))
    )


def _model_reaction_field(
    model: Mapping[str, Any],
) -> CPCMReactionField | InterfacialPCMReactionField | None:
    cpcm = _model_cpcm_field(model)
    interfacial = _model_interfacial_pcm_field(model)
    if cpcm is not None and interfacial is not None:
        raise ValueError(
            "ZAFF model cannot combine homogeneous CPCM and interfacial PCM"
        )
    return cpcm if cpcm is not None else interfacial


def _persistent_charge_response_from_model(
    model: Mapping[str, Any],
    *,
    backend: str,
    reaction_field: CPCMReactionField | InterfacialPCMReactionField | None,
    excluded_atom_pairs: tuple[tuple[int, int], ...] = (),
) -> (
    PersistentSplitChargeResponse
    | PersistentAllPairSplitChargeResponse
    | PersistentLocalPairSplitChargeResponse
):
    response_record = dict(model["charge_response"])
    sites = tuple(
        ChargeResponseSite(
            float(item["reference_charge"]),
            float(item["gaussian_width_bohr"]),
            tuple(
                (int(parent), float(weight))
                for parent, weight in item["atom_weights"]
            ),
            str(item["label"]),
        )
        for item in response_record["sites"]
    )
    nonbonded = dict(model["nonbonded"])
    generator = response_record.get("channel_generator")
    if excluded_atom_pairs:
        # The diagnostic atomic mask must not be guessed for affine/virtual sites.
        if len(sites) != len(model["atoms"]) or any(
            site.atom_weights != ((index, 1.0),) for index, site in enumerate(sites)
        ):
            raise ValueError("diagnostic 1-2/1-3 exclusions require one ordered charge site per atom")
        if generator is None:
            raise ValueError("diagnostic seed exclusions require local-pair response with rebuilt reference biases")
    if generator is not None:
        generated = dict(generator)
        local = generated.get("representation") == "IMPLICIT_LOCAL_PAIR_SITE_SCHUR"
        if not local:
            raise ValueError(
                "legacy all-pair ZAFF response is quadratic; recompile the "
                "artifact with ARCHITECT's local-pair response"
            )
        return PersistentLocalPairSplitChargeResponse(
            sites,
            np.asarray(generated["response_lengths"], dtype=float),
            np.asarray(generated["rmin_half_angstrom"], dtype=float),
            np.asarray(
                response_record["reference_geometry_angstrom"], dtype=float
            )
            / BOHR_TO_ANGSTROM,
            tolerance=1.0e-11,
            excluded_pairs=excluded_atom_pairs,
            backend=backend,
            fmm_precision=float(nonbonded["fmm_precision"]),
            fmm_minimum_sites=int(nonbonded["fmm_minimum_atoms"]),
            reaction_field=reaction_field,
            switch_minimum_angstrom=float(
                generated.get("switch_minimum_angstrom", 4.0)
            ),
            switch_scale=float(generated.get("switch_scale", 1.25)),
            switch_exponent=int(generated.get("switch_exponent", 6)),
            pair_chunk_size=int(generated.get("pair_chunk_size", 65536)),
            channel_cutoff_angstrom=float(
                generated["channel_cutoff_angstrom"]
            ),
            taper_start_fraction=float(
                generated.get("taper_start_fraction", 0.75)
            ),
        )
    channels = tuple(
        SplitChargeChannel(
            int(item["left_site"]),
            int(item["right_site"]),
            float(item["hardness_hartree"]),
            float(item["switch_radius_bohr"]),
            int(item["switch_exponent"]),
            float(item["reference_bias_hartree"]),
        )
        for item in response_record["channels"]
    )
    degree = np.zeros(len(sites), dtype=np.intp)
    for channel in channels:
        degree[channel.left_site] += 1
        degree[channel.right_site] += 1
    if int(np.max(degree, initial=0)) > 64:
        raise ValueError(
            "ZAFF explicit response exceeds the bounded local degree required "
            "for linear scaling"
        )
    return PersistentSplitChargeResponse(
        sites,
        channels,
        tolerance=1.0e-11,
        backend=backend,
        fmm_precision=float(nonbonded["fmm_precision"]),
        fmm_minimum_sites=int(nonbonded["fmm_minimum_atoms"]),
        reaction_field=reaction_field,
    )


@lru_cache(maxsize=16)
def _compile_vdw_confinement_record(payload: str) -> EllipsoidalVdwConfinement:
    return vdw_confinement_from_record(
        normalize_legacy_zaff_payload(json.loads(payload))
    )


def _model_vdw_confinement(
    model: Mapping[str, Any],
    *,
    cpcm_field: CPCMReactionField | None = None,
) -> EllipsoidalVdwConfinement | None:
    raw = model.get("vdw_confinement")
    if raw is None:
        return None
    calibration = model.get("vdw_calibration")
    if calibration is not None and str(
        dict(calibration).get("status", "")
    ) != "FROZEN_FOR_PRODUCTION":
        raise ValueError("ZAFF runtime refuses a non-frozen UvdW calibration")
    confinement = _compile_vdw_confinement_record(
        json.dumps(dict(raw), sort_keys=True, separators=(",", ":"))
    )
    if cpcm_field is not None:
        cpcm = cpcm_field.confinement
        if not (
            np.allclose(
                confinement.center_bohr, cpcm.center_bohr, rtol=0.0, atol=1.0e-10
            )
            and np.allclose(
                confinement.semiaxes_bohr,
                cpcm.semiaxes_bohr,
                rtol=0.0,
                atol=1.0e-10,
            )
            and np.allclose(
                confinement.rotation, cpcm.rotation, rtol=0.0, atol=1.0e-10
            )
        ):
            raise ValueError("CPCM and UvdW must share one confinement ellipsoid")
    return confinement



def evaluate_zaff_seed_model(
    model: Mapping[str, Any],
    coordinates_angstrom: np.ndarray,
    *,
    hessian: bool = False,
    zoom_level: str | None = None,
) -> ZaffSeedEvaluation:
    """Evaluate transferable ZAFF with analytic energy, gradient and Hessian."""

    fast = _fast_zoom_selected(zoom_level)
    energy, gradient, execution = _seed_energy_gradient(
        model, coordinates_angstrom, fast=fast
    )
    matrix = None
    if hessian:
        xyz = np.asarray(coordinates_angstrom, dtype=float)
        ncart = xyz.size
        matrix = np.zeros((ncart, ncart), dtype=float)
        interfacial_field = _model_interfacial_pcm_field(model)
        for column in range(ncart):
            direction = np.zeros(ncart)
            direction[column] = 1.0
            matrix[:, column] = _seed_hessian_vector_product(
                model,
                xyz,
                direction,
                fast=fast,
                include_fixed_reaction_field=interfacial_field is None,
            )
        if interfacial_field is not None:
            charges = np.asarray(
                [
                    float(item["atomic_charge"])
                    for item in model["nonbonded_atoms"]
                ]
            )
            coordinates_bohr = xyz / BOHR_TO_ANGSTROM
            batch_size = 32
            for start in range(0, ncart, batch_size):
                stop = min(ncart, start + batch_size)
                directions = np.zeros((stop - start, ncart), dtype=float)
                directions[np.arange(stop - start), np.arange(start, stop)] = 1.0
                products = interfacial_field.hessian_vector_products(
                    coordinates_bohr,
                    charges,
                    directions,
                )
                matrix[:, start:stop] += products.T
            execution = {
                **dict(execution),
                "interfacial_hessian_batch_size": batch_size,
            }
        matrix = 0.5 * (matrix + matrix.T)
    return ZaffSeedEvaluation(energy, gradient, matrix, execution)


def evaluate_zaff_fast_rigid_torsional_batch_energies(
    model: Mapping[str, Any],
    geometries_angstrom: Sequence[np.ndarray] | np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    """Vectorized energy-only hot loop for rigid-torsional population ranking."""

    geometries = np.asarray(geometries_angstrom, dtype=float)
    natoms = len(model["atoms"])
    if (
        geometries.ndim != 3
        or geometries.shape[1:] != (natoms, 3)
        or np.any(~np.isfinite(geometries))
    ):
        raise ValueError("fast ZAFF batch must have shape (ncandidate, natoms, 3)")
    block_size = max(1, int(chunk_size))
    if len(geometries) == 0:
        return np.zeros(0)
    pair_left, pair_right = np.triu_indices(natoms, k=1)
    excluded_vdw_pairs = _seed_excluded_vdw_pairs(model)
    vdw_pair_mask = np.asarray(
        [
            (int(left), int(right)) not in excluded_vdw_pairs
            for left, right in zip(pair_left, pair_right, strict=True)
        ],
        dtype=bool,
    )
    atom_parameters = tuple(model["nonbonded_atoms"])
    charges = np.asarray([float(item["atomic_charge"]) for item in atom_parameters])
    widths = np.asarray(
        [
            float(covalent_radius(int(atomic_number(symbol) or 0)) or 0.75)
            / BOHR_TO_ANGSTROM
            for symbol in model["atoms"]
        ]
    )
    pair_charge = charges[pair_left] * charges[pair_right]
    pair_beta = 1.0 / np.sqrt(
        2.0 * (widths[pair_left] ** 2 + widths[pair_right] ** 2)
    )
    epsilon = np.sqrt(
        np.asarray(
            [
                float(atom_parameters[index]["epsilon_hartree"])
                for index in pair_left
            ]
        )
        * np.asarray(
            [
                float(atom_parameters[index]["epsilon_hartree"])
                for index in pair_right
            ]
        )
    )
    r_min = np.asarray(
        [
            float(atom_parameters[left]["rmin_half_angstrom"])
            + float(atom_parameters[right]["rmin_half_angstrom"])
            for left, right in zip(pair_left, pair_right, strict=True)
        ]
    )
    alpha = np.asarray(
        [
            exppe_from_minimum(
                mie_minimum_derivatives(float(depth), float(radius), 12.0, 6.0)
            ).alpha
            for depth, radius in zip(epsilon, r_min, strict=True)
        ]
    )
    try:
        from scipy.special import erf as vector_erf
    except ImportError:  # pragma: no cover - MATRIX-SMITH normally supplies SciPy.
        vector_erf = np.vectorize(math.erf, otypes=[float])

    energies = np.zeros(len(geometries))
    cpcm_field = _model_cpcm_field(model)
    reaction_field = _model_reaction_field(model)
    vdw_confinement = _model_vdw_confinement(model, cpcm_field=cpcm_field)
    for start in range(0, len(geometries), block_size):
        stop = min(start + block_size, len(geometries))
        block = geometries[start:stop]
        delta = block[:, pair_left, :] - block[:, pair_right, :]
        distance_angstrom = np.linalg.norm(delta, axis=2)
        if np.any(distance_angstrom <= 1.0e-12):
            raise ValueError("coincident atoms in fast ZAFF batch")
        distance_bohr = distance_angstrom / BOHR_TO_ANGSTROM
        electrostatic = np.sum(
            pair_charge[None, :]
            * vector_erf(pair_beta[None, :] * distance_bohr)
            / distance_bohr,
            axis=1,
        )
        scaled = distance_angstrom / r_min[None, :]
        exp_full = np.exp(alpha[None, :] * (1.0 - scaled))
        exp_half = np.exp(0.5 * alpha[None, :] * (1.0 - scaled))
        polynomial = scaled**4 - 2.0 * scaled**2 + 3.0
        radial = epsilon[None, :] * (exp_full - polynomial * exp_half)
        if _seed_uses_continuous_exppe_weights(model):
            numbers = np.asarray(
                [int(atomic_number(symbol) or 0) for symbol in model["atoms"]],
                dtype=int,
            )
            topology_alpha = float(
                model["nonbonded"].get("topology_switch_alpha_per_angstrom", 12.0)
            )
            one_four_scale = float(model["nonbonded"].get("one_four_scale", 1.0))
            weights = np.asarray(
                [
                    [
                        continuous_topology_pair_scales(
                            geometry / BOHR_TO_ANGSTROM,
                            numbers,
                            one_four_scale=one_four_scale,
                            switch_alpha_per_angstrom=topology_alpha,
                        )[(int(left) + 1, int(right) + 1)]
                        for left, right in zip(pair_left, pair_right, strict=True)
                    ]
                    for geometry in block
                ],
                dtype=float,
            )
            collective = np.sum(weights * radial, axis=1)
        else:
            collective = np.sum(radial[:, vdw_pair_mask], axis=1)
        torsional = np.zeros(stop - start)
        for raw in model["torsions"]:
            i, j, k, ell = (int(value) for value in raw["atoms"])
            b1 = block[:, i, :] - block[:, j, :]
            b2 = block[:, k, :] - block[:, j, :]
            b3 = block[:, ell, :] - block[:, k, :]
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            b2_unit = b2 / np.linalg.norm(b2, axis=1)[:, None]
            phi = np.arctan2(
                np.einsum("bi,bi->b", np.cross(n1, n2), b2_unit),
                np.einsum("bi,bi->b", n1, n2),
            )
            for term in raw["terms"]:
                torsional += float(term["amplitude_hartree"]) * (
                    1.0
                    + np.cos(
                        int(term["periodicity"]) * phi
                        - float(term["phase_radian"])
                    )
                )
        reaction = (
            np.zeros(stop - start)
            if reaction_field is None
            else reaction_field.reaction_batch_energies(
                block / BOHR_TO_ANGSTROM, charges
            )
        )
        vdw = (
            np.zeros(stop - start)
            if vdw_confinement is None
            else vdw_confinement.batch_energies(
                block / BOHR_TO_ANGSTROM, model["atoms"]
            )
        )
        energies[start:stop] = (
            electrostatic + collective + torsional + reaction + vdw
        )
    return energies


def evaluate_zaff_seed_energy(
    model: Mapping[str, Any],
    coordinates_angstrom: np.ndarray,
    *,
    zoom_level: str | None = None,
    neighbor_list: PersistentVerletNeighborList | None = None,
    electrostatic_operator: PersistentGaussianElectrostaticOperator | None = None,
    charge_response_runtime: (
        PersistentSplitChargeResponse | PersistentAllPairSplitChargeResponse | None
    ) = None,
    runtime_policy: MMRuntimePolicy | None = None,
    morse_bond_arrays: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ] | None = None,
    gaussian_bond_arrays: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ] | None = None,
    local_valence_arrays: tuple[np.ndarray, ...] | None = None,
    charge_response_model: str | None = None,
    shared_valence_model: GFNFFValenceModel | None = None,
    shared_nci_model: GFNFFNCIModel | None = None,
    gfnff_eeq_session: GFNFFEEQSession | None = None,
) -> tuple[float, dict[str, Any]]:
    """Evaluate a general seed field without constructing any derivatives."""

    if model.get("schema") != ZAFF_SEED_MODEL_SCHEMA:
        raise ValueError("unsupported ZAFF seed model schema")
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.shape != (len(model["atoms"]), 3) or np.any(~np.isfinite(xyz)):
        raise ValueError("ZAFF seed evaluation geometry has wrong shape")
    fast = _fast_zoom_selected(zoom_level)
    response_model = _selected_charge_response_model(model, charge_response_model)
    if fast and response_model == "gfnff_eeq":
        raise ValueError("GFN-FF EEQ is not available in rigid-torsional ZAFF zoom")
    numerical = native_zaff_backend(len(xyz))
    energy = 0.0
    if not fast:
        morse_arrays = (
            morse_bond_arrays
            if morse_bond_arrays is not None
            else _seed_morse_bond_arrays(model)
        )
        gaussian_arrays = (
            gaussian_bond_arrays
            if gaussian_bond_arrays is not None
            else _seed_gaussian_bond_arrays(model)
        )
        if numerical.accelerated and len(morse_arrays[0]):
            bond_energy, _bond_count = _native_morse_bond_energy(
                xyz,
                *morse_arrays,
            )
            energy += bond_energy
        if numerical.accelerated and len(gaussian_arrays[0]):
            bond_energy, _bond_count = _native_gaussian_bond_energy(
                xyz,
                *gaussian_arrays,
            )
            energy += bond_energy
        dynamic_arrays = _seed_dynamic_gaussian_bond_arrays(
            model, xyz, neighbor_list
        )
        if dynamic_arrays is not None:
            if numerical.accelerated:
                bond_energy, _bond_gradient, _bond_count = (
                    _native_dynamic_gaussian_bond_energy_gradient(
                        xyz, *dynamic_arrays
                    )
                )
            else:
                bond_energy, _bond_gradient, _bond_product, _cn = (
                    GFNFFDynamicGaussianBondModel(*dynamic_arrays).evaluate(xyz)
                )
            energy += bond_energy
        for raw in _seed_inverse_distance_bond_records(model):
            distance = eval_primitive(
                Primitive("bond", tuple(int(value) for value in raw["atoms"])),
                xyz,
            )
            energy += _inverse_distance_bond_potential(raw).derivatives(distance).energy
        if not numerical.accelerated:
            for raw in model["bonds"]:
                if raw.get("reference_response") == "DYNAMIC_GFNFF_LOG_CN":
                    continue
                distance = eval_primitive(
                    Primitive("bond", tuple(int(value) for value in raw["atoms"])),
                    xyz,
                )
                functional_form = _bond_functional_form(raw)
                if functional_form in {"INVERSE_POWER_STRETCH", "SPF_INVERSE_DISTANCE"}:
                    continue
                if functional_form == "GFNFF_GAUSSIAN":
                    energy += _gaussian_bond_potential(raw).energy(distance)
                else:
                    exponent = math.exp(
                        -float(raw["alpha_per_angstrom"])
                        * (distance - float(raw["r0_angstrom"]))
                    )
                    energy += float(raw["d_e_hartree"]) * (1.0 - exponent) ** 2
        valence_arrays = (
            local_valence_arrays
            if local_valence_arrays is not None
            else _seed_local_valence_arrays(model)
        )
        if numerical.accelerated and (
            len(valence_arrays[0]) or len(valence_arrays[2])
        ):
            valence_energy, _valence_count = _native_local_valence_energy(
                xyz, *valence_arrays
            )
            energy += valence_energy
        for raw in (() if numerical.accelerated else model["angles"]):
            atoms3 = tuple(int(value) for value in raw["atoms"])
            primitives = (
                Primitive("bond", (atoms3[0], atoms3[1])),
                Primitive("bond", (atoms3[1], atoms3[2])),
                Primitive("angle", atoms3),
            )
            factors = tuple(
                BondOrderRadialFactor(
                    float(item["reference_distance_angstrom"]),
                    float(item["covalent_radius_sum_angstrom"]),
                )
                for item in raw["radial_factors"]
            )
            energy += BondOrderDampedAnglePotential(
                2.0 * float(raw["k_hartree_per_radian2"]),
                float(raw["theta0_radian"]),
                factors[0],
                factors[1],
            ).energy(*(eval_primitive(item, xyz) for item in primitives))
        for raw in (() if fast else _seed_out_of_plane_records(model)):
            primitives = _seed_out_of_plane_primitives(raw)
            potential = _seed_out_of_plane_potential(raw)
            energy += potential.energy(
                *(eval_primitive(item, xyz) for item in primitives)
            )
    for raw in (
        model["torsions"]
        if fast or not numerical.accelerated
        else ()
    ):
        atoms4 = tuple(int(value) for value in raw["atoms"])
        phi = eval_primitive(Primitive("dihedral", atoms4), xyz)
        terms = tuple(
            UFFFourierTerm(
                float(item["amplitude_hartree"]),
                int(item["periodicity"]),
                float(item["phase_radian"]),
            )
            for item in raw["terms"]
        )
        if fast:
            energy += sum(
                term.amplitude
                * (1.0 + math.cos(term.periodicity * phi - term.phase))
                for term in terms
            )
        else:
            bonds = (
                Primitive("bond", (atoms4[0], atoms4[1])),
                Primitive("bond", (atoms4[1], atoms4[2])),
                Primitive("bond", (atoms4[2], atoms4[3])),
            )
            factors = tuple(
                BondOrderRadialFactor(
                    float(item["reference_distance_angstrom"]),
                    float(item["covalent_radius_sum_angstrom"]),
                )
                for item in raw["radial_factors"]
            )
            energy += BondOrderDampedTorsionPotential(
                factors[0], factors[1], factors[2], terms
            ).energy(
                *(eval_primitive(item, xyz) for item in bonds),
                phi,
            )

    if not fast:
        shared_valence_model = shared_valence_model or _gfnff_special_torsions_from_model(model)
        shared_nci_model = shared_nci_model or _gfnff_nci_from_model(model)
    if not fast and shared_valence_model is not None:
        energy += shared_valence_model.evaluate(xyz / BOHR_TO_ANGSTROM).energy_hartree
    if not fast and shared_nci_model is not None:
        energy += shared_nci_model.evaluate(
            xyz / BOHR_TO_ANGSTROM
        ).energy_hartree

    nonbonded = dict(model["nonbonded"])
    policy = runtime_policy or select_mm_runtime_policy(
        xyz,
        cutoff_angstrom=float(nonbonded["short_range_cutoff_angstrom"]),
        neighbor_list_minimum_atoms=int(nonbonded["neighbor_list_minimum_atoms"]),
        fmm_minimum_atoms=int(nonbonded["fmm_minimum_atoms"]),
        fmm_precision=float(nonbonded["fmm_precision"]),
        require_linear_scaling=True,
    )
    atom_parameters = tuple(model["nonbonded_atoms"])
    charges = np.asarray([float(item["atomic_charge"]) for item in atom_parameters])
    coordinates_bohr = xyz / BOHR_TO_ANGSTROM
    cpcm_field = _model_cpcm_field(model)
    reaction_field = _model_reaction_field(model)
    operator = electrostatic_operator
    eeq_result = None
    if response_model == "gfnff_eeq":
        if reaction_field is not None:
            raise ValueError("GFN-FF EEQ in ZAFF does not yet support a reaction field")
        eeq_runtime = gfnff_eeq_session or GFNFFEEQSession.from_model(model)
        eeq_result = eeq_runtime.evaluate(coordinates_bohr, compute_gradient=False)
        energy += eeq_result.energy_hartree
    else:
        if operator is None:
            operator = PersistentGaussianElectrostaticOperator(
                charges,
                tuple(int(atomic_number(symbol) or 0) for symbol in model["atoms"]),
                backend=policy.electrostatic_backend,
                fmm_precision=float(nonbonded["fmm_precision"]),
                fmm_minimum_atoms=int(nonbonded["fmm_minimum_atoms"]),
            )
        energy += operator.energy(coordinates_bohr)
        if reaction_field is not None:
            energy += float(
                reaction_field.reaction_batch_energies(
                    coordinates_bohr[None, :, :], charges
                )[0]
            )
    response = None
    response_runtime = None
    if not fast and response_model == "variational_qeq_sqe":
        response_runtime = charge_response_runtime or (
            _persistent_charge_response_from_model(
                model,
                backend=policy.electrostatic_backend,
                reaction_field=reaction_field,
            )
        )
        response = response_runtime.solve(
            coordinates_bohr,
            compute_gradient=False,
        )
        energy += response.energy_correction_hartree
    vdw_confinement = _model_vdw_confinement(model, cpcm_field=cpcm_field)
    if vdw_confinement is not None:
        energy += float(
            vdw_confinement.batch_energies(
                coordinates_bohr[None, :, :], model["atoms"]
            )[0]
        )
    pairs = _seed_nonbonded_pairs(
        xyz,
        coordinates_bohr,
        policy.short_range_backend,
        float(nonbonded["short_range_cutoff_angstrom"]),
        neighbor_list,
        native=numerical.accelerated,
        excluded_pairs=_seed_excluded_vdw_pairs(model),
    )
    if numerical.accelerated:
        radial_energy, _pair_count = _native_exppe_energy(
            xyz,
            np.asarray(
                [float(item["epsilon_hartree"]) for item in atom_parameters]
            ),
            np.asarray(
                [float(item["rmin_half_angstrom"]) for item in atom_parameters]
            ),
            pairs,
            float(nonbonded["short_range_cutoff_angstrom"]),
        )
        energy += radial_energy
    else:
        for left, right in pairs:
            left_parameter = atom_parameters[left]
            right_parameter = atom_parameters[right]
            epsilon = math.sqrt(
                float(left_parameter["epsilon_hartree"])
                * float(right_parameter["epsilon_hartree"])
            )
            r_min = float(left_parameter["rmin_half_angstrom"]) + float(
                right_parameter["rmin_half_angstrom"]
            )
            distance = float(np.linalg.norm(xyz[left] - xyz[right]))
            radial = exppe_from_minimum(
                mie_minimum_derivatives(epsilon, r_min, 12.0, 6.0)
            ).energy(distance)
            energy += radial
    exppe_correction, _gradient, _product, exppe_correction_count = (
        _seed_continuous_exppe_correction(
            model,
            coordinates_bohr,
            cutoff_angstrom=float(nonbonded["short_range_cutoff_angstrom"]),
        )
    )
    morse_correction, _gradient, _product, morse_partition_count = (
        _seed_morse_partition_correction(model, xyz)
        if not fast
        else (0.0, np.zeros(xyz.size), None, 0)
    )
    energy += exppe_correction + morse_correction
    return float(energy), {
        "backend": "zaff_seed_energy_only",
        "exppe_partition": (
            "continuous_erf_paths" if _seed_uses_continuous_exppe_weights(model)
            else "discrete_graph"
        ),
        "exppe_partition_correction_count": exppe_correction_count,
        "morse_partition_count": morse_partition_count,
        "mm_runtime": policy.to_dict(),
        "electrostatic_backend": (
            eeq_result.backend if eeq_result is not None else operator.selected_backend
        ),
        "cpcm_backend": (
            "DISABLED" if cpcm_field is None else cpcm_field.modal_backend
        ),
        "reaction_field_backend": (
            "DISABLED"
            if reaction_field is None
            else reaction_field.modal_backend
        ),
        "vdw_confinement_backend": (
            "DISABLED"
            if vdw_confinement is None
            else "ELLIPSOIDAL_HOMOTHETIC_POLYNOMIAL"
        ),
        "polarization_iterations": (
            eeq_result.iterations
            if eeq_result is not None
            else 0 if response is None else response.iterations
        ),
        "polarization_channel_count": 0 if response is None else response.channel_count,
        "linear_scaling_certified": bool(
            policy.linear_scaling_certified and reaction_field is None
        ),
        "linear_scaling_blockers": (
            () if reaction_field is None else ("reaction_field",)
        ),
        "coordinate_transform_backend": "NONE_ENERGY_ONLY",
        "zaff_gfnff_components": tuple(sorted(_zaff_gfnff_components(model))),
    }


def _seed_energy_gradient(
    model: Mapping[str, Any],
    coordinates_angstrom: np.ndarray,
    *,
    fast: bool = False,
    neighbor_list: PersistentVerletNeighborList | None = None,
    electrostatic_operator: PersistentGaussianElectrostaticOperator | None = None,
    charge_response_runtime: (
        PersistentSplitChargeResponse | PersistentAllPairSplitChargeResponse | None
    ) = None,
    runtime_policy: MMRuntimePolicy | None = None,
    morse_bond_arrays: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ] | None = None,
    gaussian_bond_arrays: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ] | None = None,
    local_valence_arrays: tuple[np.ndarray, ...] | None = None,
    charge_response_model: str | None = None,
    water_charge_response_evaluator: WaterChargeResponseEvaluator | None = None,
    hbond_pair_list: HydrogenBondPairList | None = None,
    shared_valence_model: GFNFFValenceModel | None = None,
    shared_nci_model: GFNFFNCIModel | None = None,
    gfnff_eeq_session: GFNFFEEQSession | None = None,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    if model.get("schema") != ZAFF_SEED_MODEL_SCHEMA:
        raise ValueError("unsupported ZAFF seed model schema")
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.shape != (len(model["atoms"]), 3) or np.any(~np.isfinite(xyz)):
        raise ValueError("ZAFF seed evaluation geometry has wrong shape")
    energy = 0.0
    gradient_angstrom = np.zeros(xyz.size, dtype=float)
    primitives: list[Primitive] = []
    coordinate_gradients: list[float] = []
    numerical = native_zaff_backend(len(xyz))
    if not fast:
        morse_arrays = (
            morse_bond_arrays
            if morse_bond_arrays is not None
            else _seed_morse_bond_arrays(model)
        )
        gaussian_arrays = (
            gaussian_bond_arrays
            if gaussian_bond_arrays is not None
            else _seed_gaussian_bond_arrays(model)
        )
        if numerical.accelerated and len(morse_arrays[0]):
            bond_energy, bond_gradient, _bond_count = (
                _native_morse_bond_energy_gradient(
                    xyz,
                    *morse_arrays,
                )
            )
            energy += bond_energy
            gradient_angstrom += (
                bond_gradient.reshape(-1) / BOHR_TO_ANGSTROM
            )
        if numerical.accelerated and len(gaussian_arrays[0]):
            bond_energy, bond_gradient, _bond_count = (
                _native_gaussian_bond_energy_gradient(
                    xyz,
                    *gaussian_arrays,
                )
            )
            energy += bond_energy
            gradient_angstrom += (
                bond_gradient.reshape(-1) / BOHR_TO_ANGSTROM
            )
        dynamic_arrays = _seed_dynamic_gaussian_bond_arrays(
            model, xyz, neighbor_list
        )
        if dynamic_arrays is not None:
            if numerical.accelerated:
                bond_energy, bond_gradient, _bond_count = (
                    _native_dynamic_gaussian_bond_energy_gradient(
                        xyz, *dynamic_arrays
                    )
                )
                bond_gradient_angstrom = bond_gradient / BOHR_TO_ANGSTROM
            else:
                bond_energy, bond_gradient_angstrom, _bond_product, _cn = (
                    GFNFFDynamicGaussianBondModel(*dynamic_arrays).evaluate(xyz)
                )
            energy += bond_energy
            gradient_angstrom += bond_gradient_angstrom.reshape(-1)
        for raw in _seed_inverse_distance_bond_records(model):
            primitive = Primitive(
                "bond", tuple(int(value) for value in raw["atoms"])
            )
            radial = _inverse_distance_bond_potential(raw).derivatives(
                eval_primitive(primitive, xyz)
            )
            energy += radial.energy
            coordinate_gradients.append(radial.first)
            primitives.append(primitive)
        if not numerical.accelerated:
            for raw in model["bonds"]:
                if raw.get("reference_response") == "DYNAMIC_GFNFF_LOG_CN":
                    continue
                primitive = Primitive(
                    "bond", tuple(int(value) for value in raw["atoms"])
                )
                distance = eval_primitive(primitive, xyz)
                functional_form = _bond_functional_form(raw)
                if functional_form in {"INVERSE_POWER_STRETCH", "SPF_INVERSE_DISTANCE"}:
                    continue
                if functional_form == "GFNFF_GAUSSIAN":
                    radial = _gaussian_bond_potential(raw).derivatives(distance)
                    energy += radial.value
                    coordinate_gradients.append(radial.first)
                else:
                    exponent = math.exp(
                        -float(raw["alpha_per_angstrom"])
                        * (distance - float(raw["r0_angstrom"]))
                    )
                    depth = float(raw["d_e_hartree"])
                    alpha = float(raw["alpha_per_angstrom"])
                    energy += depth * (1.0 - exponent) ** 2
                    coordinate_gradients.append(
                        2.0 * depth * alpha * exponent * (1.0 - exponent)
                    )
                primitives.append(primitive)
    valence_arrays = (
        local_valence_arrays
        if local_valence_arrays is not None
        else _seed_local_valence_arrays(model)
    )
    if (
        not fast
        and numerical.accelerated
        and (len(valence_arrays[0]) or len(valence_arrays[2]))
    ):
        valence_energy, valence_gradient, _valence_count = (
            _native_local_valence_energy_gradient(xyz, *valence_arrays)
        )
        energy += valence_energy
        gradient_angstrom += valence_gradient.reshape(-1) / BOHR_TO_ANGSTROM
    for raw in (
        ()
        if fast or numerical.accelerated
        else model["angles"]
    ):
        atoms3 = tuple(int(value) for value in raw["atoms"])
        primitive = Primitive("angle", atoms3)
        left_bond = Primitive("bond", (atoms3[0], atoms3[1]))
        right_bond = Primitive("bond", (atoms3[1], atoms3[2]))
        factors = tuple(
            BondOrderRadialFactor(
                float(item["reference_distance_angstrom"]),
                float(item["covalent_radius_sum_angstrom"]),
            )
            for item in raw["radial_factors"]
        )
        potential = BondOrderDampedAnglePotential(
            2.0 * float(raw["k_hartree_per_radian2"]),
            float(raw["theta0_radian"]),
            factors[0],
            factors[1],
        )
        internal = potential.derivatives(
            eval_primitive(left_bond, xyz),
            eval_primitive(right_bond, xyz),
            eval_primitive(primitive, xyz),
        )
        energy += internal.energy
        primitives.extend((left_bond, right_bond, primitive))
        coordinate_gradients.extend(float(value) for value in internal.gradient)
    for raw in (() if fast else _seed_out_of_plane_records(model)):
        primitives_soop = _seed_out_of_plane_primitives(raw)
        potential = _seed_out_of_plane_potential(raw)
        internal = potential.derivatives(
            *(eval_primitive(item, xyz) for item in primitives_soop)
        )
        energy += internal.energy
        primitives.extend(primitives_soop)
        coordinate_gradients.extend(float(value) for value in internal.gradient)
    for raw in (
        model["torsions"]
        if fast or not numerical.accelerated
        else ()
    ):
        atoms4 = tuple(int(value) for value in raw["atoms"])
        primitive = Primitive("dihedral", atoms4)
        torsion_bonds = (
            Primitive("bond", (atoms4[0], atoms4[1])),
            Primitive("bond", (atoms4[1], atoms4[2])),
            Primitive("bond", (atoms4[2], atoms4[3])),
        )
        terms = tuple(
            UFFFourierTerm(
                float(item["amplitude_hartree"]),
                int(item["periodicity"]),
                float(item["phase_radian"]),
            )
            for item in raw["terms"]
        )
        if fast:
            phi = eval_primitive(primitive, xyz)
            value = first = 0.0
            for term in terms:
                argument = term.periodicity * phi - term.phase
                value += term.amplitude * (1.0 + math.cos(argument))
                first -= (
                    term.amplitude * term.periodicity * math.sin(argument)
                )
            energy += value
            primitives.append(primitive)
            coordinate_gradients.append(first)
            continue
        factors = tuple(
            BondOrderRadialFactor(
                float(item["reference_distance_angstrom"]),
                float(item["covalent_radius_sum_angstrom"]),
            )
            for item in raw["radial_factors"]
        )
        internal = BondOrderDampedTorsionPotential(
            factors[0], factors[1], factors[2], terms
        ).derivatives(
            *(eval_primitive(bond, xyz) for bond in torsion_bonds),
            eval_primitive(primitive, xyz),
        )
        energy += internal.energy
        primitives.extend((*torsion_bonds, primitive))
        coordinate_gradients.extend(float(value) for value in internal.gradient)
    if primitives:
        gradient_angstrom += primitive_b_matrix(primitives, xyz).T @ np.asarray(
            coordinate_gradients
        )

    nonbonded = dict(model["nonbonded"])
    policy = runtime_policy or select_mm_runtime_policy(
        xyz,
        cutoff_angstrom=float(nonbonded["short_range_cutoff_angstrom"]),
        neighbor_list_minimum_atoms=int(nonbonded["neighbor_list_minimum_atoms"]),
        fmm_minimum_atoms=int(nonbonded["fmm_minimum_atoms"]),
        fmm_precision=float(nonbonded["fmm_precision"]),
        require_linear_scaling=True,
    )
    atom_parameters = tuple(model["nonbonded_atoms"])
    charges = np.asarray([float(item["atomic_charge"]) for item in atom_parameters])
    gradient_bohr = gradient_angstrom * BOHR_TO_ANGSTROM
    if not fast:
        shared_valence_model = shared_valence_model or _gfnff_special_torsions_from_model(model)
        shared_nci_model = shared_nci_model or _gfnff_nci_from_model(model)
    if not fast and shared_valence_model is not None:
        shared_valence = shared_valence_model.evaluate(xyz / BOHR_TO_ANGSTROM)
        energy += shared_valence.energy_hartree
        gradient_bohr += shared_valence.gradient_hartree_per_bohr
    if not fast and shared_nci_model is not None:
        shared_nci = shared_nci_model.evaluate(xyz / BOHR_TO_ANGSTROM)
        energy += shared_nci.energy_hartree
        gradient_bohr += shared_nci.gradient_hartree_per_bohr
    cpcm_field = _model_cpcm_field(model)
    reaction_field = _model_reaction_field(model)
    response_model = _selected_charge_response_model(model, charge_response_model)
    local_response = None
    eeq_result = None
    if response_model == "water_hbond_surrogate":
        if water_charge_response_evaluator is None:
            raise RuntimeError(
                "water_hbond_surrogate requires an explicit perception callback"
            )
        waters = _water_groups_from_seed_model(model)
        local_response = water_charge_response_evaluator(
            xyz / BOHR_TO_ANGSTROM,
            waters,
            reference_charges_e=charges,
            recognition=hbond_pair_list,
        )
        widths = np.asarray(
            [
                float(item["gaussian_width_bohr"])
                for item in model["charge_response"]["sites"]
            ]
        )
        local_electrostatic = evaluate_local_charge_electrostatics(
            xyz / BOHR_TO_ANGSTROM,
            widths,
            local_response,
            backend=policy.electrostatic_backend,
            fmm_precision=float(nonbonded["fmm_precision"]),
            fmm_minimum_sites=int(nonbonded["fmm_minimum_atoms"]),
            reaction_field=reaction_field,
        )
        energy += local_electrostatic.energy_hartree
        gradient_bohr += local_electrostatic.gradient_hartree_per_bohr
        electrostatic_backend = local_electrostatic.backend
        pair_correction_count = 0
    elif response_model == "variational_qeq_sqe":
        electrostatic = (
            electrostatic_energy_gradient(
                xyz / BOHR_TO_ANGSTROM,
                charges,
                (),
                atomic_numbers=tuple(
                    int(atomic_number(symbol) or 0) for symbol in model["atoms"]
                ),
                electrostatic_model="gaussian_erf_all_pairs",
                backend="auto",
                fmm_precision=float(nonbonded["fmm_precision"]),
                fmm_minimum_atoms=int(nonbonded["fmm_minimum_atoms"]),
            )
            if electrostatic_operator is None
            else electrostatic_operator.evaluate(xyz / BOHR_TO_ANGSTROM)
        )
        energy += electrostatic.energy_hartree
        gradient_bohr += electrostatic.gradient_hartree_per_bohr.reshape(-1)
        electrostatic_backend = electrostatic.backend
        pair_correction_count = electrostatic.pair_correction_count
        if reaction_field is not None:
            reaction_energy, reaction_gradient = (
                reaction_field.reaction_energy_gradient(
                    xyz / BOHR_TO_ANGSTROM, charges
                )
            )
            energy += reaction_energy
            gradient_bohr += reaction_gradient.reshape(-1)
    elif response_model == "gfnff_eeq":
        if fast:
            raise ValueError("GFN-FF EEQ is not available in rigid-torsional ZAFF zoom")
        if reaction_field is not None:
            raise ValueError("GFN-FF EEQ in ZAFF does not yet support a reaction field")
        eeq_runtime = gfnff_eeq_session or GFNFFEEQSession.from_model(model)
        eeq_result = eeq_runtime.evaluate(xyz / BOHR_TO_ANGSTROM)
        energy += eeq_result.energy_hartree
        gradient_bohr += eeq_result.gradient_hartree_per_bohr
        electrostatic_backend = eeq_result.backend
        pair_correction_count = 0
    else:
        raise ValueError("unsupported ZAFF charge-response runtime model")
    vdw_confinement = _model_vdw_confinement(model, cpcm_field=cpcm_field)
    if vdw_confinement is not None:
        vdw_result = vdw_confinement.evaluate(
            xyz / BOHR_TO_ANGSTROM, model["atoms"]
        )
        energy += vdw_result.energy_hartree
        gradient_bohr += vdw_result.gradient_hartree_per_bohr
    response = None
    response_runtime = None
    if not fast and response_model == "variational_qeq_sqe":
        response_runtime = charge_response_runtime or (
            _persistent_charge_response_from_model(
                model,
                backend=policy.electrostatic_backend,
                reaction_field=reaction_field,
            )
        )
        response = response_runtime.solve(xyz / BOHR_TO_ANGSTROM)
        energy += response.energy_correction_hartree
        gradient_bohr += response.gradient_correction_hartree_per_bohr

    coordinates_bohr = xyz / BOHR_TO_ANGSTROM
    pairs = _seed_nonbonded_pairs(
        xyz,
        coordinates_bohr,
        policy.short_range_backend,
        float(nonbonded["short_range_cutoff_angstrom"]),
        neighbor_list,
        native=numerical.accelerated,
        excluded_pairs=_seed_excluded_vdw_pairs(model),
    )
    if numerical.accelerated:
        radial_energy, radial_gradient, _pair_count = (
            _native_exppe_energy_gradient(
                xyz,
                np.asarray(
                    [float(item["epsilon_hartree"]) for item in atom_parameters]
                ),
                np.asarray(
                    [
                        float(item["rmin_half_angstrom"])
                        for item in atom_parameters
                    ]
                ),
                pairs,
                float(nonbonded["short_range_cutoff_angstrom"]),
            )
        )
        energy += radial_energy
        gradient_bohr += radial_gradient.reshape(-1)
    else:
        for left, right in pairs:
            left_parameter, right_parameter = (
                atom_parameters[left],
                atom_parameters[right],
            )
            epsilon = math.sqrt(
                float(left_parameter["epsilon_hartree"])
                * float(right_parameter["epsilon_hartree"])
            )
            r_min = float(left_parameter["rmin_half_angstrom"]) + float(
                right_parameter["rmin_half_angstrom"]
            )
            potential = exppe_from_minimum(
                mie_minimum_derivatives(epsilon, r_min, 12.0, 6.0)
            )
            delta = xyz[left] - xyz[right]
            distance = float(np.linalg.norm(delta))
            radial = potential.derivatives(distance)
            energy += radial.energy
            pair_gradient_bohr = radial.first * BOHR_TO_ANGSTROM * delta / distance
            gradient_bohr[3 * left : 3 * left + 3] += pair_gradient_bohr
            gradient_bohr[3 * right : 3 * right + 3] -= pair_gradient_bohr
    exppe_correction, exppe_gradient, _product, exppe_correction_count = (
        _seed_continuous_exppe_correction(
            model,
            coordinates_bohr,
            cutoff_angstrom=float(nonbonded["short_range_cutoff_angstrom"]),
        )
    )
    morse_correction, morse_gradient, _product, morse_partition_count = (
        _seed_morse_partition_correction(model, xyz)
        if not fast
        else (0.0, np.zeros(xyz.size), None, 0)
    )
    energy += exppe_correction + morse_correction
    gradient_bohr += exppe_gradient.reshape(-1) + morse_gradient.reshape(-1)
    fast_terms = (
        (
            "electrostatics",
            "exppe_nonbonded",
            "torsions",
            "reaction_field",
            "vdw_confinement",
        )
        if fast
        else (
            "covalent_bonds",
            "angles",
            "torsions",
            "electrostatics",
            "charge_response_polarization",
            "reaction_field",
            "exppe_nonbonded",
            "vdw_confinement",
        )
    )
    omitted_terms = (
        ("covalent_bonds", "angles", "charge_response_polarization")
        if fast
        else ()
    )
    if not fast and response_model == "gfnff_eeq":
        fast_terms = tuple(
            term
            for term in fast_terms
            if term not in {"electrostatics", "charge_response_polarization"}
        ) + ("gfnff_eeq",)
    if not fast and shared_nci_model is not None:
        fast_terms = (*fast_terms, "shared_gfnff_h_x_bonds")
    if not fast and shared_valence_model is not None:
        fast_terms = (*fast_terms, "shared_gfnff_special_torsion")
    return float(energy), gradient_bohr, {
        "backend": (
            "zaff_fast_rigid_torsional"
            if fast
            else "zaff_diagonal_seed"
        ),
        "mm_runtime": policy.to_dict(),
        "electrostatic_backend": electrostatic_backend,
        "polarization_backend": (
            "DISABLED_BY_FAST_RIGID_TORSIONAL_CONTRACT"
            if fast
            else (
                "MATRIX_SHARED_LOCAL_HBOND_RESPONSE"
                if local_response is not None
                else (
                    eeq_result.backend
                    if eeq_result is not None
                    else response.backend
                )
            )
        ),
        "cpcm_backend": (
            "DISABLED"
            if cpcm_field is None
            else cpcm_field.modal_backend
        ),
        "reaction_field_backend": (
            "DISABLED"
            if reaction_field is None
            else reaction_field.modal_backend
        ),
        "vdw_confinement_backend": (
            "DISABLED"
            if vdw_confinement is None
            else "ELLIPSOIDAL_HOMOTHETIC_POLYNOMIAL"
        ),
        "polarization_iterations": (
            eeq_result.iterations
            if eeq_result is not None
            else 0 if response is None else response.iterations
        ),
        "polarization_residual_norm": (
            eeq_result.residual_norm
            if eeq_result is not None
            else 0.0 if response is None else response.residual_norm
        ),
        "polarization_channel_count": 0 if response is None else response.channel_count,
        "linear_scaling_certified": bool(
            policy.linear_scaling_certified and reaction_field is None
        ),
        "linear_scaling_blockers": (
            () if reaction_field is None else ("reaction_field",)
        ),
        "coordinate_transform_backend": (
            "NONE_DIRECT_CARTESIAN_DIHEDRALS"
            if fast
            else "MATRIX_PRIMITIVE_ANALYTIC"
        ),
        "pair_correction_count": pair_correction_count,
        "exppe_partition": (
            "continuous_erf_paths"
            if _seed_uses_continuous_exppe_weights(model)
            else "discrete_graph"
        ),
        "exppe_partition_correction_count": exppe_correction_count,
        "morse_partition_count": morse_partition_count,
        "included_terms": fast_terms,
        "omitted_terms": omitted_terms,
        "gradient_backend": "ANALYTIC_CARTESIAN",
        "zaff_gfnff_components": tuple(sorted(_zaff_gfnff_components(model))),
    }


def _water_groups_from_seed_model(
    model: Mapping[str, Any],
) -> tuple[tuple[int, int, int], ...]:
    """Return O-H-H groups without assuming contiguous atom ordering."""

    atoms = tuple(str(value).strip().upper() for value in model["atoms"])
    if any(atom not in {"O", "H"} for atom in atoms):
        raise ValueError(
            "the closed-form water response currently requires an all-water system"
        )
    adjacency: dict[int, list[int]] = {index: [] for index in range(len(atoms))}
    for item in model["bonds"]:
        left, right = (int(value) for value in item["atoms"])
        adjacency[left].append(right)
        adjacency[right].append(left)
    waters: list[tuple[int, int, int]] = []
    assigned: set[int] = set()
    for oxygen, atom in enumerate(atoms):
        if atom != "O":
            continue
        hydrogens = sorted(
            neighbor
            for neighbor in adjacency[oxygen]
            if atoms[neighbor] == "H"
        )
        if len(hydrogens) != 2:
            raise ValueError("each surrogate water oxygen must have exactly two hydrogens")
        group = (oxygen, hydrogens[0], hydrogens[1])
        if any(index in assigned for index in group):
            raise ValueError("surrogate water groups overlap")
        assigned.update(group)
        waters.append(group)
    if len(assigned) != len(atoms):
        raise ValueError("every atom must belong to one surrogate water group")
    return tuple(waters)


def _seed_hessian_vector_product(
    model: Mapping[str, Any],
    coordinates_angstrom: np.ndarray,
    vector_bohr: np.ndarray,
    *,
    fast: bool = False,
    neighbor_list: PersistentVerletNeighborList | None = None,
    electrostatic_operator: PersistentGaussianElectrostaticOperator | None = None,
    charge_response_runtime: (
        PersistentSplitChargeResponse | PersistentAllPairSplitChargeResponse | None
    ) = None,
    runtime_policy: MMRuntimePolicy | None = None,
    morse_bond_arrays: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ] | None = None,
    gaussian_bond_arrays: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ] | None = None,
    local_valence_arrays: tuple[np.ndarray, ...] | None = None,
    include_fixed_reaction_field: bool = True,
    charge_response_model: str | None = None,
    shared_valence_model: GFNFFValenceModel | None = None,
    shared_nci_model: GFNFFNCIModel | None = None,
    gfnff_eeq_session: GFNFFEEQSession | None = None,
) -> np.ndarray:
    """Apply the exact analytic seed Hessian without numerical differentiation."""

    xyz = np.asarray(coordinates_angstrom, dtype=float)
    direction = np.asarray(vector_bohr, dtype=float).reshape(-1)
    if xyz.shape != (len(model["atoms"]), 3) or direction.size != xyz.size:
        raise ValueError("ZAFF seed Hessian-vector dimensions are inconsistent")
    product = np.zeros(xyz.size)
    numerical = native_zaff_backend(len(xyz))
    if not fast:
        morse_arrays = (
            morse_bond_arrays
            if morse_bond_arrays is not None
            else _seed_morse_bond_arrays(model)
        )
        gaussian_arrays = (
            gaussian_bond_arrays
            if gaussian_bond_arrays is not None
            else _seed_gaussian_bond_arrays(model)
        )
        if numerical.accelerated and len(morse_arrays[0]):
            bond_product, _bond_count = _native_morse_bond_hessian_vector(
                xyz,
                *morse_arrays,
                direction.reshape(xyz.shape),
            )
            product += bond_product.reshape(-1)
        if numerical.accelerated and len(gaussian_arrays[0]):
            bond_product, _bond_count = _native_gaussian_bond_hessian_vector(
                xyz,
                *gaussian_arrays,
                direction.reshape(xyz.shape),
            )
            product += bond_product.reshape(-1)
        dynamic_arrays = _seed_dynamic_gaussian_bond_arrays(
            model, xyz, neighbor_list
        )
        if dynamic_arrays is not None:
            if numerical.accelerated:
                bond_product, _bond_count = (
                    _native_dynamic_gaussian_bond_hessian_vector(
                        xyz,
                        *dynamic_arrays[:7],
                        direction.reshape(xyz.shape),
                        *dynamic_arrays[7:],
                    )
                )
            else:
                _energy, _gradient, product_angstrom, _cn = (
                    GFNFFDynamicGaussianBondModel(*dynamic_arrays).evaluate(
                        xyz, direction.reshape(xyz.shape)
                    )
                )
                assert product_angstrom is not None
                bond_product = product_angstrom * BOHR_TO_ANGSTROM**2
            product += bond_product.reshape(-1)
        for raw in _seed_inverse_distance_bond_records(model):
            primitive = Primitive(
                "bond", tuple(int(value) for value in raw["atoms"])
            )
            radial = _inverse_distance_bond_potential(raw).derivatives(
                eval_primitive(primitive, xyz)
            )
            product += _internal_term_hessian_vector_product(
                (primitive,),
                np.asarray([radial.first]),
                np.asarray([[radial.second]]),
                xyz,
                direction,
            )
        if not numerical.accelerated:
            for raw in model["bonds"]:
                if raw.get("reference_response") == "DYNAMIC_GFNFF_LOG_CN":
                    continue
                primitive = Primitive(
                    "bond", tuple(int(value) for value in raw["atoms"])
                )
                distance = eval_primitive(primitive, xyz)
                functional_form = _bond_functional_form(raw)
                if functional_form in {"INVERSE_POWER_STRETCH", "SPF_INVERSE_DISTANCE"}:
                    continue
                if functional_form == "GFNFF_GAUSSIAN":
                    radial = _gaussian_bond_potential(raw).derivatives(distance)
                    gradient_internal = np.asarray([radial.first])
                    hessian_internal = np.asarray([[radial.second]])
                else:
                    alpha = float(raw["alpha_per_angstrom"])
                    depth = float(raw["d_e_hartree"])
                    exponent = math.exp(
                        -alpha * (distance - float(raw["r0_angstrom"]))
                    )
                    gradient_internal = np.asarray(
                        [2.0 * depth * alpha * exponent * (1.0 - exponent)]
                    )
                    hessian_internal = np.asarray(
                        [
                            [
                                2.0
                                * depth
                                * alpha**2
                                * (2.0 * exponent**2 - exponent)
                            ]
                        ]
                    )
                product += _internal_term_hessian_vector_product(
                    (primitive,),
                    gradient_internal,
                    hessian_internal,
                    xyz,
                    direction,
                )
    valence_arrays = (
        local_valence_arrays
        if local_valence_arrays is not None
        else _seed_local_valence_arrays(model)
    )
    if (
        not fast
        and numerical.accelerated
        and (len(valence_arrays[0]) or len(valence_arrays[2]))
    ):
        valence_product, _valence_count = _native_local_valence_hessian_vector(
            xyz, *valence_arrays, direction.reshape(xyz.shape)
        )
        product += valence_product.reshape(-1)
    for raw in (
        ()
        if fast or numerical.accelerated
        else model["angles"]
    ):
        atoms3 = tuple(int(value) for value in raw["atoms"])
        primitives = (
            Primitive("bond", (atoms3[0], atoms3[1])),
            Primitive("bond", (atoms3[1], atoms3[2])),
            Primitive("angle", atoms3),
        )
        factors = tuple(
            BondOrderRadialFactor(
                float(item["reference_distance_angstrom"]),
                float(item["covalent_radius_sum_angstrom"]),
            )
            for item in raw["radial_factors"]
        )
        internal = BondOrderDampedAnglePotential(
            2.0 * float(raw["k_hartree_per_radian2"]),
            float(raw["theta0_radian"]),
            factors[0],
            factors[1],
        ).derivatives(*(eval_primitive(item, xyz) for item in primitives))
        product += _internal_term_hessian_vector_product(
            primitives,
            internal.gradient,
            internal.hessian,
            xyz,
            direction,
        )
    for raw in (() if fast else _seed_out_of_plane_records(model)):
        primitives_soop = _seed_out_of_plane_primitives(raw)
        potential = _seed_out_of_plane_potential(raw)
        internal = potential.derivatives(
            *(eval_primitive(item, xyz) for item in primitives_soop)
        )
        product += _internal_term_hessian_vector_product(
            primitives_soop,
            internal.gradient,
            internal.hessian,
            xyz,
            direction,
        )
    for raw in (
        model["torsions"]
        if fast or not numerical.accelerated
        else ()
    ):
        atoms4 = tuple(int(value) for value in raw["atoms"])
        if fast:
            primitive = Primitive("dihedral", atoms4)
            phi = eval_primitive(primitive, xyz)
            first = second = 0.0
            for item in raw["terms"]:
                amplitude = float(item["amplitude_hartree"])
                periodicity = int(item["periodicity"])
                argument = periodicity * phi - float(item["phase_radian"])
                first -= amplitude * periodicity * math.sin(argument)
                second -= amplitude * periodicity**2 * math.cos(argument)
            product += _internal_term_hessian_vector_product(
                (primitive,),
                np.asarray([first]),
                np.asarray([[second]]),
                xyz,
                direction,
            )
            continue
        primitives = (
            Primitive("bond", (atoms4[0], atoms4[1])),
            Primitive("bond", (atoms4[1], atoms4[2])),
            Primitive("bond", (atoms4[2], atoms4[3])),
            Primitive("dihedral", atoms4),
        )
        factors = tuple(
            BondOrderRadialFactor(
                float(item["reference_distance_angstrom"]),
                float(item["covalent_radius_sum_angstrom"]),
            )
            for item in raw["radial_factors"]
        )
        internal = BondOrderDampedTorsionPotential(
            factors[0],
            factors[1],
            factors[2],
            tuple(
                UFFFourierTerm(
                    float(item["amplitude_hartree"]),
                    int(item["periodicity"]),
                    float(item["phase_radian"]),
                )
                for item in raw["terms"]
            ),
        ).derivatives(*(eval_primitive(item, xyz) for item in primitives))
        product += _internal_term_hessian_vector_product(
            primitives,
            internal.gradient,
            internal.hessian,
            xyz,
            direction,
        )

    if not fast:
        shared_valence_model = shared_valence_model or _gfnff_special_torsions_from_model(model)
        shared_nci_model = shared_nci_model or _gfnff_nci_from_model(model)
    if not fast and shared_valence_model is not None:
        product += shared_valence_model.hessian_vector_product(
            xyz / BOHR_TO_ANGSTROM,
            direction.reshape(xyz.shape),
        )
    if not fast and shared_nci_model is not None:
        product += shared_nci_model.hessian_vector_product(
            xyz / BOHR_TO_ANGSTROM,
            direction.reshape(xyz.shape),
        )

    nonbonded = dict(model["nonbonded"])
    atom_parameters = tuple(model["nonbonded_atoms"])
    charges = np.asarray([float(item["atomic_charge"]) for item in atom_parameters])
    coordinates_bohr = xyz / BOHR_TO_ANGSTROM
    cpcm_field = _model_cpcm_field(model)
    reaction_field = _model_reaction_field(model)
    response_model = _selected_charge_response_model(model, charge_response_model)
    if response_model == "gfnff_eeq":
        if fast:
            raise ValueError("GFN-FF EEQ is not available in rigid-torsional ZAFF zoom")
        if reaction_field is not None:
            raise ValueError("GFN-FF EEQ in ZAFF does not yet support a reaction field")
        eeq_runtime = gfnff_eeq_session or GFNFFEEQSession.from_model(model)
        product += eeq_runtime.hessian_vector_product(
            coordinates_bohr, direction.reshape(xyz.shape)
        )
    else:
        if electrostatic_operator is None:
            product += electrostatic_hessian_vector_product(
                coordinates_bohr,
                charges,
                direction,
                (),
                atomic_numbers=tuple(
                    int(atomic_number(symbol) or 0) for symbol in model["atoms"]
                ),
                electrostatic_model="gaussian_erf_all_pairs",
                backend="auto",
                fmm_precision=float(nonbonded["fmm_precision"]),
                fmm_minimum_atoms=int(nonbonded["fmm_minimum_atoms"]),
            )
        else:
            product += electrostatic_operator.hessian_vector_product(
                coordinates_bohr,
                direction,
            )
        if reaction_field is not None and include_fixed_reaction_field:
            product += reaction_field.hessian_vector_product(
                coordinates_bohr, charges, direction
            )
    vdw_confinement = _model_vdw_confinement(model, cpcm_field=cpcm_field)
    if vdw_confinement is not None:
        product += vdw_confinement.hessian_vector_product(
            coordinates_bohr, model["atoms"], direction
        )
    if not fast and response_model == "variational_qeq_sqe":
        if charge_response_runtime is None:
            response_runtime = _persistent_charge_response_from_model(
                model,
                backend="auto",
                reaction_field=reaction_field,
            )
            product += response_runtime.hessian_vector_product(
                coordinates_bohr,
                direction,
            )
        else:
            product += charge_response_runtime.hessian_vector_product(
                coordinates_bohr,
                direction,
            )

    policy = runtime_policy or select_mm_runtime_policy(
        xyz,
        cutoff_angstrom=float(nonbonded["short_range_cutoff_angstrom"]),
        neighbor_list_minimum_atoms=int(nonbonded["neighbor_list_minimum_atoms"]),
        fmm_minimum_atoms=int(nonbonded["fmm_minimum_atoms"]),
        fmm_precision=float(nonbonded["fmm_precision"]),
        require_linear_scaling=True,
    )
    pairs = _seed_nonbonded_pairs(
        xyz,
        coordinates_bohr,
        policy.short_range_backend,
        float(nonbonded["short_range_cutoff_angstrom"]),
        neighbor_list,
        native=numerical.accelerated,
        excluded_pairs=_seed_excluded_vdw_pairs(model),
    )
    direction_xyz = direction.reshape(xyz.shape)
    if numerical.accelerated:
        radial_product, _pair_count = _native_exppe_hessian_vector(
            xyz,
            np.asarray(
                [float(item["epsilon_hartree"]) for item in atom_parameters]
            ),
            np.asarray(
                [float(item["rmin_half_angstrom"]) for item in atom_parameters]
            ),
            pairs,
            float(nonbonded["short_range_cutoff_angstrom"]),
            direction_xyz,
        )
        product += radial_product.reshape(-1)
    else:
        for left, right in pairs:
            left_parameter, right_parameter = (
                atom_parameters[left],
                atom_parameters[right],
            )
            epsilon = math.sqrt(
                float(left_parameter["epsilon_hartree"])
                * float(right_parameter["epsilon_hartree"])
            )
            r_min = float(left_parameter["rmin_half_angstrom"]) + float(
                right_parameter["rmin_half_angstrom"]
            )
            radial = exppe_from_minimum(
                mie_minimum_derivatives(epsilon, r_min, 12.0, 6.0)
            ).derivatives(float(np.linalg.norm(xyz[left] - xyz[right])))
            delta = xyz[left] - xyz[right]
            distance = float(np.linalg.norm(delta))
            unit = delta / distance
            first = radial.first
            second = radial.second
            block = (second - first / distance) * np.outer(unit, unit)
            block += (first / distance) * np.eye(3)
            pair_direction = direction_xyz[left] - direction_xyz[right]
            pair_product = BOHR_TO_ANGSTROM**2 * (block @ pair_direction)
            product[3 * left : 3 * left + 3] += pair_product
            product[3 * right : 3 * right + 3] -= pair_product
    _energy, _gradient, exppe_product, _count = _seed_continuous_exppe_correction(
        model,
        coordinates_bohr,
        cutoff_angstrom=float(nonbonded["short_range_cutoff_angstrom"]),
        vector_bohr=direction_xyz,
    )
    _energy, _gradient, morse_product, _count = _seed_morse_partition_correction(
        model,
        xyz,
        vector_bohr=direction_xyz,
    )
    if exppe_product is not None:
        product += exppe_product.reshape(-1)
    if morse_product is not None:
        product += morse_product.reshape(-1)
    return product


def _seed_uses_continuous_exppe_weights(model: Mapping[str, Any]) -> bool:
    policy = str(model.get("nonbonded", {}).get("collective_pair_policy", ""))
    return policy in {
        ZAFF_CONTINUOUS_EXPPE_POLICY,
        ZAFF_LEGACY_COVALENT_EXPPE_PARTITION_POLICY,
    }


def _seed_uses_continuous_exppe_partition(model: Mapping[str, Any]) -> bool:
    return str(model.get("nonbonded", {}).get("collective_pair_policy", "")) == (
        ZAFF_LEGACY_COVALENT_EXPPE_PARTITION_POLICY
    )


def _seed_exppe_radial_factory(
    model: Mapping[str, Any],
    *,
    cutoff_angstrom: float,
) -> Callable[[int, int, float], Any | None]:
    atom_parameters = tuple(model["nonbonded_atoms"])

    def radial(left: int, right: int, distance_bohr: float) -> Any | None:
        distance_angstrom = float(distance_bohr) * BOHR_TO_ANGSTROM
        if distance_angstrom > float(cutoff_angstrom):
            return None
        left_parameter = atom_parameters[left]
        right_parameter = atom_parameters[right]
        epsilon = math.sqrt(
            float(left_parameter["epsilon_hartree"])
            * float(right_parameter["epsilon_hartree"])
        )
        r_min = float(left_parameter["rmin_half_angstrom"]) + float(
            right_parameter["rmin_half_angstrom"]
        )
        derivatives = exppe_from_minimum(
            mie_minimum_derivatives(epsilon, r_min, 12.0, 6.0)
        ).derivatives(distance_angstrom)
        return SimpleNamespace(
            energy=float(derivatives.energy),
            first=float(derivatives.first) * BOHR_TO_ANGSTROM,
            second=float(derivatives.second) * BOHR_TO_ANGSTROM**2,
        )

    return radial


def _seed_continuous_exppe_correction(
    model: Mapping[str, Any],
    coordinates_bohr: np.ndarray,
    *,
    cutoff_angstrom: float,
    vector_bohr: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray | None, int]:
    if not _seed_uses_continuous_exppe_weights(model):
        zeros = np.zeros_like(np.asarray(coordinates_bohr, dtype=float))
        return 0.0, zeros, (None if vector_bohr is None else zeros.copy()), 0
    nonbonded = dict(model["nonbonded"])
    return continuous_topology_pair_corrections(
        np.asarray(coordinates_bohr, dtype=float),
        np.asarray(
            [int(atomic_number(symbol) or 0) for symbol in model["atoms"]],
            dtype=int,
        ),
        _seed_exppe_radial_factory(model, cutoff_angstrom=cutoff_angstrom),
        one_four_scale=float(nonbonded.get("one_four_scale", 1.0)),
        switch_alpha_per_angstrom=float(
            nonbonded.get("topology_switch_alpha_per_angstrom", 12.0)
        ),
        vector=vector_bohr,
    )


def _seed_morse_partition_correction(
    model: Mapping[str, Any],
    coordinates_angstrom: np.ndarray,
    *,
    vector_bohr: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray | None, int]:
    """Switch dissociation-referenced Morse wells into the Exp-PE channel.

    The resident Morse kernel uses zero energy at its minimum and ``D_e`` at
    dissociation.  For a continuous edge membership ``s`` the requested
    partition is ``s*V_Morse + (1-s)*(D_e+V_ExpPE)``.  Exp-PE is handled by
    the continuous path correction above; this routine adds the missing
    ``(1-s)*(D_e-V_Morse)`` term with its complete radial chain rule.
    """

    xyz = np.asarray(coordinates_angstrom, dtype=float)
    gradient = np.zeros_like(xyz)
    product = None if vector_bohr is None else np.zeros_like(xyz)
    if not _seed_uses_continuous_exppe_partition(model):
        return 0.0, gradient.reshape(-1), (
            None if product is None else product.reshape(-1)
        ), 0
    direction = None if vector_bohr is None else np.asarray(vector_bohr, dtype=float).reshape(xyz.shape)
    numbers = tuple(int(atomic_number(symbol) or 0) for symbol in model["atoms"])
    alpha = float(model["nonbonded"].get("topology_switch_alpha_per_angstrom", 12.0))
    total = 0.0
    count = 0
    for raw in model["bonds"]:
        if _bond_functional_form(raw) != "MORSE":
            continue
        left, right = (int(value) for value in raw["atoms"])
        radius_left = covalent_radius(numbers[left])
        radius_right = covalent_radius(numbers[right])
        if radius_left is None or radius_right is None:
            continue
        delta = xyz[left] - xyz[right]
        distance = float(np.linalg.norm(delta))
        if distance <= 1.0e-12:
            raise ValueError("coincident atoms in continuous covalent/Exp-PE partition")
        argument = alpha * (
            CNA_SCALE * (float(radius_left) + float(radius_right)) - distance
        )
        gaussian = math.exp(-(argument**2)) / math.sqrt(math.pi)
        membership = 0.5 * (1.0 + math.erf(argument))
        membership_first = -alpha * gaussian
        membership_second = -2.0 * alpha**2 * argument * gaussian
        d_e = float(raw["d_e_hartree"])
        r0 = float(raw["r0_angstrom"])
        morse_alpha = float(raw["alpha_per_angstrom"])
        exponential = math.exp(-morse_alpha * (distance - r0))
        morse = d_e * (1.0 - exponential) ** 2
        morse_first = 2.0 * d_e * morse_alpha * exponential * (1.0 - exponential)
        morse_second = 2.0 * d_e * morse_alpha**2 * exponential * (2.0 * exponential - 1.0)
        complement = 1.0 - membership
        value = complement * (d_e - morse)
        first = -membership_first * (d_e - morse) - complement * morse_first
        second = (
            -membership_second * (d_e - morse)
            + 2.0 * membership_first * morse_first
            - complement * morse_second
        )
        unit = delta / distance
        pair_gradient = first * BOHR_TO_ANGSTROM * unit
        total += value
        gradient[left] += pair_gradient
        gradient[right] -= pair_gradient
        if product is not None and direction is not None:
            block = (second - first / distance) * np.outer(unit, unit)
            block += (first / distance) * np.eye(3)
            pair_direction = direction[left] - direction[right]
            pair_product = BOHR_TO_ANGSTROM**2 * (block @ pair_direction)
            product[left] += pair_product
            product[right] -= pair_product
        count += 1
    return total, gradient.reshape(-1), (
        None if product is None else product.reshape(-1)
    ), count


def _seed_nonbonded_pairs(
    coordinates_angstrom: np.ndarray,
    coordinates_bohr: np.ndarray,
    short_range_backend: str,
    cutoff_angstrom: float,
    neighbor_list: PersistentVerletNeighborList | None,
    *,
    native: bool,
    excluded_pairs: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[tuple[int, int], ...] | np.ndarray:
    if short_range_backend == "spatial_neighbor_list":
        if neighbor_list is not None:
            if native:
                return neighbor_list.native_candidates(coordinates_bohr)
            return neighbor_list.pairs(coordinates_bohr)
        pairs = build_nonbonded_neighbor_list(
            coordinates_bohr,
            cutoff_bohr=float(cutoff_angstrom) / BOHR_TO_ANGSTROM,
        )
    else:
        left, right = np.triu_indices(len(coordinates_angstrom), k=1)
        pairs = np.column_stack((left, right)).astype(np.intp, copy=False)
        if excluded_pairs:
            pairs = pairs[
                [tuple(map(int, pair)) not in excluded_pairs for pair in pairs]
            ]
        if native:
            return pairs
        return tuple(map(tuple, pairs.tolist()))
    if excluded_pairs:
        pairs = tuple(
            (int(left), int(right)) for left, right in pairs
            if tuple(sorted((int(left), int(right)))) not in excluded_pairs
        )
    if native:
        return np.asarray(pairs, dtype=np.intp).reshape(-1, 2)
    return pairs


def _seed_excluded_vdw_pairs(model: Mapping[str, Any]) -> frozenset[tuple[int, int]]:
    policy = str(model.get("nonbonded", {}).get("collective_pair_policy", ""))
    if policy != "NONCOVALENT_EXCLUDE_1_2_1_3_FULL_1_4":
        return frozenset()
    bonds = tuple(
        tuple(sorted((int(left), int(right))))
        for left, right in model.get("topology_bonds", ())
    )
    return _cached_seed_excluded_vdw_pairs(bonds)


@lru_cache(maxsize=128)
def _cached_seed_excluded_vdw_pairs(
    bonds: tuple[tuple[int, int], ...],
) -> frozenset[tuple[int, int]]:
    adjacency: dict[int, set[int]] = {}
    for left, right in bonds:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    distances = bounded_topological_distances(adjacency, maximum_distance=2)
    return frozenset(pair for pair, distance in distances.items() if distance <= 2)


def _seed_morse_bond_arrays(
    model: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    records = tuple(
        raw
        for raw in model["bonds"]
        if _bond_functional_form(raw) == "MORSE"
    )
    arrays = (
        np.asarray(
            [tuple(int(value) for value in raw["atoms"]) for raw in records],
            dtype=np.intp,
        ).reshape(-1, 2),
        np.asarray(
            [float(raw["d_e_hartree"]) for raw in records],
            dtype=float,
        ),
        np.asarray(
            [float(raw["alpha_per_angstrom"]) for raw in records],
            dtype=float,
        ),
        np.asarray(
            [float(raw["r0_angstrom"]) for raw in records],
            dtype=float,
        ),
    )
    for array in arrays:
        array.setflags(write=False)
    return arrays


def _seed_inverse_distance_bond_records(
    model: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        raw
        for raw in model["bonds"]
        if _bond_functional_form(raw)
        in {"INVERSE_POWER_STRETCH", "SPF_INVERSE_DISTANCE"}
    )


def _inverse_distance_bond_potential(
    record: Mapping[str, Any],
) -> InversePowerStretchPotential | SPFStretchPotential:
    if _bond_functional_form(record) == "INVERSE_POWER_STRETCH":
        return InversePowerStretchPotential(
            float(record.get("depth_hartree", record["d_e_hartree"])),
            float(record["r0_angstrom"]),
            float(record["exponent"]),
        )
    return SPFStretchPotential(
        float(record["r0_angstrom"]),
        tuple(float(value) for value in record["coefficients_hartree"]),
    )


def _seed_gaussian_bond_arrays(
    model: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack GFN-FF Gaussian bond records once for the native runtime."""

    records = tuple(
        raw
        for raw in model["bonds"]
        if _bond_functional_form(raw) == "GFNFF_GAUSSIAN"
        and raw.get("reference_response") != "DYNAMIC_GFNFF_LOG_CN"
    )
    arrays = (
        np.asarray(
            [tuple(int(value) for value in raw["atoms"]) for raw in records],
            dtype=np.intp,
        ).reshape(-1, 2),
        np.asarray(
            [float(raw["amplitude_hartree"]) for raw in records],
            dtype=float,
        ),
        np.asarray(
            [float(raw["exponent_per_angstrom2"]) for raw in records],
            dtype=float,
        ),
        np.asarray(
            [float(raw["r0_angstrom"]) for raw in records],
            dtype=float,
        ),
    )
    for array in arrays:
        array.setflags(write=False)
    return arrays


def _seed_dynamic_gaussian_bond_arrays(
    model: Mapping[str, Any],
    coordinates_angstrom: np.ndarray,
    neighbor_list: PersistentVerletNeighborList | None,
) -> tuple[np.ndarray, ...] | None:
    """Pack sparse dynamic-CN bonds without constructing dense derivatives."""

    records = tuple(
        raw
        for raw in model["bonds"]
        if _bond_functional_form(raw) == "GFNFF_GAUSSIAN"
        and raw.get("reference_response") == "DYNAMIC_GFNFF_LOG_CN"
    )
    if not records:
        return None
    coordination = model.get("gfnff_coordination")
    if not isinstance(coordination, Mapping):
        raise ValueError("dynamic GFN-FF bonds require a coordination record")
    cutoff_angstrom = float(coordination["cutoff_angstrom"])
    coordinates_bohr = coordinates_angstrom / BOHR_TO_ANGSTROM
    cutoff_bohr = cutoff_angstrom / BOHR_TO_ANGSTROM
    if neighbor_list is not None and neighbor_list.cutoff_bohr >= cutoff_bohr:
        candidates = neighbor_list.native_candidates(coordinates_bohr)
        delta = coordinates_angstrom[candidates[:, 0]] - coordinates_angstrom[candidates[:, 1]]
        pairs = candidates[np.sum(delta * delta, axis=1) <= cutoff_angstrom**2]
    else:
        pairs = np.asarray(
            build_nonbonded_neighbor_list(
                coordinates_bohr, cutoff_bohr=cutoff_bohr
            ),
            dtype=np.intp,
        ).reshape(-1, 2)
    atom_radii = np.asarray(coordination["covalent_radii_angstrom"], dtype=float)
    if len(atom_radii) != len(coordinates_angstrom):
        raise ValueError("GFN-FF coordination radii do not cover all atoms")
    arrays = (
        pairs,
        atom_radii[pairs[:, 0]] + atom_radii[pairs[:, 1]],
        np.asarray([raw["atoms"] for raw in records], dtype=np.intp).reshape(-1, 2),
        np.asarray([raw["amplitude_hartree"] for raw in records], dtype=float),
        np.asarray([raw["exponent_per_angstrom2"] for raw in records], dtype=float),
        np.asarray([raw["reference_base_angstrom"] for raw in records], dtype=float),
        np.asarray(
            [raw["reference_cn_coefficients_angstrom"] for raw in records],
            dtype=float,
        ).reshape(-1, 2),
        float(coordination.get("cn_max", 4.4)),
        float(coordination.get("steepness", 7.5)),
    )
    return arrays


def _bond_functional_form(record: Mapping[str, Any]) -> str:
    value = (
        str(record.get("functional_form", "MORSE"))
        .strip()
        .upper()
        .replace("-", "_")
    )
    aliases = {
        "MORSE": "MORSE",
        "INVERSE_POWER_STRETCH": "INVERSE_POWER_STRETCH",
        "GENERALIZED_KRATZER": "INVERSE_POWER_STRETCH",
        "SPF": "SPF_INVERSE_DISTANCE",
        "SPF_INVERSE_DISTANCE": "SPF_INVERSE_DISTANCE",
        "SIMONS_PARR_FINLAN_INVERSE_DISTANCE": "SPF_INVERSE_DISTANCE",
        "GFNFF_GAUSSIAN": "GFNFF_GAUSSIAN",
        "GFN_FF_GAUSSIAN": "GFNFF_GAUSSIAN",
        "GAUSSIAN": "GFNFF_GAUSSIAN",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(f"unsupported covalent bond functional form: {value}") from exc


def _seed_out_of_plane_records(model: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return SOOP records while accepting both spellings used by artifacts."""

    records = model.get("out_of_plane", model.get("out_of_planes", ()))
    if records is None:
        return ()
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("ZAFF SOOP records must be a sequence")
    return tuple(records)


def _seed_out_of_plane_primitives(raw: Mapping[str, Any]) -> tuple[Primitive, ...]:
    atoms = tuple(int(value) for value in raw["atoms"])
    if len(atoms) != 4:
        raise ValueError("ZAFF SOOP records require center, three neighbors, and one moving atom")
    center, plane1, plane2, moving = atoms
    return (
        Primitive("bond", (center, plane1)),
        Primitive("bond", (center, plane2)),
        Primitive("bond", (center, moving)),
        Primitive("out_of_plane", atoms),
    )


def _seed_out_of_plane_potential(raw: Mapping[str, Any]) -> BondOrderDampedOutOfPlanePotential:
    switches = raw.get("radial_factors", raw.get("switches"))
    if not isinstance(switches, Sequence) or len(switches) != 3:
        raise ValueError("ZAFF SOOP records require exactly three radial switches")
    factors = tuple(
        BondOrderRadialFactor(
            float(item["reference_distance_angstrom"]),
            float(item["covalent_radius_sum_angstrom"]),
        )
        for item in switches
    )
    coefficients = raw.get("coefficients_hartree", raw.get("coefficients"))
    if coefficients is None:
        # Compatibility with the provisional shifted-well representation.
        amplitude = float(raw["amplitude_hartree"])
        sine0 = math.sin(float(raw.get("chi0_radian", 0.0)))
        coefficients = (amplitude * sine0**2, -2.0 * amplitude * sine0, amplitude)
    shape = SinOutOfPlanePolynomial(tuple(float(value) for value in coefficients))
    return BondOrderDampedOutOfPlanePotential(
        factors[0], factors[1], factors[2], shape
    )


def _gaussian_bond_potential(record: Mapping[str, Any]) -> GaussianBondPotential:
    return GaussianBondPotential(
        amplitude_hartree=float(record["amplitude_hartree"]),
        exponent_per_angstrom2=float(record["exponent_per_angstrom2"]),
        reference_distance_angstrom=float(record["r0_angstrom"]),
    )


def _seed_local_valence_arrays(
    model: Mapping[str, Any],
) -> tuple[np.ndarray, ...]:
    """Pack angle/torsion records once for the portable native runtime."""

    angle_records = tuple(model["angles"])
    torsion_records = tuple(model["torsions"])
    angle_atoms = np.asarray(
        [tuple(int(value) for value in raw["atoms"]) for raw in angle_records],
        dtype=np.intp,
    ).reshape(-1, 3)
    angle_parameters = np.asarray(
        [
            (
                2.0 * float(raw["k_hartree_per_radian2"]),
                float(raw["theta0_radian"]),
                float(raw["radial_factors"][0]["reference_distance_angstrom"]),
                float(raw["radial_factors"][0]["covalent_radius_sum_angstrom"]),
                float(raw["radial_factors"][1]["reference_distance_angstrom"]),
                float(raw["radial_factors"][1]["covalent_radius_sum_angstrom"]),
            )
            for raw in angle_records
        ],
        dtype=float,
    ).reshape(-1, 6)
    torsion_atoms = np.asarray(
        [tuple(int(value) for value in raw["atoms"]) for raw in torsion_records],
        dtype=np.intp,
    ).reshape(-1, 4)
    torsion_parameters = np.asarray(
        [
            tuple(
                value
                for factor in raw["radial_factors"]
                for value in (
                    float(factor["reference_distance_angstrom"]),
                    float(factor["covalent_radius_sum_angstrom"]),
                )
            )
            for raw in torsion_records
        ],
        dtype=float,
    ).reshape(-1, 6)
    offsets = np.zeros(len(torsion_records) + 1, dtype=np.intp)
    for index, raw in enumerate(torsion_records):
        offsets[index + 1] = offsets[index] + len(raw["terms"])
    terms = np.asarray(
        [
            (
                float(term["amplitude_hartree"]),
                float(term["periodicity"]),
                float(term["phase_radian"]),
            )
            for raw in torsion_records
            for term in raw["terms"]
        ],
        dtype=float,
    ).reshape(-1, 3)
    arrays = (
        angle_atoms,
        angle_parameters,
        torsion_atoms,
        torsion_parameters,
        offsets,
        terms,
    )
    for array in arrays:
        array.setflags(write=False)
    return arrays


def _fast_zoom_selected(zoom_level: str | None) -> bool:
    if zoom_level is None:
        return False
    normalized = str(zoom_level).strip().upper().replace("-", "_")
    if normalized != ZAFF_FAST_RIGID_TORSIONAL_ZOOM:
        raise ValueError(f"unsupported transferable ZAFF zoom level: {zoom_level}")
    return True


def _internal_term_hessian_vector_product(
    primitives: Sequence[Primitive],
    gradient_internal: np.ndarray,
    hessian_internal: np.ndarray,
    coordinates_angstrom: np.ndarray,
    vector_bohr: np.ndarray,
) -> np.ndarray:
    """Apply ``B.T F B + sum(g_i B'_i)`` with analytic primitive B-prime."""

    b_matrix = primitive_b_matrix(primitives, coordinates_angstrom)
    direction_angstrom = (
        BOHR_TO_ANGSTROM * np.asarray(vector_bohr, dtype=float).reshape(-1)
    )
    product_angstrom = b_matrix.T @ (
        np.asarray(hessian_internal, dtype=float) @ (b_matrix @ direction_angstrom)
    )
    function_map = {"bond": "R", "angle": "A", "dihedral": "D", "out_of_plane": "U"}
    for primitive, coefficient in zip(
        primitives, np.asarray(gradient_internal, dtype=float), strict=True
    ):
        proxy = SimpleNamespace(
            function=function_map[primitive.kind],
            atoms=tuple(int(atom) + 1 for atom in primitive.atoms),
            ref_atoms=(),
            frame_atoms=(),
            ref_frame_atoms=(),
            identifier=f"SEED_{primitive.kind}_{primitive.atoms}",
        )
        product_angstrom += float(
            coefficient
        ) * analytic_basic_primitive_hessian_vector(
            proxy,
            coordinates_angstrom,
            direction_angstrom,
        )
    return BOHR_TO_ANGSTROM * product_angstrom



__all__ = [
    "ZAFF_FAST_RIGID_TORSIONAL_ZOOM",
    "ZAFF_SEED_MODEL_SCHEMA",
    "ZaffSeedCoverageError",
    "ZaffSeedDynamicsRuntime",
    "ZaffSecondOrderAnchoredSeedRuntime",
    "ZaffSeedEvaluation",
    "evaluate_zaff_fast_rigid_torsional_batch_energies",
    "evaluate_zaff_seed_energy",
    "evaluate_zaff_seed_model",
]
