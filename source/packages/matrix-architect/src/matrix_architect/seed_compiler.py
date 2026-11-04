"""Compile transferable diagonal ZAFF seed fields and full-fit priors."""

from __future__ import annotations

from itertools import combinations, product
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from matrix_chem.topology.elements import atomic_number
from matrix_core import (
    ElectronicObservableCompleteness,
    ElectronicObservableKind,
    ElectronicObservableOwner,
    ElectronicObservableProvenance,
)
from matrix_zaff.charge_response import (
    ChargeResponseSite,
)
from matrix_zaff.nonbonded import (
    BOHR_TO_ANGSTROM,
)
from matrix_zaff.polarization import ANGSTROM3_TO_BOHR3
from matrix_zaff.synthon_nonbonded import descriptors_from_atomic_synthons
from matrix_zaff.synthon_torsion import (
    compile_gfnff_synthon_torsion,
    synthon_type_key,
)
from matrix_zaff.seed_runtime import (
    ZAFF_FAST_RIGID_TORSIONAL_ZOOM,
    ZAFF_SEED_MODEL_SCHEMA,
    ZaffSeedCoverageError,
)
from matrix_zaff.periodic_parameters import zaff_periodic_element_prior
from matrix_zaff.radial import inverse_power_stretch_from_morse_prior

from .gaff import (
    GaffParameterSet,
    assign_initial_gaff_types,
    default_gaff_parameter_path,
)
from .mm3_morse import load_mm3_morse_catalog


ZAFF_SEED_PRIOR_SCHEMA = "matrix.zaff.full_fit_priors.v1"
KCAL_MOL_TO_HARTREE = 1.0 / 627.5094740631


def compile_zaff_seed_model(
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    cm5_charges: Sequence[float] | None = None,
    *,
    mayer_bond_orders: Mapping[tuple[int, int], float] | None = None,
    gaff_path: Path | str | None = None,
    charge_source: str = "APOC CM5 full molecule",
    bond_order_source: str = "APOC Mayer full molecule",
    charge_owner: ElectronicObservableOwner | str = ElectronicObservableOwner.APOC,
    bond_order_owner: ElectronicObservableOwner | str = ElectronicObservableOwner.APOC,
    fragment_provenance: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Compile the initial hydrocarbon/amino-acid/sugar diagonal field.

    Stretches come from continuous MM3 synthon interpolation translated once
    to the permanent generalized inverse-power form, angles
    from an official GAFF/GAFF2 parameter file, and torsions from the GFN-FF
    construction compiled onto immutable four-synthon keys. Each GAFF 12-6
    pair is transformed to the ZAFF Exp-PE modified-Morse polynomial while
    preserving its minimum, well depth and curvature.
    """

    symbols = tuple(str(value) for value in atoms)
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if not symbols or xyz.shape != (len(symbols), 3) or np.any(~np.isfinite(xyz)):
        raise ValueError("ZAFF seed geometry must have shape (natoms, 3)")
    numbers = tuple(int(atomic_number(symbol) or 0) for symbol in symbols)
    if any(number <= 0 for number in numbers):
        raise ZaffSeedCoverageError("ZAFF seed geometry contains an unknown element")
    parameter_path = Path(gaff_path) if gaff_path is not None else default_gaff_parameter_path()
    if parameter_path is None:
        raise ZaffSeedCoverageError(
            "the MATRIX installation lacks its resident GAFF2 parameter catalogue"
        )
    gaff = GaffParameterSet.from_file(parameter_path)

    from matrix_chem import build_topology_objects

    candidate_charges = (
        {}
        if cm5_charges is None
        else {
            index: float(value)
            for index, value in enumerate(_validated_atomic_charges(cm5_charges, len(symbols)))
        }
    )
    candidate_orders = {
        tuple(sorted((int(left), int(right)))): float(value)
        for (left, right), value in dict(mayer_bond_orders or {}).items()
    }
    _continuous, graph, rings, synthons, aromaticity = build_topology_objects(
        xyz,
        numbers,
        external_charges=candidate_charges,
        bond_order_overrides=candidate_orders,
        charge_source=str(charge_source),
        bond_order_source=str(bond_order_source),
    )
    charges = np.asarray(
        [float(synthons.charge(index)) for index in range(len(symbols))],
        dtype=float,
    )
    synthon_descriptors = descriptors_from_atomic_synthons(
        synthons,
        intrinsic_cm5_charges_e=charges,
    )
    synthon_type_keys = tuple(synthon_type_key(item) for item in synthon_descriptors)
    realized_charge_source = str(synthons._charge_source)
    realized_bond_order_source = str(synthons._bond_order_source)
    external_charge_level = bool(getattr(synthons, "_external_charges", None))
    external_bond_order_level = bool(
        getattr(synthons, "_external_bond_orders", None)
    )
    external_population_level = external_charge_level and external_bond_order_level
    realized_charge_owner = (
        ElectronicObservableOwner(charge_owner)
        if external_charge_level
        else ElectronicObservableOwner.ORACLE
    )
    realized_bond_order_owner = (
        ElectronicObservableOwner(bond_order_owner)
        if external_bond_order_level
        else ElectronicObservableOwner.ORACLE
    )
    charge_provenance = ElectronicObservableProvenance(
        owner=realized_charge_owner,
        observable=(
            ElectronicObservableKind.CM5
            if external_charge_level
            else ElectronicObservableKind.ELECTRONEGATIVITY_ESTIMATE
        ),
        completeness=(
            ElectronicObservableCompleteness.COMPLETE
            if external_charge_level
            else ElectronicObservableCompleteness.FALLBACK_COMPLETE
        ),
        source=realized_charge_source,
    )
    bond_order_provenance = ElectronicObservableProvenance(
        owner=realized_bond_order_owner,
        observable=(
            ElectronicObservableKind.MAYER
            if external_bond_order_level
            else ElectronicObservableKind.PAULING_ESTIMATE
        ),
        completeness=(
            ElectronicObservableCompleteness.COMPLETE
            if external_bond_order_level
            else ElectronicObservableCompleteness.FALLBACK_COMPLETE
        ),
        source=realized_bond_order_source,
    )
    if external_population_level:
        electronic_observable_level = "QM_CM5_MAYER"
    elif external_charge_level:
        electronic_observable_level = "QM_CM5_ORACLE_PAULING"
    elif external_bond_order_level:
        electronic_observable_level = "ORACLE_ELECTRONEGATIVITY_QM_MAYER"
    else:
        electronic_observable_level = "ORACLE_ELECTRONEGATIVITY_PAULING"
    ring_coordinate_records, atom_ring_coordinates = _direct_charm_ring_coordinates(
        rings,
        xyz,
    )
    atom_types = assign_initial_gaff_types(
        numbers,
        graph,
        synthons,
        aromaticity,
        allow_periodic_fallback=True,
    )
    atom_type_mixtures = tuple(
        _gaff_synthon_type_mixture(
            gaff,
            int(numbers[index]),
            float(synthons.cna(index)),
            sum(
                float(synthons.bond_order_pi(index, neighbor))
                for neighbor in graph.adjacency[index]
            ),
            float(charges[index]),
        )
        for index in range(len(symbols))
    )
    bonds = tuple(tuple(sorted(map(int, pair))) for pair in graph.bonds)
    geminal_zero_terms = [
        {
            "center_atom_zero_based": center,
            "hydrogens_zero_based": list(pair),
            "force_constant_hartree_per_bohr2": 0.0,
            "status": "UNFITTED_ZERO",
        }
        for center, number in enumerate(numbers) if number != 1
        for pair in combinations(
            sorted(
                right if left == center else left
                for left, right in bonds
                if center in (left, right)
                and numbers[right if left == center else left] == 1
            ), 2,
        )
    ]
    angles = _valence_angles(graph)
    torsions = _valence_torsions(graph)
    morse_catalog = load_mm3_morse_catalog()
    bond_records: list[dict[str, Any]] = []
    angle_records: list[dict[str, Any]] = []
    torsion_records: list[dict[str, Any]] = []
    for left, right in bonds:
        distance = float(np.linalg.norm(xyz[left] - xyz[right]))
        try:
            prior = morse_catalog.interpolate_from_synthons(
                synthons, numbers, left, right, distance
            )
        except ValueError:
            bond_records.append(
                _periodic_morse_bond_record(
                    numbers, synthons, left, right, distance
                )
            )
        else:
            bond_records.append(
                _inverse_distance_bond_record(
                    {
                    "atoms": [left, right],
                    "d_e_hartree": prior.d_e_hartree,
                    "r0_angstrom": prior.r0_angstrom,
                    "alpha_per_angstrom": prior.alpha_per_angstrom,
                    "initial_curvature_hartree_per_angstrom2": (
                        prior.harmonic_curvature_hartree_per_angstrom2
                    ),
                    "source": prior.source,
                    "uncertainty_class": "CLASS_SPECIFIC_INTERPOLATION",
                    "synthon_contributors": [
                        {
                            "types": [item.type_i, item.type_j],
                            "labels": [item.label_i, item.label_j],
                            "weight": item.weight,
                            "descriptor_distance": item.descriptor_distance,
                        }
                        for item in prior.contributions
                    ],
                    }
                )
            )
    for atoms3 in angles:
        types = tuple(atom_types[index] for index in atoms3)
        try:
            parameter, contributors = _interpolate_gaff_angle(
                gaff, tuple(atom_type_mixtures[index] for index in atoms3)
            )
        except (KeyError, ValueError):
            angle_records.append(
                _periodic_angle_record(numbers, synthons, xyz, atoms3, types)
            )
        else:
            angle_records.append(
                {
                    "atoms": list(atoms3),
                    "types": list(types),
                    "k_hartree_per_radian2": (
                        parameter.force_constant_kcal_per_mol_radian2 * KCAL_MOL_TO_HARTREE
                    ),
                    "theta0_radian": math.radians(
                        float(synthons.theta_ref(synthons.electron_domains(atoms3[1])))
                    ),
                    "radial_factors": [
                        _radial_factor_record(synthons, xyz, atoms3[0], atoms3[1]),
                        _radial_factor_record(synthons, xyz, atoms3[1], atoms3[2]),
                    ],
                    "source": (
                        f"{gaff.banner}; force constant from continuous donor interpolation; "
                        "theta0 from ORACLE synthon reference angle"
                    ),
                    "uncertainty_class": "CLASS_SPECIFIC_INTERPOLATION",
                    "synthon_contributors": contributors,
                }
            )
    for atoms4 in torsions:
        central_bond = tuple(sorted((int(atoms4[1]), int(atoms4[2]))))
        ring_indices = tuple(rings.bond_to_rings.get(central_bond, ()))
        ring_size = min(
            (len(rings.rings[index].atoms) for index in ring_indices),
            default=0,
        )
        compiled = compile_gfnff_synthon_torsion(
            synthon_descriptors,
            atoms4,
            central_bond_order=float(synthons.bond_order(*central_bond)),
            reference_coordinates_angstrom=xyz[np.asarray(atoms4, dtype=int)],
            ring_size=ring_size,
        )
        if compiled is None:
            continue
        torsion_records.append(
            {
                "schema": compiled["schema"],
                "atoms": list(atoms4),
                "synthon_types": list(compiled["synthon_types"]),
                "terms": list(compiled["terms"]),
                "radial_factors": [
                    _radial_factor_record(synthons, xyz, atoms4[0], atoms4[1]),
                    _radial_factor_record(synthons, xyz, atoms4[1], atoms4[2]),
                    _radial_factor_record(synthons, xyz, atoms4[2], atoms4[3]),
                ],
                "source": compiled["source"],
                "runtime_parameterization": compiled["runtime_parameterization"],
                "source_components": compiled["source_components"],
                "central_bond_order": compiled["central_bond_order"],
                "ring_size": compiled["ring_size"],
            }
        )
    if len(torsion_records) != len(torsions):
        raise ZaffSeedCoverageError(
            "automatic ZAFF-fast construction did not generate the complete primitive torsion set"
        )
    nonbonded_atoms: list[dict[str, Any]] = []
    for index, atom_type in enumerate(atom_types):
        coordination = float(synthons.cna(index))
        effective_radius = float(synthons.covalent_radius_eff(index))
        synthon_polarizability = float(synthons.polarizability(index))
        try:
            parameter, contributions = _interpolate_gaff_nonbonded(gaff, atom_type_mixtures[index])
        except (KeyError, ValueError):
            periodic = zaff_periodic_element_prior(numbers[index])
            parameter = (
                periodic.vdw_radius_angstrom,
                periodic.vdw_well_depth_kcal_per_mol,
            )
            contributions = [
                {
                    "type": f"Z{numbers[index]}:PERIODIC",
                    "weight": 1.0,
                    "descriptor_distance": 0.0,
                    "source": periodic.source,
                    "uncertainty_class": periodic.uncertainty_class,
                }
            ]
        nonbonded_atoms.append(
            {
                "atom": index,
                "type": atom_type,
                "rmin_half_angstrom": parameter[0],
                "epsilon_hartree": parameter[1] * KCAL_MOL_TO_HARTREE,
                "atomic_charge": float(charges[index]),
                "polarizability_bohr3": (synthon_polarizability * ANGSTROM3_TO_BOHR3),
                "oracle_synthon_descriptors": {
                    "continuous_coordination": coordination,
                    "effective_covalent_radius_angstrom": effective_radius,
                    "environmental_polarizability_angstrom3": synthon_polarizability,
                    "direct_charm_ring_coordinates": atom_ring_coordinates[index],
                },
                "synthon_contributors": contributions,
                "parameter_source": (
                    "GAFF2_SYNTHON_INTERPOLATION"
                    if atom_type_mixtures[index]
                    else "ZAFF_COMPLETE_PERIODIC_TABLE_PRIOR"
                ),
            }
        )
    charge_response = _compile_reference_charge_response(xyz, nonbonded_atoms)
    return {
        "schema": ZAFF_SEED_MODEL_SCHEMA,
        "functional_profile": "ZAFF_FAST",
        "domains": ["complete_periodic_table_Z1_Z118"],
        "parameterization_regime": "LOCAL_DONOR_THEN_COMPLETE_PERIODIC_TABLE_PRIOR",
        "zoom_contract": {
            "universal": "PERIODIC_ATOMIC_PRIORS_PLUS_ORACLE_CONTINUOUS_PERCEPTION",
            "class_specific": ("DONOR_LIBRARY_PARAMETERS_INTERPOLATED_BY_ORACLE_SYNTHONS"),
            "molecule_specific": (
                "QM_REFINEMENT_WITH_EXPLICIT_COUPLINGS_AND_PIC_SONIC_CHARM_COORDINATE_CHARTS"
            ),
            "realized_level": "LOCAL_WHEN_AVAILABLE_OTHERWISE_PERIODIC_Z1_Z118",
        },
        "parameter_inference_contract": {
            "available_routes": [
                "PHYSICS_BASED_HEURISTICS",
                "QM_FITTING",
                "DATA_SUFFICIENT_MACHINE_LEARNING",
            ],
            "realized_route": "PHYSICS_BASED_HEURISTICS_AND_DONOR_INTERPOLATION",
            "invariants": [
                "CONTINUOUS_ENERGY_AND_REQUIRED_DERIVATIVES",
                "PHYSICALLY_CORRECT_ASYMPTOTIC_LIMITS",
                "PROVENANCE_AND_UNCERTAINTY_RECORDED",
            ],
            "ml_boundary": ("ML_MAY_PREDICT_PARAMETERS_OR_RESIDUALS_INSIDE_ZAFF_PHYSICAL_FORMS"),
        },
        "architecture_contract": {
            "energy_partition": (
                "TOPOLOGY_INDEPENDENT_COLLECTIVE_FIELD_PLUS_"
                "LOCAL_SYMMETRY_AWARE_COVALENT_CORRECTIONS"
            ),
            "collective_pair_policy": (
                "CONTINUOUS_ERF_PATH_WEIGHTS_FOR_EXPPE_SEPARATE_FROM_COVALENT; "
                "ELECTROSTATICS_ALL_PAIRS"
            ),
            "local_terms": ("STRETCH_BEND_TORSION_AND_SELECTED_COUPLINGS_WITH_PHYSICAL_LIMITS"),
            "selection_metric": {
                "form": "CONSTRAINED_PARETO_VECTOR_NOT_ARBITRARY_SCALAR_SUM",
                "accuracy": "CARTESIAN_HESSIAN_AND_MODE_SUBSPACE_ERROR",
                "locality": "LOCAL_COORDINATE_AND_COUPLING_SUPPORT",
                "sparsity": "ACTIVE_PARAMETER_AND_COUPLING_COUNT",
                "symmetry": "LOCAL_SONIC_COMPRESSION_AT_EQUAL_CARTESIAN_ERROR",
            },
            "special_coordinates": {
                "rings": "SMITH_CHARM_IMPLEMENTED",
                "generic": "SERIALIZABLE_SPECIAL_BASIS_IMPLEMENTED",
                "transition_metals": (
                    "SMITH_INTERACTION_CENTER_AND_NO_CEILING_LOCAL_SALC_IMPLEMENTED"
                ),
            },
            "large_system": {
                "short_range": "PAIR_LIST",
                "long_range_electrostatics": "LAPLACE_FMM",
                "dense_hessian": "REFERENCE_VALIDATION_ONLY",
            },
        },
        "atoms": list(symbols),
        "atom_types": list(atom_types),
        "synthon_type_keys": list(synthon_type_keys),
        "atom_type_mixtures": [list(items) for items in atom_type_mixtures],
        "topology_bonds": [list(pair) for pair in bonds],
        "geminal_xh_couplings": geminal_zero_terms,
        "bonds": bond_records,
        "angles": angle_records,
        "torsions": torsion_records,
        "ring_coordinates": ring_coordinate_records,
        "nonbonded_atoms": nonbonded_atoms,
        "charge_response": charge_response,
        "fast_exploration": {
            "schema": "matrix.zaff.fast_rigid_torsional.v1",
            "zoom_level": ZAFF_FAST_RIGID_TORSIONAL_ZOOM,
            "purpose": "INITIAL_CONFORMER_AND_INTERMOLECULAR_EXPLORATION",
            "rigid_constraints": [
                "INTRAMOLECULAR_BOND_LENGTHS",
                "INTRAMOLECULAR_VALENCE_ANGLES",
                "INTERMOLECULAR_FRAGMENT_INTERNAL_GEOMETRIES",
            ],
            "active_terms": [
                "INTRAMOLECULAR_FOUR_SYNTHON_TORSIONAL_FOURIER",
                "CONTINUOUSLY_WEIGHTED_EXPPE",
                "ALL_PAIR_GAUSSIAN_ELECTROSTATICS",
            ],
            "omitted_terms": [
                "STRETCH",
                "BEND",
                "POLARIZATION",
                "GENERAL_COORDINATE_TRANSFORMATIONS",
            ],
            "coordinate_runtime": ("DIRECT_CARTESIAN_DIHEDRAL_AND_SE3_POSE_NO_PIC_SONIC_B_MATRIX"),
            "derivative_contract": "ANALYTIC_E_G_HVP",
            "parallel_contract": ("CANDIDATE_BATCH_PARALLEL_PLUS_PAIR_LIST_AND_FMM_BACKENDS"),
        },
        "nonbonded": {
            "vdw_model": "GAFF_WHEN_AVAILABLE_OTHERWISE_PERIODIC_UFF_TO_ZAFF_EXPPE",
            "mixing_rule": "LORENTZ_BERTHELOT_RMIN_HALF_SUM_EPSILON_GEOMETRIC",
            "collective_pair_policy": (
                "CONTINUOUS_ERF_PATH_WEIGHTS_FOR_EXPPE_SEPARATE_FROM_COVALENT"
            ),
            "parameter_transfer": (
                "ORACLE_SYNTHON_INTERPOLATION_IN_CNA_PI_CHARGE_WITH_RADIUS_"
                "POLARIZABILITY_AND_DIRECT_CHARM_RING_STATE"
            ),
            "short_range_overlap_switch": (
                "ORACLE_ERF_PATH_WEIGHTS_APPLY_TO_EXPPE_ONLY"
            ),
            "topology_switch_alpha_per_angstrom": 12.0,
            "one_four_scale": 1.0,
            "short_range_cutoff_angstrom": 12.0,
            "neighbor_list_minimum_atoms": 64,
            "fmm_minimum_atoms": 256,
            "fmm_precision": 1.0e-10,
            "excluded_terms": [
                "D3_DISPERSION", "D4_DISPERSION", "THREE_BODY_REPULSION"
            ],
            "polarization": {
                "status": "ACTIVE_LOCAL_SPLIT_CHARGE_QEQ_SQE",
                "schema": "matrix.zaff.split_charge_response.v1",
                "reference_charges": realized_charge_source,
                "site_policy": (
                    "ATOMIC_DEFAULT_WITH_AFFINE_LONE_PAIR_AND_OTHER_VIRTUAL_SITES_SUPPORTED"
                ),
                "charge_conservation": "EXACT_INCIDENCE_MATRIX_COLUMN_SUM_ZERO",
                "damping": "SMOOTH_DISTANCE_SWITCHED_LOCAL_CHARGE_FLOW",
                "solver": (
                    "VARIATIONAL_IMPLICIT_LOCAL_PAIR_SITE_SCHUR_WITH_"
                    "COMPACT_SUPPORT"
                ),
                "united_atom_option": "OPTIONAL_METHYL_CH3_ZOOM_LEVEL",
                "large_system_status": ("LINEAR_MEMORY_PAIR_CHUNKS_PLUS_FMM_GAUSSIAN_OPERATOR"),
                "forbidden_implicit_fallback": "EEQ",
                "induced_dipole_role": "COMPARISON_ONLY_NOT_COMBINED_BY_DEFAULT",
            },
        },
        "charge_model": "CM5" if external_charge_level else "ORACLE_ELECTRONEGATIVITY",
        "charge_source": realized_charge_source,
        "bond_order_source": realized_bond_order_source,
        "electronic_observable_level": electronic_observable_level,
        "electronic_observables": {
            "pairing_contract": (
                "INDEPENDENT_COMPLETE_OBSERVABLES_WITH_PER_VECTOR_ORACLE_FALLBACK"
            ),
            "cross_observable_mixing_allowed": True,
            "partial_vector_mixing_forbidden": True,
            "charge_provenance": charge_provenance.to_record(),
            "bond_order_provenance": bond_order_provenance.to_record(),
            "charges": [float(value) for value in charges],
            "bond_orders": [
                {
                    "atoms": [left, right],
                    "value": float(synthons.bond_order(left, right)),
                }
                for left, right in bonds
            ],
        },
        "fragment_provenance": [dict(item) for item in fragment_provenance],
        "parameter_provenance": {
            "gaff_banner": gaff.banner,
            "gaff_path": gaff.source_path,
            "gaff_sha256": gaff.source_sha256,
            "gaff_origin": gaff.source_origin,
            "gaff_license": gaff.source_license,
            "stretch_source": (
                "MM3 Morse synthon interpolation translated once to the permanent "
                "ZAFF generalized inverse-power form"
            ),
            "torsion_source": "GFN-FF-analogous synthon torsion rules in ZAFF functional form",
            "zaff_fast_automatic_inputs": ["INTRINSIC_CM5_DEPURATED_AT_USE", "MAYER", "SYNTHONS"],
            "coverage_gap_policy": (
                "RULE_BASED_EXTRAPOLATION_GFNFF_ANALOGUE_ZAFF_FORMS_CM5_MAYER_SYNTHONS"
            ),
            "zaff1_hessian_residual": (
                "H_QM_MINUS_ZAFF_FAST_NONBONDED_ELECTROSTATIC_AND_OTHER_FIXED_TERMS"
            ),
            "atom_type_contract": (
                "CONTINUOUS_ORACLE_SYNTHON_MEMBERSHIP; DOMINANT_GAFF_LABEL_IS_DIAGNOSTIC_ONLY"
            ),
            "missing_heteronuclear_alpha_rule": (
                "DIRECT_AB_THEN_GEOMETRIC_MEAN_OF_INTERPOLATED_AA_AND_BB"
            ),
        },
    }


def build_zaff_seed_force_field(
    atoms: Sequence[str],
    coordinates_angstrom: np.ndarray,
    cm5_charges: Sequence[float] | None = None,
    *,
    mayer_bond_orders: Mapping[tuple[int, int], float] | None = None,
    gaff_path: Path | str | None = None,
    charge: int = 0,
    multiplicity: int = 1,
    charge_source: str = "APOC CM5 full molecule",
    bond_order_source: str = "APOC Mayer full molecule",
    charge_owner: ElectronicObservableOwner | str = ElectronicObservableOwner.APOC,
    bond_order_owner: ElectronicObservableOwner | str = ElectronicObservableOwner.APOC,
    fragment_provenance: Sequence[Mapping[str, Any]] = (),
):
    """Build a serializable ZAFF field backed by the diagonal physical runtime."""

    from .model import ZaffForceField

    model = compile_zaff_seed_model(
        atoms,
        coordinates_angstrom,
        cm5_charges,
        mayer_bond_orders=mayer_bond_orders,
        gaff_path=gaff_path,
        charge_source=charge_source,
        bond_order_source=bond_order_source,
        charge_owner=charge_owner,
        bond_order_owner=bond_order_owner,
        fragment_provenance=fragment_provenance,
    )
    from matrix_zaff.seed_runtime import evaluate_zaff_seed_model as evaluate_runtime

    reference = evaluate_runtime(model, coordinates_angstrom, hessian=True)
    return ZaffForceField(
        atoms=tuple(str(value) for value in atoms),
        reference_coordinates_angstrom=np.asarray(coordinates_angstrom, dtype=float),
        energy_reference_hartree=reference.energy_hartree,
        gradient_reference_hartree_per_bohr=reference.gradient_hartree_per_bohr,
        hessian_hartree_per_bohr2=np.asarray(reference.hessian_hartree_per_bohr2),
        coordinate_contract={
            "owner": "ORACLE",
            "model": "TRANSFERABLE_DIAGONAL_VALENCE_PLUS_NONBONDED",
            "topology_bonds": list(model["topology_bonds"]),
        },
        metadata={
            "identifier": "ZAFF_TRANSFERABLE_SEED",
            "charge": int(charge),
            "multiplicity": int(multiplicity),
            "diagonal_seed_model": model,
            "full_fit_priors": zaff_seed_full_fit_priors(model),
            "nonbonded_parameters": {
                "regional_neighbor_cutoff_angstrom": 12.0,
                "neighbor_list_minimum_atoms": 64,
                "fmm_minimum_atoms": 256,
                "fmm_precision": 1.0e-10,
            },
        },
    )


def zaff_seed_full_fit_priors(
    model: Mapping[str, Any],
    *,
    stretch_weight: float = 1.0,
    bend_weight: float = 0.5,
    torsion_weight: float = 0.25,
    nonbonded_weight: float = 0.5,
) -> dict[str, Any]:
    """Expose a seed model as auditable regularized priors for a full QM fit."""

    if model.get("schema") != ZAFF_SEED_MODEL_SCHEMA:
        raise ValueError("full-fit priors require a ZAFF diagonal seed model")
    weights = {
        "stretch": _positive_weight(stretch_weight, "stretch"),
        "bend": _positive_weight(bend_weight, "bend"),
        "torsion": _positive_weight(torsion_weight, "torsion"),
        "nonbonded": _positive_weight(nonbonded_weight, "nonbonded"),
    }
    return {
        "schema": ZAFF_SEED_PRIOR_SCHEMA,
        "role": "INITIAL_VALUES_AND_REGULARIZED_PRIORS_NOT_HARD_CONSTRAINTS",
        "weights": weights,
        "initial_parameters": {
            "bonds": list(model["bonds"]),
            "angles": list(model["angles"]),
            "torsions": list(model["torsions"]),
            "nonbonded_atoms": list(model["nonbonded_atoms"]),
        },
        "fit_audit_contract": {
            "per_parameter": [
                "initial_value",
                "fitted_value",
                "signed_displacement",
                "prior_weight",
                "normalized_displacement",
            ],
            "aggregate": ["prior_penalty", "qm_residual", "effective_prior_rank"],
        },
    }


def assemble_fragment_cm5_charges(
    natoms: int,
    fragments: Sequence[Mapping[str, Any]],
    *,
    molecular_charge: int,
    tolerance: float = 5.0e-4,
) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    """Assemble nonoverlapping fragment CM5 vectors in full atom order."""

    charges = np.full(int(natoms), np.nan, dtype=float)
    provenance: list[dict[str, Any]] = []
    for raw in fragments:
        identifier = str(raw.get("id", "")).strip()
        atoms = tuple(int(value) for value in raw.get("atoms", ()))
        values = np.asarray(raw.get("cm5_charges", ()), dtype=float).reshape(-1)
        if not identifier or not atoms or len(set(atoms)) != len(atoms):
            raise ValueError("each CM5 fragment needs an id and unique zero-based atoms")
        if len(values) != len(atoms) or np.any(~np.isfinite(values)):
            raise ValueError(f"CM5 fragment {identifier} has an incomplete charge vector")
        if min(atoms) < 0 or max(atoms) >= int(natoms):
            raise ValueError(f"CM5 fragment {identifier} contains an out-of-range atom")
        if np.any(np.isfinite(charges[list(atoms)])):
            raise ValueError("CM5 fragments must not overlap")
        charges[list(atoms)] = values
        provenance.append(
            {
                "id": identifier,
                "atoms": list(atoms),
                "charge_sum": float(np.sum(values)),
                "source": str(raw.get("source", "APOC CM5 fragment")),
            }
        )
    if np.any(~np.isfinite(charges)):
        missing = np.flatnonzero(~np.isfinite(charges)).tolist()
        raise ValueError(f"CM5 fragments do not cover full molecule atoms: {missing}")
    residual = float(molecular_charge) - float(np.sum(charges))
    if abs(residual) > float(tolerance):
        raise ValueError(f"fragment CM5 sum differs from molecular charge by {residual:+.6f} e")
    # Only numerical quadrature drift is distributed; chemically meaningful
    # fragment charge differences are never silently renormalized.
    charges += residual / int(natoms)
    return charges, tuple(provenance)


def _compile_reference_charge_response(
    coordinates_angstrom: np.ndarray,
    nonbonded_atoms: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compile the reference local split-charge response without graph classes."""

    sites = tuple(
        ChargeResponseSite(
            reference_charge=float(item["atomic_charge"]),
            gaussian_width_bohr=max(
                0.30,
                float(item["oracle_synthon_descriptors"]["effective_covalent_radius_angstrom"])
                / BOHR_TO_ANGSTROM,
            ),
            atom_weights=((int(item["atom"]), 1.0),),
            label=f"ATOM_{int(item['atom']) + 1}",
        )
        for item in nonbonded_atoms
    )
    return {
        "schema": "matrix.zaff.split_charge_response.v1",
        "role": "PRIMARY_ZAFF_POLARIZABLE_RESPONSE",
        "reference_geometry_angstrom": np.asarray(coordinates_angstrom, dtype=float).tolist(),
        "sites": [
            {
                "reference_charge": site.reference_charge,
                "gaussian_width_bohr": site.gaussian_width_bohr,
                "atom_weights": [list(item) for item in site.atom_weights],
                "label": site.label,
            }
            for site in sites
        ],
        "channel_generator": {
            "representation": "IMPLICIT_LOCAL_PAIR_SITE_SCHUR",
            "response_lengths": [
                max(float(item["polarizability_bohr3"]), 1.0e-8) ** (-1.0 / 3.0)
                for item in nonbonded_atoms
            ],
            "rmin_half_angstrom": [float(item["rmin_half_angstrom"]) for item in nonbonded_atoms],
            "hardness_rule": "HALF_SUM_OF_SITE_RESPONSE_LENGTHS",
            "switch_minimum_angstrom": 4.0,
            "switch_scale": 1.25,
            "switch_exponent": 6,
            "channel_cutoff_angstrom": 12.0,
            "taper_start_fraction": 0.75,
            "cutoff_form": "RATIONAL_TIMES_C2_QUINTIC_COMPACT_SUPPORT",
            "pair_chunk_size": 65536,
            "reference_bias_rule": ("REFERENCE_SWITCH_TIMES_REFERENCE_POTENTIAL_DIFFERENCE"),
            "flows": "ELIMINATED_VARIATIONALLY_NOT_MATERIALIZED",
        },
        "site_extension_contract": (
            "AFFINE_LONE_PAIR_RING_INTERACTION_AND_UNITED_ATOM_SITES_SUPPORTED"
        ),
        "channel_policy": "LOCAL_VERLET_PAIRS_WITH_C2_COMPACT_SUPPORT",
        "hardness_prior": "ORACLE_POLARIZABILITY_INVERSE_RESPONSE_LENGTH",
        "decision_record": ("docs/architecture/ADR-0021-ZAFF-CONSTRAINED-CHARGE-RESPONSE.md"),
    }


def _valence_angles(graph) -> tuple[tuple[int, int, int], ...]:
    result = []
    for center in range(graph.natoms):
        neighbors = sorted(int(value) for value in graph.adjacency[center])
        for position, left in enumerate(neighbors):
            for right in neighbors[position + 1 :]:
                result.append((left, center, right))
    return tuple(result)


def _valence_torsions(graph) -> tuple[tuple[int, int, int, int], ...]:
    result: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for middle_left, middle_right in sorted(tuple(sorted(pair)) for pair in graph.bonds):
        for left in sorted(set(graph.adjacency[middle_left]) - {middle_right}):
            for right in sorted(set(graph.adjacency[middle_right]) - {middle_left}):
                raw = (int(left), int(middle_left), int(middle_right), int(right))
                if len(set(raw)) != 4:
                    # A three-membered ring or a fused closure can expose the
                    # same atom at both ends of a nominal torsion.  It is not a
                    # valid four-atom ZAFF torsion and must not reach the
                    # synthon compiler.
                    continue
                key = min(raw, tuple(reversed(raw)))
                if key not in seen:
                    seen.add(key)
                    result.append(key)
    return tuple(result)


def _validated_atomic_charges(values: Sequence[float], natoms: int) -> np.ndarray:
    charges = np.asarray(values, dtype=float).reshape(-1)
    if len(charges) != int(natoms) or np.any(~np.isfinite(charges)):
        raise ValueError("ZAFF seed compilation requires one finite charge per atom")
    total = float(np.sum(charges))
    nearest_molecular_charge = float(round(total))
    if abs(total - nearest_molecular_charge) > 5.0e-4:
        raise ValueError(
            "complete APOC CM5 charges must conserve an integral molecular charge "
            f"(sum={total:+.8f} e)"
        )
    return charges


def _periodic_morse_bond_record(numbers, synthons, left, right, distance):
    """Build and translate the low-confidence periodic-table bond prior."""

    first = zaff_periodic_element_prior(int(numbers[left]))
    second = zaff_periodic_element_prior(int(numbers[right]))
    bond_order = max(float(synthons.bond_order(left, right)), 0.20)
    radius_scale = max(
        (first.covalent_radius_angstrom + second.covalent_radius_angstrom) / 1.50,
        0.35,
    )
    depth = 0.080 * bond_order / math.sqrt(radius_scale)
    curvature = 0.35 * bond_order / (radius_scale * radius_scale)
    alpha = math.sqrt(curvature / (2.0 * depth))
    return _inverse_distance_bond_record({
        "atoms": [int(left), int(right)],
        "d_e_hartree": float(depth),
        "r0_angstrom": float(distance),
        "alpha_per_angstrom": float(alpha),
        "initial_curvature_hartree_per_angstrom2": float(curvature),
        "source": "ZAFF_GFNFF_ANALOGOUS_COMPLETE_PERIODIC_TABLE_MORSE_PRIOR",
        "uncertainty_class": "PERIODIC_TRANSFER_REQUIRES_QM_REFINEMENT",
        "synthon_contributors": [
            {
                "types": [f"Z{numbers[left]}", f"Z{numbers[right]}"],
                "labels": [first.source, second.source],
                "weight": 1.0,
                "descriptor_distance": 0.0,
                "bond_order": bond_order,
            }
        ],
    })


def _inverse_distance_bond_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a library Morse prior to the permanent ZAFF bond form."""

    result = dict(record)
    potential = inverse_power_stretch_from_morse_prior(
        float(result["d_e_hartree"]),
        float(result["r0_angstrom"]),
        float(result["alpha_per_angstrom"]),
    )
    result.update(
        {
            "functional_form": "INVERSE_POWER_STRETCH",
            "depth_hartree": float(potential.depth),
            "exponent": float(potential.exponent),
            "asymptotic_depth_hartree": float(result["d_e_hartree"]),
            "library_prior_form": "MORSE",
            "zaff_radial_variable": "rho=(r0/r)^p",
            "derivative_preservation": "MORSE_K2",
        }
    )
    return result


def _periodic_angle_record(numbers, synthons, xyz, atoms3, types):
    left, center, right = (int(value) for value in atoms3)
    central = zaff_periodic_element_prior(int(numbers[center]))
    order_left = max(float(synthons.bond_order(left, center)), 0.20)
    order_right = max(float(synthons.bond_order(center, right)), 0.20)
    size_scale = max(central.covalent_radius_angstrom / 0.75, 0.35)
    force = 0.080 * math.sqrt(order_left * order_right) / math.sqrt(size_scale)
    return {
        "atoms": [left, center, right],
        "types": list(types),
        "k_hartree_per_radian2": float(force),
        "theta0_radian": math.radians(
            float(synthons.theta_ref(synthons.electron_domains(center)))
        ),
        "radial_factors": [
            _radial_factor_record(synthons, xyz, left, center),
            _radial_factor_record(synthons, xyz, center, right),
        ],
        "source": "ZAFF_GFNFF_ANALOGOUS_COMPLETE_PERIODIC_TABLE_BEND_PRIOR",
        "uncertainty_class": "PERIODIC_TRANSFER_REQUIRES_QM_REFINEMENT",
        "synthon_contributors": [
            {
                "type": f"Z{numbers[center]}:PERIODIC",
                "weight": 1.0,
                "descriptor_distance": 0.0,
                "source": central.source,
            }
        ],
    }


def _positive_weight(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} prior weight must be finite and positive")
    return result


def _radial_factor_record(
    synthons,
    coordinates_angstrom: np.ndarray,
    left: int,
    right: int,
) -> dict[str, float]:
    return {
        "reference_distance_angstrom": float(
            np.linalg.norm(coordinates_angstrom[int(left)] - coordinates_angstrom[int(right)])
        ),
        "covalent_radius_sum_angstrom": (
            float(synthons.covalent_radius_eff(int(left)))
            + float(synthons.covalent_radius_eff(int(right)))
        ),
    }


def _direct_charm_ring_coordinates(
    rings,
    coordinates_angstrom: np.ndarray,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Evaluate SMITH CHARM rows directly on ORACLE-perceived ordered cycles."""

    from matrix_chem import eval_primitive
    from matrix_smith import build_charm_ring_coordinates

    memberships: list[list[dict[str, Any]]] = [[] for _index in range(len(coordinates_angstrom))]
    records: list[dict[str, Any]] = []
    for ring_index, ring in enumerate(rings.rings):
        atoms = tuple(int(atom) for atom in ring.atoms)
        if len(atoms) < 5:
            continue
        rows = build_charm_ring_coordinates(
            atoms,
            coordinates_angstrom,
            prefix=f"Ring{ring_index + 1:04d}RPck",
        )
        component_values: list[float] = []
        component_names: list[str] = []
        serialized_rows: list[dict[str, Any]] = []
        for row in rows:
            value = float(
                sum(
                    float(coefficient) * float(eval_primitive(primitive, coordinates_angstrom))
                    for coefficient, primitive in row.terms
                )
            )
            component_values.append(value)
            component_names.append(str(row.name))
            serialized_rows.append(
                {
                    "name": str(row.name),
                    "family": "RING_PUCKER_COMPONENT",
                    "value_angstrom": value,
                    "terms": [
                        {
                            "coefficient": float(coefficient),
                            "function": str(primitive.kind),
                            "atoms_zero_based": [int(atom) for atom in primitive.atoms],
                        }
                        for coefficient, primitive in row.terms
                    ],
                    "diagnostic": str(row.diagnostic),
                }
            )
        amplitude = float(np.linalg.norm(component_values))
        record = {
            "ring_index": ring_index,
            "ring_atoms_zero_based": list(atoms),
            "model": "CHARM",
            "coordinate_owner": "SMITH",
            "cycle_perception_owner": "ORACLE",
            "coordinate_count": len(rows),
            "coordinates": serialized_rows,
            "amplitude_angstrom": amplitude,
        }
        records.append(record)
        membership = {
            "ring_index": ring_index,
            "coordinate_names": component_names,
            "component_values_angstrom": component_values,
            "amplitude_angstrom": amplitude,
            "coordinate_owner": "SMITH",
        }
        for atom in atoms:
            memberships[atom].append(dict(membership))
    return records, memberships


_GAFF_NONBONDED_PROTOTYPES: dict[str, tuple[int, float, float, float]] = {
    "c": (6, 3.0, 1.0, 0.45),
    "c1": (6, 2.0, 2.0, 0.00),
    "c2": (6, 3.0, 1.0, 0.00),
    "c3": (6, 4.0, 0.0, 0.00),
    "ca": (6, 3.0, 1.0, 0.00),
    "hc": (1, 1.0, 0.0, 0.05),
    "h1": (1, 1.0, 0.0, 0.10),
    "ha": (1, 1.0, 0.0, 0.05),
    "hn": (1, 1.0, 0.0, 0.25),
    "ho": (1, 1.0, 0.0, 0.35),
    "hp": (1, 1.0, 0.0, 0.20),
    "hs": (1, 1.0, 0.0, 0.15),
    "n": (7, 3.0, 1.0, -0.30),
    "n2": (7, 2.0, 1.0, -0.20),
    "n3": (7, 3.0, 0.0, -0.25),
    "n4": (7, 4.0, 0.0, 0.20),
    "na": (7, 3.0, 1.0, -0.15),
    "nb": (7, 2.0, 1.0, -0.20),
    "o": (8, 1.0, 1.0, -0.45),
    "oh": (8, 2.0, 0.0, -0.35),
    "os": (8, 2.0, 0.0, -0.30),
    "p2": (15, 2.0, 1.0, 0.00),
    "p3": (15, 3.0, 0.0, 0.00),
    "p4": (15, 3.0, 1.0, 0.35),
    "p5": (15, 4.0, 0.0, 0.50),
    "pb": (15, 2.0, 1.0, 0.00),
    "s": (16, 1.0, 1.0, -0.20),
    "s2": (16, 2.0, 1.0, 0.00),
    "s4": (16, 3.0, 1.0, 0.35),
    "s6": (16, 4.0, 0.0, 0.50),
    "sh": (16, 2.0, 0.0, -0.20),
    "ss": (16, 2.0, 0.0, 0.00),
    "sx": (16, 3.0, 1.0, 0.35),
    "sy": (16, 4.0, 0.0, 0.50),
}


def _gaff_synthon_type_mixture(
    gaff: GaffParameterSet,
    atomic_number_value: int,
    coordination: float,
    pi_character: float,
    cm5_charge: float,
) -> tuple[dict[str, float | str], ...]:
    """Return continuous memberships in every element-compatible donor type."""

    candidate_types = tuple(
        atom_type
        for atom_type, prototype in _GAFF_NONBONDED_PROTOTYPES.items()
        if prototype[0] == int(atomic_number_value)
    )
    candidates = []
    for atom_type in candidate_types:
        prototype = _GAFF_NONBONDED_PROTOTYPES.get(atom_type)
        if prototype is None or prototype[0] != int(atomic_number_value):
            continue
        try:
            gaff.vdw_for(atom_type)
        except KeyError:
            continue
        distance = (
            ((float(coordination) - prototype[1]) / 0.75) ** 2
            + ((float(pi_character) - prototype[2]) / 0.60) ** 2
            + ((float(cm5_charge) - prototype[3]) / 0.35) ** 2
        )
        candidates.append((distance, atom_type))
    if not candidates:
        return ()
    minimum = min(item[0] for item in candidates)
    raw = [math.exp(-0.5 * (item[0] - minimum)) for item in candidates]
    total = sum(raw)
    weights = [value / total for value in raw]
    return tuple(
        {
            "type": item[1],
            "weight": float(weight),
            "descriptor_distance": float(item[0]),
        }
        for weight, item in zip(weights, candidates, strict=True)
    )


def _interpolate_gaff_nonbonded(
    gaff: GaffParameterSet,
    mixture: Sequence[Mapping[str, float | str]],
) -> tuple[tuple[float, float], list[dict[str, float | str]]]:
    contributions = [dict(item) for item in mixture]
    if not contributions:
        raise KeyError("GAFF has no element-compatible nonbonded donor")
    parameters = [gaff.vdw_for(str(item["type"])) for item in contributions]
    rmin_half = sum(
        float(item["weight"]) * parameter.rmin_half_angstrom
        for item, parameter in zip(contributions, parameters, strict=True)
    )
    epsilon = sum(
        float(item["weight"]) * parameter.epsilon_kcal_per_mol
        for item, parameter in zip(contributions, parameters, strict=True)
    )
    return (float(rmin_half), float(epsilon)), contributions


def _interpolate_gaff_angle(
    gaff: GaffParameterSet,
    mixtures: Sequence[Sequence[Mapping[str, float | str]]],
):
    candidates = []
    for combination in product(*mixtures):
        types = tuple(str(item["type"]) for item in combination)
        weight = math.prod(float(item["weight"]) for item in combination)
        try:
            parameter = gaff.angle(*types)
        except (KeyError, ValueError):
            continue
        candidates.append((weight, types, parameter))
    total = sum(item[0] for item in candidates)
    if total <= 0.0:
        raise KeyError("no GAFF angle donor combination is available")
    normalized = [(weight / total, types, parameter) for weight, types, parameter in candidates]
    prototype = normalized[0][2]
    parameter = type(prototype)(
        prototype.types,
        sum(
            weight * item.force_constant_kcal_per_mol_radian2 for weight, _types, item in normalized
        ),
        sum(weight * item.equilibrium_degrees for weight, _types, item in normalized),
    )
    contributors = [
        {"types": list(types), "weight": float(weight)} for weight, types, _item in normalized
    ]
    return parameter, contributors


__all__ = [
    "ZAFF_FAST_RIGID_TORSIONAL_ZOOM",
    "ZAFF_SEED_MODEL_SCHEMA",
    "ZAFF_SEED_PRIOR_SCHEMA",
    "ZaffSeedCoverageError",
    "assemble_fragment_cm5_charges",
    "build_zaff_seed_force_field",
    "compile_zaff_seed_model",
    "zaff_seed_full_fit_priors",
]
