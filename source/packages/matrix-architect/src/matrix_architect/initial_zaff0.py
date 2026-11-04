"""Final initial-geometry and coherent ZAFF0 construction protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from matrix_chem import (
    build_topology_objects,
    read_primitive_contract,
    read_xyz,
    write_xyz,
)
from matrix_chem.topology.elements import atomic_number
from matrix_core import ScientificStateManifest, canonical_sha256
from matrix_link import (
    QMScanBackend,
    OptimizerSettings,
    coordinate_model_from_xyzin,
    optimize_geometry,
)
from matrix_oracle import (
    InitialStructurePreparation,
    OracleElectronicPopulationWorkflow,
    PopulationFragmentResult,
    analyze_structure,
    assemble_overlapping_cm5_mayer,
    prepare_initial_structure,
)
from matrix_smith import (
    freeze_sonic_definition_from_xyzin,
    is_total_symmetric_irrep,
    recenter_sonic_definition_from_xyzin,
    sonic_definition_identity_sha256,
)

from .model import save_force_field
from .protocol import GeometrySource, ZaffFieldLevel, ZaffProtocolManifest
from .seed_compiler import build_zaff_seed_force_field


INITIAL_ZAFF0_SCHEMA = "matrix.architect.initial_zaff0.v1"
WHOLE_MOLECULE_L0_MAX_HEAVY_ATOMS = 60
SOFT_EXOCYCLIC_MAYER_ORDER_LIMIT = 1.25
ZAFF_FAST_MAX_BOND_DRIFT_ANGSTROM = 0.002
ZAFF_FAST_MAX_ANGLE_DRIFT_DEGREES = 0.2
# Backward-compatible name retained for callers of the first implementation.
WHOLE_MOLECULE_L0_MAX_SMILES_ATOMS = WHOLE_MOLECULE_L0_MAX_HEAVY_ATOMS


@dataclass(frozen=True)
class InitialZaff0Build:
    schema: str
    source: str
    geometry_xyz: str
    force_field: str
    manifest: str
    population_mode: str
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_initial_geometry_and_zaff0(
    source: Path | str,
    output_xyz: Path | str,
    *,
    lcb26_root: Path | str,
    population_workflow: OracleElectronicPopulationWorkflow | None = None,
    population_fragments: Sequence[PopulationFragmentResult] = (),
    population_strategy: str = "qm",
    charge: int = 0,
    multiplicity: int = 1,
    source_kind: str = "auto",
    declared_level: str = "AUTO",
    prepared_initial: InitialStructurePreparation | None = None,
) -> InitialZaff0Build:
    """Close the starting geometry, resolve its populations, and build ZAFF0.

    The SMILES or XYZ geometry is finalized once by the frozen LCB26
    atom/bond/angle and weighted-B protocol.  Up to 60 heavy atoms, one
    whole-molecule L0
    CM5/Mayer calculation is then performed on that final geometry.  Larger
    systems preserve a validated overlapping-fragment population assembly.
    ``population_strategy="empirical"`` instead asks ORACLE for its complete
    electronegativity/Pauling fallback and performs no electronic-structure
    calculation.  The resulting immutable population state is the sole input
    used to recalculate synthons and compile the complete diagonal ZAFF0 field.
    """

    target = Path(output_xyz).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    source_text = str(source)
    requested_kind = str(source_kind).strip().casefold()
    source_path = Path(source_text).expanduser()
    resolved_kind = (
        "xyz"
        if requested_kind == "auto" and source_path.is_file()
        else "smiles"
        if requested_kind == "auto"
        else requested_kind
    )
    if resolved_kind not in {"smiles", "xyz", "geometry", "enriched_xyz"}:
        raise ValueError(f"unsupported initial ZAFF0 source kind: {source_kind!r}")
    if prepared_initial is None:
        initial = prepare_initial_structure(
            source_text,
            target,
            lcb26_root=lcb26_root,
            source_kind=resolved_kind,
            declared_level=declared_level,
        )
    else:
        initial = _validate_prepared_initial(
            prepared_initial,
            source=source_text,
            target=target,
            source_kind=resolved_kind,
            declared_level=declared_level,
        )
    geometry = read_xyz(initial.output_xyz)
    sonic_definition = freeze_sonic_definition_from_xyzin(
        Path(initial.output_xyzin),
        symmetrize=True,
        separate_exocyclic_torsions=True,
    )
    source_heavy_atom_count = sum(atom != "H" for atom in geometry.atoms)
    strategy = str(population_strategy).strip().casefold().replace("-", "_")
    if strategy in {"qm", "cm5_mayer", "qm_cm5_mayer"}:
        strategy = "qm"
    elif strategy in {"empirical", "electronegativity_pauling", "geometry_only"}:
        strategy = "empirical"
    else:
        raise ValueError("population_strategy must be 'qm' or 'empirical'")
    if strategy == "empirical":
        if population_workflow is not None or population_fragments:
            raise ValueError(
                "empirical ZAFF0 cannot be mixed with a QM population workflow or fragments"
            )
        population_mode = "ORACLE_ELECTRONEGATIVITY_PAULING"
    else:
        population_mode = (
            "FINAL_GEOMETRY_WHOLE_MOLECULE_L0"
            if source_heavy_atom_count <= WHOLE_MOLECULE_L0_MAX_HEAVY_ATOMS
            else "OVERLAPPING_L0_FRAGMENTS_PRESERVED"
        )

    if population_mode == "ORACLE_ELECTRONEGATIVITY_PAULING":
        final_charges, final_orders = _oracle_empirical_population_state(geometry)
        if abs(float(charge)) > 1.0e-12:
            raise ValueError(
                "empirical ZAFF0 currently requires a neutral structure; charged "
                "systems require complete CM5/Mayer data"
            )
        population_provenance = {
            "mode": population_mode,
            "owner": "ORACLE",
            "charge_model": "electronegativity estimate",
            "bond_order_model": "Pauling estimate",
            "geometry_specific": True,
            "geometry_sha256": _sha256(target),
            "qm_calculation_count": 0,
        }
    elif population_mode == "FINAL_GEOMETRY_WHOLE_MOLECULE_L0":
        if population_workflow is None:
            raise ValueError(
                "whole-molecule finalization requires an ORACLE L0 population workflow"
            )
        if population_fragments:
            raise ValueError("whole-molecule L0 and fragment populations cannot be mixed")
        calculation = population_workflow.calculate(
            "initial-zaff0-preconformer-geometry",
            geometry.atoms,
            geometry.coordinates_angstrom,
            charge=int(charge),
            multiplicity=int(multiplicity),
            accuracy_level="L0",
        )
        final_charges = np.asarray(calculation.observables.cm5_charges, dtype=float)
        final_orders = _covalent_mayer_map(geometry, calculation.observables.mayer_bond_orders)
        population_provenance = {
            "mode": population_mode,
            "backend": calculation.backend,
            "request": calculation.request,
            "cache_key": calculation.cache_key,
            "geometry_specific": True,
            "geometry_sha256": _sha256(target),
        }
    else:
        if population_workflow is not None:
            raise ValueError(
                "large-system fragment mode cannot also use whole-molecule populations"
            )
        if not population_fragments:
            raise ValueError(
                "large-system finalization requires validated overlapping L0 fragments"
            )
        if len(population_fragments) < 2:
            raise ValueError(
                "large-system finalization requires at least two overlapping L0 fragments"
            )
        assembly = assemble_overlapping_cm5_mayer(
            len(geometry.atoms),
            _perceived_bonds(geometry),
            population_fragments,
            total_charge_e=float(charge),
        )
        final_charges = np.asarray(assembly.cm5_charges_e, dtype=float)
        final_orders = dict(assembly.mayer_bond_orders)
        population_provenance = {
            "mode": population_mode,
            "fragment_ids": list(assembly.fragment_ids),
            "level": assembly.level,
            "maximum_total_charge_correction_e": (assembly.maximum_total_charge_correction_e),
            "geometry_specific": False,
            "overlap_preserved": True,
        }

    _validate_population_state(
        geometry,
        final_charges,
        final_orders,
        molecular_charge=float(charge),
    )
    seed_field = _compile_population_field(
        geometry,
        final_charges,
        final_orders,
        charge=charge,
        multiplicity=multiplicity,
        population_mode=population_mode,
        population_provenance=population_provenance,
        geometry_role="preconformer geometry",
    )
    seed_force_field_path = target.with_suffix(".zaff-fast.seed.json")
    save_force_field(seed_force_field_path, seed_field)
    relaxation = _relax_zaff_fast_exocyclic_torsions(
        geometry,
        Path(initial.output_xyzin),
        target,
        seed_force_field_path,
        final_orders,
        sonic_definition,
        bond_order_source=(
            "ORACLE Pauling estimate"
            if population_mode == "ORACLE_ELECTRONEGATIVITY_PAULING"
            else "ORACLE Mayer"
        ),
    )
    geometry = read_xyz(target)
    if population_mode == "ORACLE_ELECTRONEGATIVITY_PAULING":
        final_charges, final_orders = _oracle_empirical_population_state(geometry)
        population_provenance.update(
            {
                "geometry_sha256": _sha256(target),
                "perception_count": 2 if relaxation["geometry_changed"] else 1,
                "preconformer_is_final_geometry": not relaxation["geometry_changed"],
            }
        )
    elif population_mode == "FINAL_GEOMETRY_WHOLE_MOLECULE_L0" and relaxation["geometry_changed"]:
        preconformer = dict(population_provenance)
        calculation = population_workflow.calculate(
            "initial-zaff0-final-geometry",
            geometry.atoms,
            geometry.coordinates_angstrom,
            charge=int(charge),
            multiplicity=int(multiplicity),
            accuracy_level="L0",
        )
        final_charges = np.asarray(calculation.observables.cm5_charges, dtype=float)
        final_orders = _covalent_mayer_map(geometry, calculation.observables.mayer_bond_orders)
        population_provenance = {
            "mode": population_mode,
            "backend": calculation.backend,
            "request": calculation.request,
            "cache_key": calculation.cache_key,
            "geometry_specific": True,
            "geometry_sha256": _sha256(target),
            "calculation_count": 2,
            "preconformer_calculation": preconformer,
        }
    elif population_mode == "FINAL_GEOMETRY_WHOLE_MOLECULE_L0":
        population_provenance["calculation_count"] = 1
        population_provenance["preconformer_is_final_geometry"] = True

    _validate_population_state(
        geometry,
        final_charges,
        final_orders,
        molecular_charge=float(charge),
    )
    field = _compile_population_field(
        geometry,
        final_charges,
        final_orders,
        charge=charge,
        multiplicity=multiplicity,
        population_mode=population_mode,
        population_provenance=population_provenance,
        geometry_role="final geometry",
    )
    scientific_state = _initial_zaff_protocol_manifest(
        source=source_text,
        target=target,
        xyzin_path=Path(initial.output_xyzin),
        field=field,
        charges=final_charges,
        orders=final_orders,
        charge=charge,
        population_mode=population_mode,
        population_provenance=population_provenance,
        sonic_identity=str(relaxation["frozen_sonic_identity_sha256"]),
    )
    pipeline_state = _initial_zaff_pipeline_state(
        source=source_text,
        source_kind=resolved_kind,
        declared_level=declared_level,
        lcb26_root=Path(lcb26_root),
        target=target,
        initial=initial,
        scientific_state=scientific_state,
        charges=final_charges,
        orders=final_orders,
        charge=charge,
        multiplicity=multiplicity,
        population_mode=population_mode,
    )
    field = replace(
        field,
        metadata={
            **dict(field.metadata),
            "zaff_scientific_state": scientific_state.to_dict(),
            "pipeline_scientific_state": pipeline_state.to_dict(),
        },
    )
    force_field_path = target.with_suffix(".zaff0.json")
    save_force_field(force_field_path, field)
    manifest_path = target.with_suffix(".initial_zaff0.json")
    result = InitialZaff0Build(
        schema=INITIAL_ZAFF0_SCHEMA,
        source=source_text,
        geometry_xyz=str(target),
        force_field=str(force_field_path),
        manifest=str(manifest_path),
        population_mode=population_mode,
        complete=True,
    )
    manifest = {
        **result.to_dict(),
        "protocol": {
            "whole_molecule_heavy_atom_limit": WHOLE_MOLECULE_L0_MAX_HEAVY_ATOMS,
            "source_kind": resolved_kind,
            "declared_structure_level": str(declared_level),
            "geometry": "LCB26_NEAREST_ATOM_BOND_ANGLE_THEN_WEIGHTED_WILSON_B",
            "geometry_reopened_after_finalization": False,
            "prepopulation_conformer_stage": ("SMITH_SONIC_SOFT_EXOCYCLIC_ONLY_LINK_ZAFF_FAST"),
            "population": population_mode,
            "synthon_recalculation": (
                "ARCHITECT_FROM_FINAL_GEOMETRY_ELECTRONEGATIVITY_PAULING"
                if population_mode == "ORACLE_ELECTRONEGATIVITY_PAULING"
                else "ARCHITECT_FROM_FINAL_GEOMETRY_CM5_MAYER"
            ),
            "field": "ZAFF0_DIAGONAL_COMPLETE",
            "mixed_population_levels_forbidden": True,
        },
        "geometry_transfer_reliability": _geometry_transfer_reliability(initial),
        "zaff_fast_relaxation": relaxation,
        "zaff_fast_seed_force_field": str(seed_force_field_path),
        "zaff_fast_seed_force_field_sha256": _sha256(seed_force_field_path),
        "population_provenance": population_provenance,
        "zaff_scientific_state": scientific_state.to_dict(),
        "pipeline_scientific_state": pipeline_state.to_dict(),
        "geometry_sha256": _sha256(target),
        "force_field_sha256": _sha256(force_field_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _initial_zaff_pipeline_state(
    *,
    source,
    source_kind,
    declared_level,
    lcb26_root,
    target,
    initial,
    scientific_state,
    charges,
    orders,
    charge,
    multiplicity,
    population_mode,
) -> ScientificStateManifest:
    """Bind every reusable downstream artifact to its exact upstream state."""

    lcb26_manifest = Path(lcb26_root).expanduser().resolve() / "manifest.json"
    inputs = {
        "final_geometry": _sha256(Path(target)),
        "electronic_population": canonical_sha256(
            {
                "cm5_charges_e": np.asarray(charges, dtype=float).tolist(),
                "mayer_bond_orders": [
                    [int(left), int(right), float(value)]
                    for (left, right), value in sorted(orders.items())
                ],
            }
        ),
        "oracle_initial_structure_report": _sha256(Path(initial.report)),
    }
    if lcb26_manifest.is_file():
        inputs["lcb26_manifest"] = _sha256(lcb26_manifest)
    source_path = Path(str(source)).expanduser()
    inputs["source"] = (
        _sha256(source_path.resolve())
        if source_path.is_file()
        else canonical_sha256({"kind": str(source_kind), "value": str(source)})
    )
    protocol = scientific_state.to_dict()
    return ScientificStateManifest(
        stage="architect.initial_zaff0",
        inputs=inputs,
        contracts={
            "zaff_protocol": canonical_sha256(protocol),
            "primitive_coordinates": protocol["primitive_contract_hash"],
            "sonic_coordinates": protocol["sonic_contract_hash"],
            "ring_coordinates": protocol["ring_contract_hash"],
        },
        parameters={
            "charge": int(charge),
            "multiplicity": int(multiplicity),
            "source_kind": str(source_kind),
            "declared_level": str(declared_level),
            "population_mode": str(population_mode),
            "geometry_reopened_after_finalization": False,
        },
    )


def _initial_zaff_protocol_manifest(
    *,
    source,
    target,
    xyzin_path,
    field,
    charges,
    orders,
    charge,
    population_mode,
    population_provenance,
    sonic_identity,
):
    model = dict(field.metadata["diagonal_seed_model"])
    whole = population_mode == "FINAL_GEOMETRY_WHOLE_MOLECULE_L0"
    empirical = population_mode == "ORACLE_ELECTRONEGATIVITY_PAULING"
    population_source = (
        "ORACLE_ELECTRONEGATIVITY_PAULING_FINAL_GEOMETRY"
        if empirical
        else "APOC_WHOLE_MOLECULE_L0_FINAL_GEOMETRY"
        if whole
        else "ORACLE_VALIDATED_OVERLAPPING_L0_FRAGMENT_ASSEMBLY"
    )
    primitive = read_primitive_contract(Path(xyzin_path))
    return ZaffProtocolManifest(
        molecule_id=str(source),
        geometry_source=GeometrySource.LCB26,
        geometry_hash=_sha256(Path(target)),
        synthon_hash=_json_sha256(
            {
                "synthon_type_keys": model["synthon_type_keys"],
                "electronic_observables": model["electronic_observables"],
            }
        ),
        cm5_provenance={
            "kind": "ELECTRONEGATIVITY_ESTIMATE" if empirical else "CM5",
            "owner": "ORACLE" if empirical else "APOC",
            "source": population_source,
            "molecular_charge": int(charge),
            "details": dict(population_provenance),
        },
        mayer_provenance={
            "kind": "PAULING_ESTIMATE" if empirical else "MAYER",
            "owner": "ORACLE" if empirical else "APOC",
            "total_source": population_source,
            "component_source": (
                "ORACLE_GEOMETRY_BASED_PAULING_COMPONENTS"
                if empirical
                else "ARCHITECT_SYNTHON_DECOMPOSITION_FROM_TOTAL_MAYER"
            ),
            "details": dict(population_provenance),
        },
        available_levels=(ZaffFieldLevel.ZERO,),
        skipped_stages=("ZAFF1_NO_QM_HESSIAN", "ZAFF2_NO_ANHARMONIC_DATA"),
        primitive_contract_hash=str(primitive.b_matrix_sha256),
        sonic_contract_hash=str(sonic_identity),
        ring_contract_hash=_json_sha256(model.get("ring_coordinates", ())),
        acceptance_diagnostics={
            "charge_conservation": bool(abs(float(np.sum(charges)) - float(charge)) <= 2.0e-6),
            "topology_synthon_identity": bool(
                len(model["atoms"]) == len(charges)
                and all(tuple(pair) in orders for pair in map(tuple, model["topology_bonds"]))
            ),
            "derivative_checks": "ZAFF0_ANALYTIC_RUNTIME_VALIDATED",
        },
    )


def _json_sha256(payload) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_prepared_initial(
    initial: InitialStructurePreparation,
    *,
    source: str,
    target: Path,
    source_kind: str,
    declared_level: str,
) -> InitialStructurePreparation:
    """Accept only an intact ORACLE preparation for the requested target.

    This hand-off prevents Keymaker from asking ORACLE to repeat the weighted-B
    geometry construction merely because ARCHITECT is now completing the L0
    population and ZAFF0 stages.
    """

    if not isinstance(initial, InitialStructurePreparation):
        raise TypeError("prepared_initial must be an ORACLE InitialStructurePreparation")
    if Path(initial.output_xyz).expanduser().resolve() != target:
        raise ValueError("prepared_initial output does not match the requested ZAFF0 target")
    if str(initial.source) != source:
        raise ValueError("prepared_initial source does not match the requested source")
    if str(initial.source_kind).casefold() != str(source_kind).casefold():
        raise ValueError("prepared_initial source kind does not match the requested source kind")
    requested_level = str(declared_level).strip().upper().replace(" ", "_")
    if str(initial.declared_level).upper() != requested_level:
        raise ValueError("prepared_initial level does not match the requested structure level")
    required = (initial.output_xyz, initial.output_xyzin, initial.report)
    if any(not Path(path).expanduser().is_file() for path in required):
        raise ValueError("prepared_initial is incomplete or its artifacts were removed")
    return initial


def _compile_population_field(
    geometry,
    charges,
    orders,
    *,
    charge,
    multiplicity,
    population_mode,
    population_provenance,
    geometry_role,
):
    whole = population_mode == "FINAL_GEOMETRY_WHOLE_MOLECULE_L0"
    empirical = population_mode == "ORACLE_ELECTRONEGATIVITY_PAULING"
    return build_zaff_seed_force_field(
        geometry.atoms,
        geometry.coordinates_angstrom,
        None if empirical else charges,
        mayer_bond_orders=None if empirical else orders,
        charge=int(charge),
        multiplicity=int(multiplicity),
        charge_source=(
            "ORACLE electronegativity estimate"
            if empirical
            else f"ORACLE L0 CM5 {geometry_role}"
            if whole
            else "ORACLE assembled overlapping-fragment L0 CM5"
        ),
        bond_order_source=(
            "ORACLE Pauling estimate"
            if empirical
            else f"ORACLE L0 Mayer {geometry_role}"
            if whole
            else "ORACLE assembled overlapping-fragment L0 Mayer"
        ),
        charge_owner="ORACLE",
        bond_order_owner="ORACLE",
        fragment_provenance=(
            ()
            if whole or empirical
            else tuple(
                {"identifier": identifier, "population_level": "L0"}
                for identifier in population_provenance["fragment_ids"]
            )
        ),
    )


def _oracle_empirical_population_state(geometry):
    """Return ORACLE's complete no-QM charge/order fallback at one geometry."""

    numbers = tuple(int(atomic_number(atom) or 0) for atom in geometry.atoms)
    _continuous, graph, _rings, synthons, _aromaticity = build_topology_objects(
        geometry.coordinates_angstrom,
        numbers,
    )
    charges = np.asarray(
        [float(synthons.charge(index)) for index in range(len(numbers))],
        dtype=float,
    )
    orders = {
        tuple(sorted(map(int, pair))): float(synthons.bond_order(*pair))
        for pair in graph.bonds
    }
    return charges, orders


def _relax_zaff_fast_exocyclic_torsions(
    geometry,
    xyzin_path,
    output_xyz,
    seed_force_field_path,
    mayer_orders,
    sonic_definition,
    *,
    bond_order_source="ORACLE Mayer",
):
    """Let LINK move only SMITH SONICs for soft exocyclic torsions."""

    selected = _soft_exocyclic_sonics(sonic_definition, mayer_orders)
    base = {
        "coordinate_owner": "SMITH",
        "optimizer_owner": "LINK",
        "energy_owner": "ZAFF",
        "torsion_classification_source": "SMITH_SONIC_FAMILY",
        "sonic_definition_built_once": True,
        "frozen_sonic_identity_sha256": sonic_definition_identity_sha256(sonic_definition),
        "sonic_coordinate_count": len(sonic_definition.gics),
        "zoom_level": "FAST_RIGID_TORSIONAL",
        "central_mayer_order_limit": SOFT_EXOCYCLIC_MAYER_ORDER_LIMIT,
        "central_bond_order_limit": SOFT_EXOCYCLIC_MAYER_ORDER_LIMIT,
        "bond_order_source": str(bond_order_source),
        "active_sonic_labels": [item["label"] for item in selected],
        "active_torsions": selected,
        "active_torsion_count": len(selected),
        "inactive_sonics_frozen": True,
        "endocyclic_torsions_active": False,
    }
    if not selected:
        return {
            **base,
            "status": "NOT_APPLICABLE_NO_SOFT_EXOCYCLIC_TORSION",
            "geometry_changed": False,
            "maximum_cartesian_displacement_angstrom": 0.0,
            "maximum_bond_drift_angstrom": 0.0,
            "maximum_angle_drift_degrees": 0.0,
        }

    run_dir = Path(output_xyz).with_suffix("").parent / (Path(output_xyz).stem + ".zaff-fast-link")
    coordinate_model = coordinate_model_from_xyzin(
        xyzin_path,
        kind="sonic",
        coordinates=tuple(item["label"] for item in selected),
        sonic_definition=sonic_definition,
    )
    result = optimize_geometry(
        xyzin_path,
        run_dir=run_dir,
        coordinate_model=coordinate_model,
        backend=QMScanBackend(
            name="zaff",
            force_field=Path(seed_force_field_path),
            properties=("energy", "gradient"),
            zaff_zoom_level="FAST_RIGID_TORSIONAL",
        ),
        settings=OptimizerSettings(
            # Large initial torsional moves need a bounded polishing phase;
            # retain the optimizer's existing force/energy/displacement gates.
            max_steps=100,
            freeze_inactive_sonic=True,
            require_exact_final_hessian=False,
            verify_irc=False,
        ),
    )
    if not result.converged:
        raise RuntimeError(
            f"LINK did not converge the ZAFF-fast soft-exocyclic SONIC stage: {result.status}"
        )
    final_coordinates = np.asarray(result.final_coordinates_angstrom, dtype=float)
    topology_before = set(_perceived_bonds(geometry))
    raw_bond_drift, raw_angle_drift = _bond_angle_drift(
        geometry.coordinates_angstrom,
        final_coordinates,
        topology_before,
        len(geometry.atoms),
    )
    cartesian_projection_used = (
        raw_bond_drift > ZAFF_FAST_MAX_BOND_DRIFT_ANGSTROM
        or raw_angle_drift > ZAFF_FAST_MAX_ANGLE_DRIFT_DEGREES
    )
    if cartesian_projection_used:
        # A frozen SOOP chart is a local tangent chart.  If its coupled
        # back-transformation distorts nominally frozen valence coordinates,
        # retain LINK's optimized torsion targets but realize them by finite
        # Cartesian rotations of the complete graph fragments on either side
        # of each exocyclic bond.  This is the fragment-SYCART fallback: it
        # preserves every bond length and valence angle by construction and
        # never substitutes a legacy U/H coordinate.
        final_coordinates = _cartesian_rigid_torsion_projection(
            np.asarray(geometry.coordinates_angstrom, dtype=float),
            final_coordinates,
            topology_before,
            selected,
        )
    final_geometry = type(geometry)(
        atoms=geometry.atoms,
        coordinates_angstrom=final_coordinates,
        comment=geometry.comment,
        charge=geometry.charge,
        multiplicity=geometry.multiplicity,
        source_format=geometry.source_format,
        metadata=geometry.metadata,
    )
    topology_after = set(_perceived_bonds(final_geometry))
    if topology_after != topology_before:
        raise RuntimeError(
            "ZAFF-fast torsional relaxation changed molecular constitution: "
            f"missing={sorted(topology_before - topology_after)}, "
            f"spurious={sorted(topology_after - topology_before)}"
        )
    maximum_bond_drift, maximum_angle_drift = _bond_angle_drift(
        geometry.coordinates_angstrom,
        final_coordinates,
        topology_before,
        len(geometry.atoms),
    )
    if maximum_bond_drift > ZAFF_FAST_MAX_BOND_DRIFT_ANGSTROM:
        raise RuntimeError(
            "ZAFF-fast torsional relaxation changed a frozen bond by "
            f"{maximum_bond_drift:.6g} angstrom"
        )
    if maximum_angle_drift > ZAFF_FAST_MAX_ANGLE_DRIFT_DEGREES:
        raise RuntimeError(
            "ZAFF-fast torsional relaxation changed a frozen angle by "
            f"{maximum_angle_drift:.6g} degrees"
        )
    displacement = np.linalg.norm(final_coordinates - geometry.coordinates_angstrom, axis=1)
    maximum_displacement = float(np.max(displacement, initial=0.0))
    write_xyz(
        output_xyz,
        geometry.atoms,
        final_coordinates,
        comment="MATRIX initial structure after ZAFF-fast soft-exocyclic relaxation",
    )
    # The downstream ORACLE contract must describe the post-ZAFF-fast geometry,
    # never the preconformer coordinates used to start LINK.
    analyze_structure(output_xyz, xyzin_path, source_kind="xyz")
    recentered = recenter_sonic_definition_from_xyzin(Path(xyzin_path), sonic_definition)
    if sonic_definition_identity_sha256(recentered) != (
        sonic_definition_identity_sha256(sonic_definition)
    ):
        raise RuntimeError("SMITH changed frozen SONIC identity while recentering R0")
    return {
        **base,
        "status": "CONVERGED",
        "coordinate_fallback": (
            "FRAGMENT_SYCART_RIGID_TORSION_PROJECTION"
            if cartesian_projection_used
            else "NONE"
        ),
        "raw_sonic_maximum_bond_drift_angstrom": raw_bond_drift,
        "raw_sonic_maximum_angle_drift_degrees": raw_angle_drift,
        "optimizer_status": result.status,
        "optimizer_summary": str(result.summary_path),
        "optimizer_trace": str(result.trace_path),
        "initial_energy_hartree": (
            float(result.iterations[0].energy_hartree)
            if result.iterations
            else float(result.final_energy_hartree)
        ),
        "final_energy_hartree": float(result.final_energy_hartree),
        "geometry_changed": maximum_displacement > 1.0e-10,
        "maximum_cartesian_displacement_angstrom": maximum_displacement,
        "maximum_bond_drift_angstrom": maximum_bond_drift,
        "maximum_angle_drift_degrees": maximum_angle_drift,
        "topology_gate": "PASS",
    }


def _soft_exocyclic_sonics(definition, mayer_orders):
    """Filter the exocyclic TORSION family already classified by SMITH."""

    primitives = {primitive.identifier: primitive for primitive in definition.primitives}
    selected = []
    for gic in definition.gics:
        if not is_total_symmetric_irrep(definition.point_group, gic.irrep):
            continue
        primitive = primitives.get(gic.primitive_id)
        if (
            gic.family != "TORSION"
            or primitive is None
            or primitive.function != "D"
            or len(primitive.atoms) != 4
        ):
            continue
        atoms = tuple(int(atom) - 1 for atom in primitive.atoms)
        central = tuple(sorted((atoms[1], atoms[2])))
        order = mayer_orders.get(central)
        if order is None or not np.isfinite(float(order)):
            raise ValueError(f"soft-torsion classification lacks Mayer order for bond {central}")
        if float(order) >= SOFT_EXOCYCLIC_MAYER_ORDER_LIMIT:
            continue
        selected.append(
            {
                "label": str(gic.identifier),
                "name": str(gic.name),
                "atoms_zero_based": list(atoms),
                "central_bond_zero_based": list(central),
                "central_mayer_order": float(order),
                "ring_type": "exocyclic",
            }
        )
    return selected


def _cartesian_rigid_torsion_projection(reference, targets, bonds, selected):
    """Realize optimized exocyclic dihedrals as exact rigid Cartesian rotations."""

    coordinates = np.asarray(reference, dtype=float).copy()
    target_coordinates = np.asarray(targets, dtype=float)
    adjacency = {atom: set() for atom in range(len(coordinates))}
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    handled = set()
    for record in selected:
        first, left, right, fourth = map(int, record["atoms_zero_based"])
        central = tuple(sorted((left, right)))
        if central in handled:
            continue
        handled.add(central)
        mobile = _bond_side_atoms(adjacency, seed=fourth, blocked_edge=central)
        if left in mobile or right not in mobile:
            mobile = _bond_side_atoms(adjacency, seed=first, blocked_edge=central)
            left, right = right, left
            first, fourth = fourth, first
        if left in mobile or right not in mobile:
            raise RuntimeError("exocyclic Cartesian fallback found a cyclic central bond")
        current = _dihedral_radian(coordinates[first], coordinates[left], coordinates[right], coordinates[fourth])
        target = _dihedral_radian(target_coordinates[first], target_coordinates[left], target_coordinates[right], target_coordinates[fourth])
        delta = float(np.arctan2(np.sin(target - current), np.cos(target - current)))
        trial_positive = _rotate_atoms_about_bond(coordinates, mobile, left, right, delta)
        trial_negative = _rotate_atoms_about_bond(coordinates, mobile, left, right, -delta)
        error_positive = abs(np.arctan2(
            np.sin(_dihedral_radian(trial_positive[first], trial_positive[left], trial_positive[right], trial_positive[fourth]) - target),
            np.cos(_dihedral_radian(trial_positive[first], trial_positive[left], trial_positive[right], trial_positive[fourth]) - target),
        ))
        error_negative = abs(np.arctan2(
            np.sin(_dihedral_radian(trial_negative[first], trial_negative[left], trial_negative[right], trial_negative[fourth]) - target),
            np.cos(_dihedral_radian(trial_negative[first], trial_negative[left], trial_negative[right], trial_negative[fourth]) - target),
        ))
        coordinates = trial_positive if error_positive <= error_negative else trial_negative
    return coordinates


def _bond_side_atoms(adjacency, *, seed, blocked_edge):
    blocked = frozenset(blocked_edge)
    visited = set()
    pending = [int(seed)]
    while pending:
        atom = pending.pop()
        if atom in visited:
            continue
        visited.add(atom)
        pending.extend(
            neighbor
            for neighbor in adjacency[atom]
            if neighbor not in visited and frozenset((atom, neighbor)) != blocked
        )
    return tuple(sorted(visited))


def _rotate_atoms_about_bond(coordinates, atoms, left, right, angle):
    result = np.asarray(coordinates, dtype=float).copy()
    origin = result[left].copy()
    axis = result[right] - origin
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-14:
        raise RuntimeError("exocyclic Cartesian fallback found a zero-length bond")
    axis /= norm
    vectors = result[np.asarray(atoms, dtype=int)] - origin
    rotated = (
        vectors * np.cos(angle)
        + np.cross(axis, vectors) * np.sin(angle)
        + np.outer(vectors @ axis, axis) * (1.0 - np.cos(angle))
    )
    result[np.asarray(atoms, dtype=int)] = origin + rotated
    return result


def _dihedral_radian(first, second, third, fourth):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    third = np.asarray(third, dtype=float)
    fourth = np.asarray(fourth, dtype=float)
    middle = third - second
    middle /= max(float(np.linalg.norm(middle)), 1.0e-15)
    left = first - second
    right = fourth - third
    left -= middle * float(left @ middle)
    right -= middle * float(right @ middle)
    return float(np.arctan2(float(middle @ np.cross(left, right)), float(left @ right)))


def _bond_angle_drift(reference, final, bonds, natoms):
    reference = np.asarray(reference, dtype=float)
    final = np.asarray(final, dtype=float)
    bond_changes = [
        abs(
            float(np.linalg.norm(final[left] - final[right]))
            - float(np.linalg.norm(reference[left] - reference[right]))
        )
        for left, right in bonds
    ]
    adjacency = {atom: set() for atom in range(natoms)}
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
    angle_changes = []
    for center, neighbors in adjacency.items():
        ordered = sorted(neighbors)
        for first_index, left in enumerate(ordered):
            for right in ordered[first_index + 1 :]:
                angle_changes.append(
                    abs(
                        _angle_degrees(final[left], final[center], final[right])
                        - _angle_degrees(reference[left], reference[center], reference[right])
                    )
                )
    return (
        max(bond_changes, default=0.0),
        max(angle_changes, default=0.0),
    )


def _angle_degrees(left, center, right):
    first = np.asarray(left, dtype=float) - np.asarray(center, dtype=float)
    second = np.asarray(right, dtype=float) - np.asarray(center, dtype=float)
    denominator = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1.0e-15)
    cosine = float(np.clip(float(first @ second) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _perceived_bonds(geometry) -> tuple[tuple[int, int], ...]:
    numbers = tuple(int(atomic_number(atom) or 0) for atom in geometry.atoms)
    _continuous, graph, _rings, _synthons, _aromaticity = build_topology_objects(
        geometry.coordinates_angstrom, numbers
    )
    return tuple(tuple(sorted(map(int, pair))) for pair in graph.bonds)


def _covalent_mayer_map(geometry, matrix) -> dict[tuple[int, int], float]:
    values = np.asarray(matrix, dtype=float)
    if values.shape != (len(geometry.atoms), len(geometry.atoms)):
        raise ValueError("the L0 Mayer matrix has an incompatible shape")
    return {pair: float(values[pair[0], pair[1]]) for pair in _perceived_bonds(geometry)}


def _validate_population_state(
    geometry,
    charges,
    orders,
    *,
    molecular_charge,
) -> None:
    values = np.asarray(charges, dtype=float).reshape(-1)
    if values.shape != (len(geometry.atoms),) or np.any(~np.isfinite(values)):
        raise ValueError("ZAFF0 requires one finite CM5 charge per atom")
    if abs(float(np.sum(values)) - float(molecular_charge)) > 1.0e-5:
        raise ValueError("ZAFF0 CM5 charges do not conserve molecular charge")
    missing = sorted(set(_perceived_bonds(geometry)) - set(orders))
    if missing:
        raise ValueError(
            "ZAFF0 Mayer orders do not cover every perceived covalent bond: "
            + ", ".join(f"{left}-{right}" for left, right in missing)
        )
    if any(not np.isfinite(float(orders[pair])) for pair in _perceived_bonds(geometry)):
        raise ValueError("ZAFF0 Mayer orders must be finite")


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _geometry_transfer_reliability(initial) -> dict[str, Any]:
    report = Path(initial.report)
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {
            "status": "MISSING_AUDIT",
            "requires_explicit_acceptance": True,
        }
    reliability = payload.get("donor_audit", {}).get("transfer_reliability")
    if not isinstance(reliability, dict):
        return {
            "status": "MISSING_AUDIT",
            "requires_explicit_acceptance": True,
        }
    return reliability


__all__ = [
    "INITIAL_ZAFF0_SCHEMA",
    "SOFT_EXOCYCLIC_MAYER_ORDER_LIMIT",
    "WHOLE_MOLECULE_L0_MAX_HEAVY_ATOMS",
    "WHOLE_MOLECULE_L0_MAX_SMILES_ATOMS",
    "ZAFF_FAST_MAX_ANGLE_DRIFT_DEGREES",
    "ZAFF_FAST_MAX_BOND_DRIFT_ANGSTROM",
    "InitialZaff0Build",
    "build_initial_geometry_and_zaff0",
]
