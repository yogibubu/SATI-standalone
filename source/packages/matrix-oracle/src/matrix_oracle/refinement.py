"""End-to-end ORACLE L1-to-PL1 Cartesian refinement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from matrix_chem import (
    AccuracyLadderPlan,
    BackTransformationResult,
    RefinementLayer,
    Structure,
    ValenceLevel,
    apply_accuracy_ladder_plan,
    build_accuracy_ladder_plan,
    build_topology_objects,
    project_aromatic_ring_planarity,
    read_enriched_xyz,
    read_primitive_contract,
    rotational_constants_MHz,
    load_pl1_gaussian_model,
)
from matrix_chem.inertia import principal_moments
from matrix_chem.geometry_io import write_xyz
from matrix_chem.topology.elements import atomic_number
from matrix_core import replace_section, sha256_file

from .api import OracleAnalysis, analyze_structure


ORACLE_REFINEMENT_SCHEMA = "oracle.xyz.accuracy_ladder_refinement.v1"


@dataclass(frozen=True)
class OracleGeometryRefinement:
    source: Path
    output: Path
    plan: AccuracyLadderPlan
    back_transformation: BackTransformationResult
    analysis: OracleAnalysis
    rotational_constants_mhz: tuple[float, float, float]
    principal_moments_amu_angstrom2: tuple[float, float, float]

    @property
    def target_count(self) -> int:
        return len(self.plan.targets)


def refine_l1_geometry(
    source: Path | str,
    output: Path | str,
    *,
    include_core_valence: bool = True,
    include_conjugation: bool = True,
    include_hydrogen_bonds: bool = True,
    cv_weight_threshold: float = 0.9,
    tolerance: float = 1.0e-8,
    max_iterations: int = 50,
    pl1_model_path: Path | str | None = None,
) -> OracleGeometryRefinement:
    """Convert an ORACLE-enriched L1 geometry into a reanalyzed PL1 state."""
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if source_path == output_path:
        raise ValueError("L1 source and PL1 output must be different files")
    geometry = read_enriched_xyz(source_path)
    contract = read_primitive_contract(source_path)
    numbers = tuple(_required_atomic_number(symbol) for symbol in geometry.atoms)
    _continuous, _discrete, _rings, synthons, aromaticity = build_topology_objects(
        geometry.coordinates_angstrom, numbers
    )
    pl1_model = load_pl1_gaussian_model(pl1_model_path) if pl1_model_path is not None else None
    plan = build_accuracy_ladder_plan(
        contract.primitives,
        numbers,
        valence_level=ValenceLevel.L1,
        include_core_valence=include_core_valence,
        coordinates_angstrom=geometry.coordinates_angstrom,
        synthons=synthons,
        include_bl1_conjugation=include_conjugation,
        include_pl1_hydrogen_bonds=include_hydrogen_bonds,
        cv_weight_threshold=cv_weight_threshold,
        pl1_model=pl1_model,
    )
    back_transformation = apply_accuracy_ladder_plan(
        plan,
        contract.primitives,
        geometry.coordinates_angstrom,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    if not back_transformation.converged:
        raise RuntimeError(
            "ORACLE L1-to-PL1 back-transformation did not converge: "
            f"maximum residual {back_transformation.maximum_residual:.6g}"
        )
    aromatic_coordinates = project_aromatic_ring_planarity(
        back_transformation.coordinates_angstrom,
        aromaticity,
    )
    back_transformation = BackTransformationResult(
        coordinates_angstrom=aromatic_coordinates,
        converged=back_transformation.converged,
        iterations=back_transformation.iterations,
        maximum_residual=back_transformation.maximum_residual,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="oracle-pl1-") as scratch_text:
        cartesian_source = Path(scratch_text) / "pl1.xyz"
        write_xyz(
            cartesian_source,
            geometry.atoms,
            back_transformation.coordinates_angstrom,
            comment=f"ORACLE PL1 refinement of {source_path.name}",
        )
        analysis = analyze_structure(cartesian_source, output_path, source_kind="xyz")
    structure = Structure(
        list(geometry.atoms),
        [tuple(float(value) for value in row) for row in aromatic_coordinates],
    )
    rotational_constants = tuple(float(value) for value in rotational_constants_MHz(structure))
    moments = tuple(float(value) for value in principal_moments(structure, isotopic=True))
    replace_section(
        output_path,
        "ACCURACY_LADDER_REFINEMENT",
        _refinement_section_lines(
            source_path,
            plan,
            back_transformation,
            include_core_valence=include_core_valence,
            include_conjugation=include_conjugation,
            include_hydrogen_bonds=include_hydrogen_bonds,
            cv_weight_threshold=cv_weight_threshold,
            pl1_model_path=pl1_model_path,
            aromatic_ring_count=len(aromaticity.aromatic_rings),
            rotational_constants_mhz=rotational_constants,
            principal_moments_amu_angstrom2=moments,
        ),
    )
    return OracleGeometryRefinement(
        source=source_path,
        output=output_path,
        plan=plan,
        back_transformation=back_transformation,
        analysis=analysis,
        rotational_constants_mhz=rotational_constants,
        principal_moments_amu_angstrom2=moments,
    )


def _refinement_section_lines(
    source: Path,
    plan: AccuracyLadderPlan,
    result: BackTransformationResult,
    **settings,
) -> list[str]:
    counts = {
        layer: sum(target.layer is layer for target in plan.targets)
        for layer in RefinementLayer
    }
    return [
        f"SCHEMA {ORACLE_REFINEMENT_SCHEMA}",
        "OWNER ORACLE",
        "INPUT_LEVEL L1",
        "OUTPUT_LEVEL PL1",
        f"SOURCE {source}",
        f"SOURCE_SHA256 {sha256_file(source)}",
        f"CORE_VALENCE {str(bool(settings['include_core_valence'])).upper()}",
        f"BL1_CONJUGATION {str(bool(settings['include_conjugation'])).upper()}",
        f"PL1_HYDROGEN_BONDS {str(bool(settings['include_hydrogen_bonds'])).upper()}",
        "CV_AMPLITUDE_MODEL RADIUS_AWARE_PERIOD_LINE",
        f"CV_WEIGHT_THRESHOLD {float(settings['cv_weight_threshold']):.16g}",
        f"PL1_MODEL {settings.get('pl1_model_path') or 'LEGACY_JCP_IS1'}",
        f"CV_TARGETS {counts[RefinementLayer.CORE_VALENCE]}",
        f"BL1_TARGETS {counts[RefinementLayer.BL1_CONJUGATION]}",
        f"PL1_TARGETS {counts[RefinementLayer.PL1_SELECTED_PAIR]}",
        f"AROMATIC_RINGS_PRESERVED {int(settings['aromatic_ring_count'])}",
        "AROMATIC_PLANARITY_PROJECTOR LOCAL_RING_PLANES_SHARED_ATOM_AVERAGE",
        "ROTATIONAL_CONSTANTS_MHZ "
        + " ".join(f"{float(value):.12g}" for value in settings["rotational_constants_mhz"]),
        "PRINCIPAL_MOMENTS_AMU_ANGSTROM2 "
        + " ".join(
            f"{float(value):.12g}" for value in settings["principal_moments_amu_angstrom2"]
        ),
        f"ITERATIONS {result.iterations}",
        f"MAXIMUM_RESIDUAL {result.maximum_residual:.16g}",
        "STATUS PASS",
    ]


def _required_atomic_number(symbol: str) -> int:
    number = atomic_number(symbol)
    if number is None or number <= 0:
        raise ValueError(f"L1-to-PL1 refinement does not support atom label {symbol!r}")
    return int(number)
