from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .sectioned_xyz import is_section_header_line, read_sectioned_lines, xyz_tail_start


PLANNED_FRAMEWORK_NAME = "MATRIX"
PLANNED_FRAMEWORK_EXPANSION = "Molecular Analysis Toolkit for Reusable Integrated eXperiments"
NANO_MATRIX_SCIENTIFIC_CONTRACT_SCHEMA = "nano-matrix.scientific-contract.v1"


@dataclass(frozen=True)
class ScientificInvariant:
    key: str
    requirement: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


NANO_MATRIX_SCIENTIFIC_INVARIANTS: tuple[ScientificInvariant, ...] = (
    ScientificInvariant(
        "CONTINUOUS_PHYSICS",
        "Physical energies, observables and parameter interpolations are continuous "
        "and provide the differentiability required by their consumers.",
    ),
    ScientificInvariant(
        "ASYMPTOTIC_CORRECTNESS",
        "Every physical model preserves its declared short-range, long-range, "
        "dissociation and boundary limits.",
    ),
    ScientificInvariant(
        "NO_TOPOLOGICAL_DISCONTINUITIES",
        "Discrete graphs, types and neighbor lists never create discontinuities in "
        "physical energies or observables; energetic topology uses continuous weights.",
    ),
    ScientificInvariant(
        "SINGLE_SCIENTIFIC_OWNER",
        "Tools consume canonical owner APIs and do not duplicate scientific formulas "
        "or parameter libraries.",
    ),
    ScientificInvariant(
        "BACKEND_EQUIVALENCE",
        "Reference, compiled CPU, parallel and GPU paths evaluate the same scientific "
        "function within declared tolerances.",
    ),
)


@dataclass(frozen=True)
class ToolContract:
    key: str
    display_name: str
    current_package: str
    standalone_command: str
    operation_key: str = ""
    required_sections: tuple[str, ...] = ()
    optional_sections: tuple[str, ...] = ()
    produced_sections: tuple[str, ...] = ()
    owned_sections: tuple[str, ...] = ()
    consumed_artifacts: tuple[str, ...] = ()
    produced_artifacts: tuple[str, ...] = ()
    owned_capabilities: tuple[str, ...] = ()
    status: str = "implemented"
    planned_name: str = ""
    expanded_name: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ToolReadiness:
    contract: ToolContract
    xyzin_path: Path
    present_sections: tuple[str, ...]
    missing_required_sections: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.missing_required_sections

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.contract.key,
            "display_name": self.contract.display_name,
            "planned_name": self.contract.planned_name,
            "xyzin_path": str(self.xyzin_path),
            "ready": self.ready,
            "required_sections": self.contract.required_sections,
            "present_sections": self.present_sections,
            "missing_required_sections": self.missing_required_sections,
        }


TOOL_CONTRACTS: tuple[ToolContract, ...] = (
    ToolContract(
        key="import",
        display_name="ORACLE import adapters",
        current_package="matrix-link (legacy package name)",
        standalone_command="link preprocess SOURCE OUTPUT",
        operation_key="switch",
        produced_sections=("SOURCE", "BASIC"),
        planned_name="ORACLE import",
        expanded_name="",
        notes=(
            "Formerly ORACLE-Babel and temporarily implemented in matrix-link. "
            "Imports external geometry/QM/SMILES/database sources into xyzin, then normally invokes "
            "ORACLE perception during preprocessing. The command name matrix link preprocess remains "
            "a compatibility alias; LINK is reserved for SONIC-based optimization and PES fitting."
        ),
    ),
    ToolContract(
        key="oracle",
        display_name="ORACLE perception",
        current_package="matrix-chem / matrix-oracle",
        standalone_command=(
            "matrix oracle analyze SOURCE -o molecule.xyzin --human-report molecule.oracle.txt; "
            "matrix oracle report molecule.xyzin; "
            "matrix oracle refine-l1 molecule.xyzin -o molecule.pl1.xyzin; "
            "matrix architect build molecule.xyzin --output molecule.zaff.json"
        ),
        operation_key="oracle",
        required_sections=("BASIC",),
        optional_sections=("CARTESIAN_HESSIAN",),
        produced_sections=(
            "SYMMETRY",
            "TOPOLOGY",
            "AROMATICITY",
            "SYNTHONS",
            "PRIMITIVES",
            "ACCURACY_LADDER_REFINEMENT",
        ),
        owned_sections=(
            "SYMMETRY",
            "TOPOLOGY",
            "AROMATICITY",
            "SYNTHONS",
            "PRIMITIVES",
            "ACCURACY_LADDER_REFINEMENT",
        ),
        produced_artifacts=(
            "matrix.xyz.primitives.v2",
            "oracle.xyz.accuracy_ladder_refinement.v1",
            "matrix.tank.perception_proposal.v1",
            "matrix.tank.geometry_proposal.v1",
        ),
        owned_capabilities=(
            "IMPORT_NORMALIZATION",
            "CONTINUOUS_PERCEPTION",
            "TOPOLOGY",
            "SYMMETRY",
            "RINGS",
            "SYNTHONS",
            "PIC_SOURCE",
            "GEOMETRY_CORRECTIONS",
            "TANK_LCB26_PERCEPTION",
            "TANK_LCB26_GEOMETRY",
        ),
        planned_name="ORACLE",
        expanded_name="Operator for Recognition, Atom-typing and Continuous Local Equivalence",
        notes=(
            "Continuous-perception development of PROXIMA; it will be described separately. "
            "Owns the molecular graph and cycle basis, point-group operations and atom "
            "permutations, atom equivalence, effective atomic number, charge, covalency, "
            "delocalization, strain, bond-order and synthon descriptors, and the redundant "
            "primitive/Wilson-B source used by SMITH and all consumers, applies posterior "
            "L1-to-PL1 geometry corrections. Hessian reduction and force-field construction "
            "belong to ARCHITECT/ZAFF; the old ORACLE entry point is hidden and deprecated. "
            "The desktop GUI remains an ORACLE client/orchestrator, not a separate chemistry engine."
            " TANK is the public LCB26 donor-search and geometry-assembly capability owned by ORACLE."
        ),
    ),
    ToolContract(
        key="apoc",
        display_name="APOC electronic analysis",
        current_package="matrix-apoc / matrix-qm",
        standalone_command=(
            "matrix apoc gaussian calculation.log --xyzin molecule.xyzin; "
            "matrix apoc molden orbitals.molden --xyzin molecule.xyzin"
        ),
        optional_sections=("ORBITALS",),
        produced_sections=("QM_POPULATION",),
        owned_sections=("QM_POPULATION",),
        produced_artifacts=(
            "matrix.apoc.analysis.v1",
            "matrix.qm.population.v1",
        ),
        owned_capabilities=(
            "HIRSHFELD_POPULATION",
            "CM5_CHARGES",
            "MAYER_BOND_ORDERS",
            "ELECTRONIC_OUTPUT_NORMALIZATION",
        ),
        planned_name="APOC",
        expanded_name="Atomic and Pairwise Observables from the Charge density",
        notes=(
            "Consumes a QM density matrix or complete molecular orbitals plus AO basis. "
            "Produces the sole backend-independent CM5/Mayer contract used by ORACLE and "
            "required by ARCHITECT/ZAFF. Alternative population definitions require an "
            "explicitly labelled override and never replace the default silently."
        ),
    ),
    ToolContract(
        key="qm_adapters",
        display_name="QM adapters",
        current_package=(
            "matrix-qm / matrix-gaussian / matrix-molpro / matrix-mrcc / matrix-orca / "
            "matrix-cfour / matrix-xtb / matrix-pyscf / matrix-et"
        ),
        standalone_command=(
            "matrix gaussian promote-fchk|promote-rovib|promote-electronic; "
            "matrix molpro promote; matrix mrcc promote; matrix orca promote; "
            "matrix cfour run|status; matrix xtb run|status|summary; "
            "matrix pyscf run|status|summary; matrix et run|status"
        ),
        operation_key="qm",
        produced_sections=(
            "SOURCE",
            "BASIC",
            "SYMMETRY",
            "TOPOLOGY",
            "SYNTHONS",
            "PRIMITIVES",
            "CARTESIAN_HESSIAN",
            "NORMAL_MODES",
            "QFF",
            "ROTATIONAL",
            "VIBRATIONAL",
            "DELTABVIB",
            "ELECTRONIC",
            "TRANSITIONS",
            "ORBITALS",
            "PROPERTIES",
        ),
        owned_sections=(
            "CARTESIAN_HESSIAN",
            "NORMAL_MODES",
            "QFF",
            "ROTATIONAL",
            "VIBRATIONAL",
            "DELTABVIB",
            "ELECTRONIC",
            "TRANSITIONS",
            "ORBITALS",
            "PROPERTIES",
        ),
        owned_capabilities=("QM_RESULT_CONTRACT", "QM_PROVIDER_DISPATCH"),
        notes=(
            "One adapter owns each external QM format. Scientific tools consume only "
            "the normalized xyzin sections and never reparse provider output. Gaussian, "
            "Molpro, MRCC, ORCA, CFOUR, xTB, PySCF and eT are registered providers. "
            "#PROPERTIES stores program-dependent QM properties with unit/conversion metadata. "
            "The adapters expose densities/orbitals to APOC; APOC alone owns #QM_POPULATION."
        ),
    ),
    ToolContract(
        key="fragments",
        display_name="Fragments / nano-lego",
        current_package="matrix-fragments",
        standalone_command="matrix fragments build molecule.xyzin",
        required_sections=("TOPOLOGY", "SYNTHONS"),
        produced_sections=("FRAGMENTS",),
        owned_sections=("FRAGMENTS",),
        status="implemented",
        notes=(
            "Shared fragment records include atom membership, center, frame and explicit optional "
            "charge/multiplicity. All tools consume the matrix-fragments parser rather than "
            "reparsing #FRAGMENTS rows."
        ),
    ),
    ToolContract(
        key="gicforge",
        display_name="SMITH / SONIC",
        current_package="matrix-smith",
        standalone_command="matrix smith standalone molecule.smith.xyz molecule.xyzin",
        operation_key="smith",
        required_sections=(),
        optional_sections=("PRIMITIVES", "FRAGMENTS"),
        produced_sections=("GIC", "SYCART"),
        owned_sections=("GIC", "SYCART"),
        consumed_artifacts=("matrix.xyz.primitives.v2",),
        produced_artifacts=(
            "oracle.gic.definition.v1",
            "matrix.smith.sonic_diagnostics.v2",
        ),
        owned_capabilities=(
            "SONIC_CONSTRUCTION",
            "SONIC_VISUALIZATION",
            "G16_COORDINATE_EXPORT",
        ),
        planned_name="SMITH",
        expanded_name="Symmetry-Mapped Internal-coordinate Template Handler",
        notes=(
            "SMITH is the MATRIX tool that consumes ORACLE's primitive/B source and builds "
            "SONIC coordinates from a frozen ORACLE state; "
            "it does not own or recompute cycles, symmetry or continuous atomic descriptors. "
            "Its Wilson B use ends with SONIC construction and first-order visualization; "
            "internal-to-Cartesian realization belongs to LINK and B-prime belongs to ARCHITECT. "
            "It can consume a populated xyzin state or one extended-XYZ input with Cartesian "
            "geometry and directives through an explicitly labelled reduced ORACLE convenience "
            "profile; "
            "matrix gicforge remains a compatibility alias for matrix smith."
        ),
    ),
    ToolContract(
        key="architect",
        display_name="ARCHITECT authoring / ZAFF runtime",
        current_package="matrix-architect / matrix-zaff",
        standalone_command=(
            "architect build molecule.xyzin --output field.zaff.json; "
            "architect validate field.zaff.json; architect evaluate field.zaff.json point.xyz"
        ),
        required_sections=("BASIC", "PRIMITIVES", "SYNTHONS", "CARTESIAN_HESSIAN"),
        optional_sections=("GIC", "ACCURACY_LADDER_REFINEMENT"),
        consumed_artifacts=(
            "matrix.xyz.primitives.v2",
            "oracle.gic.definition.v1",
        ),
        produced_artifacts=(
            "matrix.zaff.force_field.v1",
            "matrix.architect.evaluation.v1",
            "matrix.architect.derivative_validation.v1",
        ),
        owned_capabilities=(
            "ZAFF_FORCE_FIELD_CONSTRUCTION",
            "PIC_VS_LOCAL_SONIC_SELECTION",
            "COUPLING_SELECTION",
            "B_MATRIX_FIRST_DERIVATIVE",
            "HESSIAN_COORDINATE_TRANSFORMATION",
            "ZAFF_RUNTIME_HANDOFF",
        ),
        planned_name="ARCHITECT",
        expanded_name=(
            "Automated Refinement of Charges, Hessians, Internal coordinates, Torsions, "
            "Equilibria, and Coupled Terms"
        ),
        notes=(
            "Consumes frozen ORACLE PIC/synthon state and optional SMITH local-SONIC blocks. "
            "ARCHITECT owns sparse analytic B-prime, Hessian coordinate transformations, "
            "ZAFF fitting and quantitative coupling audits. matrix-zaff alone owns the "
            "stable resident analytic E/G/H runtime consumed by LINK, SENTINEL and MIFUNE."
        ),
    ),
    ToolContract(
        key="gf",
        display_name="TRINITY harmonic/GF",
        current_package="matrix-gf",
        standalone_command="matrix gf --xyzin molecule.xyzin",
        required_sections=("GIC", "CARTESIAN_HESSIAN"),
        optional_sections=("SYNTHONS", "GF_PED"),
        produced_sections=("GF_PED",),
        owned_sections=("GF_PED",),
        consumed_artifacts=("oracle.gic.definition.v1",),
        owned_capabilities=("HARMONIC_GF_PED", "CURVATURE_IMPROVEMENT"),
        planned_name="TRINITY",
        expanded_name="Torsional and Rovibrational Internal-coordinate Network for Integrated Theoretical spectroscopY",
        notes=(
            "TRINITY harmonic module: Wilson GF, PED, diagonal curvature improvement, "
            "large-amplitude classification and handoff to DVR/VPT2/VCI."
        ),
    ),
    ToolContract(
        key="morpheus",
        display_name="SEFit / MORPHEUS",
        current_package="matrix-morpheus",
        standalone_command="matrix semiexp --xyzin molecule.xyzin --job job.toml --outdir run",
        required_sections=("ISOTOPOLOGUES",),
        optional_sections=("GIC", "SYCART", "MORPHEUS"),
        produced_sections=("MORPHEUS",),
        owned_sections=("MORPHEUS", "ISOTOPOLOGUES"),
        notes=(
            "Consumes frozen coordinate models or symmetry-Cartesian state and owns "
            "semiexperimental fit state. Any requested internal-to-Cartesian realization "
            "is delegated to LINK; MORPHEUS does not own a Wilson-B inverse."
        ),
    ),
    ToolContract(
        key="sentinel",
        display_name="SENTINEL",
        current_package="matrix-sentinel",
        standalone_command="matrix sentinel run request.json response.json --config strategy.json",
        consumed_artifacts=("matrix.link.sentinel.request.v1",),
        produced_artifacts=(
            "matrix.link.sentinel.response.v1",
            "matrix.sentinel.point_selection.v1",
        ),
        owned_capabilities=(
            "SEQUENTIAL_SCAN_SELECTION",
            "GENETIC_PES_SELECTION",
            "MONTE_CARLO_PES_SELECTION",
            "SAMPLING_CHECKPOINT",
        ),
        planned_name="SENTINEL",
        expanded_name=(
            "Symmetry-adapted Evolutionary Navigation Through Internal-coordinate "
            "Nonlinear Energy Landscapes"
        ),
        notes=(
            "Owns PES point order, populations, chains, RNG state and termination. "
            "LINK owns coordinate realization and E/G/H evaluation."
        ),
    ),
    ToolContract(
        key="trinity",
        display_name="TRINITY",
        current_package="matrix-gf / matrix-dvr / matrix-vpt2-vci",
        standalone_command="matrix gf --xyzin molecule.xyzin; matrix dvr run --xyzin molecule.xyzin",
        operation_key="trinity",
        required_sections=("BASIC",),
        optional_sections=("GIC", "SYCART", "GF_PED", "DVR", "VPT2_VCI"),
        produced_sections=("GF_PED", "DVR", "VPT2_VCI"),
        owned_sections=("GF_PED", "DVR", "VPT2_VCI"),
        consumed_artifacts=("oracle.gic.definition.v1",),
        owned_capabilities=(
            "HARMONIC_GF_PED",
            "ANHARMONIC_FIELD_CONSTRUCTION",
            "ARCHITECT_ZAFF_FIELD_PROJECTION",
            "VIBRATIONAL_SCALING",
            "DVR",
            "VPT2_VCI",
        ),
        status="partly-implemented",
        planned_name="TRINITY",
        expanded_name="Torsional and Rovibrational Internal-coordinate Network for Integrated Theoretical spectroscopY",
        notes=(
            "Umbrella TRINITY state for molecular vibrational fields: harmonic GF/PED, "
            "QFF construction from direct QM or ARCHITECT/ZAFF, DVR and VPT2/VCI. "
            "Geometry optimization is owned by LINK and point selection by SENTINEL."
        ),
    ),
    ToolContract(
        key="link",
        display_name="LINK",
        current_package="matrix-link",
        standalone_command=(
            "matrix link prepare molecule.xyzin --run-dir run --engine-command CMD; "
            "matrix link scan-prepare molecule.xyzin --run-dir scan --coordinate R001"
        ),
        required_sections=("BASIC", "GIC"),
        optional_sections=("SYCART", "LINK"),
        produced_sections=("LINK",),
        owned_sections=("LINK",),
        consumed_artifacts=(
            "oracle.gic.definition.v1",
            "matrix.zaff.force_field.v1",
        ),
        produced_artifacts=(
            "matrix.link.sentinel.request.v1",
            "matrix.link.sentinel.response.v1",
            "matrix.link.sentinel.checkpoint.v1",
        ),
        owned_capabilities=(
            "INTERNAL_TO_CARTESIAN_REALIZATION",
            "GEOMETRY_OPTIMIZATION",
            "TRANSITION_STATE_OPTIMIZATION",
            "PARTIAL_OPTIMIZATION",
            "POINT_EVALUATION",
            "SENTINEL_EVALUATION_BRIDGE",
        ),
        status="implemented",
        planned_name="LINK",
        expanded_name="Level-aware Internal-coordinate Network for Kinetics and optimization",
        notes=(
            "LINK is the SONIC-based layer for minimum/TS geometry optimization, coordinate "
            "realization and E/G/H evaluation with external electronic-structure engines or "
            "ARCHITECT/ZAFF. SENTINEL owns scan/GA/Monte Carlo point selection and TRINITY owns "
            "molecular anharmonic-field fitting. The realization and optimizer APIs live in "
            "matrix-link. SENTINEL proposes SONIC points and never consumes B."
        ),
    ),
    ToolContract(
        key="rovib",
        display_name="Rovib utilities",
        current_package="matrix-rovib",
        standalone_command="matrix rovib summarize molecule.xyzin",
        required_sections=("ROTATIONAL",),
        optional_sections=("VIBRATIONAL", "DELTABVIB", "CORIOLIS", "QCENT"),
        produced_sections=("CORIOLIS", "QCENT", "ROTATIONAL_SPECTRUM"),
        owned_sections=(
            "ROTATIONAL",
            "VIBRATIONAL",
            "DELTABVIB",
            "CORIOLIS",
            "QCENT",
            "ROTATIONAL_SPECTRUM",
        ),
    ),
    ToolContract(
        key="thermo",
        display_name="Thermo",
        current_package="matrix-thermo",
        standalone_command="matrix thermo molecule.xyzin",
        required_sections=("BASIC", "ROTATIONAL"),
        optional_sections=("VIBRATIONAL",),
        produced_sections=("THERMO",),
        owned_sections=("THERMO",),
    ),
    ToolContract(
        key="kinetics",
        display_name="TST / RRKM / capture kinetics",
        current_package="matrix-kinetics",
        standalone_command=(
            "matrix kinetics single reactant.xyzin ts.xyzin --reactant-dos reactant.dos "
            "--ts-dos ts.dos --barrier-cm1 E0"
        ),
        required_sections=("VIBRATIONAL",),
        optional_sections=("ROTATIONAL", "THERMO", "MASTER_EQUATION"),
        produced_sections=("KINETICS",),
        owned_sections=("KINETICS",),
        consumed_artifacts=("matrix.kinetics.network.v1",),
        produced_artifacts=("matrix.kinetics.network.v1",),
        status="single-reaction",
        notes=(
            "Canonical unimolecular TST and microcanonical RRKM share the normalized "
            "reactant/transition-state DOS. The network manifest already represents species "
            "nodes and reaction-channel edges. The master equation consumes and updates "
            "the shared #MASTER_EQUATION XYZIN section through matrix kinetics master-equation."
            " Hard-sphere collision, legacy Gorin capture and Landau-Zener nonadiabatic "
            "TST are public channel-model calculators with explicit unit/convention metadata."
        ),
    ),
    ToolContract(
        key="master_equation",
        display_name="Chemical master equation",
        current_package="matrix-kinetics",
        operation_key="kinetics.master_equation",
        standalone_command="matrix kinetics master-equation molecule.xyzin",
        required_sections=("MASTER_EQUATION",),
        optional_sections=("BASIC", "THERMO", "VIBRATIONAL", "ROTATIONAL"),
        produced_sections=("MASTER_EQUATION",),
        owned_sections=("MASTER_EQUATION",),
        consumed_artifacts=("matrix.kinetics.master_equation.network_request.v1",),
        produced_artifacts=("matrix.kinetics.master_equation.network_result.v1",),
        notes="Single-host E or E,J networks from qualified shared state and flux inventories. "
              "The XYZIN section preserves geometry and other tools' sections.",
    ),
    ToolContract(
        key="vpt2_vci",
        display_name="TRINITY VPT2 / VCI",
        current_package="matrix-vpt2-vci",
        standalone_command=(
            "matrix vpt2-vci --xyzin molecule.xyzin --run-dir run; "
            "matrix hybrid-vibrations --fchk minimum.fchk "
            "--path-pair lower.xyz upper.xyz --report hybrid.json"
        ),
        required_sections=("QFF",),
        optional_sections=("VPT2_VCI",),
        produced_sections=("VPT2_VCI",),
        owned_sections=("VPT2_VCI",),
        planned_name="TRINITY",
        notes=(
            "Anharmonic TRINITY module. FCHK and indexed QFF text are adapter entry points; "
            "normalized standalone input is #QFF."
        ),
    ),
    ToolContract(
        key="dvr",
        display_name="TRINITY DVR",
        current_package="matrix-dvr",
        standalone_command="matrix dvr run --xyzin molecule.xyzin",
        required_sections=("DVR",),
        produced_sections=("DVR",),
        owned_sections=("DVR",),
        planned_name="TRINITY",
        notes=(
            "Large-amplitude TRINITY module. Gaussian scan logs are prepare-time adapter inputs; "
            "post-run state is collected into #DVR."
        ),
    ),
    ToolContract(
        key="gui",
        display_name="MATRIX desktop",
        current_package="matrix-gui",
        standalone_command="matrix gui [ORACLE|SMITH|LINK|MORPHEUS|TRINITY] [molecule.xyzin]",
        operation_key="keymaker",
        optional_sections=("BASIC", "GIC", "GF_PED", "MORPHEUS", "TRINITY", "VPT2_VCI", "DVR"),
        status="orchestrator",
        planned_name="MATRIX desktop",
        notes=(
            "The suite-level GUI is owned by MATRIX, not ORACLE. Each scientific tool has an "
            "independent window backed by non-Qt services. matrix-oracle remains a compatibility "
            "client while its legacy monolithic dashboard is retired incrementally."
        ),
    ),
)


def tool_contracts(*, include_gui: bool = True) -> tuple[ToolContract, ...]:
    if include_gui:
        return TOOL_CONTRACTS
    return tuple(contract for contract in TOOL_CONTRACTS if contract.key != "gui")


def tool_contract(key: str) -> ToolContract:
    normalized = key.strip().lower().replace("-", "_")
    if normalized in {"babel", "matrix_link", "oracle_babel", "preprocess", "import"}:
        normalized = "import"
    if normalized in {
        "gaussian",
        "molpro",
        "mrcc",
        "orca",
        "properties",
        "qm",
        "qm_jobs",
        "qm_adapters",
    }:
        normalized = "qm_adapters"
    if normalized in {"smith", "smith", "sonic"}:
        normalized = "gicforge"
    if normalized in {"zaff", "force_field", "forcefield"}:
        normalized = "architect"
    # Exact stable keys take precedence over display/planned-name aliases.
    # This matters for the TRINITY umbrella contract, whose planned name is
    # also used by the narrower historical ``gf`` contract.
    for contract in TOOL_CONTRACTS:
        if contract.key == normalized:
            return contract
    for contract in TOOL_CONTRACTS:
        if contract.display_name.lower() == normalized:
            return contract
        if contract.planned_name and contract.planned_name.lower() == normalized:
            return contract
    raise KeyError(f"unknown MATRIX tool contract: {key}")


def tool_contract_lines(
    contracts: tuple[ToolContract, ...] | None = None,
    *,
    include_notes: bool = True,
) -> list[str]:
    rows = tool_contracts() if contracts is None else contracts
    lines: list[str] = []
    for contract in rows:
        lines.append(f"{contract.key}: {contract.display_name}")
        if contract.planned_name:
            lines.append(f"  planned_name: {contract.planned_name}")
        if contract.expanded_name:
            lines.append(f"  expanded_name: {contract.expanded_name}")
        lines.append(f"  package: {contract.current_package}")
        lines.append(f"  command: {contract.standalone_command}")
        lines.append(f"  required: {_join_sections(contract.required_sections)}")
        lines.append(f"  optional: {_join_sections(contract.optional_sections)}")
        lines.append(f"  produced: {_join_sections(contract.produced_sections)}")
        lines.append(f"  owned: {_join_sections(contract.owned_sections)}")
        lines.append(f"  consumes artifacts: {_join_sections(contract.consumed_artifacts)}")
        lines.append(f"  produces artifacts: {_join_sections(contract.produced_artifacts)}")
        lines.append(f"  capabilities: {_join_sections(contract.owned_capabilities)}")
        lines.append(f"  status: {contract.status}")
        if include_notes and contract.notes:
            lines.append(f"  notes: {contract.notes}")
    return lines


def tool_contract_markdown_table(contracts: tuple[ToolContract, ...] | None = None) -> str:
    rows = tool_contracts() if contracts is None else contracts
    lines = [
        "| Key | Current name | Planned name | Expanded name | Package | "
        "Required sections | Produced sections | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for contract in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    contract.key,
                    contract.display_name,
                    contract.planned_name or "",
                    contract.expanded_name or "",
                    contract.current_package,
                    ", ".join(contract.required_sections) or "none",
                    ", ".join(contract.produced_sections) or "none",
                    contract.status,
                )
            )
            + " |"
        )
    return "\n".join(lines)


def tool_contracts_json(contracts: tuple[ToolContract, ...] | None = None) -> str:
    rows = tool_contracts() if contracts is None else contracts
    return json.dumps([contract.to_dict() for contract in rows], indent=2, sort_keys=True)


def nano_matrix_scientific_contract() -> dict[str, object]:
    """Return the suite-wide parent contract inherited by every tool."""

    return {
        "schema": NANO_MATRIX_SCIENTIFIC_CONTRACT_SCHEMA,
        "scope": "all nano-Matrix scientific tools and execution backends",
        "invariants": [invariant.to_dict() for invariant in NANO_MATRIX_SCIENTIFIC_INVARIANTS],
    }


def xyzin_section_names(path: Path | str) -> tuple[str, ...]:
    lines = read_sectioned_lines(Path(path))
    names: list[str] = []
    for raw in lines[xyz_tail_start(lines) :]:
        if is_section_header_line(raw):
            names.append(raw.strip()[1:].strip().upper())
    return tuple(names)


def tool_contract_readiness(path: Path | str, contract: ToolContract | str) -> ToolReadiness:
    target = Path(path)
    resolved = tool_contract(contract) if isinstance(contract, str) else contract
    present = xyzin_section_names(target)
    present_set = set(present)
    missing = tuple(section for section in resolved.required_sections if section not in present_set)
    return ToolReadiness(
        contract=resolved,
        xyzin_path=target,
        present_sections=present,
        missing_required_sections=missing,
    )


def tool_contract_readinesses(
    path: Path | str,
    contracts: tuple[ToolContract, ...] | None = None,
) -> tuple[ToolReadiness, ...]:
    rows = tool_contracts() if contracts is None else contracts
    return tuple(tool_contract_readiness(path, contract) for contract in rows)


def tool_readiness_lines(readinesses: tuple[ToolReadiness, ...]) -> list[str]:
    lines: list[str] = []
    for readiness in readinesses:
        missing = _join_sections(readiness.missing_required_sections)
        lines.append(f"{readiness.contract.key}: ready={int(readiness.ready)} missing={missing}")
    return lines


def tool_readiness_markdown_table(readinesses: tuple[ToolReadiness, ...]) -> str:
    lines = [
        "| Key | Current name | Planned name | Ready | Missing required sections |",
        "| --- | --- | --- | --- | --- |",
    ]
    for readiness in readinesses:
        lines.append(
            "| "
            + " | ".join(
                (
                    readiness.contract.key,
                    readiness.contract.display_name,
                    readiness.contract.planned_name or "",
                    str(int(readiness.ready)),
                    ", ".join(readiness.missing_required_sections) or "none",
                )
            )
            + " |"
        )
    return "\n".join(lines)


def tool_readiness_json(readinesses: tuple[ToolReadiness, ...]) -> str:
    return json.dumps(
        [readiness.to_dict() for readiness in readinesses],
        indent=2,
        sort_keys=True,
    )


def _join_sections(sections: tuple[str, ...]) -> str:
    return ", ".join(sections) if sections else "none"
