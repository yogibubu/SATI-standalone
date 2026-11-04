"""Canonical ownership registry for every MATRIX operation.

This module answers one question only: which package and API owns an
operation? Capability availability and host selection are runtime concerns and
must not change these ownership records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .package_registry import PACKAGE_CAPABILITIES


OPERATION_REGISTRY_SCHEMA = "matrix.operation-registry.v1"


@dataclass(frozen=True)
class OperationContract:
    key: str
    owner_package: str
    owner_api: str
    layer: str
    responsibility: str

    def __post_init__(self) -> None:
        if not self.key or self.key != self.key.casefold().replace("-", "_"):
            raise ValueError(f"invalid operation key: {self.key!r}")
        if not self.owner_package.startswith("matrix-"):
            raise ValueError(f"invalid operation owner: {self.owner_package!r}")
        if not self.owner_api.startswith("matrix_"):
            raise ValueError(f"invalid canonical API: {self.owner_api!r}")
        if not self.layer.strip() or not self.responsibility.strip():
            raise ValueError(f"incomplete operation contract: {self.key}")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _operation(
    key: str,
    owner_package: str,
    owner_api: str,
    layer: str,
    responsibility: str,
) -> OperationContract:
    return OperationContract(key, owner_package, owner_api, layer, responsibility)


OPERATION_CONTRACTS: tuple[OperationContract, ...] = (
    _operation("environment", "matrix-core", "matrix_core.environment", "platform", "Environment, paths and persistent configuration."),
    _operation("host_qualification", "matrix-core", "matrix_core.host_capabilities", "platform", "Host inspection and capability qualification."),
    _operation("execution_routing", "matrix-core", "matrix_core.availability", "platform", "Capability- and resource-based execution routing."),
    _operation("workflow", "matrix-core", "matrix_core.workflow", "platform", "Portable workflow contracts and state transitions."),
    _operation("numerical_backend", "matrix-numerics", "matrix_numerics.resolve_native_backend", "platform", "Equivalent portable and accelerated numerical kernels."),
    _operation("remote_transport", "matrix-engines", "matrix_engines.remote_qm", "platform", "Remote execution transport and job state."),
    _operation("cli", "matrix-cli", "matrix_cli.cli.matrix_main", "platform", "Terminal parsing and dispatch to owner APIs."),
    _operation("keymaker", "matrix-gui", "matrix_gui.capability_coverage_report", "platform", "Keymaker and desktop presentation/orchestration."),
    _operation("switch", "matrix-switch", "matrix_switch.smiles_to_cartesian", "input", "Structure resolution and Cartesian seeding."),
    _operation("chem", "matrix-chem", "matrix_chem.preprocess_to_enriched_xyz", "input", "Shared molecular data structures and perception algorithms."),
    _operation("oracle", "matrix-oracle", "matrix_oracle.analyze_structure", "input", "Public perception, topology, symmetry and primitive-source service."),
    _operation("apoc", "matrix-apoc", "matrix_apoc.analyze_source", "input", "Backend-independent electronic population analysis."),
    _operation("fragments", "matrix-fragments", "matrix_fragments.write_fragment_build_section", "input", "Fragmentation and fragment assembly."),
    _operation("qm", "matrix-qm", "matrix_qm.resources.recommend_qm_backend", "qm", "QM result, resource and provider-selection contracts."),
    _operation("gfn2", "matrix-gfn2", "matrix_gfn2.GFN2Calculator", "qm", "Resident GFN2 energy, gradient and observable evaluation."),
    _operation("gaussian", "matrix-gaussian", "matrix_gaussian", "qm", "Gaussian serialization, execution and result promotion."),
    _operation("molpro", "matrix-molpro", "matrix_molpro", "qm", "Molpro serialization, execution and result promotion."),
    _operation("mrcc", "matrix-mrcc", "matrix_mrcc", "qm", "MRCC serialization, execution and result promotion."),
    _operation("orca", "matrix-orca", "matrix_orca", "qm", "ORCA serialization, execution and result promotion."),
    _operation("cfour", "matrix-cfour", "matrix_cfour", "qm", "CFOUR serialization, execution and result promotion."),
    _operation("xtb", "matrix-xtb", "matrix_xtb", "qm", "xTB serialization, execution and result promotion."),
    _operation("pyscf", "matrix-pyscf", "matrix_pyscf", "qm", "PySCF serialization, execution and result promotion."),
    _operation("et", "matrix-et", "matrix_et", "qm", "eT serialization, execution and result promotion."),
    _operation("turbomole", "matrix-turbomole", "matrix_turbomole", "qm", "TURBOMOLE serialization, execution and result promotion."),
    _operation("composite", "matrix-composite", "matrix_composite.CompositeOrchestrator", "qm", "Composition of owner-provided electronic energies and derivatives."),
    _operation("epoc", "matrix-epoc", "matrix_epoc", "exploration", "Boundary, confinement and continuum environment fields."),
    _operation("smith", "matrix-smith", "matrix_smith.write_gicforge_build_sections", "coordinates", "Internal-coordinate and SONIC construction."),
    _operation("architect", "matrix-architect", "matrix_architect.build_zaff_force_field", "coordinates", "Field fitting and parameter authoring."),
    _operation("zaff", "matrix-zaff", "matrix_zaff.ZaffBackend", "coordinates", "Resident field evaluation."),
    _operation("link", "matrix-link", "matrix_link.GeometryEvaluationService", "coordinates", "Geometry realization and optimization."),
    _operation("sentinel", "matrix-sentinel", "matrix_sentinel.GeneticStrategy", "exploration", "Sampling and search strategy."),
    _operation("mifune", "matrix-mifune", "matrix_mifune.run_dynamics", "exploration", "Molecular dynamics and Monte Carlo propagation."),
    _operation("seraph", "matrix-seraph", "matrix_seraph.SeraphBuilder", "exploration", "Solvent and environment construction."),
    _operation("rama", "matrix-rama", "matrix_rama.ReactionPathCandidate", "exploration", "Reaction-path ensemble generation."),
    _operation("cypher", "matrix-cypher", "matrix_cypher.cluster_exploration_samples", "exploration", "Ensemble clustering and reduction."),
    _operation("niobe", "matrix-niobe", "matrix_niobe.pca_report", "inspection", "Read-only inspection and exploratory reports."),
    _operation("gf", "matrix-gf", "matrix_gf", "observables", "Harmonic GF and PED analysis."),
    _operation("trinity", "matrix-trinity", "matrix_trinity", "observables", "Anharmonic field preparation."),
    _operation("dvr", "matrix-dvr", "matrix_dvr", "observables", "Large-amplitude vibrational solution."),
    _operation("vpt2_vci", "matrix-vpt2-vci", "matrix_vpt2_vci", "observables", "VPT2 and VCI solution."),
    _operation("rovib", "matrix-rovib", "matrix_rovib", "observables", "Rovibrational analysis."),
    _operation("thermo", "matrix-thermo", "matrix_thermo", "observables", "Thermochemistry."),
    _operation("kinetics", "matrix-kinetics", "matrix_kinetics", "observables", "Kinetics and reaction networks."),
    _operation("morpheus", "matrix-morpheus", "matrix_morpheus.fit_semiexperimental_geometry", "observables", "Semiexperimental refinement."),
)


def operation_contracts() -> tuple[OperationContract, ...]:
    return OPERATION_CONTRACTS


def operation_contract(key: str) -> OperationContract:
    normalized = str(key).strip().casefold().replace("-", "_")
    normalized = {"lock": "rovib", "sati": "trinity"}.get(normalized, normalized)
    for contract in OPERATION_CONTRACTS:
        if contract.key == normalized:
            return contract
    raise KeyError(f"unknown MATRIX operation: {key}")


def operation_registry_issues() -> tuple[str, ...]:
    issues: list[str] = []
    keys = [contract.key for contract in OPERATION_CONTRACTS]
    if len(keys) != len(set(keys)):
        issues.append("operation registry contains duplicate keys")
    packages = {record.package: record for record in PACKAGE_CAPABILITIES}
    for contract in OPERATION_CONTRACTS:
        package = packages.get(contract.owner_package)
        if package is None:
            issues.append(f"{contract.key}: unknown owner package {contract.owner_package}")
            continue
        import_root = contract.owner_api.split(".", 1)[0]
        if import_root != package.import_name:
            issues.append(
                f"{contract.key}: API {contract.owner_api} does not belong to "
                f"{contract.owner_package}"
            )
    return tuple(issues)


def operation_registry_payload() -> dict[str, object]:
    issues = operation_registry_issues()
    if issues:
        raise ValueError("invalid MATRIX operation registry: " + "; ".join(issues))
    return {
        "schema": OPERATION_REGISTRY_SCHEMA,
        "operation_count": len(OPERATION_CONTRACTS),
        "operations": [contract.to_dict() for contract in OPERATION_CONTRACTS],
        "owner_index": {
            contract.key: contract.owner_package for contract in OPERATION_CONTRACTS
        },
    }


def operation_registry_json(*, indent: int = 2) -> str:
    return json.dumps(operation_registry_payload(), indent=indent, sort_keys=True)


def operation_registry_schema_path() -> Path:
    return Path(__file__).resolve().parent / "schemas" / "operation-registry-v1.schema.json"


__all__ = [
    "OPERATION_CONTRACTS",
    "OPERATION_REGISTRY_SCHEMA",
    "OperationContract",
    "operation_contract",
    "operation_contracts",
    "operation_registry_issues",
    "operation_registry_json",
    "operation_registry_payload",
    "operation_registry_schema_path",
]
