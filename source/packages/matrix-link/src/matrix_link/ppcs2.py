"""PCS2 composite energy models assembled from external backends."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .composite_protocol import load_composite_protocol
from .scan import QMBackendTerm, QMScanBackend


PCS2_ORCA_SCHEMA = "matrix.link.pcs2_orca.v1"
PCS2_ORCA_FORMULA = "F12-CCSD(T)/may-cc-pVTZ + ae-MP2/cc-pwCVTZ - fc-MP2/cc-pwCVTZ"
PPCS2_ORCA_SCHEMA = "matrix.link.ppcs2_orca.v2"
PPCS2_ORCA_FORMULA = "DLPNO-CCSD(T)-F12/may-cc-pVTZ + ae-MP2/cc-pwCVTZ - fc-MP2/cc-pwCVTZ"
PCS2_MRCC_SCHEMA = "matrix.link.pcs2_mrcc.v1"
PCS2_MRCC_FORMULA = "CCSD(F12*)(T+)/may-cc-pVTZ + ae-MP2/cc-pwCVTZ - fc-MP2/cc-pwCVTZ"
PPCS2_MRCC_SCHEMA = "matrix.link.ppcs2_mrcc.v1"
PPCS2_MRCC_FORMULA = "LNO-CCSD(T)/jun-cc-pVTZ + MP2-F12/jun-cc-pVTZ - MP2/jun-cc-pVTZ + ae-MP2/cc-pwCVTZ - fc-MP2/cc-pwCVTZ"
PCS2_MOLPRO_SCHEMA = "matrix.link.pcs2.molpro.v2"
PCS2_MOLPRO_FORMULA = "CCSD(T*)-F12b+CABS/cc-pV(T+d)Z + ae-MP2/cc-pwCVTZ - fc-MP2/cc-pwCVTZ"
PPCS2_MOLPRO_SCHEMA = "matrix.link.ppcs2_molpro.v2"
PPCS2_MOLPRO_FORMULA = "PNO-LCCSD(T*)-F12b/may-cc-pVTZ + ae-MP2/cc-pwCVTZ - fc-MP2/cc-pwCVTZ"
PL2_F12_ORCA_SCHEMA = "matrix.link.pl2_f12_orca.v2"
PL2_F12_ORCA_FORMULA = "DLPNO-CCSD(T)-F12/may-cc-pVTZ + [F12-RI-MP2/may-cc-pVTZ - DLPNO-MP2-F12/may-cc-pVTZ] + ae-MP2/cc-pwCVTZ - fc-MP2/cc-pwCVTZ"
PL2_F12_MOLPRO_SCHEMA = "matrix.link.pl2_f12_molpro.v2"
PL2_F12_MOLPRO_FORMULA = "PNO-LCCSD(T*)-F12b/may-cc-pVTZ + [MP2-F12/may-cc-pVTZ - PNO-LMP2-F12/may-cc-pVTZ] + ae-MP2/cc-pwCVTZ - fc-MP2/cc-pwCVTZ"
L2_MOLPRO_SCHEMA = "matrix.link.l2_molpro.v1"
L2_MOLPRO_FORMULA = "fc-CCSD(T)/def2-TZVPP + ae-MP2/cc-pwCVTZ - fc-MP2/cc-pwCVTZ"
PL2_ORCA_SCHEMA = "matrix.link.pl2_orca.v1"
PL2_ORCA_FORMULA = "DLPNO-CCSD(T)/def2-TZVPP + ae-MP2/cc-pwCVTZ - fc-MP2/cc-pwCVTZ"
PL2_MP2_ORCA_SCHEMA = "matrix.link.pl2_mp2_orca.v1"
PL2_MP2_ORCA_FORMULA = "DLPNO-CCSD(T)/def2-TZVPP + [fc-MP2/def2-TZVPP - DLPNO-MP2/def2-TZVPP] + (64/37)[fc-MP2/def2-QZVPP - fc-MP2/def2-TZVPP] + ae-MP2/cc-pwCVTZ - fc-MP2/cc-pwCVTZ"
FAST_MOLPRO_SCHEMA = "matrix.link.fast_molpro.v1"
FAST_MOLPRO_FORMULA = "[fc-MP2/def2-SVP + density-based BSIE(PBEC)/def2-SVP] + [fc-CCSD(T)/def2-SVP - fc-MP2/def2-SVP] + ORACLE CV exponential field"
PL2_LITE_F12_MOLPRO_SCHEMA = "matrix.link.pl2_lite_f12.molpro.v3"
PL2_LITE_MP2_MOLPRO_SCHEMA = "matrix.link.pl2_lite_mp2.molpro.v1"
PL2_LITE_DB_MOLPRO_SCHEMA = "matrix.link.pl2_lite_db.molpro.v3"
L2_LITE_DB_MOLPRO_SCHEMA = "matrix.link.l2_lite_db.molpro.v1"
L2_LITE_F12_MOLPRO_SCHEMA = "matrix.link.l2_lite_f12.molpro.v2"
JUNCHS_GDV_SCHEMA = "matrix.link.junchs_gdv.v1"
JUNCHS_GDV_FORMULA = "fc-CCSD(T)/jun-cc-pVTZ + (64/37)[fc-MP2/jun-cc-pVQZ - fc-MP2/jun-cc-pVTZ] + ae-MP2/cc-pwCVTZ - fc-MP2/cc-pwCVTZ"
JUNCHS_CFOUR_SCHEMA = "matrix.link.junchs_cfour.v1"
JUNCHS_CFOUR_FORMULA = JUNCHS_GDV_FORMULA
JUNCHS_MOLPRO_SCHEMA = "matrix.link.junchs_molpro.v1"
JUNCHS_MOLPRO_FORMULA = JUNCHS_GDV_FORMULA
FPA_Q5_T_CV_MOLPRO_SCHEMA = "matrix.link.fpa_q5_t_cv_molpro.v1"
FPA_HAQ5_HAT_CV_MOLPRO_SCHEMA = "matrix.link.fpa_haq5_hat_cv_molpro.v1"
L2FPA_LITE_MOLPRO_SCHEMA = "matrix.link.l2fpa_lite.molpro.v1"
CHEAP_LITE_MOLPRO_SCHEMA = "matrix.link.cheap_lite.molpro.v1"
SHERRILL_TQ_DZ_CV_MOLPRO_SCHEMA = "matrix.link.sherrill_tq_dz_cv_molpro.v1"

_BASIS_DATA = Path(__file__).resolve().parent / "data" / "basis"
MAY_CC_PVTZ_ORCA = _BASIS_DATA / "may-cc-pVTZ.orca"
MAY_CC_PVTZ_MOLPRO = _BASIS_DATA / "may-cc-pVTZ.molpro"


def junchs_gdv_backend(
    *,
    charge: int = 0,
    multiplicity: int = 1,
    executable: str | None = None,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    processors: int = 1,
    memory_gb: int | None = None,
    composite_parallel_workers: int = 1,
    protocol_selector: str = "junchs-gdv",
) -> QMScanBackend:
    """Return the frozen all-GDV jun-ChS composite-gradient surface."""

    protocol = load_composite_protocol(protocol_selector)
    if protocol.selector not in {"junchs-gdv", "l2-mp2-g16"}:
        raise ValueError(f"unsupported G16.C02 L2-MP2 protocol: {protocol.selector!r}")
    if int(multiplicity) != 1:
        raise ValueError(f"frozen {protocol.protocol_id} currently requires a singlet RHF reference")
    common = {
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "executable": executable,
        "timeout": timeout,
        "env": None if env is None else dict(env),
        "processors": int(processors),
        "memory_gb": memory_gb,
        "properties": ("energy",),
        # Frozen production contract: every numerical high-level displacement
        # starts from the converged central Gaussian checkpoint.  This reuses
        # both the SCF state and, for conventional CCSD/CCSD(T), the amplitudes
        # transformed by Gaussian to the displaced geometry.
        "restart_reuse_for_displacements": True,
        "gaussian_native_checkpoint_restart": True,
    }
    high = protocol.payload["high_level"]
    cbs = protocol.payload["cbs_correction"]
    cv = protocol.payload["core_valence"]

    def gdv_term(
        spec,
        *,
        gradient_mode: str,
        basis_file: Path | None = None,
    ) -> QMScanBackend:
        return QMScanBackend(
            name="gdv",
            method=str(spec["method"]),
            basis=str(spec.get("basis", cv["basis"])),
            route=str(spec["route"]),
            basis_file=basis_file,
            gradient_mode=gradient_mode,
            numerical_gradient_step_bohr=float(
                spec.get("numerical_gradient_step_bohr", high["numerical_gradient_step_bohr"])
            ),
            resolution={
                "frozen_composite_protocol_id": protocol.protocol_id,
                "frozen_composite_protocol_sha256": protocol.sha256,
            },
            **common,
        )

    lower = cbs["lower"]
    upper = cbs["upper"]
    all_electron = cv["all_electron"]
    frozen_core = cv["frozen_core"]
    high_label = str(high["label"])
    lower_label = str(lower["label"])
    upper_label = str(upper["label"])
    return QMScanBackend(
        name="linear_composite",
        charge=int(charge),
        multiplicity=int(multiplicity),
        gradient_mode="sonic_one_sided_hybrid",
        properties=("energy",),
        composite_terms=(
            QMBackendTerm(
                high_label,
                1.0,
                gdv_term(
                    high,
                    gradient_mode="energy",
                    basis_file=(protocol.basis_path if high.get("artifact") else None),
                ),
            ),
            QMBackendTerm(
                upper_label,
                float(upper["coefficient"]),
                gdv_term(
                    upper,
                    gradient_mode="analytic",
                    basis_file=(
                        protocol.cbs_upper_basis_path if upper.get("artifact") else None
                    ),
                ),
            ),
            QMBackendTerm(
                lower_label,
                float(lower["coefficient"]),
                gdv_term(
                    lower,
                    gradient_mode="analytic",
                    basis_file=(
                        protocol.cbs_lower_basis_path if lower.get("artifact") else None
                    ),
                ),
            ),
            QMBackendTerm(
                "ae-MP2/wC3",
                1.0,
                gdv_term(
                    {**all_electron, "basis": cv["basis"]},
                    gradient_mode="analytic",
                    basis_file=protocol.core_valence_basis_path,
                ),
            ),
            QMBackendTerm(
                "fc-MP2/wC3",
                -1.0,
                gdv_term(
                    {**frozen_core, "basis": cv["basis"]},
                    gradient_mode="analytic",
                    basis_file=protocol.core_valence_basis_path,
                ),
            ),
        ),
        composite_parallel_workers=int(composite_parallel_workers),
        resolution={
            "schema": JUNCHS_GDV_SCHEMA,
            "formula": str(protocol.payload["formula"]),
            "protocol_id": protocol.protocol_id,
            "protocol_selector": protocol.selector,
            "protocol_sha256": protocol.sha256,
            "implementation": "all_G16.C02",
            "correlation_model": "fc-CCSD(T)",
            "gradient_coordinates": "totally_symmetric_SONIC",
            "gradient_assembly": "one_sided_CCSD(T)_plus_analytic_MP2_CBS_and_core_valence",
            "term_derivative_sources": {
                high_label: "numerical_SONIC_totally_symmetric_one_sided",
                upper_label: "analytic_GDV",
                lower_label: "analytic_GDV",
                "ae-MP2/wC3": "analytic_GDV",
                "fc-MP2/wC3": "analytic_GDV",
            },
            "cbs_exponent": int(cbs["exponent"]),
            "cbs_cardinals": [int(cbs["lower_cardinal"]), int(cbs["upper_cardinal"])],
            "core_valence": "MP2_all_electron_minus_frozen_core",
            "state_reuse_contract": (
                "central_checkpoint_SCF_and_transformed_converged_CC_amplitudes_"
                "for_every_numerical_displacement"
            ),
        },
        restart_reuse_for_displacements=True,
    )


def junchs_cfour_backend(
    *,
    charge: int = 0,
    multiplicity: int = 1,
    executable: str | None = None,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    processors: int = 1,
    memory_gb: int | None = None,
    composite_parallel_workers: int = 1,
) -> QMScanBackend:
    """Return frozen jun-ChS with five analytic CFOUR gradients."""

    protocol = load_composite_protocol("junchs-cfour")
    if int(multiplicity) != 1:
        raise ValueError(f"frozen {protocol.protocol_id} currently requires a singlet RHF reference")
    common = {
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "executable": executable,
        "timeout": timeout,
        "env": None if env is None else dict(env),
        "processors": int(processors),
        "memory_gb": memory_gb,
        "properties": ("energy",),
    }
    high = protocol.payload["high_level"]
    cbs = protocol.payload["cbs_correction"]
    cv = protocol.payload["core_valence"]

    def cfour_term(spec, *, basis: str | None = None) -> QMScanBackend:
        return QMScanBackend(
            name="cfour",
            method=str(spec["method"]),
            basis=str(basis or spec["basis"]),
            basis_file=protocol.basis_path,
            route=str(spec["route"]),
            gradient_mode="analytic",
            resolution={
                "frozen_composite_protocol_id": protocol.protocol_id,
                "frozen_composite_protocol_sha256": protocol.sha256,
            },
            **common,
        )

    lower = cbs["lower"]
    upper = cbs["upper"]
    all_electron = cv["all_electron"]
    frozen_core = cv["frozen_core"]
    high_label = str(high["label"])
    lower_label = str(lower["label"])
    upper_label = str(upper["label"])
    return QMScanBackend(
        name="linear_composite",
        charge=int(charge),
        multiplicity=int(multiplicity),
        gradient_mode="analytic",
        properties=("energy",),
        composite_terms=(
            QMBackendTerm(high_label, 1.0, cfour_term(high)),
            QMBackendTerm(upper_label, float(upper["coefficient"]), cfour_term(upper)),
            QMBackendTerm(lower_label, float(lower["coefficient"]), cfour_term(lower)),
            QMBackendTerm(
                "ae-MP2/wC3",
                1.0,
                cfour_term(all_electron, basis=str(cv["basis"])),
            ),
            QMBackendTerm(
                "fc-MP2/wC3",
                -1.0,
                cfour_term(frozen_core, basis=str(cv["basis"])),
            ),
        ),
        composite_parallel_workers=int(composite_parallel_workers),
        resolution={
            "schema": JUNCHS_CFOUR_SCHEMA,
            "formula": str(protocol.payload["formula"]),
            "protocol_id": protocol.protocol_id,
            "protocol_selector": protocol.selector,
            "protocol_sha256": protocol.sha256,
            "basis_sha256": protocol.payload["orbital_basis"]["sha256"],
            "implementation": "all_CFOUR_2.00beta",
            "correlation_model": "fc-CCSD(T)",
            "gradient_coordinates": "Cartesian_to_SONIC",
            "gradient_assembly": "analytic_linear_composite",
            "term_derivative_sources": {
                high_label: "analytic_CFOUR",
                upper_label: "analytic_CFOUR",
                lower_label: "analytic_CFOUR",
                "ae-MP2/wC3": "analytic_CFOUR",
                "fc-MP2/wC3": "analytic_CFOUR",
            },
            "cbs_exponent": int(cbs["exponent"]),
            "cbs_cardinals": [int(cbs["lower_cardinal"]), int(cbs["upper_cardinal"])],
            "core_valence": "MP2_all_electron_minus_frozen_core",
        },
        restart_reuse_for_displacements=False,
    )


def junchs_molpro_backend(
    *,
    charge: int = 0,
    multiplicity: int = 1,
    executable: str | None = None,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    processors: int = 1,
    memory_gb: int | None = None,
    composite_parallel_workers: int = 1,
) -> QMScanBackend:
    """Return frozen conventional jun-ChS with five analytic Molpro gradients."""

    protocol = load_composite_protocol("junchs-molpro")
    if int(multiplicity) != 1:
        raise ValueError(f"frozen {protocol.protocol_id} currently requires a singlet RHF reference")
    common = {
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "executable": executable,
        "timeout": timeout,
        "env": None if env is None else dict(env),
        "processors": int(processors),
        "memory_gb": memory_gb,
        "properties": ("energy",),
        "restart_reuse_for_displacements": False,
    }
    high = protocol.payload["high_level"]
    cbs = protocol.payload["cbs_correction"]
    cv = protocol.payload["core_valence"]

    def molpro_term(
        spec,
        *,
        basis: str | None = None,
        basis_file: Path | None = None,
    ) -> QMScanBackend:
        return QMScanBackend(
            name="molpro",
            method=str(spec["method"]),
            basis=str(basis or spec["basis"]),
            basis_file=basis_file,
            route=str(spec["route"]),
            gradient_mode="analytic",
            resolution={
                "frozen_composite_protocol_id": protocol.protocol_id,
                "frozen_composite_protocol_sha256": protocol.sha256,
            },
            **common,
        )

    lower = cbs["lower"]
    upper = cbs["upper"]
    all_electron = cv["all_electron"]
    frozen_core = cv["frozen_core"]
    high_label = str(high["label"])
    lower_label = str(lower["label"])
    upper_label = str(upper["label"])
    return QMScanBackend(
        name="linear_composite",
        charge=int(charge),
        multiplicity=int(multiplicity),
        gradient_mode="analytic",
        properties=("energy",),
        composite_terms=(
            QMBackendTerm(
                high_label,
                1.0,
                molpro_term(high, basis_file=protocol.basis_path),
            ),
            QMBackendTerm(
                upper_label,
                float(upper["coefficient"]),
                molpro_term(upper, basis_file=protocol.cbs_upper_basis_path),
            ),
            QMBackendTerm(
                lower_label,
                float(lower["coefficient"]),
                molpro_term(lower, basis_file=protocol.cbs_lower_basis_path),
            ),
            QMBackendTerm(
                "ae-MP2/wC3",
                1.0,
                molpro_term(all_electron, basis=str(cv["basis"])),
            ),
            QMBackendTerm(
                "fc-MP2/wC3",
                -1.0,
                molpro_term(frozen_core, basis=str(cv["basis"])),
            ),
        ),
        composite_parallel_workers=int(composite_parallel_workers),
        resolution={
            "schema": JUNCHS_MOLPRO_SCHEMA,
            "formula": str(protocol.payload["formula"]),
            "protocol_id": protocol.protocol_id,
            "protocol_selector": protocol.selector,
            "protocol_sha256": protocol.sha256,
            "basis_sha256": {
                "jun-cc-pVTZ": lower["sha256"],
                "jun-cc-pVQZ": upper["sha256"],
            },
            "implementation": "all_MOLPRO_2025.3",
            "correlation_model": "fc-CCSD(T)",
            "gradient_coordinates": "Cartesian_to_SONIC",
            "gradient_assembly": "analytic_linear_composite",
            "term_derivative_sources": {
                high_label: "analytic_MOLPRO",
                upper_label: "analytic_MOLPRO",
                lower_label: "analytic_MOLPRO",
                "ae-MP2/wC3": "analytic_MOLPRO",
                "fc-MP2/wC3": "analytic_MOLPRO",
            },
            "cbs_exponent": int(cbs["exponent"]),
            "cbs_cardinals": [int(cbs["lower_cardinal"]), int(cbs["upper_cardinal"])],
            "core_valence": "MP2_all_electron_minus_frozen_core",
        },
        restart_reuse_for_displacements=False,
    )


def fpa_molpro_backend(
    selector: str,
    *,
    charge: int = 0,
    multiplicity: int = 1,
    executable: str | None = None,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    processors: int = 1,
    memory_gb: int | None = None,
    composite_parallel_workers: int = 1,
) -> QMScanBackend:
    """Return one frozen Warden--Sherrill FPA with explicit or empirical CV."""

    protocol = load_composite_protocol(selector)
    if protocol.selector not in {
        "fpa-q5-t-cv-molpro",
        "fpa-haq5-hat-cv-molpro",
        "sherrill-tq-dz-cv-molpro",
        "l2fpa-lite-molpro",
        "l2fpa-molpro",
        "cheap-lite-molpro",
    }:
        raise ValueError(f"unsupported Molpro focal-point protocol: {selector!r}")
    if int(multiplicity) != 1:
        raise ValueError(f"frozen {protocol.protocol_id} currently requires a singlet RHF reference")
    common = {
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "executable": executable,
        "timeout": timeout,
        "env": None if env is None else dict(env),
        "processors": int(processors),
        "memory_gb": memory_gb,
        "properties": ("energy",),
        "restart_reuse_for_displacements": False,
    }

    def molpro_term(spec, *, basis_file: Path | None) -> QMScanBackend:
        return QMScanBackend(
            name="molpro",
            method=str(spec["method"]),
            basis=str(spec.get("basis", "cc-pwCVTZ")),
            basis_file=basis_file,
            route=str(spec.get("route", "")),
            gradient_mode="analytic",
            resolution={
                "frozen_composite_protocol_id": protocol.protocol_id,
                "frozen_composite_protocol_sha256": protocol.sha256,
            },
            **common,
        )

    focal_terms = tuple(protocol.payload["focal_point"]["terms"])
    cv = protocol.payload["core_valence"]
    empirical_cv = cv.get("type") == "oracle_cv_exponential_field"
    terms = tuple(
        QMBackendTerm(
            str(spec["label"]),
            float(spec["coefficient"]),
            molpro_term(spec, basis_file=_BASIS_DATA / str(spec["artifact"])),
        )
        for spec in focal_terms
    )
    if not empirical_cv:
        terms += (
            QMBackendTerm(
                "ae-MP2/wC3",
                1.0,
                molpro_term(
                    {**cv["all_electron"], "basis": cv["basis"]},
                    basis_file=None,
                ),
            ),
            QMBackendTerm(
                "fc-MP2/wC3",
                -1.0,
                molpro_term(
                    {**cv["frozen_core"], "basis": cv["basis"]},
                    basis_file=None,
                ),
            ),
        )
    schema = {
        "fpa-q5-t-cv-molpro": FPA_Q5_T_CV_MOLPRO_SCHEMA,
        "fpa-haq5-hat-cv-molpro": FPA_HAQ5_HAT_CV_MOLPRO_SCHEMA,
        "sherrill-tq-dz-cv-molpro": SHERRILL_TQ_DZ_CV_MOLPRO_SCHEMA,
        "l2fpa-lite-molpro": L2FPA_LITE_MOLPRO_SCHEMA,
        "l2fpa-molpro": "matrix.link.l2fpa.molpro.v1",
        "cheap-lite-molpro": CHEAP_LITE_MOLPRO_SCHEMA,
    }[protocol.selector]
    derivative_sources = {
        str(spec["label"]): "analytic_MOLPRO" for spec in focal_terms
    }
    if empirical_cv:
        derivative_sources["ORACLE CV exponential field"] = "analytic_LINK"
    else:
        derivative_sources.update(
            {"ae-MP2/wC3": "analytic_MOLPRO", "fc-MP2/wC3": "analytic_MOLPRO"}
        )
    return QMScanBackend(
        name="linear_composite",
        charge=int(charge),
        multiplicity=int(multiplicity),
        gradient_mode="analytic",
        properties=("energy",),
        composite_terms=terms,
        composite_parallel_workers=int(composite_parallel_workers),
        resolution={
            "schema": schema,
            "formula": str(protocol.payload["formula"]),
            "protocol_id": protocol.protocol_id,
            "protocol_selector": protocol.selector,
            "protocol_sha256": protocol.sha256,
            "implementation": "all_MOLPRO_2025.3",
            "correlation_model": "fc-CCSD(T)",
            "gradient_coordinates": "Cartesian_to_SONIC",
            "gradient_assembly": (
                "analytic_total_MP2_CBS_plus_delta_plus_empirical_CV_field"
                if protocol.selector in {"l2fpa-lite-molpro", "cheap-lite-molpro"}
                else "analytic_correlation_only_CBS_plus_delta_plus_empirical_CV_field"
                if empirical_cv
                else "analytic_correlation_only_CBS_plus_delta_plus_core_valence"
            ),
            "term_derivative_sources": derivative_sources,
            "cbs_exponent": int(protocol.payload["focal_point"]["exponent"]),
            "cbs_cardinals": [
                int(protocol.payload["focal_point"]["lower_cardinal"]),
                int(protocol.payload["focal_point"]["upper_cardinal"]),
            ],
            "cbs_scf_source": str(protocol.payload["focal_point"]["scf_source"]),
            "core_valence": (
                "ORACLE_CV_radial_exponential"
                if empirical_cv
                else "MP2_all_electron_minus_frozen_core_cc-pwCVTZ"
            ),
            "literature_doi": protocol.payload["provenance"]["doi"],
        },
        restart_reuse_for_displacements=False,
    )


def pcs_orca_backend(
    *,
    charge: int = 0,
    multiplicity: int = 1,
    executable: str | None = None,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    processors: int = 1,
    memory_gb: int | None = None,
    composite_parallel_workers: int = 1,
    tight_pno: bool = True,
    coupled_cluster_approximation: str = "canonical",
    protocol_selector: str | None = None,
) -> QMScanBackend:
    """Return the all-ORCA PCS2 composite surface used by LINK.

    The canonical explicitly correlated coupled-cluster contribution and both
    conventional MP2 core-valence terms are evaluated at the same Cartesian
    geometry.  The coupled-cluster term is energy-only, while the MP2 terms
    advertise their analytic gradients to the general composite orchestrator.
    """

    common = {
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "executable": executable,
        "timeout": timeout,
        "env": None if env is None else dict(env),
        "processors": int(processors),
        "memory_gb": memory_gb,
        "properties": ("energy",),
    }
    approximation = str(coupled_cluster_approximation).strip().lower()
    if approximation not in {"canonical", "dlpno"}:
        raise ValueError("coupled_cluster_approximation must be 'canonical' or 'dlpno'")
    selector = protocol_selector or (
        "pcs2-orca" if approximation == "canonical" else "ppcs2-orca"
    )
    protocol = load_composite_protocol(selector)
    expected_selectors = (
        {"pcs2-orca"}
        if approximation == "canonical"
        else {"ppcs2-orca", "pl2-f12-orca", "pl2-orca"}
    )
    if protocol.selector not in expected_selectors:
        raise ValueError(
            f"protocol {protocol.selector!r} is incompatible with {approximation!r} ORCA"
        )
    if int(multiplicity) != 1:
        raise ValueError(f"frozen {protocol.protocol_id} currently requires a singlet RHF reference")
    if approximation == "dlpno" and not tight_pno:
        raise ValueError(f"frozen {protocol.protocol_id} requires TightPNO")
    high = protocol.payload["high_level"]
    high_method = str(high["method"])
    high_route = str(high["route"])
    high_label = str(high["label"])
    coupled_cluster = QMScanBackend(
        name="orca",
        method=high_method,
        basis=str(high["basis"]),
        basis_file=protocol.basis_path,
        route=high_route,
        gradient_mode="energy",
        numerical_gradient_step_bohr=float(high["numerical_gradient_step_bohr"]),
        restart_reuse_for_displacements=True,
        **common,
    )


    cv = protocol.payload["core_valence"]
    all_electron_mp2 = QMScanBackend(
        name="orca",
        method=str(cv["all_electron"]["method"]),
        basis=str(cv["basis"]),
        route=str(cv["all_electron"]["route"]),
        gradient_mode="analytic",
        restart_reuse_for_displacements=False,
        **common,
    )
    frozen_core_mp2 = QMScanBackend(
        name="orca",
        method=str(cv["frozen_core"]["method"]),
        basis=str(cv["basis"]),
        route=str(cv["frozen_core"]["route"]),
        gradient_mode="analytic",
        restart_reuse_for_displacements=False,
        **common,
    )
    local_f12_correction = protocol.payload.get("local_mp2_f12_correction")
    local_f12_terms: tuple[QMBackendTerm, ...] = ()
    local_f12_sources: dict[str, str] = {}
    if local_f12_correction is not None:
        conventional_f12 = local_f12_correction["conventional"]
        local_f12 = local_f12_correction["local"]

        def f12_mp2_backend(term_spec) -> QMScanBackend:
            return QMScanBackend(
                name="orca",
                method=str(term_spec["method"]),
                basis=str(term_spec["basis"]),
                basis_file=protocol.basis_path,
                route=str(term_spec["route"]),
                gradient_mode="energy",
                numerical_gradient_step_bohr=float(
                    term_spec["numerical_gradient_step_bohr"]
                ),
                restart_reuse_for_displacements=True,
                **common,
            )

        local_f12_terms = (
            QMBackendTerm(
                str(conventional_f12["label"]),
                1.0,
                f12_mp2_backend(conventional_f12),
            ),
            QMBackendTerm(
                str(local_f12["label"]),
                -1.0,
                f12_mp2_backend(local_f12),
            ),
        )
        local_f12_sources = {
            str(conventional_f12["label"]): "numerical_SONIC_totally_symmetric_one_sided",
            str(local_f12["label"]): "numerical_SONIC_totally_symmetric_one_sided",
        }
    return QMScanBackend(
        name="linear_composite",
        charge=int(charge),
        multiplicity=int(multiplicity),
        gradient_mode="sonic_one_sided_hybrid",
        properties=("energy",),
        composite_terms=(
            QMBackendTerm(high_label, +1.0, coupled_cluster),
            *local_f12_terms,
            QMBackendTerm("ae-MP2/wC3", +1.0, all_electron_mp2),
            QMBackendTerm("fc-MP2/wC3", -1.0, frozen_core_mp2),
        ),
        composite_parallel_workers=int(composite_parallel_workers),
        resolution={
            "schema": (
                PL2_ORCA_SCHEMA
                if protocol.selector == "pl2-orca"
                else PL2_F12_ORCA_SCHEMA
                if protocol.selector == "pl2-f12-orca"
                else PCS2_ORCA_SCHEMA
                if approximation == "canonical"
                else PPCS2_ORCA_SCHEMA
            ),
            "formula": str(protocol.payload["formula"]),
            "protocol_id": protocol.protocol_id,
            "protocol_selector": protocol.selector,
            "protocol_sha256": protocol.sha256,
            "basis_sha256": protocol.payload["orbital_basis"]["sha256"],
            "implementation": "all_ORCA",
            "gradient_coordinates": "totally_symmetric_SONIC",
            "gradient_assembly": (
                "one_sided_high_and_MP2_F12_recovery_plus_analytic_MP2_core_valence"
                if local_f12_correction is not None
                else "one_sided_high_level_plus_analytic_MP2_core_valence"
            ),
            "term_derivative_sources": {
                high_label: "numerical_SONIC_totally_symmetric_one_sided",
                **local_f12_sources,
                "ae-MP2/wC3": "analytic_ORCA",
                "fc-MP2/wC3": "analytic_ORCA",
            },
            "coupled_cluster_approximation": approximation,
            "core_valence": "conventional_MP2_all_electron_minus_frozen_core",
        },
        restart_reuse_for_displacements=True,
    )


def pcs_mrcc_backend(
    *,
    charge: int = 0,
    multiplicity: int = 1,
    executable: str | None = None,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    processors: int = 1,
    memory_gb: int | None = None,
    composite_parallel_workers: int = 1,
) -> QMScanBackend:
    """Return a PCS2 composite surface assembled from MRCC calculations.

    The explicitly correlated coupled-cluster contribution is evaluated as
    an energy surface and differentiated along SONIC coordinates.  Both
    conventional MP2 terms use MRCC analytic gradients.  The all-electron
    MP2 term carries ``core=corr`` in its MRCC route; the subtraction term uses
    the default frozen-core treatment.
    """

    protocol = load_composite_protocol("pcs2-mrcc")
    if int(multiplicity) != 1:
        raise ValueError(f"frozen {protocol.protocol_id} currently requires a singlet RHF reference")
    common = {
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "executable": executable,
        "timeout": timeout,
        "env": None if env is None else dict(env),
        "processors": int(processors),
        "memory_gb": memory_gb,
        "properties": ("energy",),
        "restart_reuse_for_displacements": False,
    }
    high = protocol.payload["high_level"]
    cv = protocol.payload["core_valence"]
    coupled_cluster = QMScanBackend(
        name="mrcc",
        method=str(high["method"]),
        basis=str(high["basis"]),
        basis_file=protocol.basis_path,
        route=str(high["route"]),
        gradient_mode="energy",
        numerical_gradient_step_bohr=float(high["numerical_gradient_step_bohr"]),
        **common,
    )

    all_electron_mp2 = QMScanBackend(
        name="mrcc", method=str(cv["all_electron"]["method"]), basis=str(cv["basis"]),
        route=str(cv["all_electron"]["route"]), gradient_mode="analytic", **common,
    )
    frozen_core_mp2 = QMScanBackend(
        name="mrcc", method=str(cv["frozen_core"]["method"]), basis=str(cv["basis"]),
        route=str(cv["frozen_core"]["route"]), gradient_mode="analytic", **common,
    )
    return QMScanBackend(
        name="linear_composite", charge=int(charge), multiplicity=int(multiplicity),
        gradient_mode="sonic_one_sided_hybrid", properties=("energy",),
        composite_terms=(
            QMBackendTerm(str(high["label"]), +1.0, coupled_cluster),
            QMBackendTerm("ae-MP2/wC3", +1.0, all_electron_mp2),
            QMBackendTerm("fc-MP2/wC3", -1.0, frozen_core_mp2),
        ),
        composite_parallel_workers=int(composite_parallel_workers),
        resolution={
            "schema": PCS2_MRCC_SCHEMA,
            "formula": str(protocol.payload["formula"]),
            "protocol_id": protocol.protocol_id,
            "protocol_selector": protocol.selector,
            "protocol_sha256": protocol.sha256,
            "basis_sha256": protocol.payload["orbital_basis"]["sha256"],
            "implementation": "all_MRCC",
            "gradient_coordinates": "totally_symmetric_SONIC",
            "gradient_assembly": "one_sided_CCSD(F12*)(T+)_plus_analytic_MP2_core_valence",
            "term_derivative_sources": {
                str(high["label"]): "numerical_SONIC_totally_symmetric_one_sided",
                "ae-MP2/wC3": "analytic_MRCC",
                "fc-MP2/wC3": "analytic_MRCC",
            },
            "core_valence": "conventional_MP2_all_electron_minus_frozen_core",
            "triples": "T+",
            "F12_ansatz": "F12*",
        },
        restart_reuse_for_displacements=False,
    )


def ppcs2_mrcc_backend(**kwargs) -> QMScanBackend:
    """Return the frozen MRCC PPCS2 LNO + delta-MP2-F12 surface."""

    protocol = load_composite_protocol("ppcs2-mrcc")
    multiplicity = int(kwargs.get("multiplicity", 1))
    if multiplicity != 1:
        raise ValueError(f"frozen {protocol.protocol_id} currently requires a singlet reference")
    common = {
        "charge": int(kwargs.get("charge", 0)),
        "multiplicity": multiplicity,
        "executable": kwargs.get("executable"),
        "timeout": kwargs.get("timeout"),
        "env": kwargs.get("env"),
        "processors": int(kwargs.get("processors", 1)),
        "memory_gb": kwargs.get("memory_gb"),
        "properties": ("energy",),
        "restart_reuse_for_displacements": False,
    }
    high = protocol.payload["high_level"]
    correction = protocol.payload["f12_correction"]
    core_valence = protocol.payload["core_valence"]

    def term_backend(spec, *, gradient_mode: str) -> QMScanBackend:
        return QMScanBackend(
            name="mrcc",
            method=str(spec["method"]),
            basis=str(spec["basis"]),
            basis_file=protocol.basis_path,
            route=str(spec["route"]),
            gradient_mode=gradient_mode,
            numerical_gradient_step_bohr=float(
                spec.get(
                    "numerical_gradient_step_bohr",
                    high["numerical_gradient_step_bohr"],
                )
            ),
            **common,
        )

    f12 = correction["explicitly_correlated"]
    conventional = correction["conventional"]
    all_electron = core_valence["all_electron"]
    frozen_core = core_valence["frozen_core"]
    return QMScanBackend(
        name="linear_composite",
        charge=common["charge"],
        multiplicity=multiplicity,
        gradient_mode="sonic_one_sided_hybrid",
        properties=("energy",),
        composite_terms=(
            QMBackendTerm(str(high["label"]), 1.0, term_backend(high, gradient_mode="energy")),
            QMBackendTerm(str(f12["label"]), 1.0, term_backend(f12, gradient_mode="energy")),
            QMBackendTerm(str(conventional["label"]), -1.0, term_backend(conventional, gradient_mode="analytic")),
            QMBackendTerm("ae-MP2/wC3", 1.0, QMScanBackend(name="mrcc", method=str(all_electron["method"]), basis=str(core_valence["basis"]), route=str(all_electron["route"]), gradient_mode="analytic", **common)),
            QMBackendTerm("fc-MP2/wC3", -1.0, QMScanBackend(name="mrcc", method=str(frozen_core["method"]), basis=str(core_valence["basis"]), route=str(frozen_core["route"]), gradient_mode="analytic", **common)),
        ),
        composite_parallel_workers=int(kwargs.get("composite_parallel_workers", 1)),
        resolution={
            "schema": PPCS2_MRCC_SCHEMA,
            "formula": str(protocol.payload["formula"]),
            "protocol_id": protocol.protocol_id,
            "protocol_selector": protocol.selector,
            "protocol_sha256": protocol.sha256,
            "implementation": "LNO_MRCC_delta_MP2_F12",
            "correlation_model": "LNO-CCSD(T)",
            "gradient_coordinates": "totally_symmetric_SONIC",
            "gradient_assembly": "one_sided_LNO_and_MP2_F12_minus_analytic_MP2_plus_analytic_MP2_core_valence",
            "term_derivative_sources": {
                str(high["label"]): "numerical_SONIC_totally_symmetric_one_sided",
                str(f12["label"]): "numerical_SONIC_totally_symmetric_one_sided",
                str(conventional["label"]): "analytic_MRCC",
                "ae-MP2/wC3": "analytic_MRCC",
                "fc-MP2/wC3": "analytic_MRCC",
            },
        },
        restart_reuse_for_displacements=False,
    )


def _molpro_composite(
    high_method: str,
    high_label: str,
    schema: str,
    formula: str,
    selector: str,
    **kwargs,
) -> QMScanBackend:
    protocol = load_composite_protocol(selector)
    if int(kwargs.get("multiplicity", 1)) != 1:
        raise ValueError(f"frozen {protocol.protocol_id} currently requires a singlet RHF reference")
    high = protocol.payload["high_level"]
    cv = protocol.payload["core_valence"]
    if (high_method, high_label, formula) != (
        high["method"],
        high["label"],
        protocol.payload["formula"],
    ):
        raise RuntimeError(f"implementation drift detected for {protocol.protocol_id}")
    # Frozen production contract: all numerical Molpro terms read the private
    # copy-on-write WFU cloned from the central point, including orbitals and
    # converged coupled-cluster amplitudes.
    common = {"charge": kwargs.get("charge", 0), "multiplicity": kwargs.get("multiplicity", 1), "executable": kwargs.get("executable"), "timeout": kwargs.get("timeout"), "env": kwargs.get("env"), "processors": kwargs.get("processors", 1), "memory_gb": kwargs.get("memory_gb"), "properties": ("energy",), "restart_reuse_for_displacements": True}
    high_is_analytic = high["derivative"] == "analytic"
    terms: tuple[QMBackendTerm, ...] = (
        QMBackendTerm(str(high["label"]), 1.0, QMScanBackend(name="molpro", method=str(high["method"]), basis=str(high["basis"]), basis_file=protocol.basis_path, route=str(high["route"]), gradient_mode="analytic" if high_is_analytic else "energy", numerical_gradient_step_bohr=float(high.get("numerical_gradient_step_bohr", 1.0e-3)), **common)),
    )
    derivative_sources = {
        str(high["label"]): (
            "analytic_Molpro"
            if high_is_analytic
            else "numerical_SONIC_totally_symmetric_one_sided"
        )
    }
    local_f12_correction = protocol.payload.get("local_mp2_f12_correction")
    if local_f12_correction is not None:
        for role, coefficient in (("conventional", 1.0), ("local", -1.0)):
            term_spec = local_f12_correction[role]
            terms += (
                QMBackendTerm(
                    str(term_spec["label"]),
                    coefficient,
                    QMScanBackend(
                        name="molpro",
                        method=str(term_spec["method"]),
                        basis=str(term_spec["basis"]),
                        basis_file=protocol.basis_path,
                        route=str(term_spec["route"]),
                        gradient_mode="energy",
                        numerical_gradient_step_bohr=float(
                            term_spec["numerical_gradient_step_bohr"]
                        ),
                        **common,
                    ),
                ),
            )
            derivative_sources[str(term_spec["label"])] = (
                "numerical_SONIC_totally_symmetric_one_sided"
            )
    terms += (
        QMBackendTerm("ae-MP2/wC3", 1.0, QMScanBackend(name="molpro", method=str(cv["all_electron"]["method"]), basis=str(cv["basis"]), route=str(cv["all_electron"]["route"]), gradient_mode="analytic", **common)),
        QMBackendTerm("fc-MP2/wC3", -1.0, QMScanBackend(name="molpro", method=str(cv["frozen_core"]["method"]), basis=str(cv["basis"]), route=str(cv["frozen_core"]["route"]), gradient_mode="analytic", **common)),
    )
    derivative_sources.update(
        {"ae-MP2/wC3": "analytic_Molpro", "fc-MP2/wC3": "analytic_Molpro"}
    )
    return QMScanBackend(name="linear_composite", charge=common["charge"], multiplicity=common["multiplicity"], gradient_mode="analytic" if high_is_analytic else "sonic_one_sided_hybrid", properties=("energy",), composite_terms=terms, composite_parallel_workers=int(kwargs.get("composite_parallel_workers", 1)), resolution={"schema": schema, "formula": formula, "protocol_id": protocol.protocol_id, "protocol_selector": protocol.selector, "protocol_sha256": protocol.sha256, "basis_sha256": protocol.payload["orbital_basis"]["sha256"], "implementation": "Molpro", "gradient_coordinates": "Cartesian_to_SONIC" if high_is_analytic else "totally_symmetric_SONIC", "gradient_assembly": "analytic_linear_composite" if high_is_analytic else "one_sided_high_and_MP2_F12_recovery_plus_analytic_MP2_core_valence" if local_f12_correction is not None else "one_sided_high_plus_analytic_MP2_core_valence", "term_derivative_sources": derivative_sources, "state_reuse_contract": "central_WFU_orbitals_and_converged_CC_amplitudes_for_every_numerical_displacement"}, restart_reuse_for_displacements=True)


def pcs2_molpro_backend(
    *, protocol_selector: str = "pcs2-molpro", **kwargs
) -> QMScanBackend:
    protocol = load_composite_protocol(protocol_selector)
    high = protocol.payload["high_level"]
    return _molpro_composite(
        str(high["method"]),
        str(high["label"]),
        str(protocol.payload["protocol_id"]),
        str(protocol.payload["formula"]),
        protocol_selector,
        **kwargs,
    )


def ppcs2_molpro_backend(
    *, protocol_selector: str = "ppcs2-molpro", **kwargs
) -> QMScanBackend:
    final_pl2_f12 = protocol_selector == "pl2-f12-molpro"
    backend = _molpro_composite(
        "PNO-LCCSD(T)-F12,SCALE_TRIP=1",
        "PNO-LCCSD(T*)-F12b",
        PL2_F12_MOLPRO_SCHEMA if final_pl2_f12 else PPCS2_MOLPRO_SCHEMA,
        PL2_F12_MOLPRO_FORMULA if final_pl2_f12 else PPCS2_MOLPRO_FORMULA,
        protocol_selector,
        **kwargs,
    )
    return QMScanBackend(
        **{
            **backend.__dict__,
            "resolution": {
                **backend.resolution,
                "correlation_model": "PNO-LCCSD(T*)-F12b",
                "triples": "T*",
                "F12_ansatz": "F12b",
            },
        }
    )


def l2_molpro_backend(**kwargs) -> QMScanBackend:
    """Return the frozen canonical L2/def2-TZVPP surface with explicit MP2 CV."""

    return _molpro_composite(
        "CCSD(T)",
        "fc-CCSD(T)/def2-TZVPP",
        L2_MOLPRO_SCHEMA,
        L2_MOLPRO_FORMULA,
        "l2-molpro",
        **kwargs,
    )


def pl2_orca_backend(**kwargs) -> QMScanBackend:
    """Return the frozen DLPNO PL2/def2-TZVPP surface with explicit MP2 CV."""

    return pcs_orca_backend(
        coupled_cluster_approximation="dlpno",
        protocol_selector="pl2-orca",
        **kwargs,
    )


def pl2_mp2_orca_backend(**kwargs) -> QMScanBackend:
    """Return PL2-MP2 with local recovery, conventional MP2 CBS, and CV."""

    protocol = load_composite_protocol("pl2-mp2-orca")
    multiplicity = int(kwargs.get("multiplicity", 1))
    if multiplicity != 1:
        raise ValueError(f"frozen {protocol.protocol_id} currently requires a singlet RHF reference")
    common = {
        "charge": int(kwargs.get("charge", 0)),
        "multiplicity": multiplicity,
        "executable": kwargs.get("executable"),
        "timeout": kwargs.get("timeout"),
        "env": kwargs.get("env"),
        "processors": int(kwargs.get("processors", 1)),
        "memory_gb": kwargs.get("memory_gb"),
        "properties": ("energy",),
    }
    high = protocol.payload["high_level"]
    local = protocol.payload["local_mp2_correction"]
    cbs = protocol.payload["cbs_correction"]
    cv = protocol.payload["core_valence"]

    def term_backend(spec, *, gradient_mode: str, basis_file: Path | None = None):
        return QMScanBackend(
            name="orca",
            method=str(spec["method"]),
            basis=str(spec.get("basis", cv["basis"])),
            basis_file=basis_file,
            route=str(spec["route"]),
            gradient_mode=gradient_mode,
            numerical_gradient_step_bohr=float(
                spec.get("numerical_gradient_step_bohr", high["numerical_gradient_step_bohr"])
            ),
            restart_reuse_for_displacements=(gradient_mode == "energy"),
            resolution={
                "frozen_composite_protocol_id": protocol.protocol_id,
                "frozen_composite_protocol_sha256": protocol.sha256,
            },
            **common,
        )

    conventional_tz = local["conventional"]
    local_tz = local["local"]
    upper = cbs["upper"]
    # The conventional TZ term occurs once in the local recovery and once
    # with -64/37 in the CBS increment.  Collect it exactly to avoid a
    # duplicate electronic-structure calculation.
    conventional_tz_coefficient = 1.0 + float(cbs["lower"]["coefficient"])
    terms = (
        QMBackendTerm(
            str(high["label"]),
            1.0,
            term_backend(high, gradient_mode="energy", basis_file=protocol.basis_path),
        ),
        QMBackendTerm(
            str(local_tz["label"]),
            -1.0,
            term_backend(local_tz, gradient_mode="analytic", basis_file=protocol.basis_path),
        ),
        QMBackendTerm(
            str(conventional_tz["label"]),
            conventional_tz_coefficient,
            term_backend(
                conventional_tz,
                gradient_mode="analytic",
                basis_file=protocol.cbs_lower_basis_path,
            ),
        ),
        QMBackendTerm(
            str(upper["label"]),
            float(upper["coefficient"]),
            term_backend(
                upper,
                gradient_mode="analytic",
                basis_file=protocol.cbs_upper_basis_path,
            ),
        ),
        QMBackendTerm(
            "ae-MP2/wC3",
            1.0,
            term_backend(cv["all_electron"], gradient_mode="analytic"),
        ),
        QMBackendTerm(
            "fc-MP2/wC3",
            -1.0,
            term_backend(cv["frozen_core"], gradient_mode="analytic"),
        ),
    )
    return QMScanBackend(
        name="linear_composite",
        charge=common["charge"],
        multiplicity=multiplicity,
        processors=common["processors"],
        memory_gb=common["memory_gb"],
        gradient_mode="sonic_one_sided_hybrid",
        properties=("energy",),
        composite_terms=terms,
        composite_parallel_workers=int(kwargs.get("composite_parallel_workers", 1)),
        resolution={
            "schema": PL2_MP2_ORCA_SCHEMA,
            "formula": str(protocol.payload["formula"]),
            "protocol_id": protocol.protocol_id,
            "protocol_selector": protocol.selector,
            "protocol_sha256": protocol.sha256,
            "implementation": "DLPNO_ORCA_local_recovery_plus_conventional_MP2_CBS",
            "gradient_coordinates": "totally_symmetric_SONIC",
            "gradient_assembly": "one_sided_DLPNO_CCSD(T)_plus_analytic_local_recovery_CBS_CV",
            "algebraic_collection": {
                str(conventional_tz["label"]): conventional_tz_coefficient,
            },
            "term_derivative_sources": {
                str(high["label"]): "numerical_SONIC_totally_symmetric_one_sided",
                str(local_tz["label"]): "analytic_ORCA",
                str(conventional_tz["label"]): "analytic_ORCA",
                str(upper["label"]): "analytic_ORCA",
                "ae-MP2/wC3": "analytic_ORCA",
                "fc-MP2/wC3": "analytic_ORCA",
            },
        },
        restart_reuse_for_displacements=True,
    )


def fast_molpro_backend(**kwargs) -> QMScanBackend:
    """Return CCSD(T)/def2-SVP plus numerical DF-BASCORR for the fast rung.

    The frozen empirical CV field is applied by the LINK optimizer, where its
    analytic energy, gradient, and Hessian are available independently of the
    electronic-structure backend.
    """

    protocol = load_composite_protocol("fast-molpro")
    if int(kwargs.get("multiplicity", 1)) != 1:
        raise ValueError(f"frozen {protocol.protocol_id} currently requires a singlet RHF reference")
    high = protocol.payload["high_level"]
    correction = protocol.payload["density_based_correction"]
    common = {
        "charge": int(kwargs.get("charge", 0)),
        "multiplicity": int(kwargs.get("multiplicity", 1)),
        "executable": kwargs.get("executable"),
        "timeout": kwargs.get("timeout"),
        "env": kwargs.get("env"),
        "processors": int(kwargs.get("processors", 1)),
        "memory_gb": kwargs.get("memory_gb"),
        "properties": ("energy",),
        "restart_reuse_for_displacements": False,
    }
    high_backend = QMScanBackend(
        name="molpro",
        method=str(high["method"]),
        basis=str(high["basis"]),
        basis_file=protocol.basis_path,
        route=str(high["route"]),
        gradient_mode="analytic",
        **common,
    )
    correction_backend = QMScanBackend(
        name="molpro",
        method=str(correction["method"]),
        basis=str(correction["basis"]),
        basis_file=protocol.basis_path,
        gradient_mode="energy",
        numerical_gradient_step_bohr=float(correction["numerical_gradient_step_bohr"]),
        **common,
    )
    return QMScanBackend(
        name="linear_composite",
        charge=common["charge"],
        multiplicity=common["multiplicity"],
        gradient_mode="sonic_one_sided_hybrid",
        properties=("energy",),
        composite_terms=(
            QMBackendTerm(str(high["label"]), 1.0, high_backend),
            QMBackendTerm(str(correction["label"]), 1.0, correction_backend),
        ),
        composite_parallel_workers=int(kwargs.get("composite_parallel_workers", 1)),
        resolution={
            "schema": FAST_MOLPRO_SCHEMA,
            "formula": str(protocol.payload["formula"]),
            "protocol_id": protocol.protocol_id,
            "protocol_selector": protocol.selector,
            "protocol_sha256": protocol.sha256,
            "basis_sha256": protocol.payload["orbital_basis"]["sha256"],
            "implementation": "MOLPRO_DF_BASCORR_plus_ORACLE_CV_field",
            "source_focal_point": protocol.payload["focal_point"]["source_model"],
            "source_doi": protocol.payload["focal_point"]["source_doi"],
            "algebraic_reduction": protocol.payload["focal_point"]["algebraic_reduction"],
            "gradient_coordinates": "totally_symmetric_SONIC",
            "gradient_assembly": "analytic_CCSD(T)_plus_one_sided_DF_BASCORR_plus_analytic_CV_field",
            "term_derivative_sources": {
                str(high["label"]): "analytic_Molpro",
                str(correction["label"]): "numerical_SONIC_totally_symmetric_one_sided",
                "ORACLE CV exponential field": "analytic_LINK",
            },
        },
        restart_reuse_for_displacements=False,
    )


def lite_molpro_backend(*, protocol_selector: str, **kwargs) -> QMScanBackend:
    """Return one frozen L2_lite/PL2_lite MOLPRO composite surface."""

    protocol = load_composite_protocol(protocol_selector)
    supported = {
        "pl2-lite-f12-molpro",
        "pl2-lite-mp2-molpro",
        "pl2-lite-db-molpro",
        "l2-lite-db-molpro",
        "l2-lite-f12-molpro",
        "l2f12-lite-hybrid-molpro",
        "pl2f12-lite-hybrid-molpro",
        "pl2f12-may-lite-molpro",
        "pfpa-lite-molpro",
        "l2f12-may-lite-molpro",
    }
    if protocol.selector not in supported:
        raise ValueError(f"unsupported lite MOLPRO protocol: {protocol.selector!r}")
    if int(kwargs.get("multiplicity", 1)) != 1:
        raise ValueError(f"frozen {protocol.protocol_id} currently requires a singlet RHF reference")

    high = protocol.payload["high_level"]
    common = {
        "charge": int(kwargs.get("charge", 0)),
        "multiplicity": int(kwargs.get("multiplicity", 1)),
        "executable": kwargs.get("executable"),
        "timeout": kwargs.get("timeout"),
        "env": kwargs.get("env"),
        "processors": int(kwargs.get("processors", 1)),
        "memory_gb": kwargs.get("memory_gb"),
        "properties": ("energy",),
        "restart_reuse_for_displacements": True,
    }

    def term_backend(spec, *, gradient_mode: str, basis_file: Path) -> QMScanBackend:
        return QMScanBackend(
            name="molpro",
            method=str(spec["method"]),
            basis=str(spec["basis"]),
            basis_file=basis_file,
            route=str(spec.get("route", "")),
            gradient_mode=gradient_mode,
            numerical_gradient_step_bohr=float(
                spec.get("numerical_gradient_step_bohr", 1.0e-3)
            ),
            resolution={
                "frozen_composite_protocol_id": protocol.protocol_id,
                "frozen_composite_protocol_sha256": protocol.sha256,
            },
            **common,
        )

    terms: tuple[QMBackendTerm, ...] = (
        QMBackendTerm(
            str(high["label"]),
            1.0,
            term_backend(high, gradient_mode="energy", basis_file=protocol.basis_path),
        ),
    )
    derivative_sources = {
        str(high["label"]): "numerical_SONIC_totally_symmetric_one_sided",
        "ORACLE CV exponential field": "analytic_LINK",
    }

    large_basis_reference = protocol.payload.get("mp2_reference")
    if large_basis_reference is None:
        large_basis_reference = protocol.payload.get("hf_reference")
    if large_basis_reference is not None:
        reference_basis = _BASIS_DATA / str(large_basis_reference["artifact"])
        terms += (
            QMBackendTerm(
                str(large_basis_reference["label"]),
                float(large_basis_reference["coefficient"]),
                term_backend(
                    large_basis_reference,
                    gradient_mode="analytic",
                    basis_file=reference_basis,
                ),
            ),
        )
        derivative_sources[str(large_basis_reference["label"])] = "analytic_MOLPRO"

    for reference in protocol.payload.get("mp2_cbs_terms", ()):
        reference_basis = _BASIS_DATA / str(reference["artifact"])
        terms += (
            QMBackendTerm(
                str(reference["label"]),
                float(reference["coefficient"]),
                term_backend(reference, gradient_mode="analytic", basis_file=reference_basis),
            ),
        )
        derivative_sources[str(reference["label"])] = "analytic_MOLPRO"

    local_f12 = protocol.payload.get("local_mp2_f12_correction")
    if local_f12 is not None:
        for role, coefficient in (("conventional", 1.0), ("local", -1.0)):
            reference = local_f12[role]
            terms += (
                QMBackendTerm(
                    str(reference["label"]),
                    coefficient,
                    term_backend(
                        reference,
                        gradient_mode="energy",
                        basis_file=protocol.basis_path,
                    ),
                ),
            )
            derivative_sources[str(reference["label"])] = (
                "numerical_SONIC_totally_symmetric_one_sided"
            )

    density_based = protocol.payload.get("density_based_correction")
    if density_based is not None:
        terms += (
            QMBackendTerm(
                str(density_based["label"]),
                1.0,
                term_backend(
                    density_based,
                    gradient_mode="energy",
                    basis_file=protocol.basis_path,
                ),
            ),
        )
        derivative_sources[str(density_based["label"])] = (
            "numerical_SONIC_totally_symmetric_one_sided"
        )

    schema = {
        "pl2-lite-f12-molpro": PL2_LITE_F12_MOLPRO_SCHEMA,
        "pl2-lite-mp2-molpro": PL2_LITE_MP2_MOLPRO_SCHEMA,
        "pl2-lite-db-molpro": PL2_LITE_DB_MOLPRO_SCHEMA,
        "l2-lite-db-molpro": L2_LITE_DB_MOLPRO_SCHEMA,
        "l2-lite-f12-molpro": L2_LITE_F12_MOLPRO_SCHEMA,
        "pl2f12-may-lite-molpro": "matrix.link.pl2f12_may_lite.molpro.v1",
    }.get(protocol.selector, protocol.protocol_id)
    return QMScanBackend(
        name="linear_composite",
        charge=common["charge"],
        multiplicity=common["multiplicity"],
        gradient_mode="sonic_one_sided_hybrid",
        properties=("energy",),
        composite_terms=terms,
        composite_parallel_workers=int(kwargs.get("composite_parallel_workers", 1)),
        resolution={
            "schema": schema,
            "formula": str(protocol.payload["formula"]),
            "protocol_id": protocol.protocol_id,
            "protocol_selector": protocol.selector,
            "protocol_sha256": protocol.sha256,
            "basis_sha256": protocol.payload["orbital_basis"]["sha256"],
            "implementation": "MOLPRO_lite_composite",
            "gradient_coordinates": "totally_symmetric_SONIC",
            "gradient_assembly": "one_sided_electronic_terms_plus_analytic_HF_QZ_and_CV_field",
            "term_derivative_sources": derivative_sources,
            "cabs_singles": high.get("cabs_singles", "not_applicable"),
            "core_valence": "ORACLE_CV_radial_exponential",
        },
        restart_reuse_for_displacements=True,
    )


__all__ = [
    "FPA_HAQ5_HAT_CV_MOLPRO_SCHEMA",
    "FPA_Q5_T_CV_MOLPRO_SCHEMA",
    "L2FPA_LITE_MOLPRO_SCHEMA",
    "CHEAP_LITE_MOLPRO_SCHEMA",
    "SHERRILL_TQ_DZ_CV_MOLPRO_SCHEMA",
    "FAST_MOLPRO_FORMULA",
    "FAST_MOLPRO_SCHEMA",
    "PL2_LITE_F12_MOLPRO_SCHEMA",
    "PL2_LITE_MP2_MOLPRO_SCHEMA",
    "PL2_LITE_DB_MOLPRO_SCHEMA",
    "L2_LITE_DB_MOLPRO_SCHEMA",
    "L2_LITE_F12_MOLPRO_SCHEMA",
    "JUNCHS_CFOUR_FORMULA",
    "JUNCHS_CFOUR_SCHEMA",
    "JUNCHS_GDV_FORMULA",
    "JUNCHS_GDV_SCHEMA",
    "JUNCHS_MOLPRO_FORMULA",
    "JUNCHS_MOLPRO_SCHEMA",
    "L2_MOLPRO_FORMULA",
    "L2_MOLPRO_SCHEMA",
    "PL2_ORCA_FORMULA",
    "PL2_ORCA_SCHEMA",
    "PL2_MP2_ORCA_FORMULA",
    "PL2_MP2_ORCA_SCHEMA",
    "PL2_F12_ORCA_FORMULA",
    "PL2_F12_ORCA_SCHEMA",
    "PL2_F12_MOLPRO_FORMULA",
    "PL2_F12_MOLPRO_SCHEMA",
    "PPCS2_MRCC_FORMULA",
    "PPCS2_MRCC_SCHEMA",
    "PCS2_MRCC_FORMULA",
    "PCS2_MRCC_SCHEMA",
    "PCS2_ORCA_FORMULA",
    "PCS2_ORCA_SCHEMA",
    "PPCS2_ORCA_FORMULA",
    "PPCS2_ORCA_SCHEMA",
    "pcs_mrcc_backend",
    "ppcs2_mrcc_backend",
    "PCS2_MOLPRO_FORMULA",
    "PCS2_MOLPRO_SCHEMA",
    "PPCS2_MOLPRO_FORMULA",
    "PPCS2_MOLPRO_SCHEMA",
    "pcs2_molpro_backend",
    "ppcs2_molpro_backend",
    "pcs_orca_backend",
    "junchs_cfour_backend",
    "junchs_gdv_backend",
    "junchs_molpro_backend",
    "fpa_molpro_backend",
    "fast_molpro_backend",
    "lite_molpro_backend",
    "l2_molpro_backend",
    "pl2_orca_backend",
    "pl2_mp2_orca_backend",
]
