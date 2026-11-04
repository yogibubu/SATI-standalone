from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from matrix_chem import (
    Structure,
    read_enriched_xyz,
    rotational_symmetry_number,
)
from matrix_chem.rotational import rotational_info
from matrix_core import has_section, read_basic_section

from .contracts import (
    RotationalSection,
    read_deltabvib_section,
    read_rotational_section,
    resolved_rotational_constants_MHz,
    write_rotational_section,
)
from .coriolis import CoriolisResult, compute_coriolis_from_xyzin, coriolis_report_lines
from .qcent import QCentResult, compute_qcent_from_xyzin, qcent_report_lines


@dataclass(frozen=True)
class RotationalAnalysisResult:
    xyzin: Path
    rotational: RotationalSection
    equilibrium_MHz: tuple[float, float, float]
    active_MHz: tuple[float, float, float]
    kappa: float
    q_rot: float | None
    qcent: QCentResult | None = None
    coriolis: CoriolisResult | None = None
    report: Path | None = None
    warnings: tuple[str, ...] = ()


def preferred_asymmetric_top_representation(kappa: float) -> str:
    """Choose the stable principal-axis representation from Ray's kappa.

    MORPHEUS uses representation I-r for the prolate half of the asymmetric
    top interval and III-r for the oblate half.  The boundary at kappa=0 is
    explicit and deterministic; callers can still preserve a user-specified
    representation from the input contract.
    """

    value = float(kappa)
    if not -1.0 - 1.0e-10 <= value <= 1.0 + 1.0e-10:
        raise ValueError("Ray asymmetry parameter kappa must lie in [-1, 1]")
    return "Ir" if value <= 0.0 else "IIIr"


def analyze_rotational_state(
    xyzin: Path | str,
    *,
    report: bool = True,
    report_path: Path | str | None = None,
    include_vibrational_analysis: bool = True,
    coriolis_threshold_cm1: float = 1.0,
) -> RotationalAnalysisResult:
    """Populate the normalized rotational state from the equilibrium geometry."""
    target = Path(xyzin)
    if target.is_dir():
        raise IsADirectoryError(f"rotational analysis expects an xyzin file, got: {target}")
    geometry = read_enriched_xyz(target)
    structure = Structure(
        symbols=list(geometry.atoms),
        coords=[tuple(row) for row in geometry.coordinates_angstrom],
        isotopes=None,
    )
    info = rotational_info(structure, isotopic=True)
    equilibrium = (float(info["A"]), float(info["B"]), float(info["C"]))
    basic = read_basic_section(target)
    existing = read_rotational_section(target)
    point_group = existing.point_group or basic.point_group or "C1"
    try:
        sigma = rotational_symmetry_number(point_group)
    except ValueError:
        sigma = existing.symmetry_number or 1
    representation = existing.representation or preferred_asymmetric_top_representation(
        float(info["kappa"])
    )
    section = replace(
        existing,
        rotor_type=info["rotor_type"],
        representation=representation,
        point_group=point_group,
        watson_reduction=existing.watson_reduction or basic.watson_reduction,
        symmetry_number=sigma,
        temperature_K=existing.temperature_K or basic.temperature_K,
        pressure_atm=existing.pressure_atm or basic.pressure_atm,
        A_MHz=equilibrium[0],
        B_MHz=equilibrium[1],
        C_MHz=equilibrium[2],
        equilibrium_MHz=equilibrium,
        constant_kind="equilibrium",
        constant_source="equilibrium-geometry",
    )
    write_rotational_section(target, section)
    deltabvib = read_deltabvib_section(target) if has_section(target, "DELTABVIB") else None
    active_raw = resolved_rotational_constants_MHz(
        section, state="ground_state", deltabvib=deltabvib
    )
    active = tuple(float(value if value is not None else 0.0) for value in active_raw)

    q_rot = None
    warnings: list[str] = []
    try:
        from matrix_thermo import rotational_thermo

        thermo = rotational_thermo(
            active[0],
            active[1],
            active[2],
            section.rotor_type,
            T_K=float(section.temperature_K or basic.temperature_K),
            sigma=section.symmetry_number,
        )
        q_rot = thermo.Q_dimless
        section = replace(section, q_rot=q_rot)
        write_rotational_section(target, section)
    except (ImportError, ValueError, ArithmeticError) as exc:
        warnings.append(f"rotational thermochemistry unavailable: {exc}")

    qcent = None
    coriolis = None
    vibin = target.parent / "vibin"
    if include_vibrational_analysis and vibin.is_file():
        try:
            qcent = compute_qcent_from_xyzin(target, vibin=vibin)
        except (OSError, ValueError, ArithmeticError) as exc:
            warnings.append(f"QCENT unavailable: {exc}")
        try:
            coriolis = compute_coriolis_from_xyzin(
                target,
                vibin=vibin,
                Geff_thr_cm1=coriolis_threshold_cm1,
            )
        except (OSError, ValueError, ArithmeticError) as exc:
            warnings.append(f"Coriolis unavailable: {exc}")

    output = None
    if report:
        output = Path(report_path) if report_path is not None else target.parent / "rotational.report"
        output.parent.mkdir(parents=True, exist_ok=True)
        result_for_report = RotationalAnalysisResult(
            xyzin=target,
            rotational=section,
            equilibrium_MHz=equilibrium,
            active_MHz=active,
            kappa=float(info["kappa"]),
            q_rot=q_rot,
            qcent=qcent,
            coriolis=coriolis,
            report=output,
            warnings=tuple(warnings),
        )
        output.write_text("\n".join(rotational_analysis_report_lines(result_for_report)) + "\n", encoding="utf-8")
    return RotationalAnalysisResult(
        xyzin=target,
        rotational=section,
        equilibrium_MHz=equilibrium,
        active_MHz=active,
        kappa=float(info["kappa"]),
        q_rot=q_rot,
        qcent=qcent,
        coriolis=coriolis,
        report=output,
        warnings=tuple(warnings),
    )


def rotational_analysis_report_lines(result: RotationalAnalysisResult) -> list[str]:
    rot = result.rotational
    lines = [
        "MATRIX ROTATIONAL ANALYSIS",
        f"xyzin = {result.xyzin}",
        f"constant_source = {rot.constant_source}",
        "equilibrium_MHz = " + " ".join(f"{value:.12g}" for value in result.equilibrium_MHz),
        "active_MHz = " + " ".join(f"{value:.12g}" for value in result.active_MHz),
        f"rotor_type = {rot.rotor_type}",
        f"kappa = {result.kappa:.12g}",
        f"point_group = {rot.point_group}",
        f"symmetry_number = {rot.symmetry_number}",
        f"representation = {rot.representation}",
        f"watson_reduction = {rot.watson_reduction}",
    ]
    if rot.delta_vib_MHz is not None:
        lines.append(
            "delta_vib_MHz = "
            + " ".join("NA" if value is None else f"{value:.12g}" for value in rot.delta_vib_MHz)
        )
    if result.q_rot is not None:
        lines.append(f"Q_rot = {result.q_rot:.12g}")
    if result.qcent is not None:
        lines.extend(("", *qcent_report_lines(result.qcent)))
    if result.coriolis is not None:
        lines.extend(("", *coriolis_report_lines(result.coriolis)))
    if result.warnings:
        lines.extend(("", "WARNINGS", *result.warnings))
    return lines
