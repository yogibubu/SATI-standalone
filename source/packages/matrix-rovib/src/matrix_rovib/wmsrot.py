from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

from matrix_core import has_section, read_basic_section

from .contracts import (
    DeltaBVibSection,
    RotationalSection,
    read_deltabvib_section,
    read_rotational_section,
    resolved_rotational_constants_MHz,
)
from .rotspec import (
    RotationalSpectrumResult,
    transitions_from_wmsrot,
    write_rotational_spectrum_latex,
    write_rotational_spectrum_plot,
    write_rotational_spectrum_section,
)
from .native_rotor import NativeRotorParameters, simulate_native_rotor


WMSROT_URL = "https://www.skies-village.it/webtools/wmsrot/"


@dataclass(frozen=True)
class LOCKRotorInputOptions:
    j_min: int = 0
    j_max: int = 30
    freq_unit: str = "MHz"
    auto_estimate_j_range: bool = False
    reduction: str | None = None


# LOCK is the production spelling; the historical name remains for explicit
# compatibility imports from this module.
WMSRotInputOptions = LOCKRotorInputOptions


@dataclass(frozen=True)
class LOCKRotorSimulationOptions:
    j_min: int = 0
    j_max: int = 30
    intensity_cut: float = 1.0e-20
    a_type: bool = True
    b_type: bool = True
    c_type: bool = True
    reduction: str | None = None
    freq_min: float | None = None
    freq_max: float | None = None
    freq_unit: str = "MHz"


# The WMS spelling remains a compatibility type alias.
WMSRotSimulationOptions = LOCKRotorSimulationOptions


class WMSRotEngineUnavailable(RuntimeError):
    pass


def wmsrot_input_text_from_xyzin(
    xyzin: Path | str,
    *,
    options: WMSRotInputOptions | None = None,
) -> str:
    target = Path(xyzin)
    rot = read_rotational_section(target)
    deltabvib = read_deltabvib_section(target) if has_section(target, "DELTABVIB") else None
    basic = read_basic_section(target) if has_section(target, "BASIC") else None
    return (
        "\n".join(wmsrot_input_lines(rot, deltabvib=deltabvib, options=options, basic=basic)) + "\n"
    )


def write_wmsrot_input(
    xyzin: Path | str,
    out: Path | str,
    *,
    options: WMSRotInputOptions | None = None,
) -> Path:
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(wmsrot_input_text_from_xyzin(xyzin, options=options), encoding="utf-8")
    return output


def lock_rotor_input_text_from_xyzin(
    xyzin: Path | str,
    *,
    options: LOCKRotorInputOptions | None = None,
) -> str:
    """Build the supported LOCK rotor compatibility text."""
    return wmsrot_input_text_from_xyzin(xyzin, options=options)


def write_lock_rotor_input(
    xyzin: Path | str,
    out: Path | str,
    *,
    options: LOCKRotorInputOptions | None = None,
) -> Path:
    """Write the supported LOCK rotor compatibility text."""
    return write_wmsrot_input(xyzin, out, options=options)


def load_wmsrot_reference_engine() -> ModuleType:
    """Load the vendored WMS-Rot engine for regression/reference work only."""
    try:
        return import_module("matrix_rovib.vendor.wmsrot_engine")
    except ModuleNotFoundError as exc:
        missing = exc.name or "dependency"
        if missing in {"numpy", "pandas", "sympy", "matplotlib"}:
            raise WMSRotEngineUnavailable(
                "WMS-Rot local engine requires numpy, pandas, sympy and matplotlib. "
                f"Missing Python package: {missing}."
            ) from exc
        raise


def load_wmsrot_engine() -> ModuleType:
    """Compatibility alias for :func:`load_wmsrot_reference_engine`."""
    return load_wmsrot_reference_engine()


def simulate_lock_rotor_spectrum_from_xyzin(
    xyzin: Path | str,
    *,
    options: WMSRotSimulationOptions | None = None,
) -> Any:
    """Run the production LOCK rigid-rotor backend from an xyzin container."""
    return simulate_wmsrot_spectrum_from_xyzin(xyzin, options=options)


def simulate_lock_rotor_spectrum(
    rotational: RotationalSection,
    *,
    deltabvib: DeltaBVibSection | None = None,
    options: WMSRotSimulationOptions | None = None,
    basic=None,
) -> Any:
    """Run the production LOCK rigid-rotor backend."""
    return simulate_wmsrot_spectrum(
        rotational,
        deltabvib=deltabvib,
        options=options,
        basic=basic,
    )


def simulate_wmsrot_spectrum_from_xyzin(
    xyzin: Path | str,
    *,
    options: WMSRotSimulationOptions | None = None,
    engine: ModuleType | Any | None = None,
) -> Any:
    target = Path(xyzin)
    rot = read_rotational_section(target)
    deltabvib = read_deltabvib_section(target) if has_section(target, "DELTABVIB") else None
    basic = read_basic_section(target) if has_section(target, "BASIC") else None
    return simulate_wmsrot_spectrum(
        rot,
        deltabvib=deltabvib,
        basic=basic,
        options=options,
        engine=engine,
    )


def simulate_wmsrot_spectrum(
    rotational: RotationalSection,
    *,
    deltabvib: DeltaBVibSection | None = None,
    options: WMSRotSimulationOptions | None = None,
    basic=None,
    engine: ModuleType | Any | None = None,
) -> Any:
    opts = options or WMSRotSimulationOptions()
    rotor_type = _wms_rotor_type(rotational.rotor_type)
    reduction = _wms_reduction(
        opts.reduction
        or rotational.watson_reduction
        or _basic_attr(basic, "watson_reduction")
        or "S"
    )
    representation = rotational.representation or "Ir"
    point_group = rotational.point_group or _basic_attr(basic, "point_group") or "C1"
    temperature = rotational.temperature_K or _basic_attr(basic, "temperature_K") or 298.15
    A, B, C = _simulation_constants(rotational, deltabvib=deltabvib, rotor_type=rotor_type)
    mu_a, mu_b, mu_c = _dipole(rotational)
    mu_a = 0.0 if mu_a is None else float(mu_a)
    mu_b = 0.0 if mu_b is None else float(mu_b)
    mu_c = 0.0 if mu_c is None else float(mu_c)
    if rotor_type == "linear":
        mu_b = 0.0
        mu_c = 0.0
    elif rotor_type == "spherical":
        mu_a = mu_b = mu_c = 0.0
    elif rotor_type == "prolate":
        mu_b = 0.0
        mu_c = 0.0
    elif rotor_type == "oblate":
        mu_a = 0.0
        mu_b = 0.0
    if engine is None:
        frequency_scale = _frequency_unit_to_MHz(opts.freq_unit)
        quartic = tuple(float(value) for value in rotational.quartic_distortion_MHz)
        sextic = tuple(float(value) for value in rotational.sextic_distortion_MHz)
        octic = tuple(float(value) for value in rotational.octic_distortion_MHz)
        quartic = quartic or (0.0,) * 5
        sextic = sextic or (0.0,) * 7
        octic = octic or (0.0,) * 9
        return simulate_native_rotor(
            NativeRotorParameters(
                A_MHz=A,
                B_MHz=B,
                C_MHz=C,
                quartic_MHz=quartic,
                sextic_MHz=sextic,
                octic_MHz=octic,
                reduction=reduction,
                representation=representation,
            ),
            temperature_K=float(temperature),
            dipole_debye=(mu_a, mu_b, mu_c),
            j_min=int(opts.j_min),
            j_max=int(opts.j_max),
            frequency_min_MHz=(
                0.0 if opts.freq_min is None else float(opts.freq_min) * frequency_scale
            ),
            frequency_max_MHz=(
                float("inf")
                if opts.freq_max is None
                else float(opts.freq_max) * frequency_scale
            ),
            intensity_cut=float(opts.intensity_cut),
            active_axes=(opts.a_type, opts.b_type, opts.c_type),
        )
    module = engine
    quartic = tuple(float(value) for value in rotational.quartic_distortion_MHz) or (0.0,) * 5
    sextic = tuple(float(value) for value in rotational.sextic_distortion_MHz) or (0.0,) * 7
    octic = tuple(float(value) for value in rotational.octic_distortion_MHz) or (0.0,) * 9
    snapshot = _snapshot_engine_globals(
        module,
        (
            "REDUCTION",
            "REPRESENTATION",
            "FREQ_MIN",
            "FREQ_MAX",
            "FREQ_UNIT",
            "groupSymmetry",
            "QROT_OVERRIDE",
        ),
    )
    try:
        module.REDUCTION = reduction
        module.REPRESENTATION = representation
        module.groupSymmetry = point_group
        module.FREQ_UNIT = opts.freq_unit
        if opts.freq_min is not None:
            module.FREQ_MIN = float(opts.freq_min)
        if opts.freq_max is not None:
            module.FREQ_MAX = float(opts.freq_max)
        if rotational.q_rot is not None:
            module.QROT_OVERRIDE = float(rotational.q_rot)
        return module.simulate_rigid_spectrum(
            float(temperature),
            A,
            B,
            C,
            *quartic,
            *sextic,
            mu_a,
            mu_b,
            mu_c,
            int(opts.j_max),
            float(opts.intensity_cut),
            point_group,
            rotor_type,
            bool(opts.a_type),
            bool(opts.b_type),
            bool(opts.c_type),
            True,
            int(opts.j_min),
        )
    finally:
        _restore_engine_globals(module, snapshot)


def write_wmsrot_spectrum_csv(
    xyzin: Path | str,
    out: Path | str,
    *,
    options: WMSRotSimulationOptions | None = None,
    engine: ModuleType | Any | None = None,
) -> Path:
    result = simulate_wmsrot_spectrum_from_xyzin(xyzin, options=options, engine=engine)
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(result, "to_csv"):
        result.to_csv(output, index=False)
    else:
        _write_rows_csv(output, result)
    return output


def write_wmsrot_spectrum_outputs(
    xyzin: Path | str,
    csv_path: Path | str,
    *,
    plot_path: Path | str | None = None,
    table_path: Path | str | None = None,
    fwhm_MHz: float = 0.0,
    options: WMSRotSimulationOptions | None = None,
    engine: ModuleType | Any | None = None,
    write_section: bool = True,
) -> RotationalSpectrumResult:
    target = Path(xyzin)
    opts = options or WMSRotSimulationOptions()
    raw = simulate_wmsrot_spectrum_from_xyzin(target, options=opts, engine=engine)
    csv_output = Path(csv_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(raw, "to_csv"):
        raw.to_csv(csv_output, index=False)
    else:
        _write_rows_csv(csv_output, raw)
    transitions = transitions_from_wmsrot(raw)
    plot_output = None
    if plot_path is not None:
        plot_output = write_rotational_spectrum_plot(
            plot_path,
            transitions,
            fwhm_MHz=fwhm_MHz,
        )
    table_output = None
    if table_path is not None:
        table_output = write_rotational_spectrum_latex(table_path, transitions)
    rotational = read_rotational_section(target)
    result = RotationalSpectrumResult(
        transitions=transitions,
        csv_path=csv_output,
        plot_path=plot_output,
        table_path=table_output,
        j_min=opts.j_min,
        j_max=opts.j_max,
        temperature_K=rotational.temperature_K,
        constant_kind=rotational.constant_kind,
        constant_source=rotational.constant_source,
        backend=(
            "matrix-rovib.lock.native_rotor"
            if engine is None
            else "matrix-rovib.vendor.wmsrot_engine"
        ),
    )
    if write_section:
        write_rotational_spectrum_section(target, result)
    return result


def write_lock_rotor_spectrum_outputs(
    xyzin: Path | str,
    csv_path: Path | str,
    *,
    plot_path: Path | str | None = None,
    table_path: Path | str | None = None,
    fwhm_MHz: float = 0.0,
    options: WMSRotSimulationOptions | None = None,
    write_section: bool = True,
) -> RotationalSpectrumResult:
    """Write production LOCK rotor outputs without a reference-engine hook."""
    return write_wmsrot_spectrum_outputs(
        xyzin,
        csv_path,
        plot_path=plot_path,
        table_path=table_path,
        fwhm_MHz=fwhm_MHz,
        options=options,
        write_section=write_section,
    )


def wmsrot_input_lines(
    rotational: RotationalSection,
    *,
    deltabvib: DeltaBVibSection | None = None,
    options: WMSRotInputOptions | None = None,
    basic=None,
) -> list[str]:
    opts = options or WMSRotInputOptions()
    rotor_type = _wms_rotor_type(rotational.rotor_type)
    reduction = _wms_reduction(
        opts.reduction
        or rotational.watson_reduction
        or _basic_attr(basic, "watson_reduction")
        or "S"
    )
    point_group = rotational.point_group or _basic_attr(basic, "point_group") or "C1"
    temperature = (
        rotational.temperature_K
        if rotational.temperature_K is not None
        else _basic_attr(basic, "temperature_K")
    )
    dvib_a, dvib_b, dvib_c = _delta_vib(rotational, deltabvib)
    dip_a, dip_b, dip_c = _dipole(rotational)
    lines = [
        "#ROTATIONAL",
        f"rotor_type = {rotor_type}",
        f"representation = {rotational.representation or 'Ir'}",
        f"Watson Reduction = {reduction}",
        f"Point Group = {point_group}",
        f"T_K = {_format_number(temperature)}",
        f"A_MHz = {_format_number(rotational.A_MHz)}",
        f"B_MHz = {_format_number(rotational.B_MHz)}",
        f"C_MHz = {_format_number(rotational.C_MHz)}",
        f"DVibA_MHz= {_format_number(dvib_a, default='0')}",
        f"DVibB_MHz= {_format_number(dvib_b, default='0')}",
        f"DVibC_MHz= {_format_number(dvib_c, default='0')}",
        f"Dipole_a_D = {_format_number(dip_a)}",
        f"Dipole_b_D = {_format_number(dip_b)}",
        f"Dipole_c_D = {_format_number(dip_c)}",
        "I_NUC = 0",
        "I_NUC_1 = 0",
        "I_NUC_2 = 0",
        "chi_aa = 0",
        "chi_bb = 0",
        "chi_cc = 0",
        "chi_ab = 0",
        "chi_ac = 0",
        "chi_bc = 0",
        "chi_aa_1 = 0",
        "chi_bb_1 = 0",
        "chi_cc_1 = 0",
        "chi_ab_1 = 0",
        "chi_ac_1 = 0",
        "chi_bc_1 = 0",
        "chi_aa_2 = 0",
        "chi_bb_2 = 0",
        "chi_cc_2 = 0",
        "chi_ab_2 = 0",
        "chi_ac_2 = 0",
        "chi_bc_2 = 0",
        f"J_MIN = {int(opts.j_min)}",
        f"J_MAX = {int(opts.j_max)}",
        f"FREQ_UNIT = {opts.freq_unit}",
        f"AUTO_ESTIMATE_J_RANGE = {'true' if opts.auto_estimate_j_range else 'false'}",
    ]
    lines.extend(_distortion_lines(reduction, rotational))
    if rotational.q_rot is not None:
        lines.append(f"Q_rot={_format_number(rotational.q_rot)}")
    return lines


def _delta_vib(
    rotational: RotationalSection,
    deltabvib: DeltaBVibSection | None,
) -> tuple[float | None, float | None, float | None]:
    if rotational.delta_vib_MHz is not None:
        return rotational.delta_vib_MHz
    if deltabvib is not None and deltabvib.available:
        return deltabvib.delta_A_MHz, deltabvib.delta_B_MHz, deltabvib.delta_C_MHz
    return None, None, None


def _simulation_constants(
    rotational: RotationalSection,
    *,
    deltabvib: DeltaBVibSection | None,
    rotor_type: str,
) -> tuple[float, float, float]:
    A, B, C = resolved_rotational_constants_MHz(
        rotational,
        state="ground_state",
        deltabvib=deltabvib,
    )
    if rotor_type == "linear":
        base = _required(B, "B_MHz")
        return base, base, base
    if rotor_type == "spherical":
        base = _required(B if B is not None else A, "B_MHz")
        return base, base, base
    if rotor_type == "prolate":
        a = _required(A, "A_MHz")
        b = _required(B, "B_MHz")
        return a, b, b
    if rotor_type == "oblate":
        b = _required(B, "B_MHz")
        c = _required(C, "C_MHz")
        return b, b, c
    return _required(A, "A_MHz"), _required(B, "B_MHz"), _required(C, "C_MHz")




def _required(value: float | None, label: str) -> float:
    if value is None:
        raise ValueError(f"WMS-Rot simulation requires {label} in #ROTATIONAL")
    return float(value)


def _dipole(rotational: RotationalSection) -> tuple[float | None, float | None, float | None]:
    if rotational.dipole_debye is None:
        return None, None, None
    return rotational.dipole_debye


def _wms_rotor_type(value: str) -> str:
    normalized = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if "linear" in normalized:
        return "linear"
    if "spherical" in normalized:
        return "spherical"
    if "oblate" in normalized:
        return "oblate"
    if "prolate" in normalized and "asymmetric" not in normalized:
        return "prolate"
    return "asymmetric"


def _wms_reduction(value: str) -> str:
    return "A" if (value or "").strip().upper().startswith("A") else "S"


def _frequency_unit_to_MHz(value: str) -> float:
    unit = str(value or "MHz").strip().lower().replace("−", "-")
    if unit == "mhz":
        return 1.0
    if unit == "ghz":
        return 1000.0
    if unit in {"cm-1", "cm^-1", "cm⁻¹"}:
        return 29979.2458
    raise ValueError(f"unsupported frequency unit {value!r}")


def _distortion_lines(reduction: str, rotational: RotationalSection) -> list[str]:
    quartic = tuple(rotational.quartic_distortion_MHz) or (0.0,) * 5
    sextic = tuple(rotational.sextic_distortion_MHz) or (0.0,) * 7
    octic = tuple(rotational.octic_distortion_MHz) or (0.0,) * 9
    if len(quartic) != 5 or len(sextic) != 7 or len(octic) != 9:
        raise ValueError(
            "Watson distortion payload requires five quartic, seven sextic, and nine octic constants"
        )
    if _wms_reduction(reduction) == "A":
        return [
            f"DELTA J_MHz =  {_format_number(quartic[0])}",
            f"DELTA JK_MHz = {_format_number(quartic[1])}",
            f"DELTA K_MHz = {_format_number(quartic[2])}",
            f"delta J_MHz = {_format_number(quartic[3])}",
            f"delta K_MHz = {_format_number(quartic[4])}",
            f"PHI N_MHz = {_format_number(sextic[0])}",
            f"PHI NK_MHz = {_format_number(sextic[1])}",
            f"PHI KN_MHz = {_format_number(sextic[2])}",
            f"PHI K_MHz = {_format_number(sextic[3])}",
            f"phi N_MHz = {_format_number(sextic[4])}",
            f"phi NK_MHz = {_format_number(sextic[5])}",
            f"phi K_MHz = {_format_number(sextic[6])}",
            f"L J_MHz = {_format_number(octic[0])}",
            f"L JJK_MHz = {_format_number(octic[1])}",
            f"L JK_MHz = {_format_number(octic[2])}",
            f"L KKJ_MHz = {_format_number(octic[3])}",
            f"L K_MHz = {_format_number(octic[4])}",
            f"l J_MHz = {_format_number(octic[5])}",
            f"l JK_MHz = {_format_number(octic[6])}",
            f"l KJ_MHz = {_format_number(octic[7])}",
            f"l K_MHz = {_format_number(octic[8])}",
        ]
    return [
        f"DJ_MHz = {_format_number(quartic[0])}",
        f"DJK_MHz = {_format_number(quartic[1])}",
        f"DK_MHz = {_format_number(quartic[2])}",
        f"d1_MHz = {_format_number(quartic[3])}",
        f"d2_MHz = {_format_number(quartic[4])}",
        f"H N_MHz = {_format_number(sextic[0])}",
        f"H NK_MHz = {_format_number(sextic[1])}",
        f"H KN_MHz = {_format_number(sextic[2])}",
        f"H K_MHz = {_format_number(sextic[3])}",
        f"h1_MHz = {_format_number(sextic[4])}",
        f"h2_MHz = {_format_number(sextic[5])}",
        f"h3_MHz = {_format_number(sextic[6])}",
        f"L J_MHz = {_format_number(octic[0])}",
        f"L JJK_MHz = {_format_number(octic[1])}",
        f"L JK_MHz = {_format_number(octic[2])}",
        f"L KKJ_MHz = {_format_number(octic[3])}",
        f"L K_MHz = {_format_number(octic[4])}",
        f"l1_MHz = {_format_number(octic[5])}",
        f"l2_MHz = {_format_number(octic[6])}",
        f"l3_MHz = {_format_number(octic[7])}",
        f"l4_MHz = {_format_number(octic[8])}",
    ]


def _format_number(value: float | int | None, *, default: str = "") -> str:
    if value is None:
        return default
    return f"{float(value):.12g}"


def _basic_attr(basic, name: str):
    return None if basic is None else getattr(basic, name, None)


def _snapshot_engine_globals(module: Any, names: Sequence[str]) -> dict[str, Any]:
    marker = object()
    snapshot: dict[str, Any] = {"__missing_marker__": marker}
    for name in names:
        snapshot[name] = getattr(module, name, marker)
    return snapshot


def _restore_engine_globals(module: Any, snapshot: dict[str, Any]) -> None:
    marker = snapshot["__missing_marker__"]
    for name, value in snapshot.items():
        if name == "__missing_marker__":
            continue
        if value is marker:
            try:
                delattr(module, name)
            except AttributeError:
                pass
        else:
            setattr(module, name, value)


def _write_rows_csv(path: Path, rows: Any) -> None:
    import csv

    materialized = list(rows)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    first = materialized[0]
    if isinstance(first, dict):
        fields = list(first)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(materialized)
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(materialized)
