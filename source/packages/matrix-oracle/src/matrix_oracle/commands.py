from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys
from collections.abc import Sequence

from matrix_core.entrypoints import matrix_command


MORBVIS_URL = "https://yasuaki-ito.github.io/morbvis/"
WMSROT_URL = "https://www.skies-village.it/webtools/wmsrot/"


@dataclass(frozen=True)
class OracleGuiCommand:
    label: str
    argv: tuple[str, ...]
    required_sections: tuple[str, ...] = ()
    produced_sections: tuple[str, ...] = ()
    cwd: Path | None = None

    def shell_line(self) -> str:
        return " ".join(shlex.quote(item) for item in self.argv)

    def run(self, *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self.argv,
            cwd=None if self.cwd is None else str(self.cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )


def validate_command(xyzin: Path | str, *, require_fragments: bool = False) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "validate", str(Path(xyzin))]
    if require_fragments:
        argv.append("--require-fragments")
    return OracleGuiCommand("Validate molecule", tuple(argv))


def avogadro_command(
    xyzin: Path | str,
    *,
    executable: str = "avogadro2",
) -> OracleGuiCommand:
    return OracleGuiCommand("Open in Avogadro", (executable, str(Path(xyzin))))


def external_viewer_command(
    target: Path | str,
    *,
    executable: str,
    label: str | None = None,
) -> OracleGuiCommand:
    return OracleGuiCommand(label or f"Open in {executable}", (executable, str(Path(target))))


def molden_command(
    target: Path | str,
    *,
    executable: str = "molden",
) -> OracleGuiCommand:
    return external_viewer_command(target, executable=executable, label="Open in Molden")


def morbvis_command(*, url: str = MORBVIS_URL) -> OracleGuiCommand:
    return OracleGuiCommand("Open MOrbVis", (sys.executable, "-m", "webbrowser", "-t", url))


def wmsrot_command(*, url: str = WMSROT_URL) -> OracleGuiCommand:
    return OracleGuiCommand("Open WMS-Rot", (sys.executable, "-m", "webbrowser", "-t", url))


def rotational_analysis_command(
    xyzin: Path | str,
    *,
    out: Path | str | None = None,
    include_vibrational_analysis: bool = True,
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "rovib", "rotational", str(Path(xyzin))]
    if out is not None:
        argv.extend(["--out", str(Path(out))])
    _append_flag(argv, "--no-vibrational-analysis", not include_vibrational_analysis)
    return OracleGuiCommand(
        "Compute rotational state",
        tuple(argv),
        required_sections=("BASIC",),
        produced_sections=("ROTATIONAL", "CORIOLIS", "QCENT"),
    )


def wmsrot_input_command(
    xyzin: Path | str,
    *,
    out: Path | str | None = None,
    j_min: int = 0,
    j_max: int = 30,
    reduction: str | None = None,
    auto_estimate_j_range: bool = False,
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "lock", "rotor-input", str(Path(xyzin))]
    if out is not None:
        argv.extend(["--out", str(Path(out))])
    argv.extend(["--j-min", str(int(j_min)), "--j-max", str(int(j_max))])
    if reduction:
        argv.extend(["--reduction", reduction])
    _append_flag(argv, "--auto-estimate-j-range", auto_estimate_j_range)
    return OracleGuiCommand(
        "Export WMS-Rot input",
        tuple(argv),
        required_sections=("ROTATIONAL",),
    )


def wmsrot_run_command(
    xyzin: Path | str,
    *,
    out: Path | str,
    j_min: int = 0,
    j_max: int = 30,
    intensity_cut: float = 1.0e-20,
    reduction: str | None = None,
) -> OracleGuiCommand:
    argv = [
        *_matrix_cli(),
        "lock",
        "rotor-spectrum",
        str(Path(xyzin)),
        "--out",
        str(Path(out)),
        "--j-min",
        str(int(j_min)),
        "--j-max",
        str(int(j_max)),
        "--intensity-cut",
        str(float(intensity_cut)),
    ]
    if reduction:
        argv.extend(["--reduction", reduction])
    return OracleGuiCommand(
        "Run LOCK rotational spectrum",
        tuple(argv),
        required_sections=("ROTATIONAL",),
        produced_sections=("ROTATIONAL_SPECTRUM",),
    )


def vibrational_spectrum_command(
    xyzin: Path | str,
    *,
    csv_path: Path | str,
    plot_path: Path | str | None = None,
    peaks_path: Path | str | None = None,
    level2_xyzin: Path | str | None = None,
    mode_match_csv_path: Path | str | None = None,
    observable: str = "IR",
    source: str = "harmonic",
    fwhm_cm1: float = 10.0,
    min_mode_overlap: float = 0.70,
) -> OracleGuiCommand:
    argv = [
        *_matrix_cli(),
        "rovib",
        "vib-spectrum",
        str(Path(xyzin)),
        "--observable",
        observable.upper(),
        "--source",
        source.lower(),
        "--csv",
        str(Path(csv_path)),
        "--min-mode-overlap",
        str(float(min_mode_overlap)),
        "--fwhm-cm1",
        str(float(fwhm_cm1)),
    ]
    if level2_xyzin is not None:
        argv.extend(["--level2-xyzin", str(Path(level2_xyzin))])
    if mode_match_csv_path is not None:
        argv.extend(["--mode-match-csv", str(Path(mode_match_csv_path))])
    if plot_path is not None:
        argv.extend(["--plot", str(Path(plot_path))])
    if peaks_path is not None:
        argv.extend(["--peaks", str(Path(peaks_path))])
    required = ("VIBRATIONAL", "NORMAL_MODES") if source.lower() == "hybrid" else ("VIBRATIONAL",)
    return OracleGuiCommand(
        f"Build {source} {observable.upper()} spectrum",
        tuple(argv),
        required_sections=required,
        produced_sections=("VIBRATIONAL_SPECTRUM",),
    )


def vibrational_spectrum_comparison_command(
    xyzin: Path | str,
    second_xyzin: Path | str | None = None,
    *,
    csv_path: Path | str,
    plot_path: Path | str | None = None,
    mode_match_csv_path: Path | str | None = None,
    observable: str = "IR",
    first_source: str = "harmonic",
    second_source: str = "anharmonic",
    min_mode_overlap: float = 0.70,
) -> OracleGuiCommand:
    argv = [
        *_matrix_cli(),
        "rovib",
        "vib-compare",
        str(Path(xyzin)),
    ]
    if second_xyzin is not None:
        argv.append(str(Path(second_xyzin)))
    argv.extend(
        [
            "--observable",
            observable.upper(),
            "--first-source",
            first_source.lower(),
            "--second-source",
            second_source.lower(),
            "--csv",
            str(Path(csv_path)),
            "--min-mode-overlap",
            str(float(min_mode_overlap)),
        ]
    )
    if mode_match_csv_path is not None:
        argv.extend(["--mode-match-csv", str(Path(mode_match_csv_path))])
    if plot_path is not None:
        argv.extend(["--plot", str(Path(plot_path))])
    required = (
        ("VIBRATIONAL", "NORMAL_MODES")
        if "hybrid" in {first_source.lower(), second_source.lower()}
        else ("VIBRATIONAL",)
    )
    return OracleGuiCommand(
        f"Compare {observable.upper()} spectra",
        tuple(argv),
        required_sections=required,
        produced_sections=("VIBRATIONAL_SPECTRUM",),
    )


def nist_ir_command(
    identifier: str,
    *,
    out: Path | str,
    index: int = 1,
) -> OracleGuiCommand:
    return OracleGuiCommand(
        "Download NIST gas-phase IR",
        (
            *_matrix_cli(),
            "rovib",
            "nist-ir",
            str(identifier),
            "--out",
            str(Path(out)),
            "--index",
            str(int(index)),
        ),
    )


def gaussian_summary_command(log: Path | str) -> OracleGuiCommand:
    return OracleGuiCommand(
        "Summarize Gaussian log",
        (*_matrix_cli(), "gaussian", "summary", str(Path(log))),
    )


def qm_remote_submit_command(
    input_path: Path | str,
    *,
    engine: str,
    host: str = "",
    remote_root: str = "",
    extra_args: Sequence[str] = (),
) -> OracleGuiCommand:
    argv = [
        *_matrix_cli(),
        "qm",
        "remote-submit",
        str(Path(input_path)),
        "--engine",
        engine,
    ]
    if str(host).strip():
        argv.extend(("--machine", str(host).strip()))
    if str(remote_root).strip():
        argv.extend(("--remote-root", str(remote_root).strip()))
    for arg in extra_args:
        argv.extend(["--extra-arg", str(arg)])
    return OracleGuiCommand("Submit remote QM job", tuple(argv))


def qm_remote_status_command(
    *,
    host: str = "",
    remote_root: str = "",
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "qm", "remote-status"]
    if str(host).strip():
        argv.extend(("--machine", str(host).strip()))
    if str(remote_root).strip():
        argv.extend(("--remote-root", str(remote_root).strip()))
    return OracleGuiCommand("Inspect remote QM jobs", tuple(argv))


def qm_remote_fetch_command(
    job: str,
    *,
    host: str = "",
    remote_root: str = "",
    destination: Path | str = "remote_qm_runs",
    promote: str = "none",
    xyzin: Path | str | None = None,
) -> OracleGuiCommand:
    argv = [
        *_matrix_cli(),
        "qm",
        "remote-fetch",
        job,
        "--dest",
        str(Path(destination)),
        "--promote",
        promote,
    ]
    if str(host).strip():
        argv.extend(("--machine", str(host).strip()))
    if str(remote_root).strip():
        argv.extend(("--remote-root", str(remote_root).strip()))
    if xyzin is not None:
        argv.extend(["--xyzin", str(Path(xyzin))])
    produced: list[str] = []
    if promote == "molpro":
        produced.extend(["SOURCE", "BASIC", "SYMMETRY", "TOPOLOGY", "AROMATICITY", "SYNTHONS"])
    elif promote == "gaussian-log-hessian":
        produced.extend(["CARTESIAN_HESSIAN", "NORMAL_MODES"])
    elif promote == "gaussian-rovib":
        produced.extend(["VIBRATIONAL", "ROTATIONAL", "DELTABVIB"])
    elif promote == "gaussian-electronic":
        produced.extend(["ELECTRONIC", "TRANSITIONS"])
    elif promote == "gaussian-fchk":
        produced.extend(["CARTESIAN_HESSIAN", "NORMAL_MODES", "QFF", "ELECTRONIC", "ORBITALS"])
    elif promote == "orca":
        produced.extend(["SOURCE", "BASIC", "SYMMETRY", "TOPOLOGY", "AROMATICITY", "SYNTHONS", "CARTESIAN_HESSIAN"])
    return OracleGuiCommand(
        "Fetch remote QM output",
        tuple(argv),
        produced_sections=tuple(produced),
    )


def gaussian_status_command(workdir: Path | str) -> OracleGuiCommand:
    return OracleGuiCommand(
        "Inspect Gaussian job",
        (*_matrix_cli(), "gaussian", "status", str(Path(workdir))),
    )


def gaussian_run_command(
    workdir: Path | str,
    *,
    executable: str | None = None,
    input_path: Path | str | None = None,
    background: bool = False,
    timeout: float | None = None,
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "gaussian", "run", str(Path(workdir))]
    if executable:
        argv.extend(["--executable", executable])
    if input_path is not None:
        argv.extend(["--input", str(Path(input_path))])
    _append_flag(argv, "--background", background)
    if timeout is not None:
        argv.extend(["--timeout", str(timeout)])
    return OracleGuiCommand("Run Gaussian job", tuple(argv))


def gaussian_formchk_command(
    chk: Path | str,
    fchk: Path | str | None = None,
    *,
    executable: str | None = None,
    timeout: float | None = None,
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "gaussian", "formchk", str(Path(chk))]
    if fchk is not None:
        argv.append(str(Path(fchk)))
    if executable:
        argv.extend(["--executable", executable])
    if timeout is not None:
        argv.extend(["--timeout", str(timeout)])
    return OracleGuiCommand("Run formchk", tuple(argv), produced_sections=("FCHK",))


def gaussian_fchk_summary_command(fchk: Path | str) -> OracleGuiCommand:
    return OracleGuiCommand(
        "Summarize Gaussian FCHK",
        (*_matrix_cli(), "gaussian", "fchk-summary", str(Path(fchk))),
    )


def preprocess_command(
    source: Path | str,
    output: Path | str,
    *,
    source_kind: str = "auto",
    symmetry_distance: float | None = None,
    symmetry_inertia: float | None = None,
    max_rotation_order: int | None = None,
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "link", "preprocess", str(Path(source)), str(Path(output))]
    if source_kind != "auto":
        argv.extend(["--source-kind", source_kind])
    if symmetry_distance is not None:
        argv.extend(["--symmetry-distance", str(symmetry_distance)])
    if symmetry_inertia is not None:
        argv.extend(["--symmetry-inertia", str(symmetry_inertia)])
    if max_rotation_order is not None:
        argv.extend(["--max-rotation-order", str(max_rotation_order)])
    return OracleGuiCommand(
        "ORACLE import molecule",
        tuple(argv),
        produced_sections=(
            "SOURCE",
            "BASIC",
            "SYMMETRY",
            "TOPOLOGY",
            "SYNTHONS",
            "PRIMITIVES",
        ),
    )


def gaussian_promote_fchk_command(
    fchk: Path | str,
    xyzin: Path | str,
    *,
    cartesian_hessian: bool = True,
    normal_modes: bool = True,
    qff: bool = True,
    electronic: bool = True,
    orbitals: bool = True,
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "gaussian", "promote-fchk", str(Path(fchk)), str(Path(xyzin))]
    _append_flag(argv, "--no-cartesian-hessian", not cartesian_hessian)
    _append_flag(argv, "--no-normal-modes", not normal_modes)
    _append_flag(argv, "--no-qff", not qff)
    _append_flag(argv, "--no-electronic", not electronic)
    _append_flag(argv, "--no-orbitals", not orbitals)
    produced: list[str] = []
    if cartesian_hessian:
        produced.append("CARTESIAN_HESSIAN")
    if normal_modes:
        produced.append("NORMAL_MODES")
    if qff:
        produced.append("QFF")
    if electronic:
        produced.append("ELECTRONIC")
    if orbitals:
        produced.append("ORBITALS")
    return OracleGuiCommand(
        "Promote Gaussian FCHK data", tuple(argv), produced_sections=tuple(produced)
    )


def gaussian_promote_electronic_command(
    log: Path | str,
    xyzin: Path | str,
    *,
    electronic: bool = True,
    transitions: bool = True,
    orbital_files: Sequence[Path | str] = (),
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "gaussian", "promote-electronic", str(Path(log)), str(Path(xyzin))]
    _append_flag(argv, "--no-electronic", not electronic)
    _append_flag(argv, "--no-transitions", not transitions)
    for orbital_file in orbital_files:
        argv.extend(["--orbital-file", str(Path(orbital_file))])
    produced: list[str] = []
    if electronic:
        produced.append("ELECTRONIC")
    if transitions:
        produced.append("TRANSITIONS")
    if orbital_files:
        produced.append("ORBITALS")
    return OracleGuiCommand(
        "Promote Gaussian electronic data", tuple(argv), produced_sections=tuple(produced)
    )


def gaussian_promote_rovib_command(
    log: Path | str,
    xyzin: Path | str,
    *,
    vibrational: bool = True,
    rotational: bool = True,
    deltabvib: bool = True,
    invert_imaginary: bool = True,
    exclude_modes: Sequence[int] = (),
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "gaussian", "promote-rovib", str(Path(log)), str(Path(xyzin))]
    _append_flag(argv, "--no-vibrational", not vibrational)
    _append_flag(argv, "--no-rotational", not rotational)
    _append_flag(argv, "--no-deltabvib", not deltabvib)
    _append_flag(argv, "--no-invert-imaginary", not invert_imaginary)
    for mode in exclude_modes:
        argv.extend(["--exclude-mode", str(int(mode))])
    produced: list[str] = []
    if vibrational:
        produced.append("VIBRATIONAL")
    if rotational:
        produced.append("ROTATIONAL")
    if deltabvib:
        produced.append("DELTABVIB")
    return OracleGuiCommand(
        "Promote Gaussian rovibrational data",
        tuple(argv),
        produced_sections=tuple(produced),
    )


def molpro_summary_command(output: Path | str) -> OracleGuiCommand:
    return OracleGuiCommand(
        "Summarize Molpro output",
        (*_matrix_cli(), "molpro", "summary", str(Path(output))),
    )


def molpro_status_command(
    workdir: Path | str,
    *,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "molpro", "status", str(Path(workdir))]
    if input_path is not None:
        argv.extend(["--input", str(Path(input_path))])
    if output_path is not None:
        argv.extend(["--output", str(Path(output_path))])
    return OracleGuiCommand("Inspect Molpro job", tuple(argv))


def molpro_run_command(
    workdir: Path | str,
    *,
    executable: str | None = None,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    background: bool = False,
    timeout: float | None = None,
    extra_args: Sequence[str] = (),
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "molpro", "run", str(Path(workdir))]
    if executable:
        argv.extend(["--executable", executable])
    if input_path is not None:
        argv.extend(["--input", str(Path(input_path))])
    if output_path is not None:
        argv.extend(["--output", str(Path(output_path))])
    _append_flag(argv, "--background", background)
    if timeout is not None:
        argv.extend(["--timeout", str(timeout)])
    for arg in extra_args:
        argv.extend(["--extra-arg", str(arg)])
    return OracleGuiCommand("Run Molpro job", tuple(argv))


def molpro_promote_command(
    output: Path | str,
    xyzin: Path | str,
    *,
    symmetry_distance: float = 1.0e-3,
    symmetry_inertia: float = 1.0e-3,
    max_rotation_order: int = 6,
) -> OracleGuiCommand:
    argv = [
        *_matrix_cli(),
        "molpro",
        "promote",
        str(Path(output)),
        str(Path(xyzin)),
        "--symmetry-distance",
        str(symmetry_distance),
        "--symmetry-inertia",
        str(symmetry_inertia),
        "--max-rotation-order",
        str(max_rotation_order),
    ]
    return OracleGuiCommand(
        "Promote Molpro output",
        tuple(argv),
        produced_sections=(
            "SOURCE",
            "BASIC",
            "SYMMETRY",
            "TOPOLOGY",
            "SYNTHONS",
            "PRIMITIVES",
        ),
    )


def molpro_molden_command(
    output: Path | str,
    xyzin: Path | str,
    *,
    molden: Path | str | None = None,
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "molpro", "molden", str(Path(output)), str(Path(xyzin))]
    if molden is not None:
        argv.extend(["--molden", str(Path(molden))])
    return OracleGuiCommand(
        "Register Molpro Molden file",
        tuple(argv),
        produced_sections=("ORBITALS",),
    )


def orca_status_command(
    workdir: Path | str,
    *,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "orca", "status", str(Path(workdir))]
    if input_path is not None:
        argv.extend(["--input", str(Path(input_path))])
    if output_path is not None:
        argv.extend(["--output", str(Path(output_path))])
    return OracleGuiCommand("Inspect ORCA job", tuple(argv))


def orca_run_command(
    workdir: Path | str,
    *,
    executable: str | None = None,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    background: bool = False,
    timeout: float | None = None,
    extra_args: Sequence[str] = (),
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "orca", "run", str(Path(workdir))]
    if executable:
        argv.extend(["--executable", executable])
    if input_path is not None:
        argv.extend(["--input", str(Path(input_path))])
    if output_path is not None:
        argv.extend(["--output", str(Path(output_path))])
    _append_flag(argv, "--background", background)
    if timeout is not None:
        argv.extend(["--timeout", str(timeout)])
    for arg in extra_args:
        argv.extend(["--extra-arg", str(arg)])
    return OracleGuiCommand("Run ORCA job", tuple(argv))


def orca_summary_command(output: Path | str) -> OracleGuiCommand:
    return OracleGuiCommand(
        "Summarize ORCA output",
        (*_matrix_cli(), "orca", "summary", str(Path(output))),
    )


def orca_promote_command(
    output: Path | str,
    xyzin: Path | str,
    *,
    symmetry_distance: float = 1.0e-3,
    symmetry_inertia: float = 1.0e-3,
    max_rotation_order: int = 6,
) -> OracleGuiCommand:
    argv = [
        *_matrix_cli(),
        "orca",
        "promote",
        str(Path(output)),
        str(Path(xyzin)),
        "--symmetry-distance",
        str(symmetry_distance),
        "--symmetry-inertia",
        str(symmetry_inertia),
        "--max-rotation-order",
        str(max_rotation_order),
    ]
    return OracleGuiCommand(
        "Promote ORCA output",
        tuple(argv),
        produced_sections=(
            "SOURCE",
            "BASIC",
            "SYMMETRY",
            "TOPOLOGY",
            "SYNTHONS",
            "PRIMITIVES",
            "CARTESIAN_HESSIAN",
        ),
    )


def orca_molden_command(
    gbw: Path | str,
    xyzin: Path | str,
    *,
    output: Path | str | None = None,
    executable: str | None = None,
    timeout: float | None = None,
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "orca", "molden", str(Path(gbw)), str(Path(xyzin))]
    if output is not None:
        argv.extend(["--output", str(Path(output))])
    if executable:
        argv.extend(["--executable", executable])
    if timeout is not None:
        argv.extend(["--timeout", str(timeout)])
    return OracleGuiCommand(
        "Convert ORCA GBW to Molden",
        tuple(argv),
        produced_sections=("ORBITALS",),
    )


def mrcc_summary_command(output: Path | str) -> OracleGuiCommand:
    return OracleGuiCommand(
        "Summarize MRCC output",
        (*_matrix_cli(), "mrcc", "summary", str(Path(output))),
    )


def xtb_summary_command(
    output: Path | str,
    *,
    geometry: Path | str | None = None,
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "xtb", "summary", str(Path(output))]
    if geometry is not None:
        argv.extend(["--geometry", str(Path(geometry))])
    return OracleGuiCommand("Summarize xTB output", tuple(argv))


def xtb_status_command(
    workdir: Path | str,
    *,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> OracleGuiCommand:
    return _simple_qm_status_command(
        "xtb", "xTB", workdir, input_path=input_path, output_path=output_path
    )


def xtb_run_command(
    workdir: Path | str,
    *,
    executable: str | None = None,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    background: bool = False,
    timeout: float | None = None,
    extra_args: Sequence[str] = (),
) -> OracleGuiCommand:
    return _simple_qm_run_command(
        "xtb",
        "xTB",
        workdir,
        executable=executable,
        input_path=input_path,
        output_path=output_path,
        background=background,
        timeout=timeout,
        extra_args=extra_args,
    )


def pyscf_summary_command(output: Path | str) -> OracleGuiCommand:
    return OracleGuiCommand(
        "Summarize PySCF output",
        (*_matrix_cli(), "pyscf", "summary", str(Path(output))),
    )


def pyscf_status_command(
    workdir: Path | str,
    *,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> OracleGuiCommand:
    return _simple_qm_status_command(
        "pyscf", "PySCF", workdir, input_path=input_path, output_path=output_path
    )


def pyscf_run_command(
    workdir: Path | str,
    *,
    executable: str | None = None,
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    background: bool = False,
    timeout: float | None = None,
    extra_args: Sequence[str] = (),
) -> OracleGuiCommand:
    return _simple_qm_run_command(
        "pyscf",
        "PySCF",
        workdir,
        executable=executable,
        input_path=input_path,
        output_path=output_path,
        background=background,
        timeout=timeout,
        extra_args=extra_args,
    )


def _simple_qm_status_command(
    command: str,
    label: str,
    workdir: Path | str,
    *,
    input_path: Path | str | None,
    output_path: Path | str | None,
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), command, "status", str(Path(workdir))]
    if input_path is not None:
        argv.extend(["--input", str(Path(input_path))])
    if output_path is not None:
        argv.extend(["--output", str(Path(output_path))])
    return OracleGuiCommand(f"Inspect {label} job", tuple(argv))


def _simple_qm_run_command(
    command: str,
    label: str,
    workdir: Path | str,
    *,
    executable: str | None,
    input_path: Path | str | None,
    output_path: Path | str | None,
    background: bool,
    timeout: float | None,
    extra_args: Sequence[str],
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), command, "run", str(Path(workdir))]
    if executable:
        argv.extend(["--executable", executable])
    if input_path is not None:
        argv.extend(["--input", str(Path(input_path))])
    if output_path is not None:
        argv.extend(["--output", str(Path(output_path))])
    _append_flag(argv, "--background", background)
    if timeout is not None:
        argv.extend(["--timeout", str(timeout)])
    for arg in extra_args:
        argv.extend(["--extra-arg", str(arg)])
    return OracleGuiCommand(f"Run {label} job", tuple(argv))


def mrcc_promote_command(
    output: Path | str,
    xyzin: Path | str,
    *,
    symmetry_distance: float = 1.0e-3,
    symmetry_inertia: float = 1.0e-3,
    max_rotation_order: int = 6,
) -> OracleGuiCommand:
    argv = [
        *_matrix_cli(),
        "mrcc",
        "promote",
        str(Path(output)),
        str(Path(xyzin)),
        "--symmetry-distance",
        str(symmetry_distance),
        "--symmetry-inertia",
        str(symmetry_inertia),
        "--max-rotation-order",
        str(max_rotation_order),
    ]
    return OracleGuiCommand(
        "Promote MRCC output",
        tuple(argv),
        produced_sections=(
            "SOURCE",
            "BASIC",
            "SYMMETRY",
            "TOPOLOGY",
            "SYNTHONS",
            "PRIMITIVES",
        ),
    )


def fragments_command(xyzin: Path | str, action: str = "build") -> OracleGuiCommand:
    if action not in {"plan", "build", "centers"}:
        raise ValueError(f"unsupported fragments action: {action}")
    return OracleGuiCommand(
        f"Fragments {action}",
        (*_matrix_cli(), "fragments", action, str(Path(xyzin))),
        required_sections=("TOPOLOGY", "SYNTHONS"),
        produced_sections=("FRAGMENTS",),
    )


def gicforge_build_command(
    xyzin: Path | str,
    *,
    symmetrize: bool = True,
    sycart: bool = True,
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "gicforge", "build", str(Path(xyzin))]
    _append_flag(argv, "--symmetrize", symmetrize)
    _append_flag(argv, "--sycart", sycart)
    return OracleGuiCommand(
        "Build GICForge coordinates",
        tuple(argv),
        required_sections=("SYMMETRY", "TOPOLOGY", "SYNTHONS"),
        produced_sections=("GIC", "SYCART"),
    )


def gicforge_report_command(
    xyzin: Path | str, output: Path | str | None = None
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "gicforge", "report", str(Path(xyzin))]
    if output is not None:
        argv.append(str(Path(output)))
    return OracleGuiCommand("Write GICForge report", tuple(argv), required_sections=("GIC",))


def gicforge_bmatrix_command(
    xyzin: Path | str, output: Path | str | None = None
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "gicforge", "bmatrix", str(Path(xyzin))]
    if output is not None:
        argv.append(str(Path(output)))
    return OracleGuiCommand("Evaluate GIC B matrix", tuple(argv), required_sections=("GIC",))


def gicforge_gaussian_input_command(
    xyzin: Path | str,
    output: Path | str,
    *,
    route: str = "#p hf/sto-3g opt=readallgic",
    title: str | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "gicforge", "gaussian-input", str(Path(xyzin)), str(Path(output))]
    argv.extend(["--route", route])
    if title is not None:
        argv.extend(["--title", title])
    if charge is not None:
        argv.extend(["--charge", str(charge)])
    if multiplicity is not None:
        argv.extend(["--multiplicity", str(multiplicity)])
    return OracleGuiCommand("Write Gaussian GIC input", tuple(argv), required_sections=("GIC",))


def gf_command(
    xyzin: Path | str,
    *,
    fchk: Path | str | None = None,
    out: Path | str | None = None,
    csv_dir: Path | str | None = None,
    scale_file: Path | str | None = None,
    scale_records: Sequence[str] = (),
    scale_class_records: Sequence[str] = (),
    local: bool = False,
    symmetry_blocks: bool = True,
    force_threshold: float | None = None,
    large_amplitude_frequency_cutoff_cm: float | None = 250.0,
    subtract_electrostatic: bool = False,
    subtract_uff_vdw: bool = False,
    nonbonded_14_scale: float = 0.5,
    write_section: bool = True,
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "gf", "--xyzin", str(Path(xyzin))]
    if fchk is not None:
        argv.extend(["--fchk", str(Path(fchk))])
    if out is not None:
        argv.extend(["--out", str(Path(out))])
    if csv_dir is not None:
        argv.extend(["--csv-dir", str(Path(csv_dir))])
    if scale_file is not None:
        argv.extend(["--scale-file", str(Path(scale_file))])
    for record in scale_records:
        if str(record).strip():
            argv.extend(["--scale", str(record).strip()])
    for record in scale_class_records:
        if str(record).strip():
            argv.extend(["--scale-class", str(record).strip()])
    _append_flag(argv, "--local", local)
    _append_flag(argv, "--symmetry-blocks", symmetry_blocks)
    if force_threshold is not None:
        argv.extend(["--force-threshold", str(force_threshold)])
    if large_amplitude_frequency_cutoff_cm is not None:
        argv.extend(
            [
                "--large-amplitude-frequency-cutoff-cm",
                str(large_amplitude_frequency_cutoff_cm),
            ]
        )
    _append_flag(argv, "--subtract-electrostatic", subtract_electrostatic)
    _append_flag(argv, "--subtract-uff-vdw", subtract_uff_vdw)
    if nonbonded_14_scale != 0.5:
        argv.extend(["--nonbonded-14-scale", str(nonbonded_14_scale)])
    _append_flag(argv, "--no-write-section", not write_section)
    return OracleGuiCommand(
        "Run TRINITY harmonic",
        tuple(argv),
        required_sections=("GIC",),
        produced_sections=("GF_PED",),
    )


def gf_scaling_preview_command(
    xyzin: Path | str,
    *,
    scale_file: Path | str | None = None,
    scale_records: Sequence[str] = (),
    scale_class_records: Sequence[str] = (),
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "gf", "--xyzin", str(Path(xyzin)), "--scale-preview"]
    if scale_file is not None:
        argv.extend(["--scale-file", str(Path(scale_file))])
    for record in scale_records:
        if str(record).strip():
            argv.extend(["--scale", str(record).strip()])
    for record in scale_class_records:
        if str(record).strip():
            argv.extend(["--scale-class", str(record).strip()])
    return OracleGuiCommand(
        "Preview GF scaling",
        tuple(argv),
        required_sections=("GIC",),
    )


def thermo_command(
    xyzin: Path | str,
    *,
    out: Path | str | None = None,
    report: bool = True,
    write_section: bool = True,
    cutoff_cm1: float = 10.0,
    keep_low_positive: bool = False,
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "thermo", str(Path(xyzin))]
    if out is not None and report:
        argv.extend(["--out", str(Path(out))])
    _append_flag(argv, "--no-report", not report)
    _append_flag(argv, "--no-write-section", not write_section)
    if cutoff_cm1 != 10.0:
        argv.extend(["--cutoff-cm1", str(cutoff_cm1)])
    _append_flag(argv, "--keep-low-positive", keep_low_positive)
    return OracleGuiCommand(
        "Run Thermo",
        tuple(argv),
        required_sections=("BASIC", "ROTATIONAL"),
        produced_sections=("THERMO",),
    )


def rovib_summary_command(xyzin: Path | str) -> OracleGuiCommand:
    return OracleGuiCommand(
        "Summarize rovibrational sections",
        (*_matrix_cli(), "rovib", "summarize", str(Path(xyzin))),
        required_sections=("ROTATIONAL",),
    )


def rovib_vibrational_dos_command(
    xyzin: Path | str,
    *,
    out: Path | str | None = None,
    vmax: int = 6,
    emax_cm1: float = 8000.0,
    bin_cm1: float = 50.0,
    ncap: float = 10.0,
) -> OracleGuiCommand:
    argv = [
        *_matrix_cli(),
        "rovib",
        "dos",
        str(Path(xyzin)),
        "--vmax",
        str(vmax),
        "--emax",
        str(emax_cm1),
        "--bin-cm1",
        str(bin_cm1),
        "--ncap",
        str(ncap),
    ]
    if out is not None:
        argv.extend(["--out", str(Path(out))])
    return OracleGuiCommand(
        "Build vibrational density of states",
        tuple(argv),
        required_sections=("VIBRATIONAL",),
    )


def rovib_density_command(
    xyzin: Path | str,
    *,
    vib_dos: Path | str | None = None,
    out: Path | str | None = None,
    rot_out: Path | str | None = None,
    q_out: Path | str | None = None,
    emax_rot: float | None = None,
    jmax: int | None = None,
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "rovib", "dos-rovib", str(Path(xyzin))]
    if vib_dos is not None:
        argv.extend(["--vib-dos", str(Path(vib_dos))])
    if out is not None:
        argv.extend(["--out", str(Path(out))])
    if rot_out is not None:
        argv.extend(["--rot-out", str(Path(rot_out))])
    if q_out is not None:
        argv.extend(["--q-out", str(Path(q_out))])
    if emax_rot is not None:
        argv.extend(["--emax-rot", str(emax_rot)])
    if jmax is not None:
        argv.extend(["--jmax", str(jmax)])
    return OracleGuiCommand(
        "Build rovibrational density of states",
        tuple(argv),
        required_sections=("ROTATIONAL", "VIBRATIONAL"),
    )


def single_reaction_kinetics_command(
    reactant_xyzin: Path | str,
    transition_state_xyzin: Path | str,
    *,
    reactant_dos: Path | str,
    transition_state_dos: Path | str,
    barrier_cm1: float,
    rrkm_out: Path | str | None = None,
    manifest: Path | str | None = None,
    report: Path | str | None = None,
    path_degeneracy: float = 1.0,
    tunneling: str = "none",
    imaginary_frequency_cm1: float | None = None,
) -> OracleGuiCommand:
    argv = [
        *_matrix_cli(), "kinetics", "single", str(Path(reactant_xyzin)),
        str(Path(transition_state_xyzin)), "--reactant-dos", str(Path(reactant_dos)),
        "--ts-dos", str(Path(transition_state_dos)), "--barrier-cm1", str(barrier_cm1),
        "--path-degeneracy", str(path_degeneracy), "--tunneling", tunneling,
    ]
    for flag, value in (
        ("--rrkm-out", rrkm_out),
        ("--manifest", manifest),
        ("--report", report),
        ("--imaginary-frequency-cm1", imaginary_frequency_cm1),
    ):
        if value is not None:
            argv.extend((flag, str(value)))
    return OracleGuiCommand(
        "Run single-reaction TST/RRKM",
        tuple(argv),
        required_sections=("VIBRATIONAL",),
        produced_sections=("KINETICS",),
    )


def vpt2_vci_command(
    xyzin: Path | str,
    *,
    run_dir: Path | str | None = None,
    max_quanta: int = 2,
    roots: int = 10,
) -> OracleGuiCommand:
    argv = [
        *_matrix_cli(),
        "vpt2-vci",
        "--xyzin",
        str(Path(xyzin)),
        "--max-quanta",
        str(max_quanta),
        "--roots",
        str(roots),
    ]
    if run_dir is not None:
        argv.extend(["--run-dir", str(Path(run_dir))])
    return OracleGuiCommand(
        "Run VPT2/VCI",
        tuple(argv),
        required_sections=("QFF",),
        produced_sections=("VPT2_VCI",),
    )


def vpt2_vci_collect_command(xyzin: Path | str, *, no_write: bool = False) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "vpt2-vci", "--collect", str(Path(xyzin))]
    _append_flag(argv, "--no-write", no_write)
    return OracleGuiCommand(
        "Collect VPT2/VCI outputs",
        tuple(argv),
        produced_sections=("VPT2_VCI",),
    )


def dvr_run_command(
    xyzin: Path | str,
    *,
    timeout: float | None = None,
    check_only: bool = False,
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "dvr", "run", "--xyzin", str(Path(xyzin))]
    _append_flag(argv, "--check-only", check_only)
    if timeout is not None:
        argv.extend(["--timeout", str(timeout)])
    return OracleGuiCommand(
        "Run DVR", tuple(argv), required_sections=("DVR",), produced_sections=("DVR",)
    )


def dvr_collect_command(xyzin: Path | str, *, no_write: bool = False) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "dvr", "collect", str(Path(xyzin))]
    _append_flag(argv, "--no-write", no_write)
    return OracleGuiCommand("Collect DVR outputs", tuple(argv), produced_sections=("DVR",))


def semiexp_command(
    job: Path | str,
    outdir: Path | str,
    *,
    xyzin: Path | str | None = None,
    backend: str = "python",
    write_section: bool = True,
    extra_args: Sequence[str] = (),
) -> OracleGuiCommand:
    argv = [
        *_matrix_cli(),
        "semiexp",
        "--job",
        str(Path(job)),
        "--outdir",
        str(Path(outdir)),
        "--backend",
        backend,
    ]
    if xyzin is not None:
        argv.extend(["--xyzin", str(Path(xyzin))])
    _append_flag(argv, "--no-write-section", not write_section)
    argv.extend(str(item) for item in extra_args)
    return OracleGuiCommand(
        "Run SEFit / MORPHEUS",
        tuple(argv),
        required_sections=("ISOTOPOLOGUES",),
        produced_sections=("MORPHEUS",),
    )


def link_prepare_command(
    xyzin: Path | str,
    *,
    run_dir: Path | str,
    engine_command: str,
    coordinate_model: str = "gic",
    active_space: str = "total_symmetric",
    max_steps: int = 50,
    trust_radius: float = 0.2,
    gradient_tolerance: float = 1.0e-5,
    step_tolerance: float = 1.0e-5,
    energy_tolerance: float = 1.0e-8,
) -> OracleGuiCommand:
    argv = [
        *_matrix_cli(),
        "link",
        "prepare",
        str(Path(xyzin)),
        "--run-dir",
        str(Path(run_dir)),
        "--engine-command",
        engine_command,
        "--coordinate-model",
        coordinate_model,
        "--active-space",
        active_space,
        "--max-steps",
        str(int(max_steps)),
        "--trust-radius",
        str(trust_radius),
        "--gradient-tolerance",
        str(gradient_tolerance),
        "--step-tolerance",
        str(step_tolerance),
        "--energy-tolerance",
        str(energy_tolerance),
    ]
    return OracleGuiCommand(
        "Prepare LINK",
        tuple(argv),
        required_sections=("BASIC",),
        produced_sections=("LINK",),
    )


def link_status_command(xyzin: Path | str) -> OracleGuiCommand:
    return OracleGuiCommand(
        "Summarize LINK",
        (*_matrix_cli(), "link", "status", str(Path(xyzin))),
        required_sections=("LINK",),
    )


def semiexp_benchmark_command(
    *,
    snapshot: Path | str | None = None,
    outdir: Path | str | None = None,
    refresh: bool = True,
    update_snapshot: bool = False,
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "semiexp-benchmark"]
    if snapshot is not None:
        argv.extend(["--snapshot", str(Path(snapshot))])
    if outdir is not None:
        argv.extend(["--outdir", str(Path(outdir))])
    _append_flag(argv, "--no-refresh", not refresh)
    _append_flag(argv, "--update-snapshot", update_snapshot)
    return OracleGuiCommand("Run SEFit/MORPHEUS paper benchmark", tuple(argv))


def gicforge_fortran_audit_command(
    *,
    root: Path | str | None = None,
    workdir: Path | str | None = None,
    molecules: Sequence[str] = (),
    limit: int | None = None,
    tolerance: float = 2.0e-8,
    output_format: str = "summary",
) -> OracleGuiCommand:
    argv = [*_matrix_cli(), "gicforge", "fortran-audit"]
    if root is not None:
        argv.extend(["--root", str(Path(root))])
    if workdir is not None:
        argv.extend(["--workdir", str(Path(workdir))])
    for molecule in molecules:
        argv.extend(["--molecule", molecule])
    if limit is not None:
        argv.extend(["--limit", str(limit)])
    argv.extend(["--tolerance", str(tolerance), "--format", output_format])
    return OracleGuiCommand("Run GICForge Python/Fortran audit", tuple(argv))


def _matrix_cli() -> tuple[str, ...]:
    return matrix_command()


def _append_flag(argv: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        argv.append(flag)
