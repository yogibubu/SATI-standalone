"""Simple, presentation-neutral contract for Meyer-Caminati tunneling runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from .outputs import collect_dvr_outputs, dvr_output_summary_lines
from .sonic_metric import build_esalc_rotational_diagnostics
from .workflow import DVRRequest, dvr_section_from_request, run_dvr_request


TUNNELING_REQUEST_SCHEMA = "matrix.tunneling.request.v1"
TUNNELING_RESULT_SCHEMA = "matrix.tunneling.result.v1"


@dataclass(frozen=True)
class TunnelingRequest:
    """Small user-facing request; backend-specific flags stay internal."""

    repo_root: Path
    input_path: Path
    outdir: Path
    figdir: Path
    prefix: str = "meyer_caminati_tunneling"
    boundary: str = "periodic"
    solver: str = "fourier"
    levels: int = 12
    workers: int = 1
    compute_rotconst: bool = True
    label_cremer_pople: bool = True
    check_only: bool = False
    oracle_xyzin: Path | None = None
    esalc_coordinates: tuple[str, ...] = ()


def tunneling_request_from_json(
    path: Path | str,
    *,
    repo_root: Path,
) -> TunnelingRequest:
    """Read the compact Keymaker/MATRIX JSON request format."""

    request_path = Path(path).expanduser().resolve()
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    if payload.get("schema") != TUNNELING_REQUEST_SCHEMA:
        raise ValueError(f"unsupported tunneling request schema: {payload.get('schema')!r}")
    source = payload.get("input", {})
    model = payload.get("model", {})
    output = payload.get("output", {})
    oracle = payload.get("oracle", {})
    scan = payload.get("scan", {})
    if not isinstance(source, Mapping) or not isinstance(model, Mapping) or not isinstance(output, Mapping):
        raise ValueError("tunneling request input/model/output must be objects")
    if not isinstance(oracle, Mapping) or not isinstance(scan, Mapping):
        raise ValueError("tunneling request oracle/scan must be objects")
    input_path = _resolve(request_path.parent, source.get("path"))
    if not input_path.is_file():
        raise FileNotFoundError(f"tunneling input does not exist: {input_path}")
    outdir = _resolve(request_path.parent, output.get("directory", "results"))
    figdir = _resolve(outdir, output.get("figures", "figures"))
    boundary = str(model.get("boundary", "periodic")).strip().casefold()
    solver = str(model.get("solver", "fourier")).strip().casefold()
    if boundary not in {"periodic", "nonperiodic"}:
        raise ValueError("model.boundary must be periodic or nonperiodic")
    if solver not in {"fourier", "sinc-dvr", "gaussian"}:
        raise ValueError("model.solver must be fourier, sinc-dvr, or gaussian")
    levels = int(model.get("levels", 12))
    workers = int(model.get("workers", payload.get("performance", {}).get("workers", 1)))
    if levels < 1 or workers < 1:
        raise ValueError("model.levels and model.workers must be positive")
    oracle_xyzin = _resolve(request_path.parent, oracle.get("xyzin")) if oracle.get("xyzin") else None
    esalc_coordinates = tuple(str(item) for item in scan.get("coordinates", ()))
    if esalc_coordinates and len(esalc_coordinates) not in {1, 2}:
        raise ValueError("scan.coordinates must contain one or two ESALC coordinates")
    if oracle_xyzin is not None and not oracle_xyzin.is_file():
        raise FileNotFoundError(f"ORACLE xyzin does not exist: {oracle_xyzin}")
    return TunnelingRequest(
        repo_root=Path(repo_root).expanduser().resolve(),
        input_path=input_path,
        outdir=outdir,
        figdir=figdir,
        prefix=str(output.get("prefix", "meyer_caminati_tunneling")) or "meyer_caminati_tunneling",
        boundary=boundary,
        solver=solver,
        levels=levels,
        workers=workers,
        compute_rotconst=bool(model.get("rotational_constants", True)),
        label_cremer_pople=bool(model.get("cremer_pople_labels", True)),
        check_only=bool(model.get("check_only", False)),
        oracle_xyzin=oracle_xyzin,
        esalc_coordinates=esalc_coordinates,
    )


def run_tunneling_request(
    request: TunnelingRequest,
    *,
    timeout: float | None = None,
) -> Path:
    """Execute a simple request and write its compact result JSON."""

    dvr = DVRRequest(
        repo_root=request.repo_root,
        log_path=request.input_path,
        outdir=request.outdir,
        figdir=request.figdir,
        prefix=request.prefix,
        boundary=request.boundary,
        solver=request.solver,
        levels=request.levels,
        compute_rotconst=request.compute_rotconst,
        label_cremer_pople=request.label_cremer_pople,
        check_only=request.check_only,
        workers=request.workers,
    )
    run = run_dvr_request(dvr, timeout=timeout)
    snapshot = collect_dvr_outputs(dvr_section_from_request(dvr, manifest_path=run.manifest_path))
    esalc = None
    if request.oracle_xyzin is not None and request.esalc_coordinates:
        esalc = build_esalc_rotational_diagnostics(
            request.oracle_xyzin,
            request.esalc_coordinates,
        )
    levels = [
        {
            "state": level.state,
            "energy_cm-1": level.energy_cm,
            "energy_above_ground_cm-1": level.energy_above_ground_cm,
        }
        for level in snapshot.levels[: request.levels]
    ]
    splittings = [
        {
            "lower_state": levels[index - 1]["state"],
            "upper_state": level["state"],
            "value_cm-1": level["energy_above_ground_cm-1"]
            - levels[index - 1]["energy_above_ground_cm-1"],
        }
        for index, level in enumerate(levels)
        if index > 0
    ]
    expectations = [
        {
            "state": expectation.state,
            "energy_cm-1": expectation.energy_cm,
            "energy_above_ground_cm-1": expectation.energy_above_ground_cm,
            "values": dict(sorted(expectation.values.items())),
        }
        for expectation in snapshot.expectations[: request.levels]
    ]
    # The DVR property grid is deliberately kept open-ended: in addition to
    # A/B/C, users may provide dipoles, distortion constants, hyperfine
    # parameters, or molecule-specific fit parameters in the input profile.
    # Do not silently discard any of them from the fitting hand-off.
    rotational_fit_parameters = [
        {
            "state": row["state"],
            "energy_cm-1": row["energy_cm-1"],
            "values": row["values"],
        }
        for row in expectations
    ]
    files = [{"role": name, "path": str(path)} for name, path in sorted(snapshot.outputs.items())]
    plots = {
        name: str(path)
        for name, path in sorted(snapshot.outputs.items())
        if name.startswith("figure_")
    }
    result_path = request.outdir / f"{request.prefix}_result.json"
    result = {
        "schema": TUNNELING_RESULT_SCHEMA,
        "status": "complete" if snapshot.status == "complete" else snapshot.status,
        "input": str(request.input_path),
        "model": {
            "boundary": request.boundary,
            "solver": request.solver,
            "levels": request.levels,
            "workers": request.workers,
        },
        "esalc": esalc,
        "levels": levels,
        "splittings": splittings,
        "expectations": expectations,
        "rotational_fit_parameters": rotational_fit_parameters,
        "plots": plots,
        "files": files,
        "summary": dvr_output_summary_lines(snapshot),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result_path


def _resolve(base: Path, value: object) -> Path:
    raw = str(value or "").strip()
    if not raw:
        return base.resolve()
    candidate = Path(raw).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


__all__ = [
    "TUNNELING_REQUEST_SCHEMA",
    "TUNNELING_RESULT_SCHEMA",
    "TunnelingRequest",
    "run_tunneling_request",
    "tunneling_request_from_json",
]
