"""Command-line interface for the standalone ORACLE perception package."""

from __future__ import annotations

import argparse
from importlib import resources
from importlib.util import find_spec
import json
from pathlib import Path
import platform
import shutil
import sys

from ._version import __version__


def build_parser(*, prog: str = "oracle") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="ORACLE topology, symmetry and continuous molecular perception",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze a structure and write enriched xyzin")
    analyze.add_argument("source", type=Path)
    analyze.add_argument("-o", "--output", type=Path, required=True)
    analyze.add_argument("--report", type=Path, help="Write the versioned JSON analysis report")
    analyze.add_argument(
        "--human-report",
        type=Path,
        help="Write the complete human-readable perception/PIC report",
    )
    analyze.add_argument("--snapshot", type=Path, help="Write a topology snapshot JSON")
    analyze.add_argument("--config", type=Path, help="TOML configuration file")
    analyze.add_argument(
        "--source-kind",
        choices=(
            "auto",
            "smiles",
            "xyz",
            "enriched_xyz",
            "gaussian",
            "fchk",
            "mol",
            "sdf",
            "mol2",
            "molpro",
            "mrcc",
            "orca",
        ),
        default="auto",
    )
    analyze.add_argument("--symmetry-distance", type=float)
    analyze.add_argument("--symmetry-inertia", type=float)
    analyze.add_argument("--max-rotation-order", type=int)
    analyze.add_argument(
        "--cartesian-symmetrization",
        choices=("inspect", "apply", "retain"),
        default="inspect",
    )
    analyze.add_argument(
        "--freeze-onic-contract",
        action="store_true",
        help=(
            "Build and serialize the ORACLE-owned frozen ONIC primitive contract "
            "for downstream SMITH chart realization"
        ),
    )
    analyze.add_argument(
        "--bond-order-model",
        choices=("auto", "pauling", "matrix-gfn2", "integer-connectivity"),
        default="auto",
        help=(
            "Choose continuous MATRIX-GFN2 orders for electronic calculations or integer "
            "1/2/3 orders from connectivity for FF seeds far from equilibrium"
        ),
    )
    analyze.add_argument(
        "--stationary-point",
        choices=("minimum", "transition-state"),
        default="minimum",
        help=(
            "Explicit scientific task. transition-state freezes the ORACLE "
            "single-geometry reaction-kernel and chart prescription"
        ),
    )
    analyze.add_argument(
        "--minimum-fragment-representation",
        choices=("automatic", "pseudo-bonds", "quaternion"),
        default="automatic",
        help=(
            "Freeze the automatic ORACLE minimum chart or an auditable "
            "pseudo-bond/quaternion comparison chart"
        ),
    )
    analyze.add_argument("--no-validate", action="store_true")
    analyze.add_argument("--json", action="store_true", help="Print the summary as JSON")

    formats = sub.add_parser("formats", help="List supported input formats and extras")
    formats.add_argument("--json", action="store_true")

    doctor = sub.add_parser("doctor", help="Check the standalone ORACLE installation")
    doctor.add_argument("--config", type=Path)
    doctor.add_argument("--json", action="store_true")

    capabilities = sub.add_parser("capabilities", help="Report local and optional remote capabilities")
    capabilities.add_argument("--remote-host")
    capabilities.add_argument("--probe-remote", action="store_true")
    capabilities.add_argument("--json", action="store_true")

    init_config = sub.add_parser("init-config", help="Write a portable configuration template")
    init_config.add_argument("output", type=Path)
    init_config.add_argument("--force", action="store_true")

    examples = sub.add_parser("examples", help="Copy the bundled publication examples")
    examples.add_argument("output", type=Path)
    examples.add_argument("--force", action="store_true")

    report = sub.add_parser(
        "report",
        help="Report an existing ORACLE state without reperceiving it",
    )
    report.add_argument("xyzin", type=Path)
    report.add_argument("--json-output", type=Path)
    report.add_argument("--human-output", type=Path)
    report.add_argument("--json", action="store_true", dest="print_json")
    migrate = sub.add_parser("migrate-report", help="Migrate a legacy JSON report")
    migrate.add_argument("input", type=Path); migrate.add_argument("-o", "--output", type=Path, required=True)
    migrate.add_argument("--backup", action="store_true")

    batch = sub.add_parser(
        "batch",
        help="Run independent ORACLE perceptions concurrently from a JSON manifest",
    )
    batch.add_argument("manifest", type=Path)
    batch.add_argument("--workers", type=int, default=0)
    batch.add_argument("--json", action="store_true", dest="print_json")
    batch.add_argument("--resume", action="store_true")
    batch.add_argument("--retries", type=int, default=0)
    batch.add_argument("--errors-output", type=Path)
    batch.add_argument("--checkpoint", type=Path)

    refine = sub.add_parser(
        "refine-l1",
        help="Convert an ORACLE-enriched L1 geometry to a reanalyzed PL1 state",
    )
    refine.add_argument("source", type=Path)
    refine.add_argument("-o", "--output", type=Path, required=True)
    refine.add_argument("--no-core-valence", action="store_true")
    refine.add_argument("--no-conjugation", action="store_true")
    refine.add_argument("--no-hydrogen-bonds", action="store_true")
    refine.add_argument("--cv-weight-threshold", type=float, default=0.9)
    refine.add_argument("--tolerance", type=float, default=1.0e-8)
    refine.add_argument("--max-iterations", type=int, default=50)
    refine.add_argument("--json", action="store_true")

    initial = sub.add_parser(
        "prepare-initial",
        help="Prepare the frozen LCB26/Cartesian-internal starting structure",
    )
    initial.add_argument("source", help="XYZ path or SMILES string")
    initial.add_argument("-o", "--output", type=Path, required=True)
    initial.add_argument("--lcb26-root", type=Path, required=True)
    initial.add_argument("--declared-level", default="AUTO")
    initial.add_argument("--source-kind", choices=("auto", "smiles", "xyz", "geometry", "enriched_xyz"), default="auto")
    initial.add_argument("--max-iterations", type=int, default=30)
    initial.add_argument("--closure-tolerance", type=float, default=1.0e-6)
    initial.add_argument("--json", action="store_true")

    refined = sub.add_parser(
        "refine-structure",
        help="Refine SMILES/XYZ with LCB26 and report fragments and rotational constants",
    )
    refined.add_argument("source", help="XYZ path or SMILES string")
    refined.add_argument("-o", "--output", type=Path, required=True)
    refined.add_argument("--lcb26-root", type=Path, required=True)
    refined.add_argument("--declared-level", default="AUTO")
    refined.add_argument("--source-kind", choices=("auto", "smiles", "xyz", "geometry", "enriched_xyz"), default="auto")
    refined.add_argument("--fragment-limit", type=int, default=5)
    refined.add_argument("--strict", action="store_true", help="Reject extrapolation without a local LCB26 donor")
    refined.add_argument("--json", action="store_true")

    gui = sub.add_parser("gui", help="Launch the optional ORACLE GUI")
    gui.add_argument("xyzin", nargs="?", type=Path)
    return parser


_SUCCESS_STATUSES = frozenset(
    {"PASS", "WARN", "AWAITING_CARTESIAN_SYMMETRY_CONFIRMATION"}
)


def main(argv: list[str] | None = None, *, prog: str = "oracle") -> int:
    args = build_parser(prog=prog).parse_args(argv)
    try:
        handler = _COMMAND_HANDLERS[args.command]
    except KeyError as exc:  # pragma: no cover - argparse enforces the choices.
        raise AssertionError(f"unhandled ORACLE command: {args.command}") from exc
    return handler(args)


def _command_analyze(args: argparse.Namespace) -> int:
    from .api import analyze_structure

    result = analyze_structure(
        args.source,
        args.output,
        source_kind=args.source_kind,
        config=args.config,
        symmetry_distance=args.symmetry_distance,
        symmetry_inertia=args.symmetry_inertia,
        max_rotation_order=args.max_rotation_order,
        report=args.report,
        human_report=args.human_report,
        topology_snapshot=args.snapshot,
        validate=not args.no_validate,
        cartesian_symmetrization=args.cartesian_symmetrization,
        bond_order_model=args.bond_order_model,
    )
    if args.freeze_onic_contract and result.status in _SUCCESS_STATUSES:
        from .sonic_contract_builder import write_oracle_sonic_contract_from_xyzin

        write_oracle_sonic_contract_from_xyzin(
            result.output,
            minimum_fragment_representation=args.minimum_fragment_representation,
        )
    if args.stationary_point == "transition-state" and result.status in _SUCCESS_STATUSES:
        from .transition_state_geometry import (
            write_oracle_transition_state_geometry_contract_from_xyzin,
        )

        write_oracle_transition_state_geometry_contract_from_xyzin(result.output)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_analysis_result(result)
    return 0 if result.status in _SUCCESS_STATUSES else 2


def _print_analysis_result(result: object) -> None:
    fields = (
        ("output", "output"),
        ("status", "status"),
        ("atoms", "atom_count"),
        ("point_group", "point_group"),
        ("symmetry_operations", "symmetry_operation_count"),
        ("bonds", "bond_count"),
        ("rings", "ring_count"),
        ("aromatic_atoms", "aromatic_atom_count"),
        ("synthons", "synthon_count"),
        ("primitives", "primitive_count"),
        ("primitive_b_rank", "primitive_b_matrix_rank"),
        ("primitive_b_sha256", "primitive_b_matrix_sha256"),
        ("topology_sha256", "topology_sha256"),
    )
    for label, attribute in fields:
        print(f"{label}: {getattr(result, attribute)}")
    for attribute in ("report", "human_report", "topology_snapshot"):
        value = getattr(result, attribute)
        if value is not None:
            print(f"{attribute}: {value}")


def _command_formats(args: argparse.Namespace) -> int:
    from .api import SUPPORTED_INPUT_FORMATS

    records = [
        {"kind": kind, "input": description, "availability": availability}
        for kind, description, availability in SUPPORTED_INPUT_FORMATS
    ]
    if args.json:
        print(json.dumps(records, indent=2))
    else:
        for record in records:
            print(
                f"{record['kind']:14s} {record['input']:38s} "
                f"{record['availability']}"
            )
    return 0


def _command_doctor(args: argparse.Namespace) -> int:
    return _doctor(args.config, as_json=args.json)


def _command_capabilities(args: argparse.Namespace) -> int:
    from .dependencies import dependency_status
    from .remote import local_qm_capabilities, probe_remote_qm, remote_qm_manifest

    payload = {
        "schema": "matrix.oracle.capabilities.v1",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": dependency_status(),
        "local_qm": local_qm_capabilities(),
    }
    if args.remote_host:
        payload["remote_qm"] = (
            probe_remote_qm(args.remote_host)
            if args.probe_remote
            else remote_qm_manifest(args.remote_host)
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _command_init_config(args: argparse.Namespace) -> int:
    from .config import write_oracle_config_template

    path = write_oracle_config_template(args.output, overwrite=args.force)
    print(path)
    return 0


def _command_examples(args: argparse.Namespace) -> int:
    for path in _copy_examples(args.output, overwrite=args.force):
        print(path)
    return 0


def _command_report(args: argparse.Namespace) -> int:
    from .api import write_oracle_analysis_reports

    human_output = args.human_output
    if human_output is None and args.json_output is None:
        human_output = args.xyzin.with_name(f"{args.xyzin.stem}.oracle.txt")
    payload = write_oracle_analysis_reports(
        args.xyzin,
        json_output=args.json_output,
        human_output=human_output,
    )
    if args.print_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for output in (args.json_output, human_output):
            if output is not None:
                print(output)
    return 0


def _command_migrate_report(args: argparse.Namespace) -> int:
    from .migrations import migrate_analysis_report

    source_text = args.input.read_text(encoding="utf-8")
    migrated = migrate_analysis_report(json.loads(source_text))
    if args.backup:
        args.input.with_suffix(args.input.suffix + ".bak").write_text(
            source_text,
            encoding="utf-8",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(migrated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


def _command_batch(args: argparse.Namespace) -> int:
    from .api import ORACLE_BATCH_SCHEMA, analyze_structures
    from .batch import checkpoint_batch, pending_requests
    from .validation import validate_xyzin_output

    requests = _batch_requests(args.manifest)
    if args.resume:
        requests = pending_requests(requests)
    results = analyze_structures(requests, workers=args.workers)
    for _ in range(max(0, args.retries)):
        failed = tuple(
            request
            for request, result in zip(requests, results, strict=True)
            if result.status not in {"PASS", "WARN"}
        )
        if not failed:
            break
        retried = analyze_structures(failed, workers=args.workers)
        result_by_output = {str(result.output): result for result in results}
        result_by_output.update({str(result.output): result for result in retried})
        results = tuple(
            result_by_output[str(request.output.resolve())] for request in requests
        )
    payload = {
        "schema": ORACLE_BATCH_SCHEMA,
        "manifest": str(args.manifest.resolve()),
        "requested_workers": args.workers,
        "count": len(results),
        "results": [
            {
                **result.to_dict(),
                "output_validation": validate_xyzin_output(result.output),
            }
            for result in results
        ],
    }
    if args.print_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for result in results:
            print(f"{result.status:5s} {result.output}")
    if args.errors_output is not None:
        failed_payload = {
            "schema": "matrix.oracle.batch_errors.v1",
            "errors": [
                result.to_dict()
                for result in results
                if result.status not in {"PASS", "WARN"}
            ],
        }
        args.errors_output.write_text(
            json.dumps(failed_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.checkpoint is not None:
        checkpoint_batch(
            [
                str(result.output)
                for result in results
                if result.status in {"PASS", "WARN"}
            ],
            args.checkpoint,
        )
    return 0 if all(result.status in {"PASS", "WARN"} for result in results) else 2


def _command_refine_l1(args: argparse.Namespace) -> int:
    from .refinement import refine_l1_geometry

    result = refine_l1_geometry(
        args.source,
        args.output,
        include_core_valence=not args.no_core_valence,
        include_conjugation=not args.no_conjugation,
        include_hydrogen_bonds=not args.no_hydrogen_bonds,
        cv_weight_threshold=args.cv_weight_threshold,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
    )
    payload = {
        "source": str(result.source),
        "output": str(result.output),
        "input_level": "L1",
        "output_level": "PL1",
        "target_count": result.target_count,
        "iterations": result.back_transformation.iterations,
        "maximum_residual": result.back_transformation.maximum_residual,
        "rotational_constants_MHz": dict(
            zip(
                ("A_e", "B_e", "C_e"),
                result.rotational_constants_mhz,
                strict=True,
            )
        ),
        "principal_moments_amu_angstrom2": list(
            result.principal_moments_amu_angstrom2
        ),
        "status": result.analysis.status,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def _command_prepare_initial(args: argparse.Namespace) -> int:
    from .initial_structure import prepare_initial_structure

    result = prepare_initial_structure(
        args.source,
        args.output,
        lcb26_root=args.lcb26_root,
        declared_level=args.declared_level,
        source_kind=args.source_kind,
        max_iterations=args.max_iterations,
        closure_tolerance=args.closure_tolerance,
    )
    print(
        json.dumps(result.to_dict(), indent=2, sort_keys=True)
        if args.json
        else result.output_xyz
    )
    return 0 if result.closure_converged else 2


def _command_refine_structure(args: argparse.Namespace) -> int:
    from .structure_refinement import refine_structure

    result = refine_structure(
        args.source,
        args.output,
        lcb26_root=args.lcb26_root,
        declared_level=args.declared_level,
        source_kind=args.source_kind,
        fragment_limit=args.fragment_limit,
        strict=args.strict,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"output: {result.output_xyz}")
        print(f"report: {result.report}")
        print(
            "rotational_constants_MHz: "
            + ", ".join(f"{value:.6f}" for value in result.rotational_constants_MHz)
        )
        print(
            "fragments: "
            + ", ".join(str(item.get("identifier")) for item in result.fragments)
        )
    return 0


def _command_gui(args: argparse.Namespace) -> int:
    from .gui import run as gui_run

    gui_args = [str(args.xyzin)] if args.xyzin is not None else []
    return gui_run(gui_args)


_COMMAND_HANDLERS = {
    "analyze": _command_analyze,
    "formats": _command_formats,
    "doctor": _command_doctor,
    "capabilities": _command_capabilities,
    "init-config": _command_init_config,
    "examples": _command_examples,
    "report": _command_report,
    "migrate-report": _command_migrate_report,
    "batch": _command_batch,
    "refine-l1": _command_refine_l1,
    "prepare-initial": _command_prepare_initial,
    "refine-structure": _command_refine_structure,
    "gui": _command_gui,
}


def _batch_requests(path: Path):
    from .api import ORACLE_BATCH_SCHEMA, OracleAnalysisRequest

    manifest = Path(path).expanduser().resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema") != ORACLE_BATCH_SCHEMA:
        raise ValueError(f"ORACLE batch manifest must use {ORACLE_BATCH_SCHEMA}")
    records = payload.get("requests")
    if not isinstance(records, list) or not records:
        raise ValueError("ORACLE batch manifest requires a non-empty requests list")
    base = manifest.parent

    def selected(record: dict[str, object], name: str, *, required: bool = False) -> Path | None:
        value = str(record.get(name, "")).strip()
        if not value:
            if required:
                raise ValueError(f"ORACLE batch request is missing {name}")
            return None
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (base / candidate).resolve()

    return tuple(
        OracleAnalysisRequest(
            source=selected(record, "source", required=True),  # type: ignore[arg-type]
            output=selected(record, "output", required=True),  # type: ignore[arg-type]
            source_kind=str(record.get("source_kind", "auto")),
            report=selected(record, "report"),
            human_report=selected(record, "human_report"),
            topology_snapshot=selected(record, "topology_snapshot"),
            config=selected(record, "config"),
            validate=bool(record.get("validate", True)),
        )
        for record in records
        if isinstance(record, dict)
    )


def matrix_main(argv: list[str] | None = None) -> int:
    return main(argv, prog="matrix oracle")


def _doctor(config_path: Path | None, *, as_json: bool) -> int:
    from .api import oracle_version
    from .config import load_oracle_config

    config = load_oracle_config(config_path)
    required_modules = (
        "numpy",
        "matrix_core",
        "matrix_chem",
        "matrix_switch",
        "matrix_oracle",
    )
    optional_modules = (
        "matrix_link",
        "matrix_gaussian",
        "matrix_molpro",
        "matrix_mrcc",
        "matrix_orca",
        "PySide6",
    )
    required = {name: find_spec(name) is not None for name in required_modules}
    optional = {name: find_spec(name) is not None for name in optional_modules}
    paths = {
        "data_dir": config.paths.data_dir,
        "cache_dir": config.paths.cache_dir,
        "work_dir": config.paths.work_dir,
    }
    payload = {
        "oracle_version": oracle_version(),
        "python": platform.python_version(),
        "python_supported": sys.version_info >= (3, 11),
        "config": str(config.source) if config.source is not None else None,
        "required_modules": required,
        "optional_modules": optional,
        "paths": {
            name: {
                "value": str(path) if path is not None else None,
                "exists": path.is_dir() if path is not None else None,
            }
            for name, path in paths.items()
        },
        "symmetry": {
            "distance_angstrom": config.symmetry.distance_angstrom,
            "inertia_relative": config.symmetry.inertia_relative,
            "max_rotation_order": config.symmetry.max_rotation_order,
        },
    }
    ok = bool(payload["python_supported"]) and all(required.values())
    payload["status"] = "PASS" if ok else "FAIL"
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"ORACLE {payload['oracle_version']}: {payload['status']}")
        print(f"Python {payload['python']}: {'PASS' if payload['python_supported'] else 'FAIL'}")
        for name, present in required.items():
            print(f"required {name}: {'PASS' if present else 'MISSING'}")
        for name, present in optional.items():
            print(f"optional {name}: {'AVAILABLE' if present else 'not installed'}")
        for name, path in paths.items():
            if path is None:
                print(f"path {name}: not configured")
            else:
                state = "exists" if path.is_dir() else "does not exist"
                print(f"path {name}: {path} ({state})")
    return 0 if ok else 1


def _copy_examples(output: Path, *, overwrite: bool) -> tuple[Path, ...]:
    target = Path(output).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    source = resources.files("matrix_oracle.data").joinpath("examples")
    copied: list[Path] = []
    for item in source.iterdir():
        if not item.is_file():
            continue
        destination = target / item.name
        if destination.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite example: {destination}")
        with resources.as_file(item) as local:
            shutil.copyfile(local, destination)
        copied.append(destination)
    return tuple(sorted(copied))


if __name__ == "__main__":
    raise SystemExit(main())
