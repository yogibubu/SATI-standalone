"""Command-line access to ARCHITECT and its ZAFF force-field runtime."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from matrix_numerics import compute_backend_report

from .acquisition import (
    ExternalZaffAnharmonicEvaluator,
    ZaffAnharmonicAcquisitionConfig,
    acquire_zaff_anharmonic_model,
    attach_acquired_anharmonic_model,
)
from .compiler import (
    EXACT_TERM_POLICY,
    HYBRID_PROTECTED_POLICY,
    REFITTED_POLICY,
    ZaffBuildConfig,
    build_zaff_force_field,
)
from .cm5_baseline import derive_cm5_intrinsic_baseline
from .model import (
    evaluate_force_field,
    load_force_field,
    save_force_field,
    validate_force_field_derivatives,
)
from .library import (
    add_force_field_to_library,
    create_zaff_library,
    default_zaff_library_path,
    list_zaff_library,
    resolve_zaff_force_field,
)
from .initial_geometry import (
    resolve_initial_geometry_provider,
    uff_point_engine_command,
)
from .seed_compiler import build_zaff_seed_force_field
from .uff import (
    compile_uff_model_from_xyzin,
    evaluate_uff_model,
    load_uff_model,
    save_uff_model,
)
from matrix_qm import (
    LinearFitProblem,
    PropertySurfaceProblem,
    fit_linear_surface,
    fit_property_surface,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="architect",
        description="ARCHITECT constructor and runtime for ZAFF force fields",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    pruning = sub.add_parser("prune-couplings", help="Remove insignificant native coupling terms under observable controls")
    pruning.add_argument("request",type=Path)
    zaff4 = sub.add_parser("zaff4", help="Build a native ZAFF4 local candidate including four-primitive products")
    zaff4.add_argument("request",type=Path)
    nonlocal_fit = sub.add_parser("zaff4-nonlocal", help="Fit local derivatives and measured nonlocal E/G/H jointly")
    nonlocal_fit.add_argument("request", type=Path)
    backend = sub.add_parser(
        "backend",
        help="Report CPU/GPU/Neural Engine backends and runtime policies",
    )
    backend.add_argument("--json", action="store_true", dest="json_output")
    library = sub.add_parser(
        "library",
        help="Create, populate, inspect or query the ZAFF force-field library",
    )
    library_sub = library.add_subparsers(dest="library_command", required=True)
    library_create = library_sub.add_parser("create", help="Create an empty library")
    library_create.add_argument("path", type=Path)
    library_add = library_sub.add_parser("add", help="Validate and add a compiled ZAFF field")
    library_add.add_argument("force_field", type=Path)
    library_add.add_argument("--library", type=Path, default=default_zaff_library_path())
    library_add.add_argument("--id")
    library_add.add_argument("--charge", type=int)
    library_add.add_argument("--multiplicity", type=int)
    library_add.add_argument("--priority", type=int, default=0)
    library_seed = library_sub.add_parser(
        "seed",
        help="Compile and register a transferable diagonal ZAFF field",
    )
    library_seed.add_argument("geometry", type=Path)
    library_seed.add_argument("--charges-file", type=Path)
    library_seed.add_argument("--bond-orders-file", type=Path)
    library_seed.add_argument("--gaff-parameters", type=Path)
    library_seed.add_argument("--library", type=Path, default=default_zaff_library_path())
    library_seed.add_argument("--output", type=Path, required=True)
    library_seed.add_argument("--id")
    library_seed.add_argument("--charge", type=int, default=0)
    library_seed.add_argument("--multiplicity", type=int, default=1)
    library_seed.add_argument("--priority", type=int, default=0)
    library_list = library_sub.add_parser("list", help="List registered fields")
    library_list.add_argument("--library", type=Path, default=default_zaff_library_path())
    library_resolve = library_sub.add_parser(
        "resolve", help="Resolve a molecule or report the GFN-FF fallback"
    )
    library_resolve.add_argument("geometry", type=Path)
    library_resolve.add_argument("--library", type=Path, default=default_zaff_library_path())
    library_resolve.add_argument("--charge", type=int, default=0)
    library_resolve.add_argument("--multiplicity", type=int, default=1)
    build = sub.add_parser(
        "build",
        help="Compile a ZAFF model after a PIC/local-SONIC Pareto comparison",
    )
    build.add_argument("xyzin", type=Path)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--apoc-source",
        type=Path,
        help="QM output, Gaussian FCHK or Molden file from which APOC creates CM5/Mayer data",
    )
    build.add_argument(
        "--apoc-format",
        choices=(
            "auto",
            "gaussian",
            "gaussian-fchk",
            "orca",
            "molden",
            "pyscf-output",
        ),
        default="auto",
    )
    build.add_argument("--apoc-grid-level", type=int, default=4)
    build.add_argument("--apoc-charge", type=int)
    build.add_argument("--charges-file", type=Path)
    build.add_argument(
        "--charge-model",
        help="Required scientific label for an explicit --charges-file",
    )
    build.add_argument("--bond-orders-file", type=Path)
    build.add_argument(
        "--bond-order-model",
        help="Required scientific label for an explicit --bond-orders-file",
    )
    build.add_argument("--no-electrostatic", action="store_true")
    build.add_argument("--no-uff-vdw", action="store_true")
    build.add_argument("--allow-inexact-equilibrium", action="store_true")
    build.add_argument(
        "--coordinate-chart",
        choices=("adaptive", "primitive", "sonic"),
        default="adaptive",
        help=(
            "select primitive PICs, stretch-preserving SMITH SONICs, or the "
            "automatic comparison (stretches are never combined)"
        ),
    )
    build.add_argument(
        "--workers",
        type=int,
        default=0,
        help="parallel coupling-score workers (0 selects up to 32 available CPUs)",
    )
    build.add_argument(
        "--coupling-policy",
        choices=("hybrid", "exact", "refitted"),
        default="exact",
        help=(
            "exact retains the complete QM harmonic field (default); hybrid and "
            "refitted provide explicit truncated diagnostic models"
        ),
    )
    build.add_argument("--protected-mixing-threshold", type=float, default=1.0)
    build.add_argument("--protected-second-order-threshold", type=float, default=0.10)
    build.add_argument("--protected-gap-floor", type=float, default=0.10)
    build.add_argument(
        "--nonbonded-cutoff",
        type=float,
        default=12.0,
        help="regional non-bonded neighbor cutoff in angstrom (default: 12)",
    )
    build.add_argument(
        "--fmm-precision",
        type=float,
        default=1.0e-10,
        help="requested relative precision for long-range FMM electrostatics",
    )
    build.add_argument(
        "--fmm-minimum-atoms",
        type=int,
        default=256,
        help="minimum atom count for automatic FMM selection (default: 256)",
    )
    build.add_argument(
        "--electrostatic-width-scale",
        type=float,
        default=1.0,
        help="scale applied to element Gaussian charge widths (default: 1)",
    )
    build.add_argument(
        "--gaff-parameters",
        type=Path,
        help="official gaff.dat/gaff2.dat used for transferable full-fit priors",
    )
    build.add_argument(
        "--no-transferable-seed-priors",
        action="store_true",
        help="disable MM3/GAFF starting values and regularized full-fit priors",
    )
    build.add_argument("--json", action="store_true", dest="json_output")
    apoc = sub.add_parser(
        "apoc",
        help="Run APOC directly and attach its CM5/Mayer contract before ZAFF construction",
    )
    apoc.add_argument("source", type=Path)
    apoc.add_argument("--xyzin", type=Path, required=True)
    apoc.add_argument("--output", type=Path)
    apoc.add_argument(
        "--format",
        choices=("auto", "gaussian", "orca", "molden", "pyscf-output"),
        default="auto",
        dest="source_format",
    )
    apoc.add_argument("--grid-level", type=int, default=4)
    apoc.add_argument("--charge", type=int)
    inspect = sub.add_parser("inspect", help="Validate and summarize a force-field model")
    inspect.add_argument("force_field", type=Path)
    inspect.add_argument("--json", action="store_true", dest="json_output")
    parameters = sub.add_parser(
        "parameters",
        help="Write the final ZAFF parameters as a human-readable report",
    )
    parameters.add_argument("force_field", type=Path)
    parameters.add_argument("--output", type=Path)
    zoom = sub.add_parser("zoom", help="Compare the quantitative ZAFF zoom levels")
    zoom.add_argument("force_field", type=Path)
    zoom.add_argument("--json", action="store_true", dest="json_output")
    validate = sub.add_parser(
        "validate",
        help="Finite-difference check of the public analytic E/G/H runtime",
    )
    validate.add_argument("force_field", type=Path)
    validate.add_argument("--geometry", type=Path)
    validate.add_argument("--zoom-level", choices=("H0", "H1", "H2", "H3"))
    validate.add_argument(
        "--xyzin",
        type=Path,
        help="Activate the compiled local-SONIC anharmonic runtime",
    )
    validate.add_argument(
        "--step-bohr",
        type=float,
        help="Defaults to 1e-4 harmonic or 5e-5 local-SONIC",
    )
    validate.add_argument("--gradient-tolerance", type=float, default=1.0e-8)
    validate.add_argument(
        "--hessian-tolerance",
        type=float,
        help="Defaults to 1e-8 harmonic or 2e-8 local-SONIC",
    )
    validate.add_argument("--json", action="store_true", dest="json_output")
    evaluate = sub.add_parser("evaluate", help="Calculate force-field E/G/H")
    evaluate.add_argument("force_field", type=Path)
    evaluate.add_argument("geometry", type=Path)
    evaluate.add_argument(
        "--property",
        choices=("energy", "gradient", "hessian"),
        action="append",
        default=[],
    )
    evaluate.add_argument("--zoom-level", choices=("H0", "H1", "H2", "H3"))
    evaluate.add_argument(
        "--xyzin",
        type=Path,
        help="Activate the compiled local-SONIC anharmonic runtime",
    )
    evaluate.add_argument("--output", type=Path)
    acquire = sub.add_parser(
        "acquire-anharmonic",
        help="Acquire 0,+/-h,+/-2h SONIC gradients and planar double-well points",
    )
    acquire.add_argument("xyzin", type=Path)
    acquire.add_argument("force_field", type=Path)
    acquire.add_argument("--output", type=Path, required=True)
    acquire.add_argument("--run-directory", type=Path, required=True)
    acquire.add_argument(
        "--command",
        dest="engine_command",
        required=True,
        help="external E/G command containing {request} and {response}",
    )
    acquire.add_argument("--workers", type=int, default=0)
    acquire.add_argument(
        "--stretch-model",
        choices=("morse", "inverse_power", "spf"),
        default="spf",
    )
    surface_fit = sub.add_parser(
        "fit-surface",
        help="Solve a weighted linear surface fit with exact constraints",
    )
    surface_fit.add_argument("problem", type=Path)
    surface_fit.add_argument("--output", type=Path, required=True)
    property_fit = sub.add_parser(
        "fit-property-surface",
        help="Fit scalar, vector or tensor property surfaces with exact constraints",
    )
    property_fit.add_argument("problem", type=Path)
    property_fit.add_argument("--output", type=Path, required=True)
    uff_compile = sub.add_parser(
        "uff-compile",
        help="Compile an ARCHITECT UFF model from an ORACLE-enriched geometry",
    )
    uff_compile.add_argument("xyzin", type=Path)
    uff_compile.add_argument("--output", type=Path, required=True)
    initial_provider = sub.add_parser(
        "initial-provider",
        help="Resolve an ARCHITECT-owned initial-geometry provider as JSON",
    )
    initial_provider.add_argument("provider")
    initial_provider.add_argument("--xtb-executable")
    initial_provider.add_argument("--uff-model", type=Path)
    uff_point = sub.add_parser(
        "uff-point",
        help="Evaluate a compiled UFF model through the LINK point-result contract",
    )
    uff_point.add_argument("model", type=Path)
    uff_point.add_argument("geometry", type=Path)
    uff_point.add_argument("--result", type=Path, required=True)
    uff_point.add_argument("--point-index", type=int, default=0)
    uff_point.add_argument(
        "--properties",
        default="energy,gradient",
        help="comma-separated energy, gradient and/or hessian",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prune-couplings":
        from .native_coupling_pruning import run_native_coupling_pruning
        try:result=run_native_coupling_pruning(args.request)
        except (OSError,ValueError,RuntimeError,MemoryError) as error:
            print(f"ARCHITECT coupling pruning: {error}");return 1
        print(json.dumps(result,indent=2))
        return 0 if result["reduction_accepted"] else 1
    if args.command == "zaff4-nonlocal":
        from .native_nonlocal_fit import run_native_nonlocal_protocol
        try:result=run_native_nonlocal_protocol(args.request)
        except (OSError,ValueError,RuntimeError,MemoryError) as error:
            print(f"ARCHITECT joint ZAFF4: {error}");return 1
        print(json.dumps(result,indent=2))
        return 0 if result["local_and_nonlocal_gates_passed"] else 1
    if args.command == "zaff4":
        from .zaff4 import run_zaff4_native_protocol
        try:result=run_zaff4_native_protocol(args.request)
        except (OSError,ValueError,RuntimeError,MemoryError) as error:
            print(f"ARCHITECT ZAFF4: {error}");return 1
        print(json.dumps(result,indent=2))
        return 0 if result["local_derivatives_passed"] else 1
    if args.command == "backend":
        payload = compute_backend_report()
        if args.json_output:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            policy = payload["policy"]
            print(f"device policy: {policy['device']}")
            print(f"GPU minimum batch: {policy['gpu_min_batch']}")
            print(f"GPU validation: {policy['gpu_validation']}")
            print(f"Neural Engine policy: {policy['neural_engine']}")
            print(f"Neural Engine validation: {policy['neural_engine_validation']}")
            for item in payload["backends"]:
                status = "available" if item["available"] else "unavailable"
                detail = f" ({item['reason']})" if item["reason"] else ""
                cores = f", {item['core_count']} cores" if item["core_count"] else ""
                print(
                    f"{item['device']}: {item['backend']} {item['precision']} "
                    f"{status}{cores}{detail}"
                )
        return 0
    if args.command == "library":
        if args.library_command == "create":
            print(create_zaff_library(args.path))
            return 0
        if args.library_command == "add":
            entry = add_force_field_to_library(
                args.library,
                args.force_field,
                identifier=args.id,
                charge=args.charge,
                multiplicity=args.multiplicity,
                priority=args.priority,
            )
            print(json.dumps(entry, indent=2, sort_keys=True))
            return 0
        if args.library_command == "seed":
            atoms, coordinates = _read_xyz(args.geometry)
            charges = _read_explicit_charges(args.charges_file)
            bond_order_rows = _read_explicit_bond_orders(args.bond_orders_file)
            mayer_bond_orders = (
                None
                if bond_order_rows is None
                else {
                    tuple(sorted((int(left), int(right)))): float(value)
                    for left, right, value in bond_order_rows
                }
            )
            field = build_zaff_seed_force_field(
                atoms,
                coordinates,
                charges,
                mayer_bond_orders=mayer_bond_orders,
                gaff_path=args.gaff_parameters,
                charge=args.charge,
                multiplicity=args.multiplicity,
                charge_source=(
                    "ORACLE electronegativity estimate"
                    if args.charges_file is None
                    else f"CM5 {args.charges_file.resolve()}"
                ),
                bond_order_source=(
                    "ORACLE Pauling estimate"
                    if args.bond_orders_file is None
                    else f"Mayer {args.bond_orders_file.resolve()}"
                ),
                charge_owner="USER",
                bond_order_owner="USER",
            )
            target = save_force_field(args.output, field)
            entry = add_force_field_to_library(
                args.library,
                target,
                identifier=args.id,
                charge=args.charge,
                multiplicity=args.multiplicity,
                priority=args.priority,
            )
            print(json.dumps(entry, indent=2, sort_keys=True))
            return 0
        if args.library_command == "list":
            print(json.dumps(list_zaff_library(args.library), indent=2, sort_keys=True))
            return 0
        atoms, coordinates = _read_xyz(args.geometry)
        resolution = resolve_zaff_force_field(
            atoms,
            coordinates,
            charge=args.charge,
            multiplicity=args.multiplicity,
            library=args.library,
        )
        print(json.dumps(resolution.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "build":
        explicit_charges = _read_explicit_charges(args.charges_file)
        field = build_zaff_force_field(
            args.xyzin,
            config=ZaffBuildConfig(
                subtract_electrostatic=not args.no_electrostatic,
                subtract_uff_vdw=not args.no_uff_vdw,
                allow_inexact_equilibrium=args.allow_inexact_equilibrium,
                coordinate_chart=args.coordinate_chart,
                parallel_workers=args.workers,
                coupling_policy=(
                    EXACT_TERM_POLICY
                    if args.coupling_policy == "exact"
                    else REFITTED_POLICY
                    if args.coupling_policy == "refitted"
                    else HYBRID_PROTECTED_POLICY
                ),
                protected_coupling_mixing_threshold=args.protected_mixing_threshold,
                protected_coupling_second_order_threshold=(args.protected_second_order_threshold),
                protected_diagonal_gap_floor_fraction=args.protected_gap_floor,
                nonbonded_neighbor_cutoff_angstrom=args.nonbonded_cutoff,
                fmm_precision=args.fmm_precision,
                fmm_minimum_atoms=args.fmm_minimum_atoms,
                electrostatic_gaussian_width_scale=args.electrostatic_width_scale,
                use_transferable_seed_priors=not args.no_transferable_seed_priors,
                gaff_parameter_path=args.gaff_parameters,
                apoc_source=(None if args.apoc_source is None else str(args.apoc_source)),
                apoc_format=args.apoc_format,
                apoc_grid_level=args.apoc_grid_level,
                apoc_charge=args.apoc_charge,
                explicit_charge_model=args.charge_model,
                explicit_atomic_charges_e=explicit_charges,
                explicit_bond_order_model=args.bond_order_model,
                explicit_bond_orders=_read_explicit_bond_orders(args.bond_orders_file),
            ),
        )
        metadata = dict(field.metadata)
        if explicit_charges is not None:
            metadata["cm5_baseline"] = derive_cm5_intrinsic_baseline(
                explicit_charges,
                molecular_charge=args.charge,
            ).to_dict()
        metadata["zaff1_residual_ledger"] = {
            "status": "REQUIRES_QM_HESSIAN_AND_EXPLICIT_TERM_COMPONENTS",
            "definition": "H_QM_MINUS_ZAFF_FAST_FIXED_COMPONENTS",
        }
        field = replace(field, metadata=metadata)
        target = save_force_field(args.output, field)
        if args.json_output:
            print(json.dumps(_summary(field), indent=2, sort_keys=True))
        else:
            print(target)
            print(f"selected coordinate basis: {field.coordinate_contract['selected_basis']}")
        return 0
    if args.command == "apoc":
        from matrix_apoc import analyze_source, attach_apoc_analysis, save_apoc_analysis

        analysis = analyze_source(
            args.source,
            source_format=args.source_format,
            grid_level=args.grid_level,
            charge=args.charge,
        )
        if args.output is not None:
            print(save_apoc_analysis(args.output, analysis))
        print(attach_apoc_analysis(args.xyzin, analysis))
        return 0
    if args.command == "fit-surface":
        payload = json.loads(args.problem.read_text(encoding="utf-8"))
        result = fit_linear_surface(LinearFitProblem.from_dict(payload))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(args.output)
        print(
            f"rank {result.diagnostics.numerical_rank}/"
            f"{result.diagnostics.free_dimension_after_constraints}; "
            f"weighted RMS {result.diagnostics.weighted_rms:.8g}; "
            f"max constraint error "
            f"{result.diagnostics.maximum_constraint_error:.3e}"
        )
        return 0
    if args.command == "fit-property-surface":
        payload = json.loads(args.problem.read_text(encoding="utf-8"))
        result = fit_property_surface(PropertySurfaceProblem.from_dict(payload))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        diagnostics = result.linear_fit.diagnostics
        print(args.output)
        print(
            f"{len(result.component_labels)} components; "
            f"rank {diagnostics.numerical_rank}/"
            f"{diagnostics.free_dimension_after_constraints}; "
            f"weighted RMS {diagnostics.weighted_rms:.8g}; "
            f"max constraint error {diagnostics.maximum_constraint_error:.3e}"
        )
        return 0
    if args.command == "uff-compile":
        print(save_uff_model(args.output, compile_uff_model_from_xyzin(args.xyzin)))
        return 0
    if args.command == "initial-provider":
        selected = resolve_initial_geometry_provider(
            args.provider,
            xtb_executable=args.xtb_executable,
        )
        payload = selected.to_dict()
        if args.uff_model is not None:
            if selected.runtime_backend != "architect_uff":
                raise ValueError("--uff-model is valid only for the UFF provider")
            payload["engine_command"] = uff_point_engine_command(args.uff_model)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "uff-point":
        model = load_uff_model(args.model)
        atoms, coordinates = _read_xyz(args.geometry)
        if atoms != model.atoms:
            raise ValueError("UFF point atoms differ from the compiled model")
        requested_properties = str(args.properties).strip() or "energy,gradient"
        properties = tuple(
            value.strip().casefold()
            for value in requested_properties.split(",")
            if value.strip()
        )
        evaluation = evaluate_uff_model(model, coordinates, properties=properties)
        payload = {
            "schema": "oracle.link.point_result.v1",
            "point_index": int(args.point_index),
            "displacement": 0.0,
            "energy_hartree": evaluation.energy_hartree,
            "backend_coordinates_angstrom": coordinates.tolist(),
            "frame_alignment_rms_angstrom": 0.0,
            "status": "completed",
            "source": f"ARCHITECT UFF {args.model.resolve()}",
            "execution": {
                "backend": "architect_uff",
                "model_schema": model.schema,
                "model": str(args.model.resolve()),
                "properties": list(properties),
                "energy_evaluations": 1,
                "gradient_evaluations": int(evaluation.gradient_hartree_per_bohr is not None),
                "hessian_evaluations": int(evaluation.hessian_hartree_per_bohr2 is not None),
                "hessian_contract": (
                    "linear_Wilson_congruence_without_B_prime_plus_exact_UFF_vdw"
                ),
            },
        }
        if evaluation.gradient_hartree_per_bohr is not None:
            payload["gradient_hartree_per_bohr"] = (
                evaluation.gradient_hartree_per_bohr.tolist()
            )
        if evaluation.hessian_hartree_per_bohr2 is not None:
            payload["hessian_hartree_per_bohr2"] = (
                evaluation.hessian_hartree_per_bohr2.tolist()
            )
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(args.result)
        return 0
    field = load_force_field(args.force_field)
    if args.command == "acquire-anharmonic":
        if field.anharmonic_model is None:
            raise ValueError("force field has no serialized ZAFF anharmonic plan")
        config = ZaffAnharmonicAcquisitionConfig(
            parallel_workers=args.workers,
            stretch_model=args.stretch_model,
        )
        evaluator = ExternalZaffAnharmonicEvaluator(args.engine_command, args.run_directory)
        model = acquire_zaff_anharmonic_model(
            args.xyzin,
            field.atoms,
            field.reference_coordinates_angstrom,
            evaluator,
            plan=field.anharmonic_model,
            config=config,
        )
        target = save_force_field(
            args.output,
            attach_acquired_anharmonic_model(field, model),
        )
        print(target)
        print(f"compiled anharmonic terms: {len(model.terms)}")
        return 0
    if args.command == "inspect":
        if args.json_output:
            print(json.dumps(_summary(field), indent=2, sort_keys=True))
        else:
            summary = _summary(field)
            for key, value in summary.items():
                print(f"{key}: {value}")
        return 0
    if args.command == "parameters":
        report = _parameter_report(field)
        if args.output is None:
            print(report)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report + "\n", encoding="utf-8")
            print(args.output)
        return 0
    if args.command == "zoom":
        payload = {
            "selected_basis": field.coordinate_contract.get("selected_basis"),
            "basis_comparison": field.metadata.get("basis_comparison", {}),
            "zoom_levels": [level.to_dict() for level in field.zoom_levels],
        }
        if args.json_output:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            baselines = _diagonal_baselines(field)
            if baselines:
                print("diagonal baselines (current force-field standard)")
                print("basis        coordinates error(%) max-error sign-mismatches")
                for baseline in baselines:
                    print(
                        f"{baseline['basis']:<12} "
                        f"{baseline['coordinate_count']:>11d} "
                        f"{baseline['relative_l2_eigenvalue_error_percent']:>8.3f} "
                        f"{baseline['maximum_absolute_eigenvalue_error_hartree_per_bohr2']:>9.3f} "
                        f"{baseline['resolved_sign_mismatch_count']:>15d}"
                    )
                print()
            print("level basis        parameters couplings error(%) sign-mismatches")
            for level in field.zoom_levels:
                print(
                    f"{level.identifier:<5} {level.coordinate_basis:<12} "
                    f"{level.parameter_count:>10d} {level.coupling_count:>9d} "
                    f"{100.0 * level.relative_l2_eigenvalue_error:>8.3f} "
                    f"{level.resolved_sign_mismatch_count:>15d}"
                )
        return 0
    if args.command == "validate":
        coordinates = None
        if args.geometry is not None:
            atoms, coordinates = _read_xyz(args.geometry)
            if atoms != field.atoms:
                raise ValueError("geometry atoms differ from the ZAFF model")
        validation = validate_force_field_derivatives(
            field,
            coordinates,
            zoom_level=args.zoom_level,
            xyzin=args.xyzin,
            step_bohr=args.step_bohr,
            gradient_tolerance=args.gradient_tolerance,
            hessian_tolerance=args.hessian_tolerance,
        )
        payload = validation.to_dict()
        if args.json_output:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            for key, value in payload.items():
                print(f"{key}: {value}")
        return 0 if validation.passed else 1
    atoms, coordinates = _read_xyz(args.geometry)
    if atoms != field.atoms:
        raise ValueError("geometry atoms differ from the ZAFF model")
    result = evaluate_force_field(
        field,
        coordinates,
        properties=tuple(args.property or ("energy", "gradient", "hessian")),
        zoom_level=args.zoom_level,
        xyzin=args.xyzin,
    )
    encoded = json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(args.output)
    return 0


def _summary(field) -> dict[str, object]:
    from .native_pic_field import NativePicField
    if isinstance(field,NativePicField):
        return dict(schema=field.schema,atoms=len(field.atoms),runtime="direct native PIC E/G/H",model_level=field.metadata.get("model_level"),construction_status=field.metadata.get("construction_status"),native_corrections=len(field.corrections),physical_field_qualified=False)
    return {
        "schema": field.schema,
        "atoms": len(field.atoms),
        "stationary": field.stationary,
        "runtime": (
            "analytic local-SONIC anharmonic E/G/H available with --xyzin"
            if field.anharmonic_model is not None and field.anharmonic_model.status == "COMPILED"
            else "exact analytic Cartesian harmonic E/G/H"
        ),
        "selected_basis": field.coordinate_contract.get("selected_basis", "UNSPECIFIED"),
        "active_zoom_level": field.active_zoom_level or "UNSPECIFIED",
        "zoom_level_count": len(field.zoom_levels),
        "anharmonic_status": (
            "ABSENT" if field.anharmonic_model is None else field.anharmonic_model.status
        ),
        "anharmonic_coordinate_count": (
            0 if field.anharmonic_model is None else len(field.anharmonic_model.coordinates)
        ),
    }


def _parameter_report(field) -> str:
    """Return the auditable, human-readable parameter view of a ZAFF model."""

    metadata = field.metadata
    lines = [
        "ZAFF FORCE-FIELD PARAMETERS",
        "===========================",
        f"Schema: {field.schema}",
        f"Atoms: {len(field.atoms)}",
        f"Selected coordinate basis: {field.coordinate_contract.get('selected_basis', 'UNSPECIFIED')}",
        f"Active zoom level: {field.active_zoom_level or 'UNSPECIFIED'}",
        "",
        "NONBONDED PARAMETERS",
        "--------------------",
        f"Charge source: {metadata.get('charge_source', 'UNSPECIFIED')}",
    ]
    charges = metadata.get("atomic_charges_e")
    if charges is None:
        lines.append("Atomic charges: not available")
    else:
        lines.extend(("Atomic charges / e:", " atom element          charge"))
        for index, (atom, charge) in enumerate(zip(field.atoms, charges, strict=True), 1):
            lines.append(f" {index:4d} {atom:>7s} {float(charge):15.8f}")
        lines.append(f" Total charge             {sum(float(value) for value in charges):15.8f}")

    nonbonded = metadata.get("nonbonded_parameters", {})
    lines.extend(
        (
            f"van der Waals model: {nonbonded.get('vdw_model', 'UNSPECIFIED')}",
            f"van der Waals radius scale: {_number(nonbonded.get('vdw_radius_scale'))}",
            f"van der Waals well-depth scale: {_number(nonbonded.get('vdw_well_depth_scale'))}",
            "Exp-PE damping: none",
        )
    )

    refit = metadata.get("equilibrium_parameter_refit", {})
    lines.extend(
        (
            "",
            "EQUILIBRIUM AND BONDED PARAMETERS",
            "---------------------------------",
            f"Refit status: {refit.get('status', 'NOT AVAILABLE')}",
            f"Cartesian gradient RMS / hartree A^-1: {_number(refit.get('total_cartesian_gradient_rms_hartree_per_angstrom'), scientific=True)}",
            f"Cartesian gradient max / hartree A^-1: {_number(refit.get('total_cartesian_gradient_max_hartree_per_angstrom'), scientific=True)}",
            "",
            " ID                         PIC coordinate(s)       kind          form                     q(QM)          q0       shift unit parameters",
        )
    )
    for row in refit.get("functional_parameters", ()):
        indices = ",".join(str(value) for value in row.get("primitive_indices", ()))
        coordinates = ",".join(row.get("primitive_labels", ())) or f"row {indices}"
        parameters = ", ".join(
            f"{name}={float(value):.8g}"
            for name, value in sorted(row.get("parameters", {}).items())
        )
        lines.append(
            f" {str(row.get('primitive_type_id', '?')):<26.26s} "
            f"{coordinates:<23.23s} {str(row.get('kind', '?')):<13.13s} "
            f"{str(row.get('form', '?')):<22.22s} "
            f"{float(row.get('reference_value', 0.0)):12.7f} "
            f"{float(row.get('optimized_equilibrium_value', 0.0)):12.7f} "
            f"{float(row.get('equilibrium_shift', 0.0)):11.3e} "
            f"{str(row.get('value_unit', '?')):<8.8s} {parameters}"
        )
    if not refit.get("functional_parameters"):
        lines.append(" No serialized bonded parameters are available.")

    lines.extend(("", "TORSIONAL FOURIER PARAMETERS", "----------------------------"))
    torsions = refit.get("torsion_fourier_parameters", ())
    if torsions:
        lines.append(
            " ID                         PIC coordinate(s)       phase/rad periodicities fit method"
        )
        for row in torsions:
            indices = ",".join(str(value) for value in row.get("primitive_indices", ()))
            coordinates = ",".join(row.get("primitive_labels", ())) or f"row {indices}"
            periodicities = ",".join(str(value) for value in row.get("periodicities", ()))
            lines.append(
                f" {str(row.get('primitive_type_id', '?')):<26.26s} "
                f"{coordinates:<23.23s} {float(row.get('reference_phase_radian', 0.0)):10.6f} "
                f"{periodicities:<13.13s} {row.get('fit_method', '?')}"
            )
    else:
        lines.append(" None")

    lines.extend(("", "ANHARMONIC SONIC PARAMETERS", "----------------------------"))
    anharmonic = field.anharmonic_model
    if anharmonic is None:
        lines.append(" Status: ABSENT")
    else:
        lines.extend(
            (
                f" Status: {anharmonic.status}",
                f" Basis: {anharmonic.basis}",
                f" Coordinates: {len(anharmonic.coordinates)}",
                f" Requests/results: {len(anharmonic.requests)}/{len(anharmonic.results)}",
            )
        )
        terms_by_index = {term.coordinate_index: term for term in anharmonic.terms}
        lines.append(
            " ID       family             form                              "
            "q(ref)       step unit             k2             k3             k4"
        )
        for coordinate in anharmonic.coordinates:
            term = terms_by_index.get(coordinate.index)
            lines.append(
                f" {coordinate.identifier:<8.8s} {coordinate.family:<18.18s} "
                f"{(term.functional_form if term else coordinate.functional_form):<33.33s} "
                f"{coordinate.reference_value:11.6g} {coordinate.step:10.3g} "
                f"{coordinate.unit:<8.8s} "
                f"{_number(None if term is None else term.quadratic, scientific=True):>14s} "
                f"{_number(None if term is None else term.cubic, scientific=True):>14s} "
                f"{_number(None if term is None else term.quartic, scientific=True):>14s}"
            )
            expansion = " ".join(
                f"{coefficient:+.10g}*{identifier}"
                for identifier, coefficient in coordinate.primitive_components
            )
            lines.append(f"          primitive expansion: {expansion or 'not available'}")
            if term is not None and term.parameters:
                parameters = ", ".join(
                    f"{name}={json.dumps(value, sort_keys=True, separators=(',', ':'))}"
                    for name, value in sorted(term.parameters.items())
                )
                lines.append(f"          parameters: {parameters}")
        if anharmonic.harmonic_reciprocity_audit:
            audit = ", ".join(
                f"{name}={json.dumps(value, sort_keys=True, separators=(',', ':'))}"
                for name, value in sorted(anharmonic.harmonic_reciprocity_audit.items())
            )
            lines.append(f" Harmonic reciprocity audit: {audit}")

        lines.extend(
            (
                "",
                "PERIODIC AND LARGE-AMPLITUDE PARAMETERS",
                "---------------------------------------",
            )
        )
        if anharmonic.periodic_coordinates:
            lines.append(
                " ID       target                 period symmetry reference/deg domain         "
                "barrier/kcal mol-1 status"
            )
            for record in anharmonic.periodic_coordinates:
                reference_degree = (
                    "undefined"
                    if record.reference_value_radian is None
                    else f"{float(np.degrees(record.reference_value_radian)):.6f}"
                )
                lines.append(
                    f" {record.identifier:<8.8s} {record.target:<22.22s} "
                    f"{record.periodicity:6d} {record.symmetry_number:8d} "
                    f"{reference_degree:>13s} {record.coordinate_domain:<14.14s} "
                    f"{record.barrier_kcal_mol:18.6f} {record.status}"
                )
                details = [
                    f"source={','.join(record.source_coordinates) or 'not available'}",
                    f"priority_atom={record.priority_atom if record.priority_atom is not None else 'not applicable'}",
                    f"reference_status={record.reference_value_status}",
                    f"barrier_source={record.barrier_source or 'not available'}",
                ]
                lines.append(f"          {'; '.join(details)}")
        else:
            lines.append(" None")

        lines.append(f" Torsion scan plans: {len(anharmonic.torsion_scan_plans)}")
        for plan in anharmonic.torsion_scan_plans:
            phases = ",".join(
                f"{float(np.degrees(value)):.3f}" for value in plan.phase_offsets_radian
            )
            harmonics = ",".join(str(value) for value in plan.harmonics)
            lines.append(
                f" {plan.coordinate_identifier}: prescan offsets/deg={phases}; "
                f"harmonics={harmonics}; barrier seed={plan.barrier_seed_kcal_mol:.6f} kcal mol-1; "
                f"{plan.escalation_policy}"
            )
        lines.append(f" Ring-puckering surfaces: {len(anharmonic.ring_puckering_surfaces)}")

    lines.extend(
        (
            "",
            "ZOOM LEVELS",
            "-----------",
            " level basis          parameters couplings Hessian error / %",
        )
    )
    for level in field.zoom_levels:
        lines.append(
            f" {level.identifier:<5s} {level.coordinate_basis:<14.14s} "
            f"{level.parameter_count:10d} {level.coupling_count:9d} "
            f"{100.0 * level.relative_l2_eigenvalue_error:17.6f}"
        )
    return "\n".join(lines)


def _number(value, *, scientific: bool = False) -> str:
    if value is None:
        return "not available"
    return f"{float(value):.6e}" if scientific else f"{float(value):.8g}"


def _diagonal_baselines(field) -> list[dict[str, object]]:
    comparison = field.metadata.get("basis_comparison", {})
    frontiers = comparison.get("frontiers", {}) if isinstance(comparison, dict) else {}
    rows = []
    for name in ("PIC", "SONIC_LOCAL", "HYBRID_LOCAL"):
        frontier = frontiers.get(name, ()) if isinstance(frontiers, dict) else ()
        if frontier and frontier[0].get("retained_coupling_count") == 0:
            rows.append(dict(frontier[0]))
    return rows


def _read_xyz(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("empty XYZ file")
    natoms = int(lines[0].strip())
    rows = [line.split() for line in lines[2 : 2 + natoms]]
    if len(rows) != natoms or any(len(row) < 4 for row in rows):
        raise ValueError("incomplete XYZ geometry")
    atoms = tuple(row[0] for row in rows)
    coordinates = np.asarray([[float(value) for value in row[1:4]] for row in rows])
    return atoms, coordinates


def _read_explicit_charges(path: Path | None) -> tuple[float, ...] | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        values = [
            float(line.split()[0])
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        values = payload.get("charges", []) if isinstance(payload, dict) else payload
    return tuple(float(value) for value in values)


def _read_explicit_bond_orders(
    path: Path | None,
) -> tuple[tuple[int, int, float], ...] | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows = [
            line.split()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        rows = payload.get("bond_orders", []) if isinstance(payload, dict) else payload
    return tuple((int(row[0]), int(row[1]), float(row[2])) for row in rows)


if __name__ == "__main__":
    raise SystemExit(main())
