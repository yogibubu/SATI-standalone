#!/usr/bin/env python3
"""Build and run the complete two-SONIC water protocol example."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from matrix_chem import preprocess_to_enriched_xyz, write_validation_section
from matrix_core import (
    CALCULATION_AUTHORIZATION_BUNDLE_ENV,
    CalculationResources,
    authorize_calculation_launch,
    build_calculation_launch_plan,
    write_calculation_authorization_bundle,
)
from matrix_smith import write_gicforge_build_sections
from matrix_link import active_variable_contract_from_file, run_external_driver_loop


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    repository = source.parents[4]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    authorization_input = args.run_dir / "authorized-link-sentinel-parent.inp"
    authorization_input.write_text(
        "deterministic LINK-SENTINEL example; no external quantum chemistry\n",
        encoding="utf-8",
    )
    authorization_plan = build_calculation_launch_plan(
        backend="TRINITY/link-sentinel-example",
        input_path=authorization_input,
        command=(sys.executable, str(Path(__file__).resolve()), "--run-dir", str(args.run_dir)),
        workdir=args.run_dir,
        resources=CalculationResources(
            process_count=1,
            threads_per_process=4,
            memory_per_job_gb=2.0,
            concurrent_jobs=4,
        ),
    )
    authorization = authorize_calculation_launch(
        authorization_plan,
        approved_plan_sha256=authorization_plan.sha256,
        authorized_by="LINK-SENTINEL-example",
    )
    authorization_bundle = write_calculation_authorization_bundle(
        args.run_dir / "calculation-authorization.json",
        authorization_plan,
        authorization,
    )
    os.environ[CALCULATION_AUTHORIZATION_BUNDLE_ENV] = str(authorization_bundle)
    xyzin = args.run_dir / "water.xyzin"
    preprocess_to_enriched_xyz(source / "water.xyz", xyzin)
    write_validation_section(xyzin)
    write_gicforge_build_sections(
        xyzin,
        symmetrize=False,
        fragment_context="exploration",
    )
    contract = active_variable_contract_from_file(
        xyzin,
        source / "active_variables.json",
        pes_exploration=True,
    )
    sentinel = repository / "tools" / "link_sentinel_v1" / "mock_sentinel.py"
    evaluator = source / "analytic_pes.py"
    result = run_external_driver_loop(
        xyzin,
        run_dir=args.run_dir / "exchange",
        driver_command=(
            f"{sys.executable} {sentinel} {{request}} {{response}} --mode scan-2d"
        ),
        coordinate_model=contract.model,
        active_variable_contract=contract,
        engine_command=(
            f"{sys.executable} {evaluator} {{xyz}} {{result}} {{index}}"
        ),
        initial_properties=("energy", "gradient", "hessian"),
        batch_workers=4,
    )
    print(f"status={result.status} cycles={result.cycles} points={result.point_count}")
    return 0 if result.status == "complete" and result.point_count == 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
