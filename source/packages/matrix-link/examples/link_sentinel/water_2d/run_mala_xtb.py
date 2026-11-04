#!/usr/bin/env python3
"""Run a short force-biased LINK--SENTINEL trajectory with xTB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from matrix_chem import preprocess_to_enriched_xyz, write_validation_section
from matrix_link import QMScanBackend
from matrix_smith import write_gicforge_build_sections
from matrix_link import active_variable_contract_from_file, run_external_driver_loop


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-workers", type=int, default=2)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    executable = shutil.which("xtb")
    if executable is None:
        raise RuntimeError("xTB is not available in PATH")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    xyzin = args.run_dir / "water.xyzin"
    preprocess_to_enriched_xyz(source / "water.xyz", xyzin)
    write_validation_section(xyzin)
    write_gicforge_build_sections(xyzin, symmetrize=False)
    contract = active_variable_contract_from_file(
        xyzin,
        source / "active_variables.json",
        pes_exploration=True,
    )
    driver = (
        f"{sys.executable} -m matrix_sentinel.cli "
        f"{{request}} {{response}} --config {source / 'mala_xtb.json'}"
    )
    result = run_external_driver_loop(
        xyzin,
        run_dir=args.run_dir / "exchange",
        driver_command=driver,
        coordinate_model=contract.model,
        active_variable_contract=contract,
        backend=QMScanBackend(
            name="xtb",
            route="--gfn 2",
            executable=executable,
        ),
        initial_properties=("energy", "gradient"),
        batch_workers=args.batch_workers,
        max_cycles=8,
        retained_group="C1",
        independent_candidates=True,
    )
    print(
        f"status={result.status} cycles={result.cycles} "
        f"points={result.point_count} completed={result.completed_point_count}"
    )
    checkpoint = json.loads(
        (
            args.run_dir
            / "exchange"
            / "sentinel_state"
            / "sampling_strategy.json"
        ).read_text(encoding="utf-8")
    )["state"]
    attempted = int(checkpoint["attempted"])
    accepted = int(checkpoint["accepted"])
    constrained = int(checkpoint.get("constraint_rejections", 0))
    print(
        f"acceptance_rate={accepted / attempted if attempted else 0.0:.3f} "
        f"constraint_rejection_rate={constrained / attempted if attempted else 0.0:.3f} "
        f"final_proposal_scale={float(checkpoint['proposal_scale']):.6f}"
    )
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
