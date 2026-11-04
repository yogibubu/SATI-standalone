#!/usr/bin/env python3
"""Run derivative-free blocked replica-exchange Monte Carlo with xTB."""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    repository = source.parents[4]
    config = args.config or source / "mc_xtb.json"
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    geometry_filter = config_payload.get("geometry_filter")
    if geometry_filter is not None and not isinstance(geometry_filter, dict):
        raise ValueError("geometry_filter must be an object")
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
    sentinel_source = repository / "packages" / "matrix-sentinel" / "src"
    os.environ["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(sentinel_source), os.environ.get("PYTHONPATH", ""))
        if part
    )
    driver = (
        f"{sys.executable} -m matrix_sentinel.cli "
        f"{{request}} {{response}} --config {config.resolve()}"
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
            env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        ),
        initial_properties=("energy",),
        batch_workers=args.batch_workers,
        max_cycles=10,
        retained_group="C1",
        independent_candidates=True,
        geometry_filter=geometry_filter,
    )
    checkpoint = json.loads(
        (
            args.run_dir / "exchange" / "sentinel_state" / "sampling_strategy.json"
        ).read_text(encoding="utf-8")
    )["state"]
    attempted = int(checkpoint["attempted"])
    accepted = int(checkpoint["accepted"])
    print(
        f"status={result.status} cycles={result.cycles} points={result.point_count} "
        f"completed={result.completed_point_count} "
        f"acceptance_rate={accepted / attempted if attempted else 0.0:.3f}"
    )
    for name, scale, block_attempted, block_accepted in zip(
        ("oh-stretch", "hoh-bend"),
        checkpoint["block_scales"],
        checkpoint["block_attempted"],
        checkpoint["block_accepted"],
        strict=True,
    ):
        rate = block_accepted / block_attempted if block_attempted else 0.0
        print(f"block={name} scale={float(scale):.6g} acceptance={rate:.3f}")
    print(
        "swap_acceptance="
        + ",".join(
            f"{accepted_swap / attempted_swap if attempted_swap else 0.0:.3f}"
            for attempted_swap, accepted_swap in zip(
                checkpoint["swap_attempted"],
                checkpoint["swap_accepted"],
                strict=True,
            )
        )
    )
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
