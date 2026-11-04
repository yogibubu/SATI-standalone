from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from matrix_qm import hessian_input_from_engine as _canonical_hessian_input_from_engine

from .internal import gf_from_cartesian_hessian_and_matrix_gics
from .large_amplitude import large_amplitude_analysis_from_gf_matrices


def hessian_input_from_engine(engine: str, path: Path):
    """Read a Cartesian Hessian from an external electronic-structure engine."""
    return _canonical_hessian_input_from_engine(
        engine,
        path,
        grd=path / "GRD" if engine == "cfour-files" else None,
        output=path / "cfour_freq.out" if engine == "cfour-files" else None,
    )


def summarize_hessian(label: str, engine: str, path: Path) -> dict[str, Any]:
    """Summarize an external Hessian with the standard TRINITY/SONIC workflow."""
    data = hessian_input_from_engine(engine, path)
    gf = gf_from_cartesian_hessian_and_matrix_gics(
        data.cartesian_hessian,
        data.cartesian_coordinates_bohr,
        data.atomic_numbers,
        data.masses_amu,
    )
    families = tuple(gf.gic_families)
    active_torsions = sum(1 for item in families if item == "torsion")
    large_blocks = tuple(block.label for block in gf.large_amplitude.blocks) if gf.large_amplitude else ()
    large_frequencies = (
        {
            block.label: [float(value) for value in block.frequencies_cm]
            for block in gf.large_amplitude.blocks
        }
        if gf.large_amplitude
        else {}
    )
    dvr_plan = (
        [
            {
                "identifier": f"GIC{item.index:03d}",
                "name": item.name,
                "family": item.family,
                "status": item.status,
                "central_bond": list(item.central_bond) if item.central_bond is not None else None,
                "periodicity": item.periodicity,
                "rotor_symmetry_number": item.rotor_symmetry_number,
                "rotor_multiplicity": item.rotor_multiplicity,
                "hindered_rotor_status": item.hindered_rotor_status,
                "barrier_cm": item.barrier_cm,
                "barrier_kcal_mol": item.barrier_kcal_mol,
                "reason": item.reason,
            }
            for item in gf.large_amplitude.dvr_candidates
        ]
        if gf.large_amplitude
        else []
    )
    torsion_frequency_models: dict[str, list[float]] = {}
    for model in ("full", "projected", "diagonal-g", "separate"):
        analysis = large_amplitude_analysis_from_gf_matrices(
            force_constants=gf.force_constants,
            g_matrix=gf.g_matrix,
            frequencies_cm=gf.frequencies_cm,
            ped=gf.ped.values,
            gic_labels=gf.gic_labels,
            gic_names=gf.gic_names,
            gic_irreps=gf.gic_irreps,
            families=("torsion",),
            frequency_cutoff_cm=None,
            block_frequency_model=model,
        )
        torsion = next((block for block in analysis.blocks if block.label == "torsion"), None)
        if torsion is not None:
            torsion_frequency_models[model] = [float(value) for value in torsion.frequencies_cm]
    return {
        "label": label,
        "engine": engine,
        "source": str(path),
        "atoms": int(len(data.atomic_numbers)),
        "cartesian_hessian_shape": list(data.cartesian_hessian.shape),
        "harmonic_frequencies": int(len(data.harmonic_frequencies_cm)),
        "sonic_coordinates": int(gf.force_constants.shape[0]),
        "point_group": gf.point_group,
        "active_torsion_rows": int(active_torsions),
        "large_amplitude_blocks": list(large_blocks),
        "large_amplitude_frequencies_cm": large_frequencies,
        "large_amplitude_dvr_plan": dvr_plan,
        "torsion_frequency_models_cm": torsion_frequency_models,
        "lowest_gf_frequencies_cm": [float(value) for value in gf.frequencies_cm[: min(8, len(gf.frequencies_cm))]],
    }


def write_benchmark_summary(output: Path, entries: list[tuple[str, str, Path]]) -> None:
    rows = [summarize_hessian(label, engine, path) for label, engine, path in entries]
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 1:
        raise SystemExit(
            "usage: python -m matrix_gf.benchmark_summary OUT.json "
            "[label engine path]..."
        )
    output = Path(args[0])
    triples = args[1:]
    if len(triples) % 3:
        raise SystemExit("benchmark entries must be label/engine/path triples")
    entries = [
        (triples[idx], triples[idx + 1], Path(triples[idx + 2]))
        for idx in range(0, len(triples), 3)
    ]
    write_benchmark_summary(output, entries)
    print(f"wrote {output} with {len(entries)} benchmark summaries")


if __name__ == "__main__":
    main()
