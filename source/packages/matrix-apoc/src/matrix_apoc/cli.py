"""Command-line interface for APOC."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from matrix_qm import read_qm_population_section

from .service import (
    analyze_gaussian,
    analyze_gaussian_fchk,
    analyze_molden,
    analyze_orca,
    analyze_pyscf_output,
    analyze_source,
    attach_apoc_analysis,
    load_apoc_analysis,
    save_apoc_analysis,
)
from .state import (
    cross_ao_overlap_from_molden,
    electronic_state_from_molden,
    match_excited_states,
    natural_orbitals,
    recanonicalize_orbitals,
)
from .trexio_adapter import read_electronic_state_trexio, write_electronic_state_trexio


def _parser(prog: str = "apoc") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="APOC — Atomic and Pairwise Observables from the Charge density",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("gaussian", "Read Hirshfeld/CM5 and Mayer data from Gaussian/GDV output"),
        ("pyscf-output", "Read APOC observables from structured PySCF output"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("source", type=Path)
        command.add_argument("--output", type=Path)
        command.add_argument("--xyzin", type=Path)
    gaussian_fchk = sub.add_parser(
        "gaussian-fchk",
        help="Compute CM5/Mayer from a Gaussian FCHK wavefunction",
    )
    gaussian_fchk.add_argument("source", type=Path)
    gaussian_fchk.add_argument("--output", type=Path)
    gaussian_fchk.add_argument("--xyzin", type=Path)
    gaussian_fchk.add_argument("--grid-level", type=int, default=4)
    gaussian_fchk.add_argument("--charge", type=int)
    orca = sub.add_parser(
        "orca",
        help="Convert an ORCA GBW to Molden and compute the common APOC observables",
    )
    orca.add_argument("source", type=Path)
    orca.add_argument("--output", type=Path)
    orca.add_argument("--xyzin", type=Path)
    orca.add_argument("--molden-output", type=Path)
    orca.add_argument("--converter", default="orca_2mkl")
    orca.add_argument("--timeout", type=float, default=120.0)
    orca.add_argument("--grid-level", type=int, default=4)
    orca.add_argument("--charge", type=int)
    molden = sub.add_parser(
        "molden",
        help="Compute Hirshfeld/CM5 and Mayer from a complete Molden orbital export",
    )
    molden.add_argument("source", type=Path)
    molden.add_argument("--output", type=Path)
    molden.add_argument("--xyzin", type=Path)
    molden.add_argument("--grid-level", type=int, default=4)
    molden.add_argument("--charge", type=int)
    inspect = sub.add_parser("inspect", help="Print an APOC JSON or #QM_POPULATION report")
    inspect.add_argument("source", type=Path)
    analyze = sub.add_parser("analyze", help="Auto-detect and analyze one QM source")
    analyze.add_argument("source", type=Path)
    analyze.add_argument(
        "--format",
        choices=(
            "auto",
            "gaussian",
            "gaussian-fchk",
            "orca",
            "molden",
            "pyscf-output",
        ),
        default="auto",
        dest="source_format",
    )
    analyze.add_argument("--output", type=Path)
    analyze.add_argument("--xyzin", type=Path)
    analyze.add_argument("--grid-level", type=int, default=4)
    analyze.add_argument("--charge", type=int)
    state = sub.add_parser(
        "state",
        help="Create, transform and compare backend-independent electronic states",
    )
    state_sub = state.add_subparsers(dest="state_command", required=True)
    import_molden = state_sub.add_parser(
        "import-molden",
        help="Convert a complete Molden wavefunction to the native APOC/TREXIO contract",
    )
    import_molden.add_argument("source", type=Path)
    import_molden.add_argument("output", type=Path)
    import_molden.add_argument("--text", action="store_true", help="Use TREXIO text backend")
    import_molden.add_argument("--overwrite", action="store_true")
    inspect_state = state_sub.add_parser("inspect", help="Summarize an APOC/TREXIO state")
    inspect_state.add_argument("source", type=Path)
    orbitals = state_sub.add_parser(
        "natural-orbitals",
        help="Build natural orbitals and optionally recanonicalize equal-occupation spaces",
    )
    orbitals.add_argument("source", type=Path)
    orbitals.add_argument("output", type=Path)
    orbitals.add_argument("--recanonicalize", action="store_true")
    orbitals.add_argument("--occupation-tolerance", type=float, default=1.0e-8)
    orbitals.add_argument("--text", action="store_true")
    orbitals.add_argument("--overwrite", action="store_true")
    track = state_sub.add_parser(
        "track",
        help="Match TD states by transition-density overlap rather than energy order",
    )
    track.add_argument("reference", type=Path)
    track.add_argument("candidate", type=Path)
    track.add_argument(
        "--cross-overlap",
        type=Path,
        help="NumPy .npy cross-AO overlap between the two geometries",
    )
    track.add_argument("--reference-molden", type=Path)
    track.add_argument("--candidate-molden", type=Path)
    track.add_argument("--minimum-overlap", type=float, default=0.70)
    track.add_argument("--ambiguity-margin", type=float, default=0.05)
    track.add_argument(
        "--strict",
        action="store_true",
        help="Return a nonzero status when continuity fails or the assignment is ambiguous",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, prog: str = "apoc") -> int:
    args = _parser(prog).parse_args(argv)
    if args.command == "state":
        return _state_command(args)
    if args.command == "inspect":
        if args.source.suffix.lower() == ".json":
            print("\n".join(load_apoc_analysis(args.source).report_lines()))
        else:
            observables, source = read_qm_population_section(args.source)
            print(f"APOC source: {source}")
            print("charge model: CM5")
            print("bond-order model: Mayer")
            print(f"atoms: {observables.natoms}")
            print(f"electrons: {observables.electron_count:.8f}")
            print(f"molecular charge: {observables.charge:.8f}")
        return 0
    if args.command == "analyze":
        analysis = analyze_source(
            args.source,
            source_format=args.source_format,
            grid_level=args.grid_level,
            charge=args.charge,
        )
    elif args.command == "gaussian":
        analysis = analyze_gaussian(args.source)
    elif args.command == "gaussian-fchk":
        analysis = analyze_gaussian_fchk(
            args.source,
            grid_level=args.grid_level,
            charge=args.charge,
        )
    elif args.command == "orca":
        analysis = analyze_orca(
            args.source,
            molden_output=args.molden_output,
            converter_executable=args.converter,
            timeout=args.timeout,
            grid_level=args.grid_level,
            charge=args.charge,
        )
    elif args.command == "pyscf-output":
        analysis = analyze_pyscf_output(args.source)
    else:
        analysis = analyze_molden(
            args.source,
            grid_level=args.grid_level,
            charge=args.charge,
        )
    output = args.output or args.source.with_suffix(".apoc.json")
    print(save_apoc_analysis(output, analysis))
    if args.xyzin is not None:
        print(attach_apoc_analysis(args.xyzin, analysis))
    return 0


def _state_command(args: argparse.Namespace) -> int:
    import numpy as np

    if args.state_command == "import-molden":
        state = electronic_state_from_molden(args.source)
        print(
            write_electronic_state_trexio(
                args.output,
                state,
                text_backend=args.text,
                overwrite=args.overwrite,
            )
        )
        return 0
    if args.state_command == "inspect":
        state = read_electronic_state_trexio(args.source)
        payload = {
            "schema": state.schema,
            "source": str(args.source),
            "atoms": state.natoms,
            "ao_count": state.nao,
            "spin_channels": state.spin_channels,
            "electron_count": state.electron_count,
            "charge": state.charge,
            "multiplicity": state.multiplicity,
            "method": state.method,
            "basis": state.basis_label,
            "excited_states": [
                {
                    "state_id": item.state_id,
                    "label": item.label,
                    "symmetry": item.symmetry,
                    "energy_hartree": item.energy_hartree,
                    "oscillator_strength": item.oscillator_strength,
                }
                for item in state.excited_states
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.state_command == "natural-orbitals":
        state = read_electronic_state_trexio(args.source)
        coefficients = []
        occupations = []
        energies = []
        for channel in range(state.spin_channels):
            transformed = natural_orbitals(state.density_ao(channel), state.overlap_ao)
            if args.recanonicalize:
                transformed = recanonicalize_orbitals(
                    transformed.coefficients,
                    state.fock_ao(channel),
                    state.overlap_ao,
                    occupations=transformed.occupations,
                    occupation_tolerance=args.occupation_tolerance,
                )
                energies.append(np.asarray(transformed.energies_hartree, dtype=float))
            coefficients.append(transformed.coefficients)
            occupations.append(transformed.occupations)
        output_state = replace(
            state,
            mo_coefficients=tuple(coefficients),
            mo_occupations=tuple(occupations),
            mo_energies_hartree=tuple(energies),
            source=f"natural orbitals from {args.source}",
        )
        print(
            write_electronic_state_trexio(
                args.output,
                output_state,
                text_backend=args.text,
                overwrite=args.overwrite,
            )
        )
        return 0
    reference = read_electronic_state_trexio(args.reference)
    candidate = read_electronic_state_trexio(args.candidate)
    if args.cross_overlap is not None:
        cross = np.asarray(np.load(args.cross_overlap), dtype=float)
    elif args.reference_molden is not None and args.candidate_molden is not None:
        cross = cross_ao_overlap_from_molden(
            args.reference_molden,
            args.candidate_molden,
        )
    elif args.reference_molden is not None or args.candidate_molden is not None:
        raise ValueError(
            "cross-overlap evaluation requires both --reference-molden and "
            "--candidate-molden"
        )
    elif (
        reference.nao == candidate.nao
        and np.allclose(reference.coordinates_bohr, candidate.coordinates_bohr)
        and np.array_equal(reference.atomic_numbers, candidate.atomic_numbers)
    ):
        cross = reference.overlap_ao
    else:
        raise ValueError(
            "state tracking at different geometries requires --cross-overlap; "
            "the producing backend or APOC basis engine must evaluate S_AB"
        )
    matches = match_excited_states(
        reference.excited_states,
        candidate.excited_states,
        cross,
        overlap_reference=reference.overlap_ao,
        overlap_candidate=candidate.overlap_ao,
        minimum_overlap=args.minimum_overlap,
        ambiguity_margin=args.ambiguity_margin,
    )
    print(
        json.dumps(
            [
                {
                    "reference_state_id": item.reference_state_id,
                    "candidate_state_id": item.candidate_state_id,
                    "overlap": item.overlap,
                    "phase": item.phase,
                    "energy_difference_hartree": item.energy_difference_hartree,
                    "runner_up_overlap": item.runner_up_overlap,
                    "margin": item.margin,
                    "continuous": item.continuous,
                    "ambiguous": item.ambiguous,
                }
                for item in matches
            ],
            indent=2,
            sort_keys=True,
        )
    )
    if args.strict and any(not item.continuous or item.ambiguous for item in matches):
        return 3
    return 0


def matrix_main(argv: Sequence[str] | None = None) -> int:
    return main(argv, prog="matrix apoc")
