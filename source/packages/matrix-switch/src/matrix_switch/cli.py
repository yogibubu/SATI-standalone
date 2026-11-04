"""Command-line interface for SWITCH."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

def _parse(args: argparse.Namespace) -> int:
    from .parser import parse_smiles

    graph = parse_smiles(args.smiles)
    print(json.dumps(graph.to_dict(), indent=2, sort_keys=True))
    return 0


def _name(args: argparse.Namespace) -> int:
    from .names import resolve_name

    result = resolve_name(
        args.name,
        allow_remote=not args.offline,
        cache_path=args.cache,
        timeout=args.timeout,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(result.smiles)
        print(f"source: {result.source}")
        if result.identifier:
            print(f"identifier: {result.identifier}")
    return 0


def _seed(args: argparse.Namespace) -> int:
    from .geometry import build_cartesian_seed
    from .names import resolve_name
    from .parser import parse_smiles

    if args.name is not None:
        resolution = resolve_name(
            args.name,
            allow_remote=not args.offline,
            cache_path=args.cache,
            timeout=args.timeout,
        )
        smiles = resolution.smiles
        title = args.title or args.name
    else:
        smiles = args.smiles
        title = args.title or smiles
    geometry = build_cartesian_seed(
        parse_smiles(smiles),
        title=title,
        multiplicity=args.multiplicity,
        complete_hydrogens=not args.heavy_atoms_only,
    )
    text = "\n".join(geometry.xyz_lines()) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


def _canonical(args: argparse.Namespace) -> int:
    from .canonical import canonical_smiles
    from .parser import parse_smiles

    print(canonical_smiles(parse_smiles(args.smiles)))
    return 0


def _depict(args: argparse.Namespace) -> int:
    from .depict import render_molecule_png, render_molecule_svg
    from .names import resolve_name
    from .parser import parse_smiles

    if args.name is not None:
        smiles = resolve_name(
            args.name,
            allow_remote=not args.offline,
            cache_path=args.cache,
            timeout=args.timeout,
        ).smiles
    else:
        smiles = args.smiles
    graph = parse_smiles(smiles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    suffix = args.output.suffix.lower()
    if suffix == ".svg":
        args.output.write_text(render_molecule_svg(graph), encoding="utf-8")
    elif suffix == ".png":
        args.output.write_bytes(render_molecule_png(graph))
    else:
        raise ValueError("SWITCH depiction output must end in .svg or .png")
    return 0


def _match(args: argparse.Namespace) -> int:
    from .matching import find_substructure_matches
    from .parser import parse_smiles
    from .smarts import parse_smarts

    target = parse_smiles(args.target)
    query = parse_smarts(args.query) if args.smarts else parse_smiles(args.query)
    matches = find_substructure_matches(
        target,
        query,
        use_chirality=not args.ignore_chirality,
        max_matches=args.max_matches,
    )
    print(json.dumps({"count": len(matches), "matches": matches}, indent=2))
    return 0


def _mcs(args: argparse.Namespace) -> int:
    from .matching import maximum_common_connected_subgraphs
    from .parser import parse_smiles

    matches = maximum_common_connected_subgraphs(
        parse_smiles(args.left),
        parse_smiles(args.right),
        minimum_atoms=args.minimum_atoms,
        timeout_seconds=args.timeout,
        max_matches=args.max_matches,
    )
    print(
        json.dumps(
            {
                "maximum_atoms": max(
                    (match.atom_count for match in matches),
                    default=0,
                ),
                "matches": [
                    {
                        "left": match.source_atoms,
                        "right": match.target_atoms,
                    }
                    for match in matches
                ],
            },
            indent=2,
        )
    )
    return 0


def _stereoisomers(args: argparse.Namespace) -> int:
    from .canonical import canonical_smiles
    from .parser import parse_smiles
    from .stereoisomers import enumerate_stereoisomers

    isomers = enumerate_stereoisomers(
        parse_smiles(args.smiles),
        max_isomers=args.max_isomers,
    )
    for isomer in isomers:
        print(canonical_smiles(isomer))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="matrix-switch",
        description="Internal SMILES parsing, common-name resolution, and Cartesian seeding.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    parse = commands.add_parser("parse", help="parse SMILES into the SWITCH graph contract")
    parse.add_argument("smiles")
    parse.set_defaults(handler=_parse)

    name = commands.add_parser("name", help="translate a common chemical name into SMILES")
    name.add_argument("name")
    name.add_argument("--offline", action="store_true")
    name.add_argument("--cache", type=Path)
    name.add_argument("--timeout", type=float, default=10.0)
    name.add_argument("--json", action="store_true")
    name.set_defaults(handler=_name)

    seed = commands.add_parser("seed", help="construct deterministic Cartesian coordinates")
    source = seed.add_mutually_exclusive_group(required=True)
    source.add_argument("--smiles")
    source.add_argument("--name")
    seed.add_argument("--title")
    seed.add_argument("--multiplicity", type=int)
    seed.add_argument("--heavy-atoms-only", action="store_true")
    seed.add_argument("--offline", action="store_true")
    seed.add_argument("--cache", type=Path)
    seed.add_argument("--timeout", type=float, default=10.0)
    seed.add_argument("-o", "--output", type=Path)
    seed.set_defaults(handler=_seed)

    canonical = commands.add_parser(
        "canonical",
        help="write deterministic canonical isomeric SMILES",
    )
    canonical.add_argument("smiles")
    canonical.set_defaults(handler=_canonical)

    depict = commands.add_parser(
        "depict",
        help="render an internal 2D SVG or PNG depiction",
    )
    depict_source = depict.add_mutually_exclusive_group(required=True)
    depict_source.add_argument("--smiles")
    depict_source.add_argument("--name")
    depict.add_argument("--offline", action="store_true")
    depict.add_argument("--cache", type=Path)
    depict.add_argument("--timeout", type=float, default=10.0)
    depict.add_argument("-o", "--output", type=Path, required=True)
    depict.set_defaults(handler=_depict)

    match = commands.add_parser("match", help="find a SMILES/SMARTS substructure")
    match.add_argument("target")
    match.add_argument("query")
    match.add_argument("--smarts", action="store_true")
    match.add_argument("--ignore-chirality", action="store_true")
    match.add_argument("--max-matches", type=int, default=10000)
    match.set_defaults(handler=_match)

    mcs = commands.add_parser("mcs", help="find maximum connected common subgraphs")
    mcs.add_argument("left")
    mcs.add_argument("right")
    mcs.add_argument("--minimum-atoms", type=int, default=1)
    mcs.add_argument("--timeout", type=float, default=5.0)
    mcs.add_argument("--max-matches", type=int, default=256)
    mcs.set_defaults(handler=_mcs)

    stereoisomers = commands.add_parser(
        "stereoisomers",
        help="enumerate unspecified tetrahedral and E/Z stereochemistry",
    )
    stereoisomers.add_argument("smiles")
    stereoisomers.add_argument("--max-isomers", type=int, default=256)
    stereoisomers.set_defaults(handler=_stereoisomers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
