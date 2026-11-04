"""Gaussian-like public input deck for LINK geometry optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import json
from typing import Sequence

import numpy as np

from matrix_core.manifest import matrix_version

from .optimizer import OptimizerResult, OptimizerSettings, optimize_geometry
from .scan import QMBackendTerm, QMScanBackend


@dataclass(frozen=True)
class OptimizationInputDeck:
    path: Path
    title: str
    atoms: tuple[str, ...]
    coordinates_angstrom: np.ndarray | None
    smiles: str
    charge: int
    multiplicity: int
    backend: str
    executable: str | None
    model: str
    route_keywords: tuple[str, ...]
    coordinate_kind: str
    orientation_weights: str
    max_steps: int
    trust_radius: float
    max_trust_radius: float
    include_cv_exponential_field: bool
    processors: int = 1
    memory_gb: int | None = None
    fd_parallel_workers: int = 1
    composite_parallel_workers: int = 1
    initial_hessian_path: Path | None = None
    ellipsoidal_cpcm_path: Path | None = None
    force_field_path: Path | None = None


def read_optimization_input(path: Path | str) -> OptimizationInputDeck:
    """Read a Gaussian-shaped MATRIX optimization input.

    Supported MATRIX link-zero keys are ``%Backend``, ``%Executable``,
    ``%MaxSteps``, ``%TrustRadius``, ``%MaxTrustRadius`` and
    ``%EllipsoidalCPCM``.  The latter points to a JSON cavity/charge
    specification and transparently turns the requested vacuum backend into
    a ``linear_composite`` surface (vacuum + analytic ellipsoidal ASC CPCM).
    The molecular specification is either ordinary Cartesian rows or one
    ``SMILES ...`` row.
    """

    target = Path(path)
    lines = target.read_text(encoding="utf-8").splitlines()
    options: dict[str, str] = {}
    route_parts: list[str] = []
    cursor = 0
    while cursor < len(lines) and lines[cursor].lstrip().startswith("%"):
        key, _, value = lines[cursor].lstrip()[1:].partition("=")
        options[key.strip().lower()] = value.strip()
        cursor += 1
    while cursor < len(lines) and not lines[cursor].lstrip().startswith("#"):
        cursor += 1
    if cursor >= len(lines):
        raise ValueError("MATRIX optimization input needs a # route line")
    while cursor < len(lines):
        text = lines[cursor].strip()
        if not text:
            break
        route_parts.append(text)
        cursor += 1
    route = " ".join(route_parts)
    cursor += 1
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    title = lines[cursor].strip() if cursor < len(lines) else target.stem
    cursor += 1
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    if cursor >= len(lines):
        raise ValueError("MATRIX optimization input lacks charge/multiplicity")
    charge_mult = lines[cursor].split()
    if len(charge_mult) != 2:
        raise ValueError("charge/multiplicity line must contain two integers")
    charge, multiplicity = int(charge_mult[0]), int(charge_mult[1])
    cursor += 1
    atoms: list[str] = []
    coordinates: list[list[float]] = []
    smiles = ""
    while cursor < len(lines):
        text = lines[cursor].strip()
        cursor += 1
        if not text:
            break
        if text.upper().startswith("SMILES "):
            smiles = text.split(None, 1)[1].strip()
            continue
        parts = text.replace(",", " ").split()
        if len(parts) != 4:
            raise ValueError(f"invalid Cartesian row: {text}")
        atoms.append(parts[0])
        coordinates.append([float(value) for value in parts[1:]])
    if bool(smiles) == bool(atoms):
        raise ValueError("provide exactly one of Cartesian coordinates or SMILES")
    route_tokens = _route_tokens(route)
    backend = options.get("backend", _route_value(route_tokens, "backend") or "gaussian")
    model = next(
        (token for token in route_tokens if "/" in token),
        "L2-MP2"
        if str(backend).strip().lower() in {
            "l2-mp2-g16",
            "junchs-gdv",
            "junchs-cfour",
            "junchs-molpro",
        }
        else "FPA-Q5/T+CV"
        if str(backend).strip().lower() == "fpa-q5-t-cv-molpro"
        else "FPA-haQ5/haT+CV"
        if str(backend).strip().lower() == "fpa-haq5-hat-cv-molpro"
        else "SHERRILL-TQ/DZ+CV"
        if str(backend).strip().lower() == "sherrill-tq-dz-cv-molpro"
        else "L2FPA_lite"
        if str(backend).strip().lower() == "l2fpa-lite-molpro"
        else "L2FPA"
        if str(backend).strip().lower() == "l2fpa-molpro"
        else "cheap_lite"
        if str(backend).strip().lower() == "cheap-lite-molpro"
        else "L2"
        if str(backend).strip().lower() == "l2-molpro"
        else "PL2-MP2"
        if str(backend).strip().lower() == "pl2-mp2-orca"
        else "PL2"
        if str(backend).strip().lower() == "pl2-orca"
        else "FL2"
        if str(backend).strip().lower() == "fast-molpro"
        else "PL2_lite-F12"
        if str(backend).strip().lower() == "pl2-lite-f12-molpro"
        else "PL2_lite-MP2(TZ/QZ)"
        if str(backend).strip().lower() == "pl2-lite-mp2-molpro"
        else "PL2_lite-DB"
        if str(backend).strip().lower() == "pl2-lite-db-molpro"
        else "L2_lite-DB"
        if str(backend).strip().lower() == "l2-lite-db-molpro"
        else "L2_lite-F12"
        if str(backend).strip().lower() == "l2-lite-f12-molpro"
        else "L2F12_lite"
        if str(backend).strip().lower() == "l2f12-lite-hybrid-molpro"
        else "PL2F12_lite"
        if str(backend).strip().lower() in {"pl2f12-lite-hybrid-molpro", "pl2f12-may-lite-molpro"}
        else "PFPA_lite"
        if str(backend).strip().lower() == "pfpa-lite-molpro"
        else "L2-F12"
        if str(backend).strip().lower() in {
            "pcs2-orca", "pcs2-mrcc", "pcs2-molpro", "l2f12-may-molpro", "l2f12-may-lite-molpro"
        }
        else "PL2-F12"
        if str(backend).strip().lower() in {
            "ppcs2-orca",
            "ppcs2-mrcc",
            "ppcs2-molpro",
            "pl2-f12-orca",
            "pl2-f12-molpro",
        }
        else "HF/STO-3G",
    )
    coordinate_kind = (options.get("coordinatekind") or _opt_value(route) or "SONIC").lower()
    if coordinate_kind not in {
        "sonic",
        "cartesian",
        "sycart",
        "simple-redundant",
        "simple-delocalized",
        "hessian-modes",
    }:
        coordinate_kind = "sonic"
    orientation_weights = options.get("orientation", "mass").lower().replace("_", "-")
    aliases = {
        "masses": "mass",
        "atomic-mass": "mass",
        "z": "atomic-number",
        "number": "atomic-number",
    }
    orientation_weights = aliases.get(orientation_weights, orientation_weights)
    if orientation_weights not in {"mass", "atomic-number"}:
        raise ValueError("%Orientation must be mass or atomic-number")
    keywords = tuple(
        token
        for token in route_tokens
        if token.upper() != "MATRIX"
        and token != model
        and not token.lower().startswith(("backend=", "opt=", "cv="))
    )
    return OptimizationInputDeck(
        path=target,
        title=title,
        atoms=tuple(atoms),
        coordinates_angstrom=None if not coordinates else np.asarray(coordinates, dtype=float),
        smiles=smiles,
        charge=charge,
        multiplicity=multiplicity,
        backend=backend.lower(),
        executable=options.get("executable") or None,
        model=model,
        route_keywords=keywords,
        coordinate_kind=coordinate_kind,
        orientation_weights=orientation_weights,
        max_steps=int(options.get("maxsteps", 100)),
        trust_radius=float(options.get("trustradius", 0.2)),
        max_trust_radius=float(options.get("maxtrustradius", 0.5)),
        include_cv_exponential_field=(
            str(backend).strip().lower()
            in {
                "fast-molpro",
                "pl2-lite-f12-molpro",
                "pl2-lite-mp2-molpro",
                "pl2-lite-db-molpro",
                "l2-lite-db-molpro",
                "l2-lite-f12-molpro",
                "l2f12-lite-hybrid-molpro",
                "pl2f12-lite-hybrid-molpro",
                "pl2f12-may-lite-molpro",
                "pfpa-lite-molpro",
            }
            or options.get("corevalence", "").strip().lower()
            in {"exponential", "cv-radial", "on", "true"}
            or (_route_value(route_tokens, "cv").strip().lower() in {"exponential", "cv-radial"})
        ),
        processors=int(options.get("processors", 1)),
        memory_gb=(int(options["memorygb"]) if options.get("memorygb") else None),
        fd_parallel_workers=int(options.get("fdworkers", 1)),
        composite_parallel_workers=int(options.get("compositeworkers", 1)),
        initial_hessian_path=(
            Path(options["initialhessian"]).expanduser()
            if options.get("initialhessian")
            else None
        ),
        ellipsoidal_cpcm_path=(
            Path(options["ellipsoidalcpcm"]).expanduser()
            if options.get("ellipsoidalcpcm")
            else None
        ),
        force_field_path=(
            Path(options["forcefield"]).expanduser()
            if options.get("forcefield") else None
        ),
    )


def run_optimization_input(
    path: Path | str,
    *,
    run_dir: Path | str,
) -> OptimizerResult:
    """Translate a public input deck into MATRIX state and run LINK."""

    deck = read_optimization_input(path)
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    xyz = root / "input.xyz"
    if deck.smiles:
        from .smiles import smiles_to_geometry

        geometry = smiles_to_geometry(
            deck.smiles,
            title=deck.title,
            charge=deck.charge,
            multiplicity=deck.multiplicity,
        )
        atoms = geometry.atoms
        coordinates = geometry.coordinates_angstrom
    else:
        assert deck.coordinates_angstrom is not None
        atoms = deck.atoms
        coordinates = deck.coordinates_angstrom
    from matrix_chem import principal_axis_orientation

    # A single ellipsoidal boundary and its finite cluster share one body-fixed
    # frame: mass centre at the origin and axes along the principal moments.
    # This same canonicalization is enforced after every proposed geometry
    # below, so the cluster can neither translate nor rotate relative to the
    # fixed ellipsoid.
    coordinates = principal_axis_orientation(
        coordinates,
        _orientation_weights(atoms, deck.orientation_weights),
    )
    _write_xyz(xyz, atoms, coordinates, deck.title)
    xyzin = root / "input.xyzin"
    from matrix_chem import preprocess_to_enriched_xyz, write_validation_section

    preprocess_to_enriched_xyz(xyz, xyzin)
    fragmented = _topology_component_count(xyzin) > 1
    if fragmented:
        from matrix_fragments import write_fragment_build_section

        write_fragment_build_section(xyzin)
    validation = write_validation_section(xyzin, require_fragments=fragmented)
    if validation.status != "PASS":
        raise ValueError("generated MATRIX input did not pass topology validation")
    if deck.coordinate_kind == "sycart":
        from matrix_smith import write_sycart_build_section

        write_sycart_build_section(xyzin)
    elif deck.coordinate_kind != "cartesian":
        from matrix_smith import write_gicforge_build_sections

        write_gicforge_build_sections(
            xyzin, symmetrize=True, out_of_plane_model="soop"
        )
    if deck.backend in {
        "gfn2_zaff_composite", "gfn2-zaff", "gfn2_zaff",
        "gfn2_zaff_composite_two_morse_energy_xh",
    }:
        if deck.force_field_path is None:
            raise ValueError("GFN2+Delta(ZAFF) requires %ForceField=<fitted field JSON>")
        backend = QMScanBackend(
            name=(
                deck.backend if deck.backend == "gfn2_zaff_composite_two_morse_energy_xh"
                else "gfn2_zaff_composite"
            ), charge=deck.charge,
            multiplicity=deck.multiplicity, force_field=deck.force_field_path,
            properties=("energy", "gradient"),
        )
    elif deck.backend in {
        "junchs-gdv",
        "l2-mp2-g16",
        "junchs-cfour",
        "junchs-molpro",
        "fpa-q5-t-cv-molpro",
        "fpa-haq5-hat-cv-molpro",
        "sherrill-tq-dz-cv-molpro",
        "l2-molpro",
        "pl2-orca",
        "pl2-mp2-orca",
        "pl2-f12-orca",
        "pl2-f12-molpro",
        "fast-molpro",
        "pl2-lite-f12-molpro",
        "pl2-lite-mp2-molpro",
        "pl2-lite-db-molpro",
        "l2-lite-db-molpro",
        "l2-lite-f12-molpro",
        "l2f12-lite-hybrid-molpro",
        "pl2f12-lite-hybrid-molpro",
        "pl2f12-may-lite-molpro",
        "pfpa-lite-molpro",
        "pcs2-orca",
        "pcs2-mrcc",
        "ppcs2-orca",
        "ppcs2-mrcc",
        "pcs2-molpro",
        "l2f12-may-molpro",
        "l2f12-may-lite-molpro",
        "ppcs2-molpro",
    }:
        if deck.backend.endswith("-mrcc"):
            from .ppcs2 import pcs_mrcc_backend, ppcs2_mrcc_backend

            factory = ppcs2_mrcc_backend if deck.backend == "ppcs2-mrcc" else pcs_mrcc_backend
            backend = factory(
                charge=deck.charge,
                multiplicity=deck.multiplicity,
                executable=deck.executable,
                processors=deck.processors,
                memory_gb=deck.memory_gb,
                composite_parallel_workers=deck.composite_parallel_workers,
            )
        else:
            from .composite_protocol import composite_protocol_backend

            backend = composite_protocol_backend(
                deck.backend,
                charge=deck.charge,
                multiplicity=deck.multiplicity,
                executable=deck.executable,
                processors=deck.processors,
                memory_gb=deck.memory_gb,
                composite_parallel_workers=deck.composite_parallel_workers,
            )
    else:
        method, basis = _method_basis(deck.model)
        vacuum_backend = QMScanBackend(
            name=deck.backend,
            route=_backend_route(deck),
            method=method,
            basis=basis,
            charge=deck.charge,
            multiplicity=deck.multiplicity,
            executable=deck.executable,
            processors=deck.processors,
            memory_gb=deck.memory_gb,
        )
        if deck.ellipsoidal_cpcm_path is not None:
            if not deck.ellipsoidal_cpcm_path.is_file():
                raise FileNotFoundError(
                    f"ellipsoidal CPCM configuration not found: {deck.ellipsoidal_cpcm_path}"
                )
            cpcm_backend = QMScanBackend(
                name="ellipsoidal_cpcm",
                resolution={"configuration": str(deck.ellipsoidal_cpcm_path)},
                properties=("energy", "gradient"),
            )
            backend = QMScanBackend(
                name="linear_composite",
                charge=deck.charge,
                multiplicity=deck.multiplicity,
                composite_terms=(
                    QMBackendTerm("vacuum", 1.0, vacuum_backend),
                    QMBackendTerm("ellipsoidal_cpcm", 1.0, cpcm_backend),
                ),
                composite_parallel_workers=deck.composite_parallel_workers,
            )
        else:
            backend = vacuum_backend
    initial_hessian = None
    initial_hessian_source = "metric-diagonal"
    if deck.initial_hessian_path is not None:
        payload = json.loads(deck.initial_hessian_path.read_text(encoding="utf-8"))
        matrix = payload.get("cartesian_hessian", payload.get("hessian"))
        if matrix is None:
            raise ValueError("initial Hessian file must contain 'cartesian_hessian'")
        from .optimizer import coordinate_model_from_xyzin, optimizer_hessian_from_cartesian

        coordinate_model = coordinate_model_from_xyzin(
            xyzin, kind=deck.coordinate_kind
        )
        initial_hessian = optimizer_hessian_from_cartesian(
            np.asarray(matrix, dtype=float), coordinate_model
        )
        initial_hessian_source = f"external:{deck.initial_hessian_path}"
    else:
        coordinate_model = None
    result = optimize_geometry(
        xyzin,
        run_dir=root / "optimizer",
        coordinate_kind=deck.coordinate_kind,
        allow_non_sonic_coordinates=(deck.coordinate_kind in {"cartesian", "sycart"}),
        coordinate_model=coordinate_model,
        backend=backend,
        settings=OptimizerSettings(
            max_steps=deck.max_steps,
            trust_radius=deck.trust_radius,
            max_trust_radius=deck.max_trust_radius,
            include_cv_exponential_field=deck.include_cv_exponential_field,
            canonical_body_frame=(deck.ellipsoidal_cpcm_path is not None),
        ),
        initial_hessian=initial_hessian,
        initial_hessian_source=initial_hessian_source,
    )
    write_optimization_log(root / "optimization.log", deck, result)
    return result


def write_optimization_log(
    path: Path | str,
    deck: OptimizationInputDeck,
    result: OptimizerResult,
) -> Path:
    """Write a compact Gaussian-inspired, non-Gaussian MATRIX log."""

    lines = [
        " Entering MATRIX LINK geometry optimizer",
        f" MATRIX version: {matrix_version()}",
        f" Backend: {deck.backend}",
        f" Model: {deck.model}",
        f" Coordinate system: {deck.coordinate_kind.upper()}",
        f" Core-valence field: {'ORACLE CV_radial exponential' if deck.include_cv_exponential_field else 'none'}",
        f" Reference orientation: center={deck.orientation_weights}; axes=principal inertia",
        "",
        " Initial Cartesian orientation (Angstrom)",
        _orientation_table(result.atoms, result.initial_coordinates_angstrom),
        "",
        " Step       Energy (Eh)       dE          Max Force    RMS Force    Max Disp     RMS Disp      Trust       Q       Status",
    ]
    for item in result.iterations:
        lines.append(
            f" {item.iteration:4d} {item.energy_hartree:18.10f} {item.energy_change_hartree:11.3e} "
            f"{item.gradient_inf_norm:11.3e} {item.gradient_rms_norm:11.3e} "
            f"{item.step_inf_norm:11.3e} {item.step_rms_norm:11.3e} "
            f"{item.trust_radius:10.3e} {item.trust_ratio:8.3f} {item.status}"
        )
    lines.extend(
        [
            "",
            f" Optimization status: {result.status}",
            f" Electronic evaluations: energy={result.energy_evaluations} gradient={result.gradient_evaluations} hessian={result.hessian_evaluations}",
            f" Final energy: {result.final_energy_hartree:.12f} Eh",
            " Final Cartesian orientation (Angstrom)",
            _orientation_table(result.atoms, result.final_coordinates_angstrom),
            "",
            " Normal termination of MATRIX LINK" if result.converged else " MATRIX LINK did not converge",
        ]
    )
    target = Path(path)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _backend_route(deck: OptimizationInputDeck) -> str:
    keywords = " ".join(deck.route_keywords)
    if deck.backend in {"gaussian", "g16", "gdv"}:
        if "pop=" not in keywords.lower():
            keywords = f"{keywords} Pop=CM5".strip()
        return " ".join(item for item in ("#p", deck.model, "Force", keywords) if item)
    if deck.backend == "orca":
        return " ".join(item for item in (deck.model.replace("/", " "), "EnGrad", keywords) if item)
    return keywords


def _method_basis(model: str) -> tuple[str, str]:
    method, separator, basis = model.partition("/")
    return method, basis if separator else "STO-3G"


def _route_tokens(route: str) -> tuple[str, ...]:
    body = re.sub(r"^#\s*[A-Za-z]*\s*", "", route.strip())
    return tuple(body.split())


def _route_value(tokens: Sequence[str], key: str) -> str:
    prefix = key.lower() + "="
    return next((token.split("=", 1)[1] for token in tokens if token.lower().startswith(prefix)), "")


def _opt_value(route: str) -> str:
    match = re.search(r"\bOpt\s*=\s*\(?\s*([A-Za-z_-]+)", route, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _topology_component_count(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    natoms = int(lines[0].strip())
    parent = list(range(natoms))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    in_bonds = False
    for line in lines:
        if line.strip() == "[BONDS]":
            in_bonds = True
            continue
        if in_bonds and line.startswith("["):
            break
        if in_bonds:
            parts = line.split()
            if len(parts) >= 2 and all(part.isdigit() for part in parts[:2]):
                left, right = int(parts[0]) - 1, int(parts[1]) - 1
                a, b = find(left), find(right)
                parent[a] = b
    return len({find(index) for index in range(natoms)})


def _write_xyz(path: Path, atoms: Sequence[str], coords: np.ndarray, title: str) -> None:
    lines = [str(len(atoms)), title]
    lines.extend(
        f"{atom:2s} {row[0]:16.10f} {row[1]:16.10f} {row[2]:16.10f}"
        for atom, row in zip(atoms, np.asarray(coords, dtype=float), strict=True)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _orientation_weights(atoms: Sequence[str], mode: str) -> np.ndarray:
    from matrix_chem import atomic_mass
    from matrix_chem.topology.elements import atomic_number

    numbers = [atomic_number(atom) for atom in atoms]
    if any(number is None for number in numbers):
        raise ValueError("reference orientation needs recognized atomic symbols")
    if mode == "atomic-number":
        return np.asarray(numbers, dtype=float)
    return np.asarray([atomic_mass(int(number)) for number in numbers], dtype=float)


def _orientation_table(atoms: Sequence[str], coords: np.ndarray) -> str:
    return "\n".join(
        f" {index:5d} {atom:3s} {row[0]:15.8f} {row[1]:15.8f} {row[2]:15.8f}"
        for index, (atom, row) in enumerate(zip(atoms, np.asarray(coords), strict=True), start=1)
    )
