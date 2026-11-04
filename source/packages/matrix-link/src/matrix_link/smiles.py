from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

from matrix_chem.geometry import MolecularGeometry
from matrix_chem.geometry_io import GeometryParseError
from matrix_chem import read_xyz
from matrix_chem.symmetry import analyze_molecular_symmetry, symmetrize_molecular_geometry
from matrix_switch import (
    RDKitFallbackUnavailableError,
    smiles_to_cartesian,
)


SMILES_SOURCE_FORMAT = "smiles_switch"
SMILES_MARKER = "SMILES"
DEFAULT_SMILES_FALLBACK_RANDOM_SEED = 61453


@dataclass(frozen=True)
class SmilesInput:
    smiles: str
    title: str = ""
    charge: int | None = None
    multiplicity: int | None = None
    route_lines: tuple[str, ...] = ()
    source_path: Path | None = None


def is_legacy_smiles_input(path: Path) -> bool:
    target = Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    return _looks_like_legacy_smiles_text(text)


def extract_legacy_smiles_input(path: Path) -> SmilesInput:
    target = Path(path)
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    route_lines = _route_lines(lines)
    if not _route_requests_smiles(route_lines):
        raise GeometryParseError("legacy input is not marked as SMILES")

    route_end = _route_end(lines)
    if route_end is None:
        raise GeometryParseError("SMILES input needs a route section starting with #")
    idx = _next_nonblank(lines, route_end)
    title_start = idx
    while idx < len(lines) and lines[idx].strip():
        idx += 1
    title = " ".join(line.strip() for line in lines[title_start:idx] if line.strip())
    idx = _next_nonblank(lines, idx)
    if idx >= len(lines) or not _is_charge_multiplicity(lines[idx]):
        raise GeometryParseError("SMILES input needs charge and multiplicity before SMILES")
    charge, multiplicity = (int(value) for value in lines[idx].split()[:2])
    idx = _next_nonblank(lines, idx + 1)
    if idx >= len(lines) or not lines[idx].strip():
        raise GeometryParseError("SMILES input contains no SMILES line")
    smiles = lines[idx].strip()
    return SmilesInput(
        smiles=smiles,
        title=title or target.stem,
        charge=charge,
        multiplicity=multiplicity,
        route_lines=route_lines,
        source_path=target,
    )


def read_legacy_smiles_input(
    path: Path,
    *,
    fallback_policy: str | None = None,
) -> MolecularGeometry:
    smiles_input = extract_legacy_smiles_input(path)
    return smiles_to_geometry(
        smiles_input.smiles,
        title=smiles_input.title,
        charge=smiles_input.charge,
        multiplicity=smiles_input.multiplicity,
        source_path=smiles_input.source_path,
        route_lines=smiles_input.route_lines,
        fallback_policy=fallback_policy,
    )


def smiles_to_geometry(
    smiles: str,
    *,
    title: str = "",
    charge: int | None = None,
    multiplicity: int | None = None,
    source_path: Path | None = None,
    route_lines: tuple[str, ...] = (),
    random_seed: int = DEFAULT_SMILES_FALLBACK_RANDOM_SEED,
    relaxation_method: str = "NONE",
    fallback_policy: str | None = None,
) -> MolecularGeometry:
    """Build the deterministic SWITCH Cartesian seed for a SMILES string.

    Structural relaxation is deliberately opt-in here.  Keymaker owns the
    automatic initialization route: it may use an LCB26 geometry or refine
    this SWITCH seed with resident GFN2-xTB in Cartesian coordinates before
    ORACLE perception and SMITH SONIC construction.  Parsing alone never
    launches a calculation as an undocumented side effect.
    """
    normalized_smiles = _normalize_legacy_smiles(smiles)
    requested_relaxation = str(relaxation_method).strip().upper().replace("-", "_")
    if requested_relaxation not in {"NONE", "GFN_FF"}:
        raise ValueError("relaxation_method must be NONE or GFN_FF")
    try:
        seed = smiles_to_cartesian(
            normalized_smiles,
            title=title or smiles,
            multiplicity=multiplicity,
            complete_hydrogens=True,
            fallback_policy=fallback_policy,
            fallback_random_seed=random_seed,
        )
    except (ValueError, RDKitFallbackUnavailableError) as exc:
        raise GeometryParseError(str(exc)) from exc
    job_charge = int(seed.charge or 0) if charge is None else int(charge)
    applied_relaxation = "NONE"
    optimize_status: int | None = None
    if requested_relaxation == "GFN_FF":
        seed = _relax_with_gfn_ff(
            seed,
            charge=int(job_charge),
            multiplicity=int(multiplicity or 1),
        )
        approximate_symmetry = analyze_molecular_symmetry(
            seed,
            distance_tolerance=5.0e-2,
            inertia_tolerance=1.0e-3,
            max_rotation_order=6,
        )
        seed = symmetrize_molecular_geometry(seed, approximate_symmetry)
        applied_relaxation = "GFN-FF"
        optimize_status = 0

    return MolecularGeometry(
        atoms=seed.atoms,
        coordinates_angstrom=seed.coordinates_angstrom,
        comment=seed.comment,
        source_format=SMILES_SOURCE_FORMAT,
        source_path=source_path,
        charge=job_charge,
        multiplicity=multiplicity,
        metadata={
            **dict(seed.metadata),
            "smiles": smiles,
            "normalized_smiles": normalized_smiles,
            "route": route_lines,
            "smiles_parser": (
                "RDKit fallback"
                if bool(seed.metadata.get("fallback_used", False))
                else "MATRIX SWITCH"
            ),
            "hydrogen_completion": (
                "RDKIT_FALLBACK"
                if bool(seed.metadata.get("fallback_used", False))
                else "MATRIX_VALENCE_COMPLETION"
            ),
            "relaxation_method": applied_relaxation,
            "requested_relaxation": requested_relaxation,
            "optimize_status": optimize_status,
        },
    )


def _relax_with_gfn_ff(
    geometry: MolecularGeometry,
    *,
    charge: int,
    multiplicity: int,
) -> MolecularGeometry:
    try:
        from matrix_xtb import run_xtb_job, write_xtb_point_input
    except ImportError as exc:
        raise GeometryParseError(
            "GFN-FF SMILES relaxation needs the matrix-link[backends] xTB adapter"
        ) from exc
    with tempfile.TemporaryDirectory(prefix="matrix-smiles-gfnff-") as scratch_text:
        scratch = Path(scratch_text)
        input_path = write_xtb_point_input(
            scratch / "input.xyz",
            list(geometry.atoms),
            geometry.coordinates_angstrom,
            charge=charge,
            multiplicity=multiplicity,
        )
        run = run_xtb_job(
            scratch,
            input_path=input_path,
            output_path=scratch / "xtb.out",
            extra_args=("--gfnff", "--opt"),
        )
        optimized = scratch / "xtbopt.xyz"
        if not run.success or not optimized.is_file():
            raise GeometryParseError(
                f"GFN-FF could not relax the SMILES seed: {run.message}"
            )
        result = read_xyz(optimized)
    return MolecularGeometry(
        atoms=result.atoms,
        coordinates_angstrom=result.coordinates_angstrom,
        comment=geometry.comment,
        source_format=geometry.source_format,
        source_path=geometry.source_path,
        charge=charge,
        multiplicity=multiplicity,
        metadata=dict(geometry.metadata),
    )


def _normalize_legacy_smiles(smiles: str) -> str:
    """Apply format-level normalization without molecule-specific rewrites."""

    return smiles.strip()


def _looks_like_legacy_smiles_text(text: str) -> bool:
    return _route_requests_smiles(_route_lines(text.splitlines()))


def _route_end(lines: list[str]) -> int | None:
    idx = 0
    while idx < len(lines):
        if lines[idx].strip().startswith("#"):
            while idx < len(lines) and lines[idx].strip():
                idx += 1
            return idx
        idx += 1
    return None


def _route_lines(lines: list[str]) -> tuple[str, ...]:
    idx = 0
    while idx < len(lines):
        if lines[idx].strip().startswith("#"):
            route: list[str] = []
            while idx < len(lines) and lines[idx].strip():
                route.append(lines[idx].strip())
                idx += 1
            return tuple(route)
        idx += 1
    return ()


def _route_requests_smiles(route_lines: tuple[str, ...]) -> bool:
    return SMILES_MARKER in " ".join(route_lines).upper().split()


def _next_nonblank(lines: list[str], idx: int) -> int:
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    return idx


def _is_charge_multiplicity(line: str) -> bool:
    parts = line.split()
    if len(parts) < 2:
        return False
    try:
        int(parts[0])
        int(parts[1])
    except ValueError:
        return False
    return True
