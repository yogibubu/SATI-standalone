"""High-level name/SMILES-to-Cartesian SWITCH workflows."""

from __future__ import annotations

from pathlib import Path

from matrix_chem.geometry import MolecularGeometry

from .geometry import build_cartesian_seed
from .names import NameResolution, resolve_name
from .backend_policy import ALLOW_RDKIT_FALLBACK, resolve_switch_backend_policy
from .parser import SwitchUnsupportedFeatureError, parse_smiles


def smiles_to_cartesian(
    smiles: str,
    *,
    title: str = "",
    multiplicity: int | None = None,
    complete_hydrogens: bool = True,
    fallback_policy: str | None = None,
    fallback_random_seed: int = 61453,
) -> MolecularGeometry:
    policy = resolve_switch_backend_policy(fallback_policy)
    try:
        graph = parse_smiles(smiles)
    except SwitchUnsupportedFeatureError as exc:
        if policy != ALLOW_RDKIT_FALLBACK:
            raise
        from .rdkit_fallback import build_rdkit_fallback_geometry

        return build_rdkit_fallback_geometry(
            smiles,
            title=title,
            multiplicity=multiplicity,
            reason=exc.message,
            random_seed=fallback_random_seed,
        )
    geometry = build_cartesian_seed(
        graph,
        title=title,
        multiplicity=multiplicity,
        complete_hydrogens=complete_hydrogens,
    )
    return MolecularGeometry(
        atoms=geometry.atoms,
        coordinates_angstrom=geometry.coordinates_angstrom,
        comment=geometry.comment,
        source_format=geometry.source_format,
        source_path=geometry.source_path,
        charge=geometry.charge,
        multiplicity=geometry.multiplicity,
        fixed_parameters=geometry.fixed_parameters,
        metadata={
            **dict(geometry.metadata),
            "primary_backend": "matrix-switch",
            "geometry_backend": "matrix-switch",
            "fallback_used": False,
            "fallback_policy": policy,
        },
    )


def name_to_cartesian(
    name: str,
    *,
    allow_remote: bool = False,
    cache_path: Path | None = None,
    timeout: float = 10.0,
    multiplicity: int | None = None,
    complete_hydrogens: bool = True,
) -> tuple[MolecularGeometry, NameResolution]:
    resolution = resolve_name(
        name,
        allow_remote=allow_remote,
        cache_path=cache_path,
        timeout=timeout,
    )
    geometry = smiles_to_cartesian(
        resolution.smiles,
        title=name,
        multiplicity=multiplicity,
        complete_hydrogens=complete_hydrogens,
    )
    geometry = MolecularGeometry(
        atoms=geometry.atoms,
        coordinates_angstrom=geometry.coordinates_angstrom,
        comment=geometry.comment,
        source_format=geometry.source_format,
        charge=geometry.charge,
        multiplicity=geometry.multiplicity,
        metadata={
            **dict(geometry.metadata),
            "name_resolution_source": resolution.source,
            "name_resolution_identifier": resolution.identifier,
        },
    )
    return geometry, resolution


__all__ = ["name_to_cartesian", "smiles_to_cartesian"]
