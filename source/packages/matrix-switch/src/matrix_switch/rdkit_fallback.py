"""The only permitted RDKit integration boundary in MATRIX.

This module is imported lazily after SWITCH reports a specifically unsupported
SMILES feature.  Invalid input and arbitrary SWITCH failures must never reach
this backend.
"""

from __future__ import annotations

import warnings

import numpy as np

from matrix_chem.geometry import MolecularGeometry

from .backend_policy import ALLOW_RDKIT_FALLBACK


RDKIT_FALLBACK_SCHEMA = "matrix.switch.rdkit-fallback.v1"
DEFAULT_RDKIT_FALLBACK_RANDOM_SEED = 61453


class RDKitFallbackUnavailableError(RuntimeError):
    """The optional fallback was requested but is not installed."""


class SwitchFallbackWarning(UserWarning):
    """SWITCH could not represent an input and selected its optional fallback."""


def rdkit_fallback_available() -> bool:
    try:
        _rdkit_modules()
    except RDKitFallbackUnavailableError:
        return False
    return True


def build_rdkit_fallback_geometry(
    smiles: str,
    *,
    title: str = "",
    multiplicity: int | None = None,
    reason: str,
    random_seed: int = DEFAULT_RDKIT_FALLBACK_RANDOM_SEED,
) -> MolecularGeometry:
    """Build a seed and record why SWITCH explicitly delegated the input."""

    Chem, AllChem, rdBase = _rdkit_modules()
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit fallback could not parse SMILES: {smiles}")
    charge = int(Chem.GetFormalCharge(molecule))
    molecule = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(random_seed)
    embed_status = int(AllChem.EmbedMolecule(molecule, parameters))
    if embed_status != 0:
        raise ValueError(f"RDKit fallback could not embed SMILES in 3D: {smiles}")
    conformer = molecule.GetConformer()
    atoms = tuple(atom.GetSymbol() for atom in molecule.GetAtoms())
    coordinates = np.asarray(
        [
            [
                float(conformer.GetAtomPosition(index).x),
                float(conformer.GetAtomPosition(index).y),
                float(conformer.GetAtomPosition(index).z),
            ]
            for index in range(len(atoms))
        ],
        dtype=float,
    )
    warnings.warn(
        f"SWITCH delegated one unsupported SMILES feature to RDKit: {reason}",
        SwitchFallbackWarning,
        stacklevel=2,
    )
    return MolecularGeometry(
        atoms=atoms,
        coordinates_angstrom=coordinates,
        comment=title or smiles,
        source_format="smiles_rdkit_fallback",
        charge=charge,
        multiplicity=multiplicity,
        metadata={
            "schema": RDKIT_FALLBACK_SCHEMA,
            "smiles": smiles,
            "primary_backend": "matrix-switch",
            "geometry_backend": "rdkit-fallback",
            "fallback_used": True,
            "fallback_policy": ALLOW_RDKIT_FALLBACK,
            "fallback_owner": "matrix-switch.rdkit_fallback",
            "fallback_reason_code": "switch.unsupported-feature",
            "fallback_reason": str(reason),
            "rdkit_version": str(rdBase.rdkitVersion),
            "rdkit_random_seed": int(random_seed),
            "rdkit_embed_status": embed_status,
        },
    )


def _rdkit_modules():
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RDKitFallbackUnavailableError(
            "SWITCH reported an unsupported feature, but the optional RDKit "
            "fallback is not installed"
        ) from exc
    return Chem, AllChem, rdBase


def rdkit_reference_modules():
    """Expose optional RDKit modules to nonproduction reference benchmarks."""
    Chem, AllChem, _ = _rdkit_modules()
    from rdkit.Chem import rdMolDescriptors

    return Chem, AllChem, rdMolDescriptors


__all__ = [
    "rdkit_reference_modules",
    "DEFAULT_RDKIT_FALLBACK_RANDOM_SEED",
    "RDKIT_FALLBACK_SCHEMA",
    "RDKitFallbackUnavailableError",
    "SwitchFallbackWarning",
    "build_rdkit_fallback_geometry",
    "rdkit_fallback_available",
]
