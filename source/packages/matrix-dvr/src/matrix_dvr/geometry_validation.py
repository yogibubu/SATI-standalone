"""Atom-order, permutation and geometry consistency checks."""

from __future__ import annotations

from itertools import permutations
from typing import Sequence

import numpy as np


def best_atom_permutation(
    reference_atoms: Sequence[str],
    reference_coordinates: np.ndarray,
    candidate_atoms: Sequence[str],
    candidate_coordinates: np.ndarray,
    *,
    tolerance_angstrom: float = 1.0e-5,
) -> dict[str, object]:
    ref_atoms = tuple(str(item) for item in reference_atoms)
    cand_atoms = tuple(str(item) for item in candidate_atoms)
    reference = np.asarray(reference_coordinates, dtype=float)
    candidate = np.asarray(candidate_coordinates, dtype=float)
    if len(ref_atoms) != len(cand_atoms) or reference.shape != candidate.shape:
        return {"valid": False, "reason": "atom_count_or_shape_mismatch"}
    groups: dict[str, list[int]] = {}
    for index, atom in enumerate(cand_atoms):
        groups.setdefault(atom.casefold(), []).append(index)
    ref_groups: dict[str, list[int]] = {}
    for index, atom in enumerate(ref_atoms):
        ref_groups.setdefault(atom.casefold(), []).append(index)
    if {key: len(value) for key, value in groups.items()} != {key: len(value) for key, value in ref_groups.items()}:
        return {"valid": False, "reason": "element_multiplicity_mismatch"}
    assignments = [list(permutations(groups[key])) for key in sorted(groups)]
    best_mapping: list[int] | None = None
    best_rmsd = float("inf")
    for choices in _product(assignments):
        mapping = [0] * len(ref_atoms)
        for key, choice in zip(sorted(groups), choices, strict=True):
            for ref_index, candidate_index in zip(ref_groups[key], choice, strict=True):
                mapping[ref_index] = candidate_index
        aligned = _kabsch(candidate[mapping], reference)
        rmsd = float(np.sqrt(np.mean((aligned - reference) ** 2)))
        if rmsd < best_rmsd:
            best_rmsd, best_mapping = rmsd, mapping
    assert best_mapping is not None
    return {
        "valid": best_rmsd <= float(tolerance_angstrom),
        "rmsd_angstrom": best_rmsd,
        "mapping_reference_to_candidate": best_mapping,
        "atom_order_changed": tuple(best_mapping) != tuple(range(len(ref_atoms))),
        "tolerance_angstrom": float(tolerance_angstrom),
    }


def _product(groups: Sequence[Sequence[tuple[int, ...]]]):
    if not groups:
        yield ()
        return
    for head in groups[0]:
        for tail in _product(groups[1:]):
            yield (head, *tail)


def _kabsch(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    candidate_centered = candidate - np.mean(candidate, axis=0)
    reference_centered = reference - np.mean(reference, axis=0)
    covariance = candidate_centered.T @ reference_centered
    left, _, right = np.linalg.svd(covariance)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return candidate_centered @ rotation + np.mean(reference, axis=0)


__all__ = ["best_atom_permutation"]
