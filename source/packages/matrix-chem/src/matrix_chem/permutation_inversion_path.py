"""Infer a confined-path permutation--inversion reflection from labelled geometries.

The ordered Cartesian path is supplied by the coordinate builder. No molecular
name, group symbol, bond graph, or manually selected atom exchange enters the
inference. An ambiguous physical action is an error, never an arbitrary choice.
"""

from __future__ import annotations

from itertools import permutations, product
import hashlib
from math import factorial

import numpy as np

from .permutation_inversion import (
    OraclePermutationInversion, PermutationInversionOperation, _closure,
)
from .permutation_inversion_grid import derive_permutation_inversion_grid_action


def infer_confined_path_reflection(
    atoms,
    nuclear_labels,
    geometries_angstrom,
    masses_amu,
    *,
    coordinates=None,
    tolerance_angstrom=2e-4,
    max_candidates=100000,
):
    """Return ``(ORACLE contract, geometry-grid action)`` for path reversal.

    The complete, ordered path must contain at least three geometries. Every
    isotope-preserving involutive atom permutation and both inversion parities
    are considered, subject to an explicit combinatorial budget. The existing
    SO(3) geometry-grid certifier verifies the full path and its cocycle. A
    symmetric distance-matrix screen avoids unnecessary alignment work.
    All certified reversals generate the returned group; a planar path can
    admit distinct operations with the same grid action but different frame
    actions, and neither may be discarded arbitrarily.

    ``coordinates`` may be supplied to certify that reversal is a reflection
    about a single coordinate midpoint. The returned action is on grid indices,
    so it can be applied by any DVR consumer without naming that coordinate.
    """
    atoms = tuple(atoms)
    labels = tuple(nuclear_labels)
    xyz = np.asarray(geometries_angstrom, dtype=float)
    masses = np.asarray(masses_amu, dtype=float)
    if (not atoms or len(labels) != len(atoms) or any(not isinstance(a, str) or not a for a in atoms)
            or any(not isinstance(x, str) or not x for x in labels)):
        raise ValueError("one element and nuclear label are required per atom")
    if xyz.ndim != 3 or xyz.shape[0] < 3 or xyz.shape[1:] != (len(atoms), 3) or not np.all(np.isfinite(xyz)):
        raise ValueError("confined path needs at least three finite labelled geometries")
    if masses.shape != (len(atoms),) or not np.all(np.isfinite(masses)) or np.any(masses <= 0):
        raise ValueError("one positive finite mass is required per atom")
    if not np.isfinite(tolerance_angstrom) or tolerance_angstrom <= 0:
        raise ValueError("path tolerance must be finite and positive")
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    if coordinates is not None:
        q = np.asarray(coordinates, dtype=float)
        if q.shape != (len(xyz),) or not np.all(np.isfinite(q)) or np.any(np.diff(q) <= 0):
            raise ValueError("confined coordinate must increase along the geometry path")
        if not np.allclose(q + q[::-1], q[0] + q[-1], rtol=0, atol=1e-10):
            raise ValueError("ordered path grid is not symmetric about one coordinate midpoint")

    classes = {}
    for i, key in enumerate(zip(atoms, labels, masses, strict=True)):
        classes.setdefault(key, []).append(i)
    groups = tuple(classes.values())
    count = 2
    for group in groups:
        count *= factorial(len(group))
        if count > max_candidates:
            raise ValueError(f"path PI candidate count exceeds {max_candidates}; reduce isotope ambiguity or raise the budget")

    distances = np.linalg.norm(xyz[:, :, None] - xyz[:, None, :], axis=-1)
    # Any pair distance can differ by at most the two corresponding Cartesian
    # errors; this conservative bound never discards a valid mass-RMSD match.
    bound = 2 * tolerance_angstrom * np.sqrt(masses.sum() / masses.min())
    reverse = np.arange(len(xyz) - 1, -1, -1)
    identity = tuple(range(len(atoms)))
    digest = hashlib.sha256(xyz.tobytes()).hexdigest()
    qualified = []
    for choices in product(*(permutations(group) for group in groups)):
        p = list(identity)
        for group, permuted in zip(groups, choices, strict=True):
            for old, new in zip(group, permuted, strict=True):
                p[old] = new
        p = tuple(p)
        if any(p[p[i]] != i for i in range(len(p))):
            continue
        inverse = np.argsort(p)
        if np.max(np.abs(distances[:, inverse][:, :, inverse] - distances[reverse])) > bound:
            continue
        for inverted in (False, True):
            op = PermutationInversionOperation(p, inverted)
            if p == identity and not inverted:
                continue
            operations, words = _closure(len(atoms), (op,), 2)
            contract = OraclePermutationInversion(
                atoms=atoms, nuclear_labels=labels, geometry_sha256=digest,
                point_group="PATH_DERIVED", generators=(op,),
                generator_sources=("PATH:CONFINED_REFLECTION",),
                operations=operations, words=words, rotors=(),
                rotor_diagnostics=(), tolerance_angstrom=tolerance_angstrom,
            )
            try:
                action = derive_permutation_inversion_grid_action(
                    contract, xyz, masses, tolerance_angstrom=tolerance_angstrom,
                )
            except ValueError:
                continue
            if np.array_equal(action.targets[1], reverse):
                qualified.append(op)
    if not qualified:
        raise ValueError("no isotope-preserving PI involution maps the complete confined path onto its reversal")
    operations, words = _closure(len(atoms), tuple(qualified), 4096)
    contract = OraclePermutationInversion(
        atoms=atoms, nuclear_labels=labels, geometry_sha256=digest,
        point_group="PATH_DERIVED", generators=tuple(qualified),
        generator_sources=("PATH:CONFINED_REFLECTION",) * len(qualified),
        operations=operations, words=words, rotors=(), rotor_diagnostics=(),
        tolerance_angstrom=tolerance_angstrom,
    )
    action = derive_permutation_inversion_grid_action(
        contract, xyz, masses, tolerance_angstrom=tolerance_angstrom,
    )
    return contract, action
