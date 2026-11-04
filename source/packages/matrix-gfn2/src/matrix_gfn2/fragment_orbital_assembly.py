"""Build a full-rank MO seed directly from capped fragment orbitals.

No global Fock diagonalization is performed. The dense overlap/QR/Cholesky
operations are deliberately retained in this proof-of-concept: they must be
replaced by domain-local factorizations before claiming linear scaling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky, qr, solve_triangular


@dataclass(frozen=True)
class AssembledFragmentOrbitals:
    coefficients: np.ndarray
    orbital_fragments: np.ndarray
    occupied: int
    density: np.ndarray
    maximum_orthonormality_error: float
    occupied_candidates: int
    virtual_candidates: int


def _orthonormalize(columns: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    gram = columns.T @ overlap @ columns
    upper = cholesky(0.5 * (gram + gram.T), lower=False)
    return solve_triangular(upper, columns.T, lower=False, trans="T").T


def assemble_fragment_orbitals(
    states: tuple[object, ...], overlap: np.ndarray, electrons: int,
) -> AssembledFragmentOrbitals:
    """Select projected fragment occupied/virtual columns and S-orthogonalize.

    Projected link-atom MOs are redundant. Pivoted QR selects an independent
    occupied span first; the remaining projected columns complete the virtual
    span. This changes the fragment density and therefore requires a genuine
    subsequent global relaxation, rather than a density-only patch.
    """
    s = np.asarray(overlap, dtype=float)
    size = len(s)
    occupied = electrons // 2
    if (s.shape != (size, size) or electrons % 2 or not 0 < occupied < size
            or not states or not np.all(np.isfinite(s))):
        raise ValueError("invalid fragment MO assembly model")
    groups: list[list[np.ndarray]] = [[], []]
    group_labels: list[list[int]] = [[], []]
    covered = np.zeros(size, dtype=bool)
    for fragment, state in enumerate(states):
        rows = np.asarray(state.global_orbitals, dtype=np.intp)
        projected = np.asarray(state.coefficients_on_real_atoms, dtype=float)
        occupations = (np.asarray(state.alpha_occupations)
                       + np.asarray(state.beta_occupations))
        if (projected.shape[0] != len(rows)
                or projected.shape[1] != len(occupations)
                or np.any(rows < 0) or np.any(rows >= size)
                or np.any(covered[rows])):
            raise ValueError("fragment orbitals overlap or have invalid dimensions")
        covered[rows] = True
        for column in range(projected.shape[1]):
            candidate = np.zeros(size)
            candidate[rows] = projected[:, column]
            if np.linalg.norm(candidate) <= 1.0e-10:
                continue
            kind = 0 if occupations[column] > 1.0 else 1
            groups[kind].append(candidate)
            group_labels[kind].append(fragment)
    if not np.all(covered) or not groups[0] or not groups[1]:
        raise ValueError("fragment MOs do not span the molecular basis")
    occupied_candidates = np.column_stack(groups[0])
    virtual_candidates = np.column_stack(groups[1])
    # Normalize before pivoting so the choice reflects linear independence,
    # not merely the norm loss caused by removing a link atom.
    normalized = occupied_candidates / np.maximum(
        np.linalg.norm(occupied_candidates, axis=0), 1.0e-15,
    )
    _, upper, pivots = qr(normalized, mode="economic", pivoting=True)
    if (len(pivots) < occupied
            or abs(upper[occupied - 1, occupied - 1]) < 1.0e-8):
        raise ValueError("projected occupied fragment MOs have insufficient rank")
    selected_occupied = occupied_candidates[:, pivots[:occupied]]
    c_occupied = _orthonormalize(selected_occupied, s)
    projected_virtual = virtual_candidates - c_occupied @ (
        c_occupied.T @ s @ virtual_candidates
    )
    normalized_virtual = projected_virtual / np.maximum(
        np.linalg.norm(projected_virtual, axis=0), 1.0e-15,
    )
    _, upper, virtual_pivots = qr(
        normalized_virtual, mode="economic", pivoting=True,
    )
    nvirtual = size - occupied
    if (len(virtual_pivots) < nvirtual
            or abs(upper[nvirtual - 1, nvirtual - 1]) < 1.0e-8):
        raise ValueError("projected virtual fragment MOs have insufficient rank")
    selected_virtual = projected_virtual[:, virtual_pivots[:nvirtual]]
    c_virtual = _orthonormalize(selected_virtual, s)
    coefficients = np.column_stack((c_occupied, c_virtual))
    error = float(np.max(abs(coefficients.T @ s @ coefficients - np.eye(size))))
    if error > 1.0e-8:
        raise RuntimeError("assembled fragment MOs are not S-orthonormal")
    labels = np.asarray(
        [group_labels[0][index] for index in pivots[:occupied]]
        + [group_labels[1][index] for index in virtual_pivots[:nvirtual]],
        dtype=np.intp,
    )
    density = 2.0 * c_occupied @ c_occupied.T
    return AssembledFragmentOrbitals(
        coefficients, labels, occupied, density, error,
        occupied_candidates.shape[1], virtual_candidates.shape[1],
    )
