"""Select native PIC couplings against the evaluated Cartesian force field."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

import numpy as np

from matrix_chem import (
    BOHR_TO_ANGSTROM,
    PrimitiveMMCoupling,
    PrimitiveMMForceField,
    classify_primitive_coupling,
    primitive_b_matrix,
    primitive_force_field_bonded_cartesian_hessian,
)
from matrix_chem.primitive_coordinates import Primitive
from matrix_qm import HESSIAN_EIGENVALUE_TO_CM


@dataclass(frozen=True)
class PhysicalPicSelection:
    field: PrimitiveMMForceField
    coupling_count: int
    relative_hessian_error: float
    maximum_frequency_error_cm: float
    rms_frequency_error_cm: float


def select_physical_pic_couplings(
    full_field: PrimitiveMMForceField,
    primitives: Sequence[Primitive],
    coordinates_angstrom: np.ndarray,
    target_cartesian_hessian_hartree_per_bohr2: np.ndarray,
    nonbonded_cartesian_hessian_hartree_per_bohr2: np.ndarray,
    masses_amu: Sequence[float],
    *,
    maximum_frequency_error_cm: float = 0.5,
    maximum_relative_hessian_error: float = 2.0e-5,
    maximum_couplings: int | None = None,
) -> PhysicalPicSelection:
    """Choose the first rank-independent support passing both physical gates.

    Local diagonal functions and their reference gradients remain fixed.  Only
    reference-centered cross coefficients are refitted, so every support has
    the same stationary reference geometry.  The target is the QM Cartesian
    Hessian after adding the independently evaluated nonbonded Hessian.
    """
    if full_field.coupling_reference_values is None:
        raise ValueError("physical selection requires reference-centered full PIC fit")
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    target = np.asarray(target_cartesian_hessian_hartree_per_bohr2, dtype=float)
    nonbonded = np.asarray(nonbonded_cartesian_hessian_hartree_per_bohr2, dtype=float)
    if target.shape != (xyz.size, xyz.size) or nonbonded.shape != target.shape:
        raise ValueError("physical selection Hessians do not match the geometry")
    masses = np.repeat(np.asarray(masses_amu, dtype=float), 3)
    if masses.shape != (xyz.size,) or np.any(masses <= 0):
        raise ValueError("physical selection requires positive atomic masses")
    b = primitive_b_matrix(primitives, xyz)
    _u, singular, vh = np.linalg.svd(b, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-10))
    v = vh[:rank].T
    bv = b @ v
    bonded_target = (target - nonbonded) / BOHR_TO_ANGSTROM**2
    full_hessian = primitive_force_field_bonded_cartesian_hessian(
        full_field, primitives, xyz
    )
    off_diagonal = np.asarray(full_field.sparse_typed_hessian, dtype=float).copy()
    np.fill_diagonal(off_diagonal, 0.0)
    base = full_hessian - b.T @ off_diagonal @ b
    upper = np.triu_indices(rank)
    weights = np.where(upper[0] == upper[1], 1.0, math.sqrt(2.0))
    pairs = [(i, j) for i in range(len(primitives)) for j in range(i + 1, len(primitives))]
    columns = []
    for i, j in pairs:
        contribution = np.outer(bv[i], bv[j]) + np.outer(bv[j], bv[i])
        columns.append(contribution[upper] * weights)
    design = np.asarray(columns, dtype=float).T
    norms = np.linalg.norm(design, axis=0)
    observable = norms > 1e-12
    design = design[:, observable] / norms[observable]
    pairs = [pair for pair, keep in zip(pairs, observable, strict=True) if keep]
    norms = norms[observable]
    residual_matrix = v.T @ (bonded_target - base) @ v
    response = residual_matrix[upper] * weights
    gram = design.T @ design
    projections = design.T @ response
    selected: list[int] = []
    reference_eigenvalues = np.linalg.eigvalsh(
        target / np.outer(np.sqrt(masses), np.sqrt(masses))
    )[-rank:]
    reference_frequencies = _signed_frequencies(reference_eigenvalues)
    limit = min(len(pairs), rank * (rank + 1) // 2)
    if maximum_couplings is not None:
        limit = min(limit, int(maximum_couplings))
    for _ in range(limit):
        correlations = np.abs(projections)
        correlations[selected] = -1.0
        index = int(np.argmax(correlations))
        if correlations[index] <= 0.0:
            break
        proposed = selected + [index]
        subgram = gram[np.ix_(proposed, proposed)]
        if np.linalg.eigvalsh(subgram)[0] < 1e-10:
            projections[index] = 0.0
            continue
        selected = proposed
        coefficients = np.linalg.solve(subgram, design[:, selected].T @ response)
        projections = design.T @ (response - design[:, selected] @ coefficients)
        couplings = tuple(
            PrimitiveMMCoupling(
                pair,
                (full_field.primitive_type_ids[pair[0]], full_field.primitive_type_ids[pair[1]]),
                float(value / norms[column]),
                abs(float(value / norms[column])) / math.sqrt(
                    full_field.diagonal_force_constants[pair[0]]
                    * full_field.diagonal_force_constants[pair[1]]
                ),
                classify_primitive_coupling(primitives[pair[0]], primitives[pair[1]]),
                "CARTESIAN_HESSIAN_AND_GF_SELECTED",
            )
            for column, value in zip(selected, coefficients, strict=True)
            for pair in (pairs[column],)
        )
        sparse = np.diag(full_field.diagonal_force_constants).copy()
        for coupling in couplings:
            i, j = coupling.primitive_indices
            sparse[i, j] = sparse[j, i] = coupling.force_constant
        estimate = base + b.T @ (sparse - np.diag(full_field.diagonal_force_constants)) @ b
        relative = float(np.linalg.norm(estimate - bonded_target) / np.linalg.norm(target / BOHR_TO_ANGSTROM**2))
        model_eigenvalues = np.linalg.eigvalsh(
            (estimate * BOHR_TO_ANGSTROM**2 + nonbonded)
            / np.outer(np.sqrt(masses), np.sqrt(masses))
        )[-rank:]
        frequency_errors = _signed_frequencies(model_eigenvalues) - reference_frequencies
        if (relative <= maximum_relative_hessian_error
                and np.max(np.abs(frequency_errors)) <= maximum_frequency_error_cm):
            field = replace(
                full_field,
                retained_couplings=couplings,
                sparse_typed_hessian=sparse,
                coupling_alerts=(),
            )
            actual = primitive_force_field_bonded_cartesian_hessian(field, primitives, xyz)
            actual_relative = float(np.linalg.norm(actual - bonded_target)
                                    / np.linalg.norm(target / BOHR_TO_ANGSTROM**2))
            if actual_relative > maximum_relative_hessian_error:
                raise ValueError("selected PIC Hessian fails evaluated-field certification")
            return PhysicalPicSelection(
                replace(field, sparse_cartesian_reconstruction_rms=float(
                    np.sqrt(np.mean((actual - bonded_target)**2))
                )),
                len(couplings), actual_relative,
                float(np.max(np.abs(frequency_errors))),
                float(np.sqrt(np.mean(frequency_errors**2))),
            )
    raise ValueError("no rank-independent PIC coupling support passes the Hessian and GF gates")


def _signed_frequencies(eigenvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(eigenvalues, dtype=float)
    return np.sign(values) * np.sqrt(np.abs(values)) * HESSIAN_EIGENVALUE_TO_CM
