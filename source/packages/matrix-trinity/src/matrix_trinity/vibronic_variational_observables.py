"""Finite-box projections and FC/linear-HT contractions without state rescaling."""

import numpy as np
from .hdc_hamiltonian import _shell_basis
from .vibronic_variational import harmonic_overlap_matrix
from .vibronic_observables import _bank


def project_sine_grid_states(coordinates, vectors, *, maximum_total_quanta):
    """Project saved dimensionless tensor-grid states onto harmonic occupations.

    Reuses the tensor Hermite recurrence of local ground-state projection for
    any saved columns. Returned coefficients are NOT normalized or orthogonalized;
    their Gram defect and individual retained norms expose projection loss.
    """
    q = np.asarray(coordinates, float)
    u = np.asarray(vectors)
    if (
        q.ndim != 2
        or not q.shape[1]
        or u.ndim != 2
        or len(u) != len(q)
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(u))
    ):
        raise ValueError("Finite tensor coordinates and eigenvector columns required")
    if not isinstance(maximum_total_quanta, int) or maximum_total_quanta < 1:
        raise ValueError("Positive integer HO projection cutoff required")
    axes = [np.unique(q[:, i]) for i in range(q.shape[1])]
    if any(len(axis) < 2 for axis in axes):
        raise ValueError("Each tensor-grid axis needs at least two points")
    shape = tuple(len(axis) for axis in axes)
    expected = np.array(np.meshgrid(*axes, indexing="ij")).reshape(q.shape[1], -1).T
    if not np.array_equal(q, expected):
        raise ValueError("Coordinates must be an ordered full tensor grid")
    coefficients = u.reshape(shape + (u.shape[1],))
    for dimension, axis in enumerate(axes):
        step = axis[1] - axis[0]
        if not np.allclose(np.diff(axis), step, rtol=1e-12, atol=1e-14):
            raise ValueError("Uniform grid required")
        functions = np.zeros((maximum_total_quanta + 1, len(axis)))
        functions[0] = np.pi ** (-0.25) * np.exp(-(axis**2) / 2)
        functions[1] = np.sqrt(2) * axis * functions[0]
        for k in range(2, len(functions)):
            functions[k] = (
                np.sqrt(2 / k) * axis * functions[k - 1] - np.sqrt((k - 1) / k) * functions[k - 2]
            )
        coefficients = np.moveaxis(
            np.tensordot(functions * np.sqrt(step), coefficients, axes=(1, dimension)), 0, dimension
        )
    basis = np.asarray(_shell_basis(q.shape[1], maximum_total_quanta), np.int64)
    projected = coefficients[tuple(basis.T)]
    gram = projected.conj().T @ projected
    return dict(
        basis=basis,
        vectors=projected,
        diagnostics=dict(
            retained_norms=np.diag(gram).real,
            maximum_gram_defect=float(np.max(abs(gram - u.conj().T @ u))),
            normalized=False,
            orthogonalized=False,
            maximum_total_quanta=maximum_total_quanta,
        ),
    )


def finite_basis_operator_spectrum(
    model,
    initial_basis,
    initial_vectors,
    final_basis,
    final_vectors,
    constants,
    forms,
    *,
    gradients=None,
    maximum_grid_entries=64_000_000,
    maximum_harmonic_pairs=4_000_000,
):
    """Contract arbitrary supplied finite wavefunctions with exact FC/q moments.

    One existing full-Duschinsky Gaussian overlap matrix supplies every operator.
    Ladder action extends the initial HO basis before evaluating q, so no top-
    shell q contribution is dropped. State norms are retained as supplied.
    """
    _, _, c, h, w, _, _, _ = _bank(model, constants, gradients, forms, "absorption")
    ib = np.asarray(initial_basis)
    fb = np.asarray(final_basis)
    ui = np.asarray(initial_vectors)
    uf = np.asarray(final_vectors)
    if (
        ui.ndim != 2
        or uf.ndim != 2
        or len(ui) != len(ib)
        or len(uf) != len(fb)
        or not np.all(np.isfinite(ui))
        or not np.all(np.isfinite(uf))
    ):
        raise ValueError("Finite coefficient columns must match supplied occupation bases")
    for basis in (ib, fb):
        if (
            basis.ndim != 2
            or basis.shape[1] != len(model.initial_frequencies_cm)
            or not len(basis)
            or basis.dtype.kind not in "iu"
            or np.any(basis < 0)
            or len(np.unique(basis, axis=0)) != len(basis)
        ):
            raise ValueError("Unique nonnegative integer occupation rows required")
    rows = {tuple(row) for row in ib}
    for row in ib:
        for mode in range(ib.shape[1]):
            up = row.copy()
            up[mode] += 1
            rows.add(tuple(up))
            if row[mode]:
                down = row.copy()
                down[mode] -= 1
                rows.add(tuple(down))
    expanded = np.asarray(sorted(rows), np.int64)
    lookup = {tuple(row): i for i, row in enumerate(expanded)}
    source = np.zeros((len(expanded), ui.shape[1], len(c)), complex)
    for i, row in enumerate(ib):
        source[lookup[tuple(row)]] += ui[i, :, None] * c[None, :]
        for mode in range(ib.shape[1]):
            up = row.copy()
            up[mode] += 1
            source[lookup[tuple(up)]] += (
                ui[i, :, None] * h[None, :, mode] * np.sqrt((row[mode] + 1) / 2)
            )
            if row[mode]:
                down = row.copy()
                down[mode] -= 1
                source[lookup[tuple(down)]] += (
                    ui[i, :, None] * h[None, :, mode] * np.sqrt(row[mode] / 2)
                )
    overlap, diagnostics = harmonic_overlap_matrix(
        model,
        expanded,
        fb,
        maximum_grid_entries=maximum_grid_entries,
        maximum_harmonic_pairs=maximum_harmonic_pairs,
    )
    moment = (uf.conj().T @ overlap @ source.reshape(len(expanded), -1)).reshape(
        uf.shape[1], ui.shape[1], len(c)
    )
    strengths = np.einsum("fia,jab,fib->jfi", moment, w, moment.conj(), optimize=True).real
    return dict(
        moments=moment,
        strengths=strengths,
        diagnostics={
            **diagnostics,
            "initial_operator_basis_size": len(expanded),
            "wavefunction_normalization_changed": False,
        },
    )


__all__ = ["project_sine_grid_states", "finite_basis_operator_spectrum"]
