"""Recover archived Gaussian electrical anharmonicity without new QM.

The ``num derivs`` arrays store derivatives, not displaced property values:
their first N records are first derivatives and the next N are diagonal
second derivatives along the archived displacement vectors. Missing fully
mixed third derivatives are returned as NaN, never silently as observations.
"""
from hashlib import sha256
from itertools import permutations
from pathlib import Path
import numpy as np
from .fchk import _read_fchk_blocks, read_gaussian_fchk


def read_gaussian_anharmonic_dipole(path):
    """Return D1, complete D2, acquired D3[i,j,j] in QMW coordinates.

    QMW has units sqrt(amu)*bohr. Cartesian dipole components share the
    original FCHK frame. The symmetric D3 acquisition mask is explicit.
    Independent value/Jacobian contractions validate array order and scale.
    """
    path = Path(path)
    b = _read_fchk_blocks(path)
    f = read_gaussian_fchk(path)
    w = np.asarray(f.harmonic_frequencies_cm)
    n, nx = len(w), 3*len(f.masses_amu)
    flags = np.asarray(b['Anharmonic displacement flags'], int)
    if sorted(flags.tolist()) != list(range(1, n+1)):
        raise ValueError('A complete unique normal-mode displacement archive is required')
    order = np.argsort(flags)
    v = np.asarray(b['Anharmonic displacements'], float).reshape(n, nx)[order]
    mass = np.repeat(f.masses_amu, 3)
    gram_error = float(np.max(abs((v*mass)@v.T-np.eye(n))))
    if gram_error > 1e-6:
        raise ValueError('Archived displacement vectors are not QMW orthonormal')
    raw = np.asarray(b['Dipole Moment num derivs'], float).reshape(2, n, 3)[:, order]
    jac = np.asarray(b['Dipole Derivatives'], float).reshape(nx, 3)
    d1 = (v@jac).T
    first_error = float(np.max(abs(d1-raw[0].T)))
    if first_error > 1e-6:
        raise ValueError('Archived dipole first derivative order or frame differs')
    jets = np.asarray(b['Dipole Derivatives num derivs'], float).reshape(2, n, nx, 3)[:, order]
    d2raw = np.einsum('ix,jxa->aij', v, jets[0])
    diagonal_error = float(np.max(abs(np.einsum('aij,ij->ai', d2raw, np.eye(n))-raw[1].T)))
    if diagonal_error > 1e-6:
        raise ValueError('Archived dipole second derivative order or scale differs')
    symmetry_error = float(np.max(abs(d2raw-d2raw.swapaxes(1, 2))))
    d2 = (d2raw+d2raw.swapaxes(1, 2))/2
    acquired = np.einsum('ix,jxa->aij', v, jets[1])
    d3 = np.full((3, n, n, n), np.nan)
    for i in range(n):
        for j in range(n):
            for indices in set(permutations((i, j, j))):
                d3[(slice(None),)+indices] = acquired[:, i, j]
    mask = np.isfinite(d3[0])
    return dict(frequencies_cm=w, displacements_qmw=v,
        first_qmw=d1, second_qmw=d2, third_qmw=d3,
        third_acquired_mask=mask, provenance=dict(
            schema='matrix.gaussian.anharmonic_dipole.v1',
            fchk_sha256=sha256(path.read_bytes()).hexdigest(),
            source=str(path), new_QM=0, modes=n,
            displacement_mass_gram_error=gram_error,
            first_derivative_closure_error=first_error,
            second_diagonal_closure_error=diagonal_error,
            second_derivative_symmetry_error=symmetry_error,
            third_coverage='All i,j,j; fully distinct triples not acquired',
            numerical_arrays='First N = first derivatives; next N = diagonal second derivatives',
            component_frame='Original FCHK Cartesian axes'))


__all__ = ['read_gaussian_anharmonic_dipole']
