"""Portable ORCA Hessian export from mass-independent Cartesian fields."""
from pathlib import Path
import numpy as np
from matrix_chem.topology.elements import atomic_symbol
from matrix_qm import cartesian_normal_modes_from_hessian


def write_orca_hessian(path, *, hessian_hartree_per_bohr2, masses_amu,
                       coordinates_bohr, atomic_numbers, energy_hartree):
    """Write a molecular .hess without changing curvature signs or geometry.

    Imaginary vibrational frequencies are retained. Selection or projection
    policies belong to the caller. Cartesian Hessians remain unweighted.
    """
    h = np.asarray(hessian_hartree_per_bohr2, dtype=float)
    m = np.asarray(masses_amu, dtype=float)
    q = np.asarray(coordinates_bohr, dtype=float)
    z = np.asarray(atomic_numbers)
    n = 3 * len(m)
    if (not len(m) or h.shape != (n, n) or q.shape != (len(m), 3)
            or z.shape != m.shape or not np.all(np.isfinite(h))
            or not np.all(np.isfinite(q)) or not np.all(np.isfinite(m))
            or np.any(m <= 0) or not np.isfinite(energy_hartree)
            or not np.all(np.isfinite(z)) or np.any(z != z.astype(int))
            or not np.allclose(h, h.T, rtol=1e-12, atol=1e-13)):
        raise ValueError('Finite symmetric Cartesian field and matching atoms/masses required')
    symbols = [atomic_symbol(int(value)) for value in z]
    normal = cartesian_normal_modes_from_hessian(h, m, q)
    lines = ['$orca_hessian_file', '', '$act_energy', f'{energy_hartree:.17g}',
             '', '$hessian', str(n)]
    for start in range(0, n, 5):
        indices = range(start, min(start + 5, n))
        lines.append(' '.join(f'{j:20d}' for j in indices))
        lines.extend(f'{i:5d} ' + ' '.join(f'{h[i, j]:24.16e}' for j in indices)
                     for i in range(n))
    frequencies = np.r_[np.zeros(normal.external_rank), normal.frequencies_cm]
    lines += ['', '$vibrational_frequencies', str(n)]
    lines += [f'{i:5d} {value:24.16e}' for i, value in enumerate(frequencies)]
    lines += ['', '$atoms', str(len(m))]
    for symbol, mass, xyz in zip(symbols, m, q):
        lines.append(f'{symbol} {mass:.16g} ' + ' '.join(f'{x:.16g}' for x in xyz))
    lines += ['', '$end']
    target = Path(path)
    target.write_text('\n'.join(lines) + '\n')
    return target
