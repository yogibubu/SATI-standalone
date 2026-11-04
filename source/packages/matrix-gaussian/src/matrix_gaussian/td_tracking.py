"""MO-invariant Gaussian TD fingerprints in their checkpoint AO convention.

Printed amplitudes remain thresholded diagnostic data. Transforming them with
that point's orbitals removes spurious changes caused by MO signs and labels.
Checkpoint frames are aligned with their AO rotations before comparison.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

REPRESENTATION = 'gaussian-td-transition-density-ao-v1'


@lru_cache(maxsize=24)
def _wavefunction(source):
    from types import SimpleNamespace
    from matrix_apoc.state import electronic_state_from_gaussian_fchk
    from matrix_apoc.population import pyscf_molecule_from_electronic_state

    state = electronic_state_from_gaussian_fchk(source)
    if len(state.mo_coefficients) != 1:
        raise ValueError('Gaussian AO TD tracking requires a restricted checkpoint')
    molecule = pyscf_molecule_from_electronic_state(state)
    overlap = molecule.intor_symmetric('int1e_ovlp')
    coefficients = state.mo_coefficients[0]
    if not np.allclose(coefficients.T @ overlap @ coefficients,
                       np.eye(coefficients.shape[1]), atol=2e-6, rtol=0.):
        raise ValueError('Gaussian tracking checkpoint orbitals are not AO-metric orthonormal')
    data = SimpleNamespace(atnums=state.atomic_numbers, atcoords=state.coordinates_bohr,
                           mo=SimpleNamespace(coeffsa=coefficients), molecule=molecule)
    return data, overlap


def write_td_ao_fingerprints(log_path, fchk_path, output):
    from .electronic import gaussian_state_fingerprint_data

    state_data = gaussian_state_fingerprint_data(log_path)
    if state_data is None:
        raise ValueError('Gaussian log contains no TD amplitudes for AO tracking')
    ids, energies, vectors, representation = state_data
    if representation != 'gaussian-td-configurations-v1':
        raise ValueError('Gaussian AO TD tracking does not interpret EOM amplitudes')
    # GDV adds a response label (e.g. TDA-FC) to the FCHK method field.
    # IOData expects three tokens. Preserve all numeric/basis/orbital records
    # and the original checkpoint; normalize only this separate import copy.
    source = Path(fchk_path)
    lines = source.read_text().splitlines(keepends=True)
    header = lines[1].split()
    if len(header) > 3:
        lines[1] = f'{header[0]:10s}{header[1]:60s}{header[-1]}\n'
    imported = Path(output).with_name('gaussian_tracking.fchk').resolve()
    imported.write_text(''.join(lines))
    _wavefunction.cache_clear()
    data, _ = _wavefunction(str(imported))
    C = data.mo.coeffsa
    dimension = int(round(np.sqrt(vectors.shape[1])))
    if dimension * dimension != vectors.shape[1]:
        raise ValueError('Gaussian configuration fingerprint is not a square orbital array')
    densities = []
    for vector in vectors:
        amplitudes = vector.reshape(dimension, dimension)
        occupied, virtual = np.nonzero(amplitudes)
        if (np.any(occupied < 1) or np.any(virtual < 1)
                or np.any(occupied > C.shape[1]) or np.any(virtual > C.shape[1])):
            raise ValueError('Gaussian printed orbital index is outside the checkpoint MO space')
        # The log parser deliberately retains Gaussian's ONE-based indices.
        density = (C[:, occupied - 1] * amplitudes[occupied, virtual]) @ C[:, virtual - 1].T
        densities.append(density.ravel())
    target = Path(output)
    np.savez_compressed(target, state_ids=ids, excitation_energies_hartree=energies,
                        vectors=np.asarray(densities), representation=np.asarray([REPRESENTATION]),
                        state_source_file=np.asarray([str(imported)]))
    return target


def follow_td_ao_fingerprint(reference, candidates, **criteria):
    from pyscf import gto
    from matrix_apoc import ExcitedState, follow_excited_state

    candidates = tuple(candidates)
    if not candidates or any(item.representation != REPRESENTATION for item in candidates):
        raise ValueError('Gaussian AO tracking requires a consistent candidate representation')
    sources = {str(item.source) for item in candidates}
    if len(sources) != 1 or not reference.source:
        raise ValueError('Gaussian AO tracking requires one checkpoint per manifold')
    left, Sl = _wavefunction(str(reference.source))
    right, Sr = _wavefunction(sources.pop())
    if not np.array_equal(left.atnums, right.atnums):
        raise ValueError('Gaussian AO state tracking requires the same ordered atoms')
    # Gaussian may reorient each checkpoint. Align the ordered atoms and
    # rotate the AO tensors, not just their centres (p/d functions rotate too).
    lc, rc = left.atcoords.mean(axis=0), right.atcoords.mean(axis=0)
    u, _, vt = np.linalg.svd((right.atcoords - rc).T @ (left.atcoords - lc))
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = (u * sign) @ vt
    candidate_molecule = right.molecule.copy()
    candidate_molecule.set_geom_((right.atcoords - rc) @ rotation + lc, unit='Bohr')
    ao_rotation = right.molecule.ao_rotation_matrix(rotation.T)
    cross = gto.intor_cross('int1e_ovlp', left.molecule, candidate_molecule)
    Sr = candidate_molecule.intor_symmetric('int1e_ovlp')
    def state(item, size, transform=None):
        density = item.vector.reshape(size, size)
        if transform is not None:
            density = transform @ density @ transform.T
        return ExcitedState(item.state_id, density,
                            energy_hartree=item.energy_hartree, label=item.label)
    selected, match = follow_excited_state(state(reference, len(Sl)),
        tuple(state(item, len(Sr), ao_rotation) for item in candidates), cross,
        overlap_reference=Sl, overlap_candidate=Sr, **criteria)
    return next(item for item in candidates if item.state_id == selected.state_id), match
