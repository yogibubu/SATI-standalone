"""Checked warm spectral guesses and exact occupation-row embeddings."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RecycledSpectralGuesses:
    basis: np.ndarray
    frequencies_cm: np.ndarray
    energies_cm: np.ndarray
    eigenvectors: np.ndarray
    source_hamiltonian_sha256: str
    basis_archive_sha256: str


def load_recycled_guesses(basis_archive, spectrum_directory):
    """Load checked warm vectors without loading the obsolete source matrix.

    The source spectrum's identity and array checksums are retained. The basis
    archive is hashed and checked for compatibility. These data are guesses
    only: they supply no count or qualification for a new target operator.
    """
    from .vibronic_spectral_cache import load_gaussian_spectrum, PreparedGaussianSpectrum, _file_sha256
    spectrum=load_gaussian_spectrum(spectrum_directory)
    if not isinstance(spectrum,PreparedGaussianSpectrum):
        raise ValueError('a dense stored spectrum is required for recycled guesses')
    with np.load(basis_archive,allow_pickle=False) as data:
        basis=data['basis'];frequencies=data['frequencies_cm']
    if (basis.ndim!=2 or basis.dtype.kind not in 'iu' or np.any(basis<0)
            or len(np.unique(basis,axis=0))!=len(basis) or basis.shape[1]!=len(frequencies)
            or np.any(frequencies<=0) or not np.all(np.isfinite(frequencies))
            or len(basis)!=spectrum.eigenvectors.shape[0]):
        raise ValueError('source basis and stored guess dimensions are incompatible')
    basis.setflags(write=False);frequencies.setflags(write=False)
    return RecycledSpectralGuesses(basis,frequencies,spectrum.energies_cm,spectrum.eigenvectors,
        spectrum.hamiltonian_sha256,_file_sha256(basis_archive))


def embed_recycled_vectors(source_basis, target_basis, vectors):
    """Embed identical occupation rows, rejecting missing or duplicate states."""
    source, target, u = np.asarray(source_basis), np.asarray(target_basis), np.asarray(vectors)
    if (source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]
            or source.dtype.kind not in 'iu' or target.dtype.kind not in 'iu'
            or u.ndim != 2 or len(u) != len(source) or not np.all(np.isfinite(u))):
        raise ValueError('compatible integer occupation tables and finite vectors required')
    lookup = {tuple(row): i for i, row in enumerate(target)}
    if len(lookup) != len(target) or len({tuple(row) for row in source}) != len(source):
        raise ValueError('duplicate occupation rows')
    if any(tuple(row) not in lookup for row in source):
        raise ValueError('target basis does not contain every source occupation')
    result = np.zeros((len(target), u.shape[1]), dtype=np.result_type(u.dtype, np.float64))
    result[[lookup[tuple(row)] for row in source]] = u
    return result
