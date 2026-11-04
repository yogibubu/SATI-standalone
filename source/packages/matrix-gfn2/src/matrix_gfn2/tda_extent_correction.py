"""Orbital size diagnostics, posteriori energies and a ground-reference K filter.

Only AO electronic dipoles and r^2 moments are required. No basis primitives,
model-potential integrals, ERIs, response solves or nuclear derivatives occur.
The inverse-radius model has an explicit empirical scale and no default fit.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .electronic_state import _readonly_array
from .imported_tda import ImportedTDAReference, ImportedTDAResult
from .sf_tda import HARTREE_TO_EV


@dataclass(frozen=True)
class OrbitalExtentDiagnostics:
    reference_model_id: str
    centroids_bohr: np.ndarray
    variances_bohr2: np.ndarray
    radii_bohr: np.ndarray

    def __post_init__(self):
        for name in ('centroids_bohr', 'variances_bohr2', 'radii_bohr'):
            object.__setattr__(self, name, _readonly_array(getattr(self, name)))
        n = len(self.radii_bohr)
        if self.centroids_bohr.shape != (n, 3) or self.variances_bohr2.shape != (n,):
            raise ValueError('orbital extent arrays have incompatible shapes')
        if np.any(self.variances_bohr2 <= 0) or not np.allclose(
                self.radii_bohr**2, self.variances_bohr2, atol=1e-10, rtol=1e-10):
            raise ValueError('orbital extents require positive consistent variances')


@dataclass(frozen=True)
class TDAExtentDiagnostics:
    reference_model_id: str
    hole_centroids_bohr: np.ndarray
    particle_centroids_bohr: np.ndarray
    hole_radii_bohr: np.ndarray
    particle_radii_bohr: np.ndarray
    particle_to_hole_radius_ratio: np.ndarray
    inverse_radius_excess_bohr_inverse: np.ndarray

    def __post_init__(self):
        arrays = ('hole_centroids_bohr', 'particle_centroids_bohr', 'hole_radii_bohr',
                  'particle_radii_bohr', 'particle_to_hole_radius_ratio',
                  'inverse_radius_excess_bohr_inverse')
        for name in arrays:
            object.__setattr__(self, name, _readonly_array(getattr(self, name)))
        n = len(self.hole_radii_bohr)
        if any(getattr(self, name).shape != (n, 3) for name in arrays[:2]) or any(
                getattr(self, name).shape != (n,) for name in arrays[2:]):
            raise ValueError('TDA extent arrays have incompatible shapes')
        h, p = self.hole_radii_bohr, self.particle_radii_bohr
        if np.any(h <= 0) or np.any(p <= 0):
            raise ValueError('TDA extents require positive radii')
        if not np.allclose(self.particle_to_hole_radius_ratio, p/h, atol=1e-10, rtol=1e-10):
            raise ValueError('extent ratio disagrees with radii')
        expected = np.maximum(1/h - 1/p, 0)
        if not np.allclose(self.inverse_radius_excess_bohr_inverse, expected, atol=1e-10, rtol=1e-10):
            raise ValueError('inverse-radius excess disagrees with radii')


@dataclass(frozen=True)
class TDAExtentCorrectionParameters:
    """Dimensionless nonnegative coefficient; explicitly supplied, never fitted here."""
    scale: float

    def __post_init__(self):
        if isinstance(self.scale, (bool, np.bool_)) or not np.isfinite(self.scale) or self.scale < 0:
            raise ValueError('extent correction scale must be finite and nonnegative')


@dataclass(frozen=True)
class TDAExtentEnergyCorrection:
    reference_model_id: str
    parameters: TDAExtentCorrectionParameters
    diagnostics: TDAExtentDiagnostics
    state_indices: np.ndarray
    original_excitation_energies_hartree: np.ndarray
    corrected_excitation_energies_hartree: np.ndarray
    corrections_ev: np.ndarray

    def __post_init__(self):
        if not isinstance(self.parameters, TDAExtentCorrectionParameters):
            raise TypeError('explicit extent parameters are required')
        if not isinstance(self.diagnostics, TDAExtentDiagnostics) or self.reference_model_id != self.diagnostics.reference_model_id:
            raise ValueError('extent correction and diagnostics must share a reference')
        before = _readonly_array(self.original_excitation_energies_hartree)
        if before.ndim != 1 or len(before) != len(self.diagnostics.hole_radii_bohr):
            raise ValueError('energies must match the extent diagnostics')
        indices = _indices(self.state_indices, len(before))
        after = _readonly_array(self.corrected_excitation_energies_hartree)
        shifts = _readonly_array(self.corrections_ev)
        if after.shape != before.shape or shifts.shape != before.shape or not np.allclose(
                after, before + shifts/HARTREE_TO_EV, atol=1e-12, rtol=0):
            raise ValueError('inconsistent corrected energies')
        mask = np.ones(len(before), dtype=bool); mask[indices] = False
        if np.any(shifts[mask] != 0) or not np.array_equal(after[mask], before[mask]):
            raise ValueError('unselected energies must be unchanged')
        for name, array in (('state_indices', indices), ('original_excitation_energies_hartree', before),
                            ('corrected_excitation_energies_hartree', after), ('corrections_ev', shifts)):
            object.__setattr__(self, name, array)


def _indices(values, count):
    raw = np.asarray(values)
    if raw.ndim != 1 or (raw.size and (raw.dtype.kind not in 'iu' or np.any(raw < 0) or np.any(raw >= count))):
        raise ValueError('state indices must be zero-based integer indices into the result')
    if len(np.unique(raw)) != len(raw):
        raise ValueError('state indices must be unique')
    return _readonly_array(raw, dtype=np.int64)


def _moments(reference, coefficients):
    if reference.dipole_ao is None or reference.radial_second_moment_ao is None:
        raise ValueError('extent diagnostics require electronic dipoles and radial_second_moment_ao')
    # The electronic dipole is -r; r^2 is supplied with positive sign.
    position = -np.einsum('mi,dmn,nj->dij', coefficients, reference.dipole_ao,
                          coefficients, optimize=True)
    radial = coefficients.T @ reference.radial_second_moment_ao @ coefficients
    return position, radial


def _radius(centroid, second):
    variance = second - np.dot(centroid, centroid)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError('moments give a nonpositive spread; check units, AO ordering and common origin')
    return np.sqrt(variance)


def imported_orbital_extents(reference: ImportedTDAReference) -> OrbitalExtentDiagnostics:
    """All canonical MO spreads sqrt(<r^2>-|<r>|^2), in bohr."""
    position, radial = _moments(reference, reference.coefficients)
    centroids = np.diagonal(position, axis1=1, axis2=2).T
    radii = np.array([_radius(c, s) for c, s in zip(centroids, np.diag(radial))])
    return OrbitalExtentDiagnostics(reference.model_id, centroids, radii**2, radii)


def _virtual_extent_filter(reference, occupied_coefficients, virtual_coefficients, beta):
    """Ground-reference-only filter, covariant under virtual basis rotations.

    F(q)=[1+beta*max(q/q0-1,0)]^(-1/4). Two virtual vertices then
    attenuate the diagonal K attraction as 1/radius at large extent.
    """
    positions, _ = _moments(reference, occupied_coefficients)
    center = np.trace(positions, axis1=1, axis2=2)/occupied_coefficients.shape[1]
    centered = (reference.radial_second_moment_ao
        + 2*np.einsum('d,dmn->mn', center, reference.dipole_ao)
        + (center@center)*reference.overlap)
    occupied = occupied_coefficients.T @ centered @ occupied_coefficients
    virtual = virtual_coefficients.T @ centered @ virtual_coefficients
    bound = float(np.linalg.eigvalsh((occupied+occupied.T)/2)[-1])
    values, vectors = np.linalg.eigh((virtual+virtual.T)/2)
    if bound <= 0 or values[0] < -1e-8:
        raise ValueError('centered r^2 must define positive spatial extents')
    weights = (1+beta*np.maximum(values/bound-1, 0))**-.25
    filtering = (vectors*weights) @ vectors.T
    return (filtering+filtering.T)/2, bound


def imported_tda_extents(reference: ImportedTDAReference, result: ImportedTDAResult) -> TDAExtentDiagnostics:
    """Full hole/particle density moments, including off-diagonal coherences.

    Supports complete canonical rectangles and projected virtual spaces.
    State radii are invariant under orbital basis rotations with transformed X.
    Individual canonical orbital radii need not be invariant under such rotations.
    """
    if not isinstance(result, ImportedTDAResult) or not result.converged or result.reference_model_id != reference.model_id:
        raise ValueError('a converged TDA result on the supplied reference is required')
    occupied = np.flatnonzero(reference.occupations > 1)
    virtual = result.retained_virtual_orbitals
    if result.virtual_space is None:
        if virtual.ndim != 1 or not len(virtual) or np.any(virtual < 0) or np.any(virtual >= len(reference.occupations)):
            raise ValueError('invalid retained virtual indices')
        if len(np.unique(virtual)) != len(virtual) or np.any(reference.occupations[virtual] > 1):
            raise ValueError('retained virtuals must be unique unoccupied orbitals')
        cv = reference.coefficients[:, virtual]
    else:
        space = result.virtual_space; space.validate_reference(reference)
        if not np.array_equal(virtual, np.arange(space.dimension)):
            raise ValueError('projected virtual labels disagree with their space')
        cv = reference.coefficients[:, space.parent_virtual_orbitals] @ space.rotation
    no, nv = len(occupied), cv.shape[1]
    if not np.array_equal(result.occupied_orbitals, np.repeat(occupied, nv)) or not np.array_equal(
            result.virtual_orbitals, np.tile(virtual, no)):
        raise ValueError('extent diagnostics require the complete TDA configuration rectangle')
    amplitudes = np.asarray(result.amplitudes)
    if amplitudes.shape != (len(result.excitation_energies_hartree), no*nv) or not np.all(np.isfinite(amplitudes)):
        raise ValueError('invalid TDA amplitudes')
    if not np.allclose(np.sum(amplitudes**2, axis=1), 1, atol=1e-7, rtol=0):
        raise ValueError('TDA amplitudes must have unit norm')
    hm, h2 = _moments(reference, reference.coefficients[:, occupied])
    pm, p2 = _moments(reference, cv)
    hc, pc, hr, pr = [], [], [], []
    for x in amplitudes.reshape(-1, no, nv):
        h, p = x@x.T, x.T@x
        h /= np.trace(h); p /= np.trace(p)
        ch = np.einsum('ij,dji->d', h, hm); cp = np.einsum('ij,dji->d', p, pm)
        hc.append(ch); pc.append(cp)
        hr.append(_radius(ch, np.einsum('ij,ji->', h, h2)))
        pr.append(_radius(cp, np.einsum('ij,ji->', p, p2)))
    hr, pr = np.array(hr), np.array(pr)
    return TDAExtentDiagnostics(reference.model_id, np.array(hc), np.array(pc), hr, pr,
                                pr/hr, np.maximum(1/hr-1/pr, 0))


def correct_imported_tda_extent_energies(reference: ImportedTDAReference, result: ImportedTDAResult, *,
        parameters: TDAExtentCorrectionParameters, state_indices) -> TDAExtentEnergyCorrection:
    """Explicit selected roots only; indices are zero-based and may be empty.

    delta omega / Ha = scale * a_response * max(1/sigma_h - 1/sigma_p, 0).
    No automatic Rydberg assignment or coefficient selection takes place.
    Full energy arrays preserve root IDs; eigenvectors, strengths and all
    original input objects remain unchanged. No geometry optimization uses this.
    """
    if not isinstance(parameters, TDAExtentCorrectionParameters):
        raise TypeError('explicit TDAExtentCorrectionParameters are required')
    if result.parameters.extent_exchange_damping not in (None, 0):
        raise ValueError('choose an extent correction in K or a posteriori; their combination is not defined')
    diagnostics = imported_tda_extents(reference, result)
    before = result.excitation_energies_hartree
    selected = _indices(state_indices, len(before))
    shifts = np.zeros(len(before))
    shifts[selected] = (parameters.scale * result.parameters.exact_exchange
                       * diagnostics.inverse_radius_excess_bohr_inverse[selected])
    return TDAExtentEnergyCorrection(reference.model_id, parameters, diagnostics, selected,
        before, before+shifts, shifts*HARTREE_TO_EV)


__all__ = ['OrbitalExtentDiagnostics', 'TDAExtentDiagnostics', 'TDAExtentCorrectionParameters',
           'TDAExtentEnergyCorrection', 'imported_orbital_extents', 'imported_tda_extents',
           'correct_imported_tda_extent_energies']
