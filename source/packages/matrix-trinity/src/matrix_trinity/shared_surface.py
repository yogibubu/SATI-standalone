"""SATI/TRINITY consumption of the owner-neutral molecular surface artifact."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from matrix_core.molecular_surface import MolecularSurface, read_molecular_surface
from matrix_gf.harmonic import CM_PER_HARTREE, HESSIAN_EIGENVALUE_TO_CM
from matrix_chem import BOHR_TO_ANGSTROM


@dataclass(frozen=True)
class TrinitySurfaceView:
    """A selected state with explicit interpolation and energy origin.

    Cartesian gradients/Hessians remain raw samples. They are not advertised as
    derivatives of the multilinear energy interpolant. No extrapolation, tail,
    polynomial field reconstruction or automatic electronic-state selection.
    """
    surface: MolecularSurface
    electronic_state: str
    reference_energy_hartree: float
    _interpolators: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.surface, MolecularSurface):
            raise TypeError('a validated MolecularSurface is required')
        self.surface.state(self.electronic_state)
        value = self.reference_energy_hartree
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError('an explicit finite energy origin in hartree is required')

    @property
    def provenance(self):
        return dict(surface_sha256=self.surface.sha256, surface_identifier=self.surface.identifier,
                    electronic_state=self.electronic_state, energy_reference=self.surface.energy_reference,
                    reference_energy_hartree=self.reference_energy_hartree,
                    cartesian_frame=self.surface.cartesian_frame,
                    evaluation='multilinear_interpolation_of_shared_samples',
                    extrapolation=False, physical_accuracy_certified=False)

    def tensor(self, channel):
        """Return the original typed channel; missing derivatives stay missing."""
        state = self.surface.state(self.electronic_state)
        if channel == 'geometries':
            return self.surface.geometries
        if channel in {'energies', 'gradients', 'hessians'}:
            value = getattr(state, channel)
            if value is None:
                raise KeyError(f'{channel} absent from electronic state {state.identifier}')
            return value
        return state.property(channel)

    def samples(self, channel):
        """Expose a read-only numerical copy in the channel's declared units."""
        tensor = self.tensor(channel)
        values = np.asarray(tensor.values, dtype=float).reshape(tensor.shape)
        values.setflags(write=False)
        return values

    def _interpolate(self, channel, coordinates):
        q = np.asarray(coordinates)
        if np.iscomplexobj(q):
            raise ValueError('surface coordinates must be real')
        q = np.asarray(q, dtype=float)
        if q.ndim != 2 or q.shape[1] != len(self.surface.coordinates) or not np.all(np.isfinite(q)):
            raise ValueError('finite (point, coordinate) queries required')
        q = q.copy()
        for i, axis in enumerate(self.surface.coordinates):
            if axis.period is not None:
                q[:, i] = axis.values[0] + np.remainder(q[:, i]-axis.values[0], axis.period)
        if channel not in self._interpolators:
            grids = [np.asarray(axis.values, dtype=float) for axis in self.surface.coordinates]
            values = self.samples(channel)
            for i, axis in enumerate(self.surface.coordinates):
                if axis.period is not None:
                    grids[i] = np.append(grids[i], grids[i][0]+axis.period)
                    values = np.concatenate((values, np.take(values, [0], axis=i)), axis=i)
            self._interpolators[channel] = RegularGridInterpolator(
                tuple(grids), values, method='linear', bounds_error=True,
            )
        return self._interpolators[channel](q)

    def energies_hartree(self, coordinates):
        return self._interpolate('energies', coordinates)

    def potential_cm(self, coordinates):
        return (self.energies_hartree(coordinates)-self.reference_energy_hartree)*CM_PER_HARTREE

    def property_values(self, name, coordinates):
        self.surface.state(self.electronic_state).property(name)
        return self._interpolate(name, coordinates)


def open_trinity_surface(source, *, electronic_state, reference_energy_hartree):
    """Open exactly the same verified artifact accepted by LOCK."""
    surface = source if isinstance(source, MolecularSurface) else read_molecular_surface(source)
    return TrinitySurfaceView(surface, electronic_state, reference_energy_hartree)


def solve_shared_surface_ground_state(view, **settings):
    """Use the existing SATI solver only for a verified linear modal embedding.

    Internal paths require their metric-aware owner. Grid interpolation and
    local box/HO errors require independent convergence; this call certifies
    neither the underlying electronic field nor a global bound state.
    """
    from .native_pes_variational import native_modal_displacements_angstrom
    from .vibronic_local_states import solve_local_potential_ground_state

    if not isinstance(view, TrinitySurfaceView):
        raise TypeError('a TrinitySurfaceView is required')
    surface = view.surface
    if surface.representation != 'LINEAR_CARTESIAN_MODAL':
        raise ValueError('internal paths cannot use the constant-diagonal modal kinetic model')
    frequencies = np.array([axis.frequency_cm for axis in surface.coordinates])
    displacement = np.asarray(surface.modal_displacements.values).reshape(surface.modal_displacements.shape)
    reference = np.asarray(surface.reference_geometry.values).reshape(surface.reference_geometry.shape)
    masses = np.asarray(surface.masses_amu)
    scales = BOHR_TO_ANGSTROM*HESSIAN_EIGENVALUE_TO_CM/np.sqrt(CM_PER_HARTREE*frequencies)
    modes = displacement*np.sqrt(masses)[None, :, None]/scales[:, None, None]
    native_modal_displacements_angstrom(frequencies, modes, masses)
    expected = np.broadcast_to(reference, surface.geometries.shape).copy()
    for i, axis in enumerate(surface.coordinates):
        shape = [1]*len(surface.coordinates)+[1, 1]
        shape[i] = len(axis.values)
        expected += np.asarray(axis.values).reshape(shape)*displacement[i]
    if not np.allclose(expected, view.samples('geometries'), atol=1e-9, rtol=0):
        raise ValueError('sampled geometries differ from the declared linear modal embedding')
    if 'potential_identifier' in settings:
        raise ValueError('the potential identity is supplied by the shared artifact')
    result = solve_local_potential_ground_state(
        frequencies, view.potential_cm, potential_identifier='shared-surface:'+surface.sha256,
        **settings,
    )
    return replace(result, diagnostics={**result.diagnostics, 'shared_surface': view.provenance,
                                       'direct_native_evaluations': False})
