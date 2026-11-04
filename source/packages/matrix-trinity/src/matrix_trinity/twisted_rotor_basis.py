"""Permutation-aware oscillator basis and rotor-sector restriction.

Symmetrization constructs a *basis reference* only. The original even and odd
force/kinetic coefficient fields must both remain in the Hamiltonian. The odd
fields carry odd rotor Fourier harmonics and change transverse parity.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator
from .covariance_adapted_bath import covariance_adapted_frame


def exchange_adapted_frame(kinetic, force, exchange, active_count):
    g, h, s = map(lambda x: np.asarray(x, float), (kinetic, force, exchange))
    if (g.shape != h.shape or s.shape != g.shape or g.ndim != 2
            or not np.allclose(s.T@s, np.eye(len(s)), atol=1e-10)
            or not np.allclose(s@s, np.eye(len(s)), atol=1e-10)
            or not np.allclose(s[:active_count], np.eye(len(s))[:active_count], atol=1e-10)):
        raise ValueError('Involutive orthogonal exchange preserving every active coordinate required')
    ge, he = .5*(g+s@g@s.T), .5*(h+s.T@h@s)
    frame = covariance_adapted_frame(ge, he, active_count)
    q, p = frame['coordinate_map'], frame['momentum_map']
    action = np.linalg.solve(q, s@q)
    if not np.allclose(action.T@action, np.eye(len(action)), atol=1e-8):
        raise ValueError('Exchange is not orthogonal in the symmetrized Gaussian basis')
    value, rotation_bath = eigh(.5*(action[active_count:, active_count:]+action[active_count:, active_count:].T))
    signs = np.where(value < 0, -1, 1)
    if max(abs(value-signs)) > 1e-7:
        raise ValueError('Transverse oscillator exchange has non-involutive eigenvalues')
    rotation = np.eye(len(s)); rotation[active_count:, active_count:] = rotation_bath
    frame['coordinate_map'] = q@rotation; frame['momentum_map'] = p@rotation
    frame['number_force_cm'] = rotation.T@frame['number_force_cm']@rotation
    frame['active_map'] = frame['coordinate_map'][:active_count, :active_count]
    expected = np.diag(np.r_[np.ones(active_count), signs])
    error = float(abs(s@frame['coordinate_map']-frame['coordinate_map']@expected).max())
    if error > 1e-8:
        raise ValueError('Active-preserving oscillator exchange diagonalization failed')
    frame.update(bath_exchange_signs=signs, kinetic_basis=ge, force_basis=he,
                 parent_kinetic_odd=g-ge, parent_force_odd=h-he,
                 exchange_closure_error=error,
                 basis_symmetrization_scope='Trial basis only; no field terms may be dropped')
    return frame


def restrict_exchange_sector(operator, modes, active_occupations, bath_occupations, bath_signs, sector):
    """Restrict by (-1)^m times the complete transverse exchange parity."""
    modes = np.asarray(modes, int); occupations = np.asarray(bath_occupations, int)
    signs = np.asarray(bath_signs, int)
    if sector not in (-1, 1) or occupations.shape[1] != len(signs) or not np.all(np.isin(signs, [-1, 1])):
        raise ValueError('Explicit physical exchange sector and oscillator signs required')
    na, nb = len(active_occupations), len(occupations)
    parity = np.prod(signs[None, :]**occupations, axis=1)
    selected = np.flatnonzero(np.broadcast_to(
        ((-1.)**modes[:, None, None])*parity[None, None, :], (len(modes), na, nb)).ravel() == sector)
    if operator.shape != (len(modes)*na*nb,)*2 or len(selected) < 2:
        raise ValueError('Fourier tensor dimension or exchange-sector rank is invalid')
    def embed(vector):
        full = np.zeros(operator.shape[0], complex); full[selected] = np.asarray(vector).reshape(-1); return full
    def action(vector):
        return (operator@embed(vector))[selected]
    restricted = LinearOperator((len(selected), len(selected)), matvec=action, dtype=complex)
    if hasattr(operator, 'diagonal_values'):
        restricted.diagonal_values = operator.diagonal_values[selected]
    return dict(operator=restricted, indices=selected, embed=embed,
                bath_parities=parity,
                sector_rule='(-1)^rotor_Fourier_index times product(bath_sign^occupation)')


__all__ = ['exchange_adapted_frame', 'restrict_exchange_sector']
