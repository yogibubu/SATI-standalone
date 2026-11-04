"""Direct inverse completion of one center and arbitrarily many partners.

Research component, not the general HDC solver. Given ALL prescribed poles,
unchanged diagonal entries and coupling phases, the arrowhead inverse problem
has a closed solution iff the poles interlace the distinct partner diagonals.
No eigenproblem, root finder, transport, fitted width or resonance cutoff is
used here. Partner-partner couplings and degenerate partner manifolds are not
covered. The inverse spectral construction is established matrix theory; its
choice as an HDC operator completion is an additional modeling assumption.
"""
from dataclasses import dataclass
from functools import cmp_to_key

import numpy as np


def _real_vector(value, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw) and np.any(raw.imag != 0):
        raise ValueError(f"{name} must be real")
    result = np.asarray(raw.real, dtype=np.longdouble)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite real vector")
    return result


def _readonly(value):
    value = np.array(value, copy=True)
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class InverseArrowheadResult:
    energy_bases_cm: np.ndarray
    energy_corrections_cm: np.ndarray
    effective_couplings_cm: np.ndarray
    effective_hamiltonian_cm: np.ndarray
    rotation: np.ndarray
    diagnostics: dict

    def transform_property(self, operator):
        """All supplied electrical derivatives use the same common functions."""
        value = np.asarray(operator, dtype=complex)
        size = len(self.rotation)
        if (value.ndim < 2 or value.shape[-2:] != (size, size)
                or not np.all(np.isfinite(value))):
            raise ValueError("finite scalar or stacked full-space operator required")
        return self.rotation.conj().T @ value @ self.rotation

    def projector(self, columns):
        columns = tuple(columns)
        size = len(self.rotation)
        if (not columns or len(set(columns)) != len(columns)
                or any(isinstance(i, (bool, np.bool_))
                       or not isinstance(i, (int, np.integer))
                       or i < 0 or i >= size for i in columns)):
            raise ValueError("distinct valid nonempty column indices required")
        frame = self.rotation[:, list(columns)]
        return frame @ frame.conj().T


def inverse_hdc_arrowhead(center_cm, partners_cm, coupling_phases,
                          energy_bases_cm, *, energy_corrections_cm=None):
    """Complete a fixed arrowhead from prescribed energies, not mechanical roots.

    Basis order is center, followed by the supplied partners. Pole/column order
    is the supplied energy order. coupling_phases may contain physical complex
    couplings: their phases alone are retained; their moduli are reconstructed
    collectively. An undefined phase with a positive reconstructed weight is
    rejected. Weak interlacing with exactly zero weights is supported.

    Energies may be supplied as base + correction. Differences use the two
    terms separately, so shifts smaller than an ulp of the base still encode
    weak couplings. This is numerical compensation, not a physical threshold.
    """
    center = _real_vector([center_cm], "center")[0]
    partners = _real_vector(partners_cm, "partners")
    bases = _real_vector(energy_bases_cm, "energy bases")
    corrections = (np.zeros_like(bases) if energy_corrections_cm is None else
                   _real_vector(energy_corrections_cm, "energy corrections"))
    m, n = len(partners), len(bases)
    phases = np.asarray(coupling_phases, dtype=complex)
    if (m < 1 or n != m+1 or corrections.shape != bases.shape
            or phases.shape != partners.shape or not np.all(np.isfinite(phases))):
        raise ValueError("one center, m partners/phases, and m+1 energies required")
    if len(np.unique(partners)) != m:
        raise ValueError("degenerate partner manifolds require a block completion")
    scale = max(np.max(abs(partners-center)), np.max(abs(bases-center)),
                np.max(abs(corrections)))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("finite nonzero spectral span required")
    # Compensated energy differences, before multiplication or division.
    to_partner = ((bases[:, None]-partners[None, :])
                  + corrections[:, None])/scale
    pole_difference = ((bases[:, None]-bases[None, :])
                       +(corrections[:, None]-corrections[None, :]))/scale
    if any(pole_difference[i, j] == 0 for i in range(n) for j in range(i)):
        raise ValueError("repeated poles require complete spectral multiplets")
    order = sorted(range(n), key=cmp_to_key(lambda i, j:
                   int(pole_difference[i, j] > 0)-int(pole_difference[i, j] < 0)))
    partner_order = np.argsort(partners)
    if any(to_partner[order[i], k] > 0 or to_partner[order[i+1], k] < 0
           for i, k in enumerate(partner_order)):
        raise ValueError("prescribed poles do not interlace partner diagonals; no Hermitian arrowhead completion")
    trace_defect = (np.sum(bases-center)-np.sum(partners-center)
                    +np.sum(corrections))/scale
    numerical_tolerance = 512*np.finfo(float).eps*n
    if abs(trace_defect) > numerical_tolerance:
        raise ValueError("prescribed poles do not preserve the declared diagonal trace")
    weights = np.empty(m, dtype=np.longdouble)
    for i in range(m):
        numerator = -to_partner[:, i]  # (d_i-E_j)/scale
        denominator = (partners[i]-np.delete(partners, i))/scale
        if np.any(numerator == 0):
            weights[i] = 0
        else:
            sign = -np.prod(np.sign(numerator))*np.prod(np.sign(denominator))
            if sign <= 0:
                raise ValueError("inverse spectral residue is not positive")
            weights[i] = np.exp(np.sum(np.log(abs(numerator)))
                                -np.sum(np.log(abs(denominator))))
            if weights[i] <= 0:
                raise FloatingPointError("positive inverse spectral weight underflowed")
    if not np.all(np.isfinite(weights)):
        raise FloatingPointError("inverse spectral weights exceed numerical range")
    amplitudes = np.sqrt(weights)
    direction = np.ones(m, dtype=complex)
    active_phase = abs(phases) > 0
    if np.any((~active_phase) & (weights > 0)):
        raise ValueError("positive reconstructed coupling requires a supplied phase")
    direction[active_phase] = phases[active_phase]/abs(phases[active_phase])
    z = np.asarray(amplitudes*scale, dtype=float)*direction
    if np.any((weights > 0) & (abs(z) == 0)) or not np.all(np.isfinite(z)):
        raise FloatingPointError("reconstructed coupling exceeds numerical range")
    u = np.zeros((n, n), complex)
    for j in range(n):
        zero = np.flatnonzero(to_partner[j] == 0)
        if len(zero):
            if weights[zero[0]] != 0:
                raise ValueError("a partner pole requires a zero coupling")
            u[1+zero[0], j] = 1
        else:
            magnitudes = amplitudes/to_partner[j]
            norm_scale = max(np.longdouble(1), np.max(abs(magnitudes)))
            if not np.isfinite(norm_scale):
                raise FloatingPointError("function coefficients exceed working precision")
            column = np.r_[np.longdouble(1)/norm_scale, magnitudes/norm_scale]
            column /= np.sqrt(np.sum(column*column))
            u[:, j] = np.asarray(column, float)*np.r_[1., direction]
    h = np.diag(np.asarray(np.r_[center, partners], float)).astype(complex)
    h[1:, 0], h[0, 1:] = z, z.conj()
    # Evaluate the residual in centered/scaled coordinates, not a huge origin.
    normalized_h = np.diag(np.asarray(np.r_[0., (partners-center)/scale], float)).astype(complex)
    normalized_h[1:, 0] = np.asarray(amplitudes, float)*direction
    normalized_h[0, 1:] = normalized_h[1:, 0].conj()
    poles = np.asarray(((bases-center)+corrections)/scale, float)
    unitarity = float(np.linalg.norm(u.conj().T @ u-np.eye(n)))
    residual = float(np.linalg.norm(normalized_h @ u-u*poles))
    if not np.all(np.isfinite(u)) or not np.isfinite(unitarity+residual) or max(unitarity, residual) > numerical_tolerance:
        raise FloatingPointError("inverse completion unresolved at working precision; no orthogonalization or spectral repair performed")
    return InverseArrowheadResult(
        _readonly(bases), _readonly(corrections), _readonly(z), _readonly(h), _readonly(u),
        dict(construction="inverse_arrowhead_with_prescribed_poles",
             physical_eigensolver_used=False, transport_used=False,
             general_graph_supported=False, degenerate_partners_supported=False,
             production_qualified=False,
             coupling_moduli_reconstructed=True,
             unitarity_frobenius=unitarity,
             normalized_prescribed_energy_residual=residual,
             normalized_trace_defect=float(trace_defect)))
