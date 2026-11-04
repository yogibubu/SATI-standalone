"""Observable contractions for vibrational circular dichroism and ROA.

Inputs are *transition* tensors in one common Cartesian frame.  Electronic
response and vibrational transition-moment acquisition are separate steps.
The ROA convention follows Eqs. 5--9 and the SCP(180) row of Table 2 in
Bloino, Biczysko, and Barone, J. Phys. Chem. A 119, 11862 (2015).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from matrix_chem import BOHR_TO_ANGSTROM


_LEVI_CIVITA = np.zeros((3, 3, 3), dtype=float)
for _i, _j, _k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    _LEVI_CIVITA[_i, _j, _k] = 1.0
    _LEVI_CIVITA[_i, _k, _j] = -1.0

_GAUSSIAN_DQ_TO_LOWER_TRIANGLE = (0, 3, 1, 4, 5, 2)
_LOWER_TRIANGLE_INDEX = np.array(((0, 1, 3), (1, 2, 4), (3, 4, 5)))
# esu cm per atomic unit of electric dipole, and amu per electron mass.
_AU_ELECTRIC_DIPOLE_ESU_CM = 4.803204712e-10 * 5.29177210903e-9
_AMU_PER_ELECTRON_MASS = 1822.888486
_SPEED_OF_LIGHT_AU = 137.035999084
_VCD_GAUSSIAN_FACTOR = (
    _AU_ELECTRIC_DIPOLE_ESU_CM**2 * 1.0e44
    / (_SPEED_OF_LIGHT_AU * _AMU_PER_ELECTRON_MASS)
)


@dataclass(frozen=True)
class GaussianHarmonicChiroptical:
    """Harmonic observables reconstructed from full-precision FCHK tensors."""

    frequencies_cm: np.ndarray
    vcd_1e_minus_44_esu2_cm2: np.ndarray
    raman_ang4_per_amu: np.ndarray
    roa_scp180_1e4_ang4_per_amu: np.ndarray
    cid_scp180_1e4: np.ndarray
    alpha_g_1e4_ang4_per_amu: np.ndarray
    beta_g_1e4_ang4_per_amu: np.ndarray
    beta_a_1e4_ang4_per_amu: np.ndarray


def gaussian_roa_response_tensors(
    optical_rotation: np.ndarray,
    dipole_quadrupole: np.ndarray,
    *,
    incident_frequency_au: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Gaussian's stored G'/omega and 2A/3 to physical ROA tensors.

    The D-Q input has 18 Gaussian-order components per displacement: three
    blocks with (xx,yy,zz,xy,xz,yz).  The returned A is A[sigma,rho,eta].
    """
    optical = np.asarray(optical_rotation, dtype=float)
    dq = np.asarray(dipole_quadrupole, dtype=float)
    if (optical.shape[-2:] != (3, 3) or dq.shape != optical.shape[:-2] + (3, 6)
            or not np.all(np.isfinite(optical)) or not np.all(np.isfinite(dq))
            or not np.isfinite(incident_frequency_au) or incident_frequency_au <= 0):
        raise ValueError("Gaussian ROA tensors have invalid dimensions or frequency")
    dq_lower = dq[..., list(_GAUSSIAN_DQ_TO_LOWER_TRIANGLE)]
    a = 1.5 * dq_lower[..., :, _LOWER_TRIANGLE_INDEX]
    return incident_frequency_au * optical, a


def gaussian_harmonic_chiroptical_from_fchk(
    path: str | Path, *, incident_frequency_au: float | None = None,
) -> GaussianHarmonicChiroptical:
    """Reconstruct Gaussian harmonic VCD/ROA from the checkpoint tensors.

    This is a full-precision backend control.  Anharmonic bands instead use
    the SONIC property surfaces and second-order polyad transport.
    """
    from matrix_gaussian.fchk import _read_fchk_blocks

    blocks = _read_fchk_blocks(Path(path))
    stored_frequencies = np.asarray(
        blocks.get("Frequencies for FD properties", []), dtype=float,
    )

    if stored_frequencies.size != 1:
        raise ValueError("FCHK must contain one dynamic response frequency")
    stored_frequency = float(stored_frequencies[0])
    if incident_frequency_au is None:
        incident_frequency_au = stored_frequency
    elif not np.isclose(incident_frequency_au, stored_frequency, rtol=1e-7, atol=1e-9):
        raise ValueError("requested incident frequency differs from FCHK response")
    vib = np.asarray(blocks["Vib-E2"], dtype=float)
    modes_flat = np.asarray(blocks["Vib-Modes"], dtype=float)
    cart_count = 3 * int(blocks["Number of atoms"])
    mode_count = modes_flat.size // cart_count
    if (mode_count < 1 or modes_flat.size != mode_count * cart_count
            or vib.size < 2 * mode_count or np.any(vib[:2 * mode_count] <= 0)):
        raise ValueError("FCHK harmonic modes or reduced masses are invalid")
    mode_cart = modes_flat.reshape(mode_count, cart_count) / np.sqrt(
        vib[mode_count:2 * mode_count, None]
    )

    def modal(label: str, shape: tuple[int, ...]) -> np.ndarray:
        data = np.asarray(blocks[label], dtype=float)
        if data.size != cart_count * int(np.prod(shape)):
            raise ValueError(f"FCHK {label} has invalid dimensions")
        return np.einsum("mc,c...->m...", mode_cart, data.reshape(cart_count, *shape))

    electric = modal("Dipole Derivatives", (3,))
    axial = modal("AAT", (3,))
    alpha = modal("Derivative Alpha(-w,w)", (3, 3))
    optical = modal("Derivative FD Optical Rotation Tensor", (3, 3))
    dq = modal("Derivative D-Q polarizability", (3, 6))
    g_prime, a = gaussian_roa_response_tensors(
        optical, dq, incident_frequency_au=incident_frequency_au,
    )
    roa = roa_scp_backscattering(
        alpha, g_prime, a, incident_frequency_au=incident_frequency_au,
    )
    ang4 = BOHR_TO_ANGSTROM**4
    return GaussianHarmonicChiroptical(
        frequencies_cm=vib[:mode_count],
        vcd_1e_minus_44_esu2_cm2=(
            _VCD_GAUSSIAN_FACTOR * np.einsum("mi,mi->m", electric, axial)
        ),
        raman_ang4_per_amu=roa.raman_activity * ang4,
        roa_scp180_1e4_ang4_per_amu=roa.roa_activity * 4.0e4 * ang4,
        cid_scp180_1e4=roa.cid * 1.0e4,
        alpha_g_1e4_ang4_per_amu=(
            roa.isotropic_alpha_g * 1.0e4 * ang4 / _SPEED_OF_LIGHT_AU
        ),
        beta_g_1e4_ang4_per_amu=(
            roa.anisotropic_alpha_g * 1.0e4 * ang4 / _SPEED_OF_LIGHT_AU
        ),
        beta_a_1e4_ang4_per_amu=(
            roa.anisotropic_alpha_a * 1.0e4 * ang4 / _SPEED_OF_LIGHT_AU
        ),
    )

def gaussian_anharmonic_roa_surfaces_from_fchk(path: str | Path):
    """Read same-job single-mode ROA response stencils from Gaussian FCHK.

    Gaussian's D2 property records contain, for every displaced normal mode,
    a first and a second numerical derivative of each Cartesian response
    Jacobian.  The latter supplies the semi-diagonal cubic property tensor.
    Phases are fixed independently by the numerical electric-dipole D1
    records.  A single incident frequency is required to avoid mixing the
    static and dynamic response blocks.
    """
    from itertools import permutations

    from matrix_gaussian.fchk import _read_fchk_blocks
    from matrix_gf.harmonic import CM_PER_HARTREE
    from matrix_qm import HESSIAN_EIGENVALUE_TO_CM

    from .gvpt2_spectroscopy import SonicPropertyDerivatives, SonicROAPropertySurfaces

    blocks = _read_fchk_blocks(Path(path))
    incident = np.asarray(blocks["Frequencies for D2FD properties"], dtype=float)
    if incident.size != 1 or incident[0] <= 0:
        raise ValueError("one positive ROA incident frequency is required")
    cart_count = 3 * int(blocks["Number of atoms"])
    modes_flat = np.asarray(blocks["Vib-Modes"], dtype=float)
    if modes_flat.size % cart_count:
        raise ValueError("invalid Gaussian normal-mode dimensions")
    mode_count = modes_flat.size // cart_count
    vib = np.asarray(blocks["Vib-E2"], dtype=float)
    frequencies = vib[:mode_count]
    reduced_masses = vib[mode_count:2 * mode_count]
    if (reduced_masses.size != mode_count or np.any(reduced_masses <= 0)
            or np.any(frequencies <= 0)):
        raise ValueError("invalid Gaussian harmonic modes")
    gamma = np.sqrt(frequencies * CM_PER_HARTREE) / HESSIAN_EIGENVALUE_TO_CM
    cart_modes = modes_flat.reshape(mode_count, cart_count).T
    cart_modes = cart_modes / np.sqrt(reduced_masses)[None, :] / gamma[None, :]
    electric_cart = np.asarray(blocks["Dipole Derivatives"], dtype=float)
    electric_cart = electric_cart.reshape(cart_count, 3).T
    numerical_electric = np.asarray(blocks["Dipole Moment num derivs"], dtype=float)
    numerical_electric = numerical_electric.reshape(2 * mode_count, 3)[:mode_count]
    expected_electric = (electric_cart @ cart_modes).T * gamma[:, None]
    phase = np.sign(np.einsum("mi,mi->m", numerical_electric, expected_electric))
    relative_error = np.linalg.norm(
        numerical_electric - phase[:, None] * expected_electric,
    ) / max(np.linalg.norm(numerical_electric), 1e-30)
    if np.any(phase == 0) or relative_error > 1e-5:
        raise ValueError("Gaussian numerical normal-mode phases are inconsistent")

    def surface(central_label: str, d2_label: str, components: int):
        central = np.asarray(blocks[central_label], dtype=float)
        displaced = np.asarray(blocks[d2_label], dtype=float)
        if (central.size != cart_count * components
                or displaced.size != 2 * mode_count * cart_count * components):
            raise ValueError(f"incomplete ROA property stencil: {d2_label}")
        first = central.reshape(cart_count, components).T @ cart_modes
        numerical = displaced.reshape(2 * mode_count, cart_count, components)
        second_directional = np.stack([
            numerical[k].T @ cart_modes * phase[k] / gamma[k]
            for k in range(mode_count)
        ], axis=2)
        asymmetry = np.linalg.norm(
            second_directional - second_directional.swapaxes(1, 2),
        ) / max(np.linalg.norm(second_directional), 1e-30)
        if asymmetry > 1e-4:
            raise ValueError(f"inconsistent modal ROA Hessian: {d2_label}")
        second = 0.5 * (second_directional + second_directional.swapaxes(1, 2))
        third = np.zeros((components, mode_count, mode_count, mode_count))
        for k in range(mode_count):
            for i in range(mode_count):
                value = numerical[mode_count + k].T @ cart_modes[:, i]
                # The second directional derivative is even under reversal
                # of the displaced mode and carries no phase[k] factor.
                value /= gamma[k] ** 2
                for indices in set(permutations((i, k, k))):
                    third[(slice(None), *indices)] = value
        return SonicPropertyDerivatives(first, second, third)

    alpha = surface("Derivative Alpha(-w,w)", "D2 Alpha(-w,w)", 9)
    optical = surface(
        "Derivative FD Optical Rotation Tensor", "D2 FD Optical Rotation Tensor", 9,
    )
    dq = surface("Derivative D-Q polarizability", "D2 D-Q polarizability", 18)

    def convert_optical(values: np.ndarray) -> np.ndarray:
        return values * incident[0]

    def convert_dq(values: np.ndarray) -> np.ndarray:
        # Gaussian stores three six-component quadrupole blocks, whereas
        # TRINITY's observable uses A[sigma,rho,eta].
        leading = values.shape[1:]
        blocks_dq = np.moveaxis(values, 0, -1).reshape(*leading, 3, 6)
        _, a = gaussian_roa_response_tensors(
            np.zeros(leading + (3, 3)), blocks_dq,
            incident_frequency_au=float(incident[0]),
        )
        return np.moveaxis(a.reshape(*leading, 27), -1, 0)

    return SonicROAPropertySurfaces(
        alpha,
        SonicPropertyDerivatives(*(convert_optical(x) for x in
                                   (optical.first, optical.second, optical.third))),
        SonicPropertyDerivatives(*(convert_dq(x) for x in
                                   (dq.first, dq.second, dq.third))),
        float(incident[0]),
    )


@dataclass(frozen=True)
class ROABackscattering:
    """Far-from-resonance SCP(180) invariants and circular ratio.

    ``activity`` omits the common scattering prefactor K; ``cid`` is
    (right-left)/(right+left) in the convention of the cited equations.
    """

    isotropic_alpha_g: np.ndarray
    anisotropic_alpha_g: np.ndarray
    anisotropic_alpha_a: np.ndarray
    raman_activity: np.ndarray
    roa_activity: np.ndarray
    cid: np.ndarray


def vcd_rotatory_strength(
    electric_if: np.ndarray, magnetic_if: np.ndarray,
) -> np.ndarray:
    """Return Im[<i|mu|f> dot <f|m|i>] in input atomic units.

    Both arrays represent ``<i|operator|f>`` in the same frame and gauge.
    The magnetic array must include the electronic and nuclear contributions.
    This contraction does not compute the magnetic transition operator.
    """
    electric = np.asarray(electric_if, dtype=complex)
    magnetic = np.asarray(magnetic_if, dtype=complex)
    if electric.shape != magnetic.shape or electric.shape[-1:] != (3,):
        raise ValueError("VCD moments require matching (..., 3) arrays")
    if not np.all(np.isfinite(electric)) or not np.all(np.isfinite(magnetic)):
        raise ValueError("VCD moments must be finite")
    return np.imag(np.sum(electric * np.conjugate(magnetic), axis=-1))


def roa_scp_backscattering(
    alpha_if: np.ndarray,
    g_prime_if: np.ndarray,
    a_if: np.ndarray,
    *,
    incident_frequency_au: float,
    speed_of_light_au: float = 137.035999084,
) -> ROABackscattering:
    """Contract transition alpha, G' and A for off-resonant SCP(180).

    Tensor shapes are ``(...,3,3)``, ``(...,3,3)`` and ``(...,3,3,3)``.
    The A index order is ``A[sigma,rho,eta]`` of the cited Eq. 9.
    All tensors must use a common origin, gauge, incident frequency and
    atomic-unit convention.  The result cannot repair inconsistent inputs.
    """
    alpha = np.asarray(alpha_if, dtype=complex)
    g_prime = np.asarray(g_prime_if, dtype=complex)
    a = np.asarray(a_if, dtype=complex)
    if (alpha.shape != g_prime.shape or alpha.shape[-2:] != (3, 3)
            or a.shape != alpha.shape[:-2] + (3, 3, 3)):
        raise ValueError("ROA tensors require matching (...,3,3) and (...,3,3,3) shapes")
    if not all(np.all(np.isfinite(value)) for value in (alpha, g_prime, a)):
        raise ValueError("ROA tensors must be finite")
    if (not np.isfinite(incident_frequency_au) or incident_frequency_au <= 0
            or not np.isfinite(speed_of_light_au) or speed_of_light_au <= 0):
        raise ValueError("incident frequency and light speed must be positive")
    trace_alpha = np.trace(alpha, axis1=-2, axis2=-1)
    trace_g = np.trace(g_prime, axis1=-2, axis2=-1)
    alpha_g = np.real(trace_alpha * np.conjugate(trace_g)) / 9.0
    beta_alpha = np.real(
        3.0 * np.einsum("...ij,...ij->...", alpha, np.conjugate(alpha))
        - trace_alpha * np.conjugate(trace_alpha)
    ) / 2.0
    beta_g = np.real(
        3.0 * np.einsum("...ij,...ij->...", alpha, np.conjugate(g_prime))
        - trace_alpha * np.conjugate(trace_g)
    ) / 2.0
    beta_a = (incident_frequency_au / 2.0) * np.real(
        np.einsum("...te,tsr,...sre->...", alpha, _LEVI_CIVITA, np.conjugate(a))
    )
    raman = 45.0 * np.real(trace_alpha * np.conjugate(trace_alpha)) / 9.0 + 7.0 * beta_alpha
    roa = (2.0 / speed_of_light_au) * (12.0 * beta_g + 4.0 * beta_a)
    cid = np.divide(roa, raman, out=np.full_like(roa, np.nan), where=raman > 0)
    return ROABackscattering(alpha_g, beta_g, beta_a, raman, roa, cid)


__all__ = [
    "GaussianHarmonicChiroptical", "ROABackscattering",
    "gaussian_harmonic_chiroptical_from_fchk",
    "gaussian_anharmonic_roa_surfaces_from_fchk", "gaussian_roa_response_tensors",
    "roa_scp_backscattering", "vcd_rotatory_strength",
]
