"""Fast Truhlar reference-structure multistructural rotor (MS-RS/MSR)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


HC_OVER_K = 1.438776877  # K cm
KINETIC_CM_AMU_A2 = 16.857629206


def one_term_fourier_barrier(
    periodicity: int,
    curvature_cm_rad2: float,
    *,
    barrier_cm: float | None = None,
) -> float:
    """Determine a one-term Fourier barrier from local data.

    For ``V(phi)=Vb/2*(1-cos(m*phi))``, the minimum curvature is
    ``V''(0)=Vb*m**2/2``.  Thus MS-RS needs only the reference minimum and
    its second derivative.  One additional energy point at the barrier can
    be supplied to replace the curvature estimate; no torsional scan is
    required.  The minimum energy itself is the zero of the returned model.
    """
    if periodicity < 1 or curvature_cm_rad2 < 0:
        raise ValueError("invalid Fourier curvature")
    if barrier_cm is not None:
        if barrier_cm < 0:
            raise ValueError("barrier_cm must be non-negative")
        return float(barrier_cm)
    return float(2.0 * curvature_cm_rad2 / periodicity**2)


def msr_local_rotor(
    minimum_cm: float,
    periodicity: int,
    curvature_cm_rad2: float,
    moment_amu_a2: float,
    *,
    barrier_cm: float | None = None,
    phase_rad: float = 0.0,
    basis_size: int = 511,
    sector: str = "full",
) -> dict[str, object]:
    """Build and solve the MS-RS rotor from local reference data only.

    ``minimum_cm`` fixes the energy origin, ``curvature_cm_rad2`` fixes the
    one-term Fourier coefficient, and ``moment_amu_a2`` is held constant.  A
    single barrier value may replace the curvature-derived barrier.  The
    returned levels include the reference minimum energy.
    """
    fitted_barrier = one_term_fourier_barrier(
        periodicity, curvature_cm_rad2, barrier_cm=barrier_cm
    )
    levels = periodic_rotor_levels(
        moment_amu_a2, periodicity, fitted_barrier, basis_size=basis_size,
        sector=sector
    )
    tunneling_splitting = float(levels[1] - levels[0])
    fundamental = float(levels[periodicity] - levels[0])
    zpe = float(levels[0])
    return {
        "minimum_cm": float(minimum_cm),
        "periodicity": int(periodicity),
        "curvature_cm_rad2": float(curvature_cm_rad2),
        "moment_amu_a2": float(moment_amu_a2),
        "fourier_coefficient_cm": float(fitted_barrier / 2.0),
        "barrier_cm": fitted_barrier,
        "phase_rad": float(phase_rad),
        "levels_cm": (float(minimum_cm) + levels).tolist(),
        "tunneling_splitting_cm": tunneling_splitting,
        "fundamental_cm": fundamental,
        "zpe_cm": zpe,
        "ground_state_cm": float(minimum_cm + zpe),
    }


def periodic_rotor_levels(
    moment_amu_a2: float,
    periodicity: int,
    barrier_cm: float,
    *,
    phi_rad: Sequence[float] | None = None,
    potential_cm: Sequence[float] | None = None,
    basis_size: int = 31,
    sector: str = "full",
) -> np.ndarray:
    """Return MS-RS one-dimensional periodic-rotor levels in cm^-1.

    The rotor is defined at one reference structure.  A supplied periodic scan
    is used directly; otherwise the fast one-term cosine barrier is used.
    Energies are shifted so the lowest level is zero.  No normal-mode
    projection, soft-soft diagonalization, or Cartesian redundancy is used.
    """
    if moment_amu_a2 <= 0 or periodicity < 1 or basis_size < 3:
        raise ValueError("invalid rotor parameters")
    if basis_size % 2 == 0:
        raise ValueError("basis_size must be odd")
    if sector not in {"even", "odd", "full"}:
        raise ValueError("sector must be 'even', 'odd', or 'full'")
    n = (np.arange(0, basis_size, dtype=float) if sector != "full"
         else np.arange(-(basis_size // 2), basis_size // 2 + 1, dtype=float))
    hamiltonian = np.diag(KINETIC_CM_AMU_A2 * n * n / moment_amu_a2).astype(complex)
    if phi_rad is None or potential_cm is None:
        grid = np.linspace(0.0, 2.0 * math.pi, 4 * basis_size, endpoint=False)
        values = 0.5 * barrier_cm * (1.0 - np.cos(periodicity * grid))
    else:
        grid = np.asarray(phi_rad, dtype=float) % (2.0 * math.pi)
        values = np.asarray(potential_cm, dtype=float)
        if grid.ndim != 1 or values.shape != grid.shape or grid.size < 4:
            raise ValueError("phi_rad and potential_cm must be matching 1D arrays")
        order = np.argsort(grid)
        grid, values = grid[order], values[order]
        values = values - values.min()
    if sector == "full":
        basis = np.exp(1j * np.outer(grid, n)) / math.sqrt(grid.size)
    elif sector == "even":
        basis = np.cos(np.outer(grid, n)) * math.sqrt(2.0 / grid.size)
        basis[:, 0] = 1.0 / math.sqrt(grid.size)
    else:
        basis = np.sin(np.outer(grid, n[1:])) * math.sqrt(2.0 / grid.size)
    hamiltonian += basis.conj().T @ (values[:, None] * basis)
    levels = np.linalg.eigvalsh(hamiltonian).real
    # The potential minimum is zero, but retain the absolute rotor ZPE.
    return np.sort(levels)


def periodic_rotor_levels_variable_inertia(
    phi_rad, potential_cm, inertia_amu_a2, *, basis_size=121,
):
    """One-dimensional periodic DVR on the arc of a positive angular metric.

    Inputs are dense values generated from an already fitted analytic model,
    including BOTH endpoints of [0, 2pi]. No source potential is evaluated.
    The Laplace--Beltrami rotor becomes a constant-inertia rotor on the
    normalized arc. Retain the absolute potential offset exactly once.
    """
    from scipy.integrate import cumulative_trapezoid
    from scipy.interpolate import PchipInterpolator
    phi=np.asarray(phi_rad,float);v=np.asarray(potential_cm,float)
    inertia=np.asarray(inertia_amu_a2,float)
    if (phi.ndim!=1 or len(phi)<4*basis_size+1 or v.shape!=phi.shape
            or inertia.shape!=phi.shape or np.any(np.diff(phi)<=0)
            or np.any(inertia<=0) or abs(phi[0])>1e-12
            or abs(phi[-1]-2*np.pi)>1e-12
            or not all(np.all(np.isfinite(a)) for a in (phi,v,inertia))
            or not np.isclose(v[0],v[-1],rtol=1e-10,atol=1e-8)
            or not np.isclose(inertia[0],inertia[-1],rtol=1e-10,atol=1e-12)):
        raise ValueError('resolved periodic fitted potential and positive angular inertia required')
    arc=cumulative_trapezoid(np.sqrt(inertia),phi,initial=0.)
    length=float(arc[-1]);moment=(length/(2*np.pi))**2
    theta=np.linspace(0.,2*np.pi,len(phi)-1,endpoint=False)
    source_phi=PchipInterpolator(2*np.pi*arc/length,phi)(theta)
    values=PchipInterpolator(phi,v)(source_phi)
    levels=periodic_rotor_levels(moment,1,float(np.ptp(values)),phi_rad=theta,
        potential_cm=values,basis_size=basis_size)+float(np.min(values))
    return dict(levels_cm=levels,effective_moment_amu_A2=moment,
        variable_metric_arclength=True,absolute_potential_offset_restored=True)


def msr_reference_structure(
    rotors: Sequence[dict[str, float | int]],
    *,
    temperature_K: float = 298.15,
    basis_size: int = 511,
    sector: str = "full",
) -> dict[str, object]:
    """Evaluate independent SONIC rotors and their fast MS-RS product.

    Each rotor dictionary requires ``identifier``, ``moment_amu_a2``,
    ``periodicity`` and ``curvature_cm_rad2`` and may contain one optional
    ``barrier_cm`` point, ``symmetry_number``, ``phi_rad`` and ``potential_cm``.
    The returned ``q_soft`` is the product
    of the one-dimensional rotor partition functions divided by the specified
    symmetry numbers.
    """
    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    q_soft = 1.0
    zpe_soft = 0.0
    output = []
    beta = HC_OVER_K / temperature_K
    for rotor in rotors:
        if "curvature_cm_rad2" not in rotor:
            raise KeyError("MS-RS requires local curvature_cm_rad2 for every rotor")
        barrier = one_term_fourier_barrier(
            int(rotor["periodicity"]),
            float(rotor["curvature_cm_rad2"]),
            barrier_cm=(None if rotor.get("barrier_cm") is None
                        else float(rotor["barrier_cm"])),
        )
        levels = periodic_rotor_levels(
            float(rotor["moment_amu_a2"]),
            int(rotor["periodicity"]),
            barrier,
            phi_rad=rotor.get("phi_rad"),
            potential_cm=rotor.get("potential_cm"),
            basis_size=basis_size, sector=sector,
        )
        sigma = int(rotor.get("symmetry_number", rotor["periodicity"]))
        relative_levels = levels - levels[0]
        q = float(np.exp(-beta * relative_levels).sum() / sigma)
        tunneling_splitting = float(levels[1] - levels[0])
        fundamental = float(levels[int(rotor["periodicity"])] - levels[0])
        zpe = float(levels[0])
        q_soft *= q
        zpe_soft += zpe
        output.append({"identifier": rotor["identifier"],
                       "tunneling_splitting_cm": tunneling_splitting,
                       "fundamental_cm": fundamental,
                       "zpe_cm": zpe, "q_rotor": q})
    return {"method": "MS-RS", "temperature_K": temperature_K,
            "basis_size": basis_size, "q_soft": q_soft,
            "zpe_soft_cm": zpe_soft, "rotors": output}
