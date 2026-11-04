"""Construction-time valence compensation in a declared local coordinate chart.

The target geometry and diagonal curvatures belong to the donor, not to the
minimum parameters of the compensated terms. Explicit terms (electrostatics,
including 1-2/1-3, nonbonded, and any retained couplings) are supplied once as
analytic derivatives in the SAME chart. This module changes no donor tables.
It fits existing Morse/Exp-PE stretches and BO-switched nonlinear bends; it
does not claim to reproduce the full coupled Hessian or to validate transfer.
"""

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.optimize import least_squares

from matrix_zaff.bonded import BondOrderDampedAnglePotential
from matrix_zaff.radial import ExpPEPotential, MorsePotential


@dataclass(frozen=True)
class RadialEquilibriumTerm:
    coordinate: int
    potential: MorsePotential | ExpPEPotential


@dataclass(frozen=True)
class AngleEquilibriumTerm:
    coordinates: tuple[int, int, int]  # adjacent stretches, then angle
    potential: BondOrderDampedAnglePotential


def refit_valence_equilibria(coordinates, target_diagonal, fixed_gradient,
                            fixed_diagonal, stretches, bends, *, provenance):
    """Preserve donor equilibria and total diagonal curvature after adding Q.

    Units: angstrom/radian and hartree, consistently for all supplied arrays.
    Each coordinate has exactly one diagonal family; bends may share bonds.
    The caller must use an independent chart or certify compatibility of its
    primitive target geometry. Linear bends require their two-component chart
    and are rejected here. No molecular gradients are numerically differentiated.

    Stretch well depths and bend radial switches remain fixed. Bend amplitude
    and center are solved analytically first; their radial derivatives then
    enter the stretch compensation, avoiding a second subtraction or an
    uncoupled treatment of the bond-order switches. Radial range/center are
    determined by two local derivative equations with explicit failure gates.
    """
    q, target, fixed_g, fixed_k = [np.array(a, dtype=float, copy=True) for a in
                                  (coordinates, target_diagonal, fixed_gradient, fixed_diagonal)]
    if (q.ndim != 1 or not len(q) or any(a.shape != q.shape or not np.all(np.isfinite(a))
                                       for a in (q, target, fixed_g, fixed_k))):
        raise ValueError("finite, equally sized one-dimensional derivative arrays required")
    if np.any(target <= 0) or not isinstance(provenance, dict) or not provenance:
        raise ValueError("positive target curvatures and explicit donor provenance required")
    stretches, bends = tuple(stretches), tuple(bends)
    if any(len(t.coordinates) != 3 for t in bends):
        raise ValueError("bend support must have three coordinates")
    radial_indices = [t.coordinate for t in stretches]
    angular_indices = [t.coordinates[2] for t in bends]
    indices = radial_indices + angular_indices
    if (any(not isinstance(i, (int, np.integer)) or isinstance(i, bool) for i in indices)
            or len(indices) != len(q) or set(indices) != set(range(len(q)))):
        raise ValueError("each coordinate needs exactly one diagonal family")
    radial_support = set(radial_indices)
    for term in bends:
        if (len(set(term.coordinates)) != 3
                or any(not isinstance(i, (int, np.integer)) or isinstance(i, bool)
                       for i in term.coordinates)
                or any(i not in radial_support for i in term.coordinates[:2])):
            raise ValueError("bend support must contain two covered stretches and its angle")
    gradient, diagonal = fixed_g.copy(), fixed_k.copy()
    fitted_bends = []
    for term in bends:
        i, j, a = term.coordinates
        original = term.potential
        if not isinstance(original, BondOrderDampedAnglePotential) or original.linear:
            raise ValueError("only nonlinear BO-damped bends are supported")
        theta = q[a]
        if not 1e-4 < theta < np.pi-1e-4:
            raise ValueError("linear bends require a two-component coordinate chart")
        sine, cosine = np.sin(theta), np.cos(theta)
        wanted_g, wanted_k = -fixed_g[a], target[a]-fixed_k[a]
        amplitude = (wanted_k-wanted_g*cosine/sine)/sine**2
        if amplitude <= 0:
            raise ValueError("bend compensation needs a positive cosine amplitude")
        cosine0 = cosine+wanted_g/(amplitude*sine)
        if not -1 < cosine0 < 1:
            raise ValueError("bend compensation has no physical equilibrium center")
        theta0 = float(np.arccos(cosine0))
        factor = original.bond1.value(q[i])*original.bond2.value(q[j])
        if factor <= 0 or abs(np.sin(theta0)) <= original.linear_sine_threshold:
            raise ValueError("bend compensation has a vanishing switch or linear center")
        potential = replace(original, theta0=theta0,
                            force_constant=float(amplitude*(1-cosine0**2)/factor))
        value = potential.derivatives(q[i], q[j], q[a])
        gradient[list(term.coordinates)] += value.gradient
        diagonal[list(term.coordinates)] += np.diag(value.hessian)
        fitted_bends.append(replace(term, potential=potential))

    fitted_stretches = []
    for term in stretches:
        i, original = term.coordinate, term.potential
        wanted_g, wanted_k = -gradient[i], target[i]-diagonal[i]
        if q[i] <= 0 or wanted_k <= 0:
            raise ValueError("stretch compensation needs a positive distance and residual curvature")
        if isinstance(original, MorsePotential):
            initial = [original.r_min, original.beta/original.r_min]
            bounds = ([.5*q[i], 1e-4], [1.5*q[i], 50.])

            def factory(p):
                return MorsePotential(original.epsilon, p[0], p[0]*p[1])
        elif isinstance(original, ExpPEPotential):
            initial = [original.r_min, original.alpha]
            bounds = ([.5*q[i], 4.+1e-7], [1.5*q[i], 80.])

            def factory(p):
                return ExpPEPotential(original.epsilon, *p)
        else:
            raise ValueError("unsupported stretch form; no implicit functional conversion")
        scale = max(target[i], abs(wanted_g)/q[i], 1e-8)

        def residual(p):
            d = factory(p).derivatives(q[i])
            return [(d.first-wanted_g)/(scale*q[i]), (d.second-wanted_k)/scale]

        result = least_squares(residual, initial, bounds=bounds, x_scale="jac",
                               ftol=1e-13, xtol=1e-13, gtol=1e-13, max_nfev=200)
        if not result.success or np.max(np.abs(residual(result.x))) > 1e-9:
            raise ValueError(f"stretch compensation failed at coordinate {i}: {result.message}")
        potential = factory(result.x)
        value = potential.derivatives(q[i])
        gradient[i] += value.first
        diagonal[i] += value.second
        fitted_stretches.append(replace(term, potential=potential))
    relative_curvature_error = (diagonal-target)/target
    if np.max(np.abs(gradient)) > 1e-8 or np.max(np.abs(relative_curvature_error)) > 1e-8:
        raise ValueError("equilibrium/curvature gate failed; no compensated model returned")
    report = {
        "schema": "matrix.architect.valence_equilibrium_refit.v1",
        "provenance": dict(provenance), "target_coordinates": q.tolist(),
        "target_diagonal": target.tolist(), "fixed_gradient": fixed_g.tolist(),
        "fixed_diagonal": fixed_k.tolist(), "total_gradient": gradient.tolist(),
        "relative_diagonal_error": relative_curvature_error.tolist(),
        "original_stretches": [asdict(t) for t in stretches],
        "fitted_stretches": [asdict(t) for t in fitted_stretches],
        "original_bends": [asdict(t) for t in bends],
        "fitted_bends": [asdict(t) for t in fitted_bends],
        "source_parameters_modified": False,
        "preserved": ["stretch_well_depth", "functional_families", "bend_radial_switches"],
        "full_hessian_matched": False,
        "construction_only": True,
    }
    return tuple(fitted_stretches), tuple(fitted_bends), report
