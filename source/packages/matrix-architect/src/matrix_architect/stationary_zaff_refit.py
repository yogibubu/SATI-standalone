"""Analytic, asymptote-preserving local refinements of a physical ZAFF prior.

The prior is supplied by the current ARCHITECT compiler. Measured local jets
correct that prior inside a C4 support in pair-distance space. The correction
vanishes in every dissociation direction and is rigid-motion invariant when
the supplied independent internal-coordinate kernel is invariant.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np

from matrix_chem import BOHR_TO_ANGSTROM as B
from .model import ArchitectEvaluation
from .native_coordinate_quartic import NativeCoordinateQuartic


class SeedZAFFPrior:
    """Resident diagonal ZAFF0 with analytic derivatives from its owner."""

    def __init__(self, model):
        self.model = model
        self.atoms = tuple(model['atoms'])

    def __call__(self, xyz):
        from matrix_zaff.seed_runtime import evaluate_zaff_seed_model
        p = evaluate_zaff_seed_model(self.model, xyz, hessian=True)
        return ArchitectEvaluation(p.energy_hartree, p.gradient_hartree_per_bohr,
            p.hessian_hartree_per_bohr2, ('energy', 'gradient', 'hessian'), source='ARCHITECT/ZAFF0')


class PairDistanceC4Support:
    """Fixed nonic window of the norm of all pair-distance changes, in angstrom."""

    def __init__(self, reference, *, inner=.5, outer=1.):
        self.reference = np.asarray(reference, float)
        if not 0 < inner < outer:
            raise ValueError('positive ordered support radii required')
        self.inner, self.outer = float(inner), float(outer)
        self.pairs = tuple(combinations(range(len(self.reference)), 2))
        self.distances = np.array([np.linalg.norm(self.reference[i]-self.reference[j])
                                   for i, j in self.pairs])
        if not len(self.pairs) or np.any(self.distances <= 0):
            raise ValueError('distinct molecular atoms required')
        self.window = np.polynomial.Polynomial([1, 0, 0, 0, 0, -126, 420, -540, 315, -70])

    def __call__(self, xyz):
        xyz = np.asarray(xyz, float)
        dimension = xyz.size
        deltas, jacobians, curvatures = [], [], []
        for (i, j), r0 in zip(self.pairs, self.distances):
            d = xyz[i]-xyz[j]
            r = np.linalg.norm(d)
            if r <= 0:
                raise ValueError('coincident atoms in molecular support')
            u = d/r
            g = np.zeros(dimension)
            g[3*i:3*i+3], g[3*j:3*j+3] = u, -u
            h = np.zeros((dimension, dimension))
            block = (np.eye(3)-np.outer(u, u))/r
            for a, sign_a in ((i, 1), (j, -1)):
                for b, sign_b in ((i, 1), (j, -1)):
                    h[3*a:3*a+3, 3*b:3*b+3] = sign_a*sign_b*block
            deltas.append(r-r0)
            jacobians.append(g)
            curvatures.append(h)
        delta, jacobian, curvature = map(np.asarray, (deltas, jacobians, curvatures))
        rho = np.linalg.norm(delta)
        if rho <= self.inner:
            return 1., np.zeros(dimension), np.zeros((dimension, dimension))
        if rho >= self.outer:
            return 0., np.zeros(dimension), np.zeros((dimension, dimension))
        g = delta@jacobian/rho
        h = (jacobian.T@jacobian+np.einsum('k,kij->ij', delta, curvature)-np.outer(g, g))/rho
        width = self.outer-self.inner
        t = (rho-self.inner)/width
        first, second = self.window.deriv(1)(t)/width, self.window.deriv(2)(t)/width**2
        return float(self.window(t)), first*g*B, (first*h+second*np.outer(g, g))*B**2


class SupportedZAFFRefit:
    """A physical prior plus a compact local native-coordinate correction."""

    schema = 'matrix.architect.supported_stationary_refit.v1'

    def __init__(self, prior, atoms, kernel, reference, *, constant=0., linear=None,
                 quadratic=None, quartic=None, support=None):
        self.prior, self.atoms, self.kernel = prior, tuple(atoms), kernel
        self.reference = np.asarray(reference, float)
        n = len(kernel.value(self.reference)[0])
        self.constant = float(constant)
        self.linear = np.zeros(n) if linear is None else np.asarray(linear, float)
        self.quadratic = np.zeros((n, n)) if quadratic is None else np.asarray(quadratic, float)
        self.quartic = quartic
        self.quartic_model = quartic
        self.support = PairDistanceC4Support(reference) if support is None else support
        if self.linear.shape != (n,) or self.quadratic.shape != (n, n):
            raise ValueError('aligned independent native-coordinate coefficients required')

    def __call__(self, xyz):
        base = self.prior(xyz)
        w, wg, wh = self.support(xyz)
        if w == 0.:
            return base
        u, j, h = self.kernel.value(xyz)
        e = self.constant+self.linear@u+.5*u@self.quadratic@u
        gu = self.linear+self.quadratic@u
        hu = self.quadratic.copy()
        if self.quartic is not None:
            de, dg, dh = self.quartic.potential_jet(u)
            e, gu, hu = e+de, gu+dg, hu+dh
        g = j@gu*B
        H = (j@hu@j.T+np.einsum('k,ijk->ij', gu, h))*B**2
        return ArchitectEvaluation(float(base.energy_hartree+w*e),
            base.gradient_hartree_per_bohr+w*g+e*wg,
            base.hessian_hartree_per_bohr2+w*H+e*wh+np.outer(wg, g)+np.outer(g, wg),
            ('energy', 'gradient', 'hessian'), source='ARCHITECT/SUPPORTED-ZAFF-REFIT')

    def energy(self, xyz):
        return self(xyz).energy_hartree

    def energy_gradient(self, xyz):
        point = self(xyz)
        return point.energy_hartree, point.gradient_hartree_per_bohr


def realized_jet(field, geometry, coordinates):
    xyz, a, b = geometry(coordinates)
    point = field(xyz)
    g = point.gradient_hartree_per_bohr/B
    H = a.T@point.hessian_hartree_per_bohr2@a/B**2+np.einsum('i,iab->ab', g, b)
    return point.energy_hartree, a.T@g, (H+H.T)/2


def refit_harmonic_jet(prior, atoms, kernel, geometry, target_energy, target_gradient, target_hessian):
    """Match E/G/H without changing the physical prior outside the support."""
    n = len(target_gradient)
    xyz, a, _ = geometry(np.zeros(n))
    _, j, _ = kernel.value(xyz)
    transform = j.T@a
    inverse = np.linalg.inv(transform)
    e, g, _ = realized_jet(prior, geometry, np.zeros(n))
    linear = inverse.T@(np.asarray(target_gradient)-g)
    first = SupportedZAFFRefit(prior, atoms, kernel, xyz,
        constant=target_energy-e, linear=linear)
    _, _, h = realized_jet(first, geometry, np.zeros(n))
    quadratic = inverse.T@(np.asarray(target_hessian)-h)@inverse
    full = SupportedZAFFRefit(prior, atoms, kernel, xyz,
        constant=target_energy-e, linear=linear, quadratic=(quadratic+quadratic.T)/2)
    return first, full, dict(native_condition_number=float(np.linalg.cond(transform)),
                             support_inner_angstrom=.5, support_outer_angstrom=1.)


def acquire_model_fourjet(field, geometry, n, *, steps=(.04, .02)):
    """Richardson differences of analytic Hessians; no electronic acquisitions."""
    from matrix_gf.tiered_anharmonic import acquire_tiered_anharmonic
    scans = []
    cache = {}
    def packet(q):
        key = tuple(q)
        if key not in cache:
            cache[key] = realized_jet(field, geometry, q)
        return cache[key]
    for step in steps:
        scans.append(acquire_tiered_anharmonic(lambda q: packet(q)[0], lambda q: packet(q)[1],
            lambda q: packet(q)[2], np.zeros(n), reaction_mode=0,
            primary_modes=range(1, n), steps=step))
    coarse, fine = steps
    result = dict(energy=packet(np.zeros(n))[0], gradient=packet(np.zeros(n))[1],
                  hessian=packet(np.zeros(n))[2], model_points=len(cache))
    for key in ('cubic', 'quartic'):
        result[key] = (coarse**2*scans[1][key]-fine**2*scans[0][key])/(coarse**2-fine**2)
    return result


def refit_anharmonic_jet(harmonic, geometry, target_cubic, target_quartic, *, cache=None):
    """Refine observed C3/semi-diagonal Q4; unidentified Q4 changes stay zero."""
    n = len(target_cubic)
    before = acquire_model_fourjet(harmonic, geometry, n) if cache is None else cache
    observed_c = np.isfinite(before['cubic']) & np.isfinite(target_cubic)
    observed_q = np.isfinite(before['quartic']) & np.isfinite(target_quartic)
    dc = np.where(observed_c, target_cubic-before['cubic'], 0.)
    dq = np.where(observed_q, target_quartic-before['quartic'], 0.)
    correction = NativeCoordinateQuartic.from_geometry_realization(
        np.zeros((n, n)), dc, dq, harmonic.kernel, geometry)
    model = SupportedZAFFRefit(harmonic.prior, harmonic.atoms, harmonic.kernel, harmonic.reference,
        constant=harmonic.constant, linear=harmonic.linear, quadratic=harmonic.quadratic,
        quartic=correction, support=harmonic.support)
    return model, before, dict(observed_cubic_entries=int(observed_c.sum()),
        observed_quartic_entries=int(observed_q.sum()),
        full_quartic_entries=n**4, zaff4_rank_complete=False,
        unobserved_quartic_change='zero correction; physical parent contribution retained')
