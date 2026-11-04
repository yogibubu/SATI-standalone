"""Protected transverse E/G/H correction of a resident reactive ZAFF.

SMITH/the caller supplies a common internal reaction chart and observations
transformed into that chart, including the nonstationary gradient term.
This owner never acquires points or changes the parent's local quartic jets.
"""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import BPoly, CubicSpline, BSpline, make_interp_spline
from .native_pic_field import NativePicPoint


@dataclass(frozen=True)
class IRCChartJet:
    coordinates: np.ndarray
    jacobian_per_bohr: np.ndarray
    hessians_per_bohr2: np.ndarray


def fit_irc_coefficient_curve(positions, values, *, first_derivatives=None):
    """C4 physical coefficient curve using values and optional Hessian slopes.

    Extra simple knots supply the degrees of freedom for slope observations;
    they are analytic representation knots, never new PES sample requests.
    """
    s, v = np.asarray(positions, float), np.asarray(values, float)
    if (s.ndim != 1 or len(s) < 6 or v.shape[:1] != s.shape or not np.all(np.isfinite(s))
            or not np.all(np.isfinite(v)) or np.any(np.diff(s) <= 0)):
        raise ValueError('at least six ordered finite samples and aligned physical coefficients required')
    template = make_interp_spline(s, v, k=5)
    observations = dict(first_derivatives or {})
    if not observations:
        return template, dict(value_observations=len(s), slope_observations=0, continuity_order=4)
    extra = []
    for i, value in observations.items():
        if (not isinstance(i, (int, np.integer)) or not 0 < i < len(s)-1
                or np.asarray(value).shape != v.shape[1:] or not np.all(np.isfinite(value))):
            raise ValueError('finite interior slope observations in the same coefficient shape required')
        # Insert toward the domain center to preserve reflection symmetry.
        neighbor = i+1 if s[i] < (s[0]+s[-1])/2 else i-1
        extra.append((s[i]+s[neighbor])/2)
    if len(set(extra)) != len(extra) or any(x in template.t for x in extra):
        raise ValueError('slope observations require distinct additional analytic knots')
    knots = np.sort(np.r_[template.t, extra]); nc = len(knots)-6
    basis = BSpline(knots, np.eye(nc), 5, extrapolate=False)
    ids = list(observations)
    design = np.vstack([basis(s), basis(s[ids], nu=1)])
    target = np.concatenate([v, np.array([observations[i] for i in ids])])
    coefficients = np.linalg.solve(design, target.reshape(len(target), -1)).reshape((nc,)+v.shape[1:])
    curve = BSpline(knots, coefficients, 5, extrapolate=False)
    return curve, dict(value_observations=len(s), slope_observations=len(ids), continuity_order=4,
        maximum_value_residual=float(np.max(abs(curve(s)-v))),
        maximum_slope_residual=float(np.max(abs(curve(s[ids], nu=1)-np.array([observations[i] for i in ids])))))


class ProtectedIRCCurve:
    """C4 Hermite coefficients; exactly zero outside a finite channel domain."""
    def __init__(self, positions, values, *, protected_indices, first_derivatives=None):
        s, v = np.asarray(positions, float), np.asarray(values, float)
        if (s.ndim != 1 or len(s) < 3 or np.any(np.diff(s) <= 0)
                or v.shape[:1] != s.shape or not np.all(np.isfinite(s))
                or not np.all(np.isfinite(v))):
            raise ValueError('ordered finite positions and coefficient values required')
        protected = tuple(sorted(set(protected_indices)))
        if (not {0, len(s)-1} <= set(protected) or any(not isinstance(i, (int, np.integer))
                or i < 0 or i >= len(s) for i in protected)
                or np.any(v[list(protected)] != 0)):
            raise ValueError('zero coefficients at both endpoints and all protected anchors required')
        guide = CubicSpline(s, v, axis=0)
        jets = [np.array([v[i], guide(s[i], 1), guide(s[i], 2), guide(s[i], 3),
                          np.zeros(v.shape[1:])]) for i in range(len(s))]
        for i, first in (first_derivatives or {}).items():
            first = np.asarray(first, float)
            if i in protected or not 0 <= i < len(s) or first.shape != v.shape[1:] or not np.all(np.isfinite(first)):
                raise ValueError('finite first-derivative observations at unprotected knots required')
            jets[i][1] = first
        for i in protected:
            jets[i][1:] = 0.
        self.positions, self.values, self.protected_indices = s.copy(), v.copy(), protected
        self.shape = v.shape[1:]
        self.protected_positions = s[list(protected)].copy()
        self.spline = BPoly.from_derivatives(s, jets, extrapolate=False)

    def __call__(self, s, derivative=0):
        if not np.isfinite(s) or derivative not in (0, 1, 2, 3, 4):
            raise ValueError('finite position and derivative order 0..4 required')
        if s <= self.positions[0] or s >= self.positions[-1]:
            return np.zeros(self.values.shape[1:])
        return np.asarray(self.spline(s, nu=derivative))


class ProtectedIRCCoefficientFunction:
    """Protected coefficients evaluated from frozen analytic fields/curves.

    The caller supplies a model-only function, never an electronic provider.
    This avoids interpolating a residual that inherits rapid variations of
    the known parent: interpolate the physical target, subtract the parent
    analytically, then differentiate that frozen function. Finite differences
    here acquire no reference-PES/QM observations.
    """
    def __init__(self, provider, shape, *, protected_positions, protection_radius, derivative_step=1e-4):
        s = np.asarray(protected_positions, float)
        if (not callable(provider) or s.ndim != 1 or len(s) < 2
                or not np.all(np.isfinite(s)) or np.any(np.diff(s) <= 0)
                or not np.isfinite(protection_radius) or protection_radius <= 0
                or 2*protection_radius >= min(np.diff(s))
                or not np.isfinite(derivative_step) or derivative_step <= 0
                or any(not isinstance(n, int) or n < 1 for n in shape)):
            raise ValueError('frozen model coefficient function, shape, ordered anchors and positive nonoverlapping radii required')
        self.provider, self.shape = provider, tuple(shape)
        self.protected_positions = s.copy()
        self.radius, self.step = float(protection_radius), float(derivative_step)
        self.smooth = np.polynomial.Polynomial([0, 0, 0, 0, 0, 126, -420, 540, -315, 70])

    def _value(self, s):
        if s <= self.protected_positions[0] or s >= self.protected_positions[-1]:
            return np.zeros(self.shape)
        distance = float(min(abs(s-self.protected_positions)))
        if distance == 0:
            return np.zeros(self.shape)
        factor = float(self.smooth(min(1., distance/self.radius)))
        value = np.asarray(self.provider(float(s)), float)
        if value.shape != self.shape or not np.all(np.isfinite(value)):
            raise ValueError('finite frozen model coefficient with declared shape required')
        return factor*value

    def __call__(self, s, derivative=0):
        if not np.isfinite(s) or derivative not in (0, 1, 2):
            raise ValueError('finite position and coefficient derivative order 0..2 required')
        if s in self.protected_positions or s <= self.protected_positions[0] or s >= self.protected_positions[-1]:
            return np.zeros(self.shape)
        if derivative == 0:
            return self._value(s)
        h = self.step
        vm2, vm, vp, vp2 = (self._value(s+k*h) for k in [-2, -1, 1, 2])
        if derivative == 1:
            return (vm2-8*vm+8*vp-vp2)/(12*h)
        return (-vp2+16*vp-30*self._value(s)+16*vm-vm2)/(12*h*h)


class IRCTransverseRefinedZAFF:
    """Parent + W(x) exp[-(||z||/L)^6] [a(s).z + z.B(s).z/2].

    Coefficient curves vanish with derivatives 1..4 at protected anchors.
    The correction vanishes identically on z=0. A native bond-support window
    W is mandatory, to retain dissociation limits away from the IRC as well.
    """
    def __init__(self, parent, chart, linear, quadratic, support, *, transverse_radius):
        if (not callable(parent) or not tuple(getattr(parent, 'atoms', ()))
                or not callable(chart) or not callable(support)
                or not isinstance(linear, (ProtectedIRCCurve, ProtectedIRCCoefficientFunction))
                or not isinstance(quadratic, (ProtectedIRCCurve, ProtectedIRCCoefficientFunction))
                or len(linear.shape) != 1 or quadratic.shape != (linear.shape[0],)*2
                or not np.isfinite(transverse_radius) or transverse_radius <= 0):
            raise ValueError('resident parent, common chart, protected coefficient curves, support and positive radius required')
        if not np.array_equal(linear.protected_positions, quadratic.protected_positions):
            raise ValueError('linear and quadratic corrections must protect the same anchors')
        self.parent, self.chart, self.linear, self.quadratic, self.support = parent, chart, linear, quadratic, support
        self.radius, self.atoms = float(transverse_radius), tuple(parent.atoms)
        self.schema = 'matrix.architect.irc_transverse_refined_zaff.v1'
        self.metadata = dict(model_level='ZAFF_IRC_HESSIAN_REFINED', physical_field_qualified=False,
                             preserves_parent_four_jets_at_protected_anchors=True,
                             energy_correction_on_IRC=0., parent_unchanged=True)

    def energy(self, xyz):
        return self(xyz).energy_hartree

    def energy_gradient(self, xyz):
        point = self(xyz)
        return point.energy_hartree, point.gradient_hartree_per_bohr

    def energy_gradient_vectorized(self, xyz):
        return self.energy_gradient(xyz)

    def energies(self, geometries, *, batch_size=64):
        xyz = np.asarray(geometries, float)
        if (xyz.ndim != 3 or xyz.shape[1:] != (len(self.atoms), 3)
                or not np.all(np.isfinite(xyz)) or isinstance(batch_size, bool)
                or not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError('finite geometry batch and positive integer batch size required')
        result = np.empty(len(xyz))
        for start in range(0, len(xyz), batch_size):
            result[start:start+batch_size] = [self.energy(x) for x in xyz[start:start+batch_size]]
        return result

    def __call__(self, xyz):
        p = self.parent(xyz); c = self.chart(xyz)
        q, j, h = (np.asarray(a, float) for a in
                   (c.coordinates, c.jacobian_per_bohr, c.hessians_per_bohr2))
        m, d = self.linear.shape[0], np.size(xyz)
        if (q.shape != (m+1,) or j.shape != (m+1, d) or h.shape != (m+1, d, d)
                or not all(np.all(np.isfinite(a)) for a in (q, j, h))):
            raise ValueError('finite common internal reaction-chart two-jet required')
        s, z = q[0], q[1:]
        a, a1, a2 = (self.linear(s, k) for k in range(3))
        b, b1, b2 = (self.quadratic(s, k) for k in range(3))
        if any(not np.allclose(v, v.T, atol=1e-12) for v in (b, b1, b2)):
            raise ValueError('symmetric transverse quadratic coefficient jets required')
        f = float(a@z+.5*z@b@z)
        gq = np.r_[a1@z+.5*z@b1@z, a+b@z]
        hq = np.empty((m+1, m+1)); hq[0, 0] = a2@z+.5*z@b2@z
        hq[0, 1:] = hq[1:, 0] = a1+b1@z; hq[1:, 1:] = b
        r2 = float(z@z)/self.radius**2
        if r2 > 700**(1/3):
            return p
        damping = np.exp(-r2**3)
        logg = np.r_[0., -6*r2**2*z/self.radius**2]
        logh = np.zeros_like(hq)
        logh[1:, 1:] = -6*r2**2*np.eye(m)/self.radius**2-24*r2*np.outer(z, z)/self.radius**4
        dh = damping*(hq+np.outer(logg, gq)+np.outer(gq, logg)+f*(logh+np.outer(logg, logg)))
        dg = damping*(gq+f*logg); de = damping*f
        gx = j.T@dg; hx = j.T@dh@j+np.einsum('i,ijk->jk', dg, h)
        w = self.support(xyz)
        wg, wh = np.asarray(w.gradient_per_bohr, float), np.asarray(w.hessian_per_bohr2, float)
        if (not np.isfinite(w.value) or not 0 <= w.value <= 1 or wg.shape != (d,)
                or wh.shape != (d, d) or not np.all(np.isfinite(wg)) or not np.all(np.isfinite(wh))):
            raise ValueError('native support window value in [0,1] and finite Cartesian derivatives required')
        return NativePicPoint(float(p.energy_hartree+w.value*de),
            p.gradient_hartree_per_bohr+w.value*gx+de*wg,
            p.hessian_hartree_per_bohr2+w.value*hx+np.outer(wg, gx)+np.outer(gx, wg)+de*wh)
