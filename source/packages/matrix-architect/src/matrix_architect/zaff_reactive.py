"""Compose resident ZAFF surfaces without changing their local functions.

Two distinct constructions are explicit: an IRC-constrained partition of
unity, and the lower eigenvalue of a real symmetric multistate EVB matrix.
Coordinates are supplied by the caller (normally SMITH); this module neither
builds an IRC nor acquires electronic points. NativePicPoint units apply:
angstrom inputs, Eh energies, derivatives with respect to Cartesian bohr.
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BPoly, CubicSpline

from .native_pic_field import NativePicPoint


@dataclass(frozen=True)
class ReactionCoordinateJet:
    value: float
    gradient_per_bohr: np.ndarray
    hessian_per_bohr2: np.ndarray


@dataclass(frozen=True)
class EVBCouplingJet:
    """Off-diagonal H, dH, d2H in Eh, Eh/bohr, Eh/bohr2."""
    matrix: np.ndarray
    gradient: np.ndarray
    hessian: np.ndarray


def _parents(fields):
    fields = tuple(fields)
    if len(fields) < 2 or any(not callable(f) for f in fields):
        raise ValueError('at least two callable resident fields are required')
    atoms = tuple(getattr(fields[0], 'atoms', ()))
    if not atoms or any(tuple(getattr(f, 'atoms', ())) != atoms for f in fields):
        raise ValueError('all fields must use the same nonempty atom order')
    return fields, atoms


def _points(fields, xyz):
    xyz = np.asarray(xyz, float)
    if (xyz.shape not in ((len(fields[0].atoms), 3), (3*len(fields[0].atoms),))
            or not np.all(np.isfinite(xyz))):
        raise ValueError('finite Cartesian geometry in the declared atom order required')
    points = [f(xyz) for f in fields]
    d = np.size(xyz)
    e = np.array([p.energy_hartree for p in points], float)
    g = np.array([p.gradient_hartree_per_bohr for p in points], float)
    h = np.array([p.hessian_hartree_per_bohr2 for p in points], float)
    if (g.shape != (len(fields), d) or h.shape != (len(fields), d, d)
            or not all(np.all(np.isfinite(a)) for a in (e, g, h))
            or not np.allclose(h, h.transpose(0, 2, 1), atol=1e-12, rtol=1e-10)):
        raise ValueError('finite symmetric native field E/G/H with flat derivatives required')
    return e, g, h


class IRCZAFFWeights:
    """C4 endpoint-flat interpolation of signed partition-of-unity weights.

    Each knot solves the energy and normalization equations with the smallest
    Euclidean change from an explicit prior. For >2 states that prior defines
    the otherwise underdetermined weights. No clipping is performed. Equal
    parent values are allowed only if the reference already agrees.
    """
    def __init__(self, positions, knot_weights, *, protected_indices=(),
                 knot_first=None, knot_second=None, report=None):
        s, w = np.asarray(positions, float), np.asarray(knot_weights, float)
        if (s.ndim != 1 or len(s) < 2 or not np.all(np.isfinite(s))
                or np.any(np.diff(s) <= 0) or w.ndim != 2 or w.shape[0] != len(s)
                or w.shape[1] < 2 or not np.all(np.isfinite(w))
                or not np.allclose(w.sum(axis=1), 1., atol=1e-10, rtol=0)):
            raise ValueError('ordered finite knots and normalized multistate weights required')
        anchors = tuple(sorted({0, len(s)-1, *protected_indices}))
        if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 or i >= len(s)
               for i in anchors):
            raise ValueError('protected knot indices must lie inside the IRC table')
        self.positions, self.knot_weights = s.copy(), w.copy()
        self.protected_indices = anchors
        # Estimate derivatives locally, then impose C4 Hermite joins. A global
        # degree-nine interpolating spline can oscillate catastrophically on
        # the strongly nonuniform near-TS knots of a physical IRC.
        first, second = knot_first, knot_second
        if (first is None) != (second is None):
            raise ValueError('first and second weight derivatives must be supplied together')
        if first is not None:
            first, second = np.asarray(first, float), np.asarray(second, float)
            if (first.shape != w.shape or second.shape != w.shape
                    or not np.all(np.isfinite(first)) or not np.all(np.isfinite(second))):
                raise ValueError('finite knot-aligned first and second weight derivatives required')
            guide = BPoly.from_derivatives(s, [np.array([v, d, h]) for v, d, h in zip(w, first, second)])
        else:
            guide = CubicSpline(s, w, axis=0)
        self.knot_first = None if first is None else first.copy()
        self.knot_second = None if second is None else second.copy()
        jets = [np.array([w[i], guide(s[i], 1), guide(s[i], 2),
                          guide(s[i], 3), np.zeros(w.shape[1])]) for i in range(len(s))]
        for i in anchors:
            jets[i][1:] = 0.
        self.spline = BPoly.from_derivatives(s, jets, extrapolate=False)
        self.report = dict(report or {})

    @classmethod
    def fit(cls, positions, parent_energies, reference_energies, *, prior,
            protected_indices=(), energy_tolerance_hartree=1e-10, gap_tolerance_hartree=1e-10,
            parent_first=None, parent_second=None, reference_first=None, reference_second=None,
            prior_first=None, prior_second=None, active_states=None):
        s = np.asarray(positions, float)
        e, target, w = (np.asarray(a, float) for a in
                        (parent_energies, reference_energies, prior))
        if (s.ndim != 1 or e.ndim != 2 or e.shape[0] != len(s)
                or e.shape[1] < 2 or target.shape != (len(s),) or w.shape != e.shape
                or not all(np.all(np.isfinite(a)) for a in (e, target, w))
                or not np.allclose(w.sum(axis=1), 1., atol=1e-10, rtol=0)
                or not np.isfinite(energy_tolerance_hartree) or energy_tolerance_hartree <= 0
                or not np.isfinite(gap_tolerance_hartree) or gap_tolerance_hartree <= 0):
            raise ValueError('finite energy table, normalized explicit prior and positive tolerances required')
        mask = np.ones(e.shape, bool) if active_states is None else np.asarray(active_states)
        if (mask.dtype.kind != 'b' or mask.shape != e.shape or np.any(mask.sum(axis=1) < 2)
                or np.any(w[~mask] != 0)):
            raise ValueError('boolean active-state mask with at least two states and zero inactive prior required')
        count = mask.sum(axis=1, keepdims=True)
        centered = np.where(mask, e-np.sum(e*mask, axis=1, keepdims=True)/count, 0.)
        norm = np.linalg.norm(centered, axis=1)
        residual = target - np.sum(w*e, axis=1)
        singular = norm <= gap_tolerance_hartree
        if np.any(np.abs(residual[singular]) > energy_tolerance_hartree):
            raise ValueError('coincident parent fields cannot reproduce the IRC reference')
        weights = w.copy()
        active = ~singular
        weights[active] += residual[active, None]*centered[active]/norm[active, None]**2
        protected = tuple(protected_indices)
        if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 or i >= len(s)
               for i in protected):
            raise ValueError('protected knot indices must lie inside the IRC table')
        if protected:
            if np.any(np.abs(residual[list(protected)]) > energy_tolerance_hartree):
                raise ValueError('protected parent weights do not reproduce the IRC anchor energy')
            weights[list(protected)] = w[list(protected)]
        derivatives = (parent_first, parent_second, reference_first, reference_second, prior_first, prior_second)
        first = second = None
        if any(a is not None for a in derivatives):
            if any(a is None for a in derivatives):
                raise ValueError('complete first/second IRC energy and prior jets required')
            e1, e2, t1, t2, p1, p2 = (np.asarray(a, float) for a in derivatives)
            if (any(a.shape != e.shape for a in (e1, e2, p1, p2))
                    or t1.shape != target.shape or t2.shape != target.shape
                    or not all(np.all(np.isfinite(a)) for a in (e1, e2, t1, t2, p1, p2))
                    or not np.allclose(p1.sum(axis=1), 0., atol=1e-10)
                    or not np.allclose(p2.sum(axis=1), 0., atol=1e-10)):
                raise ValueError('finite normalized IRC energy and prior derivative tables required')
            if np.any(singular):
                raise ValueError('derivative-assisted fit requires a nonzero parent gap at every knot')
            if np.any(p1[~mask] != 0) or np.any(p2[~mask] != 0):
                raise ValueError('inactive prior derivatives must be zero')
            c1 = np.where(mask, e1-np.sum(e1*mask, axis=1, keepdims=True)/count, 0.)
            c2 = np.where(mask, e2-np.sum(e2*mask, axis=1, keepdims=True)/count, 0.)
            denominator = norm**2
            d1 = 2*np.sum(centered*c1, axis=1)
            d2 = 2*np.sum(c1*c1+centered*c2, axis=1)
            r1 = t1-np.sum(p1*e+w*e1, axis=1)
            r2 = t2-np.sum(p2*e+2*p1*e1+w*e2, axis=1)
            a = residual/denominator
            a1 = (r1-a*d1)/denominator
            a2 = (r2-2*a1*d1-a*d2)/denominator
            first = p1+a1[:, None]*centered+a[:, None]*c1
            second = p2+a2[:, None]*centered+2*a1[:, None]*c1+a[:, None]*c2
        error = np.max(np.abs(np.sum(weights*e, axis=1)-target))
        if error > energy_tolerance_hartree:
            raise ValueError('IRC weight solve exceeds the declared energy tolerance')
        return cls(s, weights, protected_indices=protected, knot_first=first, knot_second=second, report=dict(
            construction='IRC_partition_of_unity_not_matrix_EVB',
            maximum_knot_energy_error_hartree=float(error),
            coincident_parent_knots=int(singular.sum()),
            maximum_absolute_knot_weight=float(np.max(np.abs(weights))),
            convex_knot_weights=bool(np.all((weights >= 0) & (weights <= 1))),
            endpoint_derivatives_zero_through_order=4,
            active_states=mask.tolist(),
            prior_defines_multistate_nullspace=True, weights_clipped=False))

    def __call__(self, position, derivative=0):
        if derivative not in (0, 1, 2, 3, 4) or not np.isfinite(position):
            raise ValueError('finite scalar position and derivative order 0..4 required')
        if position <= self.positions[0] or position >= self.positions[-1]:
            endpoint = self.knot_weights[0 if position <= self.positions[0] else -1]
            return endpoint.copy() if derivative == 0 else np.zeros_like(endpoint)
        return np.asarray(self.spline(position, nu=derivative))

    def to_dict(self):
        return dict(schema='matrix.architect.irc_zaff_weights.v1',
                    positions=self.positions.tolist(), knot_weights=self.knot_weights.tolist(),
                    knot_first=None if self.knot_first is None else self.knot_first.tolist(),
                    knot_second=None if self.knot_second is None else self.knot_second.tolist(),
                    protected_indices=list(self.protected_indices), report=self.report)


class _ResidentComposition:
    """The same E/G/H and batch-energy surface interface as NativePicField."""
    def energy(self, xyz):
        return self(xyz).energy_hartree

    def energy_gradient(self, xyz):
        p = self(xyz)
        return p.energy_hartree, p.gradient_hartree_per_bohr

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


class IRCZAFFBlend(_ResidentComposition):
    """Full-dimensional resident fields combined using a scalar IRC chart."""
    def __init__(self, fields, weights, coordinate):
        self.fields, self.atoms = _parents(fields)
        if not isinstance(weights, IRCZAFFWeights) or weights.knot_weights.shape[1] != len(self.fields):
            raise ValueError('one IRC weight per resident field required')
        if not callable(coordinate):
            raise ValueError('a differentiable reaction-coordinate provider is required')
        self.weights, self.coordinate = weights, coordinate
        self.schema = 'matrix.architect.irc_zaff_blend.v1'
        self.metadata = dict(model_level='ZAFF_IRC_BLEND', parent_count=len(self.fields),
                             physical_field_qualified=False, evb_matrix=False)

    def __call__(self, xyz):
        e, g, h = _points(self.fields, xyz)
        c = self.coordinate(xyz)
        ds, d2s = np.asarray(c.gradient_per_bohr, float), np.asarray(c.hessian_per_bohr2, float)
        if (ds.shape != g.shape[1:] or d2s.shape != h.shape[1:]
                or not np.all(np.isfinite(ds)) or not np.all(np.isfinite(d2s))
                or not np.allclose(d2s, d2s.T, atol=1e-12, rtol=1e-10)):
            raise ValueError('finite symmetric coordinate jet in Cartesian bohr required')
        w, first, second = (self.weights(c.value, i) for i in range(3))
        mixed = first@g
        gradient = w@g + (first@e)*ds
        hessian = (np.einsum('i,ijk->jk', w, h)
                   + np.outer(ds, mixed) + np.outer(mixed, ds)
                   + (second@e)*np.outer(ds, ds) + (first@e)*d2s)
        return NativePicPoint(float(w@e), gradient, hessian)

class ZAFFMultiStateEVB(_ResidentComposition):
    """Lowest adiabatic surface of 2+ resident ZAFF diabatic fields.

    Includes the eigenvector-response contribution to the Hessian. A
    degenerate lowest eigenvalue is rejected, rather than differentiated
    using an arbitrary eigensolver vector. Couplings are caller-owned E/G/H.
    """
    def __init__(self, fields, couplings, *, gap_tolerance_hartree=1e-10):
        self.fields, self.atoms = _parents(fields)
        if not callable(couplings) or not np.isfinite(gap_tolerance_hartree) or gap_tolerance_hartree <= 0:
            raise ValueError('a coupling-jet provider and positive gap tolerance required')
        self.couplings, self.gap_tolerance = couplings, float(gap_tolerance_hartree)
        self.schema = 'matrix.architect.multistate_zaff_evb.v1'
        self.metadata = dict(model_level='ZAFF_MULTISTATE_EVB', parent_count=len(self.fields),
                             physical_field_qualified=False, evb_matrix=True)

    def __call__(self, xyz):
        e, g, h = _points(self.fields, xyz)
        n, d = len(e), g.shape[1]
        jet = self.couplings(xyz)
        matrix, first, second = (np.array(a, float, copy=True) for a in
                                 (jet.matrix, jet.gradient, jet.hessian))
        if (matrix.shape != (n, n) or first.shape != (n, n, d)
                or second.shape != (n, n, d, d)
                or not all(np.all(np.isfinite(a)) for a in (matrix, first, second))
                or any(not np.allclose(a, a.swapaxes(0, 1), atol=1e-12, rtol=1e-10)
                       for a in (matrix, first, second))
                or not np.allclose(second, second.swapaxes(2, 3), atol=1e-12, rtol=1e-10)
                or any(np.any(a[np.arange(n), np.arange(n)] != 0) for a in (matrix, first, second))):
            raise ValueError('finite symmetric off-diagonal coupling E/G/H required')
        idx = np.arange(n)
        matrix[idx, idx], first[idx, idx], second[idx, idx] = e, g, h
        values, vectors = np.linalg.eigh(matrix)
        if values[1]-values[0] <= self.gap_tolerance:
            raise ValueError('degenerate lowest EVB surface: unique E/G/H unavailable')
        ground = vectors[:, 0]
        gradient = np.einsum('i,ija,j->a', ground, first, ground)
        hessian = np.einsum('i,ijab,j->ab', ground, second, ground)
        response = np.einsum('ik,ija,j->ka', vectors[:, 1:], first, ground)
        hessian += 2*np.einsum('ka,kb,k->ab', response, response, 1/(values[0]-values[1:]))
        return NativePicPoint(float(values[0]), gradient, hessian)

def two_state_irc_coupling_squared(parent_energies, reference_energies):
    """Invert the lower EVB eigenvalue on an IRC; do not confuse it with a blend.

    This determines H12**2, not its sign, transverse dependence or smooth
    square root. A target above either diagonal cannot be the lower root.
    """
    e, target = np.asarray(parent_energies, float), np.asarray(reference_energies, float)
    if (e.ndim != 2 or e.shape[1] != 2 or target.shape != e.shape[:1]
            or not np.all(np.isfinite(e)) or not np.all(np.isfinite(target))):
        raise ValueError('finite two-state energy table and IRC reference required')
    if np.any(target > e.min(axis=1)):
        raise ValueError('lower EVB root must not exceed either diabatic field')
    return (e[:, 0]-target)*(e[:, 1]-target)
