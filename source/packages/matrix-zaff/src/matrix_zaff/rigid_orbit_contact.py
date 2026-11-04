"""Analytic contact potentials restricted to a declared rigid fragment orbit.

Pair exponentials/inverse powers and angular contact factors retain their
geometry. No electronic callback or interpolation grid is stored in a model.
"""
from dataclasses import dataclass
import numpy as np


def _distance(vector, first, second):
    r = np.linalg.norm(vector, axis=-1)
    if np.any(r <= 1e-12):
        raise ValueError('contact orbit contains coincident atoms')
    g = np.sum(vector * first, axis=-1) / r
    h = (np.sum(first * first + vector * second, axis=-1) - g * g) / r
    return r, g, h


def _product(a, b):
    return (a[0] * b[0], a[1] * b[0] + a[0] * b[1],
            a[2] * b[0] + 2 * a[1] * b[1] + a[0] * b[2])


@dataclass(frozen=True)
class RigidOrbitContactPotential:
    xyz_A: np.ndarray
    moving_atoms: tuple
    axis: np.ndarray
    pivot_A: np.ndarray
    orbit_vector_A: np.ndarray
    terms: tuple
    translation_only: bool = True

    def __post_init__(self):
        for name in ('xyz_A', 'axis', 'pivot_A', 'orbit_vector_A'):
            a = np.array(getattr(self, name), dtype=float, copy=True)
            if not np.all(np.isfinite(a)):
                raise ValueError('finite orbit geometry required')
            a.flags.writeable = False
            object.__setattr__(self, name, a)
        if (self.xyz_A.ndim != 2 or self.xyz_A.shape[1] != 3 or
                any(getattr(self, k).shape != (3,) for k in ('axis', 'pivot_A', 'orbit_vector_A')) or
                not np.isclose(np.linalg.norm(self.axis), 1., atol=1e-10)):
            raise ValueError('Cartesian positions and unit rotation axis required')
        ids = tuple(self.moving_atoms)
        if not ids or len(ids) != len(set(ids)) or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < len(self.xyz_A) for i in ids):
            raise ValueError('distinct valid moving atom indices required')
        object.__setattr__(self, 'moving_atoms', ids)
        terms = tuple(dict(t) for t in self.terms)
        for t in terms:
            if t.get('family') not in ('exp', 'inverse') or len(t.get('pair', [])) != 2:
                raise ValueError('explicit pair exponential or inverse-power contact required')
            ids = list(t['pair']) + list(t.get('angle', []))
            if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < len(self.xyz_A) for i in ids):
                raise ValueError('valid contact atom indices required')
            if len(t.get('angle', [])) not in (0, 3):
                raise ValueError('angular contact requires three atom indices')
            if not np.all(np.isfinite([t['amplitude_cm'], t['reference_A']])) or t['reference_A'] <= 0:
                raise ValueError('finite amplitude and positive reference distance required')
            if t['family'] == 'exp' and (not np.isfinite(t['width_A_inverse']) or t['width_A_inverse'] <= 0):
                raise ValueError('positive exponential width required')
            if t['family'] == 'inverse' and t.get('power') not in (4, 6, 8):
                raise ValueError('declared inverse power 4, 6 or 8 required')
            if 'angle' in t and (t.get('angular_power') not in (0, 1, 2) or not np.isfinite(t.get('cosine_offset', 0.))):
                raise ValueError('angular power through two and finite offset required')
        object.__setattr__(self, 'terms', terms)

    def _raw(self, q):
        q = np.asarray(q, dtype=float)
        if not np.all(np.isfinite(q)):
            raise ValueError('finite angular coordinates required')
        shape = q.shape
        q = q.ravel()
        x = np.broadcast_to(self.xyz_A, (len(q),) + self.xyz_A.shape).copy()
        g = np.zeros_like(x)
        h = np.zeros_like(x)
        indices = list(self.moving_atoms)
        v = (np.repeat(self.orbit_vector_A[None, :], len(indices), axis=0)
             if self.translation_only else self.xyz_A[indices] - self.pivot_A)
        parallel = np.outer(v @ self.axis, self.axis)
        perpendicular = v - parallel
        rotated = (parallel[None] + np.cos(q)[:, None, None] * perpendicular[None]
                   + np.sin(q)[:, None, None] * np.cross(self.axis, v)[None])
        x[:, indices] += rotated - v
        g[:, indices] = np.cross(self.axis, rotated)
        h[:, indices] = np.cross(self.axis, g[:, indices])
        out = [np.zeros(len(q)) for _ in range(3)]
        for t in self.terms:
            i, j = t['pair']
            r, rg, rh = _distance(x[:, i] - x[:, j], g[:, i] - g[:, j], h[:, i] - h[:, j])
            if t['family'] == 'exp':
                w = t['width_A_inverse']
                value = t['amplitude_cm'] * np.exp(-w * (r - t['reference_A']))
                jet = (value, -w * value * rg, value * (w * w * rg * rg - w * rh))
            else:
                p = t['power']
                value = t['amplitude_cm'] * (t['reference_A'] / r) ** p
                jet = (value, -p * value * rg / r,
                       value * (p * (p + 1) * (rg / r) ** 2 - p * rh / r))
            if 'angle' in t:
                i, j, k = t['angle']
                a, b = x[:, i] - x[:, j], x[:, k] - x[:, j]
                ag, bg = g[:, i] - g[:, j], g[:, k] - g[:, j]
                ah, bh = h[:, i] - h[:, j], h[:, k] - h[:, j]
                denominator = _product(_distance(a, ag, ah), _distance(b, bg, bh))
                numerator = (np.sum(a * b, axis=-1), np.sum(ag * b + a * bg, axis=-1),
                             np.sum(ah * b + 2 * ag * bg + a * bh, axis=-1))
                d, dg, dh = denominator
                c = numerator[0] / d
                cg = numerator[1] / d - c * dg / d
                ch = numerator[2] / d - 2 * numerator[1] * dg / d ** 2 + c * (2 * (dg / d) ** 2 - dh / d)
                c = c + t.get('cosine_offset', 0.)
                p = t['angular_power']
                angular = ((np.ones(len(q)), np.zeros(len(q)), np.zeros(len(q))) if p == 0
                           else (c, cg, ch) if p == 1
                           else (c * c, 2 * c * cg, 2 * (cg * cg + c * ch)))
                jet = _product(jet, angular)
            for order in range(3):
                out[order] += jet[order]
        return tuple(v.reshape(shape) for v in out)

    def evaluate(self, angles, *, derivative=0):
        if derivative not in (0, 1, 2):
            raise ValueError('angular derivatives through order two required')
        value = self._raw(angles)[derivative]
        return value - self._raw(0.)[0] if derivative == 0 else value

    def to_dict(self):
        return dict(schema='matrix.zaff.rigid_orbit_contact.v1',
                    xyz_A=self.xyz_A.tolist(), moving_atoms=list(self.moving_atoms),
                    axis=self.axis.tolist(), pivot_A=self.pivot_A.tolist(),
                    orbit_vector_A=self.orbit_vector_A.tolist(), terms=list(self.terms),
                    translation_only=bool(self.translation_only))

    @classmethod
    def from_dict(cls, data):
        if data.get('schema') != 'matrix.zaff.rigid_orbit_contact.v1':
            raise ValueError('unsupported rigid contact orbit schema')
        return cls(*(data[k] for k in ('xyz_A', 'moving_atoms', 'axis', 'pivot_A',
                                      'orbit_vector_A', 'terms', 'translation_only')))
