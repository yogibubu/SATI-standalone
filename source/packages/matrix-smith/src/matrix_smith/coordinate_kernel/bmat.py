from __future__ import annotations

import numpy as np


def bond_grad(i, j, coords):
    rij = coords[i] - coords[j]
    r = np.linalg.norm(rij)
    if r < 1e-12:
        return np.zeros_like(coords)
    g = np.zeros_like(coords)
    g[i] = rij / r
    g[j] = -rij / r
    return g


def angle_grad(i, j, k, coords):
    # angle at j
    rji = coords[i] - coords[j]
    rjk = coords[k] - coords[j]
    rji_norm = np.linalg.norm(rji)
    rjk_norm = np.linalg.norm(rjk)
    if rji_norm < 1e-12 or rjk_norm < 1e-12:
        return np.zeros_like(coords)
    u = rji / rji_norm
    v = rjk / rjk_norm
    cosang = np.clip(np.dot(u, v), -1.0, 1.0)
    sinang = np.sqrt(max(1.0 - cosang * cosang, 1e-16))

    g = np.zeros_like(coords)
    dtheta_du = (-v + cosang * u) / sinang
    dtheta_dv = (-u + cosang * v) / sinang
    g[i] = dtheta_du / rji_norm
    g[k] = dtheta_dv / rjk_norm
    g[j] = -(g[i] + g[k])
    return g


def dihedral_grad(i, j, k, l, coords):
    # Algorithmic differentiation for exact gradient of geometry.dihedral
    class Dual:
        __slots__ = ("val", "der")

        def __init__(self, val, der):
            self.val = float(val)
            self.der = der

        def __add__(self, other):
            other = other if isinstance(other, Dual) else Dual(other, 0.0 * self.der)
            return Dual(self.val + other.val, self.der + other.der)

        def __sub__(self, other):
            other = other if isinstance(other, Dual) else Dual(other, 0.0 * self.der)
            return Dual(self.val - other.val, self.der - other.der)

        def __mul__(self, other):
            other = other if isinstance(other, Dual) else Dual(other, 0.0 * self.der)
            return Dual(self.val * other.val, self.val * other.der + other.val * self.der)

        def __truediv__(self, other):
            other = other if isinstance(other, Dual) else Dual(other, 0.0 * self.der)
            inv = 1.0 / other.val
            return Dual(self.val * inv, (self.der - self.val * other.der * inv) * inv)

        def __neg__(self):
            return Dual(-self.val, -self.der)

    def d_sqrt(x):
        v = np.sqrt(x.val)
        return Dual(v, x.der / (2.0 * v))

    def d_atan2(y, x):
        denom = x.val * x.val + y.val * y.val
        return Dual(np.arctan2(y.val, x.val), (x.val * y.der - y.val * x.der) / denom)

    def d_dot(a, b):
        v = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        return v

    def d_cross(a, b):
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]

    def d_norm(a):
        return d_sqrt(d_dot(a, a))

    def d_unit(a):
        n = d_norm(a)
        return [ai / n for ai in a]

    # build dual variables
    base = coords[[i, j, k, l]].reshape(-1)
    nvar = 12
    duals = []
    for idx, v in enumerate(base):
        der = np.zeros(nvar)
        der[idx] = 1.0
        duals.append(Dual(v, der))

    def vec(offset):
        return [duals[offset], duals[offset + 1], duals[offset + 2]]

    ri = vec(0)
    rj = vec(3)
    rk = vec(6)
    rl = vec(9)

    b1 = [ri[m] - rj[m] for m in range(3)]
    b2 = [rk[m] - rj[m] for m in range(3)]
    b3 = [rl[m] - rk[m] for m in range(3)]

    n1 = d_cross(b1, b2)
    n2 = d_cross(b2, b3)
    b2u = d_unit(b2)

    x = d_dot(n1, n2)
    y = d_dot(d_cross(n1, n2), b2u)
    phi = d_atan2(y, x)

    g_local = phi.der.reshape(4, 3)
    g = np.zeros_like(coords)
    g[i] = g_local[0]
    g[j] = g_local[1]
    g[k] = g_local[2]
    g[l] = g_local[3]
    return g


def _cross_matrix(a):
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def oop_grad(i, j, k, l, coords):
    v = coords[i] - coords[j]
    a = coords[k] - coords[j]
    b = coords[l] - coords[j]

    vv = np.linalg.norm(v)
    if vv < 1e-12:
        return np.zeros_like(coords)

    n = np.cross(a, b)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return np.zeros_like(coords)

    u = v / vv
    nh = n / nn
    s = np.dot(u, nh)
    s = np.clip(s, -0.999999, 0.999999)
    denom = np.sqrt(1.0 - s * s)
    if denom < 1e-12:
        return np.zeros_like(coords)

    Ju = (np.eye(3) - np.outer(u, u)) / vv
    Jn = (np.eye(3) - np.outer(nh, nh)) / nn

    g_v = Ju @ nh
    g_n = Jn @ u

    g = np.zeros_like(coords)

    g[i] = g_v
    Mj = _cross_matrix(b) - _cross_matrix(a)
    g[j] = -g_v + Mj.T @ g_n
    Mk = -_cross_matrix(b)
    g[k] = Mk.T @ g_n
    Ml = _cross_matrix(a)
    g[l] = Ml.T @ g_n

    g *= 1.0 / denom
    return g


def linear_grad(i, j, k, coords, mode=-1):
    rji = coords[i] - coords[j]
    rjk = coords[k] - coords[j]
    rji_n = np.linalg.norm(rji)
    rjk_n = np.linalg.norm(rjk)
    if rji_n < 1e-12 or rjk_n < 1e-12:
        return np.zeros_like(coords)

    u = rji / rji_n
    v = rjk / rjk_n

    axis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, u)) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])

    w = np.cross(u, axis)
    wn = np.linalg.norm(w)
    if wn < 1e-12:
        return np.zeros_like(coords)
    e1 = w / wn
    e2 = np.cross(u, e1)

    b = v + u

    I = np.eye(3)
    Ju = (I - np.outer(u, u)) / rji_n
    Jv = (I - np.outer(v, v)) / rjk_n

    P = (I / wn) - np.outer(w, w) / (wn**3)
    A = -_cross_matrix(axis)
    de1_du = P @ A

    de2_du = _cross_matrix(e1) + _cross_matrix(u) @ de1_du

    if mode == -1:
        dcd_u = e1 + de1_du.T @ b
        dcd_v = e1
    else:
        dcd_u = e2 + de2_du.T @ b
        dcd_v = e2

    g = np.zeros_like(coords)
    g[i] = Ju.T @ dcd_u
    g[k] = Jv.T @ dcd_v
    g[j] = -(g[i] + g[k])
    return g


def finite_diff_grad(func, coords, h=1e-4):
    nat = coords.shape[0]
    g = np.zeros_like(coords)
    for a in range(nat):
        for c in range(3):
            cp = coords.copy()
            cm = coords.copy()
            cp[a, c] += h
            cm[a, c] -= h
            fp = func(cp)
            fm = func(cm)
            g[a, c] = (fp - fm) / (2.0 * h)
    return g
