from __future__ import annotations

import numpy as np


def norm(v):
    return np.linalg.norm(v)


def unit(v, eps=1e-12):
    n = norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def angle(i, j, k, coords):
    v1 = coords[i] - coords[j]
    v2 = coords[k] - coords[j]
    u1 = unit(v1)
    u2 = unit(v2)
    dot = np.clip(np.dot(u1, u2), -1.0, 1.0)
    return np.arccos(dot)


def dihedral(i, j, k, l, coords):
    b1 = coords[i] - coords[j]
    b2 = coords[k] - coords[j]
    b3 = coords[l] - coords[k]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2u = unit(b2)
    x = np.dot(n1, n2)
    y = np.dot(np.cross(n1, n2), b2u)
    return np.arctan2(y, x)


def oop(i, j, k, l, coords):
    v = coords[i] - coords[j]
    a = coords[k] - coords[j]
    b = coords[l] - coords[j]
    n = np.cross(a, b)
    nn = norm(n)
    if nn < 1e-12:
        return 0.0
    n = n / nn
    vv = norm(v)
    if vv < 1e-12:
        return 0.0
    return np.arcsin(np.dot(v, n) / vv)


def linear_components(i, j, k, coords):
    # Two-component linear bending coordinate around j.
    u = unit(coords[i] - coords[j])
    v = unit(coords[k] - coords[j])
    # choose a reference axis not parallel to u
    axis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, u)) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])
    e1 = unit(np.cross(u, axis))
    e2 = unit(np.cross(u, e1))
    b = v + u  # small bending vector near linear
    return np.dot(b, e1), np.dot(b, e2)
