from __future__ import annotations

from functools import lru_cache
from math import gcd
import numpy as np


def rotation_matrix(axis, theta):
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def reflection_matrix_from_normal(n):
    n = np.array(n, dtype=float)
    n = n / np.linalg.norm(n)
    return np.eye(3) - 2.0 * np.outer(n, n)


@lru_cache(maxsize=16)
def candidate_ops(max_n=6):
    ops = []
    ops.append(("E", np.eye(3)))
    ops.append(("i", -np.eye(3)))

    # principal mirror planes
    for axis, name in [(0, "sigma_yz"), (1, "sigma_xz"), (2, "sigma_xy")]:
        R = np.eye(3)
        R[axis, axis] = -1.0
        ops.append((name, R))

    # Cn axes along x,y,z (all powers)
    for n in range(2, max_n + 1):
        for k in range(1, n):
            # Keep only primitive powers: Cn^k with gcd(n,k)=1.
            # This avoids aliases like C8^4 (=C2) that inflate detected n.
            if gcd(n, k) != 1:
                continue
            theta = 2.0 * np.pi * k / n
            ops.append((f"C{n}z^{k}", rotation_matrix((0, 0, 1), theta)))
            ops.append((f"C{n}x^{k}", rotation_matrix((1, 0, 0), theta)))
            ops.append((f"C{n}y^{k}", rotation_matrix((0, 1, 0), theta)))

    # sigma planes containing z (rotate plane around z)
    for n in range(3, max_n + 1):
        for k in range(n):
            theta = np.pi * k / n
            nvec = (np.cos(theta), np.sin(theta), 0.0)
            ops.append((f"sigma_v_{n}_{k}", reflection_matrix_from_normal(nvec)))
            if n % 2 == 1 and 2 * n > max_n:
                shifted = theta + 0.5 * np.pi
                shifted_nvec = (np.cos(shifted), np.sin(shifted), 0.0)
                ops.append((f"sigma_v_{n}_{k + n}", reflection_matrix_from_normal(shifted_nvec)))

    for n in range(3, max_n + 1):
        order = 2 * n
        sigma_h = np.diag((1.0, 1.0, -1.0))
        for power in range(1, order, 2):
            theta = 2.0 * np.pi * power / order
            ops.append((f"sigma_h*C{order}z^{power}", sigma_h @ rotation_matrix((0, 0, 1), theta)))
            ops.append(
                (
                    f"C2_xy_{order}_{power}",
                    rotation_matrix((np.cos(theta / 2.0), np.sin(theta / 2.0), 0), np.pi),
                )
            )
            ops.append(
                (
                    f"sigma_v_{order}_{power}",
                    reflection_matrix_from_normal((np.cos(theta / 2.0), np.sin(theta / 2.0), 0.0)),
                )
            )

    # C2 axes in xy plane (D_n families)
    for n in range(2, max_n + 1):
        for k in range(n):
            theta = np.pi * k / n
            axis = (np.cos(theta), np.sin(theta), 0.0)
            ops.append((f"C2_xy_{n}_{k}", rotation_matrix(axis, np.pi)))

    # improper rotations Sn around z (rotation + reflection in xy)
    for n in range(3, max_n + 1):
        theta = 2.0 * np.pi / n
        R = rotation_matrix((0, 0, 1), theta)
        sigma_xy = np.eye(3)
        sigma_xy[2, 2] = -1.0
        ops.append((f"S{n}", sigma_xy @ R))

    # polyhedral rotations (T, O, I)
    diag_axes = [
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (-1, 1, 1),
    ]
    for a in diag_axes:
        ops.append(("C3_t", rotation_matrix(a, 2.0 * np.pi / 3.0)))
        ops.append(("C3_t2", rotation_matrix(a, 4.0 * np.pi / 3.0)))
    for axis in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        ops.append(("C2_t", rotation_matrix(axis, np.pi)))

    for axis in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        ops.append(("C4_o", rotation_matrix(axis, np.pi / 2.0)))
        ops.append(("C4_o2", rotation_matrix(axis, 3.0 * np.pi / 2.0)))
    for a in diag_axes:
        ops.append(("C3_o", rotation_matrix(a, 2.0 * np.pi / 3.0)))
        ops.append(("C3_o2", rotation_matrix(a, 4.0 * np.pi / 3.0)))
    edge_axes = [
        (0, 1, 1),
        (0, 1, -1),
        (1, 0, 1),
        (1, 0, -1),
        (1, 1, 0),
        (1, -1, 0),
    ]
    for a in edge_axes:
        ops.append(("C2_o", rotation_matrix(a, np.pi)))

    phi = (1.0 + np.sqrt(5.0)) / 2.0
    c5_axes = [
        (0, 1, phi),
        (0, -1, phi),
        (0, 1, -phi),
        (0, -1, -phi),
        (1, phi, 0),
        (-1, phi, 0),
    ]
    for a in c5_axes:
        ops.append(("C5_i", rotation_matrix(a, 2.0 * np.pi / 5.0)))
        ops.append(("C5_i2", rotation_matrix(a, 4.0 * np.pi / 5.0)))
        ops.append(("C5_i3", rotation_matrix(a, 6.0 * np.pi / 5.0)))
        ops.append(("C5_i4", rotation_matrix(a, 8.0 * np.pi / 5.0)))

    c3_axes = [
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (-1, 1, 1),
        (0, 1, 1 / phi),
        (0, -1, 1 / phi),
        (1, 0, 1 / phi),
        (-1, 0, 1 / phi),
        (1 / phi, 1, 0),
        (-1 / phi, 1, 0),
    ]
    for a in c3_axes:
        ops.append(("C3_i", rotation_matrix(a, 2.0 * np.pi / 3.0)))
        ops.append(("C3_i2", rotation_matrix(a, 4.0 * np.pi / 3.0)))

    c2_axes = [
        (0, 1, 0),
        (1, 0, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, -1, 0),
        (1, 0, 1),
        (1, 0, -1),
        (0, 1, 1),
        (0, 1, -1),
        (phi, 1 / phi, 0),
        (phi, -1 / phi, 0),
        (0, phi, 1 / phi),
        (0, phi, -1 / phi),
        (1 / phi, 0, phi),
        (-1 / phi, 0, phi),
    ]
    for a in c2_axes:
        ops.append(("C2_i", rotation_matrix(a, np.pi)))

    return ops
