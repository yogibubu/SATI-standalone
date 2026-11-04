"""Fixed-geometry building blocks for the native NN-xTB MACE encoder."""

from __future__ import annotations

from functools import lru_cache
from math import factorial, pi, sqrt

import numpy as np


def real_spherical_harmonics(unit: np.ndarray, l_max: int) -> np.ndarray:
    """Real, orthonormal harmonics in the model's m=-l..l ordering."""
    if l_max < 0 or l_max > 2:
        raise NotImplementedError("native MACE harmonics currently support l_max <= 2")
    u = np.asarray(unit, dtype=float)
    if u.ndim != 2 or u.shape[1] != 3:
        raise ValueError("unit vectors must have shape (n, 3)")
    x, y, z = u.T
    parts = [np.full(len(u), 1.0 / sqrt(4.0 * pi))]
    if l_max >= 1:
        c = sqrt(3.0 / (4.0 * pi))
        parts.extend((c * y, c * z, c * x))
    if l_max >= 2:
        c = sqrt(15.0 / pi) / 2.0
        parts.extend((c * x * y, c * y * z,
                      sqrt(5.0 / pi) / 4.0 * (3.0 * z * z - 1.0),
                      c * x * z, c / 2.0 * (x * x - y * y)))
    return np.column_stack(parts)


def _complex_cg(j1: int, m1: int, j2: int, m2: int, j3: int, m3: int) -> float:
    if (m1 + m2 != m3 or abs(j1 - j2) > j3 or j3 > j1 + j2
            or abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3):
        return 0.0
    f = factorial
    pref = sqrt((2 * j3 + 1) * f(j1 + j2 - j3) * f(j1 - j2 + j3)
                * f(-j1 + j2 + j3) / f(j1 + j2 + j3 + 1)
                * f(j3 + m3) * f(j3 - m3)
                * f(j1 - m1) * f(j1 + m1)
                * f(j2 - m2) * f(j2 + m2))
    result = 0.0
    for k in range(max(0, j2 - j3 - m1, j1 + m2 - j3),
                   min(j1 + j2 - j3, j1 - m1, j2 + m2) + 1):
        result += (-1.0 if k % 2 else 1.0) / (
            f(k) * f(j1 + j2 - j3 - k) * f(j1 - m1 - k)
            * f(j2 + m2 - k) * f(j3 - j2 + m1 + k)
            * f(j3 - j1 - m2 + k))
    return pref * result


def _real_basis(m_real: int, m_complex: int) -> complex:
    if m_real == 0:
        return 1.0 if m_complex == 0 else 0.0
    a = abs(m_real)
    if m_real > 0:
        if m_complex == a:
            return (-1.0 if a % 2 else 1.0) / sqrt(2.0)
        if m_complex == -a:
            return 1.0 / sqrt(2.0)
    else:
        if m_complex == -a:
            return 1j / sqrt(2.0)
        if m_complex == a:
            return (-1j if a % 2 == 0 else 1j) / sqrt(2.0)
    return 0.0


@lru_cache(maxsize=None)
def real_cg(l1: int, l2: int, l3: int) -> np.ndarray:
    """Real Clebsch-Gordan coupling in the model's spherical basis."""
    if min(l1, l2, l3) < 0 or max(l1, l2, l3) > 2:
        raise ValueError("coupling supports angular momenta 0..2")
    table = np.zeros((2 * l3 + 1, 2 * l1 + 1, 2 * l2 + 1), dtype=complex)
    for m3 in range(-l3, l3 + 1):
        for m1 in range(-l1, l1 + 1):
            for m2 in range(-l2, l2 + 1):
                value = 0j
                for c1 in range(-l1, l1 + 1):
                    u1 = _real_basis(m1, c1)
                    if u1 == 0:
                        continue
                    for c2 in range(-l2, l2 + 1):
                        u2 = _real_basis(m2, c2)
                        c3 = c1 + c2
                        if u2 == 0 or abs(c3) > l3:
                            continue
                        value += (u1 * u2 * _real_basis(m3, c3).conjugate()
                                  * _complex_cg(l1, c1, l2, c2, l3, c3))
                table[m3 + l3, m1 + l1, m2 + l2] = value
    take_real = np.max(np.abs(table.real)) >= np.max(np.abs(table.imag))
    component = table.real if take_real else table.imag
    if np.max(np.abs(table.imag if take_real else table.real)) > 1e-10:
        raise ValueError("mixed complex phase in real Clebsch-Gordan table")
    component = np.where(np.abs(component) > 1e-12, component, 0.0)
    component.setflags(write=False)
    return component


def _paths(l_max: int):
    for l1 in range(l_max + 1):
        for l2 in range(l_max + 1):
            for out in range(abs(l1 - l2), min(l1 + l2, l_max) + 1):
                yield l1, l2, out


def _pair_paths(l_max: int):
    for a in range(l_max + 1):
        for b in range(a, l_max + 1):
            for out in range(abs(a - b), min(a + b, l_max) + 1):
                yield a, b, out


def _block(l: int) -> slice:
    return slice(l * l, (l + 1) * (l + 1))


def mace_layer(graph: dict, tensors: dict[str, np.ndarray], h: np.ndarray, *,
               layer: int, l_max: int, nu_max: int) -> np.ndarray:
    """Evaluate one equivariant MACE layer from a supplied atomic feature state.

    Returns ``(atoms, (l_max+1)^2, channels)``. This is a descriptor stage;
    it does not modify the GFN2 Hamiltonian or produce molecular energies.
    """
    z = graph["atomic_numbers"]
    source, target = graph["source"], graph["target"]
    radial, harmonics = graph["radial_features"], graph["spherical_harmonics"]
    n_atoms = len(z)
    n_sh = (l_max + 1) ** 2
    if h.shape != (n_atoms, n_sh, tensors["embed"].shape[1]):
        raise ValueError("MACE feature state has incompatible shape")
    prefix = f"t{layer}"
    a_feat = np.zeros_like(h)
    for l1, l2, out in _paths(l_max):
        weights = radial @ tensors[f"{prefix}.radial.{l1}{l2}{out}"]
        coupled = np.einsum(
            "ea,ebc,oab->eoc", harmonics[:, _block(l1)],
            h[target, _block(l2)], real_cg(l1, l2, out), optimize=True,
        )
        np.add.at(a_feat[:, _block(out)], source, coupled * weights[:, None, :])
    message = np.zeros_like(h)
    for l in range(l_max + 1):
        message[:, _block(l)] += a_feat[:, _block(l)] * tensors[f"{prefix}.msg_gain.nu1.{l}"]
    pairs = {}
    if nu_max >= 2:
        for la, lb, out in _pair_paths(l_max):
            coupled = np.einsum(
                "nac,nbc,oab->noc", a_feat[:, _block(la)],
                a_feat[:, _block(lb)], real_cg(la, lb, out), optimize=True,
            )
            pairs[(la, lb, out)] = coupled
            message[:, _block(out)] += coupled * tensors[f"{prefix}.msg_gain.nu2.{la}{lb}{out}"]
    if nu_max >= 3:
        for (la, lb, l12), left in pairs.items():
            for lc in range(l_max + 1):
                for out in range(abs(l12 - lc), min(l12 + lc, l_max) + 1):
                    coupled = np.einsum(
                        "nac,nbc,oab->noc", left, a_feat[:, _block(lc)],
                        real_cg(l12, lc, out), optimize=True,
                    )
                    message[:, _block(out)] += (
                        coupled * tensors[f"{prefix}.msg_gain.nu3.{la}{lb}{l12}.{lc}{out}"])
    updated = np.zeros_like(h)
    for l in range(l_max + 1):
        updated[:, _block(l)] = (
            message[:, _block(l)] @ tensors[f"{prefix}.update.U{l}"]
            + h[:, _block(l)] * tensors[f"{prefix}.vgain{l}"][z - 1, None, :]
        )
    return updated


def first_mace_layer(graph: dict, tensors: dict[str, np.ndarray], *,
                     l_max: int, nu_max: int) -> np.ndarray:
    """Evaluate the first MACE layer from the element embeddings."""
    z = graph["atomic_numbers"]
    channels = tensors["embed"].shape[1]
    h = np.zeros((len(z), (l_max + 1) ** 2, channels))
    h[:, 0] = tensors["embed"][z - 1]
    return mace_layer(graph, tensors, h, layer=0, l_max=l_max, nu_max=nu_max)


def mace_layer_states(graph: dict, tensors: dict[str, np.ndarray], *,
                      layers: int, l_max: int, nu_max: int) -> tuple[np.ndarray, ...]:
    """Return each layer state so fixed-geometry parity can be checked layerwise."""
    z = graph["atomic_numbers"]
    channels = tensors["embed"].shape[1]
    h = np.zeros((len(z), (l_max + 1) ** 2, channels))
    h[:, 0] = tensors["embed"][z - 1]
    states = []
    for layer in range(layers):
        h = mace_layer(graph, tensors, h, layer=layer, l_max=l_max, nu_max=nu_max)
        states.append(h)
    return tuple(states)
