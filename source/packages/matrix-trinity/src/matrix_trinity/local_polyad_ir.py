"""Local curvilinear IR moments dressed outside, then rotated inside polyads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .algebraic_local_ir import _apply_polynomial, _dipole_polynomials
from .algebraic_local_x import _polynomials
from .gvpt2_spectroscopy import IR_TRANSITION_FACTOR
from .local_quadratic import LocalQuadraticModel


@dataclass(frozen=True)
class LocalPolyadIRResult:
    ground_root: int
    excited_roots: tuple[int, ...]
    transition_energies_cm: np.ndarray
    moment_orders_au: np.ndarray  # order, excited root, Cartesian component
    dipole_moments_au: np.ndarray  # excited root, Cartesian component
    ir_intensities_km_mol: np.ndarray
    root_assignment: tuple[int, ...]  # local state index -> operative root


def solve_local_polyad_ir(
    model: LocalQuadraticModel,
    basis_states: tuple[tuple[int, ...], ...],
    frequencies_cm: np.ndarray,
    kq_cm: np.ndarray,
    kp_cm: np.ndarray,
    f3_cm: np.ndarray,
    f4_cm: np.ndarray,
    g1_cm: np.ndarray,
    g2_cm: np.ndarray,
    dipole_first_au: np.ndarray,
    dipole_second_au: np.ndarray,
    dipole_third_au: np.ndarray,
    *,
    ground_state_index: int,
    excited_roots: tuple[int, ...] | None = None,
    denominator_tolerance_cm: float = 1e-6,
) -> LocalPolyadIRResult:
    """Calculate IR intensities after local K/Fermi/DD polyad diagonalization.

    ``basis_states`` follows the retained-state order of ``model``. Its K
    edges must equal the matrix elements of the supplied Kq/Kp field. The
    model's anharmonic edges are the effective Fermi/DD couplings from that
    same field. The external wave operator includes K+C at first order and
    Q+(K+C)^2 at second order. States in a selected polyad are excluded from
    that wave operator and mixed by the model's operative eigenvectors.
    Thus no selected internal K or Fermi/DD contribution is added twice.
    ``excited_roots`` are operative root indices, not local state indices.
    """
    frequencies = np.asarray(frequencies_cm, dtype=float)
    if frequencies.ndim != 1 or not len(frequencies) or not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise ValueError("local frequencies must be positive and finite")
    n = len(frequencies)
    count = len(basis_states)
    if (count != len(model.reference_energies_cm) or len(set(basis_states)) != count
            or any(len(state) != n or any(not isinstance(x, int) or x < 0 for x in state)
                   for state in basis_states)):
        raise ValueError("distinct local occupations must match the model basis")
    if not 0 <= ground_state_index < count or basis_states[ground_state_index] != (0,) * n:
        raise ValueError("ground_state_index must identify the local vacuum")
    tensors = tuple(np.asarray(value, dtype=float) for value in (
        kq_cm, kp_cm, f3_cm, f4_cm, g1_cm, g2_cm,
        dipole_first_au, dipole_second_au, dipole_third_au,
    ))
    shapes = ((n, n), (n, n), (n, n, n), (n, n, n, n),
              (n, n, n), (n, n, n, n), (3, n), (3, n, n), (3, n, n, n))
    if any(value.shape != shape or not np.all(np.isfinite(value))
           for value, shape in zip(tensors, shapes)):
        raise ValueError("local Hamiltonian or dipole tensor has invalid shape")
    kq, kp, f3, f4, g1, g2, mu1, mu2, mu3 = tensors
    if (not np.allclose(kq, kq.T) or not np.allclose(kp, kp.T)
            or np.any(np.diag(kq) != 0) or np.any(np.diag(kp) != 0)):
        raise ValueError("Kq and Kp must be symmetric and off diagonal")
    if not np.isfinite(denominator_tolerance_cm) or denominator_tolerance_cm <= 0:
        raise ValueError("denominator tolerance must be positive")
    for name, tensor, pairs in (
        ("F3", f3, ((0, 1), (1, 2))),
        ("F4", f4, ((0, 1), (1, 2), (2, 3))),
        ("G'", g1, ((0, 1),)),
        ("G''", g2, ((0, 1), (2, 3))),
        ("mu2", mu2, ((1, 2),)),
        ("mu3", mu3, ((1, 2), (2, 3))),
    ):
        if any(not np.allclose(tensor, np.swapaxes(tensor, a, b), rtol=1e-8, atol=1e-10)
               for a, b in pairs):
            raise ValueError(f"{name} tensor must be symmetric")

    k, c, q = _polynomials(kq, kp, f3, f4, g1, g2)
    w = dict(c)
    for monomial, coefficient in k.items():
        w[monomial] = w.get(monomial, 0j) + coefficient
    dipole = _dipole_polynomials(mu1, mu2, mu3)
    index = {state: i for i, state in enumerate(basis_states)}
    supplied_edges = {(min(i, j), max(i, j)): value
                      for i, j, value in (*model.polyad_k_edges, *model.external_k_edges)}
    for i, state in enumerate(basis_states):
        for target, value in _apply_polynomial(state, k).items():
            j = index.get(target)
            if j is not None and i < j and abs(value - supplied_edges.get((i, j), 0.0)) > max(1e-8, model.numerical_zero_cm):
                raise ValueError("model K edges differ from the supplied local quadratic field")
    for (i, j), value in supplied_edges.items():
        actual = _apply_polynomial(basis_states[i], k).get(basis_states[j], 0j)
        if abs(actual - value) > max(1e-8, model.numerical_zero_cm):
            raise ValueError("model K edges differ from the supplied local quadratic field")

    blocks = {i: (i,) for i in range(count)}
    for members in model.polyads:
        for i in members:
            blocks[i] = members

    def energy(state: tuple[int, ...]) -> float:
        return float(np.dot(frequencies, state))

    def divide(source: tuple[int, ...], target: tuple[int, ...], value: complex) -> complex:
        gap = energy(source) - energy(target)
        if abs(gap) <= denominator_tolerance_cm:
            raise ValueError("unresolved external local IR resonance; enlarge the polyad")
        return value / gap

    first: list[dict[tuple[int, ...], complex]] = []
    for i, state in enumerate(basis_states):
        excluded = {basis_states[j] for j in blocks[i]}
        first.append({target: divide(state, target, value)
                      for target, value in _apply_polynomial(state, w).items()
                      if target not in excluded and abs(value) > 1e-14})
    second: list[dict[tuple[int, ...], complex]] = []
    for i, state in enumerate(basis_states):
        excluded = {basis_states[j] for j in blocks[i]}
        numerator = _apply_polynomial(state, q)
        for intermediate, amplitude in first[i].items():
            for target, value in _apply_polynomial(intermediate, w).items():
                numerator[target] = numerator.get(target, 0j) + value * amplitude
        internal = _apply_polynomial(state, w)
        for j in blocks[i]:
            coupling = internal.get(basis_states[j], 0j)
            if abs(coupling) > 1e-14:
                for target, amplitude in first[j].items():
                    numerator[target] = numerator.get(target, 0j) - amplitude * coupling
        second.append({target: divide(state, target, value)
                       for target, value in numerator.items()
                       if target not in excluded and abs(value) > 1e-14})

    applied: dict[tuple[int, tuple[int, ...]], dict[tuple[int, ...], complex]] = {}

    def action(rank: int, state: tuple[int, ...], component: int) -> dict[tuple[int, ...], complex]:
        key = (rank * 3 + component, state)
        if key not in applied:
            applied[key] = _apply_polynomial(state, dipole[rank - 1][component])
        return applied[key]

    def element(left: dict[tuple[int, ...], complex], rank: int,
                right: dict[tuple[int, ...], complex], component: int) -> complex:
        return sum(np.conjugate(a) * b * action(rank, state_r, component).get(state_l, 0j)
                   for state_l, a in left.items() for state_r, b in right.items())

    def overlap(i: int, j: int) -> complex:
        return sum(np.conjugate(value) * first[j].get(state, 0j)
                   for state, value in first[i].items())

    operative = model.operative_spectrum()
    ground_root = operative.root_assignment[ground_state_index]
    roots = (tuple(i for i in range(count) if i != ground_root)
             if excited_roots is None else tuple(excited_roots))
    if len(set(roots)) != len(roots) or any(not 0 <= root < count or root == ground_root for root in roots):
        raise ValueError("excited roots must be distinct and exclude the ground root")
    vectors = np.eye(count)
    for members, block_vectors in zip(model.polyads, operative.eigenvectors):
        vectors[np.ix_(members, members)] = block_vectors
    moments = np.zeros((3, len(roots), 3), dtype=complex)
    ground_support = blocks[ground_root]
    for output, root in enumerate(roots):
        excited_support = blocks[root]
        for f in excited_support:
            sf, af, bf = {basis_states[f]: 1.0}, first[f], second[f]
            for g in ground_support:
                sg, ag, bg = {basis_states[g]: 1.0}, first[g], second[g]
                weight = np.conjugate(vectors[f, root]) * vectors[g, ground_root]
                if abs(weight) < 1e-15:
                    continue
                for component in range(3):
                    m0 = element(sf, 1, sg, component)
                    m1 = (element(sf, 2, sg, component)
                          + element(af, 1, sg, component) + element(sf, 1, ag, component))
                    m2 = (element(sf, 3, sg, component)
                          + element(af, 2, sg, component) + element(sf, 2, ag, component)
                          + element(bf, 1, sg, component) + element(sf, 1, bg, component)
                          + element(af, 1, ag, component))
                    m2 -= 0.5 * sum(overlap(f, h) * element(
                        {basis_states[h]: 1.0}, 1, sg, component) for h in blocks[f])
                    m2 -= 0.5 * sum(element(sf, 1, {basis_states[h]: 1.0}, component)
                                    * overlap(h, g) for h in blocks[g])
                    moments[:, output, component] += weight * np.array((m0, m1, m2))
    if np.max(abs(moments.imag), initial=0.0) > 1e-8:
        raise ValueError("local polyad IR moments have an unexpected imaginary component")
    orders = moments.real
    total = orders.sum(axis=0)
    transitions = operative.energies_cm[list(roots)] - operative.energies_cm[ground_root]
    if np.any(transitions <= 0):
        raise ValueError("selected polyad root is not above the ground root")
    intensities = IR_TRANSITION_FACTOR * transitions * np.sum(total**2, axis=1)
    return LocalPolyadIRResult(ground_root, roots, transitions, orders, total,
                               intensities, operative.root_assignment)


__all__ = ["LocalPolyadIRResult", "solve_local_polyad_ir"]
