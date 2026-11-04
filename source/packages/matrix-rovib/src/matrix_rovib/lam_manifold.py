"""Hermitian quasi-degenerate effective Hamiltonians for arbitrary LAM manifolds."""

from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from .lam_rotation import _rotational_operators, solve_lam_basis
from .lam_fit import (
    _lam_variable,
    _apply_lam_variables,
    _operator_parameter_values,
    _projected_coupled_factors,
)
from .lam_rotational_observables import _coefficients, _PAIRS
from .lam_measurements import response_factors
from .general_lam_operator import normalize_general_lam_operator_manifest


@dataclass(frozen=True)
class LAMEffectiveManifold:
    h0: np.ndarray
    linear: np.ndarray
    quadratic: np.ndarray
    active_states: tuple[int, ...]
    derivatives: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...] = ()
    parameter_paths: tuple[str, ...] = ()

    def hamiltonian(self, J, *, scale=1.0, derivative=None):
        h, l, q = (
            (self.h0, self.linear, self.quadratic)
            if derivative is None
            else self.derivatives[derivative]
        )
        rot = _rotational_operators(J, "Ir")
        result = np.kron(h, np.eye(2 * J + 1))
        for a in range(3):
            result += scale * np.kron(l[a], rot[a])
            for b in range(3):
                result += scale**2 * np.kron(q[a, b], rot[a] @ rot[b])
        return (result + result.conj().T) / 2

    def to_dict(self):
        def complex_array(a):
            return dict(real=a.real.tolist(), imag=a.imag.tolist())

        return dict(
            schema="matrix.lock.lam_effective_manifold.v1",
            active_states=list(self.active_states),
            h0=complex_array(self.h0),
            linear=complex_array(self.linear),
            quadratic=complex_array(self.quadratic),
            derivatives=[
                dict(h0=complex_array(h), linear=complex_array(l), quadratic=complex_array(q))
                for h, l, q in self.derivatives
            ],
            parameter_paths=list(self.parameter_paths),
            definition="Hermitian second-order quasi-degenerate Van Vleck Hamiltonian",
            scope="full active manifold; homogeneous rotational orders above two omitted",
        )


def effective_lam_manifold(
    h0, linear, quadratic, active_states, *, directions=(), gauge=None, include_gap_MHz=0.0
):
    h0 = np.asarray(h0, complex)
    linear = np.asarray(linear, complex)
    quadratic = np.asarray(quadratic, complex)
    size = len(h0)
    if (
        h0.shape != (size, size)
        or linear.shape != (3, size, size)
        or quadratic.shape != (3, 3, size, size)
    ):
        raise ValueError("invalid manifold operator shapes")
    for a in (
        h0,
        *linear,
        *(quadratic[a, b] - quadratic[b, a].conj().T for a in range(3) for b in range(3)),
    ):
        if not np.all(np.isfinite(a)):
            raise ValueError("nonfinite manifold operator")
    if not np.allclose(h0, h0.conj().T) or any(not np.allclose(l, l.conj().T) for l in linear):
        raise ValueError("manifold operators must be Hermitian")
    if not np.allclose(quadratic, quadratic.swapaxes(0, 1).swapaxes(2, 3).conj()):
        raise ValueError("ordered quadratic tensors violate Hermiticity")
    energy, u = np.linalg.eigh(h0)
    active = set(int(i) for i in active_states)
    if not active or min(active) < 0 or max(active) >= size or include_gap_MHz < 0:
        raise ValueError("invalid active manifold")
    tolerance = max(1.0e-8, 128 * np.spacing(max(1.0, np.max(abs(energy)))))
    while True:
        expanded = {
            j
            for i in active
            for j in range(size)
            if abs(energy[i] - energy[j]) <= max(tolerance, include_gap_MHz)
        }
        if expanded == active:
            break
        active = expanded
    indices = tuple(sorted(active))
    outside = tuple(i for i in range(size) if i not in active)
    p = u[:, indices]
    r = u[:, outside]
    ep = energy[list(indices)]
    eq = energy[list(outside)]
    gap = ep[None, :] - eq[:, None]
    if np.any(abs(gap) <= tolerance):
        raise ValueError("active/external manifold boundary is degenerate")
    hp = p.conj().T @ h0 @ p
    lpp = np.array([p.conj().T @ l @ p for l in linear])
    lpq = np.array([p.conj().T @ l @ r for l in linear])
    lqp = lpq.conj().transpose(0, 2, 1)
    response = lqp / gap[None, :, :]
    tensor = np.empty((3, 3, len(indices), len(indices)), complex)
    for a in range(3):
        for b in range(3):
            tensor[a, b] = p.conj().T @ quadratic[a, b] @ p + 0.5 * (
                lpq[a] @ response[b] + response[a].conj().T @ lqp[b]
            )
    derivatives = []
    for dh, dl, dq in directions:
        dh = np.asarray(dh, complex)
        dl = np.asarray(dl, complex)
        dq = np.asarray(dq, complex)
        normal = (r.conj().T @ dh @ p) / gap
        dp = r @ normal
        dr = -p @ normal.conj().T
        dhp = p.conj().T @ dh @ p
        dhq = r.conj().T @ dh @ r
        dlp = np.array(
            [
                p.conj().T @ dl[a] @ p + dp.conj().T @ linear[a] @ p + p.conj().T @ linear[a] @ dp
                for a in range(3)
            ]
        )
        dlqp = np.array(
            [
                r.conj().T @ dl[a] @ p + dr.conj().T @ linear[a] @ p + r.conj().T @ linear[a] @ dp
                for a in range(3)
            ]
        )
        # Differentiate the Sylvester equation, with no internal-manifold gaps.
        ds = np.array([(dlqp[a] + dhq @ response[a] - response[a] @ dhp) / gap for a in range(3)])
        dt = np.empty_like(tensor)
        for a in range(3):
            for b in range(3):
                direct = (
                    p.conj().T @ dq[a, b] @ p
                    + dp.conj().T @ quadratic[a, b] @ p
                    + p.conj().T @ quadratic[a, b] @ dp
                )
                dt[a, b] = direct + 0.5 * (
                    dlqp[a].conj().T @ response[b]
                    + lpq[a] @ ds[b]
                    + ds[a].conj().T @ lqp[b]
                    + response[a].conj().T @ dlqp[b]
                )
        derivatives.append((dhp, dlp, dt))
    g = np.eye(len(indices), dtype=complex) if gauge is None else np.asarray(gauge, complex)
    if g.shape != hp.shape or not np.allclose(g.conj().T @ g, np.eye(len(g)), atol=1.0e-10):
        raise ValueError("manifold gauge must be unitary")
    transform = lambda a: np.einsum("ia,...ij,jb->...ab", g.conj(), a, g, optimize=True)
    return LAMEffectiveManifold(
        transform(hp),
        transform(lpp),
        transform(tensor),
        indices,
        tuple(tuple(transform(a) for a in triple) for triple in derivatives),
    )


def lam_effective_manifold(
    surface, request, values=None, *, active_states=(0,), gauge=None, include_gap_MHz=0.0
):
    raw = request.get("operator_manifest")
    manifest = None if raw is None else normalize_general_lam_operator_manifest(raw)
    variables = tuple(
        _lam_variable(v, surface, k, {} if manifest is None else manifest.parameters_MHz)
        for k, v in enumerate(request["variables"])
    )
    values = np.asarray([v["initial"] for v in variables] if values is None else values, float)
    fitted = _apply_lam_variables(surface, variables, values)
    parameters = _operator_parameter_values(manifest, variables, values)
    options = request.get("options", {})
    basis = solve_lam_basis(
        fitted,
        state_count=int(request.get("lam_states", 12)),
        primitive_basis=options.get("primitive_basis", "auto"),
        fourier_cutoffs=options.get("fourier_cutoffs"),
    )
    solved = SimpleNamespace(
        basis=basis,
        blocks={0: None, 1: None},
        operator_parameters_MHz=parameters,
        representation="Ir",
    )

    def operators(factors):
        scalar = sum(
            (a * b[0, 0] for a, b in factors[0]),
            start=np.zeros((len(basis.energies_MHz),) * 2, complex),
        )
        coeff = _coefficients(factors[1], scalar)
        tensor = np.zeros((3, 3, len(scalar), len(scalar)), complex)
        for k, (a, b) in enumerate(_PAIRS):
            tensor[a, b] = tensor[b, a] = coeff[3 + k] / (1 if a == b else 2)
        return scalar, np.array(coeff[:3]), tensor

    base = operators(
        {
            J: _projected_coupled_factors(
                fitted,
                basis.vectors,
                J,
                basis,
                operator_manifest=manifest,
                operator_parameters_MHz=parameters,
                maximum_rotation_order=2,
            )
            for J in (0, 1)
        }
    )
    directions = []
    for column in range(len(values)):
        derivative, _ = response_factors(
            fitted,
            solved,
            variables,
            values,
            column,
            manifest,
            parameters,
            int(options.get("basis_response_states", 48)),
            maximum_rotation_order=2,
            parameter_reference_surface=surface,
        )
        directions.append(operators(derivative))
    from dataclasses import replace

    result = effective_lam_manifold(
        *base, active_states, directions=directions, gauge=gauge, include_gap_MHz=include_gap_MHz
    )
    return replace(result, parameter_paths=tuple(v["path"] for v in variables))
