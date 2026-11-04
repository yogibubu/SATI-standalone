"""Analytic mixed CPSCF response for the baseline resident GFN2 surface.

AO mixed integrals come from the native analytic recurrence. Mixed directional
algebra differentiates the complete baseline Fock, including CAMMs, AES/AXC,
CN, charge-dependent two-body D4 and self-consistent ALPB. No displaced
SCF calculations, derivative steps, or automatic-differentiation dependency
enter this implementation. The route is a dense molecular reference path.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ._mixed_jet import MixedJet as Jet, stack, contract


def _coordination(r, radii, *, d4=False, electronegativities=None):
    n = r.v.shape[0]
    out = Jet.constant(np.zeros(n), r.n)
    for i in range(n):
        for j in range(i):
            distance = ((r[i] - r[j]) ** 2).sum() ** 0.5
            critical = 4 / 3 * (radii[i] + radii[j])
            if d4:
                factor = 4.10451 * np.exp(
                    -((abs(electronegativities[i] - electronegativities[j]) + 19.08857) ** 2)
                    / (2 * 11.28174**2)
                )
                contribution = 0.5 * (1 + (-7.5 * (distance - critical) / critical).erf()) * factor
            else:
                contribution = (
                    1
                    / (1 + (-10 * (critical / distance - 1)).exp())
                    / (1 + (-20 * ((critical + 2) / distance - 1)).exp())
                )
            out[i] = out[i] + contribution
            out[j] = out[j] + contribution
    return out


def _integrals(calculator, state, context, direction):
    basis = context["basis"]
    pairs = context["integrals"]
    D = state.coordinates_bohr.size
    native = calculator._active_native_context
    if native is None:
        raise RuntimeError("analytic mixed AO recurrence requires the native integral context")
    native.configure_spherical(
        basis.spherical_cartesian_offsets,
        basis.spherical_cartesian_indices,
        basis.spherical_cartesian_coefficients,
        pairs.rows,
        pairs.columns,
    )
    centers = np.array([x.center_bohr for x in basis.cartesian_orbitals])
    velocities = direction[[x.atom_index for x in basis.cartesian_orbitals]]
    left2, right2 = native.spherical_directional_seconds(
        centers, np.zeros(3), velocities, context["domains"].native_threads
    )
    value = np.concatenate(
        [pairs.overlap[:, None], pairs.dipole, pairs.second.reshape(-1, 9)], axis=1
    )
    first = np.zeros((D, *value.shape))
    mixed = np.zeros_like(first)
    assignment = np.asarray(basis.spherical_atom)
    for k in range(3):
        for p, (row, col) in enumerate(zip(pairs.rows, pairs.columns)):
            first[3 * assignment[row] + k, p] += pairs.left_derivatives[p, :, k]
            first[3 * assignment[col] + k, p] += pairs.right_derivatives[p, :, k]
            mixed[3 * assignment[row] + k, p] += left2[p, :, k]
            mixed[3 * assignment[col] + k, p] += right2[p, :, k]
    packed = Jet(value, first, np.einsum("d,dpc->pc", direction.ravel(), first), mixed)
    dense = Jet.constant(np.zeros((state.nao, state.nao, 13)), D)
    dense[pairs.rows, pairs.columns] = packed
    dense[pairs.columns, pairs.rows] = packed
    return dense[:, :, 0], dense[:, :, 1:4], dense[:, :, 4:].reshape(state.nao, state.nao, 3, 3)


def _d4_potential(calculator, context, r, charges):
    z = lambda x: Jet.constant(x, r.n)
    kernel = context["kernel"]
    prepared = kernel.d4_prepared
    parameters = kernel.d4_parameters
    # The baseline calculator uses charge-independent post-SCF ATM. Its
    # ground-state Hessian includes that term; it contributes no orbital Fock.
    models = kernel.d4_models
    n = len(models)
    from .calculator import _covalent_radii

    cn = _coordination(
        r,
        _covalent_radii()[np.asarray(calculator.atomic_numbers) - 1],
        d4=True,
        electronegativities=calculator._electronegativities,
    )
    alpha = []
    dalpha = []
    for i, model in enumerate(models):
        raw = [
            sum(
                (-6 * order * (cn[i] - ref.coordination) ** 2).exp()
                for order in range(1, ref.gaussian_count + 1)
            )
            for ref in model.references
        ]
        total = sum(raw)
        if total.v <= np.finfo(float).tiny:
            raise ValueError(
                "D4 reference extrapolation is not differentiable in this mixed-response route"
            )
        ai = z(np.zeros(len(parameters.frequency_weights)))
        di = z(np.zeros_like(ai.v))
        denominator = model.effective_nuclear_charge + charges[i]
        if denominator.v <= 0:
            raise ValueError("nonpositive D4 effective charge")
        for weight, ref in zip(raw, model.references):
            numerator = model.effective_nuclear_charge + ref.charge
            exponential = (2 * model.hardness * (1 - numerator / denominator)).exp()
            scale = (3 * (1 - exponential)).exp()
            derivative = (
                scale * (-3 * exponential) * (2 * model.hardness * numerator / denominator**2)
            )
            ai = ai + weight / total * scale * np.asarray(ref.polarizabilities)
            di = di + weight / total * derivative * np.asarray(ref.polarizabilities)
        alpha.append(ai)
        dalpha.append(di)
    potential = z(np.zeros(n))
    for i in range(n):
        for j in range(i):
            distance = ((r[i] - r[j]) ** 2).sum() ** 0.5
            ratio = prepared.ratio[i, j]
            cutoff = parameters.a1 * np.sqrt(ratio) + parameters.a2
            factor = (
                parameters.s6 / (distance**6 + cutoff**6)
                + parameters.s8 * ratio / (distance**8 + cutoff**8)
                + parameters.s10 * (49 / 40) * ratio**2 / (distance**10 + cutoff**10)
            )
            potential[i] = (
                potential[i]
                - 3 / np.pi * (dalpha[i] * alpha[j] * prepared.frequency_weights).sum() * factor
            )
            potential[j] = (
                potential[j]
                - 3 / np.pi * (dalpha[j] * alpha[i] * prepared.frequency_weights).sum() * factor
            )
    return potential


def _fock(calculator, state, context, r, S, dipole, second, P):
    from .hamiltonian import (
        _orbital_value,
        _atom_values,
        _h0_global_scalings,
        _angular_classes,
        _slater_pair_scaling,
    )

    D = r.n
    n = len(state.atomic_numbers)
    nao = state.nao
    z = lambda x: Jet.constant(x, D)
    basis = context["basis"]
    partition = context["partition"]
    numbers = calculator.atomic_numbers
    atom = np.asarray(basis.spherical_atom)
    shell = np.asarray(partition.orbital_shell)
    atom_values = _atom_values(numbers, context["atom_parameters"])
    cn = _coordination(r, np.array([x.covalent_radius_bohr for x in atom_values]))
    metadata = calculator._orbital_metadata
    pars = [
        _orbital_value(numbers, x.atom_index, x.shell, context["orbital_parameters"])
        for x in metadata
    ]
    levels = stack(
        [x.level_hartree - x.coordination_shift_hartree * cn[atom[i]] for i, x in enumerate(pars)]
    )
    angular, enscale = _h0_global_scalings(None)
    classes = _angular_classes([x.shell for x in metadata])
    H = z(np.zeros((nao, nao)))
    for i in range(nao):
        H[i, i] = levels[i]
        for j in range(i):
            if atom[i] == atom[j]:
                continue
            ai, aj = atom[i], atom[j]
            distance = ((r[ai] - r[aj]) ** 2).sum() ** 0.5
            root = (
                distance
                / (
                    0.5
                    * (
                        atom_values[ai].d4_covalent_radius_bohr
                        + atom_values[aj].d4_covalent_radius_bohr
                    )
                )
            ) ** 0.5
            exponent = _slater_pair_scaling(
                metadata[i].slater_exponent, metadata[j].slater_exponent
            )
            delta = atom_values[ai].electronegativity - atom_values[aj].electronegativity
            value = (
                0.5
                * angular[classes[i], classes[j]]
                * (levels[i] + levels[j])
                * S[i, j]
                * exponent
                * (1 + enscale * delta**2)
                * (1 + pars[i].polynomial * root)
                * (1 + pars[j].polynomial * root)
            )
            H[i, j] = value
            H[j, i] = value
    # Endpoint CAMMs: ordered AO rows give the corresponding left endpoint.
    centers = r[atom]
    dmu = centers[:, None, :] * S[:, :, None] - dipole
    raw = (
        dipole[:, :, :, None] * centers[:, None, None, :]
        + centers[:, None, :, None] * dipole[:, :, None, :]
        - S[:, :, None, None] * centers[:, None, :, None] * centers[:, None, None, :]
        - second
    )
    trace = raw[:, :, 0, 0] + raw[:, :, 1, 1] + raw[:, :, 2, 2]
    quad = 1.5 * raw - 0.5 * trace[:, :, None, None] * np.eye(3)
    population = P * S
    shell_q = stack(
        [
            partition.reference_population[k] - population[shell == k].sum()
            for k in range(partition.nshells)
        ]
    )
    charges = stack([calculator._valence[k] - population[atom == k].sum() for k in range(n)])
    mu = stack([contract("ij,ijk->k", P[atom == k], dmu[atom == k]) for k in range(n)])
    theta = stack([contract("ij,ijkl->kl", P[atom == k], quad[atom == k]) for k in range(n)])
    hardness = np.asarray(partition.hardness_hartree)
    average = 0.5 * (hardness[:, None] + hardness[None, :])
    sr = r[np.asarray(partition.shell_atom)]
    gamma = (((sr[:, None, :] - sr[None, :, :]) ** 2).sum(axis=2) + average**-2) ** -0.5
    if partition.electrostatic_profile is not None:
        gamma = partition.electrostatic_profile.modify_jet(gamma, r, partition)
    shell_pot = (
        contract("ij,j->i", gamma, shell_q) + np.asarray(partition.third_order_hartree) * shell_q**2
    )
    radii = calculator._base_multipole_radii + (5 - calculator._base_multipole_radii) / (
        1 + (-4 * (cn - calculator._valence_coordination - 1.2)).exp()
    )
    qpot = z(np.zeros(n))
    mupot = 2 * np.array([x.dipole_xc for x in calculator._element_values])[:, None] * mu
    tpot = (
        2 * np.array([x.quadrupole_xc for x in calculator._element_values])[:, None, None] * theta
    )
    for i in range(n):
        for j in range(i + 1, n):
            delta = r[j] - r[i]
            distance2 = (delta**2).sum()
            distance = distance2**0.5
            radius = 0.5 * (radii[i] + radii[j])
            f3 = distance**-3 / (1 + 6 * (radius / distance) ** 3)
            f5 = distance**-5 / (1 + 6 * (radius / distance) ** 4)
            left = (mu[i] * delta).sum()
            right = (mu[j] * delta).sum()
            leftq = contract("i,ij,j->", delta, theta[i], delta)
            rightq = contract("i,ij,j->", delta, theta[j], delta)
            qpot[i] = qpot[i] - f3 * right + f5 * rightq
            qpot[j] = qpot[j] + f3 * left + f5 * leftq
            mupot[i] = (
                mupot[i] + f3 * charges[j] * delta + f5 * (-3 * delta * right + mu[j] * distance2)
            )
            mupot[j] = (
                mupot[j] - f3 * charges[i] * delta + f5 * (-3 * delta * left + mu[i] * distance2)
            )
            tpot[i] = tpot[i] + f5 * charges[j] * delta[:, None] * delta[None, :]
            tpot[j] = tpot[j] + f5 * charges[i] * delta[:, None] * delta[None, :]
    d4pot = _d4_potential(calculator, context, r, charges)
    if calculator.alpb_parameters is not None:
        from .alpb_mixed_response import alpb_mixed_interaction

        d4pot = d4pot + contract(
            "ij,j->i",
            alpb_mixed_interaction(r, state.atomic_numbers, calculator.alpb_parameters),
            charges,
        )
    endpoint = (
        -qpot[atom, None] * S
        + contract("ik,ijk->ij", mupot[atom], dmu)
        + contract("ikl,ijkl->ij", tpot[atom], quad)
    )
    pot = shell_pot[shell] + d4pot[atom]
    F = H - 0.5 * S * (pot[:, None] + pot[None, :]) + 0.5 * (endpoint + endpoint.T)
    return F


@dataclass(frozen=True)
class MixedOrbitalResponse:
    overlap: Jet
    coefficients: Jet
    energies: Jet
    fock_primal_error: float
    fock_first_error: float
    eigen_residual: float
    normalization_residual: float


class MolecularSecondResponse:
    """Frozen molecular reference and shared CPSCF matrix for Hessian actions."""

    def __init__(self, calculator, state, *, tolerance=1e-8):
        if (
            getattr(calculator, "nn_enabled", False)
            or getattr(calculator, "_active_nn_state", None) is not None
        ):
            raise NotImplementedError(
                "mixed molecular CPSCF currently supports the baseline GFN2 surface, not NN overlays"
            )
        if len(calculator.point_charges_electron):
            raise NotImplementedError(
                "molecular TDA Hessian point-charge response is not qualified: the existing dense first-response and ground Hessian routes need embedding completion"
            )
        if state.nmo != state.nao:
            raise NotImplementedError(
                "mixed molecular CPSCF requires a complete canonical AO/MO basis"
            )
        self.calculator = calculator
        self.state = state
        self.tolerance = float(tolerance)
        self.context = calculator._prepare_density_response_gradient_context(state)
        from .dispersion import D4PreparedGeometry

        if not isinstance(self.context["kernel"].d4_prepared, D4PreparedGeometry):
            raise NotImplementedError(
                "mixed molecular CPSCF currently requires dense molecular D4 geometry"
            )
        self.first = calculator.ground_state_coupled_response(
            state, degeneracy_tolerance_hartree=tolerance
        )
        self.kernel = self.context["kernel"]
        C = state.coefficients
        eps = state.orbital_energies_hartree
        occ = state.alpha_occupations + state.beta_occupations
        if np.max(abs(occ - np.rint(occ))) > 1e-10:
            raise ValueError("mixed CPSCF requires integer occupations")
        self.occ = occ
        self.left, self.right = np.triu_indices(state.nmo, 1)
        self.gaps = eps[self.right] - eps[self.left]
        # Individual canonical derivatives are undefined within a degenerate
        # orbital block. No arbitrary split/gauge is silently substituted.
        if np.any(abs(self.gaps) <= tolerance):
            raise ValueError("mixed canonical CPSCF requires resolved orbital gaps")
        self.active = abs(occ[self.right] - occ[self.left]) > 1e-12
        left = self.left[self.active]
        right = self.right[self.active]
        gaps = self.gaps[self.active]
        self.response_matrix = np.eye(len(left))
        for column, (p, q) in enumerate(zip(left, right)):
            perturb = np.zeros((state.nmo, state.nmo))
            perturb[p, q] = perturb[q, p] = occ[q] - occ[p]
            image = C.T @ self.kernel(C @ perturb @ C.T) @ C
            self.response_matrix[:, column] -= image[left, right] / gaps

    def direction(self, direction_bohr):
        state = self.state
        calc = self.calculator
        C = state.coefficients
        eps = state.orbital_energies_hartree
        r0 = state.coordinates_bohr
        D = r0.size
        v = np.asarray(direction_bohr, dtype=float)
        if v.shape != r0.shape or not np.all(np.isfinite(v)):
            raise ValueError("invalid molecular response direction")
        # The calculator may have evaluated another geometry since this
        # operator was constructed. Restore geometry-only intermediates;
        # this does not run SCF or replace the frozen electronic reference.
        calc._model(r0, relocate_basis=True)
        S, dipole, second = _integrals(calc, state, self.context, v)
        r = Jet(r0, np.eye(D).reshape(D, *r0.shape), v)
        P0 = (C * self.occ) @ C.T
        Pa = self.first.density_derivative.reshape(D, state.nao, state.nao)
        P = Jet(P0, Pa, np.einsum("d,dij->ij", v.ravel(), Pa))
        F = _fock(calc, state, self.context, r, S, dipole, second, P)
        if not all(np.all(np.isfinite(x)) for x in (F.v, F.a, F.b, F.ab)):
            raise FloatingPointError("nonfinite mixed Fock response")
        f0 = float(np.max(abs(F.v - state.fock)))
        f1 = float(np.max(abs(F.a - self.first.fock_derivative.reshape(F.a.shape))))
        if f0 > 2e-7 or f1 > 2e-7:
            raise RuntimeError(
                f"mixed Fock disagrees with the resident reference: primal={f0:g}, first={f1:g}"
            )
        Ca = self.first.coefficient_derivative.reshape(D, state.nao, state.nmo)
        ea = self.first.orbital_energy_derivative.reshape(D, state.nmo)
        Cb = np.einsum("d,dij->ij", v.ravel(), Ca)
        eb = np.einsum("d,di->i", v.ravel(), ea)
        Cab = np.empty_like(Ca)
        eab = np.empty_like(ea)
        eres = 0.0
        nres = 0.0
        for k in range(D):
            normal = -(
                Ca[k].T @ state.overlap @ Cb
                + Cb.T @ state.overlap @ Ca[k]
                + Ca[k].T @ S.b @ C
                + C.T @ S.b @ Ca[k]
                + Cb.T @ S.a[k] @ C
                + C.T @ S.a[k] @ Cb
                + C.T @ S.ab[k] @ C
            )
            sym = 0.5 * normal
            cross = (Ca[k] * self.occ) @ Cb.T + (Cb * self.occ) @ Ca[k].T
            known_density = (C @ sym * self.occ) @ C.T + (C * self.occ) @ (C @ sym).T + cross
            extra = (
                F.a[k] @ Cb
                + F.b @ Ca[k]
                - (S.a[k] @ Cb + S.b @ Ca[k] + S.ab[k] @ C) * eps
                - (S.a[k] @ C + state.overlap @ Ca[k]) * eb
                - (S.b @ C + state.overlap @ Cb) * ea[k]
            )
            B = C.T @ (F.ab[k] @ C + extra)
            effective = B + C.T @ self.kernel(known_density) @ C
            rhs = (effective[self.left, self.right] / self.gaps - sym[self.left, self.right])[
                self.active
            ]
            rotations = np.linalg.solve(self.response_matrix, rhs)
            K = np.zeros_like(sym)
            K[self.left[self.active], self.right[self.active]] = rotations
            K -= K.T
            density = known_density + (C @ K * self.occ) @ C.T + (C * self.occ) @ (C @ K).T
            Fmixed = F.ab[k] + self.kernel(density)
            B = C.T @ (Fmixed @ C + extra)
            U = sym.copy()
            upper = B[self.left, self.right] / self.gaps
            U[self.left, self.right] = upper
            U[self.right, self.left] = normal[self.right, self.left] - upper
            Cab[k] = C @ U
            eab[k] = np.diag(B)
            residual = (
                state.fock @ Cab[k]
                - (state.overlap @ Cab[k]) * eps
                - (state.overlap @ C) * eab[k]
                + Fmixed @ C
                + extra
            )
            eres = max(eres, float(np.max(abs(residual))))
            nres = max(nres, float(np.max(abs(U + U.T - normal))))
        if not np.all(np.isfinite(Cab)) or not np.all(np.isfinite(eab)):
            raise FloatingPointError("nonfinite mixed orbital response")
        if eres > 2e-7 or nres > 2e-9:
            raise RuntimeError(f"mixed CPSCF residual failed: eigen={eres:g}, metric={nres:g}")
        return MixedOrbitalResponse(
            S, Jet(C, Ca, Cb, Cab), Jet(eps, ea, eb, eab), f0, f1, eres, nres
        )
