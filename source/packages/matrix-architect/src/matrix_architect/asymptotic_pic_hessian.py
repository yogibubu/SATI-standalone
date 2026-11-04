"""Analytic native ZAFF correction Hessians, including switches and B prime.

Cartesian lengths are in angstrom. Factor definitions and first derivatives
are checked against the production evaluator before using these columns.
"""

import numpy as np
from types import SimpleNamespace
from matrix_chem.primitive_coordinates import eval_primitives, primitive_b_matrix
from matrix_zaff.basic_primitive_derivative import analytic_basic_primitive_second_derivative
from .asymptotic_pic import periodic_outer_window


class NativeCorrectionHessianDesign:
    def __init__(self, p, f, fac):
        self.p = p
        self.f = f
        self.fac = fac
        self.q0 = np.array(f.equilibrium_values)
        bond = {frozenset(t.atoms): i for i, t in enumerate(p) if t.kind == "bond"}
        fun = {x.primitive_type_id: x for x in f.functional_parameters}
        self.radial_parameters = {i: fun[f.primitive_type_ids[i]] for i in bond.values()}
        self.alpha = {}
        for i, parameter in self.radial_parameters.items():
            values = dict(parameter.parameters)
            self.alpha[i] = (values["P"]/values["R0"] if parameter.form == "inverse_power_stretch" else values["ALPHA"])
        self.supports = []
        for t in p:
            a = t.atoms
            edges = (
                [a]
                if t.kind == "bond"
                else list(zip(a[:-1], a[1:]))
                if t.kind == "dihedral"
                else [(a[0], a[1]), (a[1], a[2])]
                if t.kind == "angle"
                else [(a[0], z) for z in a[1:]]
            )
            self.supports.append({bond[frozenset(e)] for e in edges})

    def native_radial_jet(self, index, distance):
        """Use the parent ZAFF radial function, including inverse powers."""
        parameter=self.radial_parameters[index];values=dict(parameter.parameters)
        if parameter.form == 'inverse_power_stretch':
            p,r0=values['P'],values['R0'];reference=self.q0[index]
            ratio=(r0/distance)**p;at_reference=(r0/reference)**p
            scale=p*at_reference/reference
            return ((at_reference-ratio)/scale,p*ratio/distance/scale,
                    -p*(p+1)*ratio/distance**2/scale)
        alpha=self.alpha[index];delta=distance-self.q0[index]
        first=np.exp(-alpha*delta)
        return -np.expm1(-alpha*delta)/alpha,first,-alpha*first

    def value(self, x):
        q = eval_primitives(self.p, x)
        B = primitive_b_matrix(self.p, x)
        dim = len(q)
        nf = len(self.fac)
        BP = np.zeros((dim, x.size, x.size))
        for i, t in enumerate(self.p):
            obj = SimpleNamespace(
                function={"bond": "R", "angle": "A", "dihedral": "D", "out_of_plane": "U"}[t.kind],
                atoms=tuple(k + 1 for k in t.atoms),
            )
            for a, b, z in analytic_basic_primitive_second_derivative(obj, x).entries:
                BP[i, a, b] = z
        ev = np.zeros(nf)
        eg = np.zeros((nf, dim))
        eh = np.zeros((nf, dim, dim))
        for k, fac in enumerate(self.fac):
            v = []
            d = []
            h = []
            indices = []
            for raw_factor in fac:
                if len(raw_factor) == 4:
                    i, form, power, n = raw_factor
                    window_start = window_end = 0.0
                elif len(raw_factor) == 6:
                    i, form, power, n, window_start, window_end = raw_factor
                else:
                    raise ValueError("native correction factor must have four or six fields")
                z = q[i] - self.q0[i]
                if form == "native":
                    if self.p[i].kind == "bond":
                        v0, d0, h0 = self.native_radial_jet(i,q[i])
                    elif self.p[i].kind == "angle":
                        sn = np.sin(self.q0[i])
                        v0 = (np.cos(self.q0[i]) - np.cos(q[i])) / sn
                        d0 = np.sin(q[i]) / sn
                        h0 = np.cos(q[i]) / sn
                    else:
                        v0 = np.sin(z)
                        d0 = np.cos(z)
                        h0 = -np.sin(z)
                elif form == "inverse_distance":
                    a = self.q0[i] ** 2
                    v0 = a * (1 / self.q0[i] - 1 / q[i])
                    d0 = a / q[i] ** 2
                    h0 = -2 * a / q[i] ** 3
                elif form == "sin":
                    v0 = np.sin(n * z)
                    d0 = n * np.cos(n * z)
                    h0 = -n * n * v0
                elif form == "one_minus_cos":
                    v0 = 1 - np.cos(n * z)
                    d0 = n * np.sin(n * z)
                    h0 = n * n * np.cos(n * z)
                elif form == "jet_cos":
                    v0 = np.cos(n * z) - 1 + n * n * (1 - np.cos(z))
                    d0 = -n * np.sin(n * z) + n * n * np.sin(z)
                    h0 = -n * n * np.cos(n * z) + n * n * np.cos(z)
                elif form == "jet_sin":
                    v0 = np.sin(n * z) - n * np.sin(z)
                    d0 = n * (np.cos(n * z) - np.cos(z))
                    h0 = -n * n * np.sin(n * z) + n * np.sin(z)
                elif form == "outer_window":
                    if self.p[i].kind != "dihedral" or power != 1 or n != 1:
                        raise ValueError("outer window requires a unit-power dihedral factor")
                    v0, d0, h0 = periodic_outer_window(z, window_start, window_end)
                else:
                    raise ValueError(form)
                v.append(v0**power)
                d.append(power * v0 ** (power - 1) * d0)
                h.append(
                    power * v0 ** (power - 1) * h0
                    + (power * (power - 1) * v0 ** (power - 2) * d0 * d0 if power > 1 else 0)
                )
                indices.append(i)
            val = np.prod(v)
            gq = np.zeros(dim)
            hq = np.zeros((dim, dim))
            for j, i in enumerate(indices):
                rest = np.prod([vv for l, vv in enumerate(v) if l != j])
                gq[i] += d[j] * rest
                hq[i, i] += h[j] * rest
                for l, m in enumerate(indices):
                    if j != l:
                        hq[i, m] += (
                            d[j]
                            * d[l]
                            * np.prod([vv for z, vv in enumerate(v) if z != j and z != l])
                        )
            sup = set().union(*(self.supports[i] for i in indices))
            log = 0.0
            lg = np.zeros(dim)
            lh = np.zeros(dim)
            for i in sup:
                a = self.alpha[i]
                z = q[i] - self.q0[i]
                log -= (a * z) ** 4
                lg[i] = -4 * a**4 * z**3
                lh[i] = -12 * a**4 * z * z
            sw = np.exp(log)
            ev[k] = sw * val
            eg[k] = sw * (gq + val * lg)
            eh[k] = sw * (
                hq + np.outer(lg, gq) + np.outer(gq, lg) + val * (np.outer(lg, lg) + np.diag(lh))
            )
        gc = eg @ B
        hc = np.einsum("ia,kij,jb->kab", B, eh, B, optimize=True) + np.einsum(
            "ki,iab->kab", eg, BP, optimize=True
        )
        return ev, gc.T, hc.transpose(1, 2, 0)


def asymptotic_pic_correction_hessian_design(field, primitives, coordinates_angstrom, factors):
    """Return Cartesian Hessians (3N, 3N, terms) for production native factors."""
    from .asymptotic_pic import asymptotic_pic_correction_design

    energy, gradient = asymptotic_pic_correction_design(
        field,
        primitives,
        coordinates_angstrom,
        factors,
    )
    basis = NativeCorrectionHessianDesign(
        primitives,
        field,
        [tuple((t.primitive_index, t.form, t.power, t.harmonic,
                t.window_start, t.window_end) for t in term) for term in factors],
    )
    e, g, h = basis.value(np.asarray(coordinates_angstrom, dtype=float))
    if not (
        np.allclose(e, energy, atol=1e-12, rtol=1e-10)
        and np.allclose(g, gradient, atol=1e-11, rtol=1e-9)
    ):
        raise ValueError("native Hessian factors disagree with production E/G")
    return (h + h.swapaxes(0, 1)) / 2
