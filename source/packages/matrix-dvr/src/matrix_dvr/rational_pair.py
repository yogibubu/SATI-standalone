"""Quartic-jet matched two-mode interactions from two acquired corner shells.

Parity separates the four signed corners on each shell. A [1/1] rational
amplitude for each parity retains every local mixed cubic/quartic term.
A C2 radial cap prevents poles outside the acquired disk; exterior damping
returns the interaction to zero. The caller owns the confining diagonal
potentials, native kinetic metric and independent validation points.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RationalPairInteraction:
    terms: tuple
    parameters: tuple
    outer_u: float
    cap_width: float

    def evaluate(self, coordinates):
        q = np.asarray(coordinates, float)
        if q.ndim != 2 or q.shape[1] != 2 or not np.all(np.isfinite(q)):
            raise ValueError('finite two-mode coordinate matrix required')
        a, b = q.T
        u = .5*(a*a+b*b)
        exterior = np.maximum(u-self.outer_u, 0.)
        effective = np.where(u <= self.outer_u, u,
                             self.outer_u+self.cap_width*np.tanh(exterior/self.cap_width))
        result = np.zeros(len(q))
        for parity, numerator, denominator in self.parameters:
            local = sum(c*a**i*b**j for i, j, c in self.terms
                        if (i % 2, j % 2) == tuple(parity))
            result += local*(1+numerator*effective)/(1+denominator*effective)
        return result*np.exp(-np.maximum(u/self.outer_u-1., 0.)**3)

    def to_dict(self):
        return dict(schema='matrix.dvr.rational_pair_interaction.v1', terms=self.terms,
                    parameters=self.parameters, outer_u=self.outer_u, cap_width=self.cap_width)

    @classmethod
    def from_dict(cls, data):
        if data.get('schema') != 'matrix.dvr.rational_pair_interaction.v1':
            raise ValueError('unsupported rational pair interaction schema')
        model = cls(tuple(tuple(t) for t in data['terms']),
                    tuple((tuple(p), float(a), float(b)) for p, a, b in data['parameters']),
                    float(data['outer_u']), float(data['cap_width']))
        if (not np.isfinite(model.outer_u) or model.outer_u <= 0
                or not np.isfinite(model.cap_width) or model.cap_width <= 0
                or any(not np.isfinite(a+b) or 1+b*(model.outer_u+model.cap_width) <= 0
                       for _, a, b in model.parameters)):
            raise ValueError('positive pole-free radial completion required')
        return model


def fit_rational_pair_interaction(coordinates, interaction_energies, mixed_terms):
    """Fit eight training corners; no validation data or potential callback.

    Coordinates are the four signed corners at each of two positive radii.
    Input energies have the independently fitted diagonal profiles removed.
    The five local terms are q1^2 q2, q1 q2^2, q1^3 q2, q1 q2^3, q1^2 q2^2.
    Near-zero parity anchors or poles inside the acquired disk are rejected.
    """
    q, energy = np.asarray(coordinates, float), np.asarray(interaction_energies, float)
    if (q.shape != (8, 2) or energy.shape != (8,) or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(energy)) or np.any(q == 0)
            or not np.allclose(abs(q[:, 0]), abs(q[:, 1]), rtol=0, atol=1e-12)):
        raise ValueError('eight finite signed corners on two shells required')
    terms = tuple((int(i), int(j), float(c)) for i, j, c in mixed_terms)
    expected = {(2, 1), (1, 2), (3, 1), (1, 3), (2, 2)}
    if (len(terms) != 5 or {(i,j) for i,j,_ in terms} != expected
            or not all(np.isfinite(c) for _,_,c in terms)):
        raise ValueError('five finite acquired mixed cubic/quartic terms required')
    radii = np.unique(abs(q[:, 0]))
    if len(radii) != 2:
        raise ValueError('exactly two distinct training shells required')
    for radius in radii:
        rows = q[abs(q[:, 0]) == radius]
        if len(rows) != 4 or len(np.unique(np.sign(rows), axis=0)) != 4:
            raise ValueError('four distinct signed corners on each shell required')
    parameters = []
    for parity in ((0,0), (1,0), (0,1), (1,1)):
        local = [(i,j,c) for i,j,c in terms if (i % 2,j % 2) == parity]
        anchor = sum(c for _,_,c in local)
        if abs(anchor) < 1e-12*max(1., max(abs(c) for _,_,c in terms)):
            raise ValueError('rational parity amplitude needs a resolved local anchor')
        degree = local[0][0]+local[0][1]
        amplitudes = []
        for radius in radii:
            mask = abs(q[:, 0]) == radius
            sign = np.sign(q[mask, 0])**parity[0]*np.sign(q[mask, 1])**parity[1]
            amplitudes.append(np.mean(energy[mask]*sign)/radius**degree/anchor)
        u, v = radii*radii, np.asarray(amplitudes)
        matrix = np.column_stack([u, -u*v])
        if np.linalg.cond(matrix) > 1e10:
            raise ValueError('rational amplitude is not identified by the training shells')
        a, b = np.linalg.solve(matrix, v-1.)
        parameters.append((parity, float(a), float(b)))
    outer = float(radii[-1]**2)
    first_pole = min((-1/b for _,_,b in parameters if b < 0), default=np.inf)
    if first_pole <= outer:
        raise ValueError('rational amplitude has a pole inside the acquired disk')
    width = (first_pole-outer)/2 if np.isfinite(first_pole) else outer
    model = RationalPairInteraction.from_dict(RationalPairInteraction(terms, tuple(parameters), outer, width).to_dict())
    error = model.evaluate(q)-energy
    return model, dict(training_maximum_error=float(max(abs(error))),
                       radial_cap_rule='half the distance from the outer shell to the first positive pole',
                       local_jet_preserved_through=4, axis_profiles_preserved=True)
