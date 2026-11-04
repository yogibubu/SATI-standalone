"""Compact, independently checked occupation dependence of a full operator.

Diagonal interblock interactions do not require merging resonant eigenbases.
The polynomial is qualified on caller-supplied independent occupations; this
certificate does not establish off-diagonal or transition-property accuracy.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .mode_blocks import VibrationalModePartition


def _occupations(values, mode_count):
    values = np.asarray(values)
    if (values.ndim != 2 or values.shape[1] != mode_count
            or values.dtype.kind not in 'iu' or np.any(values < 0)):
        raise ValueError('nonnegative integer occupation rows required')
    return values


@dataclass(frozen=True)
class QuadraticOccupationDiagonal:
    origin_cm: float
    linear_cm: np.ndarray
    quadratic_cm: np.ndarray
    validation_max_error_cm: float
    validation_state_count: int

    def energies(self, states):
        """Excitation energies; extrapolation beyond validated occupations is unqualified."""
        states = _occupations(states, len(self.linear_cm))
        return states @ self.linear_cm + np.einsum('ni,ij,nj->n', states, self.quadratic_cm, states)

    def interblock_energies(self, states, partition):
        """Connected diagonal part only; add it once to vacuum-local diagonals.

        For mixed local eigenstates this operator must be contracted with their
        complex columns, rather than evaluated at an assigned root label.
        """
        if not isinstance(partition, VibrationalModePartition) or partition.mode_count != len(self.linear_cm):
            raise ValueError('matching VibrationalModePartition required')
        states = _occupations(states, partition.mode_count)
        labels = np.empty(partition.mode_count, int)
        for label, block in enumerate(partition.blocks):
            labels[list(block)] = label
        cross = self.quadratic_cm * (labels[:, None] != labels[None, :])
        return np.einsum('ni,ij,nj->n', states, cross, states)


    def interblock_matrix_elements(self, model, initial, final, *, maximum_terms=100000):
        """Contract selected <final|C_interblock|initial> product columns.

        The model and this diagonal must use the same full-operator modal gauge.
        This retains occupation dependence without constructing a product matrix.
        Off-diagonal entries induced by local polyad rotations are kept.
        """
        initial, final = model.configurations(initial), model.configurations(final)
        if initial.shape != final.shape:
            raise ValueError('paired configurations required')
        columns = {}
        def column(configuration):
            key = tuple(configuration)
            if key not in columns:
                columns[key] = model.bare_column(configuration, maximum_terms=maximum_terms)
            return columns[key]
        output = []
        for left, right in zip(initial, final):
            states, weights = column(left)
            targets, target_weights = column(right)
            target = {tuple(s): weight for s, weight in zip(targets, target_weights)}
            values = self.interblock_energies(states, model.partition)
            output.append(sum(weight * np.conjugate(target.get(tuple(state), 0.)) * value
                              for state, weight, value in zip(states, weights, values)))
        return np.asarray(output, dtype=complex)


def build_occupation_diagonal(operator, *, validation_states, tolerance_cm=1e-7):
    """Extract quadratic coefficients from exact full-operator diagonals.

    All virtual modes and the operator's resonance screening are retained.
    Numerical screening can make an operator nonpolynomial: independent checks
    are mandatory and a failed check rejects this representation. Users must
    qualify the intended occupation domain rather than infer cascade accuracy.
    """
    if not np.isfinite(tolerance_cm) or tolerance_cm <= 0:
        raise ValueError('positive finite validation tolerance required')
    n = len(operator.frequencies)
    checks = _occupations(validation_states, n)
    if not len(checks) or not np.any(checks.sum(axis=1) > 2):
        raise ValueError('independent validation occupations above degree two required')
    units = np.eye(n, dtype=int)
    origin = operator.diagonal((0,) * n)
    first = np.array([operator.diagonal(tuple(e)) - origin for e in units])
    quadratic = np.zeros((n, n))
    for i, e in enumerate(units):
        quadratic[i, i] = (operator.diagonal(tuple(2 * e)) - origin - 2 * first[i]) / 2
    for i, j in combinations(range(n), 2):
        quadratic[i, j] = quadratic[j, i] = (
            operator.diagonal(tuple(units[i] + units[j])) - origin - first[i] - first[j]) / 2
    linear = first - np.diag(quadratic)
    prediction = checks @ linear + np.einsum('ni,ij,nj->n', checks, quadratic, checks)
    exact = np.array([operator.diagonal(tuple(s)) - origin for s in checks])
    error = float(np.max(abs(exact - prediction)))
    if not np.isfinite(error) or error > tolerance_cm:
        raise ValueError(f'occupation diagonal fails independent validation: {error} cm^-1')
    linear.setflags(write=False)
    quadratic.setflags(write=False)
    return QuadraticOccupationDiagonal(float(origin), linear, quadratic, error, len(checks))
