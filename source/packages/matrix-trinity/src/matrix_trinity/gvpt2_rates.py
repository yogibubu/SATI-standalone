"""Resolved GVPT2 energies and spontaneous-emission rate reference tables.

No HDC energies, populations, integrated intensities or degeneracy averages
enter this interface. All endpoints refer to the same assigned GVPT2 states.
"""
from dataclasses import dataclass, field as dataclass_field
import json
from pathlib import Path

import numpy as np


def _readonly(value, dtype=None):
    value = np.array(value, dtype=dtype, copy=True)
    value.setflags(write=False)
    return value


def _occupations(value):
    value = np.asarray(value)
    if value.ndim != 2 or value.shape[1] < 1 or value.dtype.kind not in "iu" or np.any(value < 0):
        raise ValueError("nonnegative integer occupation rows required")
    if np.any(value > np.iinfo(np.int64).max):
        raise ValueError("occupations exceed int64 range")
    return value.astype(np.int64)


def _labels(value, count, default):
    labels = (default,)*count if value is None else tuple(value)
    if len(labels) != count or any(not isinstance(s, str) or not s for s in labels):
        raise ValueError("one nonempty string label per state required")
    return labels


@dataclass(frozen=True)
class GVPT2RateReference:
    """Each pair stores its downward A, independent of pair index ordering.

    Symmetry masks concern individually resolved states. Reduced multiplet
    averages must be resolved before using this table. allowed=False requires
    exactly zero reference rate. Missing pairs are unmeasured, not forbidden.
    """
    occupations: np.ndarray
    energies_cm: np.ndarray
    pairs: np.ndarray
    rates_s_inverse: np.ndarray
    allowed: np.ndarray
    coordinate_id: str
    sectors: tuple[str, ...] | None = None
    branches: tuple[str, ...] | None = None
    provenance: dict = dataclass_field(default_factory=dict)

    def __post_init__(self):
        n = _occupations(self.occupations)
        e = np.asarray(self.energies_cm, float)
        pairs = np.asarray(self.pairs)
        rates = np.asarray(self.rates_s_inverse, float)
        allowed = np.asarray(self.allowed)
        if len(n) < 2 or e.shape != (len(n),) or not np.all(np.isfinite(e)):
            raise ValueError("at least two states with finite energies required")
        if (pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0
                or pairs.dtype.kind not in "iu" or np.any(pairs < 0) or np.any(pairs >= len(n))
                or np.any(pairs[:, 0] == pairs[:, 1])):
            raise ValueError("distinct valid state endpoint pairs required")
        if len({tuple(sorted(p)) for p in pairs}) != len(pairs):
            raise ValueError("duplicate forward/reverse pair; supply one downward rate per pair")
        if (rates.shape != (len(pairs),) or not np.all(np.isfinite(rates)) or np.any(rates < 0)
                or allowed.shape != rates.shape or allowed.dtype.kind != "b"):
            raise ValueError("finite nonnegative rates and an explicit boolean symmetry mask required")
        if np.any(rates[~allowed] != 0):
            raise ValueError("symmetry-forbidden reference pairs must have zero rate")
        if np.any((e[pairs[:, 0]] == e[pairs[:, 1]]) & (rates > 0)):
            raise ValueError("zero-frequency spontaneous-emission rates must vanish")
        if not isinstance(self.coordinate_id, str) or not self.coordinate_id.strip():
            raise ValueError("nonempty coordinate_id required")
        sectors = _labels(self.sectors, len(n), "scalar")
        branches = _labels(self.branches, len(n), "reference")
        if len({(tuple(row), s, b) for row, s, b in zip(n, sectors, branches)}) != len(n):
            raise ValueError("occupation, symmetry and branch labels must identify each state uniquely")
        for name, value in (("occupations", n), ("energies_cm", e), ("pairs", pairs.astype(np.int64)),
                            ("rates_s_inverse", rates), ("allowed", allowed)):
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "sectors", sectors)
        object.__setattr__(self, "branches", branches)
        object.__setattr__(self, "provenance", dict(self.provenance))

    def to_dict(self):
        return dict(schema="matrix.trinity.gvpt2-rate-reference.v1",
            **{name: (getattr(self, name).tolist() if isinstance(getattr(self, name), np.ndarray)
                       else getattr(self, name)) for name in self.__dataclass_fields__})

    def save(self, path):
        def encode(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(f"nonserializable provenance value: {type(value).__name__}")
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False,
                                         default=encode)+"\n")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        if payload.pop("schema", None) != "matrix.trinity.gvpt2-rate-reference.v1" or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("unsupported or incomplete GVPT2 rate-reference artifact")
        return cls(**payload)


def gvpt2_rate_reference(result, field, dipole, pairs, *, coordinate_id,
                         allowed, sectors=None, branches=None, provenance=None,
                         return_moments=False, canonical_phases=False):
    """Evaluate arbitrary included hot-band endpoints with existing PT2 algebra.

    pairs indexes result.states; no new resonance screening or diagonalization
    occurs. Assigned polyad columns are shared by energies and both endpoints
    of each property matrix element. The existing spectrum convenience API is
    restricted to ground/fundamental initials; this adapter reuses its general
    underlying contraction for any included initial occupation.
    """
    from scipy.constants import e, physical_constants
    from .gvpt2_spectroscopy import _curvilinear_basis_moments_by_order
    from .occupation_operators import occupation_dipole_emission
    from .local_quadratic import _maximum_overlap_assignment

    states = _occupations(np.asarray(result.states))
    pairs = np.asarray(pairs)
    # Validate indices/masks/identities before any property acquisition.
    empty = GVPT2RateReference(states, result.energies_cm, pairs,
        np.zeros(len(pairs)), allowed, coordinate_id, sectors, branches)
    count = len(states)
    # Sparse assigned columns avoid an N-by-N global mixing allocation.
    columns = [(np.array([i]), np.ones(1, complex)) for i in range(count)]
    occupied = set()
    for block in result.blocks:
        members = np.asarray(block["indices"], int)
        vectors = np.asarray(block["eigenvectors"], complex)
        if (vectors.shape != (len(members), len(members)) or not np.all(np.isfinite(vectors))
                or np.any(members < 0) or np.any(members >= count)
                or len(set(members)) != len(members) or occupied.intersection(members)):
            raise ValueError("invalid or overlapping GVPT2 polyad block")
        occupied.update(members)
        assignment = block.get("basis_to_root")
        if assignment is None:
            assignment = _maximum_overlap_assignment(abs(vectors)**2)
        assignment = np.asarray(assignment)
        if (assignment.dtype.kind not in "iu" or assignment.shape != (len(members),)
                or sorted(assignment.tolist()) != list(range(len(members)))):
            raise ValueError("invalid GVPT2 assigned-root permutation")
        if np.linalg.norm(vectors.conj().T @ vectors-np.eye(len(members))) > 512*np.finfo(float).eps*len(members):
            raise ValueError("GVPT2 polyad columns must be orthonormal")
        roots = np.asarray(block["eigenvalues_cm"], float)
        tolerance = 512*np.finfo(float).eps*max(1., float(np.max(abs(empty.energies_cm))))*count
        if roots.shape != (len(members),) or not np.allclose(roots[assignment], empty.energies_cm[members], rtol=0, atol=tolerance):
            raise ValueError("GVPT2 assigned energies and polyad columns disagree")
        for position, index in enumerate(members):
            column = vectors[:, assignment[position]].copy()
            if canonical_phases:
                phase_anchor = column[position]
                if abs(phase_anchor) == 0:
                    phase_anchor = column[int(np.argmax(abs(column)))]
                column *= np.conj(phase_anchor)/abs(phase_anchor)
            populated = column != 0
            columns[index] = (members[populated], column[populated])
    moments = np.zeros((len(pairs), 3), complex)
    cache = {}
    for initial in np.unique(pairs[empty.allowed, 0]):
        rows = np.flatnonzero(empty.allowed & (pairs[:, 0] == initial))
        finals = pairs[rows, 1]
        selected = tuple(sorted({int(index) for final in finals for index in columns[final][0]}))
        values = np.zeros((count, 3), complex)
        for basis_initial, coefficient in zip(*columns[initial]):
            orders = _curvilinear_basis_moments_by_order(result, dipole, 3, field,
                initial_state=tuple(states[basis_initial]), final_indices=selected,
                _reduced_cache=cache)
            values += coefficient*np.sum(orders, axis=0)
        for row, final in zip(rows, finals):
            members, coefficients = columns[final]
            moments[row] = coefficients.conj() @ values[members]
    gaps = abs(empty.energies_cm[pairs[:, 0]]-empty.energies_cm[pairs[:, 1]])
    lines = dict(amplitudes=moments, strengths=np.sum(abs(moments)**2, axis=1),
                 signed_gaps_cm=gaps, initial_rows=np.arange(len(pairs)),
                 final_occupations=states[pairs[:, 1]],
                 excluded_endpoint_strengths=np.zeros(len(pairs)))
    emission = occupation_dipole_emission(lines,
        dipole_to_coulomb_metre=e*physical_constants["Bohr radius"][0])
    rates = np.zeros(len(pairs))
    rates[emission["initial_rows"]] = emission["rates_s_inverse"]
    source = dict(provenance or {})
    source.update(reference_method="GVPT2", transition_order="second",
                  observable="resolved_vacuum_electric_dipole_A_s_inverse",
                  hdc_energies_used=False, resonance_diagnostics=result.diagnostics)
    if canonical_phases:
        source['transition_moment_phase_convention'] = 'positive_assigned_bare_state_overlap'
    reference = GVPT2RateReference(states, empty.energies_cm, pairs, rates,
        empty.allowed, coordinate_id, empty.sectors, empty.branches, source)
    return (reference, _readonly(moments)) if return_moments else reference
