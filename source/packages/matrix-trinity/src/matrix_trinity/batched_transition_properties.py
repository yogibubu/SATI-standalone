"""Vectorized selected-polyad PT2 coordinate properties at multiple fields.

All points retain their own tensors, frequencies and resolvents. Integer
contractions and common occupation supports are traversed once. Different
resonance masks must be evaluated in separate batches.
"""

from itertools import product
from math import sqrt
import hashlib
from pathlib import Path

import numpy as np
from .reduced_transition_polynomials import (
    _put,
    _nonzero,
    _property_coefficients,
    evaluate_reduced_transition,
)
from .reduced_resonant_transitions import (
    PairReducedField,
    PairReducedProperty,
    _groups,
    _plus,
    _gf2_basis,
    _gf2_reduce,
    _parity_mask,
)
from .gvpt2_spectroscopy import _direct_polyad_exclusions
from .spectroscopic_inference import assigned_columns


def _batch_hamiltonian(f3, f4, g1, g2):
    # Move the independent field-point axis to the last position. Scalar
    # Hamiltonian coefficients have shape (1, points), properties (3, points).
    f3, f4, g1, g2 = [np.moveaxis(a, 0, -1) for a in (f3, f4, g1, g2)]
    n = len(f3)
    cubic = {}
    quartic = {}
    rho = f3 - g1 + g1.transpose(0, 2, 1, 3) + g1.transpose(2, 0, 1, 3)
    sigma = f3 - g1 - g1.transpose(0, 2, 1, 3) - g1.transpose(2, 0, 1, 3)
    for i, j, k in np.argwhere(np.any(sigma != 0, axis=-1)):
        value = sigma[i, j, k][None, :] / (12 * sqrt(2))
        _put(cubic, n, value, (("x", i), ("x", j), ("x", k)))
        _put(cubic, n, value, (("y", i), ("y", j), ("y", k)))
    for i, j, k in np.argwhere(np.any(rho != 0, axis=-1)):
        value = rho[i, j, k][None, :] / (4 * sqrt(2))
        _put(cubic, n, value, (("x", i), ("x", j), ("y", k)))
        _put(cubic, n, value, (("x", k), ("y", j), ("y", i)))
    for i, j in product(range(n), repeat=2):
        _put(cubic, n, rho[i, j, j][None, :] / (4 * sqrt(2)), (("Q", i),))
        _put(
            quartic,
            n,
            np.sum(f4[i, j, np.arange(n), np.arange(n)], axis=0)[None, :] / 16,
            (("Q", i), ("Q", j)),
        )
    indices = np.argwhere(np.any(f4 != 0, axis=-1))
    if len(indices):
        codes = np.ravel_multi_index(np.sort(indices, axis=1).T, (n,) * 4)
        unique, inverse = np.unique(codes, return_inverse=True)
        weights = f4[tuple(indices.T)]
        coefficients = np.array(
            [np.bincount(inverse, weights=weights[:, p]) for p in range(weights.shape[1])]
        ).T
        indices = np.array(np.unravel_index(unique, (n,) * 4)).T
    else:
        coefficients = ()
    for (i, j, k, l), v in zip(indices, coefficients):
        _put(quartic, n, v[None, :] / 96, (("Q", i), ("Q", j), ("Q", k), ("Q", l)))
    for i, j, k, l in np.argwhere(np.any(g2 != 0, axis=-1)):
        coefficient = -g2[i, j, k, l][None, :] / 16
        _put(quartic, n, coefficient, (("R", i), ("Q", k), ("Q", l), ("R", j)))
        for delta, sign, factors in (
            (i == k, -1, (("Q", l), ("R", j))),
            (i == l, -1, (("Q", k), ("R", j))),
            (i == j, -1, (("Q", k), ("Q", l))),
            (k == l, 1, (("R", i), ("R", j))),
            (k == j, 1, (("R", i), ("Q", l))),
            (l == j, 1, (("R", i), ("Q", k))),
        ):
            if delta:
                _put(quartic, n, sign * coefficient, factors)
    return _nonzero(cubic), _nonzero(quartic)


class BatchedPairReducedField(PairReducedField):
    def __init__(self, frequencies, f3, f4, g1, g2, exclusions, *, tolerance=1e-6):
        self.points = len(frequencies)
        self.frequency_points = np.asarray(frequencies, float)
        n = self.frequency_points.shape[1]
        expected = [(self.points,) + (n,) * r for r in (3, 4, 3, 4)]
        if any(
            a.shape != s or not np.all(np.isfinite(a)) for a, s in zip((f3, f4, g1, g2), expected)
        ):
            raise ValueError("matched finite batched modal tensors required")
        self.frequencies = self.frequency_points[0]
        self.exclusions = exclusions
        self.tolerance = tolerance
        cubic, quartic = _batch_hamiltonian(f3, f4, g1, g2)
        self.base = {"c": _groups(cubic), "h": _groups(quartic)}
        self.support = {k: tuple(v) for k, v in self.base.items()}
        for name in ("a", "d", "ad"):
            self.support[name] = self.support["c"]
        self.cache = {}
        self.active_support_cache = {}

    def divide(self, polynomial, n, shift):
        polynomial = self.prune(polynomial, n)
        if not polynomial:
            return {}
        amplitude = evaluate_reduced_transition(polynomial, n, _plus(n, shift), (1, self.points))[0]
        active = abs(amplitude) > 1e-12
        if not np.any(active):
            return {}
        denominator = -self.frequency_points @ np.asarray(shift)
        if np.any(active & (abs(denominator) <= self.tolerance)):
            raise ValueError("unresolved batched transition denominator; refine the polyad")
        inverse = np.zeros(self.points)
        inverse[active] = 1 / denominator[active]
        return {key: value * inverse[None, :] for key, value in polynomial.items()}


class BatchedPairReducedProperty(PairReducedProperty):
    def __init__(self, field, first, second, third):
        self.field = field
        self.components = (3, field.points)
        self.base = {
            f"p{rank}": _groups(v)
            for rank, v in enumerate(_property_coefficients(first, second, third, False))
        }
        self.support = dict(field.support)
        self.support.update({name: tuple(groups) for name, groups in self.base.items()})
        self.cache = {}
        self.moment_cache = {}
        basis = _gf2_basis(_parity_mask(s) for name in ("c", "h") for s in field.support[name])
        self.allowed_property_cosets = frozenset(
            _gf2_reduce(_parity_mask(s), basis)
            for name in ("p0", "p1", "p2")
            for s in self.support[name]
        )
        self.h_symmetry_basis = basis


class BatchedSelectedTransitions:
    """Reusable bare coefficients and assigned moments for a field-point batch."""

    def __init__(self, results, modalized, properties, *, cache_path=None):
        if not results or len(results) != len(modalized) or len(results) != len(properties):
            raise ValueError("matching reference, modal field and property batches required")
        self.results = results
        self.modalized = modalized
        self.properties = properties
        self.points = len(results)
        self.states = results[0].states
        masks = [_direct_polyad_exclusions(r) for r in results]
        if any(r.states != self.states for r in results) or any(m != masks[0] for m in masks):
            raise ValueError("batched fields require the same support and direct resonance masks")
        self.exclusions = masks[0]
        self.path = None if cache_path is None else Path(cache_path)
        digest = hashlib.sha256(
            str(self.states).encode()
            + str(sorted((k, sorted(v)) for k, v in masks[0].items())).encode()
        )
        for tensors, prop in zip(modalized, properties):
            for a in [*tensors[0:1], *tensors[2:], prop.first, prop.second, prop.third]:
                digest.update(np.ascontiguousarray(a).tobytes())
        for name in [
            "batched_transition_properties.py",
            "reduced_resonant_transitions.py",
            "reduced_transition_polynomials.py",
            "gvpt2_spectroscopy.py",
        ]:
            digest.update(Path(__file__).with_name(name).read_bytes())
        self.digest = digest.hexdigest()
        self.bare = {}
        self.reduced = None
        if self.path is not None and self.path.exists():
            with np.load(self.path, allow_pickle=False) as saved:
                if str(saved["identity"]) != self.digest:
                    raise ValueError("batched coefficient cache identity differs")
                self.bare = {tuple(map(int, k)): v for k, v in zip(saved["pairs"], saved["orders"])}

    def _prepare(self):
        if self.reduced is not None:
            return
        frequency = np.stack([m[0] for m in self.modalized])
        tensors = [np.stack([m[i] for m in self.modalized]) for i in (2, 3, 4, 5)]
        self.field = BatchedPairReducedField(frequency, *tensors, self.exclusions)
        property_tensors = [
            np.stack([getattr(p, name) for p in self.properties], axis=-1)
            for name in ("first", "second", "third")
        ]
        self.reduced = BatchedPairReducedProperty(self.field, *property_tensors)
        self.rectilinear_points = [
            p for p, m in enumerate(self.modalized) if np.max(abs(m[4])) <= 1e-14
        ]
        self.rectilinear = None
        if self.rectilinear_points:
            tensors[-1] = np.zeros_like(tensors[-1])
            field = BatchedPairReducedField(frequency, *tensors, self.exclusions)
            property_tensors[1] = np.zeros_like(property_tensors[1])
            property_tensors[2] = np.zeros_like(property_tensors[2])
            self.rectilinear = BatchedPairReducedProperty(field, *property_tensors)

    def save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.bare)
        temporary = self.path.with_suffix(".pending.npz")
        np.savez_compressed(
            temporary,
            identity=self.digest,
            pairs=np.array(keys, int),
            orders=np.array([self.bare[k] for k in keys]),
        )
        temporary.replace(self.path)

    def moment(self, i, j):
        key = min(i, j), max(i, j)
        if key not in self.bare:
            self._prepare()
            source, target = sorted(key, key=lambda k: (sum(self.states[k]), k))
            a, b = self.states[source], self.states[target]
            orders = np.stack(self.reduced.moment(a, b))
            if (
                self.rectilinear is not None
                and not any(a)
                and sum(b) == 1
                and len(self.exclusions[b]) > 1
            ):
                from .gvpt2_spectroscopy import _rectilinear_dvpt2_fundamental_mechanical

                generic = self.rectilinear.moment(a, b)[2]
                mode = b.index(1)
                for p in self.rectilinear_points:
                    freq, _, f3, f4, _, _ = self.modalized[p]
                    collected = _rectilinear_dvpt2_fundamental_mechanical(
                        freq, f3, f4, self.properties[p].first, mode, self.exclusions
                    )
                    orders[2, :, p] += collected - generic[:, p]
            self.bare[key] = orders if source == key[0] else orders.conj()
        return self.bare[key] if i <= j else self.bare[key].conj()

    def moments(self, pairs, *, columns=None, progress=None):
        pairs = np.asarray(pairs, int)
        vectors = (
            np.asarray([assigned_columns(r) for r in self.results])
            if columns is None
            else np.asarray(columns)
        )
        count = len(self.states)
        if (
            vectors.shape != (self.points, count, count)
            or pairs.ndim != 2
            or pairs.shape[1] != 2
            or np.any(pairs < 0)
            or np.any(pairs >= count)
        ):
            raise ValueError("matching assigned columns and valid endpoint pairs required")
        missing = {}
        for a, b in pairs:
            ia = np.flatnonzero(np.any(vectors[:, :, a] != 0, axis=0))
            ib = np.flatnonzero(np.any(vectors[:, :, b] != 0, axis=0))
            for i in ia:
                for j in ib:
                    key = min(int(i), int(j)), max(int(i), int(j))
                    if key not in self.bare:
                        missing.setdefault(key[0], set()).add(key[1])
        for i, finals in missing.items():
            for j in sorted(finals):
                self.moment(i, j)
            self.save()
            if progress is not None:
                progress(dict(source=i, acquired=len(self.bare)))
        answer = np.zeros((self.points, len(pairs), 3, 3), complex)
        for row, (a, b) in enumerate(pairs):
            for i in np.flatnonzero(np.any(vectors[:, :, a] != 0, axis=0)):
                for j in np.flatnonzero(np.any(vectors[:, :, b] != 0, axis=0)):
                    # coefficient-point axis is last in the algebra, first in output.
                    answer[:, row] += (
                        np.moveaxis(self.moment(i, j), -1, 0)
                        * (vectors[:, j, b].conj() * vectors[:, i, a])[:, None, None]
                    )
        return answer
