"""Common-reference transition probabilities, assignments and uncertainty.

Molecular mode names and observations are input data. Bare PT2 moments are
acquired once per occupied endpoint pair and reused for every root column.
"""

from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from scipy.constants import c, epsilon_0, hbar, elementary_charge, physical_constants

from matrix_numerics.least_squares import CovarianceWhitening
from .curvilinear_vpt2 import SonicGVPT2Result
from .gvpt2_spectroscopy import SonicPropertyDerivatives, _curvilinear_basis_moments_by_order


def transform_property_derivatives(derivatives, coordinate_map):
    """Pull back arbitrary leading property axes through a linear modal map."""
    matrix = np.asarray(coordinate_map, float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("finite source-by-target coordinate map required")
    answer = []
    for rank, tensor in enumerate((derivatives.first, derivatives.second, derivatives.third), 1):
        if tensor is None:
            raise ValueError("complete property derivatives through third order required")
        value = np.asarray(tensor)
        if (
            value.ndim != rank + 1
            or value.shape[1:] != (matrix.shape[0],) * rank
            or not np.all(np.isfinite(value))
        ):
            raise ValueError("property and coordinate dimensions disagree")
        for axis in range(1, rank + 1):
            value = np.moveaxis(np.tensordot(value, matrix, axes=(axis, 0)), -1, axis)
        answer.append(value)
    return SonicPropertyDerivatives(*answer)


def project_property_representation(derivatives, coordinate_actions, component_actions):
    """Finite-group Reynolds projection, including exact forbidden entries.

    Paired component actions are physical polar/axial/tensor representations,
    supplied in the same group-operation order as the coordinate actions.
    Both diagonal and dense non-Abelian representations are supported.
    """
    from .field_symmetry import canonical_actions

    actions = canonical_actions(coordinate_actions)
    components = canonical_actions(component_actions)
    if len(actions) != len(components):
        raise ValueError("paired coordinate and property group actions required")
    answer = []
    residuals = []
    for rank, tensor in enumerate((derivatives.first, derivatives.second, derivatives.third), 1):
        tensor = np.asarray(tensor)
        if tensor.shape != (components.shape[1],) + (actions.shape[1],) * rank:
            raise ValueError("property tensor dimensions disagree with group actions")
        projected = np.zeros_like(tensor)
        diagonal = np.zeros_like(tensor)
        for a, r in zip(actions, components):
            term = np.einsum("ab,b...->a...", r.T, tensor)
            for axis in range(1, rank + 1):
                term = np.moveaxis(np.tensordot(term, a, axes=(axis, 0)), -1, axis)
            projected += term / len(actions)
            product = np.diag(r)
            for _ in range(rank):
                product = np.multiply.outer(product, np.diag(a))
            diagonal += product / len(actions)
        projected[abs(diagonal) <= 32 * np.finfo(float).eps * len(actions)] = 0
        residuals.append(
            float(
                np.linalg.norm(projected - tensor)
                / max(np.linalg.norm(tensor), np.finfo(float).tiny)
            )
        )
        answer.append(projected)
    return SonicPropertyDerivatives(*answer), residuals


def assigned_matrix_result(states, frequencies, matrix, *, target_columns=None, template=None):
    """Assign connected Hermitian blocks, carrying supplied target identities.

    No coefficient cutoff determines graph separation. The PT2 exclusion mask
    is the explicit set of nonzero off-diagonal Hamiltonian entries.
    """
    states = tuple(map(tuple, states))
    frequency = np.asarray(frequencies, float)
    h = np.asarray(matrix)
    count = len(states)
    if (
        len(set(states)) != count
        or not count
        or h.shape != (count, count)
        or not np.all(np.isfinite(h))
        or not np.allclose(h, h.conj().T, atol=1e-10, rtol=1e-12)
        or any(len(s) != len(frequency) or any(q < 0 for q in s) for s in states)
        or np.any(frequency <= 0)
    ):
        raise ValueError("distinct nonnegative states and a finite Hermitian matrix required")
    graph = h != 0
    np.fill_diagonal(graph, False)
    number, labels = connected_components(graph, directed=False)
    targets = {} if target_columns is None else target_columns
    positions = {s: i for i, s in enumerate(states)}
    if any(
        tuple(t) not in positions or np.asarray(v).shape != (count,) for t, v in targets.items()
    ):
        raise ValueError("target columns must match the declared occupation support")
    energies = np.zeros(count)
    blocks = []
    transported = {}
    overlaps = {}
    for label in range(number):
        ix = np.flatnonzero(labels == label)
        values, vectors = eigh(h[np.ix_(ix, ix)], check_finite=False)
        local_targets = [tuple(t) for t in targets if positions[tuple(t)] in ix]
        assigned = {}
        if local_targets:
            seed = np.column_stack([np.asarray(targets[t])[ix] for t in local_targets])
            weights = abs(seed.conj().T @ vectors) ** 2
            rr, cc = linear_sum_assignment(-weights)
            for row, col in zip(rr, cc):
                t = local_targets[row]
                assigned[list(ix).index(positions[t])] = int(col)
                overlaps[t] = float(weights[row, col])
        remaining = [j for j in range(len(ix)) if j not in assigned]
        available = [j for j in range(len(ix)) if j not in assigned.values()]
        if remaining:
            rr, cc = linear_sum_assignment(-(abs(vectors[np.ix_(remaining, available)]) ** 2))
            assigned.update({remaining[r]: available[q] for r, q in zip(rr, cc)})
        roots = [assigned[j] for j in range(len(ix))]
        energies[ix] = values[roots]
        for t in local_targets:
            v = np.zeros(count, complex)
            v[ix] = vectors[:, assigned[list(ix).index(positions[t])]]
            anchor = np.vdot(np.asarray(targets[t]), v)
            if abs(anchor):
                v *= np.conj(anchor) / abs(anchor)
            transported[t] = v
        if len(ix) > 1:
            blocks.append(
                dict(
                    indices=ix.tolist(),
                    hamiltonian_cm=h[np.ix_(ix, ix)].copy(),
                    eigenvalues_cm=values,
                    eigenvectors=vectors,
                    basis_to_root=roots,
                )
            )
    ground = positions[(0,) * len(frequency)]
    origin = float(energies[ground])
    relative = energies - origin
    for block in blocks:
        block["eigenvalues_cm"] = block["eigenvalues_cm"] - origin
    fundamentals = np.array(
        [
            relative[positions[s]] if s in positions else np.nan
            for s in (
                tuple(int(i == j) for i in range(len(frequency))) for j in range(len(frequency))
            )
        ]
    )
    if template is None:
        result = SonicGVPT2Result(
            frequency,
            states,
            np.diag(h).real - origin,
            relative,
            fundamentals,
            (),
            tuple(blocks),
            origin + float(np.sum(frequency) / 2),
            "assigned fixed-support Hermitian operator",
        )
    else:
        result = replace(
            template,
            frequencies_cm=frequency,
            states=states,
            deperturbed_cm=np.diag(h).real - origin,
            energies_cm=relative,
            fundamentals_cm=fundamentals,
            blocks=tuple(blocks),
            zero_point_cm=origin + float(np.sum(frequency) / 2),
        )
    return result, transported, overlaps


def assigned_columns(result):
    """Full assigned columns, including isolated states."""
    columns = np.eye(len(result.states), dtype=complex)
    for block in result.blocks:
        ix = np.asarray(block["indices"])
        roots = np.asarray(block["basis_to_root"])
        columns[np.ix_(ix, ix)] = np.asarray(block["eigenvectors"])[:, roots]
    return columns


_TRANSITION_WORKER = None


def _prepare_transition_worker(result, field, derivatives, modalized, prepared_cache=None):
    """One independent coefficient cache per process; never share cache files."""
    global _TRANSITION_WORKER
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass
    _TRANSITION_WORKER = SelectedTransitionWorkspace(
        result, field, derivatives, modalized=modalized,
    )
    if prepared_cache is not None:
        # Forked processes inherit immutable field/property coefficients once;
        # subsequent occupation caches remain private through copy-on-write.
        _TRANSITION_WORKER.cache = prepared_cache


def _transition_source_orders(request):
    source, selected = request
    workspace = _TRANSITION_WORKER
    orders = _curvilinear_basis_moments_by_order(
        workspace.result, workspace.derivatives,
        workspace.derivatives.first.shape[0], workspace.field,
        initial_state=workspace.result.states[source], final_indices=selected,
        _reduced_cache=workspace.cache,
    )
    # Return selected values only, not a state-by-component array per worker.
    return source, selected, np.stack([o[list(selected)] for o in orders], axis=1)


class SelectedTransitionWorkspace:
    """Resumable PT2 bare moments shared by assignments and arbitrary properties.

    Hermitian coordinate properties allow the less populated endpoint to be
    used as the source. Orders are retained separately before coherent mixing.
    An optional on-disk cache stores only numerical moment arrays and a digest.
    Endpoint columns use exact nonzero block supports, without a dense global
    matrix. ``workers`` enables independent source contractions; use the context
    manager or ``close`` to release the persistent process pool. Stacking property
    components shares the field contractions across mechanical/electrical models.
    """

    def __init__(self, result, field, derivatives, *, modalized=None, cache_path=None,
                 workers=1, start_method=None, source_chunk_size=64):
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("positive integer transition worker count required")
        if start_method is not None and start_method not in multiprocessing.get_all_start_methods():
            raise ValueError("unavailable multiprocessing start method")
        if (isinstance(source_chunk_size, bool) or not isinstance(source_chunk_size, int)
                or source_chunk_size < 1):
            raise ValueError("positive integer source chunk size required")
        self.source_chunk_size = source_chunk_size
        self.workers, self.start_method = workers, start_method
        self._executor = None
        self._source_workers = {}
        self._modalized = modalized
        self._column_cache = {}
        self._block_positions = {
            int(index): (block, local)
            for block in result.blocks
            for local, index in enumerate(block["indices"])
        }
        self.result, self.field, self.derivatives = result, field, derivatives
        self.cache = {} if modalized is None else {"modalized_field": (field, modalized)}
        self.path = None if cache_path is None else Path(cache_path)
        self.identity = hashlib.sha256()
        for value in (
            result.states,
            result.frequencies_cm,
            derivatives.first,
            derivatives.second,
            derivatives.third,
            field.harmonic_sonic,
            field.cubic_sonic,
            field.quartic_sonic,
            field.metric_sonic,
            field.metric_first_sonic,
            field.metric_second_sonic,
        ):
            a = np.ascontiguousarray(value)
            self.identity.update(str(a.shape).encode())
            self.identity.update(a.tobytes())
        for block in result.blocks:
            self.identity.update(np.ascontiguousarray(block["indices"]).tobytes())
            self.identity.update(np.ascontiguousarray(block["hamiltonian_cm"]).tobytes())
        # Cache validity includes the actual contraction implementation.
        for name in (
            "gvpt2_spectroscopy.py",
            "reduced_resonant_transitions.py",
            "reduced_transition_polynomials.py",
            "spectroscopic_inference.py",
        ):
            self.identity.update(Path(__file__).with_name(name).read_bytes())
        self.digest = self.identity.hexdigest()
        self.bare = {}
        if self.path is not None and self.path.exists():
            with np.load(self.path, allow_pickle=False) as saved:
                if str(saved["identity"]) != self.digest:
                    raise ValueError("cached transition coefficients differ from supplied model")
                self.bare = {
                    tuple(map(int, key)): value
                    for key, value in zip(saved["pairs"], saved["orders"])
                }
        self._journal = None if self.path is None else self.path.with_suffix(self.path.suffix + ".parts")
        if self._journal is not None and self._journal.exists():
            for part in sorted(self._journal.glob("part-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].npz")):
                with np.load(part, allow_pickle=False) as saved:
                    if str(saved["identity"]) != self.digest:
                        raise ValueError("cached transition coefficients differ from supplied model")
                    self.bare.update({tuple(map(int, key)): value
                                      for key, value in zip(saved["pairs"], saved["orders"])})
        self._pending_keys = set()
        self.seconds = 0.0
        self._dirty = False
        self._last_save = perf_counter()

    def save(self):
        """Atomically append completed moments without recompressing old data."""
        if self.path is None or not self._pending_keys:
            return
        self._journal.mkdir(parents=True, exist_ok=True)
        keys = sorted(self._pending_keys)
        existing = sorted(self._journal.glob("part-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].npz"))
        number = int(existing[-1].stem.split("-")[1]) + 1 if existing else 0
        destination = self._journal / f"part-{number:012d}.npz"
        temporary = destination.with_suffix(".pending.npz")
        np.savez(temporary, identity=self.digest, pairs=np.asarray(keys, int),
                 orders=np.asarray([self.bare[key] for key in keys]))
        temporary.replace(destination)
        self._pending_keys.clear()
        self._dirty = False
        self._last_save = perf_counter()

    def compact(self):
        """Consolidate the journal explicitly, outside acquisition hot loops."""
        if self.path is None:
            return
        self.save()
        parts = list(self._journal.glob("part-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].npz")) if self._journal.exists() else []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.bare)
        temporary = self.path.with_suffix(".pending.npz")
        np.savez_compressed(temporary, identity=self.digest, pairs=np.asarray(keys, int).reshape(-1, 2),
                            orders=np.asarray([self.bare[key] for key in keys]))
        temporary.replace(self.path)
        # Replace the consolidated archive before removing journal entries.
        for part in parts:
            part.unlink()

    def close(self):
        """Release workers after the last acquisition; completed data stay saved."""
        if self._dirty:
            self.save()
        if self._journal is not None and self._journal.exists():
            self.compact()
        if self._executor is not None:
            for executor in self._executor:
                executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def _column(self, index):
        """Exact block support without constructing the full square matrix."""
        index = int(index)
        if index not in self._column_cache:
            entry = self._block_positions.get(index)
            if entry is None:
                value = np.asarray([index]), np.asarray([1. + 0j])
            else:
                block, local = entry
                indices = np.asarray(block["indices"], int)
                vector = np.asarray(block["eigenvectors"])[:, block["basis_to_root"][local]]
                selected = vector != 0
                value = indices[selected], vector[selected]
            self._column_cache[index] = value
        return self._column_cache[index]

    def moments(self, pairs, *, columns=None, progress=None):
        pairs = np.asarray(pairs)
        count = len(self.result.states)
        if (
            pairs.ndim != 2 or pairs.shape[1] != 2
            or pairs.dtype.kind not in "iu"
            or np.any(pairs < 0) or np.any(pairs >= count)
        ):
            raise ValueError("valid integer endpoint pairs required")
        if columns is None:
            column = self._column
        else:
            vectors = np.asarray(columns)
            if vectors.shape != (count, count) or not np.all(np.isfinite(vectors)):
                raise ValueError("finite assigned columns on the same support required")
            selected_columns = {}
            for index in np.unique(pairs):
                support = np.flatnonzero(vectors[:, index] != 0)
                selected_columns[int(index)] = support, vectors[support, index]
            column = selected_columns.__getitem__
        requested = set()
        for a, b in pairs:
            for i in column(a)[0]:
                for j in column(b)[0]:
                    key = min(int(i), int(j)), max(int(i), int(j))
                    if key not in self.bare:
                        requested.add(key)
        requests = {}
        for i, j in sorted(requested):
            # The canonical pair is acquired once, including reversed requests.
            # A less excited source reduces active polynomial support.
            source, target = sorted((i, j), key=lambda k: (sum(self.result.states[k]), k))
            requests.setdefault(source, []).append(target)
        tasks = [(i, tuple(finals[start:start + self.source_chunk_size]))
                 for i, finals in requests.items()
                 for start in range(0, len(finals), self.source_chunk_size)]
        started = perf_counter()

        def serial():
            for i, selected in tasks:
                orders = _curvilinear_basis_moments_by_order(
                    self.result, self.derivatives, self.derivatives.first.shape[0],
                    self.field, initial_state=self.result.states[i],
                    final_indices=selected, _reduced_cache=self.cache,
                )
                yield i, selected, np.stack([o[list(selected)] for o in orders], axis=1)

        if tasks and self.workers > 1:
            if self._executor is None:
                context = multiprocessing.get_context(self.start_method)
                prepared_cache = None
                if context.get_start_method() == "fork":
                    source, targets = tasks[0]
                    _curvilinear_basis_moments_by_order(
                        self.result, self.derivatives, self.derivatives.first.shape[0],
                        self.field, initial_state=self.result.states[source],
                        final_indices=targets[:1], _reduced_cache=self.cache,
                    )
                    prepared_cache = self.cache
                # A source stays on one process across calls, preserving its
                # occupation-dependent intermediates. Each queue stays bounded.
                self._executor = [ProcessPoolExecutor(
                    max_workers=1, mp_context=context,
                    initializer=_prepare_transition_worker,
                    initargs=(self.result, self.field, self.derivatives, self._modalized, prepared_cache),
                ) for _ in range(self.workers)]
            def completed_tasks():
                loads = [0] * self.workers
                queues = [[] for _ in range(self.workers)]
                # Largest new sources first; existing assignments remain stable.
                for source in sorted(requests, key=lambda k: -len(requests[k])):
                    worker = self._source_workers.get(source)
                    if worker is None:
                        worker = min(range(self.workers), key=loads.__getitem__)
                        self._source_workers[source] = worker
                    loads[worker] += len(requests[source])
                for task in tasks:
                    queues[self._source_workers[task[0]]].append(task)
                iterators = [iter(queue) for queue in queues]
                pending = {}
                def submit(worker):
                    task = next(iterators[worker], None)
                    if task is not None:
                        future = self._executor[worker].submit(_transition_source_orders, task)
                        pending[future] = worker
                for worker in range(self.workers):
                    submit(worker)
                    submit(worker)
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        worker = pending.pop(future)
                        yield future.result()
                        submit(worker)
            acquired = completed_tasks()
        else:
            acquired = serial()
        try:
            for i, selected, values in acquired:
                for j, value in zip(selected, values):
                    key = min(i, j), max(i, j)
                    self.bare[key] = value if i <= j else value.conj()
                    self._pending_keys.add(key)
                self._dirty = True
                # Flush only new entries; complete archives are consolidated
                # explicitly outside acquisition loops.
                if perf_counter() - self._last_save >= 2.:
                    self.save()
                if progress is not None:
                    progress(dict(source=i, acquired=len(self.bare), seconds=perf_counter() - started))
        finally:
            if self._dirty:
                self.save()
            self.seconds += perf_counter() - started
        return self._contract_blocks(pairs, column)

    def _contract_blocks(self, pairs, column):
        """Coherent block products using compiled matrix multiplication.

        Components are handled separately to bound temporary storage. Exact
        nonzero supports are retained; no coefficient threshold is introduced.
        """
        answer = np.zeros((len(pairs), 3, self.derivatives.first.shape[0]), complex)
        groups = {}
        supports = {int(index): tuple(map(int, column(index)[0])) for index in np.unique(pairs)}
        for row, (a, b) in enumerate(pairs):
            key = supports[int(a)], supports[int(b)]
            groups.setdefault(key, []).append((row, int(a), int(b)))
        for (ia, ib), entries in groups.items():
            if len(ia) == len(ib) == 1:
                i, j = ia[0], ib[0]
                value = self.bare[min(i, j), max(i, j)]
                if i > j:
                    value = value.conj()
                rows = [row for row, _, _ in entries]
                factors = np.asarray([column(a)[1][0] * column(b)[1][0].conjugate()
                                      for _, a, b in entries])
                answer[rows] = factors[:, None, None] * value
                continue
            left = sorted({a for _, a, _ in entries})
            right = sorted({b for _, _, b in entries})
            ca = np.column_stack([column(a)[1] for a in left])
            cb = np.column_stack([column(b)[1] for b in right]).conj()
            left_pos = {a: k for k, a in enumerate(left)}
            right_pos = {b: k for k, b in enumerate(right)}
            rows = [row for row, _, _ in entries]
            aa = [left_pos[a] for _, a, _ in entries]
            bb = [right_pos[b] for _, _, b in entries]
            # Tile bare supports so a large connected polyad never allocates
            # a dense global moment tensor. Bound output products separately.
            for i0 in range(0, len(ia), 256):
                ii = ia[i0:i0 + 256]
                for j0 in range(0, len(ib), 256):
                    jj = ib[j0:j0 + 256]
                    values = [(self.bare[min(i, j), max(i, j)], i > j) for i in ii for j in jj]
                    for order in range(3):
                        for component in range(answer.shape[2]):
                            matrix = np.asarray([v[order, component].conjugate() if reverse
                                                 else v[order, component] for v, reverse in values])
                            matrix = matrix.reshape(len(ii), len(jj))
                            for a0 in range(0, len(left), 128):
                                active = [k for k, value in enumerate(aa) if a0 <= value < a0 + 128]
                                if not active:
                                    continue
                                partial = ca[i0:i0 + len(ii), a0:a0 + 128].T @ matrix
                                for b0 in range(0, len(right), 128):
                                    selected = [k for k in active if b0 <= bb[k] < b0 + 128]
                                    if not selected:
                                        continue
                                    mixed = partial @ cb[j0:j0 + len(jj), b0:b0 + 128]
                                    answer[np.asarray(rows)[selected], order, component] += mixed[
                                        np.asarray(aa)[selected] - a0, np.asarray(bb)[selected] - b0]
        return answer


def electric_dipole_rates(energies_cm, pairs, moment_orders_au):
    """Resolved vacuum Einstein coefficients and coherent order diagnostics."""
    e = np.asarray(energies_cm, float)
    pairs = np.asarray(pairs, int)
    orders = np.asarray(moment_orders_au)
    if (
        pairs.ndim != 2
        or pairs.shape[1] != 2
        or orders.ndim != 3
        or orders.shape[:2] != (len(pairs), 3)
        or orders.shape[2] != 3
        or not np.all(np.isfinite(orders))
    ):
        raise ValueError("three-component electric-dipole moments by PT2 order required")
    gap = abs(e[pairs[:, 0]] - e[pairs[:, 1]])
    strength = np.sum(abs(np.sum(orders, axis=1)) ** 2, axis=1)
    omega = 2 * np.pi * c * 100 * gap
    dipole_unit = elementary_charge * physical_constants["Bohr radius"][0]
    rate = omega**3 * strength * dipole_unit**2 / (3 * np.pi * epsilon_0 * hbar * c**3)
    return dict(
        gaps_cm=gap,
        strengths_au=strength,
        rates_s_inverse=rate,
        harmonic_strengths_au=np.sum(abs(orders[:, 0]) ** 2, axis=1),
        moment_orders_au=orders,
    )


def score_joint_assignments(predictions, observed, *, covariance, kinds):
    """Rank discrete candidates against measured frequencies and probabilities.

    Data must already be expressed in their declared coordinates (e.g. log A).
    Absent measurements are omitted, never replaced by a theoretical value.
    No candidate is selected by its own predicted brightness.
    """
    p, y = np.asarray(predictions, float), np.asarray(observed, float)
    if (
        p.ndim != 2
        or p.shape[1:] != y.shape
        or len(kinds) != len(y)
        or not np.all(np.isfinite(p))
        or not np.all(np.isfinite(y))
    ):
        raise ValueError("finite candidate-by-observation predictions and matched labels required")
    if any(k not in {"frequency", "probability"} for k in kinds):
        raise ValueError("observation kinds must be frequency or probability")
    covariance = np.asarray(covariance, float)
    if covariance.shape != (len(y), len(y)):
        raise ValueError("a full observation covariance matrix is required")
    whitening = CovarianceWhitening(np.sqrt(np.diag(covariance)), covariance)
    residuals = whitening.apply((p - y).T).T
    scores = np.sum(residuals**2, axis=1)
    return dict(
        scores=scores,
        ranking=np.argsort(scores),
        observed_probabilities=list(kinds).count("probability"),
        inference="joint" if "probability" in kinds else "frequency_only",
        whitened_residuals=residuals,
    )


def propagate_covariance(jacobian, parameter_covariance):
    """Full correlated delta-method covariance for frequencies, rates or branches."""
    j, cov = np.asarray(jacobian, float), np.asarray(parameter_covariance, float)
    if (
        j.ndim != 2
        or cov.shape != (j.shape[1],) * 2
        or not np.all(np.isfinite(j))
        or not np.all(np.isfinite(cov))
        or not np.allclose(cov, cov.T)
    ):
        raise ValueError("finite Jacobian and symmetric parameter covariance required")
    if np.min(np.linalg.eigvalsh(cov)) < -1e-12 * max(1.0, np.linalg.norm(cov)):
        raise ValueError("parameter covariance must be positive semidefinite")
    result = j @ cov @ j.T
    return dict(covariance=result, standard_deviations=np.sqrt(np.maximum(0, np.diag(result))))
