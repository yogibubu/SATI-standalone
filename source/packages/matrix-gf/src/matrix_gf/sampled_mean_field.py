"""Reusable conditional contractions of a full DVR potential.

Fixed product-proposal samples make the SCF map deterministic. Self-normalized
importance contraction is approximate: ESS and independent sample-size/replica
convergence must be checked separately from SCF convergence. No n-mode
truncation of the supplied residual function is made.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


class SampledResidualMeanField:
    """Full residual operator sampled on shared finite DVR grids.

    ``conditional_values[i][a, s]`` is R(a, X_{-i,s}); ``sample_values[s]``
    is R(X_s). The caller supplies independent product-proposal draws X_s,
    represented by ``sample_indices``. R excludes explicit one/pair terms.
    Samples/tables must be reused without changing the proposal during SCF.
    """

    def __init__(self, proposal, sample_indices, sample_values, conditional_values,
                 *, minimum_effective_fraction=0.01):
        raw_proposal = tuple(np.asarray(p) for p in proposal)
        raw_tables = tuple(np.asarray(v) for v in conditional_values)
        if (any(np.iscomplexobj(p) for p in raw_proposal)
                or np.iscomplexobj(sample_values)
                or any(np.iscomplexobj(v) for v in raw_tables)):
            raise ValueError('real proposal and potential tables required')
        self.proposal = tuple(np.asarray(p, float).copy() for p in raw_proposal)
        ids = np.asarray(sample_indices)
        self.sample_values = np.asarray(sample_values, float).copy()
        self.conditional_values = tuple(np.asarray(v, float).copy() for v in raw_tables)
        n = len(self.proposal)
        if (not n or ids.ndim != 2 or ids.shape[1] != n or ids.shape[0] < 2
                or not np.issubdtype(ids.dtype, np.integer)
                or not 0 < minimum_effective_fraction <= 1
                or self.sample_values.shape != (len(ids),)
                or not np.all(np.isfinite(self.sample_values))
                or len(self.conditional_values) != n):
            raise ValueError('invalid product proposal/sample tables')
        self.sample_indices = ids.copy()
        for i, (p, table) in enumerate(zip(self.proposal, self.conditional_values)):
            if (p.ndim != 1 or not len(p) or not np.all(np.isfinite(p))
                    or np.any(p <= 0) or not np.isclose(p.sum(), 1, atol=1e-12, rtol=0)
                    or np.any(ids[:, i] < 0) or np.any(ids[:, i] >= len(p))
                    or table.shape != (len(p), len(ids)) or not np.all(np.isfinite(table))):
                raise ValueError('invalid proposal or conditional table')
        self.minimum_effective_fraction = float(minimum_effective_fraction)
        self.last_effective_sample_sizes = None
        self.last_conditional_energy_offsets = None

    def __call__(self, densities):
        if len(densities) != len(self.proposal):
            raise ValueError('one normalized density per mode required')
        logs = []
        for i, (raw, p) in enumerate(zip(densities, self.proposal)):
            rho = np.asarray(raw)
            if (np.iscomplexobj(rho) or rho.shape != p.shape
                    or not np.all(np.isfinite(rho)) or np.any(rho < 0)
                    or not np.isclose(rho.sum(), 1, atol=1e-10, rtol=0)):
                raise ValueError('finite normalized real DVR densities required')
            with np.errstate(divide='ignore'):
                logs.append(np.log(rho[self.sample_indices[:, i]]) - np.log(p[self.sample_indices[:, i]]))
        logs = np.stack(logs)
        effective = []

        def weights(logw):
            norm = logsumexp(logw)
            if not np.isfinite(norm):
                raise FloatingPointError('proposal samples do not cover the current density')
            w = np.exp(logw - norm)
            ess = float(1 / (w @ w))
            effective.append(ess)
            if ess < self.minimum_effective_fraction * len(w):
                raise FloatingPointError('insufficient effective sample size for full-potential contraction')
            return w

        # Explicit exclusion avoids inf-inf for exact nodes in modal densities.
        fields = []
        for i, table in enumerate(self.conditional_values):
            others = np.delete(logs, i, axis=0).sum(axis=0)
            fields.append(table @ weights(others))
        expectation = float(self.sample_values @ weights(logs.sum(axis=0)))
        self.last_effective_sample_sizes = tuple(effective)
        # Independent conditional ratio estimators have different random scalar
        # offsets. Align their density expectations to the one total estimator.
        # Constants do not change modal functions; this enforces the VSCF energy
        # identity without multiplying conditional sampling noise by mode count.
        offsets = tuple(expectation - np.asarray(rho) @ field
                        for rho, field in zip(densities, fields))
        self.last_conditional_energy_offsets = offsets
        return tuple(field + offset for field, offset in zip(fields, offsets)), expectation


def acquire_residual_tables(axes, proposal, sample_indices, residual_evaluator, *,
                            maximum_table_bytes=256 * 1024**2, batch_size=8192):
    """Evaluate an analytic residual in bounded batches; no electronic jobs.

    residual_evaluator accepts rows of full mode coordinates and returns R in
    the same energy units as the one-mode Hamiltonians. This routine never
    constructs a tensor product of all grid axes.
    """
    axes = tuple(np.asarray(a, float) for a in axes)
    ids = np.asarray(sample_indices)
    if (not axes or ids.ndim != 2 or ids.shape[1] != len(axes)
            or not np.issubdtype(ids.dtype, np.integer) or len(ids) < 2
            or not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1
            or not isinstance(maximum_table_bytes, int) or maximum_table_bytes < 1):
        raise ValueError('finite axes, sample indices and positive budgets required')
    for i, a in enumerate(axes):
        if (a.ndim != 1 or not len(a) or not np.all(np.isfinite(a))
                or np.any(np.diff(a) <= 0) or np.any(ids[:, i] < 0) or np.any(ids[:, i] >= len(a))):
            raise ValueError('ordered finite DVR axes and valid indices required')
    required = 8 * len(ids) * (1 + sum(map(len, axes)))
    if required > maximum_table_bytes:
        raise MemoryError('full-potential conditional tables exceed declared budget')
    # Validate proposal before expensive acquisition.
    SampledResidualMeanField(proposal, ids, np.zeros(len(ids)),
                             [np.zeros((len(a), len(ids))) for a in axes])
    points = np.column_stack([a[ids[:, i]] for i, a in enumerate(axes)])

    def evaluate(q):
        result = np.empty(len(q))
        for start in range(0, len(q), batch_size):
            values = np.asarray(residual_evaluator(q[start:start + batch_size]))
            if (values.shape != (len(q[start:start + batch_size]),)
                    or np.iscomplexobj(values) or not np.all(np.isfinite(values))):
                raise ValueError('residual evaluator must return finite real scalar energies')
            result[start:start + len(values)] = values
        return result

    sample = evaluate(points)
    tables = []
    for i, a in enumerate(axes):
        table = np.empty((len(a), len(ids)))
        for j, value in enumerate(a):
            q = points.copy(); q[:, i] = value
            table[j] = evaluate(q)
        tables.append(table)
    return sample, tuple(tables)


def subtract_pair_controls(sample_indices, sample_values, conditional_values, pair_potentials):
    """Remove known pair terms from cached full residual tables, without PES calls.

    Integrate the removed pair operators exactly through the existing VSCF
    ``pair_potentials`` argument. The sampled remainder must still be retained;
    this is a control-variate decomposition, not a two-mode approximation.
    """
    ids = np.asarray(sample_indices)
    sample = np.asarray(sample_values)
    tables = tuple(np.asarray(t) for t in conditional_values)
    if (ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer)
            or len(tables) != ids.shape[1] or sample.shape != (len(ids),)
            or np.iscomplexobj(sample) or not np.all(np.isfinite(sample))
            or any(t.ndim != 2 or t.shape[1] != len(ids) or np.iscomplexobj(t)
                   or not np.all(np.isfinite(t)) for t in tables)):
        raise ValueError('valid real sample and conditional tables required')
    for i, table in enumerate(tables):
        if np.any(ids[:, i] < 0) or np.any(ids[:, i] >= len(table)):
            raise ValueError('sample indices outside conditional support')
    total = np.zeros(len(ids))
    changes = [np.zeros_like(t, dtype=float) for t in tables]
    for pair, raw in pair_potentials.items():
        if (len(pair) != 2 or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in pair)
                or not 0 <= pair[0] < pair[1] < len(tables)):
            raise ValueError('ordered bounded mode pairs required')
        i, j = pair; value = np.asarray(raw)
        if (value.shape != (len(tables[i]), len(tables[j])) or np.iscomplexobj(value)
                or not np.all(np.isfinite(value))):
            raise ValueError('finite real pair controls on the same DVR required')
        at_sample = value[ids[:, i], ids[:, j]]
        total += at_sample
        changes[i] += value[:, ids[:, j]] - at_sample[None, :]
        changes[j] += value[ids[:, i], :].T - at_sample[None, :]
    return sample - total, tuple(t - total[None, :] - c for t, c in zip(tables, changes))
