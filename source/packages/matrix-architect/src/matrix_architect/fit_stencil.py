"""Order-resolved residual observations for native ZAFF fitting.

These are linear combinations of acquired observations, not new QM values.
The baseline must already have been subtracted, consistently in the design
and target. Reference constraints remain separate exact constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedResidualBlocks:
    odd_design: np.ndarray
    odd_target: np.ndarray
    even_design: np.ndarray
    even_target: np.ndarray


def paired_residual_blocks(
    design: np.ndarray,
    target: np.ndarray,
    *,
    center: int,
    pairs: np.ndarray,
    half_steps: np.ndarray,
    target_center_offsets: np.ndarray | None = None,
) -> PairedResidualBlocks:
    """Separate odd/even information in symmetric single-coordinate stencils.

    ``design`` has shape (points, ..., coefficients), ``target`` its shape
    without the last axis. ``pairs`` lists (plus, minus) point indices;
    ``half_steps`` gives positive displacements in the declared coordinate.
    Only training points may be included when these blocks guide a fit.

    Return (A+ - A-)/(2h) and (A+ + A- - 2A0)/h², and identical combinations
    of the residual targets. Applied to Hessians these resolve cubic and
    quartic information. Applied to gradients or energies they retain the
    corresponding accessible derivative information without assuming any
    unmeasured mixed derivatives. The caller assigns explicit physical weights
    to the blocks; the count of raw tensor components is not an accuracy goal.

    ``target_center_offsets`` optionally supplies, for each pair, the difference
    between an independently established limiting central observation and the
    stored centre. Its shape is (pairs, ...). Only the even target uses this
    offset; the acquired observations, the design and the odd target are
    unchanged. This is for a documented reference-consistency correction,
    never an adjustable parameter of the fit. By default the stored centre is
    used for every pair.
    """
    a = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    raw_pairs = np.asarray(pairs)
    h = np.asarray(half_steps, dtype=float)
    if a.ndim < 2 or a.shape[:-1] != y.shape or not a.shape[-1]:
        raise ValueError("paired residual design and target shapes differ")
    if (not isinstance(center, (int, np.integer)) or isinstance(center, bool)
            or not 0 <= center < len(a)):
        raise ValueError("invalid central residual observation")
    if (raw_pairs.ndim != 2 or raw_pairs.shape[1] != 2 or len(raw_pairs) == 0
            or not np.issubdtype(raw_pairs.dtype, np.integer)):
        raise ValueError("stencil pairs require integer plus/minus indices")
    if (np.any(raw_pairs < 0) or np.any(raw_pairs >= len(a))
            or np.any(raw_pairs == center)
            or np.any(raw_pairs[:, 0] == raw_pairs[:, 1])
            or len(np.unique(raw_pairs)) != raw_pairs.size):
        raise ValueError("stencil points must be distinct from each other and the center")
    if (h.shape != (len(raw_pairs),) or not np.all(np.isfinite(h))
            or np.any(h <= 0) or not np.all(np.isfinite(a))
            or not np.all(np.isfinite(y))):
        raise ValueError("paired residual data and positive steps must be finite")
    plus, minus = raw_pairs.T
    scale = h.reshape((-1,) + (1,) * (y.ndim - 1))
    design_scale = scale[..., None]
    even_target = y[plus] + y[minus] - 2 * y[center]
    if target_center_offsets is not None:
        offsets = np.asarray(target_center_offsets, dtype=float)
        if offsets.shape != even_target.shape or not np.all(np.isfinite(offsets)):
            raise ValueError("limiting central target offsets have incompatible shape or values")
        even_target = even_target - 2 * offsets
    return PairedResidualBlocks(
        (a[plus] - a[minus]) / (2 * design_scale),
        (y[plus] - y[minus]) / (2 * scale),
        (a[plus] + a[minus] - 2 * a[center]) / design_scale**2,
        even_target / scale**2,
    )
