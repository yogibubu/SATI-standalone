"""Small dependency-free periodic interpolators shared by surface tools."""

from __future__ import annotations

import numpy as np


def periodic_linear_interpolate(x, y, query):
    """Piecewise-linear interpolation on a closed periodic grid."""

    grid = np.asarray(x, dtype=float).reshape(-1)
    values = np.asarray(y, dtype=float)
    if grid.size < 2 or values.shape[0] != grid.size:
        raise ValueError("periodic grid and values must have the same length >= 2")
    if np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("periodic grid must be finite and strictly increasing")
    period = float(grid[-1] - grid[0] + (grid[1] - grid[0]))
    q = (np.asarray(query, dtype=float) - grid[0]) % period + grid[0]
    right = np.searchsorted(grid, q, side="right")
    left = (right - 1) % grid.size
    right %= grid.size
    x_left = grid[left]
    x_right = np.where(right == 0, grid[0] + period, grid[right])
    weight = (q - x_left) / (x_right - x_left)
    return (1.0 - weight)[..., None] * values[left] + weight[..., None] * values[right]


def periodic_cubic_hermite(x, y, query):
    """C1 periodic cubic Hermite interpolation with centered slopes."""

    grid = np.asarray(x, dtype=float).reshape(-1)
    values = np.asarray(y, dtype=float)
    if values.shape[0] != grid.size or grid.size < 3:
        raise ValueError("periodic cubic interpolation requires >= 3 samples")
    period = float(grid[-1] - grid[0] + (grid[1] - grid[0]))
    q = (np.asarray(query, dtype=float) - grid[0]) % period + grid[0]
    right = np.searchsorted(grid, q, side="right") % grid.size
    left = (right - 1) % grid.size
    xl = grid[left]
    xr = np.where(right == 0, grid[0] + period, grid[right])
    h = xr - xl
    t = (q - xl) / h
    prev = (left - 1) % grid.size
    nxt = (right + 1) % grid.size
    hp = np.where(left == 0, grid[left] - (grid[-1] - period), grid[left] - grid[prev])
    hn = np.where(right == 0, grid[1] - grid[0], grid[nxt] - grid[right])
    sl = (values[left] - values[prev]) / hp[..., None]
    sr = (values[nxt] - values[right]) / hn[..., None]
    m0 = 0.5 * (sl + (values[right] - values[left]) / h[..., None])
    m1 = 0.5 * ((values[right] - values[left]) / h[..., None] + sr)
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00[..., None] * values[left] + h10[..., None] * h[..., None] * m0 + h01[..., None] * values[right] + h11[..., None] * h[..., None] * m1
