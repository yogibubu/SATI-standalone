"""Mode-resolved algebraic anharmonicities, in dimensionless normal modes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class AnharmonicityContribution:
    """One additive term in x_ij; mode indices are zero based."""

    row: int
    column: int
    modes: tuple[int, ...]
    expression: str
    numerator_cm2: float | None
    denominator_cm: float | None
    value_cm: float
    potential_cm: float
    kinetic_cm: float
    mixed_cm: float
    running_sum_cm: float


@dataclass(frozen=True)
class AnharmonicityDiagnostics:
    frequencies_cm: np.ndarray
    x_matrix_cm: np.ndarray
    contributions: tuple[AnharmonicityContribution, ...]

    def write_json(self, path: str | Path) -> None:
        """Write all terms, including exact zeros, and their final matrix."""
        Path(path).write_text(
            json.dumps(
                {
                    "mode_index_base": 0,
                    "units": "cm^-1",
                    "decomposition": "potential + kinetic + mixed",
                    "frequencies_cm": self.frequencies_cm.tolist(),
                    "x_matrix_cm": self.x_matrix_cm.tolist(),
                    "contributions": [asdict(term) for term in self.contributions],
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )


def _x_terms(w, f3, f4, g1, g2):
    """Yield i,j,modes,expression,numerator,denominator in final x units."""
    n = len(w)

    def eta(i, j, k, l):
        return float(f4[i, j, k, l] + g2[i, j, k, l] + g2[k, l, i, j])

    def s(i, j, k):
        return float(f3[i, j, k] - g1[i, j, k] - g1[i, k, j] - g1[j, k, i])

    def r(i, j, k):
        return float(f3[i, j, k] - g1[i, j, k] + g1[i, k, j] + g1[j, k, i])

    for i in range(n):
        wi = float(w[i])
        yield i, i, (i,), "eta_iiii/16", eta(i, i, i, i) / 16, None
        yield i, i, (i,), "-sigma_iii^2/(96 wi)", -(s(i, i, i) ** 2) / 96, wi
        yield i, i, (i,), "-9 rho_iii^2/(96 wi)", -9 * r(i, i, i) ** 2 / 96, wi
        for j in range(n):
            if j == i:
                continue
            wj = float(w[j])
            yield i, i, (i, j), "-rho_jii^2/(8 wj)", -(r(j, i, i) ** 2) / 8, wj
            yield i, i, (i, j), "-sigma_iij^2/[32(2wi+wj)]", -(s(i, i, j) ** 2) / 32, 2 * wi + wj
            yield i, i, (i, j), "+rho_iij^2/[32(2wi-wj)]", r(i, i, j) ** 2 / 32, 2 * wi - wj
        for j in range(i + 1, n):
            wj = float(w[j])
            yield i, j, (i, j), "eta_iijj/4", eta(i, i, j, j) / 4, None
            yield i, j, (i, j), "-sigma_iij^2/[8(2wi+wj)]", -(s(i, i, j) ** 2) / 8, 2 * wi + wj
            yield i, j, (i, j), "-rho_iij^2/[8(2wi-wj)]", -(r(i, i, j) ** 2) / 8, 2 * wi - wj
            yield i, j, (i, j), "-sigma_jji^2/[8(2wj+wi)]", -(s(j, j, i) ** 2) / 8, 2 * wj + wi
            yield i, j, (i, j), "-rho_jji^2/[8(2wj-wi)]", -(r(j, j, i) ** 2) / 8, 2 * wj - wi
            yield i, j, (i, j), "-rho_iii rho_ijj/(4wi)", -r(i, i, i) * r(i, j, j) / 4, wi
            yield i, j, (i, j), "-rho_jjj rho_jii/(4wj)", -r(j, j, j) * r(j, i, i) / 4, wj
            for k in range(n):
                if k in (i, j):
                    continue
                wk = float(w[k])
                yield (
                    i,
                    j,
                    (i, j, k),
                    "-sigma_ijk^2/[8(wi+wj+wk)]",
                    -(s(i, j, k) ** 2) / 8,
                    wi + wj + wk,
                )
                yield i, j, (i, j, k), "+rho_ijk^2/[8(wi+wj-wk)]", r(i, j, k) ** 2 / 8, wi + wj - wk
                yield (
                    i,
                    j,
                    (i, j, k),
                    "-rho_ikj^2/[8(wi-wj+wk)]",
                    -(r(i, k, j) ** 2) / 8,
                    wi - wj + wk,
                )
                yield i, j, (i, j, k), "+rho_jki^2/[8(wi-wj-wk)]", r(j, k, i) ** 2 / 8, wi - wj - wk
                yield i, j, (i, j, k), "-rho_kii rho_kjj/(4wk)", -r(k, i, i) * r(k, j, j) / 4, wk


def _value(numerator, denominator, tolerance):
    if denominator is None:
        return float(numerator)
    if numerator == 0:
        return 0.0
    if abs(denominator) <= tolerance:
        raise ValueError("SONIC VPT2 x has a resonant denominator; use GVPT2")
    return float(numerator / denominator)


def algebraic_x_matrix(w, f3, f4, g1, g2, *, denominator_tolerance_cm):
    """Production contraction; diagnostics use the very same additive terms."""
    x = np.zeros((len(w), len(w)))
    for i, j, _modes, _label, num, den in _x_terms(w, f3, f4, g1, g2):
        x[i, j] += _value(num, den, denominator_tolerance_cm)
    return x + np.triu(x, 1).T


def sonic_vpt2_x_diagnostics(field, *, denominator_tolerance_cm=1e-6):
    """Resolve x into potential, kinetic and interference terms.

    The decomposition is evaluated in one fixed normal-mode representation.
    A resonant total numerator is rejected by the production denominator rule.
    No individual cancellation is interpreted as a new resonance.
    """
    from .curvilinear_vpt2 import _modalize_sonic_field

    if not np.isfinite(denominator_tolerance_cm) or denominator_tolerance_cm <= 0:
        raise ValueError("VPT2 denominator tolerance must be positive and finite")
    w, _m, f3, f4, g1, g2 = _modalize_sonic_field(field)
    full = _x_terms(w, f3, f4, g1, g2)
    potential = _x_terms(w, f3, f4, np.zeros_like(g1), np.zeros_like(g2))
    kinetic = _x_terms(w, np.zeros_like(f3), np.zeros_like(f4), g1, g2)
    x = np.zeros((len(w), len(w)))
    rows = []
    for term, vp, kt in zip(full, potential, kinetic, strict=True):
        i, j, modes, label, num, den = term
        value = _value(num, den, denominator_tolerance_cm)
        # At an exactly zero denominator a cancelled total cannot be split.
        if den is not None and abs(den) <= denominator_tolerance_cm and (vp[4] or kt[4]):
            raise ValueError("resonant potential/kinetic decomposition is undefined; use GVPT2")
        v = _value(vp[4], den, denominator_tolerance_cm)
        k = _value(kt[4], den, denominator_tolerance_cm)
        x[i, j] += value
        rows.append(
            AnharmonicityContribution(
                i,
                j,
                modes,
                label,
                None if den is None else num,
                den,
                value,
                v,
                k,
                value - v - k,
                float(x[i, j]),
            )
        )
    return AnharmonicityDiagnostics(w, x + np.triu(x, 1).T, tuple(rows))
