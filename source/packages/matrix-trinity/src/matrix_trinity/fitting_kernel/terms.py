from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .basis import basis_value, basis_derivative


@dataclass(frozen=True)
class Term:
    exps: Tuple[Tuple[int, int], ...]  # (index, exponent) pairs, index 0-based


def generate_terms(nvib: int) -> List[Term]:
    terms = []
    # diagonals
    for i in range(nvib):
        for p in range(2, 7):
            terms.append(Term(((i, p),)))
    # two-index terms (up to 4th order)
    for i in range(nvib):
        for j in range(i + 1, nvib):
            pairs = [(1, 1), (2, 1), (1, 2), (2, 2)]
            for e1, e2 in pairs:
                terms.append(Term(((i, e1), (j, e2))))
    # three-index terms (up to 4th order)
    for i in range(nvib):
        for j in range(i + 1, nvib):
            for k in range(j + 1, nvib):
                terms.append(Term(((i, 1), (j, 1), (k, 1))))
                terms.append(Term(((i, 2), (j, 1), (k, 1))))
                terms.append(Term(((i, 1), (j, 2), (k, 1))))
                terms.append(Term(((i, 1), (j, 1), (k, 2))))
    return terms


def eval_terms(q: np.ndarray, terms: List[Term], basis_cfg: Dict[int, Dict]):
    """Return phi (nterms) and dphi (nvib, nterms)."""
    nvib = len(q)
    nterms = len(terms)
    phi = np.ones(nterms)
    dphi = np.zeros((nvib, nterms))

    for t_idx, term in enumerate(terms):
        # precompute factors
        factors = []
        derivs = []
        for i, exp in term.exps:
            cfg = basis_cfg.get(i, {})
            mode = cfg.get("mode", "poly")
            params = cfg.get("params", {})
            x = q[i]
            val = basis_value(mode, x, exp, params)
            dval = basis_derivative(mode, x, exp, params)
            factors.append((i, val))
            derivs.append((i, dval))
        # term value
        v = 1.0
        for _, val in factors:
            v *= val
        phi[t_idx] = v
        # derivatives
        for i, dval in derivs:
            if dval == 0.0:
                continue
            dv = dval
            for j, val in factors:
                if j == i:
                    continue
                dv *= val
            dphi[i, t_idx] = dv

    return phi, dphi


def load_terms(path) -> List[Term]:
    terms = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # drop trailing coefficient if present
            if any(ch in parts[-1] for ch in ".eE"):
                parts = parts[:-1]
            idxs = [int(p) for p in parts]
            exp_map = {}
            for i in idxs:
                ii = i - 1  # one-based to zero-based
                exp_map[ii] = exp_map.get(ii, 0) + 1
            exps = tuple(sorted(exp_map.items(), key=lambda x: x[0]))
            terms.append(Term(exps))
    return terms
