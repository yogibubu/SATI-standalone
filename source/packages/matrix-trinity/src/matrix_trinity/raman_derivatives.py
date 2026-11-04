"""Cartesian polarizability derivative ordering and translation checks.

The six components are xx, xy, yy, xz, yz, zz. The Cartesian Jacobian
returned to TRINITY always has shape (6, 3N); no normal-mode derivative
is subjected to a Cartesian translation test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class RamanDerivativeLayout:
    jacobian_au_per_bohr: np.ndarray
    layout: str
    relative_translation_residuals: dict[str, float]
    equivalent_layouts: bool = False


def validate_cartesian_polarizability_jacobian(
    derivative,
    *,
    relative_tolerance=1e-5,
    absolute_tolerance=1e-10,
):
    """Require finite 6 x 3N data and vanishing rigid-translation derivatives.

    Return ||sum_atoms d(alpha)/dR|| / ||d(alpha)/dR||. Acceptance uses
    absolute_tolerance + relative_tolerance * ||d(alpha)/dR||, in a.u./bohr.
    """
    if (
        not np.isfinite(relative_tolerance)
        or relative_tolerance < 0
        or not np.isfinite(absolute_tolerance)
        or absolute_tolerance < 0
    ):
        raise ValueError("Raman translation tolerances must be finite and nonnegative")
    a = np.asarray(derivative, dtype=float)
    if (
        a.ndim != 2
        or a.shape[0] != 6
        or a.shape[1] == 0
        or a.shape[1] % 3
        or not np.all(np.isfinite(a))
    ):
        raise ValueError("polarizability Jacobian must be finite with shape 6 x 3N")
    norm = float(np.linalg.norm(a))
    residual = float(np.linalg.norm(a.reshape(6, -1, 3).sum(axis=1)))
    if residual > absolute_tolerance + relative_tolerance * norm:
        raise ValueError(
            f"polarizability derivatives violate translation invariance "
            f"(absolute residual {residual:.6g}, relative {residual / max(norm, 1e-300):.6g})"
        )
    return residual / max(norm, 1e-300)


def decode_polarizability_derivatives(
    raw,
    natoms,
    *,
    layout=None,
    relative_tolerance=1e-5,
    absolute_tolerance=1e-10,
):
    """Decode flat FCHK data, rejecting invalid and ambiguous orderings.

    An explicit layout still has to satisfy translation invariance. If both
    orderings pass, automatic selection requires equivalent Jacobians.
    """
    if isinstance(natoms, bool) or int(natoms) != natoms or natoms < 1:
        raise ValueError("natoms must be a positive integer")
    n3 = 3 * int(natoms)
    a = np.asarray(raw, dtype=float)
    if a.ndim != 1 or a.size != 6 * n3 or not np.all(np.isfinite(a)):
        raise ValueError(f"Polarizability Derivatives requires {6 * n3} finite flat values")
    candidates = {"cart-major": a.reshape(n3, 6).T, "component-major": a.reshape(6, n3)}
    if layout is not None and layout not in candidates:
        raise ValueError("layout must be cart-major or component-major")
    valid = {}
    scores = {}
    for name, jacobian in candidates.items():
        norm = float(np.linalg.norm(jacobian))
        residual = float(np.linalg.norm(jacobian.reshape(6, -1, 3).sum(axis=1)))
        scores[name] = residual / max(norm, 1e-300)
        try:
            validate_cartesian_polarizability_jacobian(
                jacobian,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        except ValueError:
            continue
        valid[name] = jacobian
    # Validate tolerances even when no candidate survives.
    if (
        not np.isfinite(relative_tolerance)
        or relative_tolerance < 0
        or not np.isfinite(absolute_tolerance)
        or absolute_tolerance < 0
    ):
        raise ValueError("Raman translation tolerances must be finite and nonnegative")
    if layout is not None:
        if layout not in valid:
            raise ValueError(
                f"requested Raman layout {layout} fails translation invariance: {scores}"
            )
        return RamanDerivativeLayout(valid[layout].copy(), layout, scores)
    if not valid:
        raise ValueError(f"no Raman derivative layout satisfies translation invariance: {scores}")
    equivalent = len(valid) == 2 and np.allclose(
        candidates["cart-major"], candidates["component-major"], rtol=1e-12, atol=absolute_tolerance
    )
    if len(valid) == 2 and not equivalent:
        raise ValueError(f"ambiguous Raman derivative layout; specify layout explicitly: {scores}")
    chosen = next(iter(valid))
    return RamanDerivativeLayout(valid[chosen].copy(), chosen, scores, equivalent)


def polarizability_jacobian_from_gaussian_fchk(path, **options):
    """Read only the atom count and polarizability derivative block from FCHK."""
    from matrix_gaussian.fchk import _read_fchk_blocks

    blocks = _read_fchk_blocks(Path(path))
    atoms = np.asarray(blocks.get("Atomic numbers", ()), dtype=int)
    if not atoms.size:
        raise ValueError("FCHK atomic numbers are missing")
    if "Polarizability Derivatives" not in blocks:
        raise ValueError("FCHK Polarizability Derivatives are missing")
    return decode_polarizability_derivatives(
        blocks["Polarizability Derivatives"], len(atoms), **options
    )
