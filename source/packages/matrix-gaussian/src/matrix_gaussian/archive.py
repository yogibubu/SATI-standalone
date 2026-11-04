from __future__ import annotations

import re

import numpy as np


def archive_cartesian_dipole_derivatives(text: str, *, size: int) -> np.ndarray:
    """Return d(mu_a)/d(x_i), in atomic units and original archive axes.

    Gaussian stores the three dipole components consecutively for each
    Cartesian displacement. Only the last complete job is inspected, so a
    missing property cannot silently select a different electronic method.
    """
    if size <= 0 or size % 3:
        raise ValueError("a positive Cartesian dimension divisible by three is required")
    starts = [match.start() for match in re.finditer(r"1\\1\\GINC", text)]
    for start in reversed(starts):
        end = text.find(r"\@", start)
        if end < 0:
            continue
        compact = re.sub(r"\s+", "", text[start:end + 2])
        match = re.search(r"\\DipoleDeriv=([^\\]+)", compact)
        if match is None:
            raise ValueError("last complete Gaussian archive has no DipoleDeriv")
        values = number_array(match.group(1))
        if values.size != 3 * size:
            raise ValueError(f"DipoleDeriv size mismatch: expected {3 * size}, found {values.size}")
        return values.reshape(size, 3).T.copy()
    raise ValueError("Gaussian log contains no complete archive entry")


def last_post_nimag_groups(text: str) -> tuple[str, ...]:
    """Return numeric archive groups following the last complete ``NImag`` entry."""
    starts = [match.start() for match in re.finditer(r"1\\1\\GINC", text)]
    for start in reversed(starts):
        end = text.find(r"\@", start)
        if end < 0:
            continue
        compact = re.sub(r"\s+", "", text[start : end + 2])
        marker = re.search(r"\\NImag=-?\d+\\\\", compact)
        if marker is None:
            continue
        body = compact[marker.end() :]
        groups = tuple(item for item in body.split(r"\\") if item and item != "@")
        if groups:
            return groups
    raise ValueError("Gaussian log contains no complete archive entry with NImag")


def number_array(text: str) -> np.ndarray:
    if not text:
        return np.empty(0, dtype=float)
    try:
        values = [float(item.replace("D", "E").replace("d", "e")) for item in text.split(",")]
    except ValueError as exc:
        raise ValueError("invalid numeric value in Gaussian archive force field") from exc
    result = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite numeric value in Gaussian archive force field")
    return result


def unpack_symmetric_matrix(values: np.ndarray, size: int) -> np.ndarray:
    expected = size * (size + 1) // 2
    if values.size != expected:
        raise ValueError(
            f"Gaussian Cartesian Hessian size mismatch: expected {expected}, "
            f"found {values.size}"
        )
    result = np.zeros((size, size), dtype=float)
    result[np.tril_indices(size)] = values
    result += np.tril(result, -1).T
    return result


def archive_cartesian_hessian(text: str, *, size: int) -> np.ndarray:
    """Read packed Cartesian F2 from the last complete Gaussian archive entry."""
    groups = last_post_nimag_groups(text)
    if not groups:
        raise ValueError("Gaussian archive contains no Cartesian force-constant block")
    return unpack_symmetric_matrix(number_array(groups[0]), size)
