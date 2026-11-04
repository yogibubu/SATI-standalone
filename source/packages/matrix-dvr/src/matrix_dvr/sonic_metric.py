"""ESALC Wilson B/G and rotational diagnostics for DVR hand-offs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def build_esalc_rotational_diagnostics(
    xyzin: Path | str,
    coordinates: Sequence[str | int],
) -> dict[str, object]:
    """Build the selected ESALC B/G block and its rotational tensor.

    The frozen ORACLE/ESALC definition is the single source of coordinate
    labels.  ``B`` is built by SMITH and ``G = B M^-1 B^T`` is constructed in
    the same Wilson convention consumed by MATRIX-GF/LINK.  One or two
    coordinates are accepted because these are the 1D and product-DVR cases.
    """

    requested = tuple(coordinates)
    if len(requested) not in {1, 2}:
        raise ValueError("ESALC tunneling selection must contain one or two coordinates")
    target = Path(xyzin).expanduser().resolve()
    from matrix_chem import atomic_mass
    from matrix_chem.topology.elements import atomic_number
    from matrix_chem.xyzin_geometry import read_xyzin_geometry
    from matrix_smith import build_gic_b_matrix, read_gic_definition_from_xyzin

    definition = read_gic_definition_from_xyzin(target)
    geometry = np.asarray(definition.reference_coordinates_angstrom, dtype=float)
    atoms = tuple(str(atom) for atom in read_xyzin_geometry(target).atoms)
    masses = np.asarray([atomic_mass(atomic_number(atom) or 0) for atom in atoms], dtype=float)
    if np.any(masses <= 0.0):
        raise ValueError("ESALC diagnostic contains an atom with no positive mass")
    matrix = build_gic_b_matrix(definition, coordinates_angstrom=geometry)
    b_matrix = np.asarray(matrix.rows, dtype=float)
    labels = tuple(str(item.identifier) for item in definition.gics)
    names = tuple(str(item.name) for item in definition.gics)
    indices = tuple(_coordinate_index(item, labels, names) for item in requested)
    inverse_mass = np.repeat(1.0 / masses, 3)
    g_matrix = b_matrix @ (inverse_mass[:, None] * b_matrix.T)
    selected_b = b_matrix[np.asarray(indices, dtype=int)]
    selected_g = g_matrix[np.ix_(indices, indices)]
    inertia = _inertia_tensor(masses, geometry)
    moments = np.linalg.eigvalsh(inertia)
    constants = _rotational_constants_mhz(moments)
    return {
        "source_xyzin": str(target),
        "point_group": str(definition.point_group),
        "coordinates": [
            {"index": index + 1, "label": labels[index], "name": names[index]}
            for index in indices
        ],
        "B_shape": list(b_matrix.shape),
        "B_selected": selected_b.tolist(),
        "G_selected": selected_g.tolist(),
        "G_eigenvalues_selected": np.linalg.eigvalsh(selected_g).tolist(),
        "rotational_constants_MHz": {
            "A": constants[0],
            "B": constants[1],
            "C": constants[2],
        },
        "inertia_principal_amu_angstrom2": moments.tolist(),
        "convention": "Wilson G = B M^-1 B^T; B in inverse Angstrom; rotational tensor from ORACLE reference geometry",
    }


def _coordinate_index(coordinate: str | int, labels: tuple[str, ...], names: tuple[str, ...]) -> int:
    if isinstance(coordinate, int) or str(coordinate).strip().isdigit():
        index = int(coordinate)
        if not 1 <= index <= len(labels):
            raise ValueError(f"ESALC coordinate index {index} outside 1..{len(labels)}")
        return index - 1
    wanted = str(coordinate)
    if wanted in labels:
        return labels.index(wanted)
    if wanted in names:
        return names.index(wanted)
    raise ValueError(f"unknown ESALC coordinate {wanted!r}")


def _inertia_tensor(masses: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    center = np.sum(masses[:, None] * coordinates, axis=0) / np.sum(masses)
    centered = coordinates - center
    inertia = np.zeros((3, 3), dtype=float)
    for mass, vector in zip(masses, centered, strict=True):
        inertia += mass * (float(vector @ vector) * np.eye(3) - np.outer(vector, vector))
    return 0.5 * (inertia + inertia.T)


def _rotational_constants_mhz(moments: np.ndarray) -> tuple[float, float, float]:
    # h/(8 pi^2 I) converted from amu Angstrom^2 to MHz.
    factor = 505379.0090099937
    return tuple(float(factor / moment) if moment > 1.0e-12 else float("inf") for moment in moments)


build_sonic_rotational_diagnostics = build_esalc_rotational_diagnostics


__all__ = ["build_esalc_rotational_diagnostics"]
