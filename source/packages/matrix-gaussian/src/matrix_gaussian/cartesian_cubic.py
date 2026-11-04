from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from matrix_chem import MolecularGeometry

from .archive import last_post_nimag_groups, number_array, unpack_symmetric_matrix
from .parsers import read_gaussian_log_geometry
from .writers import GaussianWriteError, write_gaussian_point_input


@dataclass(frozen=True)
class GaussianCartesianCubicForceField:
    """Born--Oppenheimer Cartesian quadratic and cubic force field.

    Gaussian archive units are retained: hartree/bohr**2 for ``hessian`` and
    hartree/bohr**3 for ``cubic``.  Both tensors and ``geometry`` are in the
    archive (input/original) Cartesian frame.
    """

    path: Path
    geometry: MolecularGeometry
    gradient_hartree_per_bohr: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray
    cubic_unique_indices: np.ndarray
    cubic_unique_values_hartree_per_bohr3: np.ndarray

    @property
    def dimension(self) -> int:
        return 3 * self.geometry.natoms

    @property
    def cubic_hartree_per_bohr3(self) -> np.ndarray:
        """Materialize the full tensor only for compatibility or diagnostics."""
        return _dense_symmetric_rank3(
            self.cubic_unique_indices,
            self.cubic_unique_values_hartree_per_bohr3,
            self.dimension,
        )


def read_gaussian_cartesian_cubic_force_field(
    path: Path | str,
) -> GaussianCartesianCubicForceField:
    """Read the full Cartesian F2/F3 tensors written by ``Freq=Cubic``.

    Gaussian writes three archive blocks after ``NImag``: packed symmetric
    F2, the Cartesian gradient, and packed fully symmetric F3.  Their lengths
    provide a strict integrity check, preventing a harmonic-only archive from
    being mistaken for a cubic force field.
    """

    target = Path(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    geometry = read_gaussian_log_geometry(target)
    dimension = 3 * geometry.natoms
    groups = last_post_nimag_groups(text)
    expected_hessian = dimension * (dimension + 1) // 2
    expected_cubic = dimension * (dimension + 1) * (dimension + 2) // 6
    if len(groups) < 3:
        raise ValueError("Gaussian archive does not contain Cartesian F2/gradient/F3 blocks")
    packed_hessian = number_array(groups[0])
    gradient = number_array(groups[1])
    packed_cubic = number_array(groups[2])
    if packed_hessian.size != expected_hessian:
        raise ValueError(
            f"Gaussian Cartesian Hessian size mismatch: expected {expected_hessian}, "
            f"found {packed_hessian.size}"
        )
    if gradient.size != dimension:
        raise ValueError(
            f"Gaussian Cartesian gradient size mismatch: expected {dimension}, "
            f"found {gradient.size}"
        )
    if packed_cubic.size != expected_cubic:
        raise ValueError(
            "Gaussian archive is not a full Freq=Cubic result: expected "
            f"{expected_cubic} symmetric cubic constants, found {packed_cubic.size}"
        )
    cubic_indices, cubic_values = _packed_rank3_indices(packed_cubic, dimension)
    return GaussianCartesianCubicForceField(
        path=target,
        geometry=geometry,
        gradient_hartree_per_bohr=gradient,
        hessian_hartree_per_bohr2=unpack_symmetric_matrix(packed_hessian, dimension),
        cubic_unique_indices=cubic_indices,
        cubic_unique_values_hartree_per_bohr3=cubic_values,
    )


def write_gaussian_cartesian_cubic_input(
    output: Path | str,
    atoms: tuple[str, ...] | list[str],
    coordinates_angstrom,
    *,
    route: str = "#p B3LYP/6-31+G(d) EmpiricalDispersion=GD3BJ Freq=Cubic",
    title: str = "MATRIX MORPHEUS Cartesian cubic force field",
    charge: int = 0,
    multiplicity: int = 1,
    link0: tuple[str, ...] = (),
) -> Path:
    if not re.search(r"\bfreq\s*=\s*cubic\b", route, flags=re.IGNORECASE):
        raise GaussianWriteError("Cartesian cubic-force input requires Freq=Cubic")
    return write_gaussian_point_input(
        Path(output),
        atoms,
        coordinates_angstrom,
        route=route,
        title=title,
        charge=charge,
        multiplicity=multiplicity,
        link0=link0,
        ensure_force=False,
    )


def _packed_rank3_indices(values: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    indices: list[tuple[int, int, int]] = []
    retained: list[float] = []
    cursor = 0
    for i in range(size):
        for j in range(i + 1):
            for k in range(j + 1):
                value = values[cursor]
                cursor += 1
                if value != 0.0:
                    indices.append((i, j, k))
                    retained.append(float(value))
    return np.asarray(indices, dtype=int).reshape((-1, 3)), np.asarray(retained, dtype=float)


def _dense_symmetric_rank3(
    indices: np.ndarray, values: np.ndarray, size: int
) -> np.ndarray:
    result = np.zeros((size, size, size), dtype=float)
    from itertools import permutations

    for index, value in zip(indices, values, strict=True):
        for permuted in set(permutations(tuple(int(item) for item in index))):
            result[permuted] = value
    return result
