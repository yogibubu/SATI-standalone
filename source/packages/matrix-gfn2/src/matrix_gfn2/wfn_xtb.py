"""Binary ``wfn.xtb`` interoperability for reference sTDA calculations."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from pathlib import Path
import struct

import numpy as np

from .basis import ContractedGaussian, GaussianPrimitive, overlap_matrix
from .electronic_state import GFN2OrbitalBasis, GFN2OrbitalState
from .parameters import ELEMENT_SYMBOLS


def _record(payload: bytes) -> bytes:
    length = len(payload)
    marker = struct.pack("<i", length)
    return marker + payload + marker


def _integer_record(value: int) -> bytes:
    return _record(struct.pack("<i", int(value)))


def _float_record(value: float) -> bytes:
    return _record(struct.pack("<d", float(value)))


def _array_record(values: np.ndarray, dtype: str, *, order: str = "C") -> bytes:
    array = np.asarray(values, dtype=dtype)
    return _record(array.tobytes(order=order))


_CARTESIAN_POWERS = {
    1: (0, 0, 0),
    2: (1, 0, 0),
    3: (0, 1, 0),
    4: (0, 0, 1),
    5: (2, 0, 0),
    6: (0, 2, 0),
    7: (0, 0, 2),
    8: (1, 1, 0),
    9: (1, 0, 1),
    10: (0, 1, 1),
}


def _odd_double_factorial(value: int) -> int:
    result = 1
    for factor in range(value, 0, -2):
        result *= factor
    return result


def _primitive_normalization(exponent: float, ao_type: int) -> float:
    powers = _CARTESIAN_POWERS[int(ao_type)]
    angular = sum(powers)
    denominator = sqrt(
        _odd_double_factorial(2 * powers[0] - 1)
        * _odd_double_factorial(2 * powers[1] - 1)
        * _odd_double_factorial(2 * powers[2] - 1)
    )
    return (2.0 * exponent / pi) ** 0.75 * (4.0 * exponent) ** (
        angular / 2.0
    ) / denominator


@dataclass(frozen=True)
class WfnXTBMetadata:
    spin_channels: int
    atoms: int
    aos: int
    orbitals_per_spin: int
    primitives: int


class _FortranRecordReader:
    """Strict reader for the little-endian sequential records used by std2."""

    def __init__(self, path: str | Path):
        self._payload = memoryview(Path(path).read_bytes())
        self._offset = 0

    def record(self) -> bytes:
        if self._offset + 8 > len(self._payload):
            raise ValueError("truncated wfn.xtb record")
        length = struct.unpack_from("<i", self._payload, self._offset)[0]
        start = self._offset + 4
        stop = start + length
        if length < 0 or stop + 4 > len(self._payload):
            raise ValueError("invalid wfn.xtb record length")
        closing = struct.unpack_from("<i", self._payload, stop)[0]
        if closing != length:
            raise ValueError("mismatched wfn.xtb record markers")
        self._offset = stop + 4
        return bytes(self._payload[start:stop])

    def integer(self) -> int:
        payload = self.record()
        if len(payload) != 4:
            raise ValueError("wfn.xtb integer record has the wrong size")
        return int(struct.unpack("<i", payload)[0])

    def scalar(self) -> float:
        payload = self.record()
        if len(payload) != 8:
            raise ValueError("wfn.xtb real record has the wrong size")
        return float(struct.unpack("<d", payload)[0])

    def array(self, size: int) -> np.ndarray:
        payload = self.record()
        if len(payload) != 8 * int(size):
            raise ValueError("wfn.xtb real-array record has the wrong size")
        return np.frombuffer(payload, dtype="<f8").astype(float, copy=True)

    def require_end(self) -> None:
        if self._offset != len(self._payload):
            raise ValueError("unexpected trailing records in wfn.xtb")


def write_wfn_xtb(path: str | Path, state: GFN2OrbitalState) -> WfnXTBMetadata:
    """Write the sequential-unformatted file consumed by Grimme's ``stda``.

    The format intentionally mirrors ``printmos.f``/``printumos``: four-byte
    little-endian Fortran record markers, one record per scalar basis index,
    and column-major MO coefficient records.  Open-shell restricted MATRIX
    states are emitted as two spin channels with shared spatial coefficients.
    """

    destination = Path(path)
    channels = 2 if state.multiplicity > 1 else 1
    natoms = len(state.atomic_numbers)
    basis = state.basis
    nao = basis.ncartesian
    nmo = state.nmo
    metadata = WfnXTBMetadata(channels, natoms, nao, nmo, basis.nprimitives)
    chunks = [
        _integer_record(channels),
        _record(struct.pack("<4i", natoms, nao, nmo, basis.nprimitives)),
    ]
    for number in state.atomic_numbers:
        symbol = ELEMENT_SYMBOLS[int(number) - 1].ljust(2)[:2].encode("ascii")
        chunks.append(_record(symbol))
    for coordinates, number in zip(
        state.coordinates_bohr, state.atomic_numbers, strict=True
    ):
        chunks.extend(_float_record(value) for value in coordinates)
        chunks.append(_integer_record(int(number)))

    primitive_atom: list[int] = []
    primitive_ao: list[int] = []
    primitive_angular: list[int] = []
    for ao in range(nao):
        start = int(basis.primitive_offsets[ao])
        stop = int(basis.primitive_offsets[ao + 1])
        count = stop - start
        primitive_angular.extend([int(basis.cartesian_type[ao])] * count)
        primitive_atom.extend([int(basis.cartesian_atom[ao]) + 1] * count)
        primitive_ao.extend([ao + 1] * count)
    for values in (primitive_angular, primitive_atom, primitive_ao):
        chunks.extend(_integer_record(value) for value in values)
    chunks.extend(
        (
            _array_record(basis.primitive_exponents, "<f8"),
            _array_record(
                np.asarray(basis.primitive_coefficients)
                * np.asarray(
                    [
                        _primitive_normalization(exponent, ao_type)
                        for exponent, ao_type in zip(
                            basis.primitive_exponents,
                            primitive_angular,
                            strict=True,
                        )
                    ]
                ),
                "<f8",
            ),
        )
    )
    if channels == 1:
        export_coefficients = basis.spherical_to_cartesian @ state.coefficients
        chunks.extend(
            (
                _array_record(
                    state.alpha_occupations + state.beta_occupations, "<f8"
                ),
                _array_record(state.orbital_energies_hartree, "<f8"),
                _array_record(export_coefficients, "<f8", order="F"),
            )
        )
    else:
        export_coefficients = basis.spherical_to_cartesian @ state.coefficients
        chunks.extend(
            (
                _array_record(state.alpha_occupations, "<f8"),
                _array_record(state.orbital_energies_hartree, "<f8"),
                _array_record(state.beta_occupations, "<f8"),
                _array_record(state.orbital_energies_hartree, "<f8"),
                _array_record(export_coefficients, "<f8", order="F"),
                _array_record(export_coefficients, "<f8", order="F"),
            )
        )
    destination.write_bytes(b"".join(chunks))
    return metadata


def inspect_wfn_xtb(path: str | Path) -> WfnXTBMetadata:
    """Read and validate the leading records of a ``wfn.xtb`` file."""

    reader = _FortranRecordReader(path)
    channels = reader.integer()
    if channels not in (1, 2):
        raise ValueError("wfn.xtb spin-channel marker must be one or two")
    dimensions = reader.record()
    if len(dimensions) != 16:
        raise ValueError("wfn.xtb dimension record has the wrong size")
    atoms, aos, orbitals, primitives = struct.unpack("<4i", dimensions)
    if min(atoms, aos, orbitals, primitives) <= 0:
        raise ValueError("wfn.xtb dimensions must be positive")
    return WfnXTBMetadata(channels, atoms, aos, orbitals, primitives)


def read_wfn_xtb(
    path: str | Path,
    *,
    model_id: str = "WFN-XTB-IMPORTED.v1",
    molecular_charge: int = 0,
) -> GFN2OrbitalState:
    """Read a restricted Grimme ``wfn.xtb`` into the resident TDA state.

    This adapter is deliberately strict and currently accepts the one-channel
    closed-shell format used for absolute ``std2 -xtb`` qualification.  It
    reconstructs the AO metric from the serialized Gaussian basis rather than
    assuming an orthogonal basis.  The resulting state therefore lets MATRIX
    and std2 diagonalize *the same* orbital reference; it does not relabel a
    MATRIX-GFN2 ground state as Grimme's distinct ``xtb4stda`` model.
    """

    reader = _FortranRecordReader(path)
    channels = reader.integer()
    if channels != 1:
        raise ValueError(
            "read_wfn_xtb currently supports the restricted closed-shell "
            "one-channel format only"
        )
    dimensions = reader.record()
    if len(dimensions) != 16:
        raise ValueError("wfn.xtb dimension record has the wrong size")
    natoms, nao, nmo, nprimitives = struct.unpack("<4i", dimensions)
    if min(natoms, nao, nmo, nprimitives) <= 0:
        raise ValueError("wfn.xtb dimensions must be positive")
    symbols = []
    for _ in range(natoms):
        symbol = reader.record()
        if len(symbol) != 2:
            raise ValueError("wfn.xtb element-symbol record has the wrong size")
        symbols.append(symbol.decode("ascii").strip())
    coordinates = np.empty((natoms, 3), dtype=float)
    atomic_numbers = np.empty(natoms, dtype=np.int64)
    for atom in range(natoms):
        coordinates[atom] = [reader.scalar() for _ in range(3)]
        atomic_numbers[atom] = reader.integer()
    for symbol, number in zip(symbols, atomic_numbers, strict=True):
        expected = ELEMENT_SYMBOLS[int(number) - 1]
        if symbol.lower() != expected.lower():
            raise ValueError("wfn.xtb element symbol and atomic number differ")

    primitive_type = np.asarray(
        [reader.integer() for _ in range(nprimitives)], dtype=np.int64
    )
    primitive_atom = np.asarray(
        [reader.integer() - 1 for _ in range(nprimitives)], dtype=np.int64
    )
    primitive_ao = np.asarray(
        [reader.integer() - 1 for _ in range(nprimitives)], dtype=np.int64
    )
    exponents = reader.array(nprimitives)
    normalized_coefficients = reader.array(nprimitives)
    occupations = reader.array(nmo)
    energies = reader.array(nmo)
    coefficients = reader.array(nao * nmo).reshape((nao, nmo), order="F")
    reader.require_end()

    if np.any(primitive_type < 1) or np.any(primitive_type > 10):
        raise ValueError("wfn.xtb contains unsupported Cartesian AO types")
    if np.any(primitive_atom < 0) or np.any(primitive_atom >= natoms):
        raise ValueError("wfn.xtb primitive atom index is outside the molecule")
    if np.any(primitive_ao < 0) or np.any(primitive_ao >= nao):
        raise ValueError("wfn.xtb primitive AO index is outside the basis")
    if np.any(exponents <= 0.0) or not np.all(np.isfinite(exponents)):
        raise ValueError("wfn.xtb primitive exponents must be finite and positive")
    if np.any(occupations < -1.0e-10) or np.any(occupations > 2.0 + 1.0e-10):
        raise ValueError("wfn.xtb restricted occupations must lie in [0, 2]")

    offsets = np.zeros(nao + 1, dtype=np.int64)
    ao_atom = np.empty(nao, dtype=np.int64)
    ao_type = np.empty(nao, dtype=np.int64)
    raw_coefficients = np.empty(nprimitives, dtype=float)
    orbitals: list[ContractedGaussian] = []
    shell_names = {1: "s", 2: "p", 3: "p", 4: "p"}
    for ao in range(nao):
        indices = np.flatnonzero(primitive_ao == ao)
        if not len(indices):
            raise ValueError("wfn.xtb contains an AO without primitives")
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError("wfn.xtb primitives must be grouped by AO")
        if np.any(primitive_atom[indices] != primitive_atom[indices[0]]) or np.any(
            primitive_type[indices] != primitive_type[indices[0]]
        ):
            raise ValueError("wfn.xtb AO primitives disagree on atom or angular type")
        offsets[ao + 1] = offsets[ao] + len(indices)
        if int(indices[0]) != int(offsets[ao]):
            raise ValueError("wfn.xtb AO primitive groups are not ordered")
        atom = int(primitive_atom[indices[0]])
        kind = int(primitive_type[indices[0]])
        ao_atom[ao] = atom
        ao_type[ao] = kind
        for primitive in indices:
            raw_coefficients[primitive] = normalized_coefficients[primitive] / (
                _primitive_normalization(float(exponents[primitive]), kind)
            )
        angular = _CARTESIAN_POWERS[kind]
        shell = shell_names.get(kind, "d")
        orbitals.append(
            ContractedGaussian(
                tuple(float(value) for value in coordinates[atom]),
                angular,
                tuple(
                    GaussianPrimitive(
                        float(exponents[primitive]),
                        float(raw_coefficients[primitive]),
                    )
                    for primitive in indices
                ),
                atom,
                shell,
                float(np.sqrt(np.average(exponents[indices]))),
            )
        )
    overlap = overlap_matrix(orbitals)
    metric = coefficients.T @ overlap @ coefficients
    if not np.allclose(metric, np.eye(nmo), atol=2.0e-7, rtol=2.0e-7):
        raise ValueError("wfn.xtb orbitals are not orthonormal in its reconstructed AO metric")
    fock = overlap @ coefficients @ np.diag(energies) @ coefficients.T @ overlap
    fock = 0.5 * (fock + fock.T)
    basis = GFN2OrbitalBasis(
        ao_atom,
        ao_type,
        offsets,
        exponents,
        raw_coefficients,
        np.eye(nao),
    )
    electron_count = int(round(float(np.sum(occupations))))
    if abs(float(np.sum(occupations)) - electron_count) > 2.0e-8:
        raise ValueError("wfn.xtb occupations do not define an integer electron count")
    return GFN2OrbitalState(
        str(model_id),
        atomic_numbers,
        coordinates,
        overlap,
        fock,
        coefficients,
        energies,
        0.5 * occupations,
        0.5 * occupations,
        ao_atom,
        tuple(shell_names.get(int(value), "d") for value in ao_type),
        basis,
        0.0,
        electron_count,
        int(molecular_charge),
        1,
    )


__all__ = ["WfnXTBMetadata", "inspect_wfn_xtb", "read_wfn_xtb", "write_wfn_xtb"]
