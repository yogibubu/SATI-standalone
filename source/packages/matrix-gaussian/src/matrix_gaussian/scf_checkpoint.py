"""Lossless Gaussian checkpoint transport for LINK SCF guesses.

Gaussian's ordinary formatted checkpoint omits the molecular Fock matrix.
The native binary-array checkpoint export (FAF), in contrast, carries the
density, Fock, and molecular-orbital arrays as binary float64 values.  This
module provides the small, audited bridge needed to reuse MATRIX's general
SCF-state projection and extrapolation lifecycle without parsing rounded log
tables or depending on a particular Gaussian installation layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import struct
import subprocess
from typing import Mapping, Sequence

import numpy as np


GAUSSIAN_SCF_ARRAY_SCHEMA = "matrix.gaussian.scf-binary-array.v1"
_SCF_LABELS = frozenset(
    {
        "ALPHA MO COEFFICIENTS",
        "BETA MO COEFFICIENTS",
        "ALPHA DENSITY MATRIX",
        "BETA DENSITY MATRIX",
        "ALPHA SCF DENSITY MATRIX",
        "BETA SCF DENSITY MATRIX",
        "ALPHA FOCK MATRIX",
        "BETA FOCK MATRIX",
        "SCF ORBITAL OCCUPATIONS",
    }
)


class GaussianSCFCheckpointError(RuntimeError):
    """A Gaussian checkpoint cannot satisfy the joint density/Fock contract."""


@dataclass(frozen=True)
class GaussianBinaryArray:
    label: str
    values: np.ndarray
    dimensions: tuple[int, int, int, int, int]
    integer_count: int
    real_count: int
    total_count: int
    buffer_length: int
    value_type: int


@dataclass(frozen=True)
class GaussianProjectedSCFState:
    """One converged state represented in the target Gaussian AO basis."""

    source_checkpoint: Path
    projected_checkpoint: Path
    source_coordinates_angstrom: np.ndarray
    target_coordinates_angstrom: np.ndarray
    density_alpha: np.ndarray
    fock_alpha: np.ndarray
    orbitals_alpha: np.ndarray
    occupations_alpha: np.ndarray
    density_beta: np.ndarray | None = None
    fock_beta: np.ndarray | None = None
    orbitals_beta: np.ndarray | None = None
    occupations_beta: np.ndarray | None = None
    overlap: np.ndarray | None = None
    electron_count: float | None = None
    projection_residual: float = 0.0


@dataclass(frozen=True)
class GaussianSCFGuessPreparation:
    checkpoint: Path
    status: str
    source_state_count: int
    projection_count: int
    projection_residual_max: float
    electron_count_error: float
    wall_time_seconds: float
    events: tuple[str, ...] = ()
    schema: str = GAUSSIAN_SCF_ARRAY_SCHEMA


@dataclass(frozen=True)
class GaussianFAFPointResult:
    """Full-precision point data co-located with binary SCF matrices."""

    atomic_numbers: np.ndarray
    cartesian_coordinates_bohr: np.ndarray
    cartesian_gradient_hartree_per_bohr: np.ndarray
    total_energy_hartree: float


@dataclass(frozen=True)
class _FortranRecordFormat:
    endian: str
    marker_bytes: int

    @property
    def marker_code(self) -> str:
        return "I" if self.marker_bytes == 4 else "Q"


@dataclass(frozen=True)
class _ArrayLocation:
    array: GaussianBinaryArray
    data_offsets: tuple[tuple[int, int], ...]
    record_slice: tuple[int, int]


def gaussian_checkpoint_tool(
    name: str,
    *,
    gaussian_executable: str | None = None,
    explicit: str | None = None,
) -> str:
    """Resolve a Gaussian checkpoint utility next to the selected executable."""

    if explicit:
        return str(explicit)
    if gaussian_executable:
        sibling = Path(str(gaussian_executable)).expanduser().resolve().with_name(name)
        if sibling.is_file() and sibling.stat().st_mode & 0o111:
            return str(sibling)
    resolved = shutil.which(name)
    if resolved is None:
        raise GaussianSCFCheckpointError(f"Gaussian checkpoint utility not found: {name}")
    return resolved


def export_gaussian_checkpoint_faf(
    checkpoint: Path | str,
    output: Path | str,
    *,
    gaussian_executable: str | None = None,
    formchk_executable: str | None = None,
    timeout: float | None = None,
) -> Path:
    """Export a checkpoint as a native float64 Fortran binary-array file."""

    source = Path(checkpoint).expanduser().resolve()
    target = Path(output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Gaussian checkpoint not found: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    executable = gaussian_checkpoint_tool(
        "formchk", gaussian_executable=gaussian_executable, explicit=formchk_executable
    )
    completed = subprocess.run(
        (executable, "-faf", str(source), str(target)),
        cwd=target.parent,
        timeout=timeout,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0 or not target.is_file():
        raise GaussianSCFCheckpointError(
            "Gaussian FAF export failed: " + completed.stdout.strip()
        )
    # Parse immediately so a truncated or incompatible platform format never
    # reaches the extrapolation path.
    read_gaussian_binary_arrays(target, labels=("ALPHA DENSITY MATRIX",))
    return target


def import_gaussian_checkpoint_faf(
    faf: Path | str,
    checkpoint: Path | str,
    *,
    gaussian_executable: str | None = None,
    unfchk_executable: str | None = None,
    timeout: float | None = None,
) -> Path:
    """Import a native Gaussian FAF into a binary checkpoint."""

    source = Path(faf).expanduser().resolve()
    target = Path(checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Gaussian FAF not found: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    executable = gaussian_checkpoint_tool(
        "unfchk", gaussian_executable=gaussian_executable, explicit=unfchk_executable
    )
    completed = subprocess.run(
        (executable, str(source), str(target)),
        cwd=target.parent,
        timeout=timeout,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0 or not target.is_file():
        raise GaussianSCFCheckpointError(
            "Gaussian FAF import failed: " + completed.stdout.strip()
        )
    return target


def read_gaussian_binary_arrays(
    path: Path | str,
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, GaussianBinaryArray]:
    """Read selected dense float64 arrays from a Gaussian FAF.

    Both 32- and 64-bit Fortran record markers, either byte order, and both
    I4/I8 metadata labels are detected from the file itself.  Sparse/indexed
    objects are intentionally rejected: LINK only transports the dense SCF
    arrays listed above.
    """

    source = Path(path)
    raw = source.read_bytes()
    fmt = _detect_record_format(raw)
    selected = _SCF_LABELS if labels is None else frozenset(str(item).upper() for item in labels)
    locations = _locate_arrays(raw, fmt, selected)
    missing = selected - set(locations)
    # Beta-spin and auxiliary SCF-density arrays are legitimately absent.
    required_missing = missing - {
        "BETA MO COEFFICIENTS",
        "BETA DENSITY MATRIX",
        "BETA SCF DENSITY MATRIX",
        "BETA FOCK MATRIX",
        "ALPHA SCF DENSITY MATRIX",
        # A FAF written directly by Link 9999 contains the converged density,
        # Fock and MO arrays but may omit this derived reporting vector.  The
        # electron and spin counts can be recovered exactly from D and S.
        "SCF ORBITAL OCCUPATIONS",
    }
    if required_missing:
        raise GaussianSCFCheckpointError(
            "Gaussian FAF lacks required arrays: " + ", ".join(sorted(required_missing))
        )
    return {label: location.array for label, location in locations.items()}


def read_gaussian_faf_point_result(path: Path | str) -> GaussianFAFPointResult:
    """Read exact point geometry, energy, and gradient from one FAF export.

    A single ``formchk -faf`` call can therefore serve both LINK's derivative
    transport and the next accepted-point SCF reuse, avoiding a second text
    FCHK conversion of the same Gaussian checkpoint.
    """

    arrays = read_gaussian_binary_arrays(
        path,
        labels=(
            "REAL ZNUCLEAR",
            "OPT COORDINATES",
            "OPT ENERGIES",
            "NUCLEAR GRADIENT",
        ),
    )
    atomic_numbers = np.rint(arrays["REAL ZNUCLEAR"].values).astype(int)
    coordinates = np.asarray(arrays["OPT COORDINATES"].values, dtype=float)
    gradient = np.asarray(arrays["NUCLEAR GRADIENT"].values, dtype=float)
    energies = np.asarray(arrays["OPT ENERGIES"].values, dtype=float)
    expected = 3 * atomic_numbers.size
    if coordinates.size != expected or gradient.size != expected or energies.size < 1:
        raise GaussianSCFCheckpointError("Gaussian FAF point arrays are incomplete")
    if not (
        np.all(np.isfinite(coordinates))
        and np.all(np.isfinite(gradient))
        and np.all(np.isfinite(energies))
    ):
        raise GaussianSCFCheckpointError("Gaussian FAF point arrays are non-finite")
    return GaussianFAFPointResult(
        atomic_numbers=atomic_numbers,
        cartesian_coordinates_bohr=coordinates.reshape((-1, 3)),
        cartesian_gradient_hartree_per_bohr=gradient.reshape(-1),
        total_energy_hartree=float(energies[-1]),
    )


def replace_gaussian_binary_arrays(
    source: Path | str,
    target: Path | str,
    replacements: Mapping[str, np.ndarray],
    *,
    selected_only: bool = False,
) -> Path:
    """Replace dense FAF arrays in a byte-preserving full or update file.

    ``selected_only`` writes the FAF header followed only by the replaced
    array records. Gaussian ``unfchk`` treats such a binary array file as an
    update to an existing checkpoint. This avoids copying and reparsing every
    unrelated checkpoint array when LINK injects a same-basis SCF guess.
    """

    source_path = Path(source)
    target_path = Path(target)
    raw = bytearray(source_path.read_bytes())
    fmt = _detect_record_format(raw)
    normalized = {str(label).upper(): np.asarray(value, dtype=float) for label, value in replacements.items()}
    unknown = set(normalized) - _SCF_LABELS
    if unknown:
        raise GaussianSCFCheckpointError(
            "unsupported Gaussian SCF array replacement: " + ", ".join(sorted(unknown))
        )
    locations = _locate_arrays(raw, fmt, frozenset(normalized))
    missing = set(normalized) - set(locations)
    if missing:
        raise GaussianSCFCheckpointError(
            "Gaussian FAF template lacks arrays: " + ", ".join(sorted(missing))
        )
    float_dtype = np.dtype(fmt.endian + "f8")
    for label, values in normalized.items():
        location = locations[label]
        flat = np.asarray(values, dtype=float).reshape(-1)
        if flat.size != location.array.total_count or not np.all(np.isfinite(flat)):
            raise GaussianSCFCheckpointError(
                f"replacement {label} has {flat.size} values; expected "
                f"{location.array.total_count} finite values"
            )
        cursor = 0
        for offset, count in location.data_offsets:
            chunk = np.asarray(flat[cursor : cursor + count], dtype=float_dtype)
            raw[offset : offset + 8 * count] = chunk.tobytes(order="C")
            cursor += count
        if cursor != flat.size:
            raise GaussianSCFCheckpointError(f"incomplete replacement of {label}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if selected_only:
        records = _records(raw, fmt)
        if not records:
            raise GaussianSCFCheckpointError("Gaussian FAF contains no header record")
        _header_record, header_payload, header_length = records[0]
        integer_bytes = 4 if header_length == 104 else 8
        integer_code = "i" if integer_bytes == 4 else "q"
        _version, header_record_count = struct.unpack_from(
            fmt.endian + integer_code * 2,
            raw,
            header_payload + 64,
        )
        if header_record_count < 1 or header_record_count > len(records):
            raise GaussianSCFCheckpointError(
                "Gaussian FAF declares an invalid header-record count"
            )
        _last_header_record, last_header_payload, last_header_length = records[
            header_record_count - 1
        ]
        header_end = last_header_payload + last_header_length + fmt.marker_bytes
        spans = sorted({locations[label].record_slice for label in normalized})
        output = bytearray(raw[:header_end])
        for start, end in spans:
            output.extend(raw[start:end])
        target_path.write_bytes(output)
    else:
        target_path.write_bytes(raw)
    # A second parse verifies record markers and the exact float64 payload.
    reread = read_gaussian_binary_arrays(target_path, labels=tuple(normalized))
    for label, values in normalized.items():
        if not np.array_equal(reread[label].values, np.asarray(values).reshape(-1)):
            raise GaussianSCFCheckpointError(f"lossless verification failed for {label}")
    return target_path


def packed_symmetric_to_square(values: np.ndarray, dimension: int) -> np.ndarray:
    packed = np.asarray(values, dtype=float).reshape(-1)
    expected = dimension * (dimension + 1) // 2
    if packed.size != expected:
        raise GaussianSCFCheckpointError(
            f"packed Gaussian matrix has {packed.size} values; expected {expected}"
        )
    matrix = np.zeros((dimension, dimension), dtype=float)
    lower = np.tril_indices(dimension)
    matrix[lower] = packed
    matrix[(lower[1], lower[0])] = packed
    return matrix


def square_to_packed_symmetric(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise GaussianSCFCheckpointError("Gaussian SCF matrix must be square")
    if not np.all(np.isfinite(value)):
        raise GaussianSCFCheckpointError("Gaussian SCF matrix contains non-finite values")
    symmetric = 0.5 * (value + value.T)
    return symmetric[np.tril_indices(value.shape[0])]


def gaussian_array_matrix(array: GaussianBinaryArray) -> np.ndarray:
    """Expand a Gaussian dense matrix using its native dimension metadata."""

    n1, n2, n3, n4, n5 = array.dimensions
    del n3, n4, n5
    if n1 < 0 and abs(n1) == abs(n2):
        return packed_symmetric_to_square(array.values, abs(n1))
    rows = abs(int(n1))
    columns = abs(int(n2))
    if rows < 1 or columns < 1 or rows * columns != array.total_count:
        raise GaussianSCFCheckpointError(
            f"Gaussian array {array.label} has unsupported dimensions {array.dimensions}"
        )
    return np.asarray(array.values, dtype=float).reshape((rows, columns), order="F")


def gaussian_matrix_array_values(
    matrix: np.ndarray,
    template: GaussianBinaryArray,
) -> np.ndarray:
    n1, n2, _n3, _n4, _n5 = template.dimensions
    value = np.asarray(matrix, dtype=float)
    if n1 < 0 and abs(n1) == abs(n2):
        return square_to_packed_symmetric(value)
    expected = (abs(int(n1)), abs(int(n2)))
    if value.shape != expected:
        raise GaussianSCFCheckpointError(
            f"Gaussian array {template.label} expects matrix shape {expected}, got {value.shape}"
        )
    return value.reshape(-1, order="F")


def transport_fock_through_projected_orbitals(
    source_fock: np.ndarray,
    source_orbitals: np.ndarray,
    target_orbitals: np.ndarray,
    *,
    rcond: float = 1.0e-11,
) -> tuple[np.ndarray, float]:
    """Transport an AO Fock operator through Gaussian-projected MOs.

    The source operator is first represented in the source MO basis.  The
    minimum-norm target AO operator that yields the same matrix in the
    corresponding projected target MOs is then recovered.  This covariant
    operator transformation remains valid when the active MO matrix is
    rectangular, unlike applying the density congruence to a Fock matrix.
    """

    fock = np.asarray(source_fock, dtype=float)
    source = np.asarray(source_orbitals, dtype=float)
    target = np.asarray(target_orbitals, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise GaussianSCFCheckpointError("source and target MO spaces are incompatible")
    if fock.shape != (source.shape[0], source.shape[0]):
        raise GaussianSCFCheckpointError("source Fock and MO dimensions differ")
    source_mo_fock = source.T @ (0.5 * (fock + fock.T)) @ source
    target_inverse = np.linalg.pinv(target, rcond=rcond)
    transported = target_inverse.T @ source_mo_fock @ target_inverse
    transported = 0.5 * (transported + transported.T)
    residual = float(
        np.linalg.norm(target.T @ transported @ target - source_mo_fock, ord="fro")
        / max(np.linalg.norm(source_mo_fock, ord="fro"), 1.0e-16)
    )
    if not np.all(np.isfinite(transported)) or residual > 1.0e-7:
        raise GaussianSCFCheckpointError(
            f"Gaussian projected Fock residual is too large ({residual:.3e})"
        )
    return transported, residual


def transport_density_through_projected_orbitals(
    source_density: np.ndarray,
    source_orbitals: np.ndarray,
    target_orbitals: np.ndarray,
    *,
    rcond: float = 1.0e-11,
) -> tuple[np.ndarray, float]:
    """Transport an AO density through Gaussian's native MO projection.

    Gaussian supplies the target-basis images of a complete source MO set.
    Their AO map transports a contravariant density, while
    :func:`transport_fock_through_projected_orbitals` applies the dual map to
    an operator.  Keeping the two transformations distinct is required for
    electron-count and expectation-value invariance.
    """

    density = np.asarray(source_density, dtype=float)
    source = np.asarray(source_orbitals, dtype=float)
    target = np.asarray(target_orbitals, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise GaussianSCFCheckpointError("source and target MO spaces are incompatible")
    if density.shape != (source.shape[0], source.shape[0]):
        raise GaussianSCFCheckpointError("source density and MO dimensions differ")
    source_inverse = np.linalg.pinv(source, rcond=rcond)
    ao_map = target @ source_inverse
    transported = ao_map @ (0.5 * (density + density.T)) @ ao_map.T
    transported = 0.5 * (transported + transported.T)
    source_mo_density = source_inverse @ density @ source_inverse.T
    target_inverse = np.linalg.pinv(target, rcond=rcond)
    recovered = target_inverse @ transported @ target_inverse.T
    residual = float(
        np.linalg.norm(recovered - source_mo_density, ord="fro")
        / max(np.linalg.norm(source_mo_density, ord="fro"), 1.0e-16)
    )
    if not np.all(np.isfinite(transported)) or residual > 1.0e-7:
        raise GaussianSCFCheckpointError(
            f"Gaussian projected density residual is too large ({residual:.3e})"
        )
    return transported, residual


def extrapolate_projected_gaussian_states(
    states: Sequence[GaussianProjectedSCFState],
    target_coordinates_angstrom: np.ndarray,
) -> tuple[GaussianProjectedSCFState, str, float]:
    """Linearly extrapolate projected density and Fock matrices together."""

    if not states:
        raise GaussianSCFCheckpointError("Gaussian SCF extrapolation needs a state")
    latest = states[-1]
    if len(states) < 2:
        return latest, "projected", 0.0
    target = np.asarray(target_coordinates_angstrom, dtype=float)
    source_coordinates = [np.asarray(item.source_coordinates_angstrom, dtype=float) for item in states]
    direction = source_coordinates[-1] - source_coordinates[-2]
    step = float(np.linalg.norm(direction))
    if step <= 1.0e-14:
        return latest, "projected_fallback_duplicate_geometry", 0.0
    direction /= step
    origin = source_coordinates[-1]
    parameters = np.asarray([float(np.sum((coords - origin) * direction)) for coords in source_coordinates])
    if np.unique(np.round(parameters, 14)).size < 2:
        return latest, "projected_fallback_degenerate_history", 0.0
    target_parameter = float(np.sum((target - origin) * direction))
    design = np.column_stack((np.ones(len(parameters)), parameters))

    def fit(values: Sequence[np.ndarray | None]) -> np.ndarray | None:
        if any(value is None for value in values):
            return values[-1]
        arrays = [np.asarray(value, dtype=float) for value in values]
        observations = np.asarray(arrays).reshape(len(arrays), -1)
        coefficients = np.linalg.lstsq(design, observations, rcond=None)[0]
        result = (coefficients[0] + target_parameter * coefficients[1]).reshape(arrays[-1].shape)
        return 0.5 * (result + result.T)

    density_alpha = fit([item.density_alpha for item in states])
    density_beta = fit([item.density_beta for item in states])
    fock_alpha = fit([item.fock_alpha for item in states])
    fock_beta = fit([item.fock_beta for item in states])
    assert density_alpha is not None and fock_alpha is not None
    repair = 0.0
    if latest.overlap is not None and latest.electron_count is not None:
        try:
            if density_beta is None:
                density_alpha, repair = _n_representable_density(
                    density_alpha,
                    latest.overlap,
                    target_electrons=0.5 * float(latest.electron_count),
                )
            else:
                alpha_electrons = float(np.trace(latest.density_alpha @ latest.overlap))
                beta_electrons = float(np.trace(latest.density_beta @ latest.overlap))
                density_alpha, alpha_repair = _n_representable_density(
                    density_alpha,
                    latest.overlap,
                    target_electrons=alpha_electrons,
                )
                density_beta, beta_repair = _n_representable_density(
                    density_beta,
                    latest.overlap,
                    target_electrons=beta_electrons,
                )
                repair = max(alpha_repair, beta_repair)
        except GaussianSCFCheckpointError:
            return latest, "projected_fallback_excessive_density_repair", 0.0
    else:
        for density in (density_alpha, density_beta):
            if density is not None and float(np.min(np.linalg.eigvalsh(density))) < -1.0e-8:
                return latest, "projected_fallback_non_psd_extrapolation", 0.0
    electron_error = _electron_count_error(
        density_alpha,
        density_beta,
        latest.overlap,
        latest.electron_count,
    )
    if electron_error > 1.0e-7:
        return latest, "projected_fallback_electron_count", electron_error
    return (
        GaussianProjectedSCFState(
            source_checkpoint=latest.source_checkpoint,
            projected_checkpoint=latest.projected_checkpoint,
            source_coordinates_angstrom=latest.source_coordinates_angstrom,
            target_coordinates_angstrom=latest.target_coordinates_angstrom,
            density_alpha=density_alpha,
            density_beta=density_beta,
            fock_alpha=fock_alpha,
            fock_beta=fock_beta,
            orbitals_alpha=latest.orbitals_alpha,
            orbitals_beta=latest.orbitals_beta,
            occupations_alpha=latest.occupations_alpha,
            occupations_beta=latest.occupations_beta,
            overlap=latest.overlap,
            electron_count=latest.electron_count,
            projection_residual=max(item.projection_residual for item in states),
        ),
        (
            "projected_and_extrapolated_psd_repaired"
            if repair > 1.0e-12
            else "projected_and_extrapolated"
        ),
        electron_error,
    )


def inject_gaussian_projected_state(
    state: GaussianProjectedSCFState,
    *,
    template_faf: Path | str,
    output_faf: Path | str,
    output_checkpoint: Path | str,
    gaussian_executable: str | None = None,
    unfchk_executable: str | None = None,
    timeout: float | None = None,
    require_fock: bool = True,
) -> Path:
    """Inject a joint projected/extrapolated state into a target checkpoint."""

    arrays = read_gaussian_binary_arrays(template_faf)
    replacements: dict[str, np.ndarray] = {}
    matrix_values = (
        ("ALPHA DENSITY MATRIX", state.density_alpha),
        ("ALPHA SCF DENSITY MATRIX", state.density_alpha),
        ("ALPHA FOCK MATRIX", state.fock_alpha),
        ("BETA DENSITY MATRIX", state.density_beta),
        ("BETA SCF DENSITY MATRIX", state.density_beta),
        ("BETA FOCK MATRIX", state.fock_beta),
    )
    for label, matrix in matrix_values:
        template = arrays.get(label)
        if matrix is not None and template is not None:
            replacements[label] = gaussian_matrix_array_values(matrix, template)
    if "ALPHA DENSITY MATRIX" not in replacements:
        raise GaussianSCFCheckpointError("alpha density injection is incomplete")
    if require_fock and "ALPHA FOCK MATRIX" not in replacements:
        raise GaussianSCFCheckpointError("alpha Fock injection is incomplete")
    patched = replace_gaussian_binary_arrays(
        template_faf,
        output_faf,
        replacements,
        selected_only=True,
    )
    source_checkpoint = Path(state.projected_checkpoint).expanduser().resolve()
    if not source_checkpoint.is_file():
        raise FileNotFoundError(
            f"Gaussian checkpoint update template not found: {source_checkpoint}"
        )
    target_checkpoint = Path(output_checkpoint).expanduser().resolve()
    target_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_checkpoint, target_checkpoint)
    return import_gaussian_checkpoint_faf(
        patched,
        target_checkpoint,
        gaussian_executable=gaussian_executable,
        unfchk_executable=unfchk_executable,
        timeout=timeout,
    )


def _electron_count_error(
    density_alpha: np.ndarray,
    density_beta: np.ndarray | None,
    overlap: np.ndarray | None,
    electron_count: float | None,
) -> float:
    if overlap is None or electron_count is None:
        return 0.0
    total = float(np.trace(np.asarray(density_alpha) @ overlap))
    if density_beta is not None:
        total += float(np.trace(np.asarray(density_beta) @ overlap))
    else:
        # Gaussian's FAF stores one spin block for a restricted wavefunction,
        # while SCF ORBITAL OCCUPATIONS and NE count both spins.
        total *= 2.0
    return abs(total - float(electron_count))


def overlap_from_complete_orbitals(orbitals: np.ndarray, *, rcond: float = 1.0e-11) -> np.ndarray | None:
    """Recover the AO overlap from a complete orthonormal MO coefficient set."""

    coefficients = np.asarray(orbitals, dtype=float)
    if coefficients.ndim != 2 or coefficients.shape[0] != coefficients.shape[1]:
        return None
    inverse = np.linalg.pinv(coefficients, rcond=rcond)
    overlap = inverse.T @ inverse
    overlap = 0.5 * (overlap + overlap.T)
    residual = np.linalg.norm(coefficients.T @ overlap @ coefficients - np.eye(coefficients.shape[1]))
    if not np.all(np.isfinite(overlap)) or residual > 1.0e-7:
        return None
    return overlap


def _n_representable_density(
    density: np.ndarray,
    overlap: np.ndarray,
    *,
    target_electrons: float,
) -> tuple[np.ndarray, float]:
    """Project one spin density onto the PSD, trace-preserving AO set."""

    matrix = 0.5 * (np.asarray(density, dtype=float) + np.asarray(density, dtype=float).T)
    metric = 0.5 * (np.asarray(overlap, dtype=float) + np.asarray(overlap, dtype=float).T)
    eigenvalues_s, eigenvectors_s = np.linalg.eigh(metric)
    if float(np.min(eigenvalues_s)) <= 1.0e-12:
        raise GaussianSCFCheckpointError("target Gaussian AO overlap is not positive definite")
    sqrt_s = eigenvectors_s @ np.diag(np.sqrt(eigenvalues_s)) @ eigenvectors_s.T
    inverse_sqrt_s = eigenvectors_s @ np.diag(1.0 / np.sqrt(eigenvalues_s)) @ eigenvectors_s.T
    orthogonal_density = sqrt_s @ matrix @ sqrt_s
    occupations, vectors = np.linalg.eigh(0.5 * (orthogonal_density + orthogonal_density.T))
    target = float(target_electrons)
    if not 0.0 <= target <= float(occupations.size) + 1.0e-10:
        raise GaussianSCFCheckpointError("target spin electron count is outside the AO space")
    # Euclidean projection of the natural occupations onto the capped simplex
    # {0 <= n_i <= 1, sum n_i = N}.  Bisection is monotonic and deterministic.
    low = float(np.min(occupations) - 1.0)
    high = float(np.max(occupations) + 1.0)
    for _ in range(100):
        shift = 0.5 * (low + high)
        projected = np.clip(occupations - shift, 0.0, 1.0)
        if float(np.sum(projected)) > target:
            low = shift
        else:
            high = shift
    projected = np.clip(occupations - 0.5 * (low + high), 0.0, 1.0)
    trace_error = target - float(np.sum(projected))
    if abs(trace_error) > 1.0e-12:
        free = np.flatnonzero((projected > 1.0e-12) & (projected < 1.0 - 1.0e-12))
        if free.size:
            projected[free] += trace_error / free.size
    repaired_orthogonal = vectors @ np.diag(projected) @ vectors.T
    repaired = inverse_sqrt_s @ repaired_orthogonal @ inverse_sqrt_s
    repaired = 0.5 * (repaired + repaired.T)
    relative = float(
        np.linalg.norm(repaired_orthogonal - orthogonal_density, ord="fro")
        / max(np.linalg.norm(orthogonal_density, ord="fro"), 1.0e-16)
    )
    if relative > 0.25:
        raise GaussianSCFCheckpointError(
            f"extrapolated Gaussian density needs excessive PSD repair ({relative:.3e})"
        )
    return repaired, relative


def _detect_record_format(raw: bytes | bytearray) -> _FortranRecordFormat:
    for endian in ("<", ">"):
        for marker_bytes, code in ((4, "I"), (8, "Q")):
            if len(raw) < 2 * marker_bytes:
                continue
            length = int(struct.unpack_from(endian + code, raw, 0)[0])
            end = marker_bytes + length
            if length not in (104, 144) or end + marker_bytes > len(raw):
                continue
            trailer = int(struct.unpack_from(endian + code, raw, end)[0])
            payload = bytes(raw[marker_bytes:end])
            if trailer == length and payload.startswith(b"Gaussian"):
                return _FortranRecordFormat(endian=endian, marker_bytes=marker_bytes)
    raise GaussianSCFCheckpointError("unsupported or corrupt Gaussian FAF record format")


def _records(
    raw: bytes | bytearray,
    fmt: _FortranRecordFormat,
) -> tuple[tuple[int, int, int], ...]:
    records: list[tuple[int, int, int]] = []
    offset = 0
    marker = fmt.endian + fmt.marker_code
    while offset < len(raw):
        if offset + 2 * fmt.marker_bytes > len(raw):
            raise GaussianSCFCheckpointError("truncated Gaussian FAF record marker")
        length = int(struct.unpack_from(marker, raw, offset)[0])
        payload_offset = offset + fmt.marker_bytes
        end = payload_offset + length
        if length < 0 or end + fmt.marker_bytes > len(raw):
            raise GaussianSCFCheckpointError("invalid Gaussian FAF record length")
        trailer = int(struct.unpack_from(marker, raw, end)[0])
        if trailer != length:
            raise GaussianSCFCheckpointError("inconsistent Gaussian FAF record markers")
        records.append((offset, payload_offset, length))
        offset = end + fmt.marker_bytes
    return tuple(records)


def _locate_arrays(
    raw: bytes | bytearray,
    fmt: _FortranRecordFormat,
    selected: frozenset[str],
) -> dict[str, _ArrayLocation]:
    records = _records(raw, fmt)
    locations: dict[str, _ArrayLocation] = {}
    for index, (_record_offset, payload_offset, length) in enumerate(records):
        if length not in (104, 144):
            continue
        payload = bytes(raw[payload_offset : payload_offset + length])
        try:
            label = payload[:64].decode("ascii").rstrip().upper()
        except UnicodeDecodeError:
            continue
        if label not in selected:
            continue
        integer_bytes = 4 if length == 104 else 8
        code = "i" if integer_bytes == 4 else "q"
        metadata = struct.unpack(fmt.endian + code * 10, payload[64:])
        ni, nr, total_count, buffer_length, n1, n2, n3, n4, n5, value_type = (
            int(item) for item in metadata
        )
        if ni != 0 or nr != 1 or total_count < 1 or buffer_length < 1:
            raise GaussianSCFCheckpointError(
                f"Gaussian FAF array {label} is not a dense real scalar array"
            )
        data_offsets: list[tuple[int, int]] = []
        remaining = total_count
        next_index = index + 1
        while remaining > 0:
            if next_index >= len(records):
                raise GaussianSCFCheckpointError(f"truncated Gaussian FAF array {label}")
            _next_record, data_offset, data_length = records[next_index]
            expected = buffer_length * 8
            if data_length < expected:
                raise GaussianSCFCheckpointError(
                    f"Gaussian FAF array {label} has a short data buffer"
                )
            count = min(buffer_length, remaining)
            data_offsets.append((data_offset, count))
            remaining -= count
            next_index += 1
        dtype = np.dtype(fmt.endian + "f8")
        values = np.concatenate(
            [np.frombuffer(raw, dtype=dtype, count=count, offset=offset).astype(float) for offset, count in data_offsets]
        )
        locations[label] = _ArrayLocation(
            GaussianBinaryArray(
                label=label,
                values=values,
                dimensions=(n1, n2, n3, n4, n5),
                integer_count=ni,
                real_count=nr,
                total_count=total_count,
                buffer_length=buffer_length,
                value_type=value_type,
            ),
            tuple(data_offsets),
            (
                records[index][0],
                records[next_index][0] if next_index < len(records) else len(raw),
            ),
        )
    return locations


__all__ = [
    "GAUSSIAN_SCF_ARRAY_SCHEMA",
    "GaussianBinaryArray",
    "GaussianProjectedSCFState",
    "GaussianFAFPointResult",
    "GaussianSCFCheckpointError",
    "GaussianSCFGuessPreparation",
    "export_gaussian_checkpoint_faf",
    "gaussian_array_matrix",
    "gaussian_checkpoint_tool",
    "gaussian_matrix_array_values",
    "import_gaussian_checkpoint_faf",
    "inject_gaussian_projected_state",
    "overlap_from_complete_orbitals",
    "packed_symmetric_to_square",
    "read_gaussian_binary_arrays",
    "read_gaussian_faf_point_result",
    "replace_gaussian_binary_arrays",
    "square_to_packed_symmetric",
    "transport_fock_through_projected_orbitals",
    "transport_density_through_projected_orbitals",
    "extrapolate_projected_gaussian_states",
]
