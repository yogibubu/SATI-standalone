"""Field-free fine and hyperfine Hamiltonians for the LOCK rigid rotor.

The production representation couples ``N + S = J``, sequentially couples all
nuclear spins to ``I``, and finally couples ``J + I = F``.  Quantum numbers
are stored as doubled integers so half-integer triangle rules remain exact.

The electric-quadrupole matrix elements use a direct contraction over one
space-fixed projection.  This is slower than a hand-written chain of 6-j/9-j
symbols while the cache is populated, but it is much less vulnerable to phase
errors and supports any number of nuclear spins and repeated coupling paths.
The resulting scalar factors are cached and reused for every K-space matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from matrix_core import read_sectioned_lines, replace_section, section_content
from matrix_numerics import diagonalize_hermitian, resolve_native_backend

from .native_rotor import (
    MHZ_TO_KELVIN,
    NativeRotorParameters,
    diagonalize_rotor,
    dipole_transition_matrix,
    normalize_representation,
    rotational_hamiltonian,
)

try:
    from . import _lock_native
except ImportError:  # pragma: no cover - source trees can run before extension builds
    _lock_native = None


SPIN_HYPERFINE_SCHEMA = "matrix.lock.spin_hyperfine.v1"
SPIN_HYPERFINE_RESULT_SCHEMA = "matrix.lock.spin_hyperfine_result.v1"
_TENSOR_ZERO = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class NuclearSpin:
    """One isotope-resolved nuclear spin and its Cartesian quadrupole tensor.

    Tensor order is ``aa, bb, cc, ab, ac, bc`` in MHz.  Its isotropic trace is
    removed before conversion to rank-2 spherical components.
    """

    label: str
    two_I: int
    quadrupole_tensor_MHz: tuple[float, float, float, float, float, float] = _TENSOR_ZERO
    fermi_contact_MHz: float = 0.0
    isotropic_nuclear_spin_rotation_MHz: float = 0.0
    electron_nuclear_dipolar_tensor_MHz: tuple[
        float, float, float, float, float, float
    ] = _TENSOR_ZERO

    def __post_init__(self) -> None:
        label = self.label.strip()
        if not label:
            raise ValueError("a nuclear-spin label is required")
        if isinstance(self.two_I, bool) or int(self.two_I) != self.two_I or self.two_I < 0:
            raise ValueError("two_I must be a nonnegative integer")
        if len(self.quadrupole_tensor_MHz) != 6:
            raise ValueError("quadrupole_tensor_MHz must contain aa, bb, cc, ab, ac, bc")
        tensor = np.asarray(self.quadrupole_tensor_MHz, dtype=float)
        dipolar = np.asarray(self.electron_nuclear_dipolar_tensor_MHz, dtype=float)
        if dipolar.shape != (6,):
            raise ValueError(
                "electron_nuclear_dipolar_tensor_MHz must contain aa, bb, cc, ab, ac, bc"
            )
        if not np.all(np.isfinite(tensor)) or not np.all(np.isfinite(dipolar)):
            raise ValueError("nuclear spin tensor components must be finite")
        if not math.isfinite(float(self.fermi_contact_MHz)):
            raise ValueError("the Fermi-contact constant must be finite")
        if not math.isfinite(float(self.isotropic_nuclear_spin_rotation_MHz)):
            raise ValueError("the nuclear spin-rotation constant must be finite")
        if np.any(tensor != 0.0) and self.two_I < 2:
            raise ValueError("electric quadrupole coupling requires nuclear spin I >= 1")
        if self.two_I == 0 and (
            self.fermi_contact_MHz != 0.0
            or self.isotropic_nuclear_spin_rotation_MHz != 0.0
            or np.any(dipolar != 0.0)
        ):
            raise ValueError("magnetic hyperfine coupling requires nuclear spin I > 0")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "two_I", int(self.two_I))
        object.__setattr__(
            self,
            "quadrupole_tensor_MHz",
            tuple(float(value) for value in self.quadrupole_tensor_MHz),
        )
        object.__setattr__(self, "fermi_contact_MHz", float(self.fermi_contact_MHz))
        object.__setattr__(
            self,
            "electron_nuclear_dipolar_tensor_MHz",
            tuple(float(value) for value in self.electron_nuclear_dipolar_tensor_MHz),
        )
        object.__setattr__(
            self,
            "isotropic_nuclear_spin_rotation_MHz",
            float(self.isotropic_nuclear_spin_rotation_MHz),
        )


@dataclass(frozen=True)
class ElectronSpin:
    """Electron spin for an isolated case-(b) state.

    The isotropic spin-rotation eigenvalue and the anisotropic spin-rotation
    and zero-field-splitting tensors are evaluated in the coupled
    ``|N,S;J>`` basis.
    """

    two_S: int = 0
    isotropic_spin_rotation_MHz: float = 0.0
    anisotropic_spin_rotation_tensor_MHz: tuple[
        float, float, float, float, float, float
    ] = _TENSOR_ZERO
    zero_field_splitting_tensor_MHz: tuple[
        float, float, float, float, float, float
    ] = _TENSOR_ZERO
    spin_lam_momentum_vectors_MHz: tuple[tuple[float, float, float], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.two_S, bool) or int(self.two_S) != self.two_S or self.two_S < 0:
            raise ValueError("two_S must be a nonnegative integer")
        if not math.isfinite(float(self.isotropic_spin_rotation_MHz)):
            raise ValueError("the electron spin-rotation constant must be finite")
        anisotropic = np.asarray(self.anisotropic_spin_rotation_tensor_MHz, dtype=float)
        zfs = np.asarray(self.zero_field_splitting_tensor_MHz, dtype=float)
        spin_lam = np.asarray(self.spin_lam_momentum_vectors_MHz, dtype=float)
        if anisotropic.shape != (6,) or zfs.shape != (6,):
            raise ValueError("electron tensors must contain aa, bb, cc, ab, ac, bc")
        if not np.all(np.isfinite(anisotropic)) or not np.all(np.isfinite(zfs)):
            raise ValueError("electron tensor components must be finite")
        if spin_lam.size and (spin_lam.ndim != 2 or spin_lam.shape[1] != 3):
            raise ValueError(
                "spin_lam_momentum_vectors_MHz must contain one a,b,c vector per LAM"
            )
        if not np.all(np.isfinite(spin_lam)):
            raise ValueError("spin-LAM momentum coupling components must be finite")
        if self.two_S == 0 and (
            self.isotropic_spin_rotation_MHz != 0.0
            or np.any(anisotropic != 0.0)
            or np.any(spin_lam != 0.0)
        ):
            raise ValueError("electron spin rotation requires S > 0")
        if np.any(zfs != 0.0) and self.two_S < 2:
            raise ValueError("zero-field splitting requires electron spin S >= 1")
        object.__setattr__(self, "two_S", int(self.two_S))
        object.__setattr__(
            self, "isotropic_spin_rotation_MHz", float(self.isotropic_spin_rotation_MHz)
        )
        object.__setattr__(
            self,
            "anisotropic_spin_rotation_tensor_MHz",
            tuple(float(value) for value in self.anisotropic_spin_rotation_tensor_MHz),
        )
        object.__setattr__(
            self,
            "zero_field_splitting_tensor_MHz",
            tuple(float(value) for value in self.zero_field_splitting_tensor_MHz),
        )
        object.__setattr__(
            self,
            "spin_lam_momentum_vectors_MHz",
            tuple(tuple(float(value) for value in row) for row in spin_lam),
        )


@dataclass(frozen=True)
class NuclearSpinSpinCoupling:
    """Pairwise nuclear spin-spin coupling in MHz.

    ``isotropic_MHz`` multiplies ``I_left . I_right``.  The Cartesian tensor
    is the symmetric traceless dipolar tensor in ``aa,bb,cc,ab,ac,bc`` order.
    """

    left: str
    right: str
    isotropic_MHz: float = 0.0
    dipolar_tensor_MHz: tuple[float, float, float, float, float, float] = _TENSOR_ZERO

    def __post_init__(self) -> None:
        left, right = self.left.strip(), self.right.strip()
        tensor = np.asarray(self.dipolar_tensor_MHz, dtype=float)
        if not left or not right or left.casefold() == right.casefold():
            raise ValueError("a nuclear spin-spin pair requires two distinct labels")
        if not math.isfinite(float(self.isotropic_MHz)):
            raise ValueError("the isotropic nuclear spin-spin constant must be finite")
        if tensor.shape != (6,) or not np.all(np.isfinite(tensor)):
            raise ValueError("dipolar_tensor_MHz must contain six finite components")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "isotropic_MHz", float(self.isotropic_MHz))
        object.__setattr__(self, "dipolar_tensor_MHz", tuple(float(x) for x in tensor))


@dataclass(frozen=True)
class SpinHyperfineModel:
    nuclei: tuple[NuclearSpin, ...] = ()
    electron: ElectronSpin = ElectronSpin()
    nuclear_spin_spin: tuple[NuclearSpinSpinCoupling, ...] = ()
    schema: str = SPIN_HYPERFINE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SPIN_HYPERFINE_SCHEMA:
            raise ValueError(f"unsupported spin/hyperfine schema: {self.schema}")
        if not isinstance(self.electron, ElectronSpin):
            raise TypeError("electron must be an ElectronSpin")
        nuclei = tuple(self.nuclei)
        if any(not isinstance(nucleus, NuclearSpin) for nucleus in nuclei):
            raise TypeError("nuclei must contain NuclearSpin values")
        labels = tuple(nucleus.label.casefold() for nucleus in nuclei)
        if len(labels) != len(set(labels)):
            raise ValueError("nuclear-spin labels must be unique")
        if self.electron.two_S == 0 and any(
            nucleus.fermi_contact_MHz != 0.0
            or any(value != 0.0 for value in nucleus.electron_nuclear_dipolar_tensor_MHz)
            for nucleus in nuclei
        ):
            raise ValueError("electron-nuclear coupling requires electron spin S > 0")
        pairs = tuple(self.nuclear_spin_spin)
        if any(not isinstance(pair, NuclearSpinSpinCoupling) for pair in pairs):
            raise TypeError("nuclear_spin_spin must contain NuclearSpinSpinCoupling values")
        label_indices = {label: index for index, label in enumerate(labels)}
        canonical_pairs: set[tuple[int, int]] = set()
        for pair in pairs:
            try:
                indices = tuple(sorted((label_indices[pair.left.casefold()], label_indices[pair.right.casefold()])))
            except KeyError as exc:
                raise ValueError(f"unknown nucleus in spin-spin pair: {exc.args[0]}") from exc
            if indices in canonical_pairs:
                raise ValueError("each nuclear spin-spin pair may be specified only once")
            if (
                nuclei[indices[0]].two_I == 0 or nuclei[indices[1]].two_I == 0
            ) and (
                pair.isotropic_MHz != 0.0
                or any(value != 0.0 for value in pair.dipolar_tensor_MHz)
            ):
                raise ValueError("nuclear spin-spin coupling requires both I > 0")
            canonical_pairs.add(indices)
        object.__setattr__(self, "nuclei", nuclei)
        object.__setattr__(self, "nuclear_spin_spin", pairs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "tensor_frame": "body-fixed a,b,c",
            "coupling_tree": "N+S=J; sequential-nuclear-spins=I; J+I=F",
            "electron": {
                "two_S": self.electron.two_S,
                "isotropic_spin_rotation_MHz": self.electron.isotropic_spin_rotation_MHz,
                "anisotropic_spin_rotation_tensor_MHz": list(
                    self.electron.anisotropic_spin_rotation_tensor_MHz
                ),
                "zero_field_splitting_tensor_MHz": list(
                    self.electron.zero_field_splitting_tensor_MHz
                ),
                "spin_lam_momentum_vectors_MHz": [
                    list(row) for row in self.electron.spin_lam_momentum_vectors_MHz
                ],
            },
            "nuclei": [
                {
                    "label": nucleus.label,
                    "two_I": nucleus.two_I,
                    "quadrupole_tensor_MHz": list(nucleus.quadrupole_tensor_MHz),
                    "fermi_contact_MHz": nucleus.fermi_contact_MHz,
                    "isotropic_nuclear_spin_rotation_MHz": (
                        nucleus.isotropic_nuclear_spin_rotation_MHz
                    ),
                    "electron_nuclear_dipolar_tensor_MHz": list(
                        nucleus.electron_nuclear_dipolar_tensor_MHz
                    ),
                }
                for nucleus in self.nuclei
            ],
            "nuclear_spin_spin": [
                {
                    "left": pair.left,
                    "right": pair.right,
                    "isotropic_MHz": pair.isotropic_MHz,
                    "dipolar_tensor_MHz": list(pair.dipolar_tensor_MHz),
                }
                for pair in self.nuclear_spin_spin
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SpinHyperfineModel:
        if payload.get("schema") != SPIN_HYPERFINE_SCHEMA:
            raise ValueError(f"unsupported spin/hyperfine schema: {payload.get('schema')!r}")
        frame = payload.get("tensor_frame", "body-fixed a,b,c")
        if frame != "body-fixed a,b,c":
            raise ValueError("spin/hyperfine tensors must use the body-fixed a,b,c frame")
        electron_payload = payload.get("electron", {})
        if not isinstance(electron_payload, Mapping):
            raise ValueError("electron must be an object")
        nuclei_payload = payload.get("nuclei", [])
        if not isinstance(nuclei_payload, list):
            raise ValueError("nuclei must be an array")
        pair_payload = payload.get("nuclear_spin_spin", [])
        if not isinstance(pair_payload, list) or any(
            not isinstance(item, Mapping) for item in pair_payload
        ):
            raise ValueError("nuclear_spin_spin must be an array of objects")
        nuclei: list[NuclearSpin] = []
        for item in nuclei_payload:
            if not isinstance(item, Mapping):
                raise ValueError("each nuclear spin must be an object")
            tensor = tuple(float(value) for value in item.get("quadrupole_tensor_MHz", _TENSOR_ZERO))
            nuclei.append(
                NuclearSpin(
                    label=str(item.get("label", "")),
                    two_I=item.get("two_I", -1),  # type: ignore[arg-type]
                    quadrupole_tensor_MHz=tensor,  # type: ignore[arg-type]
                    fermi_contact_MHz=float(item.get("fermi_contact_MHz", 0.0)),
                    isotropic_nuclear_spin_rotation_MHz=float(
                        item.get("isotropic_nuclear_spin_rotation_MHz", 0.0)
                    ),
                    electron_nuclear_dipolar_tensor_MHz=tuple(
                        float(value)
                        for value in item.get(
                            "electron_nuclear_dipolar_tensor_MHz", _TENSOR_ZERO
                        )
                    ),  # type: ignore[arg-type]
                )
            )
        return cls(
            nuclei=tuple(nuclei),
            electron=ElectronSpin(
                two_S=electron_payload.get("two_S", 0),  # type: ignore[arg-type]
                isotropic_spin_rotation_MHz=float(
                    electron_payload.get("isotropic_spin_rotation_MHz", 0.0)
                ),
                anisotropic_spin_rotation_tensor_MHz=tuple(
                    float(value)
                    for value in electron_payload.get(
                        "anisotropic_spin_rotation_tensor_MHz", _TENSOR_ZERO
                    )
                ),  # type: ignore[arg-type]
                zero_field_splitting_tensor_MHz=tuple(
                    float(value)
                    for value in electron_payload.get(
                        "zero_field_splitting_tensor_MHz", _TENSOR_ZERO
                    )
                ),  # type: ignore[arg-type]
                spin_lam_momentum_vectors_MHz=tuple(
                    tuple(float(value) for value in row)
                    for row in electron_payload.get(
                        "spin_lam_momentum_vectors_MHz", ()
                    )
                ),
            ),
            nuclear_spin_spin=tuple(
                NuclearSpinSpinCoupling(
                    left=str(item.get("left", "")),
                    right=str(item.get("right", "")),
                    isotropic_MHz=float(item.get("isotropic_MHz", 0.0)),
                    dipolar_tensor_MHz=tuple(
                        float(value)
                        for value in item.get("dipolar_tensor_MHz", _TENSOR_ZERO)
                    ),  # type: ignore[arg-type]
                )
                for item in pair_payload
            ),
        )


@dataclass(frozen=True)
class SpinHyperfineBasisState:
    N: int
    k_index: int
    two_J: int
    nuclear_path: tuple[int, ...]
    two_F: int

    @property
    def two_I(self) -> int:
        return self.nuclear_path[-1] if self.nuclear_path else 0


@dataclass(frozen=True)
class SpinHyperfineLevel:
    energy_MHz: float
    two_F: int
    N: int
    two_J: int
    nuclear_path: tuple[int, ...]
    Ka: int
    Kc: int
    parent_purity: float
    lam_parent: int | None = None
    torsional_state: int | None = None
    lam_symmetry: str | None = None
    spin_weight: float = 1.0


@dataclass(frozen=True)
class SpinHyperfineBlock:
    two_F: int
    energies_MHz: np.ndarray
    vectors: np.ndarray
    basis: tuple[SpinHyperfineBasisState, ...]
    levels: tuple[SpinHyperfineLevel, ...]
    hermiticity_error_MHz: float
    maximum_scaled_residual: float
    orthogonality_error: float
    assembly_backend: str
    diagonalizer_backend: str


@dataclass(frozen=True)
class SpinHyperfineResult:
    model: SpinHyperfineModel
    n_values: tuple[int, ...]
    blocks: tuple[SpinHyperfineBlock, ...]
    representation: str = "Ir"

    @property
    def maximum_scaled_residual(self) -> float:
        return max((block.maximum_scaled_residual for block in self.blocks), default=0.0)

    @property
    def maximum_hermiticity_error_MHz(self) -> float:
        return max((block.hermiticity_error_MHz for block in self.blocks), default=0.0)


@dataclass(frozen=True)
class SpinHyperfineTransition:
    frequency_MHz: float
    line_strength_D2: float
    intensity: float
    relative_intensity: float
    lower_two_F: int
    upper_two_F: int
    lower_level: int
    upper_level: int
    lower_energy_MHz: float
    upper_energy_MHz: float
    lower_N: int
    upper_N: int
    lower_two_J: int
    upper_two_J: int
    lower_nuclear_path: tuple[int, ...]
    upper_nuclear_path: tuple[int, ...]
    lower_Ka: int
    upper_Ka: int
    lower_Kc: int
    upper_Kc: int
    lower_lam_parent: int | None = None
    upper_lam_parent: int | None = None


def read_spin_hyperfine_model(path: Path | str) -> SpinHyperfineModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("a spin/hyperfine model must be a JSON object")
    return SpinHyperfineModel.from_dict(payload)


def write_spin_hyperfine_model(path: Path | str, model: SpinHyperfineModel) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target


def spin_hyperfine_section_lines(model: SpinHyperfineModel) -> list[str]:
    compact = json.dumps(model.to_dict(), separators=(",", ":"), ensure_ascii=False)
    return [f"SCHEMA = {model.schema}", f"MODEL_JSON = {compact}"]


def parse_spin_hyperfine_section(lines: Sequence[str]) -> SpinHyperfineModel:
    values: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(("#", "!")) or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().upper()] = value.strip()
    if values.get("SCHEMA") != SPIN_HYPERFINE_SCHEMA:
        raise ValueError(f"unsupported #SPIN_HYPERFINE schema: {values.get('SCHEMA')!r}")
    encoded = values.get("MODEL_JSON")
    if encoded is None:
        raise ValueError("#SPIN_HYPERFINE requires MODEL_JSON")
    payload = json.loads(encoded)
    if not isinstance(payload, Mapping):
        raise ValueError("#SPIN_HYPERFINE MODEL_JSON must contain an object")
    return SpinHyperfineModel.from_dict(payload)


def read_spin_hyperfine_section(path: Path | str) -> SpinHyperfineModel:
    content = section_content(read_sectioned_lines(Path(path)), "SPIN_HYPERFINE")
    if not content:
        raise ValueError(f"missing #SPIN_HYPERFINE section: {path}")
    return parse_spin_hyperfine_section(content)


def write_spin_hyperfine_section(path: Path | str, model: SpinHyperfineModel) -> Path:
    target = Path(path)
    replace_section(target, "SPIN_HYPERFINE", spin_hyperfine_section_lines(model))
    return target


def write_spin_hyperfine_result(
    path: Path | str,
    result: SpinHyperfineResult,
    *,
    model_path: Path | str | None = None,
    rotational_path: Path | str | None = None,
    lam_surface_path: Path | str | None = None,
    transitions: Sequence[SpinHyperfineTransition] = (),
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SPIN_HYPERFINE_RESULT_SCHEMA,
        "model_schema": result.model.schema,
        "model_path": None if model_path is None else str(model_path),
        "rotational_path": None if rotational_path is None else str(rotational_path),
        "lam_surface_path": None if lam_surface_path is None else str(lam_surface_path),
        "representation": result.representation,
        "n_values": list(result.n_values),
        "maximum_hermiticity_error_MHz": result.maximum_hermiticity_error_MHz,
        "maximum_scaled_residual": result.maximum_scaled_residual,
        "blocks": [
            {
                "two_F": block.two_F,
                "dimension": len(block.basis),
                "assembly_backend": block.assembly_backend,
                "diagonalizer_backend": block.diagonalizer_backend,
                "hermiticity_error_MHz": block.hermiticity_error_MHz,
                "maximum_scaled_residual": block.maximum_scaled_residual,
                "orthogonality_error": block.orthogonality_error,
                "levels": [
                    {
                        "energy_MHz": level.energy_MHz,
                        "two_F": level.two_F,
                        "N": level.N,
                        "two_J": level.two_J,
                        "nuclear_path": list(level.nuclear_path),
                        "Ka": level.Ka,
                        "Kc": level.Kc,
                        "parent_purity": level.parent_purity,
                        "lam_parent": level.lam_parent,
                        "torsional_state": level.torsional_state,
                        "lam_symmetry": level.lam_symmetry,
                        "spin_weight": level.spin_weight,
                    }
                    for level in block.levels
                ],
            }
            for block in result.blocks
        ],
        "transitions": [
            {
                "frequency_MHz": line.frequency_MHz,
                "line_strength_D2": line.line_strength_D2,
                "intensity": line.intensity,
                "relative_intensity": line.relative_intensity,
                "lower_two_F": line.lower_two_F,
                "upper_two_F": line.upper_two_F,
                "lower_level": line.lower_level,
                "upper_level": line.upper_level,
                "lower_energy_MHz": line.lower_energy_MHz,
                "upper_energy_MHz": line.upper_energy_MHz,
                "lower_N": line.lower_N,
                "upper_N": line.upper_N,
                "lower_two_J": line.lower_two_J,
                "upper_two_J": line.upper_two_J,
                "lower_nuclear_path": list(line.lower_nuclear_path),
                "upper_nuclear_path": list(line.upper_nuclear_path),
                "lower_Ka": line.lower_Ka,
                "upper_Ka": line.upper_Ka,
                "lower_Kc": line.lower_Kc,
                "upper_Kc": line.upper_Kc,
                "lower_lam_parent": line.lower_lam_parent,
                "upper_lam_parent": line.upper_lam_parent,
            }
            for line in transitions
        ],
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


@dataclass(frozen=True)
class _BasisGroup:
    N: int
    two_J: int
    nuclear_path: tuple[int, ...]
    start: int
    stop: int

    @property
    def two_I(self) -> int:
        return self.nuclear_path[-1] if self.nuclear_path else 0


def coupled_nuclear_spin_paths(two_spins: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Return every sequential coupling path for the supplied nuclear spins."""
    spins = _validate_two_spins(two_spins)
    return _coupled_nuclear_spin_paths(spins)


@lru_cache(maxsize=256)
def _coupled_nuclear_spin_paths(two_spins: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    if not two_spins:
        return ((),)
    paths: tuple[tuple[int, ...], ...] = ((two_spins[0],),)
    for two_spin in two_spins[1:]:
        expanded: list[tuple[int, ...]] = []
        for path in paths:
            for total in range(abs(path[-1] - two_spin), path[-1] + two_spin + 1, 2):
                expanded.append((*path, total))
        paths = tuple(expanded)
    return paths


def nuclear_coupling_transform(
    two_spins: Sequence[int],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[tuple[int, ...], int], ...], np.ndarray]:
    """Return product states, coupled states and their real unitary transform."""
    return _nuclear_coupling_transform(_validate_two_spins(two_spins))


@lru_cache(maxsize=128)
def _nuclear_coupling_transform(
    two_spins: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[tuple[int, ...], int], ...], np.ndarray]:
    if not two_spins:
        transform = np.ones((1, 1), dtype=float)
        transform.setflags(write=False)
        return ((),), (((), 0),), transform

    product_states = tuple(
        itertools.product(*(range(-spin, spin + 1, 2) for spin in two_spins))
    )
    coupled_states = tuple(
        (path, two_M)
        for path in _coupled_nuclear_spin_paths(two_spins)
        for two_M in range(-path[-1], path[-1] + 1, 2)
    )
    transform = np.zeros((len(product_states), len(coupled_states)), dtype=float)
    for row, projections in enumerate(product_states):
        for column, (path, target_M) in enumerate(coupled_states):
            if sum(projections) != target_M:
                continue
            coefficient = 1.0
            previous_M = projections[0]
            for index in range(1, len(two_spins)):
                new_M = previous_M + projections[index]
                coefficient *= clebsch_gordan_doubled(
                    path[index - 1],
                    previous_M,
                    two_spins[index],
                    projections[index],
                    path[index],
                    new_M,
                )
                if coefficient == 0.0:
                    break
                previous_M = new_M
            transform[row, column] = coefficient
    # The Clebsch-Gordan products are analytically normalized.  Their direct
    # floating-point evaluation can nevertheless leave column norms a few ulps
    # away from one, with the exact drift depending on libm and architecture.
    # Normalize each coupled state explicitly so the public transform remains
    # reproducibly unitary on every supported runner.
    column_norms = np.sqrt(np.einsum("ij,ij->j", transform, transform))
    if not np.all(np.isfinite(column_norms)) or np.any(column_norms == 0.0):
        raise RuntimeError("nuclear-spin coupling transform contains an invalid column")
    transform /= column_norms
    gram = np.einsum("ki,kj->ij", transform, transform)
    error = float(np.max(np.abs(gram - np.eye(len(coupled_states)))))
    if error > 2.0e-12:
        raise RuntimeError(f"nuclear-spin coupling transform is not unitary: {error:.3e}")
    transform.setflags(write=False)
    return product_states, coupled_states, transform


def solve_spin_hyperfine_rotor(
    parameters: NativeRotorParameters,
    model: SpinHyperfineModel,
    *,
    n_values: Sequence[int],
    f_max: float | None = None,
    levels_per_f: int | None = None,
    assembly_backend: str | None = None,
    diagonalizer_backend: str | None = None,
    workers: int = 1,
) -> SpinHyperfineResult:
    """Solve field-free fine/hyperfine blocks for a native LOCK rotor."""
    raw_values = tuple(n_values)
    if not raw_values or any(
        isinstance(value, bool) or int(value) != value or value < 0 for value in raw_values
    ):
        raise ValueError("n_values must contain nonnegative integers")
    values = tuple(sorted(set(int(value) for value in raw_values)))
    if f_max is not None and (not math.isfinite(float(f_max)) or f_max < 0.0):
        raise ValueError("f_max must be finite and nonnegative")
    if levels_per_f is not None and (
        isinstance(levels_per_f, bool)
        or int(levels_per_f) != levels_per_f
        or levels_per_f <= 0
    ):
        raise ValueError("levels_per_f must be a positive integer")
    if workers < 1:
        raise ValueError("workers must be at least one")

    two_spins = tuple(nucleus.two_I for nucleus in model.nuclei)
    paths = _coupled_nuclear_spin_paths(two_spins)
    groups_by_f: dict[int, list[_BasisGroup]] = {}
    basis_by_f: dict[int, list[SpinHyperfineBasisState]] = {}
    for N in values:
        for two_J in range(abs(2 * N - model.electron.two_S), 2 * N + model.electron.two_S + 1, 2):
            for path in paths:
                two_I = path[-1] if path else 0
                for two_F in range(abs(two_J - two_I), two_J + two_I + 1, 2):
                    if f_max is not None and two_F > int(math.floor(2.0 * f_max + 1.0e-12)):
                        continue
                    basis = basis_by_f.setdefault(two_F, [])
                    start = len(basis)
                    basis.extend(
                        SpinHyperfineBasisState(N, k, two_J, path, two_F)
                        for k in range(2 * N + 1)
                    )
                    groups_by_f.setdefault(two_F, []).append(
                        _BasisGroup(N, two_J, path, start, len(basis))
                    )

    arguments = [
        (
            parameters,
            model,
            two_spins,
            two_F,
            tuple(basis_by_f[two_F]),
            tuple(groups_by_f[two_F]),
            levels_per_f,
            assembly_backend,
            diagonalizer_backend,
        )
        for two_F in sorted(basis_by_f)
    ]
    if workers == 1 or len(arguments) < 2:
        blocks = tuple(_solve_f_block(*item) for item in arguments)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lock-f-block") as pool:
            blocks = tuple(pool.map(lambda item: _solve_f_block(*item), arguments))
    return SpinHyperfineResult(
        model=model,
        n_values=values,
        blocks=blocks,
        representation=parameters.representation,
    )


def spin_hyperfine_transitions(
    result: SpinHyperfineResult,
    dipole_debye: Sequence[float],
    *,
    temperature_K: float | None = None,
    intensity_cut: float = 0.0,
    _transition_blocks: Mapping[tuple[int, int], np.ndarray] | None = None,
    _reduced_matrices: Mapping[tuple[int, int], np.ndarray] | None = None,
) -> tuple[SpinHyperfineTransition, ...]:
    """Return E1 transitions including interference from mixed hyperfine states."""
    dipole = np.asarray(tuple(dipole_debye), dtype=float)
    if dipole.shape != (3,) or not np.all(np.isfinite(dipole)):
        raise ValueError("dipole_debye must contain three finite a,b,c components")
    if temperature_K is not None and temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    if intensity_cut < 0.0 or not math.isfinite(float(intensity_cut)):
        raise ValueError("intensity_cut must be finite and nonnegative")
    blocks = tuple(result.blocks)
    ground = min((float(block.energies_MHz[0]) for block in blocks), default=0.0)
    beta = None if temperature_K is None else MHZ_TO_KELVIN / temperature_K
    partition = 1.0
    if beta is not None:
        partition = math.fsum(
            (block.two_F + 1.0)
            * level.spin_weight
            * math.exp(-(float(energy) - ground) * beta)
            for block in blocks
            for energy, level in zip(block.energies_MHz, block.levels, strict=True)
        )
    reduced = (
        spin_hyperfine_reduced_dipole_matrices(
            result, dipole, _transition_blocks=_transition_blocks
        )
        if _reduced_matrices is None
        else _reduced_matrices
    )
    raw: list[SpinHyperfineTransition] = []
    for lower_index, lower in enumerate(blocks):
        for upper_index in range(lower_index, len(blocks)):
            upper = blocks[upper_index]
            amplitudes = reduced.get((lower.two_F, upper.two_F))
            if amplitudes is None:
                continue
            for il, energy_l in enumerate(lower.energies_MHz):
                start_iu = il + 1 if lower_index == upper_index else 0
                for iu in range(start_iu, len(upper.energies_MHz)):
                    energy_u = float(upper.energies_MHz[iu])
                    first_energy = float(energy_l)
                    if first_energy <= energy_u:
                        low_energy, up_energy = first_energy, energy_u
                        low_f, up_f = lower.two_F, upper.two_F
                        low_level, up_level = il, iu
                        low_info, up_info = lower.levels[il], upper.levels[iu]
                    else:
                        low_energy, up_energy = energy_u, first_energy
                        low_f, up_f = upper.two_F, lower.two_F
                        low_level, up_level = iu, il
                        low_info, up_info = upper.levels[iu], lower.levels[il]
                    low_weight = low_info.spin_weight
                    frequency = up_energy - low_energy
                    strength = float(abs(amplitudes[il, iu]) ** 2)
                    if frequency <= 1.0e-9 or strength <= 1.0e-18:
                        continue
                    intensity = strength
                    intensity *= low_weight
                    if beta is not None:
                        intensity *= (
                            frequency
                            * (-math.expm1(-frequency * beta))
                            * math.exp(-(low_energy - ground) * beta)
                            / partition
                        )
                    if intensity <= 0.0 or intensity < intensity_cut:
                        continue
                    raw.append(
                        SpinHyperfineTransition(
                            frequency_MHz=frequency,
                            line_strength_D2=strength,
                            intensity=intensity,
                            relative_intensity=0.0,
                            lower_two_F=low_f,
                            upper_two_F=up_f,
                            lower_level=low_level,
                            upper_level=up_level,
                            lower_energy_MHz=low_energy,
                            upper_energy_MHz=up_energy,
                            lower_N=low_info.N,
                            upper_N=up_info.N,
                            lower_two_J=low_info.two_J,
                            upper_two_J=up_info.two_J,
                            lower_nuclear_path=low_info.nuclear_path,
                            upper_nuclear_path=up_info.nuclear_path,
                            lower_Ka=low_info.Ka,
                            upper_Ka=up_info.Ka,
                            lower_Kc=low_info.Kc,
                            upper_Kc=up_info.Kc,
                            lower_lam_parent=low_info.lam_parent,
                            upper_lam_parent=up_info.lam_parent,
                        )
                    )
    maximum = max((line.intensity for line in raw), default=1.0)
    return tuple(
        SpinHyperfineTransition(
            **{
                **line.__dict__,
                "relative_intensity": 100.0 * line.intensity / maximum,
            }
        )
        for line in sorted(raw, key=lambda item: item.frequency_MHz)
    )


def spin_hyperfine_reduced_dipole_matrices(
    result: SpinHyperfineResult,
    dipole_debye: Sequence[float],
    *,
    _transition_blocks: Mapping[tuple[int, int], np.ndarray] | None = None,
    _raw_operators: bool = False,
) -> dict[tuple[int, int], np.ndarray]:
    """Return reduced E1 matrices between resolved field-free F blocks."""
    dipole = np.asarray(tuple(dipole_debye), dtype=float)
    if dipole.shape != (3,) or not np.all(np.isfinite(dipole)):
        raise ValueError("dipole_debye must contain three finite a,b,c components")
    reduced: dict[tuple[int, int], np.ndarray] = {}
    blocks = tuple(result.blocks)
    for lower_index, lower in enumerate(blocks):
        lower_groups = _basis_group_slices(lower.basis)
        for upper_index in range(lower_index, len(blocks)):
            upper = blocks[upper_index]
            if abs(lower.two_F - upper.two_F) > 2 or lower.two_F == upper.two_F == 0:
                continue
            transition = np.zeros((len(lower.basis), len(upper.basis)), dtype=complex)
            for lg in lower_groups:
                for ug in _basis_group_slices(upper.basis):
                    if lg.nuclear_path != ug.nuclear_path or abs(lg.N - ug.N) > 1:
                        continue
                    if lg.N == ug.N == 0:
                        continue
                    factor = _e1_recoupling_amplitude(
                        lg.N,
                        ug.N,
                        result.model.electron.two_S,
                        lg.two_J,
                        ug.two_J,
                        lg.two_I,
                        lower.two_F,
                        upper.two_F,
                    )
                    if abs(factor) <= 1.0e-15:
                        continue
                    if _transition_blocks is None:
                        body = np.zeros((2 * lg.N + 1, 2 * ug.N + 1), dtype=complex)
                        for mu, axis in zip(dipole, ("a", "b", "c"), strict=True):
                            if mu != 0.0:
                                body += mu * dipole_transition_matrix(
                                    lg.N, ug.N, axis, result.representation
                                )
                    else:
                        body = _transition_blocks[(lg.N, ug.N)]
                    transition[lg.start : lg.stop, ug.start : ug.stop] += factor * body
            reduced[(lower.two_F, upper.two_F)] = transition if _raw_operators else (
                lower.vectors.conj().T @ transition @ upper.vectors)
    return reduced


def spin_hyperfine_lam_transitions(
    surface: Any,
    lam_result: Any,
    result: SpinHyperfineResult,
    *,
    dipole_au: Sequence[float] | None = None,
    temperature_K: float | None = None,
    intensity_cut: float = 0.0,
) -> tuple[SpinHyperfineTransition, ...]:
    """Contract the dipole surface and return mixed LAM/hyperfine intensities."""
    blocks = spin_hyperfine_lam_reduced_dipole_matrices(
        surface, lam_result, result, dipole_au=dipole_au
    )
    # The body blocks already contain the dipole surface; the dummy vector is
    # used only by the shared input validator.
    return spin_hyperfine_transitions(
        result,
        (0.0, 0.0, 0.0),
        temperature_K=temperature_K,
        intensity_cut=intensity_cut,
        _reduced_matrices=blocks,
    )


def spin_hyperfine_lam_reduced_dipole_matrices(
    surface: Any,
    lam_result: Any,
    result: SpinHyperfineResult,
    *,
    dipole_au: Sequence[float] | None = None,
    _raw_operators: bool = False,
) -> dict[tuple[int, int], np.ndarray]:
    """Contract a LAM dipole surface and return resolved reduced F matrices."""
    primitive_dipole = surface.dipole_au
    if primitive_dipole is None:
        if dipole_au is None or np.asarray(dipole_au).shape != (3,):
            raise ValueError("provide a three-component dipole or a dipole surface")
        primitive_dipole = np.broadcast_to(
            np.asarray(dipole_au, dtype=float), surface.grid_shape + (3,)
        )
    primitive = np.asarray(primitive_dipole, dtype=float).reshape((-1, 3))
    U = np.asarray(lam_result.basis.vectors)
    contracted_mu = tuple(U.conj().T @ (primitive[:, a, None] * U) for a in range(3))
    rotational_blocks: dict[tuple[int, int], np.ndarray] = {}
    for N in result.n_values:
        for Np in result.n_values:
            if abs(N - Np) > 1 or N == Np == 0:
                continue
            lower = np.asarray(lam_result.blocks[N].vectors)
            upper = np.asarray(lam_result.blocks[Np].vectors)
            contracted_transition = np.zeros(
                (lower.shape[1], upper.shape[1]), dtype=complex
            )
            for axis, name in enumerate(("a", "b", "c")):
                contracted_transition += _contract_kron_between_states(
                    lower,
                    contracted_mu[axis],
                    dipole_transition_matrix(N, Np, name, result.representation),
                    upper,
                )
            rotational_blocks[(N, Np)] = contracted_transition
    return spin_hyperfine_reduced_dipole_matrices(
        result,
        (0.0, 0.0, 0.0),
        _transition_blocks=rotational_blocks,
        _raw_operators=_raw_operators,
    )


def _basis_group_slices(
    basis: tuple[SpinHyperfineBasisState, ...],
) -> tuple[_BasisGroup, ...]:
    groups: list[_BasisGroup] = []
    start = 0
    while start < len(basis):
        state = basis[start]
        stop = start + 1
        key = (state.N, state.two_J, state.nuclear_path)
        while stop < len(basis):
            candidate = basis[stop]
            if (candidate.N, candidate.two_J, candidate.nuclear_path) != key:
                break
            stop += 1
        groups.append(_BasisGroup(state.N, state.two_J, state.nuclear_path, start, stop))
        start = stop
    return tuple(groups)


@lru_cache(maxsize=32768)
def _e1_recoupling_amplitude(
    N: int,
    Np: int,
    two_S: int,
    two_J: int,
    two_Jp: int,
    two_I: int,
    two_F: int,
    two_Fp: int,
) -> float:
    phase_j = -1.0 if (((2 * N + two_S + two_Jp + 2) // 2) & 1) else 1.0
    first = phase_j * math.sqrt((two_J + 1.0) * (two_Jp + 1.0))
    first *= wigner_6j_doubled(2 * N, two_J, two_S, two_Jp, 2 * Np, 2)
    phase_f = -1.0 if (((two_J + two_I + two_Fp + 2) // 2) & 1) else 1.0
    second = phase_f * math.sqrt((two_F + 1.0) * (two_Fp + 1.0))
    second *= wigner_6j_doubled(two_J, two_F, two_I, two_Fp, two_Jp, 2)
    return first * second


def _assemble_f_block_hamiltonian(
    parameters: NativeRotorParameters,
    model: SpinHyperfineModel,
    two_spins: tuple[int, ...],
    two_F: int,
    basis: tuple[SpinHyperfineBasisState, ...],
    groups: tuple[_BasisGroup, ...],
    assembly_backend: str | None,
) -> tuple[np.ndarray, str]:
    dimension = len(basis)
    diagonal_terms: list[tuple[int, np.ndarray]] = []
    coupling_terms: list[tuple[int, int, complex, np.ndarray]] = []
    for group in groups:
        block = np.array(rotational_hamiltonian(parameters, group.N), copy=True)
        shift = _isotropic_spin_rotation_shift(
            group.N,
            model.electron.two_S,
            group.two_J,
            model.electron.isotropic_spin_rotation_MHz,
        )
        block.flat[:: block.shape[0] + 1] += shift
        diagonal_terms.append((group.start, block))

    anisotropic_sr = model.electron.anisotropic_spin_rotation_tensor_MHz
    if any(value != 0.0 for value in anisotropic_sr):
        _append_rank2_terms(
            coupling_terms,
            groups,
            two_F,
            two_spins,
            model.electron.two_S,
            anisotropic_sr,
            parameters.representation,
            actor="spin-rotation",
        )
    zfs = model.electron.zero_field_splitting_tensor_MHz
    if any(value != 0.0 for value in zfs):
        _append_rank2_terms(
            coupling_terms,
            groups,
            two_F,
            two_spins,
            model.electron.two_S,
            zfs,
            parameters.representation,
            actor="electron-electron",
        )

    for active_index, nucleus in enumerate(model.nuclei):
        if any(value != 0.0 for value in nucleus.quadrupole_tensor_MHz):
            for left_index, left in enumerate(groups):
                for right in groups[left_index:]:
                    if abs(left.N - right.N) > 2 or left.N + right.N < 2:
                        continue
                    body = quadrupole_body_block(
                        left.N,
                        right.N,
                        nucleus.quadrupole_tensor_MHz,
                        parameters.representation,
                        nucleus.two_I,
                    )
                    if not np.any(np.abs(body) > 1.0e-15):
                        continue
                    factor = _quadrupole_scalar_factor(
                        left.N,
                        right.N,
                        model.electron.two_S,
                        left.two_J,
                        right.two_J,
                        left.nuclear_path,
                        right.nuclear_path,
                        two_F,
                        two_spins,
                        active_index,
                    )
                    if abs(factor) <= 1.0e-15:
                        continue
                    coupling_terms.append((left.start, right.start, factor, body))
                    if left is not right:
                        coupling_terms.append(
                            (right.start, left.start, factor.conjugate(), body.conj().T)
                        )

        if nucleus.fermi_contact_MHz != 0.0:
            _append_isotropic_nuclear_terms(
                coupling_terms,
                groups,
                two_F,
                two_spins,
                active_index,
                model.electron.two_S,
                nucleus.fermi_contact_MHz,
                actor="electron",
            )
        if nucleus.isotropic_nuclear_spin_rotation_MHz != 0.0:
            _append_isotropic_nuclear_terms(
                coupling_terms,
                groups,
                two_F,
                two_spins,
                active_index,
                model.electron.two_S,
                nucleus.isotropic_nuclear_spin_rotation_MHz,
                actor="rotation",
            )
        if any(value != 0.0 for value in nucleus.electron_nuclear_dipolar_tensor_MHz):
            _append_rank2_terms(
                coupling_terms,
                groups,
                two_F,
                two_spins,
                model.electron.two_S,
                nucleus.electron_nuclear_dipolar_tensor_MHz,
                parameters.representation,
                actor="electron-nuclear",
                active_indices=(active_index,),
            )

    label_index = {nucleus.label.casefold(): i for i, nucleus in enumerate(model.nuclei)}
    for pair in model.nuclear_spin_spin:
        indices = (label_index[pair.left.casefold()], label_index[pair.right.casefold()])
        if pair.isotropic_MHz != 0.0:
            _append_isotropic_nuclear_pair_terms(
                coupling_terms,
                groups,
                two_F,
                two_spins,
                model.electron.two_S,
                indices,
                pair.isotropic_MHz,
            )
        if any(value != 0.0 for value in pair.dipolar_tensor_MHz):
            _append_rank2_terms(
                coupling_terms,
                groups,
                two_F,
                two_spins,
                model.electron.two_S,
                pair.dipolar_tensor_MHz,
                parameters.representation,
                actor="nuclear-nuclear",
                active_indices=indices,
            )

    hamiltonian, selected_assembly = _assemble_spin_hyperfine_block(
        dimension,
        diagonal_terms,
        coupling_terms,
        backend=assembly_backend,
    )

    return hamiltonian, selected_assembly


def _solve_f_block(
    parameters: NativeRotorParameters,
    model: SpinHyperfineModel,
    two_spins: tuple[int, ...],
    two_F: int,
    basis: tuple[SpinHyperfineBasisState, ...],
    groups: tuple[_BasisGroup, ...],
    levels_per_f: int | None,
    assembly_backend: str | None,
    diagonalizer_backend: str | None,
) -> SpinHyperfineBlock:
    dimension = len(basis)
    hamiltonian, selected_assembly = _assemble_f_block_hamiltonian(
        parameters, model, two_spins, two_F, basis, groups, assembly_backend
    )

    raw_error = float(np.max(np.abs(hamiltonian - hamiltonian.conj().T), initial=0.0))
    scale = max(1.0, float(np.max(np.abs(hamiltonian), initial=0.0)))
    if raw_error > 2.0e-10 * scale:
        raise RuntimeError(
            f"spin/hyperfine F={0.5 * two_F:g} block is not Hermitian: {raw_error:.3e} MHz"
        )
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    subset = None
    if levels_per_f is not None and levels_per_f < dimension:
        subset = (0, levels_per_f - 1)
    diagonalized = diagonalize_hermitian(
        hamiltonian,
        backend=diagonalizer_backend,
        subset_by_index=subset,
        check_finite=True,
    )
    if diagonalized.eigenvectors is None:
        raise RuntimeError("the spin/hyperfine diagonalizer did not return eigenvectors")
    energies = np.asarray(diagonalized.eigenvalues, dtype=float)
    vectors = np.asarray(diagonalized.eigenvectors, dtype=complex)
    residual = hamiltonian @ vectors - vectors * energies[None, :]
    matrix_scale = float(np.linalg.norm(hamiltonian, ord=np.inf))
    denominators = np.maximum(1.0, matrix_scale + np.abs(energies))
    scaled_residual = float(
        np.max(np.linalg.norm(residual, axis=0) / denominators, initial=0.0)
    )
    gram_error = float(
        np.max(np.abs(vectors.conj().T @ vectors - np.eye(vectors.shape[1])), initial=0.0)
    )
    if scaled_residual > 1.0e-10 or gram_error > 1.0e-10:
        raise RuntimeError(
            "spin/hyperfine eigensystem failed qualification: "
            f"residual={scaled_residual:.3e}, orthogonality={gram_error:.3e}"
        )
    levels = _assign_levels(parameters, two_F, energies, vectors, groups)
    energies.setflags(write=False)
    vectors.setflags(write=False)
    return SpinHyperfineBlock(
        two_F=two_F,
        energies_MHz=energies,
        vectors=vectors,
        basis=basis,
        levels=levels,
        hermiticity_error_MHz=raw_error,
        maximum_scaled_residual=scaled_residual,
        orthogonality_error=gram_error,
        assembly_backend=selected_assembly,
        diagonalizer_backend=diagonalized.backend,
    )


def solve_spin_hyperfine_lam_rotation(
    surface: Any,
    lam_result: Any,
    model: SpinHyperfineModel,
    *,
    f_max: float | None = None,
    levels_per_f: int | None = None,
    assembly_backend: str | None = None,
    diagonalizer_backend: str | None = None,
    matrix_free: bool | None = None,
    workers: int = 1,
) -> SpinHyperfineResult:
    """Contract spin tensors over DVR/LAM functions and solve the resulting F blocks.

    Coordinate-dependent rank-2 tensors are read from
    ``surface.spin_tensor_surfaces_MHz``.  Missing surfaces fall back to the
    constants in ``model``.  The accepted keys are documented in
    :func:`_lam_spin_tensor_key` and deliberately include isotope labels.
    """
    blocks_by_n = dict(lam_result.blocks)
    if not blocks_by_n:
        raise ValueError("the LAM-rotation result contains no rotational blocks")
    if f_max is not None and (not math.isfinite(float(f_max)) or f_max < 0.0):
        raise ValueError("f_max must be finite and nonnegative")
    if levels_per_f is not None and levels_per_f <= 0:
        raise ValueError("levels_per_f must be positive")
    if workers < 1:
        raise ValueError("workers must be at least one")
    _validate_lam_spin_surface_keys(surface, model)
    two_spins, basis_by_f, groups_by_f = _lam_spin_basis_registry(lam_result, model, f_max)
    arguments = [
        (
            surface,
            lam_result,
            model,
            two_spins,
            two_F,
            tuple(basis_by_f[two_F]),
            tuple(groups_by_f[two_F]),
            levels_per_f,
            assembly_backend,
            diagonalizer_backend,
            matrix_free,
        )
        for two_F in sorted(basis_by_f)
    ]
    if workers == 1 or len(arguments) < 2:
        blocks = tuple(_solve_lam_f_block(*item) for item in arguments)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lock-f-block") as pool:
            blocks = tuple(pool.map(lambda item: _solve_lam_f_block(*item), arguments))
    return SpinHyperfineResult(
        model=model,
        n_values=tuple(sorted(blocks_by_n)),
        blocks=blocks,
        representation=lam_result.representation,
    )


def _lam_spin_basis_registry(lam_result, model, f_max=None, only_two_F=None):
    """The common coupled basis for native solves and effective operators."""
    blocks_by_n = dict(lam_result.blocks)
    two_spins = tuple(nucleus.two_I for nucleus in model.nuclei)
    paths = _coupled_nuclear_spin_paths(two_spins)
    groups_by_f: dict[int, list[_BasisGroup]] = {}
    basis_by_f: dict[int, list[SpinHyperfineBasisState]] = {}
    for N, lam_block in sorted(blocks_by_n.items()):
        width = len(lam_block.energies_MHz)
        for two_J in range(abs(2 * N - model.electron.two_S), 2 * N + model.electron.two_S + 1, 2):
            for path in paths:
                two_I = path[-1] if path else 0
                for two_F in range(abs(two_J - two_I), two_J + two_I + 1, 2):
                    if f_max is not None and two_F > int(math.floor(2.0 * f_max + 1.0e-12)):
                        continue
                    if only_two_F is not None and two_F != only_two_F:
                        continue
                    basis = basis_by_f.setdefault(two_F, [])
                    start = len(basis)
                    basis.extend(
                        SpinHyperfineBasisState(N, index, two_J, path, two_F)
                        for index in range(width)
                    )
                    groups_by_f.setdefault(two_F, []).append(
                        _BasisGroup(N, two_J, path, start, len(basis))
                    )
    return two_spins, basis_by_f, groups_by_f


def _validate_lam_spin_surface_keys(surface: Any, model: SpinHyperfineModel) -> None:
    supplied = set((surface.spin_tensor_surfaces_MHz or {}).keys())
    electron_keys = {
        "electron.isotropic_spin_rotation_MHz",
        "electron.anisotropic_spin_rotation_tensor_MHz",
        "electron.zero_field_splitting_tensor_MHz",
        "electron.spin_lam_momentum_vectors_MHz",
    }
    allowed = set(electron_keys)
    for nucleus in model.nuclei:
        prefix = f"nucleus:{nucleus.label}."
        allowed.update(
            prefix + suffix
            for suffix in (
                "quadrupole_tensor_MHz",
                "fermi_contact_MHz",
                "isotropic_nuclear_spin_rotation_MHz",
                "electron_nuclear_dipolar_tensor_MHz",
            )
        )
    for pair in model.nuclear_spin_spin:
        prefix = f"pair:{pair.left}|{pair.right}."
        allowed.update(prefix + suffix for suffix in ("isotropic_MHz", "dipolar_tensor_MHz"))
    unknown = sorted(supplied - allowed)
    if unknown:
        raise ValueError(f"spin coupling surfaces do not match the model: {unknown}")
    if model.electron.two_S == 0 and supplied.intersection(electron_keys):
        raise ValueError("electron spin coupling surfaces require S > 0")
    spin_lam = model.electron.spin_lam_momentum_vectors_MHz
    if spin_lam and len(spin_lam) != surface.dimension:
        raise ValueError("spin-LAM momentum coupling requires one vector per LAM coordinate")
    if model.electron.two_S < 2 and "electron.zero_field_splitting_tensor_MHz" in supplied:
        raise ValueError("zero-field-splitting surfaces require S >= 1")
    if model.electron.two_S == 0 and any(
        key.endswith(("fermi_contact_MHz", "electron_nuclear_dipolar_tensor_MHz"))
        for key in supplied
    ):
        raise ValueError("electron-nuclear coupling surfaces require S > 0")
    spin_by_label = {nucleus.label: nucleus.two_I for nucleus in model.nuclei}
    for key in supplied:
        if key.startswith("nucleus:"):
            label = key.split(":", 1)[1].split(".", 1)[0]
            if spin_by_label[label] == 0:
                raise ValueError(f"spin coupling surface for spin-zero nucleus {label!r}")
            if key.endswith("quadrupole_tensor_MHz") and spin_by_label[label] < 2:
                raise ValueError(
                    f"quadrupole surface requires I >= 1 for nucleus {label!r}"
                )
        elif key.startswith("pair:"):
            labels = key.split(":", 1)[1].split(".", 1)[0].split("|")
            if any(spin_by_label[label] == 0 for label in labels):
                raise ValueError("nuclear spin-spin surfaces require both I > 0")


def _lam_f_block_operator_terms(surface, lam_result, model, two_spins, two_F, basis, groups,
                                include_spin_free=True,spin_free_blocks=None,contractions=None):
    """Shared native Wigner/contracted-tensor producer for solving and reduction."""
    contractions = contractions or {}
    scalar = contractions.get("scalar", _contract_lam_scalar_block)
    rank2 = contractions.get("rank2", _contract_lam_rank2_block)
    momentum = contractions.get("momentum", _contract_lam_spin_momentum_block)
    diagonal_terms: list[tuple[int, np.ndarray]] = []
    coupling_terms: list[tuple[int, int, complex, np.ndarray]] = []
    for group in groups:
        energies = np.asarray(lam_result.blocks[group.N].energies_MHz, dtype=float)
        block = np.diag(energies).astype(complex) if include_spin_free else np.zeros((len(energies),)*2,complex)
        if spin_free_blocks is not None and include_spin_free:
            block = np.array(spin_free_blocks[group.N], dtype=complex, copy=True)
        sr_factor = _isotropic_spin_rotation_shift(
            group.N, model.electron.two_S, group.two_J, 1.0
        )
        if sr_factor != 0.0:
            block += sr_factor * scalar(
                surface,
                lam_result,
                group.N,
                model.electron.isotropic_spin_rotation_MHz,
                "electron.isotropic_spin_rotation_MHz",
            )
        diagonal_terms.append((group.start, block))

    spin_lam_key = "electron.spin_lam_momentum_vectors_MHz"
    if (
        model.electron.spin_lam_momentum_vectors_MHz
        or spin_lam_key in (surface.spin_tensor_surfaces_MHz or {})
    ):
        for left_index, left in enumerate(groups):
            for right in groups[left_index:]:
                if abs(left.N - right.N) > 1 or left.N + right.N < 1:
                    continue
                spatial = momentum(
                    surface,
                    lam_result,
                    left.N,
                    right.N,
                    model.electron.spin_lam_momentum_vectors_MHz,
                    spin_lam_key,
                )
                if not np.any(np.abs(spatial) > 1.0e-15):
                    continue
                factor = _rank1_electron_spin_scalar_factor(
                    left.N,
                    right.N,
                    model.electron.two_S,
                    left.two_J,
                    right.two_J,
                    left.nuclear_path,
                    right.nuclear_path,
                    two_F,
                    two_spins,
                )
                if abs(factor) <= 1.0e-15:
                    continue
                coupling_terms.append((left.start, right.start, factor, spatial))
                if left is not right:
                    coupling_terms.append(
                        (right.start, left.start, factor.conjugate(), spatial.conj().T)
                    )

    def append_rank2(
        tensor: tuple[float, float, float, float, float, float],
        actor: str,
        key: str,
        active_indices: tuple[int, ...] = (),
        divisor: float = 1.0,
    ) -> None:
        if not any(value != 0.0 for value in tensor) and key not in (
            surface.spin_tensor_surfaces_MHz or {}
        ):
            return
        for left_index, left in enumerate(groups):
            for right in groups[left_index:]:
                if abs(left.N - right.N) > 2 or left.N + right.N < 2:
                    continue
                body = rank2(
                    surface,
                    lam_result,
                    left.N,
                    right.N,
                    tensor,
                    key,
                ) / divisor
                if not np.any(np.abs(body) > 1.0e-15):
                    continue
                if actor == "quadrupole":
                    factor = _quadrupole_scalar_factor(
                        left.N, right.N, model.electron.two_S,
                        left.two_J, right.two_J, left.nuclear_path,
                        right.nuclear_path, two_F, two_spins, active_indices[0]
                    )
                elif actor == "spin-rotation":
                    factor = _anisotropic_spin_rotation_scalar_factor(
                        left.N, right.N, model.electron.two_S,
                        left.two_J, right.two_J, left.nuclear_path,
                        right.nuclear_path, two_F, two_spins
                    )
                else:
                    factor = _rank2_spin_scalar_factor(
                        left.N, right.N, model.electron.two_S,
                        left.two_J, right.two_J, left.nuclear_path,
                        right.nuclear_path, two_F, two_spins, actor, active_indices
                    )
                if abs(factor) <= 1.0e-15:
                    continue
                coupling_terms.append((left.start, right.start, factor, body))
                if left is not right:
                    coupling_terms.append(
                        (right.start, left.start, factor.conjugate(), body.conj().T)
                    )

    append_rank2(
        model.electron.anisotropic_spin_rotation_tensor_MHz,
        "spin-rotation",
        "electron.anisotropic_spin_rotation_tensor_MHz",
    )
    append_rank2(
        model.electron.zero_field_splitting_tensor_MHz,
        "electron-electron",
        "electron.zero_field_splitting_tensor_MHz",
    )
    for active_index, nucleus in enumerate(model.nuclei):
        reduced = _pickett_quadrupole_reduced_spin_factor(nucleus.two_I)
        append_rank2(
            nucleus.quadrupole_tensor_MHz,
            "quadrupole",
            f"nucleus:{nucleus.label}.quadrupole_tensor_MHz",
            (active_index,),
            reduced if reduced != 0.0 else 1.0,
        )
        append_rank2(
            nucleus.electron_nuclear_dipolar_tensor_MHz,
            "electron-nuclear",
            f"nucleus:{nucleus.label}.electron_nuclear_dipolar_tensor_MHz",
            (active_index,),
        )
        fermi_key = f"nucleus:{nucleus.label}.fermi_contact_MHz"
        if (
            nucleus.fermi_contact_MHz != 0.0
            or fermi_key in (surface.spin_tensor_surfaces_MHz or {})
        ):
            _append_contracted_scalar_terms(
                coupling_terms, groups, two_F, two_spins, model.electron.two_S,
                nucleus.fermi_contact_MHz, actor="electron", active_indices=(active_index,),
                surface=surface, lam_result=lam_result,
                surface_key=fermi_key,contract_scalar=scalar,
            )
        nuclear_sr_key = f"nucleus:{nucleus.label}.isotropic_nuclear_spin_rotation_MHz"
        if (
            nucleus.isotropic_nuclear_spin_rotation_MHz != 0.0
            or nuclear_sr_key in (surface.spin_tensor_surfaces_MHz or {})
        ):
            _append_contracted_scalar_terms(
                coupling_terms, groups, two_F, two_spins, model.electron.two_S,
                nucleus.isotropic_nuclear_spin_rotation_MHz,
                actor="rotation", active_indices=(active_index,), surface=surface,
                lam_result=lam_result,
                surface_key=nuclear_sr_key,contract_scalar=scalar,
            )
    label_index = {nucleus.label.casefold(): i for i, nucleus in enumerate(model.nuclei)}
    for pair in model.nuclear_spin_spin:
        indices = (label_index[pair.left.casefold()], label_index[pair.right.casefold()])
        append_rank2(
            pair.dipolar_tensor_MHz,
            "nuclear-nuclear",
            f"pair:{pair.left}|{pair.right}.dipolar_tensor_MHz",
            indices,
        )
        pair_key = f"pair:{pair.left}|{pair.right}.isotropic_MHz"
        if (
            pair.isotropic_MHz != 0.0
            or pair_key in (surface.spin_tensor_surfaces_MHz or {})
        ):
            _append_contracted_scalar_terms(
                coupling_terms, groups, two_F, two_spins, model.electron.two_S,
                pair.isotropic_MHz, actor="nuclear-pair", active_indices=indices,
                surface=surface, lam_result=lam_result, surface_key=pair_key,contract_scalar=scalar,
            )

    return diagonal_terms, coupling_terms


def _solve_lam_f_block(
    surface: Any,
    lam_result: Any,
    model: SpinHyperfineModel,
    two_spins: tuple[int, ...],
    two_F: int,
    basis: tuple[SpinHyperfineBasisState, ...],
    groups: tuple[_BasisGroup, ...],
    levels_per_f: int | None,
    assembly_backend: str | None,
    diagonalizer_backend: str | None,
    matrix_free: bool | None,
) -> SpinHyperfineBlock:
    diagonal_terms, coupling_terms = _lam_f_block_operator_terms(
        surface, lam_result, model, two_spins, two_F, basis, groups)

    use_matrix_free = bool(matrix_free) or (
        matrix_free is None
        and len(basis) >= 512
        and levels_per_f is not None
        and levels_per_f < len(basis) - 1
    )
    if use_matrix_free:
        if levels_per_f is None or levels_per_f >= len(basis) - 1:
            raise ValueError(
                "matrix-free spin diagonalization requires levels_per_f < dimension - 1"
            )
        return _solve_lam_f_block_matrix_free(
            lam_result,
            two_F,
            basis,
            groups,
            levels_per_f,
            diagonal_terms,
            coupling_terms,
        )

    hamiltonian, selected_assembly = _assemble_spin_hyperfine_block(
        len(basis), diagonal_terms, coupling_terms, backend=assembly_backend
    )
    raw_error = float(np.max(np.abs(hamiltonian - hamiltonian.conj().T), initial=0.0))
    scale = max(1.0, float(np.max(np.abs(hamiltonian), initial=0.0)))
    if raw_error > 2.0e-10 * scale:
        raise RuntimeError(
            f"contracted LAM spin F={0.5 * two_F:g} block is not Hermitian: {raw_error:.3e} MHz"
        )
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    subset = None
    if levels_per_f is not None and levels_per_f < len(basis):
        subset = (0, levels_per_f - 1)
    diagonalized = diagonalize_hermitian(
        hamiltonian, backend=diagonalizer_backend, subset_by_index=subset, check_finite=True
    )
    if diagonalized.eigenvectors is None:
        raise RuntimeError("the contracted LAM spin diagonalizer did not return eigenvectors")
    energies = np.asarray(diagonalized.eigenvalues, dtype=float)
    vectors = np.asarray(diagonalized.eigenvectors, dtype=complex)
    residual = hamiltonian @ vectors - vectors * energies[None, :]
    denominators = np.maximum(1.0, np.linalg.norm(hamiltonian, ord=np.inf) + np.abs(energies))
    scaled_residual = float(np.max(np.linalg.norm(residual, axis=0) / denominators, initial=0.0))
    gram_error = float(np.max(np.abs(vectors.conj().T @ vectors - np.eye(vectors.shape[1])), initial=0.0))
    if scaled_residual > 1.0e-10 or gram_error > 1.0e-10:
        raise RuntimeError(
            "contracted LAM spin eigensystem failed qualification: "
            f"residual={scaled_residual:.3e}, orthogonality={gram_error:.3e}"
        )
    levels = _assign_lam_spin_levels(lam_result, two_F, energies, vectors, groups)
    energies.setflags(write=False)
    vectors.setflags(write=False)
    return SpinHyperfineBlock(
        two_F=two_F, energies_MHz=energies, vectors=vectors, basis=basis, levels=levels,
        hermiticity_error_MHz=raw_error, maximum_scaled_residual=scaled_residual,
        orthogonality_error=gram_error, assembly_backend=selected_assembly,
        diagonalizer_backend=diagonalized.backend,
    )


def _solve_lam_f_block_matrix_free(
    lam_result: Any,
    two_F: int,
    basis: tuple[SpinHyperfineBasisState, ...],
    groups: tuple[_BasisGroup, ...],
    levels_per_f: int,
    diagonal_terms: list[tuple[int, np.ndarray]],
    coupling_terms: list[tuple[int, int, complex, np.ndarray]],
) -> SpinHyperfineBlock:
    """Iteratively solve an F block without allocating its dense Hamiltonian."""
    from scipy.sparse.linalg import LinearOperator, eigsh

    dimension = len(basis)

    def apply(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=complex)
        is_vector = array.ndim == 1
        columns = array.reshape((dimension, -1))
        result = np.zeros_like(columns)
        for start, block in diagonal_terms:
            stop = start + block.shape[0]
            result[start:stop] += block @ columns[start:stop]
        for row_start, column_start, factor, block in coupling_terms:
            row_stop = row_start + block.shape[0]
            column_stop = column_start + block.shape[1]
            result[row_start:row_stop] += factor * (block @ columns[column_start:column_stop])
        return result[:, 0] if is_vector else result

    operator = LinearOperator(
        (dimension, dimension), matvec=apply, matmat=apply, dtype=np.dtype(complex)
    )
    rng = np.random.default_rng(271828 + two_F)
    raw_error = 0.0
    for _ in range(3):
        left = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
        right = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
        defect = abs(np.vdot(left, apply(right)) - np.vdot(apply(left), right))
        raw_error = max(raw_error, float(defect / (np.linalg.norm(left) * np.linalg.norm(right))))
    scale = max(
        1.0,
        max((float(np.max(np.abs(block), initial=0.0)) for _start, block in diagonal_terms), default=1.0),
    )
    if raw_error > 2.0e-10 * scale:
        raise RuntimeError(
            f"matrix-free LAM spin F={0.5 * two_F:g} block is not Hermitian: {raw_error:.3e} MHz"
        )
    energies, vectors = eigsh(
        operator,
        k=levels_per_f,
        which="SA",
        tol=2.0e-12,
        maxiter=max(1000, 20 * dimension),
        v0=rng.normal(size=dimension) + 1j * rng.normal(size=dimension),
    )
    order = np.argsort(energies.real)
    energies = np.asarray(energies.real[order], dtype=float)
    vectors = np.asarray(vectors[:, order], dtype=complex)
    # ARPACK may return visibly nonorthogonal vectors inside an unresolved
    # multiplet.  Orthonormalize the selected invariant subspace, then perform
    # a Rayleigh--Ritz diagonalization in that small subspace.
    vectors, _ = np.linalg.qr(vectors, mode="reduced")
    projected = 0.5 * (
        vectors.conj().T @ apply(vectors)
        + (vectors.conj().T @ apply(vectors)).conj().T
    )
    energies, rotation = np.linalg.eigh(projected)
    vectors = vectors @ rotation
    from ._iterative_roots import complete_lowest_ritz_subspace

    row_bound = np.zeros(dimension)
    for start, block in diagonal_terms:
        row_bound[start:start+len(block)] += np.sum(np.abs(block), axis=1)
    for row, _column, factor, block in coupling_terms:
        row_bound[row:row+len(block)] += abs(factor) * np.sum(np.abs(block), axis=1)
    energies, vectors = complete_lowest_ritz_subspace(
        apply, energies, vectors, spectral_bound=float(np.max(row_bound)), seed=271828+two_F,
    )
    residual = apply(vectors) - vectors * energies[None, :]
    denominators = np.maximum(1.0, np.abs(energies) + scale)
    scaled_residual = float(
        np.max(np.linalg.norm(residual, axis=0) / denominators, initial=0.0)
    )
    gram_error = float(
        np.max(
            np.abs(vectors.conj().T @ vectors - np.eye(vectors.shape[1])), initial=0.0
        )
    )
    if scaled_residual > 2.0e-10 or gram_error > 2.0e-10:
        raise RuntimeError(
            "matrix-free LAM spin eigensystem failed qualification: "
            f"residual={scaled_residual:.3e}, orthogonality={gram_error:.3e}"
        )
    levels = _assign_lam_spin_levels(lam_result, two_F, energies, vectors, groups)
    energies.setflags(write=False)
    vectors.setflags(write=False)
    return SpinHyperfineBlock(
        two_F=two_F,
        energies_MHz=energies,
        vectors=vectors,
        basis=basis,
        levels=levels,
        hermiticity_error_MHz=raw_error,
        maximum_scaled_residual=scaled_residual,
        orthogonality_error=gram_error,
        assembly_backend="matrix-free",
        diagonalizer_backend="scipy-arpack-matrix-free",
    )


def _append_contracted_scalar_terms(
    terms: list[tuple[int, int, complex, np.ndarray]],
    groups: tuple[_BasisGroup, ...],
    two_F: int,
    two_spins: tuple[int, ...],
    two_S: int,
    constant: float,
    *,
    actor: str,
    active_indices: tuple[int, ...],
    surface: Any,
    lam_result: Any,
    surface_key: str,
    contract_scalar=None,
) -> None:
    for left_index, left in enumerate(groups):
        for right in groups[left_index:]:
            if left.N != right.N:
                continue
            if actor == "nuclear-pair":
                factor = _isotropic_nuclear_pair_scalar_factor(
                    left.N, two_S, left.two_J, right.two_J, left.nuclear_path,
                    right.nuclear_path, two_F, two_spins,
                    (active_indices[0], active_indices[1])
                )
            else:
                factor = _isotropic_nuclear_scalar_factor(
                    left.N, two_S, left.two_J, right.two_J, left.nuclear_path,
                    right.nuclear_path, two_F, two_spins, active_indices[0], actor
                )
            if abs(factor) <= 1.0e-15:
                continue
            spatial = (contract_scalar or _contract_lam_scalar_block)(
                surface, lam_result, left.N, constant, surface_key
            )
            terms.append((left.start, right.start, factor, spatial))
            if left is not right:
                terms.append(
                    (right.start, left.start, factor.conjugate(), spatial.conj().T)
                )


def _contract_lam_rank2_block(
    surface: Any,
    lam_result: Any,
    N: int,
    Np: int,
    constant_tensor: tuple[float, float, float, float, float, float],
    surface_key: str,
) -> np.ndarray:
    primitive_surface = (surface.spin_tensor_surfaces_MHz or {}).get(surface_key)
    grid_size = int(np.prod(surface.grid_shape))
    if primitive_surface is None:
        values = np.broadcast_to(np.asarray(constant_tensor, dtype=float), (grid_size, 6))
    else:
        values = np.asarray(primitive_surface, dtype=float).reshape((grid_size, 6))
    U = np.asarray(lam_result.basis.vectors)
    lower = np.asarray(lam_result.blocks[N].vectors)
    upper = np.asarray(lam_result.blocks[Np].vectors)
    result = np.zeros((lower.shape[1], upper.shape[1]), dtype=complex)
    for component in range(6):
        if not np.any(values[:, component] != 0.0):
            continue
        contracted = U.conj().T @ (values[:, component, None] * U)
        unit = [0.0] * 6
        unit[component] = 1.0
        body = rank2_body_block(N, Np, tuple(unit), lam_result.representation)
        result += _contract_kron_between_states(lower, contracted, body, upper)
    return result


def _contract_lam_spin_momentum_block(
    surface: Any,
    lam_result: Any,
    N: int,
    Np: int,
    constant_vectors: tuple[tuple[float, float, float], ...],
    surface_key: str,
) -> np.ndarray:
    """Contract ``1/2 {e_mu(q), p_mu} . S`` over the DVR/LAM basis."""
    from .lam_rotation import primitive_momentum_operators

    grid_size = int(np.prod(surface.grid_shape))
    primitive_surface = (surface.spin_tensor_surfaces_MHz or {}).get(surface_key)
    if primitive_surface is None:
        values = np.broadcast_to(
            np.asarray(constant_vectors, dtype=float),
            (grid_size, surface.dimension, 3),
        )
    else:
        values = np.asarray(primitive_surface, dtype=float).reshape(
            (grid_size, surface.dimension, 3)
        )
    U = np.asarray(lam_result.basis.vectors)
    momenta = lam_result.basis.momentum_operators or primitive_momentum_operators(surface)
    lower = np.asarray(lam_result.blocks[N].vectors)
    upper = np.asarray(lam_result.blocks[Np].vectors)
    result = np.zeros((lower.shape[1], upper.shape[1]), dtype=complex)
    for mu, momentum in enumerate(momenta):
        for axis in range(3):
            coefficient = values[:, mu, axis]
            if not np.any(coefficient != 0.0):
                continue
            primitive = 0.5 * (
                coefficient[:, None] * momentum + momentum * coefficient[None, :]
            )
            contracted = U.conj().T @ primitive @ U
            unit = [0.0, 0.0, 0.0]
            unit[axis] = 1.0
            body = rank1_body_block(N, Np, tuple(unit), lam_result.representation)
            result += _contract_kron_between_states(lower, contracted, body, upper)
    return result


def _contract_lam_scalar_block(
    surface: Any,
    lam_result: Any,
    N: int,
    constant: float,
    surface_key: str,
) -> np.ndarray:
    primitive_surface = (surface.spin_tensor_surfaces_MHz or {}).get(surface_key)
    block = np.asarray(lam_result.blocks[N].vectors)
    if primitive_surface is None:
        return float(constant) * np.eye(block.shape[1], dtype=complex)
    values = np.asarray(primitive_surface, dtype=float).reshape(-1)
    U = np.asarray(lam_result.basis.vectors)
    contracted = U.conj().T @ (values[:, None] * U)
    return _contract_kron_between_states(
        block, contracted, np.eye(2 * N + 1), block
    )


def _contract_kron_between_states(
    lower: np.ndarray,
    coordinate_operator: np.ndarray,
    rotational_operator: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Evaluate ``L^H (C kron R) U`` without materializing the Kronecker matrix."""
    nlam_l, nlam_u = coordinate_operator.shape
    nrot_l, nrot_u = rotational_operator.shape
    left = np.asarray(lower).reshape((nlam_l, nrot_l, lower.shape[1]))
    right = np.asarray(upper).reshape((nlam_u, nrot_u, upper.shape[1]))
    return np.einsum(
        "xka,xy,kl,ylb->ab",
        left.conj(),
        coordinate_operator,
        rotational_operator,
        right,
        optimize=True,
    )


def _assign_lam_spin_levels(
    lam_result: Any,
    two_F: int,
    energies: np.ndarray,
    vectors: np.ndarray,
    groups: tuple[_BasisGroup, ...],
) -> tuple[SpinHyperfineLevel, ...]:
    levels: list[SpinHyperfineLevel] = []
    for index, energy in enumerate(energies):
        vector = vectors[:, index]
        weights = np.asarray([
            float(np.vdot(vector[group.start:group.stop], vector[group.start:group.stop]).real)
            for group in groups
        ])
        group = groups[int(np.argmax(weights))]
        local = np.abs(vector[group.start:group.stop]) ** 2
        parent_index = int(np.argmax(local))
        parent = lam_result.blocks[group.N].levels[parent_index]
        levels.append(
            SpinHyperfineLevel(
                energy_MHz=float(energy), two_F=two_F, N=group.N,
                two_J=group.two_J, nuclear_path=group.nuclear_path,
                Ka=parent.Ka, Kc=parent.Kc,
                parent_purity=float(weights.max() * local[parent_index] / max(weights.max(), 1.0e-300)),
                lam_parent=parent.lam_parent, torsional_state=parent.torsional_state,
                lam_symmetry=parent.lam_symmetry,
                spin_weight=parent.spin_weight,
            )
        )
    return tuple(levels)


def spin_hyperfine_native_available() -> bool:
    """Return whether the installed LOCK extension contains the spin kernel."""
    return _lock_native is not None and hasattr(_lock_native, "assemble_spin_hyperfine")


def hellmann_feynman_diagonal(
    vectors: np.ndarray,
    operators: Sequence[np.ndarray],
    *,
    backend: str = "auto",
) -> np.ndarray:
    """Return ``<psi|dH/dp|psi>`` for a batch of Hermitian operators."""
    states = np.asarray(vectors, dtype=complex)
    derivatives = np.asarray(tuple(operators), dtype=complex)
    if states.ndim != 2 or derivatives.ndim != 3 or derivatives.shape[1:] != (
        states.shape[0], states.shape[0]
    ):
        raise ValueError("operators must have shape (nparameter, nbasis, nbasis)")
    if backend not in {"auto", "python", "compiled"}:
        raise ValueError("Hellmann-Feynman backend must be auto, python, or compiled")
    native = _lock_native is not None and hasattr(_lock_native, "hellmann_feynman_diagonal")
    if backend == "compiled" and not native:
        raise RuntimeError("compiled LOCK Hellmann-Feynman kernel is unavailable")
    if native and backend != "python":
        return np.asarray(
            _lock_native.hellmann_feynman_diagonal(states, derivatives), dtype=float
        )
    return np.real(np.einsum("is,pij,js->ps", states.conj(), derivatives, states, optimize=True))


def _assemble_spin_hyperfine_block(
    dimension: int,
    diagonal_terms: Sequence[tuple[int, np.ndarray]],
    coupling_terms: Sequence[tuple[int, int, complex, np.ndarray]],
    *,
    backend: str | None,
) -> tuple[np.ndarray, str]:
    selected = resolve_native_backend(
        extension_available=spin_hyperfine_native_available(),
        extension_name="lock-cpp-spin",
        workload_size=dimension * dimension,
        requested=backend,
        minimum_work=256,
    )
    if selected.accelerated and _lock_native is not None:
        matrix = _lock_native.assemble_spin_hyperfine(
            dimension,
            tuple((offset, np.asarray(block, dtype=complex)) for offset, block in diagonal_terms),
            tuple(
                (row, column, complex(scale), np.asarray(block, dtype=complex))
                for row, column, scale, block in coupling_terms
            ),
        )
        return np.asarray(matrix, dtype=complex), selected.name

    matrix = np.zeros((dimension, dimension), dtype=complex)
    for offset, block in diagonal_terms:
        stop = offset + block.shape[0]
        matrix[offset:stop, offset:stop] += block
    for row, column, scale, block in coupling_terms:
        matrix[row : row + block.shape[0], column : column + block.shape[1]] += scale * block
    return matrix, selected.name


def _assign_levels(
    parameters: NativeRotorParameters,
    two_F: int,
    energies: np.ndarray,
    vectors: np.ndarray,
    groups: tuple[_BasisGroup, ...],
) -> tuple[SpinHyperfineLevel, ...]:
    levels: list[SpinHyperfineLevel] = []
    for index, energy in enumerate(energies):
        vector = vectors[:, index]
        weights = np.asarray(
            [float(np.vdot(vector[group.start : group.stop], vector[group.start : group.stop]).real) for group in groups]
        )
        group_index = int(np.argmax(weights))
        group = groups[group_index]
        coefficients = vector[group.start : group.stop]
        rotor = diagonalize_rotor(parameters, group.N)
        parent_amplitudes = rotor.vectors.conj().T @ coefficients
        parent = int(np.argmax(np.abs(parent_amplitudes) ** 2))
        Ka, Kc, _tau = rotor.assignments[parent]
        levels.append(
            SpinHyperfineLevel(
                energy_MHz=float(energy),
                two_F=two_F,
                N=group.N,
                two_J=group.two_J,
                nuclear_path=group.nuclear_path,
                Ka=Ka,
                Kc=Kc,
                parent_purity=float(weights[group_index]),
            )
        )
    return tuple(levels)


def _isotropic_spin_rotation_shift(N: int, two_S: int, two_J: int, gamma_MHz: float) -> float:
    if two_S == 0 or gamma_MHz == 0.0:
        return 0.0
    S = 0.5 * two_S
    J = 0.5 * two_J
    return 0.5 * gamma_MHz * (J * (J + 1.0) - N * (N + 1.0) - S * (S + 1.0))


def _append_isotropic_nuclear_terms(
    terms: list[tuple[int, int, complex, np.ndarray]],
    groups: tuple[_BasisGroup, ...],
    two_F: int,
    two_spins: tuple[int, ...],
    active_index: int,
    two_S: int,
    constant_MHz: float,
    *,
    actor: str,
) -> None:
    identity_by_n: dict[int, np.ndarray] = {}
    for left_index, left in enumerate(groups):
        for right in groups[left_index:]:
            if left.N != right.N:
                continue
            factor = _isotropic_nuclear_scalar_factor(
                left.N,
                two_S,
                left.two_J,
                right.two_J,
                left.nuclear_path,
                right.nuclear_path,
                two_F,
                two_spins,
                active_index,
                actor,
            )
            if abs(factor) <= 1.0e-15:
                continue
            identity = identity_by_n.setdefault(left.N, np.eye(2 * left.N + 1, dtype=complex))
            scale = constant_MHz * factor
            terms.append((left.start, right.start, scale, identity))
            if left is not right:
                terms.append((right.start, left.start, scale.conjugate(), identity))


def _append_isotropic_nuclear_pair_terms(
    terms: list[tuple[int, int, complex, np.ndarray]],
    groups: tuple[_BasisGroup, ...],
    two_F: int,
    two_spins: tuple[int, ...],
    two_S: int,
    active_indices: tuple[int, int],
    constant_MHz: float,
) -> None:
    identity_by_n: dict[int, np.ndarray] = {}
    for left_index, left in enumerate(groups):
        for right in groups[left_index:]:
            if left.N != right.N:
                continue
            factor = _isotropic_nuclear_pair_scalar_factor(
                left.N,
                two_S,
                left.two_J,
                right.two_J,
                left.nuclear_path,
                right.nuclear_path,
                two_F,
                two_spins,
                active_indices,
            )
            if abs(factor) <= 1.0e-15:
                continue
            identity = identity_by_n.setdefault(left.N, np.eye(2 * left.N + 1, dtype=complex))
            scale = constant_MHz * factor
            terms.append((left.start, right.start, scale, identity))
            if left is not right:
                terms.append((right.start, left.start, scale.conjugate(), identity))


@lru_cache(maxsize=32768)
def _isotropic_nuclear_pair_scalar_factor(
    N: int,
    two_S: int,
    two_J: int,
    two_Jp: int,
    path: tuple[int, ...],
    pathp: tuple[int, ...],
    two_F: int,
    two_spins: tuple[int, ...],
    active_indices: tuple[int, int],
) -> complex:
    left = _coupled_f_vector(N, two_S, two_J, path, two_F, two_spins)
    right = _coupled_f_vector(N, two_S, two_Jp, pathp, two_F, two_spins)
    pair = np.zeros((left.shape[2], left.shape[2]), dtype=complex)
    for q in (-1, 0, 1):
        phase = -1.0 if (q & 1) else 1.0
        first = _coupled_nuclear_rank1_component(two_spins, active_indices[0], q)
        second = _coupled_nuclear_rank1_component(two_spins, active_indices[1], -q)
        pair += phase * (first @ second)
    total = sum(
        np.vdot(left[n_index, spin_index], pair @ right[n_index, spin_index])
        for n_index in range(left.shape[0])
        for spin_index in range(left.shape[1])
    )
    return _clean_complex(total)


def _append_rank2_terms(
    terms: list[tuple[int, int, complex, np.ndarray]],
    groups: tuple[_BasisGroup, ...],
    two_F: int,
    two_spins: tuple[int, ...],
    two_S: int,
    tensor_MHz: tuple[float, float, float, float, float, float],
    representation: str,
    *,
    actor: str,
    active_indices: tuple[int, ...] = (),
) -> None:
    for left_index, left in enumerate(groups):
        for right in groups[left_index:]:
            if abs(left.N - right.N) > 2 or left.N + right.N < 2:
                continue
            body = rank2_body_block(left.N, right.N, tensor_MHz, representation)
            if not np.any(np.abs(body) > 1.0e-15):
                continue
            if actor == "spin-rotation":
                factor = _anisotropic_spin_rotation_scalar_factor(
                    left.N,
                    right.N,
                    two_S,
                    left.two_J,
                    right.two_J,
                    left.nuclear_path,
                    right.nuclear_path,
                    two_F,
                    two_spins,
                )
            else:
                factor = _rank2_spin_scalar_factor(
                    left.N,
                    right.N,
                    two_S,
                    left.two_J,
                    right.two_J,
                    left.nuclear_path,
                    right.nuclear_path,
                    two_F,
                    two_spins,
                    actor,
                    active_indices,
                )
            if abs(factor) <= 1.0e-15:
                continue
            terms.append((left.start, right.start, factor, body))
            if left is not right:
                terms.append((right.start, left.start, factor.conjugate(), body.conj().T))


@lru_cache(maxsize=32768)
def _rank2_spin_scalar_factor(
    N: int,
    Np: int,
    two_S: int,
    two_J: int,
    two_Jp: int,
    path: tuple[int, ...],
    pathp: tuple[int, ...],
    two_F: int,
    two_spins: tuple[int, ...],
    actor: str,
    active_indices: tuple[int, ...],
) -> complex:
    left = _coupled_f_vector(N, two_S, two_J, path, two_F, two_spins)
    right = _coupled_f_vector(Np, two_S, two_Jp, pathp, two_F, two_spins)
    total = 0.0 + 0.0j
    for p in range(-2, 3):
        phase = -1.0 if (p & 1) else 1.0
        rotational = _rank2_rotational_space_component(N, Np, p)
        internal = _coupled_internal_rank2_component(
            two_S, two_spins, actor, active_indices, -p
        )
        right_flat = right.reshape((right.shape[0], -1))
        transformed = rotational @ right_flat @ internal.T
        total += phase * np.vdot(left.reshape((left.shape[0], -1)), transformed)
    # With the Cartesian/spherical conventions used here,
    # A:T(A,B) = sqrt(2/3) sum_q (-1)^q A_q [A x B]^2_-q.
    return _clean_complex(math.sqrt(2.0 / 3.0) * total)


@lru_cache(maxsize=32768)
def _rank1_electron_spin_scalar_factor(
    N: int,
    Np: int,
    two_S: int,
    two_J: int,
    two_Jp: int,
    path: tuple[int, ...],
    pathp: tuple[int, ...],
    two_F: int,
    two_spins: tuple[int, ...],
) -> complex:
    """Space-fixed scalar factor for a body-fixed vector coupled to electron spin."""
    left = _coupled_f_vector(N, two_S, two_J, path, two_F, two_spins)
    right = _coupled_f_vector(Np, two_S, two_Jp, pathp, two_F, two_spins)
    nuclear_identity = np.eye(left.shape[2], dtype=complex)
    total = 0.0 + 0.0j
    for p in (-1, 0, 1):
        phase = -1.0 if (p & 1) else 1.0
        rotational = _rank1_rotational_space_component(N, Np, p)
        spin = _angular_momentum_rank1_component(two_S, -p)
        internal = np.kron(spin, nuclear_identity)
        transformed = rotational @ right.reshape((right.shape[0], -1)) @ internal.T
        total += phase * np.vdot(left.reshape((left.shape[0], -1)), transformed)
    return _clean_complex(total)


@lru_cache(maxsize=32768)
def _anisotropic_spin_rotation_scalar_factor(
    N: int,
    Np: int,
    two_S: int,
    two_J: int,
    two_Jp: int,
    path: tuple[int, ...],
    pathp: tuple[int, ...],
    two_F: int,
    two_spins: tuple[int, ...],
) -> complex:
    left = _coupled_f_vector(N, two_S, two_J, path, two_F, two_spins)
    right = _coupled_f_vector(Np, two_S, two_Jp, pathp, two_F, two_spins)
    nuclear_identity = np.eye(left.shape[2], dtype=complex)
    total = 0.0 + 0.0j
    for p in range(-2, 3):
        D = _rank2_rotational_space_component(N, Np, p)
        phase = -1.0 if (p & 1) else 1.0
        for qn in (-1, 0, 1):
            qs = -p - qn
            if qs not in {-1, 0, 1}:
                continue
            cg = clebsch_gordan_doubled(2, 2 * qn, 2, 2 * qs, 4, -2 * p)
            if cg == 0.0:
                continue
            N_left = _angular_momentum_rank1_component(2 * N, qn)
            N_right = _angular_momentum_rank1_component(2 * Np, qn)
            ordered = 0.5 * (N_left @ D + D @ N_right)
            spin = _angular_momentum_rank1_component(two_S, qs)
            internal = np.kron(spin, nuclear_identity)
            transformed = ordered @ right.reshape((right.shape[0], -1)) @ internal.T
            total += phase * cg * np.vdot(left.reshape((left.shape[0], -1)), transformed)
    return _clean_complex(math.sqrt(2.0 / 3.0) * total)


@lru_cache(maxsize=4096)
def _coupled_internal_rank2_component(
    two_S: int,
    two_spins: tuple[int, ...],
    actor: str,
    active_indices: tuple[int, ...],
    q: int,
) -> np.ndarray:
    nuclear_dimension = _nuclear_coupling_transform(two_spins)[2].shape[1]
    result = np.zeros(((two_S + 1) * nuclear_dimension,) * 2, dtype=complex)
    for q1 in (-1, 0, 1):
        q2 = q - q1
        if q2 not in {-1, 0, 1}:
            continue
        cg = clebsch_gordan_doubled(2, 2 * q1, 2, 2 * q2, 4, 2 * q)
        if cg == 0.0:
            continue
        if actor == "electron-electron":
            electronic = (
                _angular_momentum_rank1_component(two_S, q1)
                @ _angular_momentum_rank1_component(two_S, q2)
            )
            component = np.kron(electronic, np.eye(nuclear_dimension))
        elif actor == "electron-nuclear":
            electronic = _angular_momentum_rank1_component(two_S, q1)
            nuclear = _coupled_nuclear_rank1_component(two_spins, active_indices[0], q2)
            component = np.kron(electronic, nuclear)
        elif actor == "nuclear-nuclear":
            first = _coupled_nuclear_rank1_component(two_spins, active_indices[0], q1)
            second = _coupled_nuclear_rank1_component(two_spins, active_indices[1], q2)
            component = np.kron(np.eye(two_S + 1), first @ second)
        else:
            raise ValueError(f"unsupported rank-2 spin actor: {actor}")
        result += cg * component
    result.setflags(write=False)
    return result


def _clean_complex(value: complex) -> complex:
    cleaned = complex(np.real_if_close(value, tol=1000))
    return complex(cleaned.real if abs(cleaned.imag) < 1.0e-13 else cleaned)


@lru_cache(maxsize=32768)
def _isotropic_nuclear_scalar_factor(
    N: int,
    two_S: int,
    two_J: int,
    two_Jp: int,
    path: tuple[int, ...],
    pathp: tuple[int, ...],
    two_F: int,
    two_spins: tuple[int, ...],
    active_index: int,
    actor: str,
) -> complex:
    if actor not in {"electron", "rotation"}:
        raise ValueError("the isotropic nuclear actor must be electron or rotation")
    if (actor == "electron" and two_S == 0) or (actor == "rotation" and N == 0):
        return 0.0 + 0.0j
    left = _coupled_f_vector(N, two_S, two_J, path, two_F, two_spins)
    right = _coupled_f_vector(N, two_S, two_Jp, pathp, two_F, two_spins)
    nuclear_components = {
        q: _coupled_nuclear_rank1_component(two_spins, active_index, q)
        for q in (-1, 0, 1)
    }
    total = 0.0 + 0.0j
    for q in (-1, 0, 1):
        phase = -1.0 if (q & 1) else 1.0
        if actor == "electron":
            actor_component = _angular_momentum_rank1_component(two_S, q)
            contraction = sum(
                np.vdot(
                    left[n_index],
                    actor_component @ right[n_index] @ nuclear_components[-q].T,
                )
                for n_index in range(left.shape[0])
            )
        else:
            actor_component = _angular_momentum_rank1_component(2 * N, q)
            contraction = sum(
                np.vdot(
                    left[:, spin_index, :],
                    actor_component
                    @ right[:, spin_index, :]
                    @ nuclear_components[-q].T,
                )
                for spin_index in range(left.shape[1])
            )
        total += phase * contraction
    value = complex(np.real_if_close(total, tol=1000))
    return complex(value.real if abs(value.imag) < 1.0e-13 else value)


@lru_cache(maxsize=2048)
def quadrupole_body_block(
    N: int,
    Np: int,
    tensor_MHz: tuple[float, float, float, float, float, float],
    representation: str,
    two_I: int,
) -> np.ndarray:
    """Return the body-fixed K-space quadrupole block for one nucleus."""
    reduced = _pickett_quadrupole_reduced_spin_factor(two_I)
    if reduced == 0.0:
        block = np.zeros((2 * N + 1, 2 * Np + 1), dtype=complex)
        block.setflags(write=False)
        return block
    spherical = cartesian_rank2_spherical(tensor_MHz, representation)
    block = np.zeros((2 * N + 1, 2 * Np + 1), dtype=complex)
    for q, value in spherical.items():
        if value != 0.0:
            block += (value / reduced) * _rank2_rotational_body_component(N, Np, q)
    block.setflags(write=False)
    return block


@lru_cache(maxsize=4096)
def rank2_body_block(
    N: int,
    Np: int,
    tensor_MHz: tuple[float, float, float, float, float, float],
    representation: str,
) -> np.ndarray:
    """Return the body block of a generic symmetric traceless rank-2 tensor."""
    spherical = cartesian_rank2_spherical(tensor_MHz, representation)
    block = np.zeros((2 * N + 1, 2 * Np + 1), dtype=complex)
    for q, value in spherical.items():
        if value != 0.0:
            block += value * _rank2_rotational_body_component(N, Np, q)
    block.setflags(write=False)
    return block


@lru_cache(maxsize=4096)
def rank1_body_block(
    N: int,
    Np: int,
    vector_MHz: tuple[float, float, float],
    representation: str,
) -> np.ndarray:
    """Return the body block of a Cartesian a,b,c rank-1 vector."""
    values = np.asarray(vector_MHz, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("a finite three-component Cartesian vector is required")
    block = np.zeros((2 * N + 1, 2 * Np + 1), dtype=complex)
    for axis, value in zip(("a", "b", "c"), values, strict=True):
        if value != 0.0:
            block += value * dipole_transition_matrix(N, Np, axis, representation)
    block.setflags(write=False)
    return block


def cartesian_rank2_spherical(
    tensor: Sequence[float], representation: str = "Ir"
) -> dict[int, complex]:
    """Convert ``aa,bb,cc,ab,ac,bc`` to traceless rank-2 components."""
    values = np.asarray(tuple(tensor), dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("a finite six-component Cartesian tensor is required")
    aa, bb, cc, ab, ac, bc = values
    matrix = np.asarray(((aa, ab, ac), (ab, bb, bc), (ac, bc, cc)), dtype=float)
    matrix -= np.eye(3) * (float(np.trace(matrix)) / 3.0)
    rep = normalize_representation(representation)
    z_axis, x_axis, y_axis = (("c", "b", "a") if rep.endswith("l") else ("a", "b", "c"))
    axes = {"a": 0, "b": 1, "c": 2}
    iz, ix, iy = axes[z_axis], axes[x_axis], axes[y_axis]
    zz, xx, yy = matrix[iz, iz], matrix[ix, ix], matrix[iy, iy]
    zx, zy, xy = matrix[iz, ix], matrix[iz, iy], matrix[ix, iy]
    # Racah-normalized Cartesian components, scaled so that the conventional
    # spectroscopic axial component remains T20=(2Tzz-Txx-Tyy)/2.  Applying
    # the same sqrt(3/2) scale to all normalized spherical components is
    # essential: without it, rotations mixing diagonal and off-diagonal
    # Cartesian elements change the tensor norm and the Hamiltonian spectrum.
    transverse_scale = math.sqrt(3.0 / 2.0)
    return {
        -2: complex(0.5 * transverse_scale * ((xx - yy) - 2.0j * xy)),
        -1: complex(transverse_scale * (zx - 1.0j * zy)),
        0: complex(0.5 * (2.0 * zz - xx - yy)),
        1: complex(-transverse_scale * (zx + 1.0j * zy)),
        2: complex(0.5 * transverse_scale * ((xx - yy) + 2.0j * xy)),
    }


@lru_cache(maxsize=4096)
def _rank2_rotational_body_component(N: int, Np: int, q: int) -> np.ndarray:
    block = np.zeros((2 * N + 1, 2 * Np + 1), dtype=complex)
    prefactor = math.sqrt((2 * N + 1) * (2 * Np + 1))
    for row, K in enumerate(range(-N, N + 1)):
        Kp = K - q
        if abs(Kp) > Np:
            continue
        phase = -1.0 if ((N - K) & 1) else 1.0
        block[row, Kp + Np] = phase * prefactor * wigner_3j_doubled(
            2 * N, 4, 2 * Np, -2 * K, 2 * q, 2 * Kp
        )
    block.setflags(write=False)
    return block


@lru_cache(maxsize=32768)
def _quadrupole_scalar_factor(
    N: int,
    Np: int,
    two_S: int,
    two_J: int,
    two_Jp: int,
    path: tuple[int, ...],
    pathp: tuple[int, ...],
    two_F: int,
    two_spins: tuple[int, ...],
    active_index: int,
) -> complex:
    left = _coupled_f_vector(N, two_S, two_J, path, two_F, two_spins)
    right = _coupled_f_vector(Np, two_S, two_Jp, pathp, two_F, two_spins)
    components = {
        q: _coupled_nuclear_rank2_component(two_spins, active_index, q)
        for q in range(-2, 3)
    }
    total = 0.0 + 0.0j
    for p in range(-2, 3):
        phase = -1.0 if (p & 1) else 1.0
        rotational = _rank2_rotational_space_component(N, Np, p)
        total += phase * sum(
            np.vdot(
                left[:, spin_index, :],
                rotational @ right[:, spin_index, :] @ components[-p].T,
            )
            for spin_index in range(left.shape[1])
        )
    # Body blocks divide out the Pickett reduced spin factor, while the
    # nuclear tensor above restores it. Their product leaves a unit-reduced
    # Wigner tensor, which still needs the physical Cartesian chi scale.
    total *= _quadrupole_cartesian_normalization(two_spins[active_index])
    value = complex(np.real_if_close(total, tol=1000))
    return complex(value.real if abs(value.imag) < 1.0e-13 else value)


@lru_cache(maxsize=32768)
def _coupled_f_vector(
    N: int,
    two_S: int,
    two_J: int,
    path: tuple[int, ...],
    two_F: int,
    two_spins: tuple[int, ...],
) -> np.ndarray:
    two_I = path[-1] if path else 0
    _product, coupled, _transform = _nuclear_coupling_transform(two_spins)
    coupled_index = {state: index for index, state in enumerate(coupled)}
    vector = np.zeros((2 * N + 1, two_S + 1, len(coupled)), dtype=float)
    for n_index, two_MN in enumerate(range(-2 * N, 2 * N + 1, 2)):
        for s_index, two_MS in enumerate(range(-two_S, two_S + 1, 2)):
            two_MJ = two_MN + two_MS
            if abs(two_MJ) > two_J:
                continue
            two_MI = two_F - two_MJ
            nuclear_index = coupled_index.get((path, two_MI))
            if nuclear_index is None:
                continue
            coefficient = clebsch_gordan_doubled(
                2 * N, two_MN, two_S, two_MS, two_J, two_MJ
            )
            coefficient *= clebsch_gordan_doubled(
                two_J, two_MJ, two_I, two_MI, two_F, two_F
            )
            vector[n_index, s_index, nuclear_index] = coefficient
    norm = float(np.linalg.norm(vector))
    if abs(norm - 1.0) > 2.0e-12:
        raise RuntimeError(f"coupled F basis vector is not normalized: {norm:.16g}")
    vector.setflags(write=False)
    return vector


@lru_cache(maxsize=4096)
def _rank2_rotational_space_component(N: int, Np: int, p: int) -> np.ndarray:
    matrix = np.zeros((2 * N + 1, 2 * Np + 1), dtype=float)
    for row, two_MN in enumerate(range(-2 * N, 2 * N + 1, 2)):
        two_MNp = two_MN - 2 * p
        if abs(two_MNp) > 2 * Np:
            continue
        phase = -1.0 if (((2 * N - two_MN) // 2) & 1) else 1.0
        matrix[row, (two_MNp + 2 * Np) // 2] = phase * wigner_3j_doubled(
            2 * N, 4, 2 * Np, -two_MN, 2 * p, two_MNp
        )
    matrix.setflags(write=False)
    return matrix


@lru_cache(maxsize=4096)
def _rank1_rotational_space_component(N: int, Np: int, p: int) -> np.ndarray:
    matrix = np.zeros((2 * N + 1, 2 * Np + 1), dtype=float)
    for row, two_MN in enumerate(range(-2 * N, 2 * N + 1, 2)):
        two_MNp = two_MN - 2 * p
        if abs(two_MNp) > 2 * Np:
            continue
        phase = -1.0 if (((2 * N - two_MN) // 2) & 1) else 1.0
        matrix[row, (two_MNp + 2 * Np) // 2] = phase * wigner_3j_doubled(
            2 * N, 2, 2 * Np, -two_MN, 2 * p, two_MNp
        )
    matrix.setflags(write=False)
    return matrix


@lru_cache(maxsize=2048)
def _coupled_nuclear_rank2_component(
    two_spins: tuple[int, ...], active_index: int, q: int
) -> np.ndarray:
    if not 0 <= active_index < len(two_spins):
        raise IndexError("active nuclear-spin index is outside the coupling tree")
    product_states, _coupled_states, transform = _nuclear_coupling_transform(two_spins)
    target_spin = two_spins[active_index]
    single = _single_nuclear_rank2_component(target_spin, q)
    lookup = {state: index for index, state in enumerate(product_states)}
    product_operator = np.zeros((len(product_states), len(product_states)), dtype=complex)
    for row, state in enumerate(product_states):
        for target_Mp in range(-target_spin, target_spin + 1, 2):
            value = single[(state[active_index] + target_spin) // 2, (target_Mp + target_spin) // 2]
            if abs(value) <= 1.0e-15:
                continue
            statep = list(state)
            statep[active_index] = target_Mp
            column = lookup[tuple(statep)]
            product_operator[row, column] = value
    coupled = transform.T @ product_operator @ transform
    coupled.setflags(write=False)
    return coupled


@lru_cache(maxsize=2048)
def _coupled_nuclear_rank1_component(
    two_spins: tuple[int, ...], active_index: int, q: int
) -> np.ndarray:
    if not 0 <= active_index < len(two_spins):
        raise IndexError("active nuclear-spin index is outside the coupling tree")
    product_states, _coupled_states, transform = _nuclear_coupling_transform(two_spins)
    target_spin = two_spins[active_index]
    single = _angular_momentum_rank1_component(target_spin, q)
    lookup = {state: index for index, state in enumerate(product_states)}
    product_operator = np.zeros((len(product_states), len(product_states)), dtype=complex)
    for row, state in enumerate(product_states):
        for target_Mp in range(-target_spin, target_spin + 1, 2):
            value = single[
                (state[active_index] + target_spin) // 2,
                (target_Mp + target_spin) // 2,
            ]
            if abs(value) <= 1.0e-15:
                continue
            statep = list(state)
            statep[active_index] = target_Mp
            product_operator[row, lookup[tuple(statep)]] = value
    coupled = transform.T @ product_operator @ transform
    coupled.setflags(write=False)
    return coupled


@lru_cache(maxsize=256)
def _angular_momentum_rank1_component(two_j: int, q: int) -> np.ndarray:
    if q not in {-1, 0, 1}:
        raise ValueError("a rank-1 angular-momentum component requires q=-1,0,1")
    dimension = two_j + 1
    matrix = np.zeros((dimension, dimension), dtype=complex)
    j = 0.5 * two_j
    for two_mp in range(-two_j, two_j + 1, 2):
        column = (two_mp + two_j) // 2
        mp = 0.5 * two_mp
        if q == 0:
            matrix[column, column] = mp
        elif q == 1 and two_mp + 2 <= two_j:
            row = (two_mp + 2 + two_j) // 2
            matrix[row, column] = -math.sqrt(j * (j + 1.0) - mp * (mp + 1.0)) / math.sqrt(2.0)
        elif q == -1 and two_mp - 2 >= -two_j:
            row = (two_mp - 2 + two_j) // 2
            matrix[row, column] = math.sqrt(j * (j + 1.0) - mp * (mp - 1.0)) / math.sqrt(2.0)
    matrix.setflags(write=False)
    return matrix


@lru_cache(maxsize=256)
def _single_nuclear_rank2_component(two_I: int, q: int) -> np.ndarray:
    reduced = _pickett_quadrupole_reduced_spin_factor(two_I)
    dimension = two_I + 1
    matrix = np.zeros((dimension, dimension), dtype=complex)
    for two_M in range(-two_I, two_I + 1, 2):
        row = (two_M + two_I) // 2
        phase = -1.0 if (((two_I - two_M) // 2) & 1) else 1.0
        for two_Mp in range(-two_I, two_I + 1, 2):
            column = (two_Mp + two_I) // 2
            matrix[row, column] = phase * reduced * wigner_3j_doubled(
                two_I, 4, two_I, -two_M, 2 * q, two_Mp
            )
    matrix.setflags(write=False)
    return matrix


def _quadrupole_cartesian_normalization(two_I: int) -> float:
    """Normalize the unit-reduced nuclear tensor to spectroscopic chi.

    For an axial Cartesian tensor (chi, -chi/2, -chi/2), our body T20
    is 3 chi/2. The physical operator is
    chi [3 Iz**2 - I(I+1)] / [4 I(2I-1)], whose stretched-state
    expectation is chi/4 for every I >= 1. Thus the remaining unit-reduced
    spin tensor must be multiplied by 1/(6 * (I 2 I; -I 0 I)).
    This factor applies once to both rigid and LAM quadrupole contractions;
    generic rank-2 fine-structure tensors do not use this normalization.
    """
    if two_I < 2:
        return 0.0
    stretched = wigner_3j_doubled(two_I, 4, two_I, -two_I, 0, two_I)
    return 1.0 / (6.0 * stretched)


def _pickett_quadrupole_reduced_spin_factor(two_I: int) -> float:
    if two_I < 2:
        return 0.0
    I = 0.5 * two_I
    first = math.sqrt(I * (I + 1.0) * (two_I + 1.0))
    second = 0.25 * math.sqrt((I + 1.5) / (I - 0.5)) / I
    return first * second


@lru_cache(maxsize=None)
def wigner_3j_doubled(
    two_j1: int,
    two_j2: int,
    two_j3: int,
    two_m1: int,
    two_m2: int,
    two_m3: int,
) -> float:
    values = (two_j1, two_j2, two_j3, two_m1, two_m2, two_m3)
    if any(isinstance(value, bool) or int(value) != value for value in values):
        raise ValueError("Wigner quantum numbers must be doubled integers")
    if two_m1 + two_m2 + two_m3 != 0:
        return 0.0
    if any(
        two_j < 0 or abs(two_m) > two_j or ((two_j - two_m) & 1)
        for two_j, two_m in (
            (two_j1, two_m1),
            (two_j2, two_m2),
            (two_j3, two_m3),
        )
    ):
        return 0.0
    if (
        two_j3 < abs(two_j1 - two_j2)
        or two_j3 > two_j1 + two_j2
        or ((two_j1 + two_j2 + two_j3) & 1)
    ):
        return 0.0
    if _lock_native is not None and hasattr(_lock_native, "wigner_3j_doubled"):
        return float(_lock_native.wigner_3j_doubled(*values))

    triangle_a = (two_j1 + two_j2 - two_j3) // 2
    triangle_b = (two_j1 - two_j2 + two_j3) // 2
    triangle_c = (-two_j1 + two_j2 + two_j3) // 2
    triangle_sum = (two_j1 + two_j2 + two_j3) // 2
    jm = (
        (two_j1 + two_m1) // 2,
        (two_j1 - two_m1) // 2,
        (two_j2 + two_m2) // 2,
        (two_j2 - two_m2) // 2,
        (two_j3 + two_m3) // 2,
        (two_j3 - two_m3) // 2,
    )
    log_norm = 0.5 * (
        math.lgamma(triangle_a + 1.0)
        + math.lgamma(triangle_b + 1.0)
        + math.lgamma(triangle_c + 1.0)
        - math.lgamma(triangle_sum + 2.0)
        + sum(math.lgamma(value + 1.0) for value in jm)
    )

    bound_b = (two_j1 - two_m1) // 2
    bound_c = (two_j2 + two_m2) // 2
    offset_d = (two_j3 - two_j2 + two_m1) // 2
    offset_e = (two_j3 - two_j1 - two_m2) // 2
    z_min = max(0, -offset_d, -offset_e)
    z_max = min(triangle_a, bound_b, bound_c)
    if z_min > z_max:
        return 0.0
    logarithms: list[tuple[int, float]] = []
    for z in range(z_min, z_max + 1):
        denominator = (
            z,
            triangle_a - z,
            bound_b - z,
            bound_c - z,
            offset_d + z,
            offset_e + z,
        )
        logarithms.append(
            (z, -sum(math.lgamma(value + 1.0) for value in denominator))
        )
    maximum = max(value for _z, value in logarithms)
    positive = math.fsum(
        math.exp(value - maximum) for z, value in logarithms if not (z & 1)
    )
    negative = math.fsum(
        math.exp(value - maximum) for z, value in logarithms if z & 1
    )
    phase_exponent = (two_j1 - two_j2 - two_m3) // 2
    phase = -1.0 if (phase_exponent & 1) else 1.0
    return phase * math.exp(log_norm + maximum) * (positive - negative)


@lru_cache(maxsize=None)
def clebsch_gordan_doubled(
    two_j1: int,
    two_m1: int,
    two_j2: int,
    two_m2: int,
    two_J: int,
    two_M: int,
) -> float:
    if two_m1 + two_m2 != two_M:
        return 0.0
    exponent = (two_j1 - two_j2 + two_M) // 2
    phase = -1.0 if (exponent & 1) else 1.0
    return phase * math.sqrt(two_J + 1.0) * wigner_3j_doubled(
        two_j1, two_j2, two_J, two_m1, two_m2, -two_M
    )


@lru_cache(maxsize=None)
def wigner_6j_doubled(
    two_a: int,
    two_b: int,
    two_c: int,
    two_d: int,
    two_e: int,
    two_f: int,
) -> float:
    """Numerically stable Racah formula for a Wigner 6-j symbol."""
    values = (two_a, two_b, two_c, two_d, two_e, two_f)
    if any(isinstance(value, bool) or int(value) != value or value < 0 for value in values):
        raise ValueError("Wigner quantum numbers must be nonnegative doubled integers")
    triangles = ((two_a, two_b, two_c), (two_a, two_e, two_f),
                 (two_d, two_b, two_f), (two_d, two_e, two_c))
    if any(
        z < abs(x - y) or z > x + y or ((x + y + z) & 1)
        for x, y, z in triangles
    ):
        return 0.0
    if _lock_native is not None and hasattr(_lock_native, "wigner_6j_doubled"):
        return float(_lock_native.wigner_6j_doubled(*values))

    def log_delta(x: int, y: int, z: int) -> float:
        p = (x + y - z) // 2
        q = (x - y + z) // 2
        r = (-x + y + z) // 2
        s = (x + y + z) // 2
        return 0.5 * (
            math.lgamma(p + 1.0)
            + math.lgamma(q + 1.0)
            + math.lgamma(r + 1.0)
            - math.lgamma(s + 2.0)
        )

    x_values = (
        (two_a + two_b + two_c) // 2,
        (two_a + two_e + two_f) // 2,
        (two_d + two_b + two_f) // 2,
        (two_d + two_e + two_c) // 2,
    )
    y_values = (
        (two_a + two_b + two_d + two_e) // 2,
        (two_a + two_c + two_d + two_f) // 2,
        (two_b + two_c + two_e + two_f) // 2,
    )
    z_min, z_max = max(x_values), min(y_values)
    if z_min > z_max:
        return 0.0
    logarithms: list[tuple[int, float]] = []
    for z in range(z_min, z_max + 1):
        log_term = math.lgamma(z + 2.0)
        log_term -= sum(math.lgamma(z - x + 1.0) for x in x_values)
        log_term -= sum(math.lgamma(y - z + 1.0) for y in y_values)
        logarithms.append((z, log_term))
    maximum = max(value for _z, value in logarithms)
    signed = math.fsum(
        (-1.0 if (z & 1) else 1.0) * math.exp(value - maximum)
        for z, value in logarithms
    )
    log_prefactor = sum(log_delta(*triangle) for triangle in triangles)
    return math.exp(log_prefactor + maximum) * signed


def _validate_two_spins(two_spins: Sequence[int]) -> tuple[int, ...]:
    values = tuple(two_spins)
    if any(isinstance(value, bool) or int(value) != value or value < 0 for value in values):
        raise ValueError("doubled spins must be nonnegative integers")
    return tuple(int(value) for value in values)


__all__ = [
    "ElectronSpin",
    "NuclearSpin",
    "NuclearSpinSpinCoupling",
    "SPIN_HYPERFINE_SCHEMA",
    "SPIN_HYPERFINE_RESULT_SCHEMA",
    "SpinHyperfineBasisState",
    "SpinHyperfineBlock",
    "SpinHyperfineLevel",
    "SpinHyperfineModel",
    "SpinHyperfineResult",
    "SpinHyperfineTransition",
    "cartesian_rank2_spherical",
    "clebsch_gordan_doubled",
    "coupled_nuclear_spin_paths",
    "nuclear_coupling_transform",
    "quadrupole_body_block",
    "rank2_body_block",
    "read_spin_hyperfine_model",
    "read_spin_hyperfine_section",
    "parse_spin_hyperfine_section",
    "solve_spin_hyperfine_rotor",
    "solve_spin_hyperfine_lam_rotation",
    "spin_hyperfine_lam_transitions",
    "spin_hyperfine_lam_reduced_dipole_matrices",
    "spin_hyperfine_transitions",
    "spin_hyperfine_reduced_dipole_matrices",
    "spin_hyperfine_section_lines",
    "spin_hyperfine_native_available",
    "hellmann_feynman_diagonal",
    "write_spin_hyperfine_model",
    "write_spin_hyperfine_result",
    "write_spin_hyperfine_section",
    "wigner_3j_doubled",
    "wigner_6j_doubled",
]
