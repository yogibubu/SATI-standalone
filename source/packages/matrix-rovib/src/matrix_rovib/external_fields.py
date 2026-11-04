"""Collinear and arbitrary-vector Stark/Zeeman dressing of LOCK manifolds."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .native_rotor import MHZ_TO_KELVIN

from .spin_hyperfine import (
    SpinHyperfineResult,
    spin_hyperfine_reduced_dipole_matrices,
    wigner_3j_doubled,
)


EXTERNAL_FIELD_MODEL_SCHEMA = "matrix.lock.external_field_model.v1"
EXTERNAL_FIELD_RESULT_SCHEMA = "matrix.lock.external_field_result.v1"
NONCOLLINEAR_FIELD_RESULT_SCHEMA = "matrix.lock.noncollinear_field_result.v1"
DEBYE_V_CM_TO_MHZ = 3.33564e-30 * 100.0 / 6.62607015e-34 / 1.0e6
BOHR_MAGNETON_MHZ_T = 9.2740100783e-24 / 6.62607015e-34 / 1.0e6


@dataclass(frozen=True)
class ExternalFieldState:
    label: str
    energy_MHz: float
    two_F: int
    effective_g: float = 0.0
    assignment: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if (
            self.two_F < 0
            or not math.isfinite(self.energy_MHz)
            or not math.isfinite(self.effective_g)
        ):
            raise ValueError(
                "external-field states require finite energy, g, and nonnegative two_F"
            )


@dataclass(frozen=True)
class ExternalFieldCoupling:
    left: int
    right: int
    electric_reduced_D: complex = 0.0j
    magnetic_reduced_MHz_T: complex = 0.0j


@dataclass(frozen=True)
class ExternalFieldModel:
    states: tuple[ExternalFieldState, ...]
    couplings: tuple[ExternalFieldCoupling, ...] = ()

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("an external-field model requires at least one field-free state")
        for coupling in self.couplings:
            if not (
                0 <= coupling.left < len(self.states) and 0 <= coupling.right < len(self.states)
            ):
                raise ValueError("external-field coupling index is outside the state manifold")
            if coupling.left == coupling.right:
                raise ValueError(
                    "diagonal moments belong in effective_g or a parity-expanded basis"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXTERNAL_FIELD_MODEL_SCHEMA,
            "convention": "reduced spherical tensor, Condon-Shortley; arbitrary laboratory vectors",
            "states": [
                {
                    "label": state.label,
                    "energy_MHz": state.energy_MHz,
                    "two_F": state.two_F,
                    "effective_g": state.effective_g,
                    "assignment": state.assignment,
                }
                for state in self.states
            ],
            "couplings": [
                {
                    "left": item.left,
                    "right": item.right,
                    "electric_reduced_D": _complex_pair(item.electric_reduced_D),
                    "magnetic_reduced_MHz_T": _complex_pair(item.magnetic_reduced_MHz_T),
                }
                for item in self.couplings
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExternalFieldModel":
        if payload.get("schema") != EXTERNAL_FIELD_MODEL_SCHEMA:
            raise ValueError(f"unsupported external-field model schema: {payload.get('schema')!r}")
        states = tuple(
            ExternalFieldState(
                str(item["label"]),
                float(item["energy_MHz"]),
                int(item["two_F"]),
                float(item.get("effective_g", 0.0)),
                item.get("assignment"),
            )
            for item in payload.get("states", ())
        )
        couplings = tuple(
            ExternalFieldCoupling(
                int(item["left"]),
                int(item["right"]),
                _as_complex(item.get("electric_reduced_D", (0.0, 0.0))),
                _as_complex(item.get("magnetic_reduced_MHz_T", (0.0, 0.0))),
            )
            for item in payload.get("couplings", ())
        )
        return cls(states, couplings)


@dataclass(frozen=True)
class ExternalFieldBlock:
    two_M: int
    basis_states: tuple[int, ...]
    energies_MHz: np.ndarray
    vectors: np.ndarray
    parent_states: tuple[int, ...]
    parent_purities: tuple[float, ...]
    hermiticity_error_MHz: float


@dataclass(frozen=True)
class ExternalFieldResult:
    model: ExternalFieldModel
    electric_field_V_cm: float
    magnetic_field_T: float
    blocks: tuple[ExternalFieldBlock, ...]


@dataclass(frozen=True)
class NonCollinearFieldResult:
    """A field-dressed manifold in which M_F is allowed to mix."""

    model: ExternalFieldModel
    electric_field_V_cm: tuple[float, float, float]
    magnetic_field_T: tuple[float, float, float]
    basis: tuple[tuple[int, int], ...]
    energies_MHz: np.ndarray
    vectors: np.ndarray
    parent_states: tuple[int, ...]
    parent_two_M: tuple[int, ...]
    parent_purities: tuple[float, ...]
    hermiticity_error_MHz: float
    residual_norms_MHz: np.ndarray
    diagonalizer_backend: str


def _rank1_factor(
    two_F_left: int, two_M_left: int, two_F_right: int, two_M_right: int, q: int
) -> float:
    if -two_M_left + 2 * q + two_M_right != 0:
        return 0.0
    phase = -1.0 if ((two_F_left - two_M_left) // 2) % 2 else 1.0
    return phase * wigner_3j_doubled(two_F_left, 2, two_F_right, -two_M_left, 2 * q, two_M_right)


def solve_external_fields(
    model: ExternalFieldModel,
    *,
    electric_field_V_cm: float = 0.0,
    magnetic_field_T: float = 0.0,
) -> ExternalFieldResult:
    """Diagonalize collinear dc electric and magnetic fields in each M_F block."""
    if not math.isfinite(electric_field_V_cm) or not math.isfinite(magnetic_field_T):
        raise ValueError("external fields must be finite")
    maximum_two_F = max(state.two_F for state in model.states)
    blocks = []
    for two_M in range(-maximum_two_F, maximum_two_F + 1):
        basis = tuple(
            index
            for index, state in enumerate(model.states)
            if abs(two_M) <= state.two_F and (state.two_F - two_M) % 2 == 0
        )
        if not basis:
            continue
        location = {state: index for index, state in enumerate(basis)}
        hamiltonian = np.diag([model.states[index].energy_MHz for index in basis]).astype(complex)
        M = two_M / 2.0
        for position, state_index in enumerate(basis):
            state = model.states[state_index]
            hamiltonian[position, position] += (
                state.effective_g * BOHR_MAGNETON_MHZ_T * magnetic_field_T * M
            )
        for coupling in model.couplings:
            if coupling.left not in location or coupling.right not in location:
                continue
            left, right = model.states[coupling.left], model.states[coupling.right]
            angular = _rank1_factor(left.two_F, two_M, right.two_F, two_M, 0)
            value = (
                -electric_field_V_cm * DEBYE_V_CM_TO_MHZ * coupling.electric_reduced_D
                + magnetic_field_T * coupling.magnetic_reduced_MHz_T
            ) * angular
            i, j = location[coupling.left], location[coupling.right]
            hamiltonian[i, j] += value
            hamiltonian[j, i] += value.conjugate()
        hermiticity = float(np.max(np.abs(hamiltonian - hamiltonian.conj().T)))
        energies, vectors = np.linalg.eigh(hamiltonian)
        parents, purities = [], []
        for column in range(vectors.shape[1]):
            weights = np.abs(vectors[:, column]) ** 2
            parent_local = int(np.argmax(weights))
            parents.append(basis[parent_local])
            purities.append(float(weights[parent_local]))
        blocks.append(
            ExternalFieldBlock(
                two_M, basis, energies, vectors, tuple(parents), tuple(purities), hermiticity
            )
        )
    return ExternalFieldResult(
        model, float(electric_field_V_cm), float(magnetic_field_T), tuple(blocks)
    )


def external_field_hamiltonian(
    model: ExternalFieldModel,
    two_M: int,
    electric_field_V_cm: float = 0.0,
    magnetic_field_T: float = 0.0,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Build one conserved ``M_F`` block without diagonalizing it."""
    if not math.isfinite(electric_field_V_cm) or not math.isfinite(magnetic_field_T):
        raise ValueError("external fields must be finite")
    basis = tuple(
        index
        for index, state in enumerate(model.states)
        if abs(two_M) <= state.two_F and (state.two_F - two_M) % 2 == 0
    )
    location = {state: index for index, state in enumerate(basis)}
    hamiltonian = np.diag([model.states[index].energy_MHz for index in basis]).astype(complex)
    M = two_M / 2.0
    for position, state_index in enumerate(basis):
        state = model.states[state_index]
        hamiltonian[position, position] += (
            state.effective_g * BOHR_MAGNETON_MHZ_T * magnetic_field_T * M
        )
    for coupling in model.couplings:
        if coupling.left not in location or coupling.right not in location:
            continue
        left, right = model.states[coupling.left], model.states[coupling.right]
        angular = _rank1_factor(left.two_F, two_M, right.two_F, two_M, 0)
        value = (
            -electric_field_V_cm * DEBYE_V_CM_TO_MHZ * coupling.electric_reduced_D
            + magnetic_field_T * coupling.magnetic_reduced_MHz_T
        ) * angular
        i, j = location[coupling.left], location[coupling.right]
        hamiltonian[i, j] += value
        hamiltonian[j, i] += value.conjugate()
    return basis, hamiltonian


def external_field_model_from_spin_result(
    result: SpinHyperfineResult,
    dipole_debye: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    effective_g: Mapping[tuple[int, int], float] | None = None,
    electron_g: float = 2.00231930436,
    threshold_D: float = 1.0e-14,
    reduced_dipole_matrices: Mapping[tuple[int, int], np.ndarray] | None = None,
) -> ExternalFieldModel:
    """Promote a resolved LOCK calculation to a Stark/Zeeman model."""
    states = []
    locations = {}
    if not math.isfinite(electron_g):
        raise ValueError("electron_g must be finite")
    for block in result.blocks:
        automatic_g = _effective_electron_g(block, result.model.electron.two_S, electron_g)
        for index, (energy, level) in enumerate(zip(block.energies_MHz, block.levels, strict=True)):
            locations[(block.two_F, index)] = len(states)
            assignment = {
                "N": level.N,
                "two_J": level.two_J,
                "two_F": level.two_F,
                "Ka": level.Ka,
                "Kc": level.Kc,
                "nuclear_path": list(level.nuclear_path),
            }
            states.append(
                ExternalFieldState(
                    f"F={block.two_F}/2 level={index}",
                    float(energy),
                    block.two_F,
                    automatic_g[index]
                    if effective_g is None
                    else float(effective_g.get((block.two_F, index), automatic_g[index])),
                    assignment,
                )
            )
    couplings = []
    reduced = (
        spin_hyperfine_reduced_dipole_matrices(result, dipole_debye)
        if reduced_dipole_matrices is None
        else reduced_dipole_matrices
    )
    for (left_f, right_f), matrix in reduced.items():
        for il in range(matrix.shape[0]):
            start = il + 1 if left_f == right_f else 0
            for ir in range(start, matrix.shape[1]):
                value = complex(matrix[il, ir])
                if abs(value) > threshold_D:
                    couplings.append(
                        ExternalFieldCoupling(
                            locations[(left_f, il)], locations[(right_f, ir)], value, 0.0j
                        )
                    )
    return ExternalFieldModel(tuple(states), tuple(couplings))


def _effective_electron_g(block: Any, two_S: int, electron_g: float) -> np.ndarray:
    """Weak-field diagonal electron-spin g from the actual mixed eigenvectors."""
    F = block.two_F / 2.0
    diagonal = np.zeros(len(block.basis))
    for index, basis in enumerate(block.basis):
        J, S, N, I = basis.two_J / 2.0, two_S / 2.0, float(basis.N), basis.two_I / 2.0
        if J == 0.0 or F == 0.0:
            continue
        g_J = electron_g * (J * (J + 1.0) + S * (S + 1.0) - N * (N + 1.0)) / (2.0 * J * (J + 1.0))
        diagonal[index] = (
            g_J * (F * (F + 1.0) + J * (J + 1.0) - I * (I + 1.0)) / (2.0 * F * (F + 1.0))
        )
    return np.real(np.einsum("is,i,is->s", block.vectors.conj(), diagonal, block.vectors))


def read_external_field_model(path: Path | str) -> ExternalFieldModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("an external-field model must be a JSON object")
    return ExternalFieldModel.from_dict(payload)


def write_external_field_model(path: Path | str, model: ExternalFieldModel) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target


def write_external_field_result(path: Path | str, result: ExternalFieldResult) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": EXTERNAL_FIELD_RESULT_SCHEMA,
        "model_schema": EXTERNAL_FIELD_MODEL_SCHEMA,
        "electric_field_V_cm": result.electric_field_V_cm,
        "magnetic_field_T": result.magnetic_field_T,
        "blocks": [
            {
                "two_M": block.two_M,
                "dimension": len(block.energies_MHz),
                "hermiticity_error_MHz": block.hermiticity_error_MHz,
                "levels": [
                    {
                        "energy_MHz": float(energy),
                        "parent_state": parent,
                        "parent_label": result.model.states[parent].label,
                        "parent_purity": purity,
                    }
                    for energy, parent, purity in zip(
                        block.energies_MHz, block.parent_states, block.parent_purities, strict=True
                    )
                ],
            }
            for block in result.blocks
        ],
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def external_field_transitions(
    result: ExternalFieldResult,
    *,
    temperature_K: float | None = None,
    intensity_cut: float = 0.0,
) -> list[dict[str, Any]]:
    """Return polarized E1 transitions between field-dressed eigenstates."""
    if temperature_K is not None and temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    if intensity_cut < 0.0:
        raise ValueError("intensity_cut must be nonnegative")
    blocks = result.blocks
    ground = min(float(block.energies_MHz[0]) for block in blocks)
    beta = None if temperature_K is None else MHZ_TO_KELVIN / temperature_K
    partition = 1.0
    if beta is not None:
        partition = math.fsum(
            math.exp(-(float(energy) - ground) * beta)
            for block in blocks
            for energy in block.energies_MHz
        )
    rows: list[dict[str, Any]] = []
    for left_index, left_block in enumerate(blocks):
        for right_index in range(left_index, len(blocks)):
            right_block = blocks[right_index]
            for q in (-1, 0, 1):
                operator = _field_dipole_operator(result.model, left_block, right_block, q)
                if not np.any(np.abs(operator) > 1.0e-15):
                    continue
                amplitudes = left_block.vectors.conj().T @ operator @ right_block.vectors
                for il, left_energy in enumerate(left_block.energies_MHz):
                    start = il + 1 if left_index == right_index else 0
                    for ir in range(start, len(right_block.energies_MHz)):
                        right_energy = float(right_block.energies_MHz[ir])
                        first_energy = float(left_energy)
                        strength = float(abs(amplitudes[il, ir]) ** 2)
                        if strength <= 1.0e-18 or abs(right_energy - first_energy) <= 1.0e-9:
                            continue
                        if first_energy < right_energy:
                            low_block, low_level = left_block, il
                            up_block, up_level = right_block, ir
                            low_energy, up_energy, polarization = first_energy, right_energy, q
                        else:
                            low_block, low_level = right_block, ir
                            up_block, up_level = left_block, il
                            low_energy, up_energy, polarization = right_energy, first_energy, -q
                        frequency = up_energy - low_energy
                        intensity = strength
                        if beta is not None:
                            intensity *= (
                                frequency
                                * (-math.expm1(-frequency * beta))
                                * math.exp(-(low_energy - ground) * beta)
                                / partition
                            )
                        if intensity < intensity_cut:
                            continue
                        rows.append(
                            {
                                "frequency_MHz": frequency,
                                "line_strength_D2": strength,
                                "intensity": intensity,
                                "polarization_q": polarization,
                                "lower_two_M": low_block.two_M,
                                "lower_level": low_level,
                                "lower_parent": low_block.parent_states[low_level],
                                "upper_two_M": up_block.two_M,
                                "upper_level": up_level,
                                "upper_parent": up_block.parent_states[up_level],
                            }
                        )
    maximum = max((row["intensity"] for row in rows), default=1.0)
    for row in rows:
        row["relative_intensity"] = 100.0 * row["intensity"] / maximum
    return sorted(rows, key=lambda row: (row["frequency_MHz"], row["polarization_q"]))


def _field_dipole_operator(
    model: ExternalFieldModel,
    left_block: ExternalFieldBlock,
    right_block: ExternalFieldBlock,
    q: int,
) -> np.ndarray:
    left_location = {state: index for index, state in enumerate(left_block.basis_states)}
    right_location = {state: index for index, state in enumerate(right_block.basis_states)}
    operator = np.zeros((len(left_location), len(right_location)), dtype=complex)
    for coupling in model.couplings:
        if coupling.left in left_location and coupling.right in right_location:
            left = model.states[coupling.left]
            right = model.states[coupling.right]
            angular = _rank1_factor(left.two_F, left_block.two_M, right.two_F, right_block.two_M, q)
            operator[left_location[coupling.left], right_location[coupling.right]] += (
                angular * coupling.electric_reduced_D
            )
        if coupling.right in left_location and coupling.left in right_location:
            left = model.states[coupling.right]
            right = model.states[coupling.left]
            phase = -1.0 if ((left.two_F - right.two_F) // 2) % 2 else 1.0
            reduced = phase * coupling.electric_reduced_D.conjugate()
            angular = _rank1_factor(left.two_F, left_block.two_M, right.two_F, right_block.two_M, q)
            operator[left_location[coupling.right], right_location[coupling.left]] += (
                angular * reduced
            )
    return operator


def _finite_vector3(value: Sequence[float], label: str) -> tuple[float, float, float]:
    vector = tuple(float(component) for component in value)
    if len(vector) != 3 or not all(math.isfinite(component) for component in vector):
        raise ValueError(f"{label} must contain three finite Cartesian components")
    return vector


def _angular_momentum_cartesian(two_F: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    F = two_F / 2.0
    two_m = np.arange(-two_F, two_F + 1, 2, dtype=int)
    raising = np.zeros((len(two_m), len(two_m)), dtype=complex)
    for index, value in enumerate(two_m[:-1]):
        m = value / 2.0
        raising[index + 1, index] = math.sqrt(max(0.0, F * (F + 1.0) - m * (m + 1.0)))
    lowering = raising.conj().T
    return (
        0.5 * (raising + lowering),
        (raising - lowering) / (2.0j),
        np.diag(two_m.astype(float) / 2.0),
    )


def _spherical_vector(vector: Sequence[float]) -> dict[int, complex]:
    x, y, z = _finite_vector3(vector, "field vector")
    root2 = math.sqrt(2.0)
    return {-1: (x - 1.0j * y) / root2, 0: complex(z), 1: -(x + 1.0j * y) / root2}


def _rank1_field_contraction(
    field: Mapping[int, complex],
    left: ExternalFieldState,
    two_M_left: int,
    right: ExternalFieldState,
    two_M_right: int,
    reduced: complex,
) -> complex:
    value = 0.0j
    for q in (-1, 0, 1):
        angular = _rank1_factor(left.two_F, two_M_left, right.two_F, two_M_right, q)
        value += ((-1.0) if q else 1.0) * field[-q] * angular * reduced
    return value


def noncollinear_external_field_hamiltonian(
    model: ExternalFieldModel,
    *,
    electric_field_V_cm: Sequence[float] = (0.0, 0.0, 0.0),
    magnetic_field_T: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    """Build the full Stark--Zeeman Hamiltonian for arbitrary field vectors."""

    electric = _finite_vector3(electric_field_V_cm, "electric_field_V_cm")
    magnetic = _finite_vector3(magnetic_field_T, "magnetic_field_T")
    basis = tuple(
        (state_index, two_M)
        for state_index, state in enumerate(model.states)
        for two_M in range(-state.two_F, state.two_F + 1, 2)
    )
    location = {state: index for index, state in enumerate(basis)}
    matrix = np.zeros((len(basis), len(basis)), dtype=complex)
    for state_index, state in enumerate(model.states):
        indices = [
            location[(state_index, two_M)] for two_M in range(-state.two_F, state.two_F + 1, 2)
        ]
        matrix[indices, indices] = state.energy_MHz
        if state.effective_g != 0.0 and any(magnetic):
            components = _angular_momentum_cartesian(state.two_F)
            zeeman = (
                state.effective_g
                * BOHR_MAGNETON_MHZ_T
                * sum(
                    component * field for component, field in zip(components, magnetic, strict=True)
                )
            )
            matrix[np.ix_(indices, indices)] += zeeman

    electric_spherical = _spherical_vector(electric)
    magnetic_spherical = _spherical_vector(magnetic)
    for coupling in model.couplings:
        left, right = model.states[coupling.left], model.states[coupling.right]
        for two_M_left in range(-left.two_F, left.two_F + 1, 2):
            for two_M_right in range(-right.two_F, right.two_F + 1, 2):
                value = -DEBYE_V_CM_TO_MHZ * _rank1_field_contraction(
                    electric_spherical,
                    left,
                    two_M_left,
                    right,
                    two_M_right,
                    coupling.electric_reduced_D,
                )
                value += _rank1_field_contraction(
                    magnetic_spherical,
                    left,
                    two_M_left,
                    right,
                    two_M_right,
                    coupling.magnetic_reduced_MHz_T,
                )
                if abs(value) <= 1.0e-18:
                    continue
                i, j = (
                    location[(coupling.left, two_M_left)],
                    location[(coupling.right, two_M_right)],
                )
                matrix[i, j] += value
                matrix[j, i] += value.conjugate()
    return basis, matrix


def solve_noncollinear_external_fields(
    model: ExternalFieldModel,
    *,
    electric_field_V_cm: Sequence[float] = (0.0, 0.0, 0.0),
    magnetic_field_T: Sequence[float] = (0.0, 0.0, 0.0),
    level_count: int | None = None,
    diagonalizer_backend: str | None = None,
) -> NonCollinearFieldResult:
    """Diagonalize arbitrary dc fields after expanding all M_F projections."""

    from .adaptive_solver import EigenproblemBlock, solve_eigenproblem

    electric = _finite_vector3(electric_field_V_cm, "electric_field_V_cm")
    magnetic = _finite_vector3(magnetic_field_T, "magnetic_field_T")
    basis, matrix = noncollinear_external_field_hamiltonian(
        model,
        electric_field_V_cm=electric,
        magnetic_field_T=magnetic,
    )
    retained = len(basis) if level_count is None else int(level_count)
    if not 1 <= retained <= len(basis):
        raise ValueError("non-collinear field level_count is outside the basis")
    solved = solve_eigenproblem(
        EigenproblemBlock("noncollinear-fields", len(basis), retained, matrix=matrix),
        diagonalizer_backend=diagonalizer_backend,
    )
    parent_states = []
    parent_two_m = []
    purities = []
    for column in range(solved.eigenvectors.shape[1]):
        weights = np.abs(solved.eigenvectors[:, column]) ** 2
        parent = int(np.argmax(weights))
        parent_states.append(basis[parent][0])
        parent_two_m.append(basis[parent][1])
        purities.append(float(weights[parent]))
    error = float(np.max(np.abs(matrix - matrix.conj().T), initial=0.0))
    return NonCollinearFieldResult(
        model,
        electric,
        magnetic,
        basis,
        solved.eigenvalues,
        solved.eigenvectors,
        tuple(parent_states),
        tuple(parent_two_m),
        tuple(purities),
        error,
        solved.residual_norms,
        solved.backend,
    )


def noncollinear_field_transitions(
    result: NonCollinearFieldResult,
    *,
    polarization_lab: Sequence[float] = (0.0, 0.0, 1.0),
    temperature_K: float | None = None,
    intensity_cut: float = 0.0,
) -> list[dict[str, Any]]:
    """Return polarized electric-dipole strengths between fully mixed levels."""

    if temperature_K is not None and temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    if intensity_cut < 0.0:
        raise ValueError("intensity_cut must be nonnegative")
    polarization = _spherical_vector(polarization_lab)
    basis = result.basis
    location = {item: index for index, item in enumerate(basis)}
    operator = np.zeros((len(basis), len(basis)), dtype=complex)
    for coupling in result.model.couplings:
        left, right = result.model.states[coupling.left], result.model.states[coupling.right]
        for two_M_left in range(-left.two_F, left.two_F + 1, 2):
            for two_M_right in range(-right.two_F, right.two_F + 1, 2):
                value = _rank1_field_contraction(
                    polarization,
                    left,
                    two_M_left,
                    right,
                    two_M_right,
                    coupling.electric_reduced_D,
                )
                i, j = (
                    location[(coupling.left, two_M_left)],
                    location[(coupling.right, two_M_right)],
                )
                operator[i, j] += value
                operator[j, i] += value.conjugate()
    mixed = result.vectors.conj().T @ operator @ result.vectors
    ground = float(result.energies_MHz[0])
    beta = None if temperature_K is None else MHZ_TO_KELVIN / temperature_K
    partition = (
        1.0
        if beta is None
        else math.fsum(math.exp(-(float(energy) - ground) * beta) for energy in result.energies_MHz)
    )
    rows = []
    for lower in range(len(result.energies_MHz)):
        for upper in range(lower + 1, len(result.energies_MHz)):
            strength = float(abs(mixed[lower, upper]) ** 2)
            frequency = float(result.energies_MHz[upper] - result.energies_MHz[lower])
            intensity = strength
            if beta is not None:
                intensity *= (
                    frequency
                    * (-math.expm1(-frequency * beta))
                    * math.exp(-(float(result.energies_MHz[lower]) - ground) * beta)
                    / partition
                )
            if intensity < intensity_cut:
                continue
            rows.append(
                {
                    "frequency_MHz": frequency,
                    "line_strength_D2": strength,
                    "intensity": intensity,
                    "lower_level": lower,
                    "upper_level": upper,
                    "lower_parent": result.parent_states[lower],
                    "upper_parent": result.parent_states[upper],
                }
            )
    maximum = max((row["intensity"] for row in rows), default=1.0)
    for row in rows:
        row["relative_intensity"] = 100.0 * row["intensity"] / maximum
    return sorted(rows, key=lambda row: row["frequency_MHz"])


def write_noncollinear_field_result(path: Path | str, result: NonCollinearFieldResult) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": NONCOLLINEAR_FIELD_RESULT_SCHEMA,
        "electric_field_V_cm": list(result.electric_field_V_cm),
        "magnetic_field_T": list(result.magnetic_field_T),
        "basis": [{"state": state, "two_M": two_M} for state, two_M in result.basis],
        "diagnostics": {
            "dimension": len(result.basis),
            "hermiticity_error_MHz": result.hermiticity_error_MHz,
            "maximum_residual_MHz": float(np.max(result.residual_norms_MHz, initial=0.0)),
            "diagonalizer_backend": result.diagonalizer_backend,
        },
        "levels": [
            {
                "energy_MHz": float(energy),
                "parent_state": parent,
                "parent_two_M": two_M,
                "parent_purity": purity,
            }
            for energy, parent, two_M, purity in zip(
                result.energies_MHz,
                result.parent_states,
                result.parent_two_M,
                result.parent_purities,
                strict=True,
            )
        ],
    }
    target.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return target


def write_external_field_transitions(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ())
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return target


def _complex_pair(value: complex) -> list[float]:
    return [float(complex(value).real), float(complex(value).imag)]


def _as_complex(value: Any) -> complex:
    if isinstance(value, (int, float)):
        return complex(value)
    if isinstance(value, Sequence) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    raise ValueError("a reduced moment must be a number or [real, imaginary]")


__all__ = [
    "EXTERNAL_FIELD_MODEL_SCHEMA",
    "EXTERNAL_FIELD_RESULT_SCHEMA",
    "DEBYE_V_CM_TO_MHZ",
    "NONCOLLINEAR_FIELD_RESULT_SCHEMA",
    "BOHR_MAGNETON_MHZ_T",
    "ExternalFieldState",
    "ExternalFieldCoupling",
    "ExternalFieldModel",
    "ExternalFieldBlock",
    "ExternalFieldResult",
    "NonCollinearFieldResult",
    "solve_external_fields",
    "solve_noncollinear_external_fields",
    "external_field_hamiltonian",
    "noncollinear_external_field_hamiltonian",
    "noncollinear_field_transitions",
    "external_field_model_from_spin_result",
    "read_external_field_model",
    "write_external_field_model",
    "write_external_field_result",
    "write_noncollinear_field_result",
    "external_field_transitions",
    "write_external_field_transitions",
]
