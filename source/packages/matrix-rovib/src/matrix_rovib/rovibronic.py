"""Contracted multi-electronic Hund-case-(a) effective Hamiltonians."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .native_rotor import MHZ_TO_KELVIN
from .spin_hyperfine import wigner_3j_doubled


ROVIBRONIC_MODEL_SCHEMA = "matrix.lock.rovibronic_model.v1"
ROVIBRONIC_RESULT_SCHEMA = "matrix.lock.rovibronic_result.v1"
_KINDS = {"spin-orbit", "lambda-doubling", "Renner-Teller", "Coriolis", "other"}
_J_DEPENDENCE = {"constant", "J(J+1)", "case-a-ladder"}


@dataclass(frozen=True)
class RovibronicBasisState:
    label: str
    electronic_label: str
    vibronic_label: str
    term_energy_MHz: float
    two_Lambda: int
    two_Sigma: int
    two_Omega: int
    parity: int
    B_MHz: float
    D_MHz: float = 0.0
    spin_orbit_A_MHz: float = 0.0

    def __post_init__(self) -> None:
        if self.parity not in {-1, 1}:
            raise ValueError("rovibronic parity must be -1 or +1")
        if (self.two_Lambda + self.two_Sigma - self.two_Omega) % 2:
            raise ValueError("Lambda, Sigma and Omega doubled quantum numbers have inconsistent parity")
        values = (self.term_energy_MHz, self.B_MHz, self.D_MHz, self.spin_orbit_A_MHz)
        if not all(math.isfinite(value) for value in values) or self.B_MHz <= 0.0:
            raise ValueError("rovibronic energies and constants must be finite with B > 0")


@dataclass(frozen=True)
class RovibronicInteraction:
    left: int
    right: int
    value_MHz: complex
    kind: str
    j_dependence: str = "constant"
    parity_factor: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unknown rovibronic interaction kind: {self.kind}")
        if self.j_dependence not in _J_DEPENDENCE:
            raise ValueError(f"unknown J dependence: {self.j_dependence}")
        if not math.isfinite(complex(self.value_MHz).real) or not math.isfinite(complex(self.value_MHz).imag):
            raise ValueError("rovibronic interaction values must be finite")


@dataclass(frozen=True)
class RovibronicTransitionMoment:
    left: int
    right: int
    body_moment_D: complex


@dataclass(frozen=True)
class RovibronicModel:
    states: tuple[RovibronicBasisState, ...]
    interactions: tuple[RovibronicInteraction, ...] = ()
    transition_moments: tuple[RovibronicTransitionMoment, ...] = ()

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("a rovibronic model requires contracted basis states")
        labels = [state.label for state in self.states]
        if len(labels) != len(set(labels)):
            raise ValueError("rovibronic state labels must be unique")
        for item in self.interactions:
            if not (0 <= item.left < len(self.states) and 0 <= item.right < len(self.states)):
                raise ValueError("rovibronic interaction index is outside the basis")
            if self.states[item.left].parity != self.states[item.right].parity:
                raise ValueError("a field-free rovibronic interaction must preserve parity")
            if item.parity_factor and item.left != item.right:
                raise ValueError("parity_factor is defined for diagonal effective terms")
        for item in self.transition_moments:
            if not (0 <= item.left < len(self.states) and 0 <= item.right < len(self.states)):
                raise ValueError("rovibronic transition moment index is outside the basis")
            delta = self.states[item.left].two_Omega - self.states[item.right].two_Omega
            if delta not in {-2, 0, 2}:
                raise ValueError("a rank-one body moment requires Delta Omega = 0, +/-1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ROVIBRONIC_MODEL_SCHEMA,
            "basis": "contracted parity-adapted Hund case (a)",
            "states": [state.__dict__ for state in self.states],
            "interactions": [
                {
                    "left": item.left, "right": item.right,
                    "value_MHz": [complex(item.value_MHz).real, complex(item.value_MHz).imag],
                    "kind": item.kind, "j_dependence": item.j_dependence,
                    "parity_factor": item.parity_factor,
                }
                for item in self.interactions
            ],
            "transition_moments": [
                {
                    "left": item.left,
                    "right": item.right,
                    "body_moment_D": [
                        complex(item.body_moment_D).real,
                        complex(item.body_moment_D).imag,
                    ],
                }
                for item in self.transition_moments
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RovibronicModel":
        if payload.get("schema") != ROVIBRONIC_MODEL_SCHEMA:
            raise ValueError(f"unsupported rovibronic schema: {payload.get('schema')!r}")
        states = tuple(
            RovibronicBasisState(
                label=str(item["label"]), electronic_label=str(item["electronic_label"]),
                vibronic_label=str(item["vibronic_label"]),
                term_energy_MHz=float(item["term_energy_MHz"]),
                two_Lambda=int(item["two_Lambda"]), two_Sigma=int(item["two_Sigma"]),
                two_Omega=int(item["two_Omega"]), parity=int(item["parity"]),
                B_MHz=float(item["B_MHz"]), D_MHz=float(item.get("D_MHz", 0.0)),
                spin_orbit_A_MHz=float(item.get("spin_orbit_A_MHz", 0.0)),
            )
            for item in payload.get("states", ())
        )
        interactions = tuple(
            RovibronicInteraction(
                int(item["left"]), int(item["right"]), _as_complex(item["value_MHz"]),
                str(item["kind"]), str(item.get("j_dependence", "constant")),
                bool(item.get("parity_factor", False)),
            )
            for item in payload.get("interactions", ())
        )
        moments = tuple(
            RovibronicTransitionMoment(
                int(item["left"]), int(item["right"]), _as_complex(item["body_moment_D"])
            )
            for item in payload.get("transition_moments", ())
        )
        return cls(states, interactions, moments)


@dataclass(frozen=True)
class RovibronicLevel:
    energy_MHz: float
    two_J: int
    parity: int
    parent_state: int
    parent_label: str
    electronic_label: str
    vibronic_label: str
    parent_purity: float


@dataclass(frozen=True)
class RovibronicBlock:
    two_J: int
    parity: int
    basis_states: tuple[int, ...]
    energies_MHz: np.ndarray
    vectors: np.ndarray
    levels: tuple[RovibronicLevel, ...]
    hermiticity_error_MHz: float


@dataclass(frozen=True)
class RovibronicResult:
    model: RovibronicModel
    blocks: tuple[RovibronicBlock, ...]


def solve_rovibronic(
    model: RovibronicModel,
    *,
    two_j_values: Sequence[int] | range,
    levels_per_block: int | None = None,
) -> RovibronicResult:
    """Diagonalize each conserved ``(J, parity)`` effective-Hamiltonian block."""
    blocks = []
    for two_J in sorted({int(value) for value in two_j_values}):
        if two_J < 0:
            raise ValueError("two_J must be nonnegative")
        J = two_J / 2.0
        for parity in (-1, 1):
            basis = tuple(
                index for index, state in enumerate(model.states)
                if state.parity == parity and abs(state.two_Omega) <= two_J
                and (two_J - state.two_Omega) % 2 == 0
            )
            if not basis:
                continue
            location = {state: index for index, state in enumerate(basis)}
            hamiltonian = np.zeros((len(basis), len(basis)), dtype=complex)
            for local, state_index in enumerate(basis):
                state = model.states[state_index]
                omega = state.two_Omega / 2.0
                rotation = J * (J + 1.0) - omega * omega
                spin_orbit = state.spin_orbit_A_MHz * state.two_Lambda * state.two_Sigma / 4.0
                hamiltonian[local, local] = (
                    state.term_energy_MHz + spin_orbit
                    + state.B_MHz * rotation - state.D_MHz * rotation * rotation
                )
            for item in model.interactions:
                if item.left not in location or item.right not in location:
                    continue
                factor = _interaction_factor(item, model.states, J)
                value = complex(item.value_MHz) * factor
                i, j = location[item.left], location[item.right]
                if i == j:
                    hamiltonian[i, i] += value.real
                else:
                    hamiltonian[i, j] += value
                    hamiltonian[j, i] += value.conjugate()
            hermiticity = float(np.max(np.abs(hamiltonian - hamiltonian.conj().T)))
            energies, vectors = np.linalg.eigh(hamiltonian)
            if levels_per_block is not None:
                if levels_per_block < 1:
                    raise ValueError("levels_per_block must be positive")
                energies, vectors = energies[:levels_per_block], vectors[:, :levels_per_block]
            levels = []
            for column, energy in enumerate(energies):
                weights = np.abs(vectors[:, column]) ** 2
                local_parent = int(np.argmax(weights))
                parent = basis[local_parent]
                state = model.states[parent]
                levels.append(RovibronicLevel(
                    float(energy), two_J, parity, parent, state.label, state.electronic_label,
                    state.vibronic_label, float(weights[local_parent]),
                ))
            blocks.append(
                RovibronicBlock(two_J, parity, basis, energies, vectors, tuple(levels), hermiticity)
            )
    return RovibronicResult(model, tuple(blocks))


def rovibronic_hamiltonian(
    model: RovibronicModel, two_J: int, parity: int
) -> tuple[tuple[int, ...], np.ndarray]:
    """Build one conserved ``(J, parity)`` block without diagonalizing it."""
    if two_J < 0 or parity not in {-1, 1}:
        raise ValueError("two_J must be nonnegative and parity must be -1 or +1")
    J = two_J / 2.0
    basis = tuple(
        index for index, state in enumerate(model.states)
        if state.parity == parity and abs(state.two_Omega) <= two_J
        and (two_J - state.two_Omega) % 2 == 0
    )
    location = {state: index for index, state in enumerate(basis)}
    hamiltonian = np.zeros((len(basis), len(basis)), dtype=complex)
    for local, state_index in enumerate(basis):
        state = model.states[state_index]
        omega = state.two_Omega / 2.0
        rotation = J * (J + 1.0) - omega * omega
        spin_orbit = state.spin_orbit_A_MHz * state.two_Lambda * state.two_Sigma / 4.0
        hamiltonian[local, local] = (
            state.term_energy_MHz + spin_orbit
            + state.B_MHz * rotation - state.D_MHz * rotation * rotation
        )
    for item in model.interactions:
        if item.left not in location or item.right not in location:
            continue
        factor = _interaction_factor(item, model.states, J)
        value = complex(item.value_MHz) * factor
        i, j = location[item.left], location[item.right]
        if i == j:
            hamiltonian[i, i] += value.real
        else:
            hamiltonian[i, j] += value
            hamiltonian[j, i] += value.conjugate()
    return basis, hamiltonian


def _interaction_factor(
    item: RovibronicInteraction,
    states: tuple[RovibronicBasisState, ...],
    J: float,
) -> float:
    factor = 1.0
    if item.j_dependence == "J(J+1)":
        factor = J * (J + 1.0)
    elif item.j_dependence == "case-a-ladder":
        left, right = states[item.left], states[item.right]
        if abs(left.two_Omega - right.two_Omega) != 2:
            raise ValueError("case-a-ladder requires Delta Omega = +/-1")
        omega_left, omega_right = left.two_Omega / 2.0, right.two_Omega / 2.0
        argument = J * (J + 1.0) - omega_left * omega_right
        if argument < -1.0e-12:
            raise ValueError("case-a ladder factor is outside the selected J manifold")
        factor = math.sqrt(max(0.0, argument))
    if item.parity_factor:
        factor *= states[item.left].parity
    return factor


def read_rovibronic_model(path: Path | str) -> RovibronicModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("a rovibronic model must be a JSON object")
    return RovibronicModel.from_dict(payload)


def write_rovibronic_model(path: Path | str, model: RovibronicModel) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target


def write_rovibronic_result(path: Path | str, result: RovibronicResult) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": ROVIBRONIC_RESULT_SCHEMA,
        "model_schema": ROVIBRONIC_MODEL_SCHEMA,
        "blocks": [
            {
                "two_J": block.two_J, "parity": block.parity,
                "dimension": len(block.energies_MHz),
                "basis_states": list(block.basis_states),
                "hermiticity_error_MHz": block.hermiticity_error_MHz,
                "levels": [level.__dict__ for level in block.levels],
            }
            for block in result.blocks
        ],
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def rovibronic_transitions(
    result: RovibronicResult,
    *,
    temperature_K: float | None = None,
    intensity_cut: float = 0.0,
) -> list[dict[str, Any]]:
    """Return mixed case-(a) E1 bands from explicit contracted body moments."""
    if temperature_K is not None and temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    if intensity_cut < 0.0:
        raise ValueError("intensity_cut must be nonnegative")
    ground = min(float(block.energies_MHz[0]) for block in result.blocks)
    beta = None if temperature_K is None else MHZ_TO_KELVIN / temperature_K
    partition = 1.0
    if beta is not None:
        partition = math.fsum(
            (block.two_J + 1.0) * math.exp(-(float(energy) - ground) * beta)
            for block in result.blocks for energy in block.energies_MHz
        )
    rows: list[dict[str, Any]] = []
    for lower_index, lower in enumerate(result.blocks):
        for upper in result.blocks[lower_index:]:
            if abs(lower.two_J - upper.two_J) > 2 or lower.two_J == upper.two_J == 0:
                continue
            operator = _rovibronic_reduced_dipole(result.model, lower, upper)
            if not np.any(np.abs(operator) > 1.0e-15):
                continue
            amplitudes = lower.vectors.conj().T @ operator @ upper.vectors
            for il, energy_l in enumerate(lower.energies_MHz):
                for iu, energy_u in enumerate(upper.energies_MHz):
                    if lower is upper and iu <= il:
                        continue
                    first, second = float(energy_l), float(energy_u)
                    strength = float(abs(amplitudes[il, iu]) ** 2)
                    if strength <= 1.0e-18 or abs(second - first) <= 1.0e-9:
                        continue
                    if first < second:
                        low, up, low_i, up_i, low_e, up_e = lower, upper, il, iu, first, second
                    else:
                        low, up, low_i, up_i, low_e, up_e = upper, lower, iu, il, second, first
                    frequency = up_e - low_e
                    intensity = strength
                    if beta is not None:
                        intensity *= (
                            frequency * (-math.expm1(-frequency * beta))
                            * math.exp(-(low_e - ground) * beta) / partition
                        )
                    if intensity < intensity_cut:
                        continue
                    rows.append({
                        "frequency_MHz": frequency,
                        "line_strength_D2": strength,
                        "intensity": intensity,
                        "lower_two_J": low.two_J,
                        "lower_parity": low.parity,
                        "lower_level": low_i,
                        "upper_two_J": up.two_J,
                        "upper_parity": up.parity,
                        "upper_level": up_i,
                    })
    maximum = max((row["intensity"] for row in rows), default=1.0)
    for row in rows:
        row["relative_intensity"] = 100.0 * row["intensity"] / maximum
    return sorted(rows, key=lambda row: row["frequency_MHz"])


def _rovibronic_reduced_dipole(
    model: RovibronicModel, lower: RovibronicBlock, upper: RovibronicBlock
) -> np.ndarray:
    lower_location = {state: index for index, state in enumerate(lower.basis_states)}
    upper_location = {state: index for index, state in enumerate(upper.basis_states)}
    operator = np.zeros((len(lower_location), len(upper_location)), dtype=complex)
    for moment in model.transition_moments:
        if moment.left not in lower_location or moment.right not in upper_location:
            continue
        left, right = model.states[moment.left], model.states[moment.right]
        p = (left.two_Omega - right.two_Omega) // 2
        phase = -1.0 if ((lower.two_J - left.two_Omega) // 2) % 2 else 1.0
        factor = phase * math.sqrt((lower.two_J + 1.0) * (upper.two_J + 1.0))
        factor *= wigner_3j_doubled(
            lower.two_J, 2, upper.two_J, -left.two_Omega, 2 * p, right.two_Omega
        )
        operator[lower_location[moment.left], upper_location[moment.right]] += (
            factor * moment.body_moment_D
        )
    return operator


def _as_complex(value: Any) -> complex:
    if isinstance(value, (int, float)):
        return complex(value)
    if isinstance(value, Sequence) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    raise ValueError("an interaction value must be a number or [real, imaginary]")


__all__ = [
    "ROVIBRONIC_MODEL_SCHEMA", "ROVIBRONIC_RESULT_SCHEMA", "RovibronicBasisState",
    "RovibronicInteraction", "RovibronicTransitionMoment", "RovibronicModel", "RovibronicLevel", "RovibronicBlock",
    "RovibronicResult", "solve_rovibronic", "rovibronic_hamiltonian", "read_rovibronic_model",
    "write_rovibronic_model", "write_rovibronic_result", "rovibronic_transitions",
]
