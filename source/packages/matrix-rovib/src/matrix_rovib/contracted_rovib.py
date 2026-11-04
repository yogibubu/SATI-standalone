"""Contracted vibrational/polyad states coupled to the native LOCK rotor."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .native_rotor import (
    MHZ_TO_KELVIN,
    NativeRotorParameters,
    diagonalize_rotor,
    dipole_transition_matrix,
    rotational_hamiltonian,
    angular_momentum_operators,
)


CONTRACTED_ROVIB_MODEL_SCHEMA = "matrix.lock.contracted_rovib_model.v1"
CONTRACTED_ROVIB_RESULT_SCHEMA = "matrix.lock.contracted_rovib_result.v1"


@dataclass(frozen=True)
class ContractedVibrationalState:
    label: str
    energy_MHz: float
    rotor: NativeRotorParameters
    symmetry: str = "A"
    quanta: tuple[int, ...] = ()
    dipole_debye: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ContractedVibrationalCoupling:
    left: int
    right: int
    value_MHz: float
    kind: str = "polyad"
    rotation_word: tuple[int, ...] = ()
    imaginary_MHz: float = 0.0

    def __post_init__(self):
        if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in (self.left, self.right)):
            raise ValueError("coupling state indices must be integers")
        if any(isinstance(a, bool) or not isinstance(a, (int, np.integer)) or a not in (0, 1, 2)
               for a in self.rotation_word):
            raise ValueError("ordered rotation words require integer Cartesian axes")
        object.__setattr__(self, 'rotation_word', tuple(int(a) for a in self.rotation_word))


@dataclass(frozen=True)
class ContractedRovibModel:
    states: tuple[ContractedVibrationalState, ...]
    couplings: tuple[ContractedVibrationalCoupling, ...] = ()

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("a contracted rovibrational model requires at least one state")
        labels = [state.label for state in self.states]
        if len(labels) != len(set(labels)):
            raise ValueError("contracted vibrational state labels must be unique")
        for coupling in self.couplings:
            if not 0 <= coupling.left < len(self.states) or not 0 <= coupling.right < len(self.states):
                raise ValueError("contracted vibrational coupling index is outside the state basis")
            if ((coupling.left == coupling.right and not coupling.rotation_word) or not math.isfinite(coupling.value_MHz)
                    or not math.isfinite(coupling.imaginary_MHz)
                    or any(isinstance(a, bool) or a not in (0, 1, 2) for a in coupling.rotation_word)):
                raise ValueError("contracted couplings must be finite; scalar couplings must be off-diagonal")
            if not coupling.rotation_word and self.states[coupling.left].symmetry != self.states[coupling.right].symmetry:
                raise ValueError("scalar polyad coupling requires equal vibrational symmetry")
        if len({s.rotor.representation for s in self.states}) != 1:
            raise ValueError("contracted states require a common rotational frame")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACTED_ROVIB_MODEL_SCHEMA,
            "states": [
                {
                    "label": state.label,
                    "energy_MHz": state.energy_MHz,
                    "symmetry": state.symmetry,
                    "quanta": list(state.quanta),
                    "dipole_debye": list(state.dipole_debye),
                    "rotor": _rotor_to_dict(state.rotor),
                }
                for state in self.states
            ],
            "couplings": [coupling.__dict__ for coupling in self.couplings],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractedRovibModel":
        if payload.get("schema") != CONTRACTED_ROVIB_MODEL_SCHEMA:
            raise ValueError(f"unsupported contracted rovib schema: {payload.get('schema')!r}")
        states = tuple(
            ContractedVibrationalState(
                label=str(item["label"]),
                energy_MHz=float(item["energy_MHz"]),
                symmetry=str(item.get("symmetry", "A")),
                quanta=tuple(int(value) for value in item.get("quanta", ())),
                dipole_debye=tuple(float(value) for value in item.get("dipole_debye", (0.0, 0.0, 0.0))),
                rotor=_rotor_from_dict(item["rotor"]),
            )
            for item in payload.get("states", ())
        )
        couplings = tuple(
            ContractedVibrationalCoupling(
                int(item["left"]),
                int(item["right"]),
                float(item["value_MHz"]),
                str(item.get("kind", "polyad")),
                tuple(item.get("rotation_word", ())),
                float(item.get("imaginary_MHz", 0.0)),
            )
            for item in payload.get("couplings", ())
        )
        return cls(states, couplings)


@dataclass(frozen=True)
class ContractedRovibLevel:
    energy_MHz: float
    J: int
    vibrational_parent: int
    vibrational_label: str
    symmetry: str
    Ka: int
    Kc: int
    parent_purity: float


@dataclass(frozen=True)
class ContractedRovibBlock:
    J: int
    energies_MHz: np.ndarray
    vectors: np.ndarray
    levels: tuple[ContractedRovibLevel, ...]
    hermiticity_error_MHz: float


@dataclass(frozen=True)
class ContractedRovibResult:
    model: ContractedRovibModel
    blocks: tuple[ContractedRovibBlock, ...]


def contracted_coupling_matrix(model, coupling, J):
    """Ordered upper off-diagonal block; the lower block is its adjoint."""
    angular = angular_momentum_operators(J, model.states[0].rotor.representation)
    operator = np.eye(2 * J + 1, dtype=complex)
    for axis in coupling.rotation_word:
        operator = operator @ angular[axis]
    return complex(coupling.value_MHz, coupling.imaginary_MHz) * operator


def retain_contracted_manifold(model, target_states, *, j_values):
    """Close targets under every nonzero coupling over the requested J range.

    This conservative closure omits only disconnected states, so it introduces
    no small-denominator approximation or molecule-dependent gap threshold.
    Complete symmetry partner sets must be present in the producer basis.
    """
    targets = tuple(target_states)
    if (not targets or any(isinstance(i, bool) or int(i) != i or not 0 <= i < len(model.states)
                           for i in targets)):
        raise ValueError("valid target-state indices required")
    js = tuple(j_values)
    if not js or any(isinstance(j, bool) or int(j) != j or j < 0 for j in js):
        raise ValueError("nonempty nonnegative integer J range required")
    # Test each angular word once, not once per vibrational matrix element.
    from dataclasses import replace
    present = {}
    for c in model.couplings:
        if c.rotation_word not in present:
            unit = replace(c, value_MHz=1., imaginary_MHz=0.)
            present[c.rotation_word] = any(np.any(contracted_coupling_matrix(model, unit, int(j))) for j in js)
    edges = [(c.left, c.right) for c in model.couplings
             if (c.value_MHz or c.imaginary_MHz) and present[c.rotation_word]]
    selected = set(map(int, targets))
    while True:
        expanded = selected | {b for a, b in edges if a in selected} | {a for a, b in edges if b in selected}
        if expanded == selected:
            break
        selected = expanded
    indices = tuple(sorted(selected))
    mapping = {old: new for new, old in enumerate(indices)}
    reduced = ContractedRovibModel(tuple(model.states[i] for i in indices), tuple(
        replace(c, left=mapping[c.left], right=mapping[c.right]) for c in model.couplings
        if c.left in mapping and c.right in mapping))
    return reduced, dict(original_indices=list(indices), omitted_indices=[i for i in range(len(model.states))
        if i not in mapping], j_values=list(js), approximation="exact disconnected-component closure")


def solve_contracted_rovib(
    model: ContractedRovibModel,
    *,
    j_values: Sequence[int] | range,
    levels_per_j: int | None = None,
) -> ContractedRovibResult:
    """Diagonalize a direct-product vibrational/polyad and rotational Hamiltonian."""
    blocks = []
    count = len(model.states)
    js = tuple(j_values)
    if not js or any(isinstance(j, bool) or int(j) != j or j < 0 for j in js):
        raise ValueError("J values must be nonnegative integers")
    for J in tuple(sorted({int(value) for value in js})):
        rotational_size = 2 * J + 1
        hamiltonian = np.zeros((count * rotational_size,) * 2, dtype=complex)
        for state_index, state in enumerate(model.states):
            section = slice(state_index * rotational_size, (state_index + 1) * rotational_size)
            hamiltonian[section, section] = rotational_hamiltonian(state.rotor, J)
            hamiltonian[section, section] += np.eye(rotational_size) * state.energy_MHz
        for coupling in model.couplings:
            left = slice(coupling.left * rotational_size, (coupling.left + 1) * rotational_size)
            right = slice(coupling.right * rotational_size, (coupling.right + 1) * rotational_size)
            block = contracted_coupling_matrix(model, coupling, J)
            if coupling.left == coupling.right:
                hamiltonian[left, left] += .5*(block+block.conj().T)
            else:
                hamiltonian[left, right] += block
                hamiltonian[right, left] += block.conj().T
        hermiticity = float(np.max(np.abs(hamiltonian - hamiltonian.conj().T)))
        energies, vectors = np.linalg.eigh(hamiltonian)
        if levels_per_j is not None:
            if levels_per_j < 1:
                raise ValueError("levels_per_j must be positive")
            energies, vectors = energies[:levels_per_j], vectors[:, :levels_per_j]
        rotor_blocks = [diagonalize_rotor(state.rotor, J) for state in model.states]
        levels = []
        for level_index, energy in enumerate(energies):
            weights = np.asarray([
                float(np.vdot(
                    vectors[state_index * rotational_size : (state_index + 1) * rotational_size, level_index],
                    vectors[state_index * rotational_size : (state_index + 1) * rotational_size, level_index],
                ).real)
                for state_index in range(count)
            ])
            parent = int(np.argmax(weights))
            component = vectors[parent * rotational_size : (parent + 1) * rotational_size, level_index]
            overlaps = np.abs(rotor_blocks[parent].vectors.conj().T @ component) ** 2
            rotor_parent = int(np.argmax(overlaps))
            Ka, Kc, _label = rotor_blocks[parent].assignments[rotor_parent]
            state = model.states[parent]
            levels.append(
                ContractedRovibLevel(
                    float(energy), J, parent, state.label, state.symmetry, Ka, Kc, float(weights[parent])
                )
            )
        blocks.append(ContractedRovibBlock(J, energies, vectors, tuple(levels), hermiticity))
    return ContractedRovibResult(model, tuple(blocks))


def contracted_rovib_transitions(
    result: ContractedRovibResult,
    *,
    temperature_K: float | None = None,
    intensity_cut: float = 0.0,
) -> list[dict[str, Any]]:
    """Return pure rotational transitions with intensity borrowing inside polyads."""
    if temperature_K is not None and temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    ground = min(float(block.energies_MHz[0]) for block in result.blocks)
    beta = None if temperature_K is None else MHZ_TO_KELVIN / temperature_K
    partition = 1.0
    if beta is not None:
        partition = math.fsum(
            (2 * block.J + 1) * math.exp(-(float(energy) - ground) * beta)
            for block in result.blocks for energy in block.energies_MHz
        )
    raw = []
    by_j = {block.J: block for block in result.blocks}
    for J, lower in by_j.items():
        upper = by_j.get(J + 1)
        if upper is None:
            continue
        nl, nu = 2 * J + 1, 2 * (J + 1) + 1
        operator = np.zeros((len(result.model.states) * nl, len(result.model.states) * nu), complex)
        for index, state in enumerate(result.model.states):
            body = np.zeros((nl, nu), complex)
            for mu, axis in zip(state.dipole_debye, ("a", "b", "c"), strict=True):
                if mu:
                    body += mu * dipole_transition_matrix(J, J + 1, axis, state.rotor.representation)
            operator[index * nl : (index + 1) * nl, index * nu : (index + 1) * nu] = body
        amplitudes = lower.vectors.conj().T @ operator @ upper.vectors
        for il, lower_energy in enumerate(lower.energies_MHz):
            for iu, upper_energy in enumerate(upper.energies_MHz):
                frequency = float(upper_energy - lower_energy)
                strength = float(abs(amplitudes[il, iu]) ** 2)
                if frequency <= 1.0e-9 or strength <= 1.0e-18:
                    continue
                intensity = strength
                if beta is not None:
                    intensity *= (
                        frequency * (-math.expm1(-frequency * beta))
                        * math.exp(-(float(lower_energy) - ground) * beta) / partition
                    )
                if intensity < intensity_cut:
                    continue
                raw.append({
                    "frequency_MHz": frequency,
                    "line_strength_D2": strength,
                    "intensity": intensity,
                    "lower_J": J,
                    "lower_level": il,
                    "lower_vibrational_label": lower.levels[il].vibrational_label,
                    "lower_Ka": lower.levels[il].Ka,
                    "lower_Kc": lower.levels[il].Kc,
                    "upper_J": J + 1,
                    "upper_level": iu,
                    "upper_vibrational_label": upper.levels[iu].vibrational_label,
                    "upper_Ka": upper.levels[iu].Ka,
                    "upper_Kc": upper.levels[iu].Kc,
                })
    maximum = max((row["intensity"] for row in raw), default=1.0)
    for row in raw:
        row["relative_intensity"] = 100.0 * row["intensity"] / maximum
    return sorted(raw, key=lambda row: row["frequency_MHz"])


def read_contracted_rovib_model(path: Path | str) -> ContractedRovibModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("a contracted rovib model must be a JSON object")
    return ContractedRovibModel.from_dict(payload)


def write_contracted_rovib_model(path: Path | str, model: ContractedRovibModel) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target


def write_contracted_rovib_result(path: Path | str, result: ContractedRovibResult) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CONTRACTED_ROVIB_RESULT_SCHEMA,
        "model_schema": CONTRACTED_ROVIB_MODEL_SCHEMA,
        "blocks": [
            {
                "J": block.J,
                "dimension": len(block.energies_MHz),
                "hermiticity_error_MHz": block.hermiticity_error_MHz,
                "levels": [level.__dict__ for level in block.levels],
            }
            for block in result.blocks
        ],
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def write_contracted_rovib_lines(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ())
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return target


def _rotor_to_dict(parameters: NativeRotorParameters) -> dict[str, Any]:
    return {
        "A_MHz": parameters.A_MHz, "B_MHz": parameters.B_MHz, "C_MHz": parameters.C_MHz,
        "quartic_MHz": list(parameters.quartic_MHz), "sextic_MHz": list(parameters.sextic_MHz),
        "octic_MHz": list(parameters.octic_MHz),
        "reduction": parameters.reduction, "representation": parameters.representation,
    }


def _rotor_from_dict(payload: Mapping[str, Any]) -> NativeRotorParameters:
    return NativeRotorParameters(
        float(payload["A_MHz"]), float(payload["B_MHz"]), float(payload["C_MHz"]),
        tuple(float(value) for value in payload.get("quartic_MHz", (0.0,) * 5)),
        tuple(float(value) for value in payload.get("sextic_MHz", (0.0,) * 7)),
        str(payload.get("reduction", "S")), str(payload.get("representation", "Ir")),
        tuple(float(value) for value in payload.get("octic_MHz", (0.0,) * 9)),
    )


__all__ = [
    "CONTRACTED_ROVIB_MODEL_SCHEMA", "CONTRACTED_ROVIB_RESULT_SCHEMA",
    "ContractedVibrationalState", "ContractedVibrationalCoupling", "ContractedRovibModel",
    "ContractedRovibLevel", "ContractedRovibBlock", "ContractedRovibResult",
    "solve_contracted_rovib", "contracted_rovib_transitions", "read_contracted_rovib_model",
    "write_contracted_rovib_model", "write_contracted_rovib_result", "write_contracted_rovib_lines",
]
