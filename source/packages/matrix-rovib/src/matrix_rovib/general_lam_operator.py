"""Declarative, molecule-independent operator algebra for coupled LAMs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .lam_rotation import LAMBasis as LAMBasis, LAMRotationSurface, _rotational_operators
from .operator_engine import CompiledOperatorModel, FactorizedOperatorTerm, OperatorSpace

GENERAL_LAM_OPERATOR_SCHEMA = "matrix.lock.general_lam_operator.v1"
_FUNCTIONS = {"constant", "cos", "sin", "one_minus_cos"}


@dataclass(frozen=True)
class GeneralLAMOperatorTerm:
    """One ordered product plus its adjoint, controlled by a named parameter."""

    label: str
    parameter: str
    prefactor: float = 1.0
    harmonics: tuple[int, ...] = ()
    functions: tuple[str, ...] = ()
    momentum_axes: tuple[int, ...] = ()
    rotation_word: tuple[int, ...] = ()
    symmetry: str = "totally-symmetric"

    def validate(self, surface: LAMRotationSurface) -> None:
        if not self.label.strip() or not self.parameter.strip():
            raise ValueError("general LAM operator terms require labels and parameters")
        if not self.symmetry.strip() or not math.isfinite(float(self.prefactor)):
            raise ValueError("operator symmetry and prefactor must be valid")
        if len(self.harmonics) != surface.dimension or len(self.functions) != surface.dimension:
            raise ValueError("each operator Fourier field must span every LAM coordinate")
        if any(function not in _FUNCTIONS for function in self.functions):
            raise ValueError("unsupported general LAM Fourier function")
        if any(axis not in range(surface.dimension) for axis in self.momentum_axes):
            raise ValueError("LAM momentum word references an absent coordinate")
        if any(axis not in range(3) for axis in self.rotation_word):
            raise ValueError("rotation words use axes 0, 1, 2 for a, b, c")
        for harmonic, function, periodic, order in zip(
            self.harmonics, self.functions, surface.periodic, surface.symmetry_orders,
            strict=True,
        ):
            if harmonic < 0 or int(harmonic) != harmonic:
                raise ValueError("Fourier harmonics must be nonnegative integers")
            if function == "constant" and harmonic != 0:
                raise ValueError("constant Fourier factors require harmonic zero")
            if function != "constant" and (not periodic or harmonic < 1):
                raise ValueError("nonconstant Fourier factors require a periodic coordinate")
            if self.symmetry == "totally-symmetric" and function != "constant" and harmonic % order:
                raise ValueError("a totally symmetric harmonic must preserve the LAM sector")


@dataclass(frozen=True)
class GeneralLAMOperatorManifest:
    """Immutable parameter registry and arbitrary-order operator expansion."""

    parameters_MHz: Mapping[str, float]
    terms: tuple[GeneralLAMOperatorTerm, ...]
    schema: str = GENERAL_LAM_OPERATOR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != GENERAL_LAM_OPERATOR_SCHEMA or not self.terms:
            raise ValueError("unsupported or empty general LAM operator manifest")
        parameters = {str(name): float(value) for name, value in self.parameters_MHz.items()}
        if not parameters or any(not name.strip() for name in parameters):
            raise ValueError("general LAM manifests require named parameters")
        if not all(math.isfinite(value) for value in parameters.values()):
            raise ValueError("general LAM parameter values must be finite")
        if {term.parameter for term in self.terms} != set(parameters):
            raise ValueError("parameter registry does not match its operator terms")
        labels = [term.label for term in self.terms]
        if len(labels) != len(set(labels)):
            raise ValueError("general LAM operator term labels must be unique")
        object.__setattr__(self, "parameters_MHz", MappingProxyType(parameters))
        object.__setattr__(self, "terms", tuple(self.terms))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GeneralLAMOperatorManifest":
        if payload.get("schema") != GENERAL_LAM_OPERATOR_SCHEMA:
            raise ValueError(f"unsupported general LAM schema: {payload.get('schema')!r}")
        raw_terms = payload.get("terms")
        if not isinstance(raw_terms, Sequence) or isinstance(raw_terms, (str, bytes)):
            raise ValueError("general LAM manifest terms must be an array")
        terms = []
        for raw in raw_terms:
            if not isinstance(raw, Mapping):
                raise ValueError("general LAM operator terms must be objects")
            terms.append(GeneralLAMOperatorTerm(
                label=str(raw["label"]), parameter=str(raw["parameter"]),
                prefactor=float(raw.get("prefactor", 1.0)),
                harmonics=tuple(map(int, raw.get("harmonics", ()))),
                functions=tuple(map(str, raw.get("functions", ()))),
                momentum_axes=tuple(map(int, raw.get("momentum_axes", ()))),
                rotation_word=tuple(map(int, raw.get("rotation_word", ()))),
                symmetry=str(raw.get("symmetry", "totally-symmetric")),
            ))
        parameters = payload.get("parameters_MHz")
        if not isinstance(parameters, Mapping):
            raise ValueError("general LAM manifest parameters_MHz must be an object")
        return cls(parameters, tuple(terms))

    def to_dict(self) -> dict[str, object]:
        return {"schema": self.schema, "parameters_MHz": dict(self.parameters_MHz), "terms": [
            {"label": term.label, "parameter": term.parameter, "prefactor": term.prefactor,
             "harmonics": list(term.harmonics), "functions": list(term.functions),
             "momentum_axes": list(term.momentum_axes), "rotation_word": list(term.rotation_word),
             "symmetry": term.symmetry}
            for term in self.terms
        ]}

    @property
    def digest(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def parameters_with(self, overrides: Mapping[str, float] | None = None) -> Mapping[str, float]:
        values = dict(self.parameters_MHz)
        if overrides:
            unknown = set(overrides) - set(values)
            if unknown:
                raise ValueError(f"unknown general LAM parameters: {sorted(unknown)}")
            values.update({str(name): float(value) for name, value in overrides.items()})
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("general LAM parameter overrides must be finite")
        return MappingProxyType(values)


def normalize_general_lam_operator_manifest(manifest):
    return manifest if isinstance(manifest, GeneralLAMOperatorManifest) else GeneralLAMOperatorManifest.from_dict(manifest)


def _fourier_field(surface, term):
    field = np.ones(surface.grid_shape, dtype=float)
    for axis, (coordinate, harmonic, function) in enumerate(zip(surface.coordinates, term.harmonics, term.functions, strict=True)):
        if function == "constant":
            factor = np.ones(len(coordinate))
        else:
            phase = harmonic * np.asarray(coordinate)
            factor = np.sin(phase) if function == "sin" else np.cos(phase)
            if function == "one_minus_cos":
                factor = 1.0 - factor
        shape = [1] * surface.dimension
        shape[axis] = len(factor)
        field *= factor.reshape(shape)
    return field.reshape(-1)


@dataclass(frozen=True)
class ContractedGeneralLAMOperators:
    """J-independent contraction reused by every rotational block."""

    manifest: GeneralLAMOperatorManifest
    lam_factors: tuple[np.ndarray, ...]

    def compile(self, J, *, representation="Ir", parameters_MHz=None):
        return _compile_contracted_model(
            self, J, representation=representation, parameters_MHz=parameters_MHz
        )


def contract_general_lam_operators(surface, basis, manifest):
    """Project ordered words by acting on retained vectors, without grid matrices."""
    from .lam_rotation import _apply_momentum

    normalized = normalize_general_lam_operator_manifest(manifest)
    vectors = np.asarray(basis.vectors, dtype=complex)
    # Shared prefixes in the applied (right-to-left) momentum words are
    # evaluated once. This bounds storage by grid size times retained states.
    acted = {(): vectors}
    projected = {}
    factors = []
    for term in normalized.terms:
        term.validate(surface)
        word = tuple(reversed(term.momentum_axes))
        for length in range(1, len(word) + 1):
            prefix = word[:length]
            if prefix not in acted:
                axis = prefix[-1]
                previous = acted[prefix[:-1]]
                acted[prefix] = (
                    basis.momentum_operators[axis] @ previous
                    if basis.momentum_operators
                    else _apply_momentum(surface, axis, previous)
                )
        key = (term.harmonics, term.functions, term.momentum_axes)
        if key not in projected:
            field = _fourier_field(surface, term)
            invariant = term.symmetry == "totally-symmetric" and surface.symmetry_sector_count > 1
            if invariant:
                tensor = field.reshape(surface.grid_shape)
                for axis, order in enumerate(surface.symmetry_orders):
                    if order > 1 and not np.allclose(tensor, np.roll(tensor, surface.grid_shape[axis]//order, axis=axis), rtol=0., atol=1.e-10):
                        raise ValueError(f"operator {term.label!r} does not preserve the declared translation symmetry")
            from .lam_rotation import _project_lam_values
            matrix = _project_lam_values(basis, field[:, None] * acted[word], preserve_sectors=invariant)
            if term.symmetry == "totally-symmetric" and surface.symmetry_sector_count > 1:
                sectors = np.asarray(basis.symmetry_sectors)
                forbidden = sectors[:, None] != sectors[None, :]
                error = float(np.max(np.abs(matrix[forbidden]), initial=0.0))
                scale = max(1.0, float(np.max(np.abs(matrix), initial=0.0)))
                if error > 1.0e-9 * scale:
                    raise ValueError(f"operator {term.label!r} mixes distinct product sectors")
                matrix[forbidden] = 0.0
            matrix.setflags(write=False)
            projected[key] = matrix
        factors.append(projected[key])
    return ContractedGeneralLAMOperators(normalized, tuple(factors))


def _compile_contracted_model(contraction, J, *, representation, parameters_MHz):
    normalized = contraction.manifest
    rotational = _rotational_operators(int(J), representation)
    identity_rot = np.eye(2 * int(J) + 1, dtype=complex)
    terms = []
    for term, lam in zip(normalized.terms, contraction.lam_factors, strict=True):
        rot = identity_rot
        for axis in term.rotation_word:
            rot = rot @ rotational[axis]
        terms.append(FactorizedOperatorTerm(
            label=term.label, factors=(("lam", lam), ("rotation", rot)),
            coefficient=term.prefactor, parameter=term.parameter,
            adjoint_symmetrized=True, symmetry=term.symmetry,
        ))
    return CompiledOperatorModel(
        spaces=(OperatorSpace("lam", contraction.lam_factors[0].shape[0]), OperatorSpace("rotation", 2 * int(J) + 1)),
        terms=tuple(terms), parameter_defaults=normalized.parameters_with(parameters_MHz),
    )


def compile_general_lam_model(surface, basis, J, manifest, *, representation="Ir", parameters_MHz=None):
    """Compile once for dense, matrix-free and exact-derivative backends."""
    contraction = (
        manifest if isinstance(manifest, ContractedGeneralLAMOperators)
        else contract_general_lam_operators(surface, basis, manifest)
    )
    return contraction.compile(J, representation=representation, parameters_MHz=parameters_MHz)


def compile_general_lam_factors(surface, basis, J, manifest, *, representation="Ir", parameters_MHz=None):
    """Expand the Hermitian products for existing coupled-DVR backends."""
    model = compile_general_lam_model(surface, basis, J, manifest, representation=representation, parameters_MHz=parameters_MHz)
    factors = []
    for term in model.terms:
        matrices = dict(term.factors)
        scale = float(term.coefficient.real) * model.parameter_defaults[term.parameter]
        factors.append((0.5 * scale * matrices["lam"], matrices["rotation"]))
        factors.append((0.5 * scale * matrices["lam"].conj().T, matrices["rotation"].conj().T))
    return tuple(factors)


__all__ = ["GENERAL_LAM_OPERATOR_SCHEMA", "GeneralLAMOperatorManifest", "GeneralLAMOperatorTerm",
           "ContractedGeneralLAMOperators", "contract_general_lam_operators",
           "compile_general_lam_factors", "compile_general_lam_model",
           "normalize_general_lam_operator_manifest"]


def truncate_rotational_order(manifest,order):
    """Retain a homogeneous rotational order without changing parameter IDs."""
    from dataclasses import replace
    normalized=normalize_general_lam_operator_manifest(manifest)
    return replace(normalized,terms=tuple(term if len(term.rotation_word)<=order else
        replace(term,prefactor=0.,rotation_word=()) for term in normalized.terms))
