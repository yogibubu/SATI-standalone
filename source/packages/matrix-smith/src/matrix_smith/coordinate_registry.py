from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from matrix_chem import coordinate_capability


COMMUTATIVE_FUNCTIONS = {"SUM", "PRODUCT"}
FRAGMENT_BODY_PAIR_FUNCTIONS = frozenset({"FC_DIST", "FTRANS", "FLIN_TRANS", "FAXIS", "FROT"})
FRAGMENT_BODY_SINGLE_FUNCTIONS = frozenset({"FCA_DIST"})
MODE_BEARING_PRIMITIVE_FUNCTIONS = frozenset({"L", "FTRANS", "FLIN_TRANS", "FAXIS", "FROT"})
GAUSSIAN_PRIMITIVE_LOWERINGS = {
    "FROT": "REFERENCE_RELATIVE_QUATERNION_STEREOGRAPHIC_4K_OVER_1_PLUS_KW",
}


def gaussian_primitive_lowering(operator: str) -> str:
    """Return the registered ReadAllGIC realization of a SMITH primitive.

    The registry keeps backend algebra separate from both scientific primitive
    selection and the serializer that emits it.  Missing entries are explicit:
    a non-native primitive must never acquire an implicit writer-side lowering.
    """

    normalized = _canonical_operator(operator)
    coordinate_capability(normalized, layer="PRIMITIVE")
    try:
        return GAUSSIAN_PRIMITIVE_LOWERINGS[normalized]
    except KeyError as exc:
        raise KeyError(f"no registered Gaussian lowering for primitive: {normalized}") from exc


@dataclass(frozen=True, order=True)
class CoordinateSignature:
    """Canonical identity for a generated primitive or derived coordinate."""

    kind: str
    operator: str
    atoms: tuple[int, ...] = ()
    mode: int = 0
    ref_atoms: tuple[int, ...] = ()
    arguments: tuple["CoordinateSignature", ...] = ()
    parameters: tuple[tuple[str, str], ...] = ()

    def text(self) -> str:
        parts = [self.kind.upper(), self.operator.upper()]
        if self.atoms:
            parts.append("atoms=" + "-".join(str(atom) for atom in self.atoms))
        if self.ref_atoms:
            parts.append("refs=" + "-".join(str(atom) for atom in self.ref_atoms))
        if self.mode:
            parts.append(f"mode={self.mode}")
        if self.arguments:
            parts.append("args=(" + ",".join(argument.text() for argument in self.arguments) + ")")
        if self.parameters:
            parts.append(
                "params=(" + ",".join(f"{key}={value}" for key, value in self.parameters) + ")"
            )
        return "|".join(parts)


@dataclass(frozen=True)
class RegisteredCoordinate:
    signature: CoordinateSignature
    identifier: str
    label: str
    payload: Any = None


class CoordinateRegistry:
    """Deterministic interning registry for primitive and function coordinates.

    This is deliberately a semantic registry rather than a Gaussian-style input
    parser.  It prevents duplicate generated coordinate records while preserving
    the distinction between primitive model construction and iterative B-row
    evaluation.
    """

    def __init__(self, *, prefix: str = "C") -> None:
        self._prefix = str(prefix)
        self._records: dict[CoordinateSignature, RegisteredCoordinate] = {}
        self._order: list[CoordinateSignature] = []

    def register(
        self,
        signature: CoordinateSignature,
        *,
        label: str | None = None,
        payload: Any = None,
    ) -> RegisteredCoordinate:
        existing = self._records.get(signature)
        if existing is not None:
            return existing
        identifier = f"{self._prefix}{len(self._order) + 1:05d}"
        record = RegisteredCoordinate(
            signature=signature,
            identifier=identifier,
            label=label or signature.text(),
            payload=payload,
        )
        self._records[signature] = record
        self._order.append(signature)
        return record

    def primitive(
        self,
        operator: str,
        atoms: Iterable[int],
        *,
        mode: int = 0,
        ref_atoms: Iterable[int] = (),
        label: str | None = None,
        payload: Any = None,
    ) -> RegisteredCoordinate:
        return self.register(
            primitive_signature(operator, atoms, mode=mode, ref_atoms=ref_atoms),
            label=label,
            payload=payload,
        )

    def function(
        self,
        operator: str,
        arguments: Iterable[CoordinateSignature],
        *,
        parameters: Mapping[str, Any] | Iterable[tuple[str, Any]] = (),
        label: str | None = None,
        payload: Any = None,
    ) -> RegisteredCoordinate:
        return self.register(
            function_signature(operator, arguments, parameters=parameters),
            label=label,
            payload=payload,
        )

    def get(self, signature: CoordinateSignature) -> RegisteredCoordinate | None:
        return self._records.get(signature)

    def records(self) -> tuple[RegisteredCoordinate, ...]:
        return tuple(self._records[signature] for signature in self._order)

    def ordered_records(self) -> tuple[RegisteredCoordinate, ...]:
        return tuple(self._records[signature] for signature in sorted(self._records))

    def ordered_signatures(self) -> tuple[CoordinateSignature, ...]:
        return tuple(sorted(self._records))

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, signature: CoordinateSignature) -> bool:
        return signature in self._records


def primitive_signature(
    operator: str,
    atoms: Iterable[int],
    *,
    mode: int = 0,
    ref_atoms: Iterable[int] = (),
) -> CoordinateSignature:
    op = _canonical_operator(operator)
    coordinate_capability(op, layer="PRIMITIVE")
    normalized_atoms = _canonical_atoms(op, tuple(int(atom) for atom in atoms))
    normalized_refs = tuple(int(atom) for atom in ref_atoms)
    if op in {"FC_DIST"}:
        left = tuple(sorted(normalized_atoms))
        right = tuple(sorted(normalized_refs))
        if right < left:
            left, right = right, left
        normalized_atoms, normalized_refs = left, right
    return CoordinateSignature(
        kind="PRIMITIVE",
        operator=op,
        atoms=normalized_atoms,
        mode=int(mode),
        ref_atoms=normalized_refs,
    )


def function_signature(
    operator: str,
    arguments: Iterable[CoordinateSignature],
    *,
    parameters: Mapping[str, Any] | Iterable[tuple[str, Any]] = (),
) -> CoordinateSignature:
    op = str(operator).strip().upper()
    coordinate_capability(op, layer="FUNCTION")
    args = tuple(arguments)
    if op in COMMUTATIVE_FUNCTIONS:
        args = tuple(sorted(args))
    return CoordinateSignature(
        kind="FUNCTION",
        operator=op,
        arguments=args,
        parameters=_canonical_parameters(parameters),
    )


def _canonical_atoms(operator: str, atoms: tuple[int, ...]) -> tuple[int, ...]:
    if operator in {"R", "DISTANCE", "HBOND_DISTANCE"} and len(atoms) == 2:
        return tuple(sorted(atoms))
    if operator in {"A", "ANGLE"} and len(atoms) == 3:
        left, center, right = atoms
        if right < left:
            return (right, center, left)
        return atoms
    if operator in {"D", "DIHEDRAL"} and len(atoms) == 4:
        reversed_atoms = tuple(reversed(atoms))
        return min(atoms, reversed_atoms)
    return atoms


def _canonical_operator(operator: str) -> str:
    normalized = str(operator).strip().upper()
    return {
        "DISTANCE": "R",
        "HBOND_DISTANCE": "R",
        "ANGLE": "A",
        "DIHEDRAL": "D",
    }.get(normalized, normalized)


def _canonical_parameters(
    parameters: Mapping[str, Any] | Iterable[tuple[str, Any]],
) -> tuple[tuple[str, str], ...]:
    items = parameters.items() if isinstance(parameters, Mapping) else tuple(parameters)
    return tuple(sorted((str(key), _parameter_text(value)) for key, value in items))


def _parameter_text(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.17g}"
    if isinstance(value, (tuple, list)):
        return "[" + ",".join(_parameter_text(item) for item in value) + "]"
    return str(value)
