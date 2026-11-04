"""ORACLE-owned hierarchy of primitive-coordinate classes.

The hierarchy is chemical metadata. It tells SMITH which local primitive
families may be delocalized first; it does not perform linear algebra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


ORACLE_PRIMITIVE_CLASS_SCHEMA = "matrix.oracle.primitive_classes.v1"
PRIMITIVE_DOMAIN_INTRA = "INTRAFRAGMENT"
PRIMITIVE_DOMAIN_INTER = "INTERFRAGMENT"
SYCART_CLASS_RANK = 10_000


@dataclass(frozen=True)
class PrimitiveClassAssignment:
    primitive_id: str
    family: str
    class_id: str
    class_rank: int
    domain_kind: str
    domain_id: str

    def __post_init__(self) -> None:
        if self.domain_kind not in {PRIMITIVE_DOMAIN_INTRA, PRIMITIVE_DOMAIN_INTER}:
            raise ValueError("primitive domain must be intra- or interfragment")
        if self.class_rank < 0 or self.class_rank >= SYCART_CLASS_RANK:
            raise ValueError("chemical primitive class rank is outside the allowed range")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": ORACLE_PRIMITIVE_CLASS_SCHEMA,
            "primitive_id": self.primitive_id,
            "family": self.family,
            "class_id": self.class_id,
            "class_rank": self.class_rank,
            "domain_kind": self.domain_kind,
            "domain_id": self.domain_id,
        }


# Lower ranks are admitted first. These ranks are neither force constants nor masses.
_FUNCTION_CLASS: dict[str, tuple[str, int]] = {
    "R": ("DISTANCE", 10),
    "A": ("VALENCE_ANGLE", 20),
    "L": ("LINEAR_BEND", 20),
    "U": ("OUT_OF_PLANE", 30),
    "SOOP": ("SELECTIVE_OUT_OF_PLANE", 30),
    "IMPD": ("IMPROPER_DIHEDRAL", 30),
    "D": ("TORSION", 40),
    "FACE_FLAP": ("FACE_FLAP", 50),
    "FACE_HINGE": ("FACE_HINGE", 50),
    "H": ("RING_FLAP", 50),
    "CENTER_FACE_DIST": ("CENTER_FACE_DISTANCE", 60),
    "FACE_DIRECTION_ANGLE": ("FACE_DIRECTION", 60),
    "CONTACT_CENTER_ANGLE": ("ORIENTED_CONTACT", 60),
    "CENTER_ATOM_DIST": ("CENTER_ATOM_DISTANCE", 60),
    "FCA_DIST": ("FRAGMENT_CENTER_ATOM_DISTANCE", 70),
    "FC_DIST": ("FRAGMENT_DISTANCE", 70),
    "FTRANS": ("FRAGMENT_TRANSLATION", 70),
    "FLIN_TRANS": ("FRAGMENT_TRANSLATION", 70),
    "FROT": ("FRAGMENT_ORIENTATION", 80),
    "FAXIS": ("FRAGMENT_ORIENTATION", 80),
}

_FAMILY_CLASS: dict[str, tuple[str, int]] = {
    "FULLERENE_FACE_DIAGONAL": ("FULLERENE_FACE", 50),
    "RING_PUCKER_COMPONENT": ("RING_FLAP", 50),
    "BUTTERFLY": ("RING_FLAP", 50),
    "METAL_COORDINATION": ("MULTICENTER_COORDINATION", 60),
    "MULTICENTER_DISTANCE": ("MULTICENTER_COORDINATION", 60),
    "MULTICENTER_BEND": ("MULTICENTER_COORDINATION", 60),
    "MULTICENTER_BRIDGE_PLANE_COUPLING": ("MULTICENTER_COORDINATION", 60),
    "AUXILIARY_CONTACT": ("ORIENTED_CONTACT", 60),
    "CONTACT_SUPPORT": ("ORIENTED_CONTACT", 60),
    "FRAG_DISTANCE": ("FRAGMENT_TRANSLATION", 70),
    "FRAG_TRANSLATION": ("FRAGMENT_TRANSLATION", 70),
    "FRAG_ORIENTATION": ("FRAGMENT_ORIENTATION", 80),
}


def primitive_class(function: str, family: str) -> tuple[str, int]:
    """Return the extensible ORACLE class identifier and ordering rank."""

    family_key = str(family).strip().upper()
    if family_key in _FAMILY_CLASS:
        return _FAMILY_CLASS[family_key]
    key = str(function).strip().upper()
    if key in _FUNCTION_CLASS:
        return _FUNCTION_CLASS[key]
    label = family_key or key or "EXTENSION"
    return f"EXTENSION::{label}", 9000


def assign_primitive_classes(
    primitives: Iterable[object],
    *,
    fragment_by_atom: Mapping[int, str] | None = None,
    rank_overrides: Mapping[str, int] | None = None,
) -> tuple[PrimitiveClassAssignment, ...]:
    """Attach class and strict fragment-domain metadata to primitives."""

    fragments = dict(fragment_by_atom or {})
    overrides = {str(key).upper(): int(value) for key, value in (rank_overrides or {}).items()}
    result: list[PrimitiveClassAssignment] = []
    inter_functions = {"FC_DIST", "FCA_DIST", "FTRANS", "FLIN_TRANS", "FROT", "FAXIS"}
    for primitive in primitives:
        identifier = str(getattr(primitive, "identifier"))
        family = str(getattr(primitive, "family"))
        function = str(getattr(primitive, "function")).upper()
        class_id, default_rank = primitive_class(function, family)
        rank = overrides.get(class_id, overrides.get(family.upper(), default_rank))
        if rank < 0 or rank >= SYCART_CLASS_RANK:
            raise ValueError(f"invalid class rank {rank} for {identifier}")
        atoms = tuple(int(atom) for atom in getattr(primitive, "atoms", ()))
        ref_atoms = tuple(int(atom) for atom in getattr(primitive, "ref_atoms", ()))
        fragment_ids = tuple(
            sorted({fragments.get(atom, "MOLECULE") for atom in (*atoms, *ref_atoms)})
        )
        is_inter = function in inter_functions or len(fragment_ids) > 1
        kind = PRIMITIVE_DOMAIN_INTER if is_inter else PRIMITIVE_DOMAIN_INTRA
        domain_id = (
            "INTER::" + "::".join(fragment_ids)
            if is_inter
            else "INTRA::" + (fragment_ids[0] if fragment_ids else "MOLECULE")
        )
        result.append(
            PrimitiveClassAssignment(
                primitive_id=identifier,
                family=family,
                class_id=class_id,
                class_rank=rank,
                domain_kind=kind,
                domain_id=domain_id,
            )
        )
    return tuple(result)


__all__ = [
    "ORACLE_PRIMITIVE_CLASS_SCHEMA",
    "PRIMITIVE_DOMAIN_INTER",
    "PRIMITIVE_DOMAIN_INTRA",
    "SYCART_CLASS_RANK",
    "PrimitiveClassAssignment",
    "assign_primitive_classes",
    "primitive_class",
]
