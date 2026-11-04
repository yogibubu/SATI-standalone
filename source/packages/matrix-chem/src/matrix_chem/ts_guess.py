"""Deterministic Cartesian images used to audit a two-endpoint TS seed.

The chemically meaningful coordinate construction lives in
``reaction_topology``.  This module only aligns the endpoints and emits
Cartesian images; it does not select coordinates, claim that the middle image
is a saddle, or run an electronic-structure optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Iterable, Sequence

import numpy as np

from .geometry import MolecularGeometry
from .geometry_alignment import kabsch_align
from .primitive_coordinates import Primitive, eval_primitive


TS_GUESS_SCHEMA = "matrix.chem.ts_guess.path_seed.v1"


class TransitionStateGuessError(ValueError):
    """Raised when two endpoint geometries cannot define a TS-seed path."""


@dataclass(frozen=True)
class CoordinateUnionEntry:
    """One coordinate present in the union of the endpoint coordinate sets."""

    label: str
    kind: str
    atoms: tuple[int, ...]
    start_value: float
    end_value: float
    displacement: float


@dataclass(frozen=True)
class TSGuessImage:
    """One Cartesian image in the endpoint path, including its parameter."""

    index: int
    fraction: float
    geometry: MolecularGeometry


@dataclass(frozen=True)
class TransitionStateGuessPath:
    """Self-contained Cartesian image path and its coordinate audit."""

    schema: str
    start: MolecularGeometry
    end: MolecularGeometry
    images: tuple[TSGuessImage, ...]
    coordinate_union: tuple[CoordinateUnionEntry, ...]
    atom_mapping: tuple[int, ...]
    aligned_end_rms_displacement_angstrom: float
    symmetry_preserved: bool
    symmetry_lowering_requested: bool

    @property
    def guess(self) -> MolecularGeometry:
        """Return the central image, the initial TS guess for downstream LINK."""

        return self.images[len(self.images) // 2].geometry


def build_transition_state_guess(
    start: MolecularGeometry,
    end: MolecularGeometry,
    *,
    images: int = 7,
    atom_mapping: Sequence[int] | None = None,
    coordinates: Iterable[Primitive] = (),
    start_coordinates: Iterable[Primitive] | None = None,
    end_coordinates: Iterable[Primitive] | None = None,
    align_end: bool = True,
    allow_symmetry_lowering: bool = False,
) -> TransitionStateGuessPath:
    """Build a deterministic two-minimum Cartesian image path.

    ``atom_mapping[i]`` is the atom index in ``end`` corresponding to atom
    ``i`` in ``start``.  Omitting it means the declared order is used; no
    chemical guess or hidden reordering is performed.  Primitive coordinates
    are evaluated in the declared start order and deduplicated by definition.
    No primitive or SONIC coordinate is selected here.  Dihedrals use the
    shortest periodic displacement.

    Symmetry lowering is deliberately not implemented as an implicit fallback:
    requesting it raises until a caller supplies the exact Hessian/mode
    transport contract used by LINK.
    """

    if allow_symmetry_lowering:
        raise PermissionError(
            "TS guess symmetry lowering requires an explicit Hessian-mode request"
        )
    if not isinstance(start, MolecularGeometry) or not isinstance(end, MolecularGeometry):
        raise TypeError("TS-seed endpoints must be MolecularGeometry instances")
    if start.natoms != end.natoms:
        raise TransitionStateGuessError("TS-seed endpoints must contain the same number of atoms")
    start_state = (
        0 if start.charge is None else int(start.charge),
        1 if start.multiplicity is None else int(start.multiplicity),
    )
    end_state = (
        0 if end.charge is None else int(end.charge),
        1 if end.multiplicity is None else int(end.multiplicity),
    )
    if start_state != end_state:
        raise TransitionStateGuessError(
            "TS-seed endpoints must have the same charge and multiplicity"
        )
    mapping = _validate_mapping(start, end, atom_mapping)
    mapped_end_atoms = tuple(end.atoms[index] for index in mapping)
    if mapped_end_atoms != start.atoms:
        raise TransitionStateGuessError(
            "atom mapping changes element identity; provide a chemically consistent bijection"
        )
    if not isinstance(images, int) or images < 3 or images % 2 == 0:
        raise TransitionStateGuessError("images must be an odd integer >= 3")

    end_xyz = end.coordinates_angstrom[np.asarray(mapping, dtype=int)]
    if align_end:
        end_xyz = kabsch_align(end_xyz, start.coordinates_angstrom)
    aligned_error = float(
        np.sqrt(np.mean(np.sum((end_xyz - start.coordinates_angstrom) ** 2, axis=1)))
    )
    aligned_end = MolecularGeometry(
        atoms=start.atoms,
        coordinates_angstrom=end_xyz,
        comment=end.comment,
        source_format=end.source_format,
        source_path=end.source_path,
        charge=end.charge,
        multiplicity=end.multiplicity,
        fixed_parameters=end.fixed_parameters,
        metadata={**end.metadata, "qst2_atom_mapping": mapping},
    )
    start_pool = tuple(coordinates) if start_coordinates is None else tuple(start_coordinates)
    end_pool = (
        start_pool
        if end_coordinates is None
        else tuple(_remap_primitive(primitive, mapping) for primitive in end_coordinates)
    )
    selected = _deduplicate_primitives(start_pool + end_pool)
    union = _coordinate_union(start, aligned_end, selected)
    path_images = tuple(
        TSGuessImage(
            index=index,
            fraction=fraction,
            geometry=MolecularGeometry(
                atoms=start.atoms,
                coordinates_angstrom=(1.0 - fraction) * start.coordinates_angstrom
                + fraction * end_xyz,
                comment=f"TS-seed image {index}/{images - 1} (s={fraction:.8f})",
                source_format="matrix-ts-seed",
                charge=start.charge,
                multiplicity=start.multiplicity,
                metadata={"qst2_fraction": fraction, "qst2_schema": TS_GUESS_SCHEMA},
            ),
        )
        for index, fraction in enumerate(np.linspace(0.0, 1.0, images))
    )
    return TransitionStateGuessPath(
        schema=TS_GUESS_SCHEMA,
        start=start,
        end=aligned_end,
        images=path_images,
        coordinate_union=union,
        atom_mapping=mapping,
        aligned_end_rms_displacement_angstrom=aligned_error,
        symmetry_preserved=True,
        symmetry_lowering_requested=False,
    )


def _validate_mapping(
    start: MolecularGeometry, end: MolecularGeometry, mapping: Sequence[int] | None
) -> tuple[int, ...]:
    values = tuple(range(start.natoms)) if mapping is None else tuple(int(i) for i in mapping)
    if len(values) != start.natoms or sorted(values) != list(range(end.natoms)):
        raise TransitionStateGuessError("atom mapping must be a bijection over the endpoint atoms")
    return values


def _coordinate_union(
    start: MolecularGeometry,
    end: MolecularGeometry,
    coordinates: Iterable[Primitive],
) -> tuple[CoordinateUnionEntry, ...]:
    entries: list[CoordinateUnionEntry] = []
    seen: set[tuple[str, tuple[int, ...], tuple[int, ...]]] = set()
    for primitive in coordinates:
        key = (primitive.kind, tuple(primitive.atoms), tuple(primitive.ref))
        if key in seen:
            continue
        seen.add(key)
        left = float(eval_primitive(primitive, start.coordinates_angstrom))
        right = float(eval_primitive(primitive, end.coordinates_angstrom))
        displacement = _coordinate_delta(primitive.kind, right, left)
        entries.append(
            CoordinateUnionEntry(
                label=primitive.label,
                kind=primitive.kind,
                atoms=tuple(primitive.atoms),
                start_value=left,
                end_value=right,
                displacement=displacement,
            )
        )
    return tuple(entries)


def _coordinate_delta(kind: str, target: float, source: float) -> float:
    if kind != "dihedral":
        return target - source
    period = 2.0 * pi
    return (target - source + pi) % period - pi


def _deduplicate_primitives(primitives: Sequence[Primitive]) -> tuple[Primitive, ...]:
    result: list[Primitive] = []
    seen: set[tuple[object, ...]] = set()
    for primitive in primitives:
        key = (primitive.kind, tuple(primitive.atoms), int(primitive.mode), tuple(primitive.ref))
        if key not in seen:
            seen.add(key)
            result.append(primitive)
    return tuple(result)


def _remap_primitive(primitive: Primitive, mapping: Sequence[int]) -> Primitive:
    """Express an end-point primitive in the canonical start-point atom order."""

    inverse = {int(end_index): start_index for start_index, end_index in enumerate(mapping)}
    try:
        atoms = tuple(inverse[int(atom)] for atom in primitive.atoms)
        ref = tuple(inverse[int(atom)] for atom in primitive.ref)
    except KeyError as exc:
        raise TransitionStateGuessError("primitive references an unmapped endpoint atom") from exc
    return Primitive(primitive.kind, atoms, primitive.mode, ref)


__all__ = [
    "TS_GUESS_SCHEMA",
    "CoordinateUnionEntry",
    "TransitionStateGuessError",
    "TSGuessImage",
    "TransitionStateGuessPath",
    "build_transition_state_guess",
]
