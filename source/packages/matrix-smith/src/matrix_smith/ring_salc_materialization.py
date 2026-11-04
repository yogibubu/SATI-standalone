"""Materialize analytic ring SALCs on the native primitive library."""

from __future__ import annotations

from dataclasses import replace
import re

from .analytic_salc import cyclic_out_of_plane_atom_orders
from .contracts import GICForgeContractError
from .models import FrozenGIC, GICPrimitive
from .numerics import _ring_pucker_terms_from_refs


def materialize_ring_out_of_plane_salcs(
    primitives: tuple[GICPrimitive, ...],
    gics: tuple[FrozenGIC, ...],
) -> tuple[tuple[GICPrimitive, ...], tuple[FrozenGIC, ...]]:
    """Replace internal ring-SALC candidates by native Gaussian U support."""

    salcs = tuple(primitive for primitive in primitives if primitive.function == "RPU")
    if not salcs:
        return primitives, gics
    groups: dict[tuple[int, ...], list[GICPrimitive]] = {}
    for salc in salcs:
        groups.setdefault(salc.atoms, []).append(salc)

    occupied = {
        primitive.identifier for primitive in primitives if primitive.function != "RPU"
    }
    native_blocks: dict[tuple[int, ...], tuple[GICPrimitive, ...]] = {}
    expansion_by_salc: dict[str, tuple[tuple[str, float], ...]] = {}
    for ring_atoms, ring_salcs in groups.items():
        template = ring_salcs[0]
        identifier_match = re.fullmatch(r"P(\d+)", template.identifier)
        name_match = re.fullmatch(r"(.*?)(\d+)", template.name)
        if identifier_match is None or name_match is None:
            raise GICForgeContractError(
                f"ring SALC {template.identifier!r} has no stable catalog slot"
            )
        identifier_start = int(identifier_match.group(1))
        name_start = int(name_match.group(2))
        ring_ref = "RING:" + "-".join(str(atom) for atom in ring_atoms)
        native_block = tuple(
            replace(
                template,
                identifier=f"P{identifier_start + index:03d}",
                name=f"{name_match.group(1)}{name_start + index:04d}",
                function="U",
                atoms=atoms,
                refs=(ring_ref,),
            )
            for index, atoms in enumerate(cyclic_out_of_plane_atom_orders(ring_atoms))
        )
        collisions = occupied.intersection(
            primitive.identifier for primitive in native_block
        )
        if collisions:
            raise GICForgeContractError(
                "native-U materialization collides with primitive IDs: "
                + ",".join(sorted(collisions))
            )
        occupied.update(primitive.identifier for primitive in native_block)
        native_blocks[ring_atoms] = native_block
        native_by_atoms = {primitive.atoms: primitive for primitive in native_block}
        for salc in ring_salcs:
            expansion_by_salc[salc.identifier] = tuple(
                (native_by_atoms[atoms].identifier, float(coefficient))
                for coefficient, atoms in _ring_pucker_terms_from_refs(salc)
            )

    output_primitives: list[GICPrimitive] = []
    inserted: set[tuple[int, ...]] = set()
    for primitive in primitives:
        if primitive.function != "RPU":
            output_primitives.append(primitive)
            continue
        if primitive.atoms not in inserted:
            output_primitives.extend(native_blocks[primitive.atoms])
            inserted.add(primitive.atoms)

    output_gics: list[FrozenGIC] = []
    for gic in gics:
        combined: dict[str, float] = {}
        order: list[str] = []
        for primitive_id, outer_coefficient in (
            gic.coefficients or ((gic.primitive_id, 1.0),)
        ):
            terms = expansion_by_salc.get(primitive_id, ((primitive_id, 1.0),))
            for native_id, inner_coefficient in terms:
                if native_id not in combined:
                    order.append(native_id)
                    combined[native_id] = 0.0
                combined[native_id] += float(outer_coefficient) * float(inner_coefficient)
        coefficients = tuple(
            (primitive_id, combined[primitive_id])
            for primitive_id in order
            if abs(combined[primitive_id]) > 1.0e-14
        )
        if not coefficients:
            raise GICForgeContractError(
                f"empty native-U expansion for ring SALC {gic.identifier}"
            )
        output_gics.append(
            replace(
                gic,
                primitive_id=coefficients[0][0],
                gaussian_expression="LINEAR_COMBINATION",
                coefficients=coefficients,
            )
        )
    return tuple(output_primitives), tuple(output_gics)


__all__ = ["materialize_ring_out_of_plane_salcs"]
