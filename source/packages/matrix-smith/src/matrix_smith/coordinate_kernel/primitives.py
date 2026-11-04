"""Compatibility facade for ORACLE and frozen SMITH primitive coordinates.

MORPHEUS and the legacy coordinate consumers consume the compact ``Primitive``
contract.  A primitive carrying ``smith_function`` is nevertheless evaluated
by SMITH's complete frozen-coordinate atlas.  Consequently every family
admitted by SMITH has the same scalar value and Wilson row in every consumer;
no downstream package needs to maintain a second function whitelist.
"""

from __future__ import annotations

import numpy as np

from matrix_chem.primitive_coordinates import (
    Primitive,
    build_primitives,
    eval_primitive as _chem_eval_primitive,
    eval_primitives as _chem_eval_primitives,
    grad_primitive as _chem_grad_primitive,
)


_SMITH_KIND = {
    "R": "bond",
    "IR": "inverse_bond",
    "IR3": "inverse_cubic_bond",
    "ER": "exponential_bond",
    "A": "angle",
    "L": "linear_bend",
    "D": "dihedral",
    "IMPD": "dihedral",
    "U": "out_of_plane",
    "H": "out_of_plane_height",
    "SOOP": "selective_out_of_plane",
    "FC_DIST": "frag_dist",
    "FCA_DIST": "frag_atom_dist",
    "CENTER_ATOM_DIST": "frag_atom_dist",
    "FTRANS": "frag_trans",
    "FLIN_TRANS": "frag_linear_trans",
    "FROT": "frag_rot",
    "FAXIS": "frag_axis",
}


def primitive_from_gic(primitive, *, reference_coordinates_angstrom=None) -> Primitive:
    """Preserve the numerical contract of one frozen SMITH primitive."""

    function = str(primitive.function).upper()
    family = str(primitive.family)
    kind = _SMITH_KIND.get(function, f"smith_{function.lower()}")
    if family.upper() == "HBOND_DISTANCE" and function == "R":
        kind = "hbond_dist"
    reference = ()
    if reference_coordinates_angstrom is not None:
        reference = tuple(
            tuple(float(value) for value in row)
            for row in np.asarray(reference_coordinates_angstrom, dtype=float)
        )
    return Primitive(
        kind=kind,
        atoms=tuple(int(atom) - 1 for atom in primitive.atoms),
        mode=int(primitive.mode),
        ref=tuple(int(atom) - 1 for atom in primitive.ref_atoms),
        frame_atoms=tuple(int(atom) - 1 for atom in primitive.frame_atoms),
        ref_frame_atoms=tuple(int(atom) - 1 for atom in primitive.ref_frame_atoms),
        rotation_reference=reference,
        refs=tuple(str(value) for value in primitive.refs),
        chart=str(primitive.chart),
        smith_function=function,
        smith_family=family,
        smith_identifier=str(primitive.identifier),
        chart_reference_radian=primitive.chart_reference_radian,
    )


def _as_gic_primitive(primitive: Primitive):
    from matrix_smith.models import GICPrimitive

    return GICPrimitive(
        identifier=primitive.smith_identifier or primitive.label,
        name=primitive.smith_identifier or primitive.label,
        family=primitive.smith_family or primitive.kind.upper(),
        function=primitive.smith_function,
        atoms=tuple(int(atom) + 1 for atom in primitive.atoms),
        mode=int(primitive.mode),
        ref_atoms=tuple(int(atom) + 1 for atom in primitive.ref),
        refs=tuple(str(value) for value in primitive.refs),
        frame_atoms=tuple(int(atom) + 1 for atom in primitive.frame_atoms),
        ref_frame_atoms=tuple(int(atom) + 1 for atom in primitive.ref_frame_atoms),
        chart=str(primitive.chart),
        chart_reference_radian=primitive.chart_reference_radian,
    )


def eval_primitive(primitive: Primitive, coords) -> float:
    if not primitive.smith_function:
        return float(_chem_eval_primitive(primitive, coords))
    from matrix_smith.numerics import _primitive_value

    reference = primitive.rotation_reference or None
    return float(
        _primitive_value(
            _as_gic_primitive(primitive),
            np.asarray(coords, dtype=float),
            reference_coords=None if reference is None else np.asarray(reference, dtype=float),
        )
    )


def grad_primitive(primitive: Primitive, coords, fd_step: float = 1.0e-4):
    if not primitive.smith_function:
        return _chem_grad_primitive(primitive, coords, fd_step)
    from matrix_smith.numerics import _analytic_b_row

    reference = primitive.rotation_reference or None
    row = _analytic_b_row(
        _as_gic_primitive(primitive),
        np.asarray(coords, dtype=float),
        reference_coords=None if reference is None else np.asarray(reference, dtype=float),
    )
    return np.asarray(row, dtype=float).reshape(np.asarray(coords).shape)


def eval_primitives(primitives, coords):
    items = tuple(primitives)
    if all(not item.smith_function for item in items):
        return _chem_eval_primitives(items, coords)
    return np.asarray([eval_primitive(item, coords) for item in items], dtype=float)

__all__ = [
    "Primitive",
    "build_primitives",
    "eval_primitive",
    "eval_primitives",
    "grad_primitive",
    "primitive_from_gic",
]
