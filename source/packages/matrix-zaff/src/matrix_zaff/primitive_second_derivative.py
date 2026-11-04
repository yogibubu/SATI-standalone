"""Analytic primitive derivatives used by the resident SONIC runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class PrimitiveSecondDerivative:
    """Sparse Hessian of one primitive, with global Cartesian indices."""

    entries: tuple[tuple[int, int, float], ...]
    analytic: bool = True


class _Jet2:
    __slots__ = ("val", "grad", "hess")

    def __init__(self, val: float, grad: np.ndarray, hess: np.ndarray):
        self.val = float(val)
        self.grad = np.asarray(grad, dtype=float)
        self.hess = np.asarray(hess, dtype=float)

    def _coerce(self, other: object) -> "_Jet2":
        if isinstance(other, _Jet2):
            return other
        return _Jet2(float(other), np.zeros_like(self.grad), np.zeros_like(self.hess))

    def __add__(self, other: object) -> "_Jet2":
        rhs = self._coerce(other)
        return _Jet2(self.val + rhs.val, self.grad + rhs.grad, self.hess + rhs.hess)

    __radd__ = __add__

    def __sub__(self, other: object) -> "_Jet2":
        rhs = self._coerce(other)
        return _Jet2(self.val - rhs.val, self.grad - rhs.grad, self.hess - rhs.hess)

    def __rsub__(self, other: object) -> "_Jet2":
        return self._coerce(other).__sub__(self)

    def __neg__(self) -> "_Jet2":
        return _Jet2(-self.val, -self.grad, -self.hess)

    def __mul__(self, other: object) -> "_Jet2":
        rhs = self._coerce(other)
        hess = (
            self.val * rhs.hess
            + rhs.val * self.hess
            + np.outer(self.grad, rhs.grad)
            + np.outer(rhs.grad, self.grad)
        )
        return _Jet2(self.val * rhs.val, self.val * rhs.grad + rhs.val * self.grad, hess)

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> "_Jet2":
        return self * _unary(self._coerce(other), lambda x: 1.0 / x, lambda x: -1.0 / x**2, lambda x: 2.0 / x**3)

    def __rtruediv__(self, other: object) -> "_Jet2":
        return self._coerce(other).__truediv__(self)


def analytic_primitive_second_derivative(
    primitive,
    coordinates_angstrom: np.ndarray,
    *,
    reference_coordinates: np.ndarray | None = None,
    oracle_primitive_convention: bool = False,
    zero_tolerance: float = 1.0e-12,
) -> PrimitiveSecondDerivative:
    """Return an analytic sparse primitive Hessian by second-order AD."""

    from matrix_smith.numerics import (
        _angle_component_terms_from_refs,
        _fragment_frame_anchor_atoms,
        _improper_dihedral_atoms,
        _ring_pucker_terms_from_refs,
    )

    coords = np.asarray(coordinates_angstrom, dtype=float)
    function = str(primitive.function).upper()
    if function == "SOOP":
        from matrix_chem.soop_contract import frozen_soop_covector
        frozen_soop_covector(primitive, len(coords))
        return PrimitiveSecondDerivative(())
    terms: list[tuple[float, str, tuple[int, ...], tuple[int, ...]]] = []
    if function == "L" and len(primitive.ref_atoms) == 1:
        # A referenced L is a four-atom angle/dihedral chart, not the
        # laboratory-axis gauge used by an unreferenced three-atom L.
        # Branch offsets in the perpendicular scalar are constant locally
        # and therefore do not contribute to its Cartesian Hessian.
        left, center, right = primitive.atoms
        anchor = primitive.ref_atoms[0]
        if primitive.mode == -1:
            terms.extend(((1.0, "A", (left, center, anchor), ()),
                          (1.0, "A", (anchor, center, right), ())))
        elif primitive.mode == -2:
            terms.extend(((0.5, "D", (left, center, anchor, right), ()),
                          (-0.5, "D", (left, anchor, center, right), ())))
        else:
            raise ValueError("referenced linear bend requires mode -1 or -2")
    elif function == "RPCB":
        terms.extend((float(c), "A", tuple(a), ()) for c, a in _angle_component_terms_from_refs(primitive))
    elif function == "RPCK":
        terms.extend((float(c), "D", tuple(a), ()) for c, a in _ring_pucker_terms_from_refs(primitive))
    elif function == "IMPD":
        terms.append((1.0, "D", tuple(_improper_dihedral_atoms(primitive.atoms)), ()))
    else:
        terms.append((1.0, function, tuple(primitive.atoms), tuple(primitive.ref_atoms)))

    all_atoms: set[int] = set()
    for _coefficient, term_function, atoms, refs in terms:
        all_atoms.update(atoms)
        all_atoms.update(refs)
        if term_function == "FTRANS":
            all_atoms.update(primitive.ref_frame_atoms)
        if term_function == "FROT":
            frame_atoms = tuple(primitive.frame_atoms) or _fragment_frame_anchor_atoms(atoms, coords=coords)
            ref_frame_atoms = tuple(primitive.ref_frame_atoms) or _fragment_frame_anchor_atoms(refs, coords=coords)
            all_atoms.update(frame_atoms)
            all_atoms.update(ref_frame_atoms)
    ordered_atoms = tuple(sorted(all_atoms))
    if not ordered_atoms:
        return PrimitiveSecondDerivative(())
    atom_position = {atom: index for index, atom in enumerate(ordered_atoms)}
    dimension = 3 * len(ordered_atoms)
    jets: dict[int, list[_Jet2]] = {}
    for atom in ordered_atoms:
        row: list[_Jet2] = []
        for axis in range(3):
            local = 3 * atom_position[atom] + axis
            gradient = np.zeros(dimension)
            gradient[local] = 1.0
            row.append(_Jet2(coords[atom - 1, axis], gradient, np.zeros((dimension, dimension))))
        jets[atom] = row

    value = _constant_like(next(iter(jets.values()))[0], 0.0)
    for coefficient, term_function, atoms, refs in terms:
        value += coefficient * _primitive_value_jet(
            term_function,
            atoms,
            refs,
            jets,
            primitive=primitive,
            coords=coords,
            reference_coordinates=reference_coordinates,
            oracle_primitive_convention=oracle_primitive_convention,
        )
    if not np.all(np.isfinite(value.hess)):
        raise FloatingPointError(f"non-finite analytic B derivative for {primitive.identifier}")
    entries: list[tuple[int, int, float]] = []
    for local_a in range(dimension):
        atom_a, axis_a = ordered_atoms[local_a // 3], local_a % 3
        global_a = 3 * (atom_a - 1) + axis_a
        for local_b in range(dimension):
            element = float(value.hess[local_a, local_b])
            if abs(element) <= zero_tolerance:
                continue
            atom_b, axis_b = ordered_atoms[local_b // 3], local_b % 3
            global_b = 3 * (atom_b - 1) + axis_b
            entries.append((global_a, global_b, element))
    return PrimitiveSecondDerivative(tuple(entries))


def _primitive_value_jet(
    function,
    atoms,
    refs,
    jets,
    *,
    primitive,
    coords,
    reference_coordinates,
    oracle_primitive_convention,
):
    points = [jets[atom] for atom in atoms]
    if function == "R":
        return _norm(_sub(points[0], points[1]))
    if function == "A":
        left = _unit(_sub(points[0], points[1]))
        right = _unit(_sub(points[2], points[1]))
        return _acos(_dot(left, right))
    if function == "D":
        p0, p1, p2, p3 = points
        b0 = _neg(_sub(p1, p0))
        b1 = _unit(_sub(p2, p1))
        b2 = _sub(p3, p2)
        v = _sub(b0, _scale(b1, _dot(b0, b1)))
        w = _sub(b2, _scale(b1, _dot(b2, b1)))
        return _atan2(_dot(_cross(b1, v), w), _dot(v, w))
    if function == "U":
        center, plane1, plane2, out = points
        r1 = _sub(plane1, center)
        r2 = _sub(plane2, center)
        rout = _sub(out, center)
        return -_asin(_dot(_unit(rout), _unit(_cross(r1, r2))))
    if function == "H":
        center, plane1, plane2, out = points
        r1 = _sub(plane1, center)
        r2 = _sub(plane2, center)
        rout = _sub(out, center)
        return _dot(rout, _unit(_cross(r1, r2)))
    if function == "L":
        left = _unit(_sub(points[0], points[1]))
        right = _unit(_sub(points[2], points[1]))
        trial = (0.0, 1.0, 0.0) if abs(left[0].val) > 0.9 else (1.0, 0.0, 0.0)
        trial_jet = [_constant_like(left[0], item) for item in trial]
        first = _unit(_cross(left, trial_jet))
        second = _cross(left, first)
        return _dot(_add(left, right), first if primitive.mode == -1 else second)
    if function in {"FC_DIST", "FCA_DIST", "CENTER_ATOM_DIST"}:
        left = _center([jets[atom] for atom in atoms])
        right = _center([jets[atom] for atom in refs])
        return _norm(_sub(left, right))
    if function == "FTRANS":
        left = _center([jets[atom] for atom in atoms])
        right = _center([jets[atom] for atom in refs])
        delta = _sub(left, right)
        if primitive.ref_frame_atoms:
            if len(primitive.ref_frame_atoms) != 2 or primitive.mode not in (0, 1, 2):
                raise ValueError("body-fixed FTRANS requires two reference anchors and mode 0, 1 or 2")
            frame = _fragment_frame_jet(jets, refs, tuple(primitive.ref_frame_atoms))
            return _dot(delta, frame[primitive.mode])
        return delta[primitive.mode]
    if function == "FROT":
        from matrix_smith.numerics import (
            _fragment_frame,
            _fragment_frame_anchor_atoms,
            _linear_fragment_stereographic_axes,
        )

        frame_atoms = tuple(primitive.frame_atoms) or _fragment_frame_anchor_atoms(atoms, coords=coords)
        ref_frame_atoms = tuple(primitive.ref_frame_atoms) or _fragment_frame_anchor_atoms(refs, coords=coords)
        frame_ref = _fragment_frame_jet(jets, refs, ref_frame_atoms)
        if len(frame_atoms) == 1:
            selection_coords = (
                coords
                if reference_coordinates is None
                else np.asarray(reference_coordinates, dtype=float)
            )
            pole_index, pole_sign, transverse = _linear_fragment_stereographic_axes(
                selection_coords,
                atoms,
                frame_atoms[0],
                refs,
                ref_frame_atoms,
            )
            center = _center([jets[atom] for atom in atoms])
            direction = _unit(_sub(jets[frame_atoms[0]], center))
            components = tuple(_dot(direction, axis) for axis in frame_ref)
            denominator = 1.0 + pole_sign * components[pole_index]
            zero = _constant_like(components[0], 0.0)
            values = (
                components[transverse[0]] / denominator,
                components[transverse[1]] / denominator,
                zero,
            )
            return values[primitive.mode]
        frame_frag = _fragment_frame_jet(jets, atoms, frame_atoms)
        rotation = [[_dot(frame_frag[i], frame_ref[j]) for j in range(3)] for i in range(3)]
        if reference_coordinates is not None:
            reference_frag = _fragment_frame(np.asarray(reference_coordinates), atoms, frame_atoms=frame_atoms)
            reference_ref = _fragment_frame(np.asarray(reference_coordinates), refs, frame_atoms=ref_frame_atoms)
            reference_rotation = reference_frag.T @ reference_ref
            rotation = [[sum(rotation[i][k] * reference_rotation[j, k] for k in range(3)) for j in range(3)] for i in range(3)]
        return _rotation_vector(rotation)[primitive.mode]
    raise NotImplementedError(function)


def _fragment_frame_jet(jets, atoms, frame_atoms):
    center = _center([jets[atom] for atom in atoms])
    p_axis = _unit(_sub(jets[frame_atoms[0]], center))
    q_axis = _unit(_cross(p_axis, _sub(jets[frame_atoms[1]], center)))
    return [p_axis, q_axis, _unit(_cross(p_axis, q_axis))]


def _rotation_vector(rotation):
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace.val <= -1.0 + 1.0e-10:
        raise FloatingPointError("fragment exponential map is singular near 180 degrees")
    kw = 0.5 * _sqrt(trace + 1.0)
    denom = 4.0 * kw
    kx = (rotation[1][2] - rotation[2][1]) / denom
    ky = (rotation[2][0] - rotation[0][2]) / denom
    kz = (rotation[0][1] - rotation[1][0]) / denom
    norm2 = kx * kx + ky * ky + kz * kz
    if norm2.val <= 1.0e-20:
        return 2.0 * kx, 2.0 * ky, 2.0 * kz
    norm = _sqrt(norm2)
    factor = 2.0 * _atan2(norm, kw) / norm
    return factor * kx, factor * ky, factor * kz


def _unary(value: _Jet2, f: Callable[[float], float], fp, fpp) -> _Jet2:
    return _Jet2(
        f(value.val),
        fp(value.val) * value.grad,
        fp(value.val) * value.hess + fpp(value.val) * np.outer(value.grad, value.grad),
    )


def _sqrt(value):
    if value.val <= 0.0:
        raise FloatingPointError("square root derivative is singular")
    return _unary(value, np.sqrt, lambda x: 0.5 / np.sqrt(x), lambda x: -0.25 / x**1.5)


def _acos(value):
    if abs(value.val) >= 1.0:
        raise FloatingPointError("acos derivative is singular")
    return _unary(value, np.arccos, lambda x: -1.0 / np.sqrt(1.0 - x * x), lambda x: -x / (1.0 - x * x) ** 1.5)


def _asin(value):
    if abs(value.val) >= 1.0:
        raise FloatingPointError("asin derivative is singular")
    return _unary(value, np.arcsin, lambda x: 1.0 / np.sqrt(1.0 - x * x), lambda x: x / (1.0 - x * x) ** 1.5)


def _atan2(y, x):
    radius2 = x * x + y * y
    if radius2.val <= 1.0e-20:
        raise FloatingPointError("atan2 derivative is singular")
    # Use the better-conditioned local chart.  The additive quadrant constant
    # has zero derivatives, so restoring atan2's value is sufficient.
    if abs(x.val) >= abs(y.val):
        ratio = y / x
        result = _atan(ratio)
    else:
        ratio = x / y
        result = -_atan(ratio)
    return _Jet2(float(np.arctan2(y.val, x.val)), result.grad, result.hess)


def _atan(value):
    return _unary(
        value,
        np.arctan,
        lambda z: 1.0 / (1.0 + z * z),
        lambda z: -2.0 * z / (1.0 + z * z) ** 2,
    )


def _constant_like(template, value):
    return _Jet2(value, np.zeros_like(template.grad), np.zeros_like(template.hess))


def _add(a, b):
    return [a[i] + b[i] for i in range(3)]


def _sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def _neg(a):
    return [-item for item in a]


def _scale(a, factor):
    return [factor * item for item in a]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _norm(a):
    return _sqrt(_dot(a, a))


def _unit(a):
    return _scale(a, 1.0 / _norm(a))


def _center(points):
    if not points:
        raise FloatingPointError("empty fragment center")
    return _scale([sum(point[axis] for point in points) for axis in range(3)], 1.0 / len(points))
