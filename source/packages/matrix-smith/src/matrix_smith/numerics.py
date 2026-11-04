"""Analytic primitive-coordinate values and B-matrix kernels for SMITH."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from matrix_chem import (
    Primitive,
    grad_primitive,
    is_linear_geometry,
    referenced_linear_bend_perpendicular_value,
)

from .bmatrix import SparseBRow
from .contracts import GICForgeContractError
from .models import GICPrimitive
from .policy import DIAGNOSTIC_FINITE_DIFFERENCE_STEP, RANK_TOLERANCE


def _angle_component_terms_from_refs(
    primitive: GICPrimitive,
) -> tuple[tuple[float, tuple[int, int, int]], ...]:
    terms: list[tuple[float, tuple[int, int, int]]] = []
    for ref in primitive.refs:
        if ":" not in ref:
            raise GICForgeContractError(
                f"invalid RPCB term {ref!r} in primitive {primitive.identifier}"
            )
        coefficient_text, atom_text = ref.split(":", 1)
        try:
            coefficient = float(coefficient_text)
            atoms = tuple(int(atom) for atom in atom_text.split("-") if atom)
        except ValueError as exc:
            raise GICForgeContractError(
                f"invalid RPCB term {ref!r} in primitive {primitive.identifier}"
            ) from exc
        if len(atoms) != 3:
            raise GICForgeContractError(
                f"invalid RPCB angle term {ref!r} in primitive {primitive.identifier}"
            )
        terms.append((coefficient, atoms))
    return tuple(terms)


def _ring_pucker_terms_from_refs(
    primitive: GICPrimitive,
) -> tuple[tuple[float, tuple[int, int, int, int]], ...]:
    terms: list[tuple[float, tuple[int, int, int, int]]] = []
    for ref in primitive.refs:
        if ":" not in ref:
            raise GICForgeContractError(
                f"invalid RPck term {ref!r} in primitive {primitive.identifier}"
            )
        coefficient_text, atom_text = ref.split(":", 1)
        try:
            coefficient = float(coefficient_text)
            atoms = tuple(int(atom) for atom in atom_text.split("-") if atom)
        except ValueError as exc:
            raise GICForgeContractError(
                f"invalid RPck term {ref!r} in primitive {primitive.identifier}"
            ) from exc
        if len(atoms) != 4:
            raise GICForgeContractError(
                f"invalid RPck dihedral term {ref!r} in primitive {primitive.identifier}"
            )
        terms.append((coefficient, atoms))
    return tuple(terms)


def _sparse_analytic_b_row(
    primitive: GICPrimitive,
    coords: np.ndarray,
    *,
    reference_coords: np.ndarray | None = None,
) -> SparseBRow:
    coords = np.asarray(coords, dtype=float)
    natoms = int(coords.shape[0])
    # Face/contact primitives are implemented by the analytic face kernel.
    # Keep their support local while accepting the one-based ORACLE atom
    # numbering used by GICPrimitive; routing them through the generic
    # fallback used to subtract one twice and reject valid FACE_FLAP rows.
    from matrix_chem.face_primitives import FACE_OPERATORS
    if primitive.function in FACE_OPERATORS.values():
        return SparseBRow.from_dense(_analytic_b_row(primitive, coords))
    if primitive.function in {"R", "IR", "IR3", "ER"}:
        i, j = (atom - 1 for atom in primitive.atoms)
        delta = coords[i] - coords[j]
        distance = float(np.linalg.norm(delta))
        if distance <= RANK_TOLERANCE:
            raise FloatingPointError("zero-length distance coordinate")
        unit = delta / distance
        if primitive.function in {"IR", "IR3"}:
            if primitive.function == "IR":
                unit = -unit / distance**2
            else:
                # The radial derivative of r^-3 carries the exponent 3.
                unit = -3.0 * unit / distance**4
        elif primitive.function == "ER":
            unit *= -np.exp(-distance)
        return SparseBRow.from_atom_gradients(
            natoms,
            ((primitive.atoms[0], unit), (primitive.atoms[1], -unit)),
        )
    if primitive.function in {"FC_DIST"}:
        center = _fragment_center(coords, primitive.atoms)
        ref_center = _fragment_center(coords, primitive.ref_atoms)
        delta = center - ref_center
        distance = float(np.linalg.norm(delta))
        if distance <= RANK_TOLERANCE:
            raise FloatingPointError("coincident fragment centers")
        unit = delta / distance
        gradients = tuple((atom, unit / len(primitive.atoms)) for atom in primitive.atoms) + tuple(
            (atom, -unit / len(primitive.ref_atoms)) for atom in primitive.ref_atoms
        )
        return SparseBRow.from_atom_gradients(natoms, gradients)
    if primitive.function in {"FCA_DIST", "CENTER_ATOM_DIST"}:
        if len(primitive.ref_atoms) != 1:
            raise FloatingPointError("center-atom distance needs exactly one reference atom")
        atom = primitive.ref_atoms[0]
        delta = _fragment_center(coords, primitive.atoms) - coords[atom - 1]
        distance = float(np.linalg.norm(delta))
        if distance <= RANK_TOLERANCE:
            raise FloatingPointError("fragment center and atom are coincident")
        unit = delta / distance
        gradients = tuple((item, unit / len(primitive.atoms)) for item in primitive.atoms) + (
            (atom, -unit),
        )
        return SparseBRow.from_atom_gradients(natoms, gradients)
    if primitive.function in {"FTRANS", "FLIN_TRANS"}:
        if primitive.ref_frame_atoms:
            return SparseBRow.from_dense(
                _dual_primitive_value(
                    primitive,
                    _dual_coordinates(coords),
                    coords,
                    reference_coords=reference_coords,
                ).der
            )
        axis = np.zeros(3, dtype=float)
        axis[primitive.mode] = 1.0
        gradients = tuple((atom, axis / len(primitive.atoms)) for atom in primitive.atoms) + tuple(
            (atom, -axis / len(primitive.ref_atoms)) for atom in primitive.ref_atoms
        )
        return SparseBRow.from_atom_gradients(natoms, gradients)
    if primitive.function in {"A", "D", "U", "L", "H"}:
        support = tuple(
            dict.fromkeys(
                (
                    *primitive.atoms,
                    *primitive.ref_atoms,
                    *primitive.frame_atoms,
                    *primitive.ref_frame_atoms,
                )
            )
        )
        local_index = {atom: index + 1 for index, atom in enumerate(support)}
        local = replace(
            primitive,
            atoms=tuple(local_index[atom] for atom in primitive.atoms),
            ref_atoms=tuple(local_index[atom] for atom in primitive.ref_atoms),
            frame_atoms=tuple(local_index[atom] for atom in primitive.frame_atoms),
            ref_frame_atoms=tuple(
                local_index[atom] for atom in primitive.ref_frame_atoms
            ),
        )
        local_row = _analytic_b_row(
            local,
            coords[np.asarray([atom - 1 for atom in support], dtype=int)],
            reference_coords=(
                None
                if reference_coords is None
                else np.asarray(reference_coords, dtype=float)[
                    np.asarray([atom - 1 for atom in support], dtype=int)
                ]
            ),
        ).reshape((-1, 3))
        return SparseBRow.from_atom_gradients(
            natoms,
            tuple(
                (atom, local_row[index]) for index, atom in enumerate(support)
            ),
        )
    return SparseBRow.from_dense(
        _analytic_b_row(primitive, coords, reference_coords=reference_coords)
    )


def _analytic_b_row(
    primitive: GICPrimitive,
    coords: np.ndarray,
    *,
    reference_coords: np.ndarray | None = None,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if primitive.function in {"SOOP", "SYCART"}:
        from matrix_chem.soop_contract import frozen_soop_covector
        return frozen_soop_covector(primitive, len(coords))[0]
    from matrix_chem.face_primitives import FACE_OPERATORS, face_from_smith
    if primitive.function in FACE_OPERATORS.values():
        from matrix_chem.primitive_coordinates import grad_primitive
        return grad_primitive(face_from_smith(primitive), coords).reshape(-1)
    if primitive.function == "R":
        return _distance_b_row(coords, primitive.atoms)
    if primitive.function == "IR":
        row = _distance_b_row(coords, primitive.atoms)
        i, j = primitive.atoms[0] - 1, primitive.atoms[1] - 1
        distance = float(np.linalg.norm(coords[i] - coords[j]))
        if distance <= RANK_TOLERANCE:
            raise FloatingPointError("inverse distance is singular")
        return row / -(distance * distance)
    if primitive.function == "IR3":
        row = _distance_b_row(coords, primitive.atoms)
        i, j = primitive.atoms[0] - 1, primitive.atoms[1] - 1
        distance = float(np.linalg.norm(coords[i] - coords[j]))
        if distance <= RANK_TOLERANCE:
            raise FloatingPointError("inverse cubic distance is singular")
        # d(r^-3)/dr = -3 r^-4; the missing factor of three made the
        # compiled SMITH IR3 rows disagree with the analytic Python chart.
        return row * (-3.0 / distance**4)
    if primitive.function == "ER":
        row = _distance_b_row(coords, primitive.atoms)
        i, j = primitive.atoms[0] - 1, primitive.atoms[1] - 1
        distance = float(np.linalg.norm(coords[i] - coords[j]))
        return row * -np.exp(-distance)
    if primitive.function == "A":
        return _angle_b_row(coords, primitive.atoms)
    if primitive.function == "FC_DIST":
        return _fragment_center_distance_b_row(coords, primitive.atoms, primitive.ref_atoms)
    if primitive.function == "FCA_DIST":
        return _fragment_center_atom_distance_b_row(coords, primitive.atoms, primitive.ref_atoms)
    if primitive.function == "CENTER_ATOM_DIST":
        return _fragment_center_atom_distance_b_row(coords, primitive.atoms, primitive.ref_atoms)
    if primitive.function in {"FTRANS", "FLIN_TRANS"}:
        if primitive.ref_frame_atoms:
            return _dual_primitive_value(
                primitive,
                _dual_coordinates(coords),
                coords,
                reference_coords=reference_coords,
            ).der
        return _fragment_translation_b_row(
            coords,
            primitive.atoms,
            primitive.ref_atoms,
            mode=primitive.mode,
        )
    if primitive.function == "RPCB":
        return _angle_component_b_row(primitive, coords)
    if primitive.function == "RPCK":
        return _ring_pucker_component_b_row(primitive, coords)
    if primitive.function == "RPU":
        return _ring_out_of_plane_component_b_row(primitive, coords)
    if primitive.function in {"L", "D", "IMPD", "U", "H"}:
        try:
            return _local_valence_b_row(primitive, coords)
        except (ArithmeticError, FloatingPointError):
            return _dual_b_row(
                primitive,
                coords,
                reference_coords=reference_coords,
            )
    return _dual_b_row(primitive, coords, reference_coords=reference_coords)


def _local_valence_b_row(
    primitive: GICPrimitive,
    coords: np.ndarray,
) -> np.ndarray:
    """Evaluate compact-support rows without full-molecule dual arrays."""

    atoms = (
        _improper_dihedral_atoms(primitive.atoms)
        if primitive.function == "IMPD"
        else primitive.atoms
    )
    kind = {
        "L": "linear_bend",
        "D": "dihedral",
        "IMPD": "dihedral",
        "U": "out_of_plane",
        "H": "out_of_plane_height",
    }[primitive.function]
    zero_based = tuple(int(atom) - 1 for atom in atoms)
    zero_based_refs = tuple(int(atom) - 1 for atom in primitive.ref_atoms)
    support = tuple(dict.fromkeys(zero_based + zero_based_refs))
    remap = {atom: index for index, atom in enumerate(support)}
    local = Primitive(
        kind,
        tuple(remap[atom] for atom in zero_based),
        int(primitive.mode),
        tuple(remap[atom] for atom in zero_based_refs),
    )
    with np.errstate(all="raise"):
        local_gradient = grad_primitive(
            local,
            np.asarray(coords, dtype=float)[np.asarray(support, dtype=int)],
        )
    row = np.zeros(np.asarray(coords).size, dtype=float)
    for local_atom, global_atom in enumerate(support):
        row[3 * global_atom : 3 * global_atom + 3] = local_gradient[local_atom]
    return row


def _distance_b_row(coords: np.ndarray, atoms: tuple[int, ...]) -> np.ndarray:
    i, j = (atom - 1 for atom in atoms)
    delta = coords[i] - coords[j]
    distance = float(np.linalg.norm(delta))
    if distance <= RANK_TOLERANCE:
        raise FloatingPointError("zero-length distance coordinate")
    row = np.zeros(coords.size, dtype=float)
    unit = delta / distance
    row[3 * i : 3 * i + 3] = unit
    row[3 * j : 3 * j + 3] = -unit
    return row


def _angle_b_row(coords: np.ndarray, atoms: tuple[int, ...]) -> np.ndarray:
    i, j, k = (atom - 1 for atom in atoms)
    rji = coords[i] - coords[j]
    rjk = coords[k] - coords[j]
    dji = float(np.linalg.norm(rji))
    djk = float(np.linalg.norm(rjk))
    if dji <= RANK_TOLERANCE or djk <= RANK_TOLERANCE:
        raise FloatingPointError("zero-length angle arm")
    eji = rji / dji
    ejk = rjk / djk
    cosine = float(np.clip(np.dot(eji, ejk), -1.0, 1.0))
    sine = float(np.sqrt(max(1.0 - cosine * cosine, 0.0)))
    if sine <= RANK_TOLERANCE:
        raise FloatingPointError("linear angle has no ordinary bend derivative")
    gi = (cosine * eji - ejk) / (dji * sine)
    gk = (cosine * ejk - eji) / (djk * sine)
    row = np.zeros(coords.size, dtype=float)
    row[3 * i : 3 * i + 3] = gi
    row[3 * k : 3 * k + 3] = gk
    row[3 * j : 3 * j + 3] = -(gi + gk)
    return row


def _fragment_center_distance_b_row(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
) -> np.ndarray:
    center = _fragment_center(coords, atoms)
    ref_center = _fragment_center(coords, ref_atoms)
    delta = center - ref_center
    distance = float(np.linalg.norm(delta))
    if distance <= RANK_TOLERANCE:
        raise FloatingPointError("coincident fragment centers")
    unit = delta / distance
    row = np.zeros(coords.size, dtype=float)
    _accumulate_center_gradient(row, atoms, unit / len(atoms))
    _accumulate_center_gradient(row, ref_atoms, -unit / len(ref_atoms))
    return row


def _fragment_center_atom_distance_b_row(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
) -> np.ndarray:
    if len(ref_atoms) != 1:
        raise FloatingPointError("center-atom distance needs exactly one reference atom")
    atom = ref_atoms[0]
    delta = _fragment_center(coords, atoms) - coords[atom - 1]
    distance = float(np.linalg.norm(delta))
    if distance <= RANK_TOLERANCE:
        raise FloatingPointError("fragment center and atom are coincident")
    unit = delta / distance
    row = np.zeros(coords.size, dtype=float)
    _accumulate_center_gradient(row, atoms, unit / len(atoms))
    row[3 * (atom - 1) : 3 * atom] -= unit
    return row


def _fragment_translation_b_row(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
    *,
    mode: int,
) -> np.ndarray:
    row = np.zeros(coords.size, dtype=float)
    axis = np.zeros(3, dtype=float)
    axis[mode] = 1.0
    _accumulate_center_gradient(row, atoms, axis / len(atoms))
    _accumulate_center_gradient(row, ref_atoms, -axis / len(ref_atoms))
    return row


def _ring_pucker_component_b_row(
    primitive: GICPrimitive,
    coords: np.ndarray,
) -> np.ndarray:
    row = np.zeros(coords.size, dtype=float)
    for coefficient, atoms in _ring_pucker_terms_from_refs(primitive):
        if coefficient == 0.0:
            continue
        term = GICPrimitive(
            identifier=f"{primitive.identifier}_D",
            name="RPckD",
            family="TORSION",
            function="D",
            atoms=atoms,
        )
        row += coefficient * _dual_b_row(term, coords)
    return row


def _ring_out_of_plane_component_b_row(
    primitive: GICPrimitive,
    coords: np.ndarray,
) -> np.ndarray:
    """Evaluate one topology-frozen SALC of native Gaussian U primitives."""

    row = np.zeros(coords.size, dtype=float)
    for coefficient, atoms in _ring_pucker_terms_from_refs(primitive):
        if coefficient == 0.0:
            continue
        term = GICPrimitive(
            identifier=f"{primitive.identifier}_U",
            name="RingU",
            family="RING_PUCKER_COMPONENT",
            function="U",
            atoms=atoms,
        )
        row += coefficient * _local_valence_b_row(term, coords)
    return row


def _angle_component_b_row(
    primitive: GICPrimitive,
    coords: np.ndarray,
) -> np.ndarray:
    row = np.zeros(coords.size, dtype=float)
    for coefficient, atoms in _angle_component_terms_from_refs(primitive):
        if coefficient == 0.0:
            continue
        row += coefficient * _angle_b_row(coords, atoms)
    return row


def _accumulate_center_gradient(
    row: np.ndarray,
    atoms: tuple[int, ...],
    gradient: np.ndarray,
) -> None:
    for atom in atoms:
        start = 3 * (atom - 1)
        row[start : start + 3] += gradient


def _dual_b_row(
    primitive: GICPrimitive,
    coords: np.ndarray,
    *,
    reference_coords: np.ndarray | None = None,
) -> np.ndarray:
    dcoords = _dual_coordinates(coords)
    value = _dual_primitive_value(
        primitive,
        dcoords,
        coords,
        reference_coords=reference_coords,
    )
    if not np.isfinite(value.val):
        raise FloatingPointError("non-finite analytic derivative value")
    return np.asarray(value.der, dtype=float)


def _finite_difference_b_row(
    primitive: GICPrimitive,
    coords: np.ndarray,
    *,
    step_angstrom: float = DIAGNOSTIC_FINITE_DIFFERENCE_STEP,
) -> np.ndarray:
    flat = np.asarray(coords, dtype=float).reshape(-1)
    base = _primitive_value(primitive, coords)
    row = np.zeros_like(flat)
    for idx in range(flat.size):
        plus = flat.copy()
        minus = flat.copy()
        plus[idx] += step_angstrom
        minus[idx] -= step_angstrom
        try:
            value_plus = _primitive_value(primitive, plus.reshape(coords.shape))
            value_minus = _primitive_value(primitive, minus.reshape(coords.shape))
        except FloatingPointError:
            row[idx] = np.nan
            continue
        if primitive.function == "RPCK":
            delta = _ring_pucker_component_periodic_delta(
                primitive,
                plus.reshape(coords.shape),
                minus.reshape(coords.shape),
            )
        elif primitive.function in {"D", "IMPD"}:
            delta = _periodic_delta(value_plus, value_minus)
        else:
            delta = value_plus - value_minus
        if not np.isfinite(delta) or not np.isfinite(base):
            row[idx] = np.nan
        else:
            row[idx] = delta / (2.0 * step_angstrom)
    return row


def _ring_pucker_component_periodic_delta(
    primitive: GICPrimitive,
    plus_coords: np.ndarray,
    minus_coords: np.ndarray,
) -> float:
    delta = 0.0
    for coefficient, atoms in _ring_pucker_terms_from_refs(primitive):
        delta += coefficient * _periodic_delta(
            _dihedral_value(plus_coords, atoms),
            _dihedral_value(minus_coords, atoms),
        )
    return float(delta)


class _Dual:
    __slots__ = ("val", "der")

    def __init__(self, val: float, der: np.ndarray):
        self.val = float(val)
        self.der = np.asarray(der, dtype=float)

    def _coerce(self, other: object) -> "_Dual":
        if isinstance(other, _Dual):
            return other
        return _Dual(float(other), np.zeros_like(self.der))

    def __add__(self, other: object) -> "_Dual":
        rhs = self._coerce(other)
        return _Dual(self.val + rhs.val, self.der + rhs.der)

    def __radd__(self, other: object) -> "_Dual":
        return self.__add__(other)

    def __sub__(self, other: object) -> "_Dual":
        rhs = self._coerce(other)
        return _Dual(self.val - rhs.val, self.der - rhs.der)

    def __rsub__(self, other: object) -> "_Dual":
        lhs = self._coerce(other)
        return _Dual(lhs.val - self.val, lhs.der - self.der)

    def __mul__(self, other: object) -> "_Dual":
        rhs = self._coerce(other)
        return _Dual(self.val * rhs.val, self.val * rhs.der + rhs.val * self.der)

    def __rmul__(self, other: object) -> "_Dual":
        return self.__mul__(other)

    def __truediv__(self, other: object) -> "_Dual":
        rhs = self._coerce(other)
        inv = 1.0 / rhs.val
        return _Dual(self.val * inv, (self.der - self.val * rhs.der * inv) * inv)

    def __rtruediv__(self, other: object) -> "_Dual":
        lhs = self._coerce(other)
        inv = 1.0 / self.val
        return _Dual(lhs.val * inv, (lhs.der - lhs.val * self.der * inv) * inv)

    def __neg__(self) -> "_Dual":
        return _Dual(-self.val, -self.der)


def _dual_coordinates(coords: np.ndarray) -> list[list[_Dual]]:
    flat = np.asarray(coords, dtype=float).reshape(-1)
    dim = flat.size
    out: list[list[_Dual]] = []
    for atom in range(coords.shape[0]):
        row = []
        for axis in range(3):
            idx = 3 * atom + axis
            der = np.zeros(dim, dtype=float)
            der[idx] = 1.0
            row.append(_Dual(float(coords[atom, axis]), der))
        out.append(row)
    return out


def _dual_primitive_value(
    primitive: GICPrimitive,
    dcoords: list[list[_Dual]],
    coords: np.ndarray,
    *,
    reference_coords: np.ndarray | None = None,
) -> _Dual:
    if primitive.function == "L":
        if len(primitive.ref_atoms) == 1:
            return _dual_four_atom_linear_bend_value(
                dcoords,
                primitive.atoms,
                primitive.ref_atoms[0],
                mode=primitive.mode,
            )
        return _dual_linear_bend_value(dcoords, primitive.atoms, mode=primitive.mode)
    if primitive.function == "D":
        return _dual_dihedral_value(dcoords, primitive.atoms)
    if primitive.function == "U":
        return _dual_out_of_plane_value(dcoords, primitive.atoms)
    if primitive.function == "H":
        return _dual_out_of_plane_height_value(dcoords, primitive.atoms)
    if primitive.function == "IMPD":
        return _dual_dihedral_value(dcoords, _improper_dihedral_atoms(primitive.atoms))
    if primitive.function == "FROT":
        return _dual_fragment_rotation_value(
            dcoords,
            coords,
            primitive.atoms,
            primitive.ref_atoms,
            mode=primitive.mode,
            frame_atoms=primitive.frame_atoms,
            ref_frame_atoms=primitive.ref_frame_atoms,
            reference_coords=reference_coords,
        )
    if primitive.function == "FAXIS":
        if len(primitive.frame_atoms) != 1 or len(primitive.ref_frame_atoms) != 1:
            raise GICForgeContractError("FAXIS requires one axis anchor per body")
        return _dual_fragment_axis_axis_values(
            dcoords,
            coords,
            primitive.atoms,
            primitive.ref_atoms,
            frame_atom=primitive.frame_atoms[0],
            ref_frame_atom=primitive.ref_frame_atoms[0],
            reference_coords=reference_coords,
        )[primitive.mode]
    if primitive.function == "FLIN_TRANS":
        if len(primitive.ref_frame_atoms) != 1:
            raise GICForgeContractError("FLIN_TRANS requires one reference-axis anchor")
        return _dual_axial_jacobi_translation_values(
            dcoords,
            primitive.atoms,
            primitive.ref_atoms,
            ref_frame_atom=primitive.ref_frame_atoms[0],
        )[primitive.mode]
    if primitive.function == "FTRANS" and primitive.ref_frame_atoms:
        return _dual_fragment_translation_value(
            dcoords,
            primitive.atoms,
            primitive.ref_atoms,
            mode=primitive.mode,
            ref_frame_atoms=primitive.ref_frame_atoms,
        )
    raise GICForgeContractError(
        f"analytic B row is not implemented for function {primitive.function}"
    )


def _dual_linear_bend_value(
    dcoords: list[list[_Dual]],
    atoms: tuple[int, ...],
    *,
    mode: int,
) -> _Dual:
    i, j, k = (atom - 1 for atom in atoms)
    left = _d_unit(_d_vec_sub(dcoords[i], dcoords[j]))
    right = _d_unit(_d_vec_sub(dcoords[k], dcoords[j]))
    # Keep the linear-bend gauge identical to matrix_chem.linear_components
    # and grad_primitive: the first component is u x trial, with a
    # deterministic trial axis selected from u.  Using (right-left) here
    # defines a different orientation and reverses mode -1 for near-linear
    # bends, breaking the ORACLE/SMITH B-row contract.
    trial = [1.0, 0.0, 0.0]
    if abs(left[0].val) > 0.9:
        trial = [0.0, 1.0, 0.0]
    trial_dual = [_d_constant(left[0], value) for value in trial]
    e1 = _d_unit(_d_cross(left, trial_dual))
    e2 = _d_unit(_d_cross(left, e1))
    bend = _d_vec_add(left, right)
    return _d_dot(bend, e1 if mode == -1 else e2)


def _dual_angle_value(
    dcoords: list[list[_Dual]],
    atoms: tuple[int, int, int],
) -> _Dual:
    i, j, k = (atom - 1 for atom in atoms)
    left = _d_unit(_d_vec_sub(dcoords[i], dcoords[j]))
    right = _d_unit(_d_vec_sub(dcoords[k], dcoords[j]))
    return _d_acos(_d_dot(left, right))


def _dual_four_atom_linear_bend_value(
    dcoords: list[list[_Dual]],
    atoms: tuple[int, ...],
    reference: int,
    *,
    mode: int,
) -> _Dual:
    endpoint1, center, endpoint2 = atoms
    if mode == -1:
        return _dual_angle_value(
            dcoords, (endpoint1, center, reference)
        ) + _dual_angle_value(dcoords, (reference, center, endpoint2))
    if mode == -2:
        first = _dual_dihedral_value(
            dcoords, (endpoint1, center, reference, endpoint2)
        )
        second = _dual_dihedral_value(
            dcoords, (endpoint1, reference, center, endpoint2)
        )
        value = referenced_linear_bend_perpendicular_value(first.val, second.val)
        return _Dual(value, 0.5 * (first.der - second.der))
    raise GICForgeContractError(f"invalid linear-bend mode: {mode}")


def _dual_dihedral_value(dcoords: list[list[_Dual]], atoms: tuple[int, ...]) -> _Dual:
    i, j, k, ell = (atom - 1 for atom in atoms)
    p0, p1, p2, p3 = dcoords[i], dcoords[j], dcoords[k], dcoords[ell]
    b0 = _d_vec_neg(_d_vec_sub(p1, p0))
    b1 = _d_vec_sub(p2, p1)
    b2 = _d_vec_sub(p3, p2)
    b1 = _d_unit(b1)
    v = _d_vec_sub(b0, _d_vec_scale(b1, _d_dot(b0, b1)))
    w = _d_vec_sub(b2, _d_vec_scale(b1, _d_dot(b2, b1)))
    x = _d_dot(v, w)
    y = _d_dot(_d_cross(b1, v), w)
    # The projected-vector construction reverses both plane normals relative
    # to the canonical ORACLE/matrix-chem dihedral.  Negating both atan2
    # arguments preserves the derivative and places the value on that chart.
    return _d_atan2(-y, -x)


def _dual_out_of_plane_value(
    dcoords: list[list[_Dual]],
    atoms: tuple[int, ...],
) -> _Dual:
    center, plane1, plane2, out = (atom - 1 for atom in atoms)
    r1 = _d_vec_sub(dcoords[plane1], dcoords[center])
    r2 = _d_vec_sub(dcoords[plane2], dcoords[center])
    rout = _d_vec_sub(dcoords[out], dcoords[center])
    normal = _d_unit(_d_cross(r1, r2))
    return -_d_asin(_d_dot(_d_unit(rout), normal))


def _dual_out_of_plane_height_value(
    dcoords: list[list[_Dual]],
    atoms: tuple[int, ...],
) -> _Dual:
    center, plane1, plane2, out = (atom - 1 for atom in atoms)
    r1 = _d_vec_sub(dcoords[plane1], dcoords[center])
    r2 = _d_vec_sub(dcoords[plane2], dcoords[center])
    rout = _d_vec_sub(dcoords[out], dcoords[center])
    return _d_dot(rout, _d_unit(_d_cross(r1, r2)))


def _dual_fragment_translation_value(
    dcoords: list[list[_Dual]],
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
    *,
    mode: int,
    ref_frame_atoms: tuple[int, ...],
) -> _Dual:
    """Body-fixed fragment-center translation in the reference-fragment frame."""

    moving_center = _d_fragment_center(dcoords, atoms)
    reference_center = _d_fragment_center(dcoords, ref_atoms)
    delta = _d_vec_sub(moving_center, reference_center)
    reference_frame = _d_fragment_frame(
        dcoords,
        ref_atoms,
        frame_atoms=ref_frame_atoms,
    )
    return _d_dot(delta, reference_frame[int(mode)])


def _dual_axial_jacobi_frame(
    dcoords: list[list[_Dual]],
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
    *,
    ref_frame_atom: int,
) -> tuple[list[_Dual], list[_Dual], list[_Dual], list[_Dual]]:
    """Return center displacement and a rotation-invariant linear-body gauge."""

    moving_center = _d_fragment_center(dcoords, atoms)
    reference_center = _d_fragment_center(dcoords, ref_atoms)
    delta = _d_vec_sub(moving_center, reference_center)
    primary = _d_unit(_d_vec_sub(dcoords[ref_frame_atom - 1], reference_center))
    parallel = _d_dot(delta, primary)
    transverse = _d_vec_sub(delta, _d_vec_scale(primary, parallel))
    secondary = _d_unit(transverse)
    normal = _d_unit(_d_cross(primary, secondary))
    return delta, primary, secondary, normal


def _dual_axial_jacobi_translation_values(
    dcoords: list[list[_Dual]],
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
    *,
    ref_frame_atom: int,
) -> tuple[_Dual, _Dual]:
    delta, primary, secondary, _normal = _dual_axial_jacobi_frame(
        dcoords,
        atoms,
        ref_atoms,
        ref_frame_atom=ref_frame_atom,
    )
    return _d_dot(delta, primary), _d_dot(delta, secondary)


def _axis_axis_stereographic_chart(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
    *,
    frame_atom: int,
    ref_frame_atom: int,
) -> tuple[int, float, tuple[int, int]]:
    frame = _axial_jacobi_frame(
        coords,
        atoms,
        ref_atoms,
        ref_frame_atom=ref_frame_atom,
    )[1:]
    moving_center = _fragment_center(coords, atoms)
    direction = _unit(coords[frame_atom - 1] - moving_center)
    components = np.asarray([float(direction @ axis) for axis in frame])
    pole_index = max(range(3), key=lambda index: (abs(components[index]), -index))
    pole_sign = 1.0 if components[pole_index] >= 0.0 else -1.0
    transverse = tuple(index for index in range(3) if index != pole_index)
    return pole_index, pole_sign, (transverse[0], transverse[1])


def _dual_fragment_axis_axis_values(
    dcoords: list[list[_Dual]],
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
    *,
    frame_atom: int,
    ref_frame_atom: int,
    reference_coords: np.ndarray | None,
) -> tuple[_Dual, _Dual]:
    selection = coords if reference_coords is None else np.asarray(reference_coords, dtype=float)
    pole_index, pole_sign, transverse = _axis_axis_stereographic_chart(
        selection,
        atoms,
        ref_atoms,
        frame_atom=frame_atom,
        ref_frame_atom=ref_frame_atom,
    )
    _delta, primary, secondary, normal = _dual_axial_jacobi_frame(
        dcoords,
        atoms,
        ref_atoms,
        ref_frame_atom=ref_frame_atom,
    )
    moving_center = _d_fragment_center(dcoords, atoms)
    direction = _d_unit(_d_vec_sub(dcoords[frame_atom - 1], moving_center))
    components = tuple(
        _d_dot(direction, axis) for axis in (primary, secondary, normal)
    )
    denominator = 1.0 + pole_sign * components[pole_index]
    if denominator.val <= 1.0e-12:
        raise FloatingPointError("axis--axis stereographic chart reached its antipode")
    return (
        components[transverse[0]] / denominator,
        components[transverse[1]] / denominator,
    )


def _dual_fragment_rotation_value(
    dcoords: list[list[_Dual]],
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
    *,
    mode: int,
    frame_atoms: tuple[int, ...] = (),
    ref_frame_atoms: tuple[int, ...] = (),
    reference_coords: np.ndarray | None = None,
) -> _Dual:
    return _dual_fragment_rotation_values(
        dcoords,
        coords,
        atoms,
        ref_atoms,
        frame_atoms=frame_atoms,
        ref_frame_atoms=ref_frame_atoms,
        reference_coords=reference_coords,
    )[mode]


def _dual_fragment_rotation_values(
    dcoords: list[list[_Dual]],
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
    *,
    frame_atoms: tuple[int, ...] = (),
    ref_frame_atoms: tuple[int, ...] = (),
    reference_coords: np.ndarray | None = None,
) -> tuple[_Dual, _Dual, _Dual]:
    """Evaluate a complete FROT triplet from one dual-coordinate tape."""

    ref_frame_atoms = ref_frame_atoms or _fragment_frame_anchor_atoms(ref_atoms, coords=coords)
    frame_ref = _d_fragment_frame(dcoords, ref_atoms, frame_atoms=ref_frame_atoms)
    if len(frame_atoms) == 1:
        selection_coords = (
            coords if reference_coords is None else np.asarray(reference_coords, dtype=float)
        )
        pole_index, pole_sign, transverse = _linear_fragment_stereographic_axes(
            selection_coords,
            atoms,
            frame_atoms[0],
            ref_atoms,
            ref_frame_atoms,
        )
        center = _d_fragment_center(dcoords, atoms)
        direction = _d_unit(_d_vec_sub(dcoords[frame_atoms[0] - 1], center))
        components = tuple(_d_dot(direction, axis) for axis in frame_ref)
        denominator = 1.0 + pole_sign * components[pole_index]
        zero = _d_zero_like(components[0])
        return (
            components[transverse[0]] / denominator,
            components[transverse[1]] / denominator,
            zero,
        )

    frame_atoms = frame_atoms or _fragment_frame_anchor_atoms(atoms, coords=coords)
    frame_frag = _d_fragment_frame(dcoords, atoms, frame_atoms=frame_atoms)
    rotation = [
        [_d_dot(frame_frag[left], frame_ref[right]) for right in range(3)] for left in range(3)
    ]
    if reference_coords is not None:
        reference_frag, reference_ref = _fragment_relative_frames(
            np.asarray(reference_coords, dtype=float),
            atoms,
            ref_atoms,
            frame_atoms=frame_atoms,
            ref_frame_atoms=ref_frame_atoms,
            gauge_reference_coords=np.asarray(reference_coords, dtype=float),
        )
        reference_rotation = reference_frag.T @ reference_ref
        rotation = [
            [
                sum(rotation[left][axis] * reference_rotation[right, axis] for axis in range(3))
                for right in range(3)
            ]
            for left in range(3)
        ]
    return _d_rotation_vector(rotation)


def _d_fragment_frame(
    dcoords: list[list[_Dual]],
    atoms: tuple[int, ...],
    *,
    frame_atoms: tuple[int, ...],
) -> list[list[_Dual]]:
    p_atom, q_atom = frame_atoms
    center = _d_fragment_center(dcoords, atoms)
    p_axis = _d_unit(_d_vec_sub(dcoords[p_atom - 1], center))
    q_raw = _d_cross(p_axis, _d_vec_sub(dcoords[q_atom - 1], center))
    q_axis = _d_unit(q_raw)
    s_axis = _d_unit(_d_cross(p_axis, q_axis))
    return [p_axis, q_axis, s_axis]




def _periodic_delta(value_plus: float, value_minus: float) -> float:
    delta = float(value_plus - value_minus)
    while delta > np.pi:
        delta -= 2.0 * np.pi
    while delta < -np.pi:
        delta += 2.0 * np.pi
    return delta

def _d_fragment_center(
    dcoords: list[list[_Dual]],
    atoms: tuple[int, ...],
) -> list[_Dual]:
    if not atoms:
        raise FloatingPointError("fragment has no atoms")
    total = [_d_zero_like(dcoords[atoms[0] - 1][0]) for _axis in range(3)]
    for atom in atoms:
        total = _d_vec_add(total, dcoords[atom - 1])
    return _d_vec_scale(total, 1.0 / len(atoms))


def _d_quaternion_vector(rotation: list[list[_Dual]]) -> tuple[_Dual, _Dual, _Dual]:
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace.val <= -1.0 + RANK_TOLERANCE:
        raise FloatingPointError("fragment quaternion is singular near 180 degrees")
    kw = 0.5 * _d_sqrt(trace + 1.0)
    denom = 4.0 * kw
    return (
        (rotation[1][2] - rotation[2][1]) / denom,
        (rotation[2][0] - rotation[0][2]) / denom,
        (rotation[0][1] - rotation[1][0]) / denom,
    )


def _d_rotation_vector(rotation: list[list[_Dual]]) -> tuple[_Dual, _Dual, _Dual]:
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace.val <= -1.0 + RANK_TOLERANCE:
        raise FloatingPointError("fragment exponential map is singular near 180 degrees")
    kw = 0.5 * _d_sqrt(trace + 1.0)
    kx, ky, kz = _d_quaternion_vector(rotation)
    kn2 = kx * kx + ky * ky + kz * kz
    if kn2.val <= RANK_TOLERANCE * RANK_TOLERANCE:
        return 2.0 * kx, 2.0 * ky, 2.0 * kz
    kn = _d_sqrt(kn2)
    factor = (2.0 * _d_atan2(kn, kw)) / kn
    return factor * kx, factor * ky, factor * kz


def _d_zero_like(value: _Dual) -> _Dual:
    return _Dual(0.0, np.zeros_like(value.der))


def _d_vec_add(left: list[_Dual], right: list[_Dual]) -> list[_Dual]:
    return [left[idx] + right[idx] for idx in range(3)]


def _d_vec_sub(left: list[_Dual], right: list[_Dual]) -> list[_Dual]:
    return [left[idx] - right[idx] for idx in range(3)]


def _d_vec_neg(vector: list[_Dual]) -> list[_Dual]:
    return [-item for item in vector]


def _d_vec_scale(vector: list[_Dual], scale: float | _Dual) -> list[_Dual]:
    return [scale * item for item in vector]


def _d_dot(left: list[_Dual], right: list[_Dual]) -> _Dual:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _d_cross(left: list[_Dual], right: list[_Dual]) -> list[_Dual]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _d_norm(vector: list[_Dual]) -> _Dual:
    return _d_sqrt(_d_dot(vector, vector))


def _d_unit(vector: list[_Dual]) -> list[_Dual]:
    norm = _d_norm(vector)
    if norm.val <= RANK_TOLERANCE:
        raise FloatingPointError("zero-length vector")
    return [item / norm for item in vector]




def _d_constant(template: _Dual, value: float) -> _Dual:
    return _Dual(value, np.zeros_like(template.der))


def _d_sqrt(value: _Dual) -> _Dual:
    if value.val <= 0.0:
        raise FloatingPointError("square root of non-positive value")
    root = float(np.sqrt(value.val))
    return _Dual(root, value.der / (2.0 * root))


def _d_acos(value: _Dual) -> _Dual:
    clipped = float(np.clip(value.val, -1.0, 1.0))
    denom = float(np.sqrt(max(1.0 - clipped * clipped, 0.0)))
    if denom <= RANK_TOLERANCE:
        raise FloatingPointError("acos derivative is singular")
    return _Dual(float(np.arccos(clipped)), -value.der / denom)


def _d_asin(value: _Dual) -> _Dual:
    clipped = float(np.clip(value.val, -1.0, 1.0))
    denom = float(np.sqrt(max(1.0 - clipped * clipped, 0.0)))
    if denom <= RANK_TOLERANCE:
        raise FloatingPointError("asin derivative is singular")
    return _Dual(float(np.arcsin(clipped)), value.der / denom)


def _d_atan2(y_value: _Dual, x_value: _Dual) -> _Dual:
    denom = x_value.val * x_value.val + y_value.val * y_value.val
    if denom <= RANK_TOLERANCE:
        raise FloatingPointError("atan2 derivative is singular")
    der = (x_value.val * y_value.der - y_value.val * x_value.der) / denom
    return _Dual(float(np.arctan2(y_value.val, x_value.val)), der)


def _primitive_value(
    primitive: GICPrimitive,
    coords: np.ndarray,
    *,
    reference_coords: np.ndarray | None = None,
) -> float:
    from matrix_chem.face_primitives import FACE_OPERATORS, face_from_smith
    if primitive.function in {"SOOP", "SYCART"}:
        from matrix_chem.soop_contract import frozen_soop_covector
        row, offset = frozen_soop_covector(primitive, len(coords))
        return float(row @ np.asarray(coords).ravel() - offset)
    if primitive.function in FACE_OPERATORS.values():
        from matrix_chem.primitive_coordinates import eval_primitive
        return eval_primitive(face_from_smith(primitive), coords)
    if primitive.function in {"R", "IR", "IR3", "ER"}:
        distance = _distance_value(coords, primitive.atoms)
        if primitive.function == "R":
            return distance
        if distance <= RANK_TOLERANCE:
            raise FloatingPointError(
                f"{primitive.function} distance is singular"
            )
        if primitive.function == "IR":
            return 1.0 / distance
        if primitive.function == "IR3":
            return 1.0 / distance**3
        return float(np.exp(-distance))
    if primitive.function == "A":
        return _angle_value(coords, primitive.atoms)
    if primitive.function in {"L", "D", "U", "H", "IMPD"}:
        return _local_valence_value(primitive, coords)
    if primitive.function == "FC_DIST":
        return _fragment_center_distance_value(coords, primitive.atoms, primitive.ref_atoms)
    if primitive.function == "FCA_DIST":
        return _fragment_center_atom_distance_value(coords, primitive.atoms, primitive.ref_atoms)
    if primitive.function == "CENTER_ATOM_DIST":
        return _fragment_center_atom_distance_value(coords, primitive.atoms, primitive.ref_atoms)
    if primitive.function == "FLIN_TRANS":
        if len(primitive.ref_frame_atoms) != 1:
            raise GICForgeContractError("FLIN_TRANS requires one reference-axis anchor")
        return _axial_jacobi_translation_values(
            coords,
            primitive.atoms,
            primitive.ref_atoms,
            ref_frame_atom=primitive.ref_frame_atoms[0],
        )[primitive.mode]
    if primitive.function == "FTRANS":
        return _fragment_translation_value(
            coords,
            primitive.atoms,
            primitive.ref_atoms,
            mode=primitive.mode,
            ref_frame_atoms=primitive.ref_frame_atoms,
        )
    if primitive.function == "RPCB":
        return _angle_component_value(primitive, coords)
    if primitive.function == "RPCK":
        return _ring_pucker_component_value(primitive, coords)
    if primitive.function == "RPU":
        return _ring_out_of_plane_component_value(primitive, coords)
    if primitive.function == "FAXIS":
        if len(primitive.frame_atoms) != 1 or len(primitive.ref_frame_atoms) != 1:
            raise GICForgeContractError("FAXIS requires one axis anchor per body")
        return _fragment_axis_axis_values(
            coords,
            primitive.atoms,
            primitive.ref_atoms,
            frame_atom=primitive.frame_atoms[0],
            ref_frame_atom=primitive.ref_frame_atoms[0],
            reference_coords=reference_coords,
        )[primitive.mode]
    if primitive.function == "FROT":
        value = _fragment_rotation_value(
            coords,
            primitive.atoms,
            primitive.ref_atoms,
            mode=primitive.mode,
            frame_atoms=primitive.frame_atoms,
            ref_frame_atoms=primitive.ref_frame_atoms,
            reference_coords=reference_coords,
        )
        if reference_coords is None:
            return value
        if len(primitive.frame_atoms) == 1:
            reference_value = _fragment_rotation_value(
                np.asarray(reference_coords, dtype=float),
                primitive.atoms,
                primitive.ref_atoms,
                mode=primitive.mode,
                frame_atoms=primitive.frame_atoms,
                ref_frame_atoms=primitive.ref_frame_atoms,
                reference_coords=np.asarray(reference_coords, dtype=float),
            )
            return float(value - reference_value)
        reference_frag, reference_ref = _fragment_relative_frames(
            np.asarray(reference_coords, dtype=float),
            primitive.atoms,
            primitive.ref_atoms,
            frame_atoms=primitive.frame_atoms,
            ref_frame_atoms=primitive.ref_frame_atoms,
            gauge_reference_coords=np.asarray(reference_coords, dtype=float),
        )
        current_frag, current_ref = _fragment_relative_frames(
            coords,
            primitive.atoms,
            primitive.ref_atoms,
            frame_atoms=primitive.frame_atoms,
            ref_frame_atoms=primitive.ref_frame_atoms,
            gauge_reference_coords=np.asarray(reference_coords, dtype=float),
        )
        delta_rotation = (current_frag.T @ current_ref) @ (reference_frag.T @ reference_ref).T
        return float(_rotation_vector(delta_rotation)[primitive.mode])
    raise GICForgeContractError(f"unsupported primitive function: {primitive.function}")


def _local_valence_value(primitive: GICPrimitive, coords: np.ndarray) -> float:
    """Evaluate the finite scalar paired with `_local_valence_b_row`."""

    from matrix_chem.primitive_coordinates import eval_primitive

    atoms = (
        _improper_dihedral_atoms(primitive.atoms)
        if primitive.function == "IMPD"
        else primitive.atoms
    )
    kind = {
        "L": "linear_bend",
        "D": "dihedral",
        "IMPD": "dihedral",
        "U": "out_of_plane",
        "H": "out_of_plane_height",
    }[primitive.function]
    zero_based = tuple(int(atom) - 1 for atom in atoms)
    zero_based_refs = tuple(int(atom) - 1 for atom in primitive.ref_atoms)
    support = tuple(dict.fromkeys(zero_based + zero_based_refs))
    remap = {atom: index for index, atom in enumerate(support)}
    local = Primitive(
        kind,
        tuple(remap[atom] for atom in zero_based),
        int(primitive.mode),
        tuple(remap[atom] for atom in zero_based_refs),
    )
    with np.errstate(all="raise"):
        return float(
            eval_primitive(
                local,
                np.asarray(coords, dtype=float)[np.asarray(support, dtype=int)],
            )
        )


def _ring_pucker_component_value(primitive: GICPrimitive, coords: np.ndarray) -> float:
    value = 0.0
    for coefficient, atoms in _ring_pucker_terms_from_refs(primitive):
        value += coefficient * _dihedral_value(coords, atoms)
    return float(value)


def _ring_out_of_plane_component_value(
    primitive: GICPrimitive,
    coords: np.ndarray,
) -> float:
    value = 0.0
    for coefficient, atoms in _ring_pucker_terms_from_refs(primitive):
        value += coefficient * _out_of_plane_value(coords, atoms)
    return float(value)


def _angle_component_value(primitive: GICPrimitive, coords: np.ndarray) -> float:
    value = 0.0
    for coefficient, atoms in _angle_component_terms_from_refs(primitive):
        value += coefficient * _angle_value(coords, atoms)
    return float(value)


def _distance_value(coords: np.ndarray, atoms: tuple[int, ...]) -> float:
    i, j = (atom - 1 for atom in atoms)
    return float(np.linalg.norm(coords[i] - coords[j]))


def _angle_value(coords: np.ndarray, atoms: tuple[int, ...]) -> float:
    i, j, k = (atom - 1 for atom in atoms)
    u = coords[i] - coords[j]
    v = coords[k] - coords[j]
    return float(np.arccos(np.clip(_dot_unit(u, v), -1.0, 1.0)))


def _linear_bend_value(coords: np.ndarray, atoms: tuple[int, ...], *, mode: int) -> float:
    i, j, k = (atom - 1 for atom in atoms)
    left = _unit(coords[i] - coords[j])
    right = _unit(coords[k] - coords[j])
    trial = np.array([1.0, 0.0, 0.0])
    if abs(float(left[0])) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(left, trial))
    e2 = _unit(np.cross(left, e1))
    bend = left + right
    return float(np.dot(bend, e1 if mode == -1 else e2))


def _four_atom_linear_bend_value(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    reference: int,
    *,
    mode: int,
) -> float:
    endpoint1, center, endpoint2 = atoms
    if mode == -1:
        return _angle_value(
            coords, (endpoint1, center, reference)
        ) + _angle_value(coords, (reference, center, endpoint2))
    if mode == -2:
        return referenced_linear_bend_perpendicular_value(
            _dihedral_value(coords, (endpoint1, center, reference, endpoint2)),
            _dihedral_value(coords, (endpoint1, reference, center, endpoint2)),
        )
    raise GICForgeContractError(f"invalid linear-bend mode: {mode}")


def _dihedral_value(coords: np.ndarray, atoms: tuple[int, ...]) -> float:
    i, j, k, ell = (atom - 1 for atom in atoms)
    p0, p1, p2, p3 = coords[i], coords[j], coords[k], coords[ell]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = _unit(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.arctan2(-y, -x))


def _out_of_plane_value(coords: np.ndarray, atoms: tuple[int, ...]) -> float:
    center, plane1, plane2, out = (atom - 1 for atom in atoms)
    r1 = coords[plane1] - coords[center]
    r2 = coords[plane2] - coords[center]
    rout = coords[out] - coords[center]
    normal = _unit(np.cross(r1, r2))
    return float(-np.arcsin(np.clip(np.dot(_unit(rout), normal), -1.0, 1.0)))


def _out_of_plane_height_value(coords: np.ndarray, atoms: tuple[int, ...]) -> float:
    center, plane1, plane2, out = (atom - 1 for atom in atoms)
    r1 = coords[plane1] - coords[center]
    r2 = coords[plane2] - coords[center]
    rout = coords[out] - coords[center]
    return float(np.dot(rout, _unit(np.cross(r1, r2))))


def _fragment_center_distance_value(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
) -> float:
    delta = _fragment_center(coords, atoms) - _fragment_center(coords, ref_atoms)
    return float(np.linalg.norm(delta))


def _fragment_center_atom_distance_value(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
) -> float:
    if len(ref_atoms) != 1:
        raise FloatingPointError("center-atom distance needs exactly one reference atom")
    return float(np.linalg.norm(_fragment_center(coords, atoms) - coords[ref_atoms[0] - 1]))


def _fragment_translation_value(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
    *,
    mode: int,
    ref_frame_atoms: tuple[int, ...] = (),
) -> float:
    delta = _fragment_center(coords, atoms) - _fragment_center(coords, ref_atoms)
    if not ref_frame_atoms:
        return float(delta[mode])
    reference_frame = _fragment_frame(
        coords,
        ref_atoms,
        frame_atoms=ref_frame_atoms,
    )
    return float((delta @ reference_frame)[mode])


def _axial_jacobi_frame(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
    *,
    ref_frame_atom: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    moving_center = _fragment_center(coords, atoms)
    reference_center = _fragment_center(coords, ref_atoms)
    delta = moving_center - reference_center
    primary = _unit(coords[ref_frame_atom - 1] - reference_center)
    transverse = delta - float(delta @ primary) * primary
    secondary = _unit(transverse)
    normal = _unit(np.cross(primary, secondary))
    return delta, primary, secondary, normal


def _axial_jacobi_translation_values(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
    *,
    ref_frame_atom: int,
) -> tuple[float, float]:
    delta, primary, secondary, _normal = _axial_jacobi_frame(
        coords,
        atoms,
        ref_atoms,
        ref_frame_atom=ref_frame_atom,
    )
    return float(delta @ primary), float(delta @ secondary)


def _fragment_axis_axis_values(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
    *,
    frame_atom: int,
    ref_frame_atom: int,
    reference_coords: np.ndarray | None,
) -> tuple[float, float]:
    selection = coords if reference_coords is None else np.asarray(reference_coords, dtype=float)
    pole_index, pole_sign, transverse = _axis_axis_stereographic_chart(
        selection,
        atoms,
        ref_atoms,
        frame_atom=frame_atom,
        ref_frame_atom=ref_frame_atom,
    )
    _delta, primary, secondary, normal = _axial_jacobi_frame(
        coords,
        atoms,
        ref_atoms,
        ref_frame_atom=ref_frame_atom,
    )
    moving_center = _fragment_center(coords, atoms)
    direction = _unit(coords[frame_atom - 1] - moving_center)
    components = tuple(float(direction @ axis) for axis in (primary, secondary, normal))
    denominator = 1.0 + pole_sign * components[pole_index]
    if denominator <= 1.0e-12:
        raise FloatingPointError("axis--axis stereographic chart reached its antipode")
    return (
        components[transverse[0]] / denominator,
        components[transverse[1]] / denominator,
    )


def _fragment_rotation_value(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
    *,
    mode: int,
    frame_atoms: tuple[int, ...] = (),
    ref_frame_atoms: tuple[int, ...] = (),
    reference_coords: np.ndarray | None = None,
) -> float:
    if len(frame_atoms) == 1:
        selection_coords = coords if reference_coords is None else np.asarray(
            reference_coords,
            dtype=float,
        )
        return float(
            _linear_fragment_stereographic_values(
                coords,
                atoms,
                frame_atoms[0],
                ref_atoms,
                ref_frame_atoms,
                selection_coords=selection_coords,
            )[mode]
        )
    frame_frag, frame_ref = _fragment_relative_frames(
        coords,
        atoms,
        ref_atoms,
        frame_atoms=frame_atoms,
        ref_frame_atoms=ref_frame_atoms,
        gauge_reference_coords=reference_coords,
    )
    rotation = frame_frag.T @ frame_ref
    return float(_rotation_vector(rotation)[mode])


def _fragment_center(coords: np.ndarray, atoms: tuple[int, ...]) -> np.ndarray:
    if not atoms:
        raise FloatingPointError("fragment has no atoms")
    return np.mean(coords[[atom - 1 for atom in atoms]], axis=0)


def _fragment_frame_rank(coords: np.ndarray, atoms: tuple[int, ...]) -> int:
    if len(atoms) < 2:
        return 0
    subset = coords[[atom - 1 for atom in atoms]]
    if is_linear_geometry(subset, tolerance=1.0e-6):
        return 1
    centered = subset - _fragment_center(coords, atoms)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    return int(np.sum(singular_values > 1.0e-8))


def _fragment_linear_anchor_atom(atoms: tuple[int, ...], *, coords: np.ndarray) -> int:
    """Choose a deterministic signed-axis anchor for a linear fragment."""

    if _fragment_frame_rank(coords, atoms) != 1:
        raise FloatingPointError("linear fragment axis requires a rank-one body")
    center = _fragment_center(coords, atoms)
    return min(
        atoms,
        key=lambda atom: (-float(np.linalg.norm(coords[atom - 1] - center)), atom),
    )


def _linear_fragment_gauge_axis_index(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    anchor_atom: int,
    ref_atoms: tuple[int, ...],
    ref_frame_atoms: tuple[int, ...],
) -> int:
    """Select the reference-frame axis furthest from the frozen linear axis."""

    center = _fragment_center(coords, atoms)
    axis = _unit(coords[anchor_atom - 1] - center)
    reference_frame = _fragment_frame(coords, ref_atoms, frame_atoms=ref_frame_atoms)
    return min(range(3), key=lambda index: (abs(float(axis @ reference_frame[:, index])), index))


def _linear_fragment_stereographic_axes(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    anchor_atom: int,
    ref_atoms: tuple[int, ...],
    ref_frame_atoms: tuple[int, ...],
) -> tuple[int, float, tuple[int, int]]:
    """Freeze the best-conditioned stereographic pole for a linear body.

    A linear rigid body has an orientation on the unit sphere, not on SO(3).
    Two fixed components of an SO(3) exponential map can therefore become
    dependent as the molecular axis rotates.  Choose the signed reference-
    frame axis closest to the initial molecular direction as the projection
    pole; the two remaining axes are the exact, nonredundant coordinates.
    """

    center = _fragment_center(coords, atoms)
    direction = _unit(coords[anchor_atom - 1] - center)
    reference_frame = _fragment_frame(coords, ref_atoms, frame_atoms=ref_frame_atoms)
    components = direction @ reference_frame
    pole_index = max(range(3), key=lambda index: (abs(float(components[index])), -index))
    pole_sign = 1.0 if float(components[pole_index]) >= 0.0 else -1.0
    transverse = tuple(index for index in range(3) if index != pole_index)
    return pole_index, pole_sign, (transverse[0], transverse[1])


def _linear_fragment_stereographic_values(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    anchor_atom: int,
    ref_atoms: tuple[int, ...],
    ref_frame_atoms: tuple[int, ...],
    *,
    selection_coords: np.ndarray,
) -> tuple[float, float, float]:
    """Return two exact linear-body orientation coordinates and a zero pad."""

    pole_index, pole_sign, transverse = _linear_fragment_stereographic_axes(
        np.asarray(selection_coords, dtype=float),
        atoms,
        anchor_atom,
        ref_atoms,
        ref_frame_atoms,
    )
    center = _fragment_center(coords, atoms)
    direction = _unit(coords[anchor_atom - 1] - center)
    reference_frame = _fragment_frame(coords, ref_atoms, frame_atoms=ref_frame_atoms)
    components = direction @ reference_frame
    denominator = 1.0 + pole_sign * float(components[pole_index])
    if denominator <= 1.0e-12:
        raise FloatingPointError("linear-fragment stereographic chart reached its antipode")
    return (
        float(components[transverse[0]]) / denominator,
        float(components[transverse[1]]) / denominator,
        0.0,
    )


def _fragment_relative_frames(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    ref_atoms: tuple[int, ...],
    *,
    frame_atoms: tuple[int, ...] = (),
    ref_frame_atoms: tuple[int, ...] = (),
    gauge_reference_coords: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return moving/reference frames for nonlinear or linear rigid bodies."""

    reference_frame = _fragment_frame(coords, ref_atoms, frame_atoms=ref_frame_atoms)
    if len(frame_atoms) != 1:
        return _fragment_frame(coords, atoms, frame_atoms=frame_atoms), reference_frame
    gauge_coords = coords if gauge_reference_coords is None else np.asarray(
        gauge_reference_coords,
        dtype=float,
    )
    gauge_index = _linear_fragment_gauge_axis_index(
        gauge_coords,
        atoms,
        frame_atoms[0],
        ref_atoms,
        ref_frame_atoms,
    )
    center = _fragment_center(coords, atoms)
    p_axis = _unit(coords[frame_atoms[0] - 1] - center)
    q_axis = _unit(np.cross(p_axis, reference_frame[:, gauge_index]))
    s_axis = _unit(np.cross(p_axis, q_axis))
    return np.column_stack((p_axis, q_axis, s_axis)), reference_frame


def _fragment_frame(
    coords: np.ndarray,
    atoms: tuple[int, ...],
    *,
    frame_atoms: tuple[int, ...] = (),
) -> np.ndarray:
    if _fragment_frame_rank(coords, atoms) < 2:
        raise FloatingPointError("fragment orientation is underdefined")
    p_atom, q_atom = (
        tuple(frame_atoms) if frame_atoms else _fragment_frame_anchor_atoms(atoms, coords=coords)
    )
    center = _fragment_center(coords, atoms)
    p_axis = _unit(coords[p_atom - 1] - center)
    q_raw = np.cross(p_axis, coords[q_atom - 1] - center)
    q_axis = _unit(q_raw)
    s_axis = _unit(np.cross(p_axis, q_axis))
    return np.column_stack([p_axis, q_axis, s_axis])


def _fragment_frame_anchor_atoms(
    atoms: tuple[int, ...],
    *,
    coords: np.ndarray,
) -> tuple[int, int]:
    center = _fragment_center(coords, atoms)
    ranked = sorted(
        atoms,
        key=lambda atom: (-float(np.linalg.norm(coords[atom - 1] - center)), atom),
    )
    p_atom = ranked[0]
    p_axis = _unit(coords[p_atom - 1] - center)
    q_candidates = []
    for atom in atoms:
        if atom == p_atom:
            continue
        vector = coords[atom - 1] - center
        norm = float(np.linalg.norm(vector))
        if norm <= RANK_TOLERANCE:
            continue
        dot = abs(float(np.dot(p_axis, vector / norm)))
        q_candidates.append((dot, -norm, atom))
    if not q_candidates:
        raise FloatingPointError("fragment has no second orientation anchor")
    _dot, _norm, q_atom = min(q_candidates)
    return p_atom, q_atom


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / scale
        qx = (rotation[1, 2] - rotation[2, 1]) * scale
        qy = (rotation[2, 0] - rotation[0, 2]) * scale
        qz = (rotation[0, 1] - rotation[1, 0]) * scale
    else:
        qw, qx, qy, qz = _fallback_quaternion(rotation)
    quat = np.array([qw, qx, qy, qz], dtype=float)
    if quat[0] < 0.0:
        quat = -quat
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12:
        return np.zeros(3, dtype=float)
    quat /= norm
    vector = quat[1:]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm < 1.0e-12:
        return np.zeros(3, dtype=float)
    angle = 2.0 * np.arctan2(vector_norm, quat[0])
    return vector / vector_norm * angle


def _fallback_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]))
        return (
            (rotation[2, 1] - rotation[1, 2]) / scale,
            0.25 * scale,
            (rotation[0, 1] + rotation[1, 0]) / scale,
            (rotation[0, 2] + rotation[2, 0]) / scale,
        )
    if rotation[1, 1] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]))
        return (
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[0, 1] + rotation[1, 0]) / scale,
            0.25 * scale,
            (rotation[1, 2] + rotation[2, 1]) / scale,
        )
    scale = 2.0 * np.sqrt(max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]))
    return (
        (rotation[1, 0] - rotation[0, 1]) / scale,
        (rotation[0, 2] + rotation[2, 0]) / scale,
        (rotation[1, 2] + rotation[2, 1]) / scale,
        0.25 * scale,
    )


def _improper_dihedral_atoms(atoms: tuple[int, ...]) -> tuple[int, ...]:
    center, n1, n2, n3 = atoms
    return (n1, center, n3, n2)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= RANK_TOLERANCE:
        raise FloatingPointError("zero-length vector")
    return vector / norm


def _dot_unit(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(_unit(left), _unit(right)))
