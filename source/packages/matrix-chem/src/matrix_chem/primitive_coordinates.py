"""ORACLE-owned redundant primitive coordinates and Wilson B matrix.

The primitive layer is deliberately redundant and geometry local.  SMITH is a
consumer of this layer: it selects, combines and symmetry adapts these rows to
construct SONIC coordinates, but it does not own molecular reperception.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from matrix_core import read_sectioned_lines, replace_section, section_content

from .coordinate_library import CoordinateComponent, coordinate_selection_units
from .structural_corrections import pseudo_bond_pairs
from .face_primitives import FACE_OPERATORS, compiled_face_value_gradient, face_value_gradient


try:
    from . import _chem_native
except ImportError:
    _chem_native = None


MATRIX_XYZ_PRIMITIVES_SCHEMA = "matrix.xyz.primitives.v2"
LEGACY_MATRIX_XYZ_PRIMITIVES_SCHEMA = "matrix.xyz.primitives.v1"
LEGACY_ORACLE_XYZ_PRIMITIVES_SCHEMA = "oracle.xyz.primitives.v1"
LINEAR_ANGLE_DEGREES = 165.0
EXPONENTIAL_BOND_ALPHA = 1.0
# v5 records the complete SMITH radial atlas and its corrected IR3 Jacobian.
# Cached plans produced by the pre-correction kernel must not be reused.
COMPILED_CHEM_ABI_VERSION = "matrix-chem-native-abi.v5"


def compiled_chem_kernel_identity() -> dict[str, object]:
    """Stable cache identity for the loaded native chemistry kernel."""

    if _chem_native is None:
        return {"abi": COMPILED_CHEM_ABI_VERSION, "implementation": "unavailable"}
    return {"abi": COMPILED_CHEM_ABI_VERSION, **dict(_chem_native.build_info())}


@dataclass(frozen=True)
class Primitive:
    """One ORACLE primitive; atom indices are zero based in memory."""

    kind: str
    atoms: tuple[int, ...]
    mode: int = 0
    ref: tuple[int, ...] = ()
    frame_atoms: tuple[int, ...] = ()
    ref_frame_atoms: tuple[int, ...] = ()
    rotation_reference: tuple[tuple[float, float, float], ...] = ()
    refs: tuple[str, ...] = ()
    chart: str = "PRINCIPAL"
    smith_function: str = ""
    smith_family: str = ""
    smith_identifier: str = ""
    chart_reference_radian: float | None = None

    @property
    def function(self) -> str:
        if self.smith_function:
            return self.smith_function
        if self.kind in FACE_OPERATORS:
            return FACE_OPERATORS[self.kind]
        return {
            "bond": "R",
            "inverse_bond": "IR",
            "inverse_cubic_bond": "IR3",
            "exponential_bond": "ER",
            "hbond_dist": "R",
            "pseudo_bond": "R",
            "angle": "A",
            "linear_bend": "L",
            "dihedral": "D",
            "out_of_plane": "U",
            "out_of_plane_height": "H",
            "selective_out_of_plane": "SOOP",
            "frag_dist": "R",
            "frag_atom_dist": "R",
            "frag_trans": "FTRANS",
            "frag_rot": "FROT",
            "frag_linear_trans": "FLIN_TRANS",
            "frag_axis": "FAXIS",
        }.get(self.kind, self.kind.upper())

    @property
    def label(self) -> str:
        atoms = ",".join(str(atom + 1) for atom in self.atoms)
        if self.kind == "linear_bend":
            reference = self.ref[0] + 1 if len(self.ref) == 1 else 0
            return f"L({atoms},{reference},{self.mode})"
        return f"{self.function}({atoms})"


@dataclass(frozen=True)
class PrimitiveCoordinateContract:
    schema: str
    primitives: tuple[Primitive, ...]
    reference_values: tuple[float, ...]
    reference_geometry_sha256: str
    b_matrix_sha256: str
    b_matrix_rank: int
    b_matrix_columns: int
    b_matrix_storage: str = "DENSE"
    b_matrix_rank_method: str = "NUMERICAL_SVD"


def build_primitives(
    discrete_graph,
    coords: np.ndarray,
    linear_threshold: float = np.deg2rad(LINEAR_ANGLE_DEGREES),
    *,
    include_pseudo_bonds: bool = True,
    pair_selector_threshold: float = 1.0e-3,
) -> list[Primitive]:
    """Build the redundant local primitive set from frozen ORACLE topology."""
    xyz = np.asarray(coords, dtype=float)
    n = int(discrete_graph.natoms)
    bonds = [Primitive("bond", tuple(sorted((i, j)))) for i, j in discrete_graph.bonds]
    pseudo: list[Primitive] = []
    if include_pseudo_bonds:
        for pair, source in pseudo_bond_pairs(
            tuple(int(value) for value in discrete_graph.Z),
            xyz,
            discrete_graph.bonds,
            selector_threshold=pair_selector_threshold,
        ):
            kind = "hbond_dist" if "HBOND" in source else "pseudo_bond"
            pseudo.append(Primitive(kind, tuple(sorted(pair)), ref=()))

    angles: list[Primitive] = []
    linears: list[Primitive] = []
    linear_keys: set[tuple[int, int, int]] = set()
    for center in range(n):
        neighbors = sorted(discrete_graph.adjacency[center])
        for left, right in combinations(neighbors, 2):
            atoms = (left, center, right)
            if angle(*atoms, xyz) >= linear_threshold:
                linear_keys.add(_angle_key(atoms))
                reference_candidates = None
                if n >= 256:
                    neighborhood = set(atoms)
                    frontier = set(atoms)
                    for _shell in range(2):
                        frontier = {
                            int(neighbor)
                            for atom in frontier
                            for neighbor in discrete_graph.adjacency[atom]
                        } - neighborhood
                        neighborhood.update(frontier)
                    reference_candidates = tuple(sorted(neighborhood - set(atoms)))
                reference = linear_bend_reference_atom(
                    atoms,
                    xyz,
                    candidate_indices=reference_candidates,
                )
                refs = (reference,) if reference is not None else ()
                linears.extend(
                    (
                        Primitive("linear_bend", atoms, mode=-1, ref=refs),
                        Primitive("linear_bend", atoms, mode=-2, ref=refs),
                    )
                )
            else:
                angles.append(Primitive("angle", atoms))

    dihedrals: list[Primitive] = []
    seen: set[tuple[int, int, int, int]] = set()
    for j, k in sorted(tuple(sorted(pair)) for pair in discrete_graph.bonds):
        for i in sorted(set(discrete_graph.adjacency[j]) - {k}):
            for ell in sorted(set(discrete_graph.adjacency[k]) - {j}):
                atoms = (i, j, k, ell)
                if len(set(atoms)) != 4:
                    continue
                canonical = min(atoms, tuple(reversed(atoms)))
                if canonical in seen:
                    continue
                seen.add(canonical)
                if _angle_key((i, j, k)) in linear_keys or _angle_key((j, k, ell)) in linear_keys:
                    continue
                dihedrals.append(Primitive("dihedral", canonical))

    out_of_planes: list[Primitive] = []
    for center in range(n):
        neighbors = sorted(discrete_graph.adjacency[center])
        for n1, n2, n3 in combinations(neighbors, 3):
            out_of_planes.append(Primitive("out_of_plane", (center, n1, n2, n3)))
    return bonds + pseudo + angles + linears + dihedrals + out_of_planes


def eval_primitive(primitive: Primitive, coords: np.ndarray) -> float:
    xyz = np.asarray(coords, dtype=float)
    if primitive.kind == "selective_out_of_plane":
        from .soop_contract import frozen_soop_covector

        row, offset = frozen_soop_covector(primitive, xyz.shape[0])
        return float(row @ xyz.reshape(-1) - offset)
    if primitive.kind in FACE_OPERATORS:
        return compiled_face_value_gradient(primitive, xyz)[0]
    if primitive.kind in {"frag_rot", "frag_trans", "frag_linear_trans", "frag_axis"}:
        return float(_compiled_fragment_rotation(primitive, xyz)[0][0])
    if primitive.kind in {"bond", "hbond_dist", "pseudo_bond", "inverse_bond", "inverse_cubic_bond", "exponential_bond"}:
        i, j = primitive.atoms
        distance = float(np.linalg.norm(xyz[i] - xyz[j]))
        if primitive.kind == "inverse_bond":
            return float(1.0 / distance)
        if primitive.kind == "inverse_cubic_bond":
            return float(1.0 / distance**3)
        if primitive.kind == "exponential_bond":
            return float(np.exp(-EXPONENTIAL_BOND_ALPHA * distance))
        return distance
    if primitive.kind == "angle":
        return float(angle(*primitive.atoms, xyz))
    if primitive.kind == "dihedral":
        return float(dihedral(*primitive.atoms, xyz))
    if primitive.kind == "out_of_plane":
        return float(out_of_plane(*primitive.atoms, xyz))
    if primitive.kind == "out_of_plane_height":
        return float(out_of_plane_height(*primitive.atoms, xyz))
    if primitive.kind == "linear_bend":
        if len(primitive.ref) == 1:
            return _four_atom_linear_bend_value(
                *primitive.atoms,
                primitive.ref[0],
                xyz,
                mode=primitive.mode,
            )
        if len(primitive.atoms) == 4:
            return _four_atom_linear_bend_value(
                *primitive.atoms[:3], primitive.atoms[3], xyz, mode=primitive.mode
            )
        first, second = linear_components(*primitive.atoms, xyz)
        return float(first if primitive.mode == -1 else second)
    if primitive.kind in {"frag_dist", "frag_atom_dist", "frag_trans", "frag_rot"}:
        return _eval_fragment_primitive(primitive, xyz)
    raise ValueError(f"unknown primitive kind: {primitive.kind}")


def eval_primitives(primitives: Iterable[Primitive], coords: np.ndarray) -> np.ndarray:
    primitive_tuple = tuple(primitives)
    compiled = compiled_local_primitive_values_and_sparse_gradients(
        primitive_tuple, coords
    )
    if compiled is not None:
        return compiled[0]
    return np.asarray([eval_primitive(primitive, coords) for primitive in primitive_tuple], dtype=float)


_COMPILED_LOCAL_PRIMITIVE_CODES = {
    "bond": 1,
    "hbond_dist": 1,
    "pseudo_bond": 1,
    "angle": 2,
    "dihedral": 3,
    "out_of_plane": 4,
    "out_of_plane_height": 5,
    "linear_bend": 6,
    "inverse_bond": 7,
    "inverse_cubic_bond": 8,
    "exponential_bond": 9,
}


@dataclass(frozen=True)
class CompiledLocalPrimitivePlan:
    """Immutable native dispatch arrays for one frozen primitive atlas."""

    kinds: np.ndarray
    atoms: np.ndarray
    counts: np.ndarray
    modes: np.ndarray
    extended_metadata: tuple[np.ndarray, ...] | None = None


def compile_local_primitive_plan(
    primitives: Sequence[Primitive],
) -> CompiledLocalPrimitivePlan | None:
    """Compile geometry-independent primitive metadata exactly once."""

    primitive_tuple = tuple(primitives)
    from .face_primitives import FACE_OPERATORS

    if any(p.kind in FACE_OPERATORS for p in primitive_tuple):
        codes = dict(_COMPILED_LOCAL_PRIMITIVE_CODES)
        codes.update({kind: 18 + i for i, kind in enumerate(FACE_OPERATORS)})
        if any(p.kind not in codes for p in primitive_tuple):
            return None
        kinds = np.asarray([codes[p.kind] for p in primitive_tuple], dtype=np.intp)
        modes = np.asarray([p.mode for p in primitive_tuple], dtype=np.intp)
        metadata = []
        for field in ("atoms", "ref", "frame_atoms", "ref_frame_atoms"):
            data, pointers = [], [0]
            for primitive in primitive_tuple:
                data.extend(getattr(primitive, field))
                pointers.append(len(data))
            metadata.extend((np.asarray(data, dtype=np.intp), np.asarray(pointers, dtype=np.intp)))
        for array in (kinds, modes, *metadata):
            array.setflags(write=False)
        return CompiledLocalPrimitivePlan(kinds, np.empty((0, 4), dtype=np.intp),
                                         np.empty(0, dtype=np.intp), modes, tuple(metadata))
    kinds = np.empty(len(primitive_tuple), dtype=np.intp)
    atoms = np.full((len(primitive_tuple), 4), -1, dtype=np.intp)
    counts = np.empty(len(primitive_tuple), dtype=np.intp)
    modes = np.empty(len(primitive_tuple), dtype=np.intp)
    for index, primitive in enumerate(primitive_tuple):
        code = _COMPILED_LOCAL_PRIMITIVE_CODES.get(primitive.kind)
        if code is None:
            return None
        support = (
            tuple(primitive.atoms) + tuple(primitive.ref)
            if primitive.kind == "linear_bend"
            else tuple(primitive.atoms)
        )
        if not 2 <= len(support) <= 4 or len(set(support)) != len(support):
            return None
        kinds[index] = code
        atoms[index, : len(support)] = support
        counts[index] = len(support)
        modes[index] = primitive.mode
    for array in (kinds, atoms, counts, modes):
        array.setflags(write=False)
    return CompiledLocalPrimitivePlan(kinds, atoms, counts, modes)


def evaluate_compiled_local_primitive_plan(
    plan: CompiledLocalPrimitivePlan,
    coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a compiled plan directly as primitive values plus CSR rows."""

    if plan.extended_metadata is not None:
        integer = lambda values: np.asarray(values, dtype=np.intp)
        result = compiled_smith_primitive_plan_csr(
            coords, coords, plan.kinds, plan.modes, *plan.extended_metadata,
            integer([]), integer([]).reshape(0, 4), np.empty(0),
            np.zeros(len(plan.kinds) + 1, dtype=np.intp),
        )
        if result is None:
            raise RuntimeError("the compiled ORACLE primitive kernel is unavailable")
        return result

    if _chem_native is None or not hasattr(
        _chem_native, "local_primitive_values_sparse_gradients"
    ):
        # Older architecture wheels may not carry the optional local-plan
        # entry point.  Reconstruct the immutable primitive rows and use the
        # same analytic evaluators as the native path; this preserves values
        # and Jacobians exactly while keeping the wheel usable.
        inverse_codes = {value: key for key, value in _COMPILED_LOCAL_PRIMITIVE_CODES.items()}
        values: list[float] = []
        data: list[float] = []
        indices: list[int] = []
        indptr = [0]
        for kind_code, row, count, mode in zip(
            plan.kinds, plan.atoms, plan.counts, plan.modes
        ):
            kind = inverse_codes.get(int(kind_code))
            if kind is None:
                raise RuntimeError("the compiled ORACLE primitive kernel is unavailable")
            primitive = Primitive(kind, tuple(int(atom) for atom in row[: int(count)]), mode=int(mode))
            values.append(eval_primitive(primitive, coords))
            gradient = np.asarray(grad_primitive(primitive, coords), dtype=float).reshape(-1)
            nz = np.flatnonzero(gradient)
            indices.extend(int(index) for index in nz)
            data.extend(float(gradient[index]) for index in nz)
            indptr.append(len(data))
        return (
            np.asarray(values, dtype=float),
            np.asarray(data, dtype=float),
            np.asarray(indices, dtype=np.intp),
            np.asarray(indptr, dtype=np.intp),
        )
    values, data, indices, indptr = _chem_native.local_primitive_values_sparse_gradients(
        np.asarray(coords, dtype=float),
        plan.kinds,
        plan.atoms,
        plan.counts,
        plan.modes,
    )
    return (
        np.asarray(values, dtype=float),
        np.asarray(data, dtype=float),
        np.asarray(indices, dtype=np.intp),
        np.asarray(indptr, dtype=np.intp),
    )


def assemble_compiled_gic_csr(
    primitive_data: np.ndarray,
    primitive_indices: np.ndarray,
    primitive_indptr: np.ndarray,
    combination,
    *,
    column_count: int,
    zero_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble sparse GIC rows wholly inside the portable C++ kernel."""

    if _chem_native is None or not hasattr(_chem_native, "assemble_gic_csr"):
        raise RuntimeError("the compiled GIC CSR assembly kernel is unavailable")
    data, indices, indptr = _chem_native.assemble_gic_csr(
        np.asarray(primitive_data, dtype=float),
        np.asarray(primitive_indices, dtype=np.intp),
        np.asarray(primitive_indptr, dtype=np.intp),
        np.asarray(combination.data, dtype=float),
        np.asarray(combination.indices, dtype=np.intp),
        np.asarray(combination.indptr, dtype=np.intp),
        int(column_count),
        float(zero_tolerance),
    )
    return (
        np.asarray(data, dtype=float),
        np.asarray(indices, dtype=np.intp),
        np.asarray(indptr, dtype=np.intp),
    )


def compiled_csr_matvec(data, indices, indptr, vector, *, column_count: int, transpose: bool = False):
    """Apply native CSR ``B`` or ``B.T`` without constructing SciPy objects."""

    if _chem_native is None:
        raise RuntimeError("the compiled CSR product kernel is unavailable")
    function = _chem_native.csr_transpose_matvec if transpose else _chem_native.csr_matvec
    return np.asarray(
        function(
            np.asarray(data, dtype=float), np.asarray(indices, dtype=np.intp),
            np.asarray(indptr, dtype=np.intp), np.asarray(vector, dtype=float),
            int(column_count),
        ),
        dtype=float,
    )


def compiled_csr_incremental_matching(indices, indptr, matching, *, column_count: int, start_row: int):
    """Extend a support matching in the native kernel; return ``None`` on deficiency."""

    if _chem_native is None or not hasattr(_chem_native, "csr_incremental_matching"):
        raise RuntimeError("the compiled incremental matching kernel is unavailable")
    result = _chem_native.csr_incremental_matching(
        np.asarray(indices, dtype=np.intp), np.asarray(indptr, dtype=np.intp),
        np.asarray(matching, dtype=np.intp), int(column_count), int(start_row),
    )
    return None if result is None else np.asarray(result, dtype=np.intp)


def compiled_csr_support_certificate(indices, indptr, *, column_count: int):
    """Return the fused native structural certificate for one CSR support."""

    if _chem_native is None or not hasattr(_chem_native, "csr_support_certificate"):
        raise RuntimeError("the compiled CSR support-certificate kernel is unavailable")
    payload = _chem_native.csr_support_certificate(
        np.asarray(indices, dtype=np.intp),
        np.asarray(indptr, dtype=np.intp),
        int(column_count),
    )
    return {
        "rank": int(payload["rank"]),
        "connected_components": int(payload["connected_components"]),
        "unsupported_rows": tuple(int(value) for value in payload["unsupported_rows"]),
        "column_to_row": np.asarray(payload["column_to_row"], dtype=np.intp),
    }


def compiled_local_primitive_values_and_sparse_gradients(
    primitives: Sequence[Primitive],
    coords: np.ndarray,
) -> tuple[
    np.ndarray,
    tuple[tuple[tuple[int, float], ...], ...],
] | None:
    """Evaluate a complete local primitive batch in the C++ ORACLE kernel.

    ``None`` requests the established Python fallback when the extension is
    unavailable or the batch contains a nonlocal fragment coordinate.  The
    native return stays row-sparse: no ``nprimitive x 3N`` array is formed.
    """

    raw = compiled_local_primitive_values_and_csr(primitives, coords)
    if raw is None:
        return None
    values, data, indices, indptr = raw
    primitive_tuple = tuple(primitives)
    rows = tuple(
        tuple(
            (int(column), float(value))
            for column, value in zip(
                indices[indptr[row] : indptr[row + 1]],
                data[indptr[row] : indptr[row + 1]],
                strict=True,
            )
        )
        for row in range(len(primitive_tuple))
    )
    return values, rows


def compiled_local_primitive_values_and_csr(
    primitives: Sequence[Primitive],
    coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return the native primitive batch directly as CSR arrays."""

    if _chem_native is None or not hasattr(
        _chem_native, "local_primitive_values_sparse_gradients"
    ):
        return None
    plan = compile_local_primitive_plan(primitives)
    if plan is None:
        return None
    return evaluate_compiled_local_primitive_plan(plan, coords)


def compiled_smith_primitive_plan_csr(*arrays):
    """Execute a fully serialized SMITH primitive plan in the C++ kernel."""

    if _chem_native is None or not hasattr(
        _chem_native, "smith_primitive_values_sparse_gradients"
    ):
        return None
    return tuple(_chem_native.smith_primitive_values_sparse_gradients(*arrays))


def grad_primitive(primitive: Primitive, coords: np.ndarray, fd_step: float = 1.0e-4) -> np.ndarray:
    xyz = np.asarray(coords, dtype=float)
    if primitive.kind == "selective_out_of_plane":
        from .soop_contract import frozen_soop_covector

        row, _offset = frozen_soop_covector(primitive, xyz.shape[0])
        return row.reshape(xyz.shape)
    if primitive.kind in FACE_OPERATORS:
        result = np.zeros(xyz.size)
        for column, value in compiled_face_value_gradient(primitive, xyz)[1]:
            result[column] = value
        return result.reshape(xyz.shape)
    if primitive.kind in {"bond", "hbond_dist", "pseudo_bond", "inverse_bond", "inverse_cubic_bond", "exponential_bond"}:
        gradient = _bond_grad(*primitive.atoms, xyz)
        distance = float(np.linalg.norm(xyz[primitive.atoms[0]] - xyz[primitive.atoms[1]]))
        if distance <= 1.0e-12:
            raise FloatingPointError("zero-length distance is singular")
        if primitive.kind in {"inverse_bond", "inverse_cubic_bond"}:
            power = 2 if primitive.kind == "inverse_bond" else 4
            exponent = 1 if primitive.kind == "inverse_bond" else 3
            gradient *= -exponent / distance**power
        elif primitive.kind == "exponential_bond":
            gradient *= -EXPONENTIAL_BOND_ALPHA * np.exp(-EXPONENTIAL_BOND_ALPHA * distance)
        return gradient
    if primitive.kind == "angle":
        return _angle_grad(*primitive.atoms, xyz)
    if primitive.kind == "dihedral":
        return _dihedral_grad(*primitive.atoms, xyz)
    if primitive.kind == "out_of_plane":
        return _out_of_plane_grad(*primitive.atoms, xyz)
    if primitive.kind == "out_of_plane_height":
        return _out_of_plane_height_grad(*primitive.atoms, xyz)
    if primitive.kind == "linear_bend":
        if len(primitive.ref) == 1:
            return _four_atom_linear_bend_grad(
                *primitive.atoms,
                primitive.ref[0],
                xyz,
                mode=primitive.mode,
            )
        if len(primitive.atoms) == 4:
            return _four_atom_linear_bend_grad(
                *primitive.atoms[:3], primitive.atoms[3], xyz, mode=primitive.mode
            )
        return _linear_grad(*primitive.atoms, xyz, mode=primitive.mode)
    if primitive.kind in {"frag_dist", "frag_atom_dist"}:
        return _fragment_distance_grad(primitive, xyz)
    if primitive.kind in {"frag_rot", "frag_trans", "frag_linear_trans", "frag_axis"}:
        values, data, indices, indptr = _compiled_fragment_rotation(primitive, xyz)
        gradient = np.zeros(xyz.size)
        gradient[indices] = data
        return gradient.reshape(xyz.shape)
    return _finite_difference_gradient(lambda value: eval_primitive(primitive, value), xyz, fd_step)


def _compiled_fragment_rotation(primitive: Primitive, coords: np.ndarray):
    """Evaluate a frozen anchor-frame SO(3) chart using native analytic AD.

    Reference positions follow the sorted union of the two body atom sets.
    Anchors are fixed atom identities, never reselected at displaced points.
    """
    if primitive.mode not in (0, 1, 2):
        raise ValueError("frag_rot mode must be 0, 1, or 2")
    if primitive.mode == 2 and (primitive.kind in {"frag_axis", "frag_linear_trans"}
                              or (primitive.kind == "frag_rot" and len(primitive.frame_atoms) == 1)):
        raise ValueError("a linear fragment chart has only two components")
    if primitive.kind == "frag_rot" and (len(primitive.frame_atoms) not in (1, 2) or len(primitive.ref_frame_atoms) != 2):
        raise ValueError("frag_rot requires two frozen anchors per nonlinear body")
    if not set(primitive.frame_atoms) <= set(primitive.atoms) or not set(primitive.ref_frame_atoms) <= set(primitive.ref):
        raise ValueError("frag_rot anchors must belong to their body")
    support = tuple(sorted(set(primitive.atoms + primitive.ref)))
    reference = np.asarray(primitive.rotation_reference if primitive.rotation_reference
                           else np.asarray(coords)[list(support)] if primitive.kind != "frag_rot"
                           else (), dtype=float)
    if reference.shape != (len(support), 3) or not np.all(np.isfinite(reference)):
        raise ValueError("frag_rot requires frozen reference positions on its support")
    mapping = {atom: index for index, atom in enumerate(support)}
    integer = lambda values: np.asarray(values, dtype=np.intp)
    arrays = []
    for atoms in (primitive.atoms, primitive.ref, primitive.frame_atoms, primitive.ref_frame_atoms):
        arrays.extend((integer([mapping[a] for a in atoms]), integer([0, len(atoms)])))
    result = compiled_smith_primitive_plan_csr(
        np.asarray(coords)[list(support)], reference,
        integer([{"frag_trans": 10, "frag_linear_trans": 11, "frag_axis": 15, "frag_rot": 16}[primitive.kind]]), integer([primitive.mode]),
        *arrays, integer([]), integer([]).reshape(0, 4), np.asarray([], dtype=float), integer([0, 0]),
    )
    if result is None:
        raise RuntimeError("frag_rot requires the compiled analytic primitive kernel")
    values, data, indices, indptr = result
    indices = np.asarray(indices, dtype=np.intp)
    global_indices = 3 * np.asarray(support)[indices // 3] + indices % 3
    return values, data, global_indices, indptr


def primitive_b_matrix(
    primitives: Sequence[Primitive], coords: np.ndarray, *, fd_step: float = 1.0e-5
) -> np.ndarray:
    """Evaluate the Wilson matrix without constructing one dense row at a time.

    Ordinary valence primitives touch at most four atoms.  Their derivatives
    are therefore evaluated in compact local regions and scattered directly
    into the preallocated matrix.  Fragment coordinates retain the full
    numerical path because their support is intentionally nonlocal.
    """
    xyz = np.asarray(coords, dtype=float)
    matrix = np.zeros((len(primitives), 3 * xyz.shape[0]), dtype=float)
    for row, primitive in enumerate(primitives):
        if primitive.kind == "selective_out_of_plane":
            matrix[row] = grad_primitive(primitive, xyz).reshape(-1)
            continue
        if primitive.kind in FACE_OPERATORS:
            for column, value in face_value_gradient(primitive, xyz)[1]:
                matrix[row, column] = value
            continue
        if primitive.kind.startswith("frag_"):
            matrix[row] = grad_primitive(primitive, xyz, fd_step=fd_step).reshape(-1)
            continue
        support_atoms = (
            primitive.atoms + primitive.ref
            if primitive.kind == "linear_bend"
            else primitive.atoms
        )
        support = tuple(dict.fromkeys(int(atom) for atom in support_atoms))
        remap = {atom: index for index, atom in enumerate(support)}
        local = Primitive(
            primitive.kind,
            tuple(remap[int(atom)] for atom in primitive.atoms),
            primitive.mode,
            tuple(
                remap[int(atom)]
                for atom in primitive.ref
                if int(atom) in remap
            ),
        )
        local_gradient = grad_primitive(
            local,
            xyz[np.asarray(support, dtype=int)],
            fd_step=fd_step,
        )
        for local_atom, global_atom in enumerate(support):
            matrix[row, 3 * global_atom : 3 * global_atom + 3] = local_gradient[local_atom]
    return matrix


def primitive_b_matrix_sparse(
    primitives: Sequence[Primitive], coords: np.ndarray, *, fd_step: float = 1.0e-5
):
    """Evaluate a local Wilson matrix in CSR storage."""

    from scipy.sparse import csr_matrix

    xyz = np.asarray(coords, dtype=float)
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    for primitive in primitives:
        entries = sparse_grad_primitive_entries(primitive, xyz, fd_step=fd_step)
        indices.extend(column for column, _value in entries)
        data.extend(value for _column, value in entries)
        indptr.append(len(data))
    return csr_matrix(
        (
            np.asarray(data, dtype=float),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(primitives), 3 * xyz.shape[0]),
    )


def sparse_grad_primitive_entries(
    primitive: Primitive,
    coords: np.ndarray,
    *,
    fd_step: float = 1.0e-5,
) -> tuple[tuple[int, float], ...]:
    """Return one primitive Wilson row as sorted global column/value pairs."""

    xyz = np.asarray(coords, dtype=float)
    if primitive.kind == "selective_out_of_plane":
        gradient = grad_primitive(primitive, xyz).reshape(-1)
        return tuple(
            (int(column), float(gradient[column]))
            for column in np.flatnonzero(np.abs(gradient) > 1.0e-15)
        )
    if primitive.kind in FACE_OPERATORS:
        return compiled_face_value_gradient(primitive, xyz)[1]
    if primitive.kind in {"frag_rot", "frag_trans", "frag_linear_trans", "frag_axis"}:
        _, data, indices, _ = _compiled_fragment_rotation(primitive, xyz)
        return tuple((int(i), float(v)) for i, v in zip(indices, data) if abs(v) > 1.e-15)
    if primitive.kind.startswith("frag_"):
        gradient = grad_primitive(primitive, xyz, fd_step=fd_step).reshape(-1)
        return tuple(
            (int(column), float(gradient[column]))
            for column in np.flatnonzero(np.abs(gradient) > 1.0e-15)
        )
    support_atoms = (
        primitive.atoms + primitive.ref
        if primitive.kind == "linear_bend"
        else primitive.atoms
    )
    support = tuple(dict.fromkeys(int(atom) for atom in support_atoms))
    remap = {atom: index for index, atom in enumerate(support)}
    local = Primitive(
        primitive.kind,
        tuple(remap[int(atom)] for atom in primitive.atoms),
        primitive.mode,
        tuple(remap[int(atom)] for atom in primitive.ref if int(atom) in remap),
    )
    local_gradient = grad_primitive(
        local,
        xyz[np.asarray(support, dtype=int)],
        fd_step=fd_step,
    )
    entries = []
    for local_atom, global_atom in enumerate(support):
        for component, value in enumerate(local_gradient[local_atom]):
            if abs(float(value)) > 1.0e-15:
                entries.append((3 * global_atom + component, float(value)))
    return tuple(sorted(entries))


def build_primitive_contract(
    discrete_graph, coords: np.ndarray, *, include_pseudo_bonds: bool = True
) -> PrimitiveCoordinateContract:
    xyz = np.asarray(coords, dtype=float)
    primitives = tuple(
        build_primitives(discrete_graph, xyz, include_pseudo_bonds=include_pseudo_bonds)
    )
    values = tuple(float(value) for value in eval_primitives(primitives, xyz))
    estimated_dense_bytes = len(primitives) * 3 * xyz.shape[0] * 8
    large_sparse = xyz.shape[0] >= 256 or estimated_dense_bytes >= 64 * 1024 * 1024
    b_matrix = (
        primitive_b_matrix_sparse(primitives, xyz)
        if large_sparse
        else primitive_b_matrix(primitives, xyz)
    )
    if large_sparse:
        structural = compiled_csr_support_certificate(
            b_matrix.indices,
            b_matrix.indptr,
            column_count=b_matrix.shape[1],
        )["rank"]
        b_rank = min(int(structural), _rigid_motion_limited_rank(discrete_graph, xyz))
        b_sha256 = _sparse_array_sha256(b_matrix)
        storage = "CSR_LOCAL"
        rank_method = "STRUCTURAL_BIPARTITE_MATCHING_RIGID_GAUGE_CEILING"
    else:
        b_rank = int(np.linalg.matrix_rank(b_matrix, tol=1.0e-10))
        b_sha256 = _array_sha256(b_matrix)
        storage = "DENSE"
        rank_method = "NUMERICAL_SVD"
    return PrimitiveCoordinateContract(
        schema=MATRIX_XYZ_PRIMITIVES_SCHEMA,
        primitives=primitives,
        reference_values=values,
        reference_geometry_sha256=_array_sha256(xyz),
        b_matrix_sha256=b_sha256,
        b_matrix_rank=b_rank,
        b_matrix_columns=int(b_matrix.shape[1]),
        b_matrix_storage=storage,
        b_matrix_rank_method=rank_method,
    )


def _rigid_motion_limited_rank(discrete_graph, coordinates: np.ndarray) -> int:
    """Return the Cartesian rank ceiling after component rigid motions."""

    xyz = np.asarray(coordinates, dtype=float)
    pending = set(range(int(discrete_graph.natoms)))
    rank = 0
    while pending:
        seed = min(pending)
        stack = [seed]
        component: list[int] = []
        pending.remove(seed)
        while stack:
            atom = stack.pop()
            component.append(atom)
            for neighbor in discrete_graph.adjacency[atom]:
                if neighbor in pending:
                    pending.remove(neighbor)
                    stack.append(neighbor)
        count = len(component)
        if count <= 1:
            continue
        if count == 2:
            rank += 1
            continue
        points = xyz[np.asarray(component, dtype=int)]
        centered = points - np.mean(points, axis=0)
        linear = np.linalg.matrix_rank(centered, tol=1.0e-10) <= 1
        rank += 3 * count - (5 if linear else 6)
    return int(rank)


def primitive_contract_section_lines(contract: PrimitiveCoordinateContract) -> list[str]:
    lines = [
        f"SCHEMA {contract.schema}",
        "OWNER ORACLE",
        "INDEXING ATOMS=ONE_BASED",
        "B_MATRIX_EVALUATION REFERENCE_GEOMETRY",
        "B_MATRIX_DERIVATIVE ANALYTIC_LOCAL_NUMERICAL_FRAGMENT",
        "B_MATRIX_FD_STEP_ANGSTROM 1e-5",
        f"REFERENCE_GEOMETRY_SHA256 {contract.reference_geometry_sha256}",
        f"PRIMITIVE_COUNT {len(contract.primitives)}",
        f"B_MATRIX_ROWS {len(contract.primitives)}",
        f"B_MATRIX_COLUMNS {contract.b_matrix_columns}",
        f"B_MATRIX_RANK {contract.b_matrix_rank}",
        f"B_MATRIX_SHA256 {contract.b_matrix_sha256}",
        f"B_MATRIX_STORAGE {contract.b_matrix_storage}",
        f"B_MATRIX_RANK_METHOD {contract.b_matrix_rank_method}",
        "[DEFINITIONS]",
        "COLUMNS ID KIND FUNCTION ATOMS MODE REF VALUE",
    ]
    for index, (primitive, value) in enumerate(
        zip(contract.primitives, contract.reference_values, strict=True), start=1
    ):
        atoms = ",".join(str(atom + 1) for atom in primitive.atoms)
        ref = ",".join(str(atom + 1) for atom in primitive.ref) or "NONE"
        lines.append(
            f"P{index:04d} {primitive.kind} {primitive.function} {atoms} "
            f"{primitive.mode} {ref} {value:.16g}"
        )
        if primitive.kind.startswith("frag_") or primitive.kind in FACE_OPERATORS:
            import json
            lines[-1] += " " + json.dumps({
                "frame_atoms": primitive.frame_atoms,
                "ref_frame_atoms": primitive.ref_frame_atoms,
                "rotation_reference": primitive.rotation_reference,
            }, separators=(",", ":"))
    return lines


def write_primitive_contract(path: Path, contract: PrimitiveCoordinateContract) -> None:
    replace_section(Path(path), "PRIMITIVES", primitive_contract_section_lines(contract))


def read_primitive_contract(path: Path) -> PrimitiveCoordinateContract:
    content = section_content(read_sectioned_lines(Path(path)), "PRIMITIVES")
    if not content:
        raise ValueError(f"missing #PRIMITIVES section in {Path(path)}")
    metadata: dict[str, str] = {}
    primitives: list[Primitive] = []
    values: list[float] = []
    in_definitions = False
    for raw in content:
        text = raw.strip()
        if text == "[DEFINITIONS]":
            in_definitions = True
            continue
        if not text or text.startswith("COLUMNS "):
            continue
        if not in_definitions:
            fields = text.split(maxsplit=1)
            if len(fields) == 2:
                metadata[fields[0]] = fields[1]
            continue
        fields = text.split()
        if len(fields) not in {7, 8} or not fields[0].startswith("P"):
            raise ValueError(f"invalid ORACLE primitive record: {text}")
        atoms = tuple(int(value) - 1 for value in fields[3].split(","))
        ref = () if fields[5] == "NONE" else tuple(int(value) - 1 for value in fields[5].split(","))
        import json
        extra = json.loads(fields[7]) if len(fields) > 7 else {}
        primitives.append(Primitive(fields[1], atoms, int(fields[4]), ref,
            tuple(extra.get("frame_atoms", ())), tuple(extra.get("ref_frame_atoms", ())),
            tuple(tuple(row) for row in extra.get("rotation_reference", ()))))
        values.append(float(fields[6]))
    schema = metadata.get("SCHEMA", "")
    if schema not in {
        MATRIX_XYZ_PRIMITIVES_SCHEMA,
        LEGACY_MATRIX_XYZ_PRIMITIVES_SCHEMA,
        LEGACY_ORACLE_XYZ_PRIMITIVES_SCHEMA,
    }:
        raise ValueError(f"unsupported ORACLE primitive schema: {schema or 'missing'}")
    if schema in {LEGACY_MATRIX_XYZ_PRIMITIVES_SCHEMA, LEGACY_ORACLE_XYZ_PRIMITIVES_SCHEMA}:
        # v1 evaluated the serialized tuple (i,j,k,l) as the angle of i from
        # the k-j-l plane.  U(j,l,k,i) in the Gaussian convention has exactly
        # the same value and Wilson row, so this permutation losslessly
        # migrates frozen contracts without regenerating their fingerprints.
        primitives = [
            Primitive(primitive.kind, (atoms[1], atoms[3], atoms[2], atoms[0]), primitive.mode, primitive.ref)
            if primitive.kind == "out_of_plane" and len((atoms := primitive.atoms)) == 4
            else primitive
            for primitive in primitives
        ]
    if int(metadata.get("PRIMITIVE_COUNT", "-1")) != len(primitives):
        raise ValueError("#PRIMITIVES count does not match its definitions")
    return PrimitiveCoordinateContract(
        schema=schema,
        primitives=tuple(primitives),
        reference_values=tuple(values),
        reference_geometry_sha256=metadata.get("REFERENCE_GEOMETRY_SHA256", ""),
        b_matrix_sha256=metadata.get("B_MATRIX_SHA256", ""),
        b_matrix_rank=int(metadata.get("B_MATRIX_RANK", "0")),
        b_matrix_columns=int(metadata.get("B_MATRIX_COLUMNS", "0")),
        b_matrix_storage=metadata.get("B_MATRIX_STORAGE", "DENSE"),
        b_matrix_rank_method=metadata.get("B_MATRIX_RANK_METHOD", "NUMERICAL_SVD"),
    )


def validate_primitive_contract(
    contract: PrimitiveCoordinateContract, coords: np.ndarray, *, atol: float = 2.0e-8
) -> None:
    xyz = np.asarray(coords, dtype=float)
    coordinate_selection_units(
        tuple(
            CoordinateComponent(
                operator=primitive.function,
                atoms=primitive.atoms,
                mode=primitive.mode,
                ref_atoms=primitive.ref,
                context=(primitive.kind,),
            )
            for primitive in contract.primitives
        )
    )
    if contract.reference_geometry_sha256 and contract.reference_geometry_sha256 != _array_sha256(xyz):
        raise ValueError("#PRIMITIVES reference geometry does not match the Cartesian block")
    values = eval_primitives(contract.primitives, xyz)
    differences = values - np.asarray(contract.reference_values, dtype=float)
    for index, primitive in enumerate(contract.primitives):
        if primitive.kind == "dihedral":
            differences[index] = (
                differences[index] + np.pi
            ) % (2.0 * np.pi) - np.pi
    if not np.allclose(differences, 0.0, atol=atol, rtol=0.0):
        raise ValueError("#PRIMITIVES reference values do not match the Cartesian block")
    if contract.b_matrix_storage == "CSR_LOCAL":
        b_matrix = primitive_b_matrix_sparse(contract.primitives, xyz)
        fingerprint = _sparse_array_sha256(b_matrix)
    else:
        b_matrix = primitive_b_matrix(contract.primitives, xyz)
        fingerprint = _array_sha256(b_matrix)
    if contract.b_matrix_sha256 and contract.b_matrix_sha256 != fingerprint:
        raise ValueError("#PRIMITIVES Wilson B fingerprint does not match the Cartesian block")


def angle(i: int, j: int, k: int, coords: np.ndarray) -> float:
    u = _unit(coords[i] - coords[j])
    v = _unit(coords[k] - coords[j])
    return float(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0)))


def linear_bend_reference_atom(
    atoms: tuple[int, int, int],
    coords: np.ndarray,
    *,
    minimum_angular_deviation: float = 1.0e-3,
    candidate_indices: Sequence[int] | None = None,
) -> int | None:
    """Choose Gaussian's deterministic fourth atom for a local linear bend.

    The score follows Gaussian's ``CrdCon`` rule: maximize the smaller
    deviation from 0 or pi of the two angles made by the candidate reference
    with the two arms of ``A1-O-A2``.  A fully linear molecule has no valid
    real reference and retains the plane-based three-atom convention.
    """

    xyz = np.asarray(coords, dtype=float)
    endpoint1, center, endpoint2 = atoms
    candidates: list[tuple[float, int]] = []
    references = (
        range(len(xyz))
        if candidate_indices is None
        else tuple(int(value) for value in candidate_indices)
    )
    for reference in references:
        if reference < 0 or reference >= len(xyz):
            raise ValueError("linear-bend reference candidate is outside the geometry")
        if reference in atoms:
            continue
        first = angle(endpoint1, center, reference, xyz)
        second = angle(reference, center, endpoint2, xyz)
        score = min(first, second, np.pi - first, np.pi - second)
        if np.isfinite(score):
            candidates.append((float(score), reference))
    if not candidates:
        return None
    score, reference = max(candidates, key=lambda item: (item[0], -item[1]))
    return reference if score > float(minimum_angular_deviation) else None


def dihedral(i: int, j: int, k: int, ell: int, coords: np.ndarray) -> float:
    b1 = coords[i] - coords[j]
    b2 = coords[k] - coords[j]
    b3 = coords[ell] - coords[k]
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    return float(np.arctan2(np.dot(np.cross(n1, n2), _unit(b2)), np.dot(n1, n2)))


def out_of_plane(i: int, j: int, k: int, ell: int, coords: np.ndarray) -> float:
    """Gaussian U(i,j,k,l): center i, plane i-j-k, out-of-plane atom l."""

    vector = coords[ell] - coords[i]
    normal = np.cross(coords[j] - coords[i], coords[k] - coords[i])
    denominator = np.linalg.norm(vector) * np.linalg.norm(normal)
    if denominator < 1.0e-12:
        return 0.0
    return float(-np.arcsin(np.clip(np.dot(vector, normal) / denominator, -1.0, 1.0)))


def out_of_plane_height(i: int, j: int, k: int, ell: int, coords: np.ndarray) -> float:
    """Gaussian H(i,j,k,l): signed distance of l from the j-i-k plane."""

    normal = np.cross(coords[j] - coords[i], coords[k] - coords[i])
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1.0e-12:
        return 0.0
    return float(np.dot(coords[ell] - coords[i], normal / normal_norm))


def linear_components(i: int, j: int, k: int, coords: np.ndarray) -> tuple[float, float]:
    u = _unit(coords[i] - coords[j])
    v = _unit(coords[k] - coords[j])
    axis = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, u))) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])
    first = _unit(np.cross(u, axis))
    second = _unit(np.cross(u, first))
    bending = v + u
    return float(np.dot(bending, first)), float(np.dot(bending, second))


def referenced_linear_bend_perpendicular_value(
    first_dihedral: float,
    second_dihedral: float,
) -> float:
    """Return the continuous perpendicular component of a four-atom L chart."""

    difference = (
        float(first_dihedral) - float(second_dihedral) + np.pi
    ) % (2.0 * np.pi) - np.pi
    return float(np.pi + 0.5 * difference)


def _four_atom_linear_bend_value(
    i: int,
    j: int,
    k: int,
    reference: int,
    coords: np.ndarray,
    *,
    mode: int,
) -> float:
    """Evaluate Gaussian's four-atom L(A1,O,A2,A4,component)."""

    if mode == -1:
        return angle(i, j, reference, coords) + angle(reference, j, k, coords)
    if mode == -2:
        return referenced_linear_bend_perpendicular_value(
            dihedral(i, j, reference, k, coords),
            dihedral(i, reference, j, k, coords),
        )
    raise ValueError(f"invalid linear-bend mode: {mode}")


def _eval_fragment_primitive(primitive: Primitive, coords: np.ndarray) -> float:
    fragment = np.asarray(primitive.atoms, dtype=int)
    reference = np.asarray(primitive.ref, dtype=int)
    delta = coords[fragment].mean(axis=0) - coords[reference].mean(axis=0)
    if primitive.kind == "frag_trans":
        return float(delta[primitive.mode])
    if primitive.kind in {"frag_dist", "frag_atom_dist"}:
        return float(np.linalg.norm(delta))
    def frame(indices: np.ndarray) -> np.ndarray:
        centered = coords[indices] - coords[indices].mean(axis=0)
        inertia = np.zeros((3, 3), dtype=float)
        for vector in centered:
            inertia += np.dot(vector, vector) * np.eye(3) - np.outer(vector, vector)
        values, vectors = np.linalg.eigh(inertia)
        result = vectors[:, np.argsort(values)]
        if np.linalg.det(result) < 0.0:
            result[:, -1] *= -1.0
        return result

    rotation = frame(reference).T @ frame(fragment)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        quaternion = np.asarray(
            [
                0.25 / scale,
                (rotation[2, 1] - rotation[1, 2]) * scale,
                (rotation[0, 2] - rotation[2, 0]) * scale,
                (rotation[1, 0] - rotation[0, 1]) * scale,
            ]
        )
    else:
        diagonal = np.diag(rotation)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            quaternion = np.asarray(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            quaternion = np.asarray(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            quaternion = np.asarray(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    vector = quaternion[1:]
    norm = float(np.linalg.norm(vector))
    rotation_vector = (
        np.zeros(3) if norm < 1.0e-12 else vector / norm * (2.0 * np.arctan2(norm, quaternion[0]))
    )
    return float(rotation_vector[primitive.mode])


def _bond_grad(i: int, j: int, coords: np.ndarray) -> np.ndarray:
    delta = coords[i] - coords[j]
    distance = np.linalg.norm(delta)
    gradient = np.zeros_like(coords)
    if distance > 1.0e-12:
        gradient[i], gradient[j] = delta / distance, -delta / distance
    return gradient


def _fragment_distance_grad(primitive: Primitive, coords: np.ndarray) -> np.ndarray:
    fragment = tuple(int(atom) for atom in primitive.atoms)
    reference = tuple(int(atom) for atom in primitive.ref)
    if not fragment or not reference:
        raise ValueError("fragment distance requires two nonempty atom sets")
    delta = coords[np.asarray(fragment)].mean(axis=0) - coords[np.asarray(reference)].mean(axis=0)
    distance = float(np.linalg.norm(delta))
    if distance <= 1.0e-12:
        raise FloatingPointError("fragment centers are coincident")
    unit = delta / distance
    gradient = np.zeros_like(coords)
    for atom in fragment:
        gradient[atom] += unit / len(fragment)
    for atom in reference:
        gradient[atom] -= unit / len(reference)
    return gradient


def _fragment_translation_grad(primitive: Primitive, coords: np.ndarray) -> np.ndarray:
    fragment = tuple(int(atom) for atom in primitive.atoms)
    reference = tuple(int(atom) for atom in primitive.ref)
    if not fragment or not reference or primitive.mode not in {0, 1, 2}:
        raise ValueError("fragment translation requires two atom sets and mode 0, 1, or 2")
    axis = np.zeros(3, dtype=float)
    axis[primitive.mode] = 1.0
    gradient = np.zeros_like(coords)
    for atom in fragment:
        gradient[atom] += axis / len(fragment)
    for atom in reference:
        gradient[atom] -= axis / len(reference)
    return gradient


def _angle_grad(i: int, j: int, k: int, coords: np.ndarray) -> np.ndarray:
    left, right = coords[i] - coords[j], coords[k] - coords[j]
    left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
    gradient = np.zeros_like(coords)
    if left_norm < 1.0e-12 or right_norm < 1.0e-12:
        return gradient
    u, v = left / left_norm, right / right_norm
    cosine = np.clip(np.dot(u, v), -1.0, 1.0)
    sine = np.sqrt(max(1.0 - cosine * cosine, 1.0e-16))
    gradient[i] = (-v + cosine * u) / (sine * left_norm)
    gradient[k] = (-u + cosine * v) / (sine * right_norm)
    gradient[j] = -(gradient[i] + gradient[k])
    return gradient


def _dihedral_grad(i: int, j: int, k: int, ell: int, coords: np.ndarray) -> np.ndarray:
    """Algorithmic derivative of the signed dihedral definition."""
    class Dual:
        __slots__ = ("value", "derivative")

        def __init__(self, value, derivative):
            self.value = float(value)
            self.derivative = derivative

        def __add__(self, other):
            other = other if isinstance(other, Dual) else Dual(other, 0.0 * self.derivative)
            return Dual(self.value + other.value, self.derivative + other.derivative)

        def __sub__(self, other):
            other = other if isinstance(other, Dual) else Dual(other, 0.0 * self.derivative)
            return Dual(self.value - other.value, self.derivative - other.derivative)

        def __mul__(self, other):
            other = other if isinstance(other, Dual) else Dual(other, 0.0 * self.derivative)
            return Dual(
                self.value * other.value,
                self.value * other.derivative + other.value * self.derivative,
            )

        def __truediv__(self, other):
            other = other if isinstance(other, Dual) else Dual(other, 0.0 * self.derivative)
            inverse = 1.0 / other.value
            return Dual(
                self.value * inverse,
                (self.derivative - self.value * other.derivative * inverse) * inverse,
            )

    def dot(left, right):
        return sum((left[index] * right[index] for index in range(3)), start=Dual(0.0, np.zeros(12)))

    def cross(left, right):
        return [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ]

    def unit(vector):
        squared = dot(vector, vector)
        length_value = np.sqrt(squared.value)
        length = Dual(length_value, squared.derivative / (2.0 * length_value))
        return [component / length for component in vector]

    base = coords[[i, j, k, ell]].reshape(-1)
    variables: list[Dual] = []
    for index, value in enumerate(base):
        derivative = np.zeros(12)
        derivative[index] = 1.0
        variables.append(Dual(value, derivative))
    vectors = [variables[start : start + 3] for start in (0, 3, 6, 9)]
    ri, rj, rk, rell = vectors
    b1 = [ri[m] - rj[m] for m in range(3)]
    b2 = [rk[m] - rj[m] for m in range(3)]
    b3 = [rell[m] - rk[m] for m in range(3)]
    n1, n2 = cross(b1, b2), cross(b2, b3)
    x = dot(n1, n2)
    y = dot(cross(n1, n2), unit(b2))
    denominator = x.value * x.value + y.value * y.value
    derivative = (x.value * y.derivative - y.value * x.derivative) / denominator
    local = derivative.reshape(4, 3)
    gradient = np.zeros_like(coords)
    gradient[i], gradient[j], gradient[k], gradient[ell] = local
    return gradient


def _cross_matrix(vector: np.ndarray) -> np.ndarray:
    return np.asarray(
        [[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]]
    )


def _out_of_plane_grad(i: int, j: int, k: int, ell: int, coords: np.ndarray) -> np.ndarray:
    vector = coords[ell] - coords[i]
    left, right = coords[j] - coords[i], coords[k] - coords[i]
    vector_norm = np.linalg.norm(vector)
    normal = np.cross(left, right)
    normal_norm = np.linalg.norm(normal)
    if vector_norm < 1.0e-12 or normal_norm < 1.0e-12:
        return np.zeros_like(coords)
    unit_vector, unit_normal = vector / vector_norm, normal / normal_norm
    sine = np.clip(np.dot(unit_vector, unit_normal), -0.999999, 0.999999)
    denominator = np.sqrt(1.0 - sine * sine)
    if denominator < 1.0e-12:
        return np.zeros_like(coords)
    unit_jacobian = (np.eye(3) - np.outer(unit_vector, unit_vector)) / vector_norm
    normal_jacobian = (np.eye(3) - np.outer(unit_normal, unit_normal)) / normal_norm
    gradient_vector = unit_jacobian @ unit_normal
    gradient_normal = normal_jacobian @ unit_vector
    gradient = np.zeros_like(coords)
    gradient[ell] = gradient_vector
    gradient[i] = -gradient_vector + (_cross_matrix(right) - _cross_matrix(left)).T @ gradient_normal
    gradient[j] = (-_cross_matrix(right)).T @ gradient_normal
    gradient[k] = _cross_matrix(left).T @ gradient_normal
    return -gradient / denominator


def _out_of_plane_height_grad(
    i: int, j: int, k: int, ell: int, coords: np.ndarray
) -> np.ndarray:
    vector = coords[ell] - coords[i]
    left, right = coords[j] - coords[i], coords[k] - coords[i]
    normal = np.cross(left, right)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1.0e-12:
        return np.zeros_like(coords)
    unit_normal = normal / normal_norm
    normal_jacobian = (np.eye(3) - np.outer(unit_normal, unit_normal)) / normal_norm
    gradient_normal = normal_jacobian @ vector
    gradient = np.zeros_like(coords)
    gradient[ell] = unit_normal
    gradient[i] = -unit_normal + (_cross_matrix(right) - _cross_matrix(left)).T @ gradient_normal
    gradient[j] = (-_cross_matrix(right)).T @ gradient_normal
    gradient[k] = _cross_matrix(left).T @ gradient_normal
    return gradient


def _linear_grad(i: int, j: int, k: int, coords: np.ndarray, *, mode: int) -> np.ndarray:
    left, right = coords[i] - coords[j], coords[k] - coords[j]
    left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
    if left_norm < 1.0e-12 or right_norm < 1.0e-12:
        return np.zeros_like(coords)
    u, v = left / left_norm, right / right_norm
    axis = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, u))) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])
    cross = np.cross(u, axis)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1.0e-12:
        return np.zeros_like(coords)
    first = cross / cross_norm
    second = np.cross(u, first)
    bending = v + u
    identity = np.eye(3)
    jacobian_u = (identity - np.outer(u, u)) / left_norm
    jacobian_v = (identity - np.outer(v, v)) / right_norm
    projector = identity / cross_norm - np.outer(cross, cross) / cross_norm**3
    first_u = projector @ (-_cross_matrix(axis))
    # d( u x first )/du = -[first]_x + [u]_x d(first)/du.
    second_u = -_cross_matrix(first) + _cross_matrix(u) @ first_u
    if mode == -1:
        derivative_u, derivative_v = first + first_u.T @ bending, first
    else:
        derivative_u, derivative_v = second + second_u.T @ bending, second
    gradient = np.zeros_like(coords)
    gradient[i] = jacobian_u.T @ derivative_u
    gradient[k] = jacobian_v.T @ derivative_v
    gradient[j] = -(gradient[i] + gradient[k])
    return gradient


def _four_atom_linear_bend_grad(
    i: int,
    j: int,
    k: int,
    reference: int,
    coords: np.ndarray,
    *,
    mode: int,
) -> np.ndarray:
    """Analytic Wilson row for Gaussian's referenced local linear bend."""

    if mode == -1:
        return _angle_grad(i, j, reference, coords) + _angle_grad(
            reference,
            j,
            k,
            coords,
        )
    if mode == -2:
        return 0.5 * (
            _dihedral_grad(i, j, reference, k, coords)
            - _dihedral_grad(i, reference, j, k, coords)
        )
    raise ValueError(f"invalid linear-bend mode: {mode}")


def _finite_difference_gradient(function, coords: np.ndarray, step: float) -> np.ndarray:
    gradient = np.zeros_like(coords)
    for atom in range(coords.shape[0]):
        for axis in range(3):
            plus, minus = coords.copy(), coords.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            delta = float(function(plus) - function(minus))
            if abs(delta) > np.pi and abs(delta) < 2.0 * np.pi + 1.0e-8:
                delta -= np.sign(delta) * 2.0 * np.pi
            gradient[atom, axis] = delta / (2.0 * step)
    return gradient


def _unit(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    return np.zeros_like(vector) if length < 1.0e-12 else vector / length


def _angle_key(atoms: tuple[int, int, int]) -> tuple[int, int, int]:
    left, center, right = atoms
    return min((left, center, right), (right, center, left))


def _array_sha256(values: np.ndarray) -> str:
    # Primitive rows can differ in their final floating-point bits across
    # Python/NumPy platforms even when they implement the same contract.
    # Quantizing below the validation tolerance makes the fingerprint stable
    # without masking a chemically meaningful change.  Normalize signed zero
    # as well, since it has no geometric meaning but a different byte pattern.
    canonical = np.round(np.asarray(values, dtype="<f8"), decimals=12)
    canonical[canonical == 0.0] = 0.0
    canonical = np.ascontiguousarray(canonical, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _sparse_array_sha256(values) -> str:
    """Fingerprint CSR content without expanding absent zeros."""

    matrix = values.tocsr(copy=True)
    matrix.sort_indices()
    data = np.round(np.asarray(matrix.data, dtype="<f8"), decimals=12)
    data[data == 0.0] = 0.0
    payload = np.asarray(matrix.shape, dtype="<i8").tobytes()
    payload += np.asarray(matrix.indptr, dtype="<i8").tobytes()
    payload += np.asarray(matrix.indices, dtype="<i8").tobytes()
    payload += np.ascontiguousarray(data, dtype="<f8").tobytes()
    return hashlib.sha256(payload).hexdigest()
