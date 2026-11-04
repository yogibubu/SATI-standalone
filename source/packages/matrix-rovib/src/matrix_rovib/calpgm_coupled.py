"""Reduced coupled-F CALPGM Hamiltonian assembly.

The rotational K coordinate and the space-fixed angular-momentum coupling
chain are independent.  This module keeps them separate and contracts the
space-fixed part with Wigner coefficients before any Hamiltonian is built.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .calpgm_adapter import (
    build_calpgm_coupled_basis,
    build_calpgm_symmetry_subblocks,
    calpgm_wang_transform,
)
from .calpgm_directional import (
    CALPGMDirectionalRoute,
    directional_route,
    directional_subblock_elements,
    electron_spin_tensor_factor,
    equivalent_spin_tensor_factor,
    rotational_reduced_factor,
)
from .calpgm_native import (
    CALPGMOperatorCode,
    calpgm_fourier_modulate,
    calpgm_fourier_options,
    calpgm_fourier_rho_values,
    calpgm_selected_k_modulate,
    calpgm_rotational_projection,
    calpgm_spin_twice_from_model,
    calpgm_vibrational_aliases,
    calpgm_vibrational_l_values,
    calpgm_vibrational_projection,
    decode_calpgm_identifier,
)
from .native_rotor import angular_momentum_operators
from .spin_hyperfine import (
    _anisotropic_spin_rotation_scalar_factor,
    _coupled_f_vector,
    _e1_recoupling_amplitude,
    _isotropic_nuclear_scalar_factor,
    _isotropic_spin_rotation_shift,
    _rank1_electron_spin_scalar_factor,
    _rank1_rotational_space_component,
    _rank2_rotational_space_component,
    _rank2_spin_scalar_factor,
    _angular_momentum_rank1_component,
    _coupled_nuclear_rank1_component,
    clebsch_gordan_doubled,
    rank1_body_block,
    rank2_body_block,
)


@dataclass(frozen=True)
class CALPGMCoupledGroup:
    vibration: int
    nn: int
    N: int
    two_J: int
    two_I: int
    two_F: int
    nuclear_symmetry: int
    start: int
    stop: int


@dataclass(frozen=True)
class CALPGMCoupledPiece:
    row: int
    column: int
    matrix: np.ndarray


@dataclass(frozen=True)
class CALPGMCoupledOperator:
    label: str
    identifier: int
    pieces: tuple[CALPGMCoupledPiece, ...]
    symmetry_pieces: tuple[CALPGMCoupledPiece, ...] = ()


@dataclass(frozen=True)
class CALPGMCoupledBlock:
    F: int
    two_S: int
    basis_labels: tuple[Mapping[str, Any], ...]
    groups: tuple[CALPGMCoupledGroup, ...]
    operators: tuple[CALPGMCoupledOperator, ...]
    parameter_defaults: Mapping[str, float]
    symmetry_subblocks: tuple[Mapping[str, Any], ...] = ()
    working_basis_labels: tuple[Mapping[str, Any], ...] = ()
    symmetry_transform: np.ndarray | None = None
    symmetry_transform_sparse: Any | None = None
    symmetry_block_ids: tuple[int, ...] = ()
    nuclear_symmetry_order: int = 0

    @property
    def dimension(self) -> int:
        return len(self.basis_labels)


def compile_calpgm_coupled_block(
    model: Mapping[str, Any], aggregate_map: Mapping[str, Any], F: int,
    *, n_max: int | None = None,
) -> CALPGMCoupledBlock:
    """Compile one reduced total-F block into sparse K-space pieces."""

    spin_twice = calpgm_spin_twice_from_model(model)
    if not spin_twice:
        raise ValueError("the coupled-F CALPGM assembler requires a declared spin")
    if any(spin_twice) and not bool(aggregate_map.get("itot_basis")):
        raise ValueError("the reduced coupled-F assembler requires CALPGM ITOT basis labels")
    two_S = spin_twice[0]
    working_labels = tuple(build_calpgm_coupled_basis(aggregate_map, F, n_max=n_max))
    if not working_labels:
        raise ValueError(f"CALPGM aggregate map contains no states for F={F}")
    groups = _groups_from_labels(working_labels)
    symmetry_subblocks = tuple(
        build_calpgm_symmetry_subblocks(aggregate_map, F, n_max=n_max)
    )
    symmetry_dimension = sum(int(item["dimension"]) for item in symmetry_subblocks)
    if symmetry_dimension > len(working_labels):
        raise ValueError("CALPGM GETQQ reconstruction exceeds the working basis")
    labels, symmetry_transform = _calpgm_symmetry_transform(
        working_labels, groups, symmetry_subblocks
    )
    state_count = max(1, abs(int(model["options"][0]["nvib"])))
    aliases = calpgm_vibrational_aliases(model.get("options", ()), state_count)
    l_values = calpgm_vibrational_l_values(model.get("options", ()), state_count)
    nitot = _calpgm_nuclear_symmetry_order(model.get("options", ()))
    parameters = tuple(model.get("parameters", ()))
    rho_values = calpgm_fourier_rho_values(model)
    nofc, g12_phase = calpgm_fourier_options(model)
    keys = _parameter_keys(parameters)
    operators = []
    defaults = {}
    for key, parameter in zip(keys, parameters, strict=True):
        code = decode_calpgm_identifier(int(parameter["identifier"]), state_count=state_count)
        if code.operator_family == "series-constant":
            symmetry_pieces = ()
            pieces = ()
        else:
            symmetry_pieces = _directional_operator_pieces(
                code, symmetry_subblocks, spin_twice, l_values, nitot,
                rho_values, nofc, g12_phase,
            )
            pieces = () if symmetry_pieces is not None else _operator_pieces(
                code, groups, two_S, state_count, aliases, l_values,
                rho_values, nofc, g12_phase,
            )
        operators.append(CALPGMCoupledOperator(
            key, int(parameter["identifier"]), pieces, symmetry_pieces or ()
        ))
        defaults[key] = float(parameter["value"])
    sparse_transform = None
    if symmetry_transform is not None:
        from scipy.sparse import csr_matrix

        sparse_transform = csr_matrix(symmetry_transform)
    return CALPGMCoupledBlock(
        F=int(F), two_S=two_S, basis_labels=labels, groups=groups,
        operators=tuple(operators), parameter_defaults=defaults,
        symmetry_subblocks=symmetry_subblocks,
        working_basis_labels=working_labels,
        symmetry_transform=symmetry_transform,
        symmetry_transform_sparse=sparse_transform,
        symmetry_block_ids=tuple(int(label["calpgm_block"]) for label in labels),
        nuclear_symmetry_order=nitot,
    )


def assemble_calpgm_coupled_block(
    block: CALPGMCoupledBlock, parameters: Mapping[str, float] | None = None
) -> np.ndarray:
    values = dict(block.parameter_defaults)
    if parameters is not None:
        unknown = set(parameters) - set(values)
        if unknown:
            raise ValueError(f"unknown CALPGM parameters: {sorted(unknown)}")
        values.update({str(key): float(value) for key, value in parameters.items()})
    working_dimension = (
        len(block.working_basis_labels)
        if block.symmetry_transform is not None else block.dimension
    )
    matrix = np.zeros((working_dimension, working_dimension), dtype=complex)
    for operator in block.operators:
        coefficient = values[operator.label]
        for piece in operator.pieces:
            rows, columns = piece.matrix.shape
            matrix[piece.row:piece.row + rows, piece.column:piece.column + columns] += (
                coefficient * piece.matrix
            )
    matrix = 0.5 * (matrix + matrix.conj().T)
    if block.symmetry_transform is not None:
        matrix = block.symmetry_transform.conj().T @ matrix @ block.symmetry_transform
    if block.symmetry_block_ids:
        block_ids = np.asarray(block.symmetry_block_ids)
        matrix[block_ids[:, None] != block_ids[None, :]] = 0.0
    for operator in block.operators:
        _accumulate_symmetry_pieces(matrix, operator.symmetry_pieces, values[operator.label])
    return matrix


def assemble_calpgm_coupled_sparse(
    block: CALPGMCoupledBlock, parameters: Mapping[str, float] | None = None
):
    """Assemble one coupled block directly as a Hermitian CSR matrix."""

    from scipy.sparse import coo_matrix, csr_matrix

    values = dict(block.parameter_defaults)
    if parameters is not None:
        unknown = set(parameters) - set(values)
        if unknown:
            raise ValueError(f"unknown CALPGM parameters: {sorted(unknown)}")
        values.update({str(key): float(value) for key, value in parameters.items()})
    working_dimension = (
        len(block.working_basis_labels)
        if block.symmetry_transform is not None else block.dimension
    )

    def embedded(piece: CALPGMCoupledPiece, dimension: int):
        local = np.asarray(piece.matrix)
        row, column = np.nonzero(local)
        return coo_matrix(
            (local[row, column], (piece.row + row, piece.column + column)),
            shape=(dimension, dimension),
            dtype=complex,
        ).tocsr()

    working = csr_matrix((working_dimension, working_dimension), dtype=complex)
    for operator in block.operators:
        coefficient = values[operator.label]
        for piece in operator.pieces:
            working += coefficient * embedded(piece, working_dimension)
    working = 0.5 * (working + working.getH())
    transform = block.symmetry_transform_sparse
    matrix = transform.getH() @ working @ transform if transform is not None else working
    for operator in block.operators:
        coefficient = values[operator.label]
        for piece in operator.symmetry_pieces:
            contribution = coefficient * embedded(piece, block.dimension)
            matrix += contribution
            if piece.row != piece.column:
                matrix += contribution.getH()
    matrix = matrix.tocoo()
    if block.symmetry_block_ids:
        ids = np.asarray(block.symmetry_block_ids, dtype=int)
        keep = ids[matrix.row] == ids[matrix.col]
        matrix = coo_matrix(
            (matrix.data[keep], (matrix.row[keep], matrix.col[keep])),
            shape=matrix.shape,
        )
    result = matrix.tocsr()
    result.sum_duplicates()
    result.eliminate_zeros()
    return result


def apply_calpgm_coupled_block(
    block: CALPGMCoupledBlock,
    vector: np.ndarray,
    parameters: Mapping[str, float] | None = None,
    *,
    symmetry_block_id: int | None = None,
) -> np.ndarray:
    values = dict(block.parameter_defaults)
    if parameters is not None:
        unknown = set(parameters) - set(values)
        if unknown:
            raise ValueError(f"unknown CALPGM parameters: {sorted(unknown)}")
        values.update({str(key): float(value) for key, value in parameters.items()})
    source = np.asarray(vector, dtype=complex)
    if source.shape != (block.dimension,):
        raise ValueError("coupled-F CALPGM vector has the wrong dimension")
    transform = block.symmetry_transform
    sparse_transform = block.symmetry_transform_sparse
    projectors = (None,)
    if transform is not None and block.symmetry_block_ids:
        ids = np.asarray(block.symmetry_block_ids)
        symmetry_values = (
            (int(symmetry_block_id),)
            if symmetry_block_id is not None
            else tuple(sorted(set(block.symmetry_block_ids)))
        )
        if symmetry_block_id is not None and symmetry_block_id not in set(
            block.symmetry_block_ids
        ):
            raise ValueError("unknown CALPGM symmetry block")
        projectors = tuple(ids == value for value in symmetry_values)
    result = np.zeros_like(source)
    for projector in projectors:
        symmetry_source = source if projector is None else np.where(projector, source, 0.0)
        if sparse_transform is not None:
            working_source = np.asarray(sparse_transform @ symmetry_source).reshape(-1)
        else:
            working_source = transform @ symmetry_source if transform is not None else symmetry_source
        forward = np.zeros_like(working_source)
        adjoint = np.zeros_like(working_source)
        for operator in block.operators:
            coefficient = values[operator.label]
            for piece in operator.pieces:
                rows, columns = piece.matrix.shape
                forward[piece.row:piece.row + rows] += (
                    coefficient
                    * piece.matrix
                    @ working_source[piece.column:piece.column + columns]
                )
                adjoint[piece.column:piece.column + columns] += (
                    coefficient
                    * piece.matrix.conj().T
                    @ working_source[piece.row:piece.row + rows]
                )
        partial = 0.5 * (forward + adjoint)
        if sparse_transform is not None:
            partial = np.asarray(sparse_transform.conj().T @ partial).reshape(-1)
        elif transform is not None:
            partial = transform.conj().T @ partial
        result += partial if projector is None else np.where(projector, partial, 0.0)
    for operator in block.operators:
        coefficient = values[operator.label]
        for piece in operator.symmetry_pieces:
            rows, columns = piece.matrix.shape
            row = slice(piece.row, piece.row + rows)
            column = slice(piece.column, piece.column + columns)
            result[row] += coefficient * piece.matrix @ source[column]
            if piece.row != piece.column:
                result[column] += coefficient * piece.matrix.conj().T @ source[row]
    return result


def calpgm_coupled_block_diagonal(
    block: CALPGMCoupledBlock, parameters: Mapping[str, float] | None = None
) -> np.ndarray:
    """Return the real diagonal without materializing the reduced block."""

    values = dict(block.parameter_defaults)
    if parameters is not None:
        unknown = set(parameters) - set(values)
        if unknown:
            raise ValueError(f"unknown CALPGM parameters: {sorted(unknown)}")
        values.update({str(key): float(value) for key, value in parameters.items()})
    diagonal = np.zeros(block.dimension, dtype=float)
    transform = block.symmetry_transform_sparse
    if transform is not None:
        from scipy.sparse import csr_matrix

        for operator in block.operators:
            coefficient = values[operator.label]
            for piece in operator.pieces:
                rows, columns = piece.matrix.shape
                left = transform[piece.row:piece.row + rows, :]
                right = transform[piece.column:piece.column + columns, :]
                product = csr_matrix(piece.matrix) @ right
                projected = left.conjugate().multiply(product).sum(axis=0)
                diagonal += coefficient * np.asarray(projected).reshape(-1).real
    else:
        for operator in block.operators:
            coefficient = values[operator.label]
            for piece in operator.pieces:
                if piece.row != piece.column:
                    continue
                diagonal[piece.row:piece.row + piece.matrix.shape[0]] += (
                    coefficient * np.real(np.diag(piece.matrix))
                )
    for operator in block.operators:
        coefficient = values[operator.label]
        for piece in operator.symmetry_pieces:
            if piece.row != piece.column:
                continue
            diagonal[piece.row:piece.row + piece.matrix.shape[0]] += (
                coefficient * np.real(np.diag(piece.matrix))
            )
    return diagonal


def calpgm_coupled_operator_expectations(
    block: CALPGMCoupledBlock, vectors: np.ndarray
) -> tuple[tuple[str, ...], np.ndarray]:
    states = np.asarray(vectors, dtype=complex)
    if states.ndim != 2 or states.shape[0] != block.dimension:
        raise ValueError("coupled-F CALPGM eigenvectors have the wrong shape")
    derivatives = np.zeros((states.shape[1], len(block.operators)), dtype=float)
    projected_states = (states,)
    if block.symmetry_transform is not None and block.symmetry_block_ids:
        ids = np.asarray(block.symmetry_block_ids)
        projected_states = tuple(
            block.symmetry_transform @ np.where((ids == value)[:, None], states, 0.0)
            for value in sorted(set(block.symmetry_block_ids))
        )
    elif block.symmetry_transform is not None:
        projected_states = (block.symmetry_transform @ states,)
    for parameter, operator in enumerate(block.operators):
        for projected in projected_states:
            forward = np.zeros_like(projected)
            adjoint = np.zeros_like(projected)
            for piece in operator.pieces:
                rows, columns = piece.matrix.shape
                forward[piece.row:piece.row + rows] += (
                    piece.matrix @ projected[piece.column:piece.column + columns]
                )
                adjoint[piece.column:piece.column + columns] += (
                    piece.matrix.conj().T @ projected[piece.row:piece.row + rows]
                )
            applied = 0.5 * (forward + adjoint)
            derivatives[:, parameter] += np.real(
                np.einsum("ij,ij->j", projected.conj(), applied)
            )
        for piece in operator.symmetry_pieces:
            rows, columns = piece.matrix.shape
            row = slice(piece.row, piece.row + rows)
            column = slice(piece.column, piece.column + columns)
            applied = piece.matrix @ states[column]
            derivatives[:, parameter] += np.real(
                np.einsum("ij,ij->j", states[row].conj(), applied)
            )
            if piece.row != piece.column:
                applied = piece.matrix.conj().T @ states[row]
                derivatives[:, parameter] += np.real(
                    np.einsum("ij,ij->j", states[column].conj(), applied)
                )
    return tuple(operator.label for operator in block.operators), derivatives


def calpgm_coupled_dipole_matrix(
    lower: CALPGMCoupledBlock,
    upper: CALPGMCoupledBlock,
    dipole_debye: Sequence[float],
    vibrational_dipoles: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Return the reduced E1 matrix between two Wigner-coupled F blocks."""

    dipole = np.asarray(tuple(dipole_debye), dtype=float)
    if dipole.shape != (3,) or not np.all(np.isfinite(dipole)):
        raise ValueError("dipole_debye must contain three finite components")
    state_count = 1 + max(label["v"] for label in lower.basis_labels + upper.basis_labels)
    if vibrational_dipoles is None:
        vibration = (np.eye(state_count, dtype=complex),) * 3
    else:
        vibration = tuple(np.asarray(value, dtype=complex) for value in vibrational_dipoles)
        if len(vibration) != 3 or any(value.shape != (state_count, state_count) for value in vibration):
            raise ValueError("vibrational dipoles require three state_count square matrices")
    transition = np.zeros((lower.dimension, upper.dimension), dtype=complex)
    if abs(lower.F - upper.F) > 1 or lower.F == upper.F == 0:
        return transition
    # Diagonal vibrational moments are assembled directly in CALPGM's
    # GETQQ/Wang basis.  This is the electric-dipole analogue of DIRCOS in
    # SPINV and preserves the relative phases needed for intensity borrowing
    # between mixed states.  A raw signed-K transform is isospectral but does
    # not carry these discrete phase conventions.
    diagonal_vibration = tuple(np.diag(np.diag(value)) for value in vibration)
    transition += _calpgm_symmetry_dipole_matrix(
        lower, upper, dipole, diagonal_vibration
    )

    residual_vibration = tuple(value - diagonal for value, diagonal in zip(
        vibration, diagonal_vibration, strict=True
    ))
    if not any(np.any(np.abs(value) > 1.0e-15) for value in residual_vibration):
        return transition

    # General off-diagonal vibrational transition surfaces remain in the
    # explicit signed-K representation because they can encode arbitrary LAM
    # selection rules beyond those present in a CALPGM .int record.
    working_transition = np.zeros(
        (len(lower.working_basis_labels), len(upper.working_basis_labels)),
        dtype=complex,
    )
    for left in lower.groups:
        for right in upper.groups:
            if (
                left.two_I != right.two_I
                or left.nuclear_symmetry != right.nuclear_symmetry
                or abs(left.N - right.N) > 1
                or left.N == right.N == 0
            ):
                continue
            factor = _e1_recoupling_amplitude(
                left.N, right.N, lower.two_S, left.two_J, right.two_J,
                left.two_I, left.two_F, right.two_F,
            )
            if abs(factor) <= 1.0e-15:
                continue
            _left_labels, left_wang = calpgm_wang_transform(left.N)
            _right_labels, right_wang = calpgm_wang_transform(right.N)
            body = np.zeros((2 * left.N + 1, 2 * right.N + 1), dtype=complex)
            for axis, name in enumerate(("a", "b", "c")):
                vib = residual_vibration[axis][left.vibration, right.vibration]
                if dipole[axis] != 0.0 and vib != 0.0:
                    body += dipole[axis] * vib * rank1_body_block(
                        left.N,
                        right.N,
                        tuple(1.0 if component == axis else 0.0 for component in range(3)),
                        "Ir",
                    )
            rows = slice(left.start, left.stop)
            columns = slice(right.start, right.stop)
            working_transition[rows, columns] += factor * (
                left_wang.conj().T @ body @ right_wang
            )
    if lower.symmetry_transform is not None:
        working_transition = (
            lower.symmetry_transform.conj().T @ working_transition
        )
    if upper.symmetry_transform is not None:
        working_transition = working_transition @ upper.symmetry_transform
    return transition + working_transition


def _calpgm_symmetry_dipole_matrix(
    lower: CALPGMCoupledBlock,
    upper: CALPGMCoupledBlock,
    dipole: np.ndarray,
    vibration: Sequence[np.ndarray],
) -> np.ndarray:
    """Build diagonal-vibrational E1 moments with CALPGM DIRCOS phases."""

    result = np.zeros((lower.dimension, upper.dimension), dtype=complex)
    options = lower.symmetry_subblocks
    if not options or not upper.symmetry_subblocks:
        return result
    if lower.nuclear_symmetry_order != upper.nuclear_symmetry_order:
        raise ValueError("CALPGM dipole blocks use different nuclear symmetry orders")
    nitot = lower.nuclear_symmetry_order

    lower_starts: list[int] = []
    cursor = 0
    for block in lower.symmetry_subblocks:
        lower_starts.append(cursor)
        cursor += int(block["dimension"])
    upper_starts: list[int] = []
    cursor = 0
    for block in upper.symmetry_subblocks:
        upper_starts.append(cursor)
        cursor += int(block["dimension"])

    # CALPGM dipole symmetry after the prolate REVSYM conversion:
    # a -> 3 (delta K=0), b -> 2 and c -> 1 (delta K=1).
    axis_routes = (
        (3, 0),
        (2, 1),
        (1, 1),
    )
    for axis, (required_symmetry, delta_k) in enumerate(axis_routes):
        if dipole[axis] == 0.0:
            continue
        route = CALPGMDirectionalRoute(
            direction_rank=1,
            delta_k=delta_k,
            tensor_rank=1,
            delta_l=0,
            factor=1.0,
            block_symmetries=(required_symmetry,),
            identity_itot=True,
        )
        for left_index, left in enumerate(lower.symmetry_subblocks):
            left_group = _subblock_group(left, lower.two_S)
            coefficient = vibration[axis][left_group.vibration, left_group.vibration]
            if coefficient == 0.0:
                continue
            for right_index, right in enumerate(upper.symmetry_subblocks):
                right_group = _subblock_group(right, upper.two_S)
                block_symmetry = (
                    int(left["wang_symmetry_code"])
                    ^ int(right["wang_symmetry_code"])
                ) & 3
                if (
                    block_symmetry != required_symmetry
                    or left_group.vibration != right_group.vibration
                    or left_group.two_I != right_group.two_I
                    or left_group.nuclear_symmetry != right_group.nuclear_symmetry
                    or abs(left_group.N - right_group.N) > 1
                    or left_group.N == right_group.N == 0
                ):
                    continue
                spin_factor = _e1_recoupling_amplitude(
                    left_group.N,
                    right_group.N,
                    lower.two_S,
                    left_group.two_J,
                    right_group.two_J,
                    left_group.two_I,
                    left_group.two_F,
                    right_group.two_F,
                )
                if abs(spin_factor) <= 1.0e-15:
                    continue
                scalar = dipole[axis] * coefficient * spin_factor
                for row, column, value in directional_subblock_elements(
                    left,
                    right,
                    route,
                    block_symmetry=block_symmetry,
                    alpha=0,
                    nitot=nitot,
                    ifup=1,
                ):
                    result[lower_starts[left_index] + row, upper_starts[right_index] + column] += (
                        scalar * value
                    )
    return result


def _groups_from_labels(
    labels: Sequence[Mapping[str, Any]],
) -> tuple[CALPGMCoupledGroup, ...]:
    groups = []
    start = 0
    while start < len(labels):
        first = labels[start]
        key = (first["v"], first["nn"], first["N"])
        stop = start + 1
        while stop < len(labels):
            candidate = labels[stop]
            if (candidate["v"], candidate["nn"], candidate["N"]) != key:
                break
            stop += 1
        intermediates = first.get("coupled_angular_momenta", ())
        J = float(intermediates[0]) if intermediates else float(first["N"])
        I = float(first.get("I_tot", 0.0))
        groups.append(CALPGMCoupledGroup(
            vibration=int(first["v"]), nn=int(first["nn"]), N=int(first["N"]),
            two_J=int(round(2.0 * J)), two_I=int(round(2.0 * I)),
            two_F=int(round(2.0 * float(first["F"]))),
            nuclear_symmetry=int(first.get("nuclear_permutation_symmetry", 0)),
            start=start, stop=stop,
        ))
        start = stop
    return tuple(groups)


def _calpgm_symmetry_transform(
    working_labels: tuple[Mapping[str, Any], ...],
    groups: tuple[CALPGMCoupledGroup, ...],
    subblocks: tuple[Mapping[str, Any], ...],
) -> tuple[tuple[Mapping[str, Any], ...], np.ndarray]:
    """Return the working-Wang to GETQQ/GETMASK coupled-l transform."""

    working_dimension = len(working_labels)
    symmetry_dimension = sum(int(item["dimension"]) for item in subblocks)
    raw_from_working = np.zeros(
        (working_dimension, working_dimension), dtype=float
    )
    group_by_key = {}
    for group in groups:
        group_by_key[(group.vibration, group.nn, group.N)] = group
        _signed, wang = calpgm_wang_transform(group.N)
        raw_from_working[group.start:group.stop, group.start:group.stop] = wang
    label_by_key = {
        (int(label["v"]), int(label["nn"]), int(label["N"]), int(label["K"])): label
        for label in working_labels
    }
    raw_from_symmetry = np.zeros(
        (working_dimension, symmetry_dimension), dtype=float
    )
    symmetry_labels = []
    column = 0
    for subblock in subblocks:
        vibration = int(subblock["vibration"])
        nn = int(subblock["nn"])
        N = int(subblock["N"])
        symmetry = int(subblock["wang_symmetry_code"])
        l_value = int(subblock.get("l_value", 0))
        group = group_by_key[(vibration, nn, N)]
        for magnitude in subblock["k_values"]:
            magnitude = int(magnitude)
            signed_k = magnitude if symmetry in {0, 2} else -magnitude
            printed_vibration = vibration
            if magnitude == 0 and l_value < 0 and symmetry == 3:
                partner_vibration = vibration - 1
                candidates = [
                    key for key in group_by_key
                    if key == (partner_vibration, nn, N)
                    and int(working_labels[group_by_key[key].start].get("l_value", 0)) == -l_value
                ]
                if len(candidates) != 1:
                    raise ValueError("ambiguous CALPGM K=0 l-doublet partner")
                printed_vibration = candidates[0][0]
            label = dict(label_by_key[(printed_vibration, nn, N, signed_k)])
            label.update({
                "calpgm_block": int(subblock["block"]),
                "wang_symmetry": str(subblock["wang_symmetry"]),
                "wang_symmetry_code": symmetry,
                "l_value": l_value,
                "K_magnitude": magnitude,
                "internal_vibration": vibration,
                "statistical_weight": int(subblock.get("weight", 1)),
            })
            symmetry_labels.append(label)

            if l_value == 0:
                signed_labels, wang = calpgm_wang_transform(N)
                raw_from_symmetry[
                    group.start:group.stop, column
                ] = wang[:, signed_labels.index(signed_k)]
            else:
                partner_vibration = vibration + (1 if l_value > 0 else -1)
                partners = [
                    candidate for candidate in groups
                    if candidate.vibration == partner_vibration
                    and candidate.nn == nn and candidate.N == N
                    and int(working_labels[candidate.start].get("l_value", 0)) == -l_value
                ]
                if len(partners) != 1:
                    raise ValueError("ambiguous CALPGM signed-l partner")
                partner = partners[0]
                phase = -1.0 if ((N + (symmetry & 1)) & 1) else 1.0
                raw_from_symmetry[group.start + N + magnitude, column] = 1.0 / math.sqrt(2.0)
                raw_from_symmetry[partner.start + N - magnitude, column] = phase / math.sqrt(2.0)
            column += 1
    if column != symmetry_dimension:
        raise ValueError("CALPGM symmetry transform has the wrong dimension")
    transform = raw_from_working.T @ raw_from_symmetry
    error = float(np.max(np.abs(
        transform.T @ transform - np.eye(symmetry_dimension)
    )))
    if error > 1.0e-10:
        raise ValueError(f"CALPGM symmetry transform is not orthogonal: {error:.3e}")
    transform.setflags(write=False)
    return tuple(symmetry_labels), transform


def _parameter_keys(parameters: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    keys = []
    for index, parameter in enumerate(parameters):
        base = str(parameter.get("label") or abs(int(parameter["identifier"]))).strip()
        count = counts.get(base, 0)
        counts[base] = count + 1
        keys.append(base if count == 0 else f"{base}@{abs(int(parameter['identifier']))}:{index}")
    return tuple(keys)


def _calpgm_nuclear_symmetry_order(options: Sequence[Mapping[str, Any]]) -> int:
    if not options:
        return 0
    axis = abs(int(options[0].get("symmetry_axis", 0)))
    return {4: 2, 5: 2, 6: 3, 7: 4, 8: 4, 9: 5, 10: 6, 11: 6}.get(axis, 0)


def _subblock_group(label: Mapping[str, Any], two_S: int) -> CALPGMCoupledGroup:
    intermediates = tuple(label.get("intermediate_angular_momenta_minus_F", ()))
    J = float(label["F"]) + (float(intermediates[0]) if intermediates else 0.0)
    return CALPGMCoupledGroup(
        vibration=int(label["vibration"]),
        nn=int(label["nn"]),
        N=int(label["N"]),
        two_J=int(round(2.0 * J)),
        two_I=int(round(2.0 * float(label.get("I_tot", 0.0)))),
        two_F=int(round(2.0 * float(label["F"]))),
        nuclear_symmetry=int(label.get("nuclear_permutation_symmetry", 0)),
        start=0,
        stop=int(label["dimension"]),
    )


def _directional_operator_pieces(
    code: CALPGMOperatorCode,
    subblocks: tuple[Mapping[str, Any], ...],
    spin_twice: tuple[int, ...],
    l_values: tuple[int, ...],
    nitot: int,
    rho_values: tuple[float, ...],
    nofc: bool,
    g12_phase: bool,
) -> tuple[CALPGMCoupledPiece, ...] | None:
    two_S = spin_twice[0]
    route = directional_route(code, l_values, nitot)
    if route is None:
        return None
    starts: list[int] = []
    offsets: dict[int, int] = {}
    for subblock in subblocks:
        block_id = int(subblock["block"])
        start = offsets.get(block_id, 0)
        starts.append(start)
        offsets[block_id] = start + int(subblock["dimension"])
    block_offsets: dict[int, int] = {}
    cursor = 0
    for block_id in sorted(offsets):
        block_offsets[block_id] = cursor
        cursor += offsets[block_id]
    starts = [
        block_offsets[int(subblock["block"])] + local
        for subblock, local in zip(subblocks, starts, strict=True)
    ]
    explicit_spins = int(bool(code.spin_1)) + int(bool(code.spin_2))
    pieces: list[CALPGMCoupledPiece] = []
    for left_index, left in enumerate(subblocks):
        left_group = _subblock_group(left, two_S)
        for right_index in range(left_index + 1):
            right = subblocks[right_index]
            # The subblock ``vibration`` field distinguishes the signed-l
            # components used to build a symmetry-adapted state.  It is not
            # an independent vibrational-state selector here.  GETMASK already
            # applies the physical delta-l and symmetry selection rules; an
            # extra raw-index filter would remove valid diagonal matrix
            # elements of off-diagonal l-doublet operators.
            pair_route = route
            if int(left["block"]) != int(right["block"]):
                continue
            block_symmetry = (
                int(left["wang_symmetry_code"]) ^ int(right["wang_symmetry_code"])
            ) & 3
            if block_symmetry not in pair_route.block_symmetries:
                continue
            right_group = _subblock_group(right, two_S)
            if explicit_spins == 2 and abs(left_group.N - right_group.N) > pair_route.tensor_rank:
                continue
            if explicit_spins == 0:
                if (
                    abs(left_group.N - right_group.N) > pair_route.tensor_rank
                    or left_group.two_J != right_group.two_J
                    or left_group.two_I != right_group.two_I
                ):
                    continue
                spin_factor = 1.0
            elif explicit_spins == 1:
                spin_factor = electron_spin_tensor_factor(
                    left_group.N,
                    right_group.N,
                    left_group.two_J,
                    right_group.two_J,
                    left_group.two_I,
                    right_group.two_I,
                    two_S,
                )
            elif pair_route.tensor_rank == 0:
                try:
                    spin_factor = _electron_total_nuclear_scalar(
                        left_group, right_group, two_S
                    )
                except (RuntimeError, ValueError):
                    spin_factor = 0.0
            else:
                try:
                    spin_factor = _electron_total_nuclear_rank2(left_group, right_group, two_S)
                except (RuntimeError, ValueError):
                    spin_factor = 0.0
            if abs(spin_factor) <= 1.0e-15:
                if explicit_spins < 2:
                    continue
            if pair_route.commutator_n2 and left_group.N == right_group.N:
                continue
            scalar = pair_route.factor * rotational_reduced_factor(
                pair_route.direction_rank,
                pair_route.tensor_rank,
                left_group.N,
                right_group.N,
            )
            if pair_route.commutator_n2:
                scalar *= left_group.N
            if code.n_square_power:
                nsq = 0.5 * (
                    (left_group.N * (left_group.N + 1.0)) ** code.n_square_power
                    + (right_group.N * (right_group.N + 1.0)) ** code.n_square_power
                )
                scalar *= nsq
            if abs(scalar) <= 1.0e-15:
                continue
            matrix = np.zeros(
                (int(left["dimension"]), int(right["dimension"])), dtype=complex
            )
            alpha0 = (
                (pair_route.delta_l - pair_route.delta_k) % nitot
                if nitot >= 3
                else 0
            )
            alphas = (alpha0, alpha0 + nitot) if nitot >= 3 else (alpha0,)
            for alpha in alphas:
                alpha_spin_factor = spin_factor
                if explicit_spins == 2 and nitot == 3:
                    symmetry_i = (
                        (2 * int(left.get("nuclear_permutation_symmetry", 0))) >> 2
                    ) % nitot
                    symmetry_j = (
                        (2 * int(right.get("nuclear_permutation_symmetry", 0))) >> 2
                    ) % nitot
                    equivalent = equivalent_spin_tensor_factor(
                        left_group.N,
                        right_group.N,
                        left_group.two_J,
                        right_group.two_J,
                        left_group.two_I,
                        right_group.two_I,
                        left_group.two_F,
                        symmetry_i,
                        symmetry_j,
                        pair_route.tensor_rank,
                        alpha,
                        tuple(spin_twice[1:]),
                        int(left.get("nuclear_spin_multiplicity", 0)),
                        int(right.get("nuclear_spin_multiplicity", 0)),
                    )
                    if equivalent is not None:
                        alpha_spin_factor = equivalent
                if abs(alpha_spin_factor) <= 1.0e-15:
                    continue
                elements = directional_subblock_elements(
                    left,
                    right,
                    pair_route,
                    block_symmetry=block_symmetry,
                    alpha=alpha,
                    nitot=nitot,
                    ifup=left_index - right_index,
                )
                for row, column, value in elements:
                    if code.k_square_power:
                        kleft = int(left["k_start"]) + 2 * row
                        kright = int(right["k_start"]) + 2 * column
                        value *= (
                            0.5 * (
                                kleft ** (2 * code.k_square_power)
                                + kright ** (2 * code.k_square_power)
                            )
                        )
                    matrix[row, column] += scalar * value * alpha_spin_factor
            if not np.any(np.abs(matrix) > 1.0e-15):
                continue
            if code.fourier_flag:
                k_left = int(left["k_start"]) + 2 * np.arange(matrix.shape[0])
                k_right = int(right["k_start"]) + 2 * np.arange(matrix.shape[1])
                if nofc:
                    matrix = calpgm_selected_k_modulate(
                        code, matrix, k_left, k_right
                    )
                else:
                    matrix = calpgm_fourier_modulate(
                        code,
                        matrix,
                        k_left,
                        k_right,
                        rho_values[int(left["vibration"])],
                        rho_values[int(right["vibration"])],
                        g12_phase=g12_phase,
                    )
            if left_index == right_index:
                lower = np.tril(matrix)
                matrix = lower + lower.conj().T - np.diag(np.diag(lower))
            matrix.setflags(write=False)
            pieces.append(CALPGMCoupledPiece(starts[left_index], starts[right_index], matrix))
    return tuple(pieces)


def _accumulate_symmetry_pieces(
    target: np.ndarray,
    pieces: tuple[CALPGMCoupledPiece, ...],
    coefficient: float,
) -> None:
    for piece in pieces:
        rows, columns = piece.matrix.shape
        row = slice(piece.row, piece.row + rows)
        column = slice(piece.column, piece.column + columns)
        target[row, column] += coefficient * piece.matrix
        if piece.row != piece.column:
            target[column, row] += coefficient * piece.matrix.conj().T


def _operator_pieces(
    code: CALPGMOperatorCode,
    groups: tuple[CALPGMCoupledGroup, ...],
    two_S: int,
    state_count: int,
    aliases: tuple[int, ...],
    l_values: tuple[int, ...],
    rho_values: tuple[float, ...],
    nofc: bool,
    g12_phase: bool,
) -> tuple[CALPGMCoupledPiece, ...]:
    vibration = calpgm_vibrational_projection(code, state_count, aliases=aliases)
    if (
        code.vibrational_upper != code.vibrational_lower
        and l_values[code.vibrational_upper]
        and l_values[code.vibrational_upper] == -l_values[code.vibrational_lower]
    ):
        # SPINV normalizes an off-diagonal l-doublet tensor with one
        # 1/sqrt(2) reduced vibrational factor.  The generic projector uses
        # 1/2 because it represents a Hermitian state projector rather than
        # a reduced CALPGM tensor component.
        vibration = (2.0 ** 0.5) * vibration
    implicit_lz = (
        code.vibrational_upper == code.vibrational_lower
        and code.projection_type == 20
        and not code.fourier_flag
        and any(l_values)
    )
    if implicit_lz:
        canonical = aliases[code.vibrational_upper]
        vibration = np.diag([
            float(l_value) if aliases[index] == canonical else 0.0
            for index, l_value in enumerate(l_values)
        ]).astype(complex)
    pieces = []
    for left in groups:
        for right in groups:
            # CALPGM constructs one Hamiltonian triangle and reflects it.
            # Odd tensor ranks have direction-dependent Wigner phases, so
            # independently evaluating both directions and averaging them
            # can cancel a valid matrix element.
            if left.start > right.start:
                continue
            vib = complex(vibration[left.vibration, right.vibration])
            if abs(vib) <= 1.0e-15:
                continue
            if not _spin_symmetry_allowed(left, right, code):
                continue
            raw = _coupled_k_block(code, left, right, two_S)
            if raw is None or not np.any(np.abs(raw) > 1.0e-15):
                continue
            if code.fourier_flag:
                signed_k_left = np.arange(-left.N, left.N + 1)
                signed_k_right = np.arange(-right.N, right.N + 1)
                if nofc:
                    raw = calpgm_selected_k_modulate(
                        code, raw, signed_k_left, signed_k_right
                    )
                else:
                    raw = calpgm_fourier_modulate(
                        code,
                        raw,
                        signed_k_left,
                        signed_k_right,
                        rho_values[left.vibration],
                        rho_values[right.vibration],
                        g12_phase=g12_phase,
                    )
            if implicit_lz:
                # SPINV's MLZ branch carries an extra (-1)^N relative to the
                # ordinary odd-rank Wang convention.  For a non-spin Na term
                # this removes the phase already applied in
                # _coupled_k_block; for Na.S it supplies the missing phase.
                raw = ((-1.0) ** left.N) * raw
            _left_labels, left_wang = calpgm_wang_transform(left.N)
            _right_labels, right_wang = calpgm_wang_transform(right.N)
            matrix = vib * (left_wang.conj().T @ raw @ right_wang)
            if code.projection_rank & 1 and left.N != right.N and not implicit_lz:
                # DIRcos changes the negative signed-K ket phase for odd
                # tensor order between distinct N manifolds.  This phase is
                # separate from the Wang basis phase above and determines
                # whether the upper/lower parity partners are coupled.
                negative = np.asarray(_right_labels) < 0
                matrix[:, negative] *= -1.0
            if left.start != right.start:
                matrix = 2.0 * matrix
            matrix = np.asarray(matrix, dtype=complex)
            matrix.setflags(write=False)
            pieces.append(CALPGMCoupledPiece(left.start, right.start, matrix))
    return tuple(pieces)


def _spin_symmetry_allowed(
    left: CALPGMCoupledGroup, right: CALPGMCoupledGroup, code: CALPGMOperatorCode
) -> bool:
    if left.two_I != right.two_I:
        return False
    if left.nuclear_symmetry != right.nuclear_symmetry:
        return False
    if not (code.spin_1 or code.spin_2 or code.n_dot_s_power or code.n_z_s_z):
        return left.two_J == right.two_J and left.N == right.N
    return True


def _coupled_k_block(
    code: CALPGMOperatorCode,
    left: CALPGMCoupledGroup,
    right: CALPGMCoupledGroup,
    two_S: int,
) -> np.ndarray | None:
    explicit = tuple(value for value in (code.spin_1, code.spin_2) if value)
    if any(actor not in {1, 2} for actor in explicit):
        raise NotImplementedError(
            "the ITOT reduced basis supports spin 1 (electron) and spin 2 "
            "(total nuclear spin) identifiers"
        )
    spin_dependent = bool(explicit or code.n_dot_s_power or code.n_z_s_z)
    if not spin_dependent:
        if left.N != right.N or left.two_J != right.two_J:
            return None
        raw = calpgm_rotational_projection(
            code, angular_momentum_operators(left.N), prolate=True
        )
        if code.projection_rank & 1:
            raw = ((-1.0) ** left.N) * raw
        # calpgm_rotational_projection already applies KSQ and NSQ through
        # half anticommutators; applying them again changes the centrifugal
        # distortion order (for example DN from [N(N+1)]^2 to its square).
        return raw

    if code.n_dot_s_power:
        if left.N != right.N or left.two_J != right.two_J:
            return None
        shift = _isotropic_spin_rotation_shift(left.N, two_S, left.two_J, 1.0)
        raw = np.eye(2 * left.N + 1, dtype=complex) * (shift ** code.n_dot_s_power)
        return _apply_scalar_modifiers(raw, code, left.N, right.N)

    if code.n_z_s_z:
        tensor = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return _electron_spin_rotation_block(code, left, right, two_S, tensor)

    if len(explicit) == 1:
        actor = explicit[0]
        if code.projection_type in {20, 40, 60}:
            vector = _rank1_vector(code.projection_type)
            body = rank1_body_block(left.N, right.N, vector, "Ir")
            factor = _rank1_actor_factor(actor, left, right, two_S)
            return _apply_scalar_modifiers(body * factor, code, left.N, right.N)
        tensor = _rank2_tensor(code.projection_type)
        if tensor is None:
            return None
        if actor == 1:
            return _electron_spin_rotation_block(code, left, right, two_S, tensor)
        return _nuclear_spin_rotation_block(code, left, right, two_S, tensor)

    if len(explicit) == 2:
        if code.projection_type == 0:
            if left.N != right.N:
                return None
            factor = _electron_total_nuclear_scalar(left, right, two_S)
            raw = np.eye(2 * left.N + 1, dtype=complex) * factor
            return _apply_scalar_modifiers(raw, code, left.N, right.N)
        tensor = _rank2_tensor(code.projection_type)
        if tensor is None:
            return None
        body = rank2_body_block(left.N, right.N, tensor, "Ir")
        factor = _electron_total_nuclear_rank2(left, right, two_S)
        return _apply_scalar_modifiers(body * factor, code, left.N, right.N)
    return None


def _rank1_actor_factor(actor: int, left: CALPGMCoupledGroup, right: CALPGMCoupledGroup, two_S: int) -> complex:
    two_F = left.two_F
    if actor == 1:
        return _rank1_electron_spin_scalar_factor(
            left.N, right.N, two_S, left.two_J, right.two_J,
            (left.two_I,), (right.two_I,), two_F, (left.two_I,),
        )
    return _rank1_total_nuclear_factor(left, right, two_S)


def _rank1_total_nuclear_factor(left: CALPGMCoupledGroup, right: CALPGMCoupledGroup, two_S: int) -> complex:
    two_F = left.two_F
    lvec = _coupled_f_vector(left.N, two_S, left.two_J, (left.two_I,), two_F, (left.two_I,))
    rvec = _coupled_f_vector(right.N, two_S, right.two_J, (right.two_I,), two_F, (right.two_I,))
    total = 0.0j
    for p in (-1, 0, 1):
        phase = -1.0 if (p & 1) else 1.0
        rotation = _rank1_rotational_space_component(left.N, right.N, p)
        nuclear = _coupled_nuclear_rank1_component((left.two_I,), 0, -p)
        internal = np.kron(np.eye(two_S + 1), nuclear)
        transformed = rotation @ rvec.reshape((rvec.shape[0], -1)) @ internal.T
        total += phase * np.vdot(lvec.reshape((lvec.shape[0], -1)), transformed)
    return complex(np.real_if_close(total))


def _electron_spin_rotation_block(code, left, right, two_S, tensor):
    body = rank2_body_block(left.N, right.N, tensor, "Ir")
    factor = _anisotropic_spin_rotation_scalar_factor(
        left.N, right.N, two_S, left.two_J, right.two_J,
        (left.two_I,), (right.two_I,), left.two_F, (left.two_I,),
    )
    isotropic = sum(tensor[:3]) / 3.0
    raw = body * factor
    if isotropic and left.N == right.N and left.two_J == right.two_J:
        raw = raw + np.eye(2 * left.N + 1) * isotropic * _isotropic_spin_rotation_shift(
            left.N, two_S, left.two_J, 1.0
        )
    return _apply_scalar_modifiers(raw, code, left.N, right.N)


def _nuclear_spin_rotation_block(code, left, right, two_S, tensor):
    # Rank-2 N-I is evaluated explicitly in the same fixed-F projection used
    # by the electron spin-rotation implementation.
    body = rank2_body_block(left.N, right.N, tensor, "Ir")
    factor = _anisotropic_nuclear_rotation_factor(left, right, two_S)
    isotropic = sum(tensor[:3]) / 3.0
    raw = body * factor
    if isotropic and left.N == right.N:
        raw = raw + np.eye(2 * left.N + 1) * isotropic * _isotropic_nuclear_scalar_factor(
            left.N, two_S, left.two_J, right.two_J,
            (left.two_I,), (right.two_I,), left.two_F,
            (left.two_I,), 0, "rotation",
        )
    return _apply_scalar_modifiers(raw, code, left.N, right.N)


def _anisotropic_nuclear_rotation_factor(left, right, two_S):
    two_F = left.two_F
    lvec = _coupled_f_vector(left.N, two_S, left.two_J, (left.two_I,), two_F, (left.two_I,))
    rvec = _coupled_f_vector(right.N, two_S, right.two_J, (right.two_I,), two_F, (right.two_I,))
    total = 0.0j
    for p in range(-2, 3):
        D = _rank2_rotational_space_component(left.N, right.N, p)
        phase = -1.0 if (p & 1) else 1.0
        for qn in (-1, 0, 1):
            qi = -p - qn
            if qi not in {-1, 0, 1}:
                continue
            cg = clebsch_gordan_doubled(2, 2 * qn, 2, 2 * qi, 4, -2 * p)
            if cg == 0.0:
                continue
            nl = _angular_momentum_rank1_component(2 * left.N, qn)
            nr = _angular_momentum_rank1_component(2 * right.N, qn)
            ordered = 0.5 * (nl @ D + D @ nr)
            nuclear = _coupled_nuclear_rank1_component((left.two_I,), 0, qi)
            internal = np.kron(np.eye(two_S + 1), nuclear)
            transformed = ordered @ rvec.reshape((rvec.shape[0], -1)) @ internal.T
            total += phase * cg * np.vdot(lvec.reshape((lvec.shape[0], -1)), transformed)
    return math.sqrt(2.0 / 3.0) * total


def _electron_total_nuclear_scalar(left, right, two_S):
    if left.N != right.N:
        return 0.0
    value = _isotropic_nuclear_scalar_factor(
        left.N, two_S, left.two_J, right.two_J,
        (left.two_I,), (right.two_I,), left.two_F,
        (left.two_I,), 0, "electron",
    )
    # CALPGM's scalar two-spin convention is S.(S+I_tot).  The S.S part is
    # diagonal in the coupled chain and must be retained even when I_tot=0.
    if left.two_J == right.two_J and left.two_I == right.two_I:
        spin = 0.5 * two_S
        value += spin * (spin + 1.0)
    return value


def _electron_total_nuclear_rank2(left, right, two_S):
    return _rank2_spin_scalar_factor(
        left.N, right.N, two_S, left.two_J, right.two_J,
        (left.two_I,), (right.two_I,), left.two_F,
        (left.two_I,), "electron-nuclear", (0,),
    )


def _apply_scalar_modifiers(raw, code, N, Np):
    result = np.asarray(raw, dtype=complex)
    if code.k_square_power:
        left_k = np.arange(-N, N + 1, dtype=float) ** (2 * code.k_square_power)
        right_k = np.arange(-Np, Np + 1, dtype=float) ** (2 * code.k_square_power)
        result = 0.5 * (left_k[:, None] + right_k[None, :]) * result
    if code.n_square_power:
        left = (N * (N + 1.0)) ** code.n_square_power
        right = (Np * (Np + 1.0)) ** code.n_square_power
        result = 0.5 * (left + right) * result
    return result


def _rank1_vector(projection_type: int) -> tuple[float, float, float]:
    return {20: (1.0, 0.0, 0.0), 40: (0.0, 1.0, 0.0), 60: (0.0, 0.0, 1.0)}[projection_type]


def _rank2_tensor(projection_type: int):
    return {
        1: (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        3: (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        # Prolate CALPGM uses x=c and y=b.  TYP=4 is
        # N_+^2+N_-^2=2(Nc^2-Nb^2); off-diagonal Cartesian entries are the
        # coefficients of the complete symmetrized products.
        4: (0.0, -2.0, 2.0, 0.0, 0.0, 0.0),
        21: (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        41: (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        61: (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    }.get(projection_type)


__all__ = [
    "CALPGMCoupledBlock",
    "CALPGMCoupledGroup",
    "CALPGMCoupledOperator",
    "CALPGMCoupledPiece",
    "apply_calpgm_coupled_block",
    "assemble_calpgm_coupled_block",
    "assemble_calpgm_coupled_sparse",
    "calpgm_coupled_block_diagonal",
    "calpgm_coupled_dipole_matrix",
    "calpgm_coupled_operator_expectations",
    "compile_calpgm_coupled_block",
]
