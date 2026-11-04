"""Native, factorized CALPGM block solving, fitting and prediction for LOCK."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import json
import math
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .adaptive_solver import (
    AdaptiveSolverPolicy,
    EigenproblemBlock,
    EigensolverWorkspace,
    solve_eigenblocks,
    track_eigenstates_by_subspace,
    read_eigensolver_workspace,
    write_eigensolver_workspace,
)
from .calpgm_native import (
    CALPGMOperatorCode,
    calpgm_fourier_modulate,
    calpgm_fourier_options,
    calpgm_fourier_rho_values,
    calpgm_selected_k_modulate,
    calpgm_spin_rotational_projection,
    calpgm_spin_twice_from_model,
    calpgm_vibrational_aliases,
    calpgm_vibrational_projection,
    decode_calpgm_identifier,
)
from .calpgm_coupled import (
    CALPGMCoupledBlock,
    apply_calpgm_coupled_block,
    assemble_calpgm_coupled_block,
    assemble_calpgm_coupled_sparse,
    calpgm_coupled_block_diagonal,
    calpgm_coupled_dipole_matrix,
    calpgm_coupled_operator_expectations,
    compile_calpgm_coupled_block,
)
from .calpgm_adapter import infer_calpgm_scalar_aggregate_map
from .native_rotor import MHZ_TO_KELVIN, angular_momentum_operators, dipole_transition_matrix
from .operator_engine import (
    CompiledOperatorModel,
    FactorizedOperatorTerm,
    OperatorSpace,
    apply_operator,
    assemble_operator,
    hellmann_feynman_factorized_jacobian,
)
from .qm_seeded import validate_qm_seed


CALPGM_NATIVE_RESULT_SCHEMA = "matrix.lock.calpgm_native_result.v1"
CALPGM_NATIVE_CATALOG_SCHEMA = "matrix.lock.calpgm_native_catalog.v1"
CALPGM_NATIVE_FIT_SCHEMA = "matrix.lock.calpgm_native_fit.v1"
CALPGM_NATIVE_FIT_RESULT_SCHEMA = "matrix.lock.calpgm_native_fit_result.v1"


@dataclass(frozen=True)
class CALPGMCompiledBlock:
    J: int
    state_count: int
    spin_twice: tuple[int, ...]
    operator_model: CompiledOperatorModel | None
    coupled_block: CALPGMCoupledBlock | None
    codes: tuple[CALPGMOperatorCode, ...]
    parameter_keys: tuple[str, ...]
    parameter_identifiers: tuple[int, ...]
    ratio_constrained: tuple[bool, ...]
    prolate: bool
    n_max: int | None = None

    @property
    def dimension(self) -> int:
        if self.coupled_block is not None:
            return self.coupled_block.dimension
        if self.operator_model is None:
            raise RuntimeError("CALPGM block has no compiled assembler")
        return self.operator_model.dimension

    @property
    def block_quantum_number(self) -> str:
        return "F" if self.coupled_block is not None and any(self.spin_twice) else "J"


@dataclass(frozen=True)
class CALPGMSolvedBlock:
    J: int
    energies_MHz: np.ndarray
    vectors: np.ndarray
    residual_norms_MHz: np.ndarray
    backend: str
    reused_subspace: bool
    dimension: int
    block_quantum_number: str = "J"


@dataclass(frozen=True)
class CALPGMNativeResult:
    model: Mapping[str, Any]
    compiled_blocks: Mapping[int, CALPGMCompiledBlock]
    blocks: tuple[CALPGMSolvedBlock, ...]
    matrix_free_threshold: int
    factorized_backend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CALPGM_NATIVE_RESULT_SCHEMA,
            "model_schema": self.model.get("schema"),
            "matrix_free_threshold": self.matrix_free_threshold,
            "factorized_backend": self.factorized_backend,
            "block_quantum_number": (
                self.blocks[0].block_quantum_number if self.blocks else "J"
            ),
            "n_max": (
                next(iter(self.compiled_blocks.values())).n_max
                if self.compiled_blocks else None
            ),
            "spin_twice": list(calpgm_spin_twice_from_model(self.model)),
            "parameter_count": len(self.model.get("parameters", ())),
            "blocks": [
                {
                    block.block_quantum_number: block.J,
                    "basis": (
                        "coupled-F-Wigner"
                        if block.block_quantum_number == "F"
                        else "spin-free-rotational"
                    ),
                    "dimension": block.dimension,
                    "working_dimension": (
                        len(self.compiled_blocks[block.J].coupled_block.working_basis_labels)
                        if self.compiled_blocks[block.J].coupled_block is not None
                        else block.dimension
                    ),
                    "retained_levels": len(block.energies_MHz),
                    "energies_MHz": [float(value) for value in block.energies_MHz],
                    "maximum_residual_MHz": float(
                        np.max(block.residual_norms_MHz, initial=0.0)
                    ),
                    "backend": block.backend,
                    "reused_subspace": block.reused_subspace,
                }
                for block in self.blocks
            ],
        }


def _parameter_keys(parameters: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    result = []
    for index, parameter in enumerate(parameters):
        base = str(parameter.get("label") or abs(int(parameter["identifier"]))).strip()
        count = counts.get(base, 0)
        counts[base] = count + 1
        result.append(base if count == 0 else f"{base}@{abs(int(parameter['identifier']))}:{index}")
    return tuple(result)


def compile_calpgm_block(
    model: Mapping[str, Any],
    J: int,
    *,
    aggregate_map: Mapping[str, Any] | None = None,
    prolate: bool = True,
    n_max: int | None = None,
) -> CALPGMCompiledBlock:
    """Compile one block, using the reduced Wigner-coupled F basis for spin."""

    if isinstance(J, bool) or int(J) != J or J < 0:
        raise ValueError("CALPGM J must be a nonnegative integer")
    options = model.get("options", ())
    parameters = tuple(model.get("parameters", ()))
    if not options or not parameters:
        raise ValueError("CALPGM compilation requires options and parameters")
    state_count = max(1, abs(int(options[0]["nvib"])))
    spin_twice = calpgm_spin_twice_from_model(model)
    if spin_twice:
        if not prolate:
            raise NotImplementedError(
                "the reduced coupled-F CALPGM assembler currently uses the prolate axis convention"
            )
        if aggregate_map is None and any(spin_twice):
            raise ValueError(
                "spinful CALPGM compilation requires an aggregate map; "
                "the block index is total F"
            )
        if aggregate_map is None:
            aggregate_map = infer_calpgm_scalar_aggregate_map(model)
        coupled = compile_calpgm_coupled_block(
            model, aggregate_map, int(J), n_max=n_max
        )
        keys = _parameter_keys(parameters)
        return CALPGMCompiledBlock(
            J=int(J),
            state_count=state_count,
            spin_twice=spin_twice,
            operator_model=None,
            coupled_block=coupled,
            codes=tuple(
                decode_calpgm_identifier(int(parameter["identifier"]), state_count=state_count)
                for parameter in parameters
            ),
            parameter_keys=keys,
            parameter_identifiers=tuple(int(parameter["identifier"]) for parameter in parameters),
            ratio_constrained=tuple(bool(parameter.get("ratio_constrained", False)) for parameter in parameters),
            prolate=bool(prolate),
            n_max=None if n_max is None else int(n_max),
        )
    spin_dimension = math.prod(value + 1 for value in spin_twice)
    momenta = angular_momentum_operators(int(J))
    aliases = calpgm_vibrational_aliases(model.get("options", ()), state_count)
    rho_values = calpgm_fourier_rho_values(model)
    nofc, g12_phase = calpgm_fourier_options(model)
    keys = _parameter_keys(parameters)
    terms = []
    codes = []
    identifiers = []
    constrained = []
    defaults = {}
    for key, parameter in zip(keys, parameters, strict=True):
        code = decode_calpgm_identifier(int(parameter["identifier"]), state_count=state_count)
        if code.operator_family == "series-constant":
            vibration = np.zeros((state_count, state_count), dtype=complex)
            rotation_spin = np.zeros((2 * int(J) + 1, 2 * int(J) + 1), dtype=complex)
        else:
            vibration = calpgm_vibrational_projection(code, state_count, aliases=aliases)
            rotation_spin = calpgm_spin_rotational_projection(
                code, momenta, spin_twice, prolate=prolate
            )
        if code.fourier_flag:
            k_labels = np.arange(-int(J), int(J) + 1)
            for upper, lower in np.argwhere(np.abs(vibration) > 1.0e-15):
                selector = np.zeros_like(vibration)
                selector[upper, lower] = vibration[upper, lower]
                if nofc:
                    modulated = calpgm_selected_k_modulate(
                        code, rotation_spin, k_labels, k_labels
                    )
                else:
                    modulated = calpgm_fourier_modulate(
                        code,
                        rotation_spin,
                        k_labels,
                        k_labels,
                        rho_values[upper],
                        rho_values[lower],
                        g12_phase=g12_phase,
                    )
                terms.append(FactorizedOperatorTerm(
                    label=f"{key}:v{upper},{lower}",
                    factors=(("vibration", selector), ("rotation_spin", modulated)),
                    parameter=key,
                ))
        else:
            terms.append(FactorizedOperatorTerm(
                label=key,
                factors=(("vibration", vibration), ("rotation_spin", rotation_spin)),
                parameter=key,
            ))
        codes.append(code)
        identifiers.append(int(parameter["identifier"]))
        constrained.append(bool(parameter.get("ratio_constrained", False)))
        defaults[key] = float(parameter["value"])
    operator_model = CompiledOperatorModel(
        spaces=(
            OperatorSpace("vibration", state_count),
            OperatorSpace("rotation_spin", (2 * int(J) + 1) * spin_dimension),
        ),
        terms=tuple(terms),
        parameter_defaults=defaults,
    )
    return CALPGMCompiledBlock(
        J=int(J),
        state_count=state_count,
        spin_twice=spin_twice,
        operator_model=operator_model,
        coupled_block=None,
        codes=tuple(codes),
        parameter_keys=keys,
        parameter_identifiers=tuple(identifiers),
        ratio_constrained=tuple(constrained),
        prolate=bool(prolate),
        n_max=None,
    )


def calpgm_block_diagonal(
    block: CALPGMCompiledBlock,
    parameters: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Return the exact diagonal of a factorized CALPGM Hamiltonian."""

    if block.coupled_block is not None:
        return calpgm_coupled_block_diagonal(block.coupled_block, parameters)
    if block.operator_model is None:
        raise RuntimeError("CALPGM block has no compiled assembler")
    values = dict(block.operator_model.parameter_defaults)
    if parameters:
        unknown = set(parameters) - set(values)
        if unknown:
            raise ValueError(f"unknown CALPGM parameters: {sorted(unknown)}")
        values.update({name: float(value) for name, value in parameters.items()})
    diagonal = np.zeros(block.dimension, dtype=complex)
    dimensions = {space.name: space.dimension for space in block.operator_model.spaces}
    for term in block.operator_model.terms:
        factors = dict(term.factors)
        value = np.ones(1, dtype=complex)
        for space in block.operator_model.spaces:
            factor = factors.get(space.name, np.eye(dimensions[space.name], dtype=complex))
            value = np.kron(value, np.diag(factor))
        diagonal += term.coefficient * values[term.parameter] * value
    return diagonal.real


def solve_calpgm_native(
    model: Mapping[str, Any],
    *,
    J_values: Sequence[int] | range | None = None,
    F_values: Sequence[int] | range | None = None,
    aggregate_map: Mapping[str, Any] | None = None,
    levels_per_block: int | Mapping[int, int] | None = None,
    parameters: Mapping[str, float] | None = None,
    workers: int = 1,
    matrix_free_threshold: int = 1024,
    factorized_backend: str = "auto",
    workspace: EigensolverWorkspace | None = None,
    policy: AdaptiveSolverPolicy | None = None,
    prolate: bool = True,
    compiled_blocks: Mapping[int, CALPGMCompiledBlock] | None = None,
    blas_threads: int | None = None,
    n_max: int | None = None,
) -> CALPGMNativeResult:
    """Solve independent CALPGM blocks; spinful models use total-F blocks."""

    spinful = any(calpgm_spin_twice_from_model(model))
    if J_values is not None and F_values is not None:
        raise ValueError("specify either J_values or F_values, not both")
    selected = F_values if F_values is not None else J_values
    if selected is None:
        raise ValueError("native CALPGM solving requires block quantum numbers")
    if spinful and F_values is None:
        raise ValueError("spinful CALPGM solving requires F_values")
    if not spinful and F_values is not None:
        raise ValueError("spin-free CALPGM solving requires J_values")
    requested = tuple(dict.fromkeys(int(value) for value in selected))
    if not requested or any(value < 0 for value in requested):
        raise ValueError("native CALPGM solving requires nonnegative J blocks")
    if matrix_free_threshold < 1:
        raise ValueError("matrix_free_threshold must be positive")
    if n_max is not None and (
        isinstance(n_max, bool) or int(n_max) != n_max or n_max < 0
    ):
        raise ValueError("n_max must be a nonnegative integer")
    if blas_threads is not None and blas_threads < 1:
        raise ValueError("blas_threads must be positive")
    if compiled_blocks is None:
        compiled = {
            value: compile_calpgm_block(
                model, value, aggregate_map=aggregate_map, prolate=prolate,
                n_max=n_max,
            )
            for value in requested
        }
    else:
        missing = set(requested) - set(compiled_blocks)
        if missing:
            raise ValueError(f"compiled CALPGM cache lacks blocks: {sorted(missing)}")
        compiled = {value: compiled_blocks[value] for value in requested}
        expected_kind = "F" if spinful else "J"
        if any(block.block_quantum_number != expected_kind for block in compiled.values()):
            raise ValueError("compiled CALPGM cache has the wrong block kind")
        if spinful and any(block.n_max != n_max for block in compiled.values()):
            raise ValueError("compiled CALPGM cache was built with a different n_max")
    problems = []
    problem_metadata: list[tuple[int, np.ndarray]] = []
    for J in requested:
        block = compiled[J]
        roots = (
            block.dimension
            if levels_per_block is None
            else int(levels_per_block[J])
            if isinstance(levels_per_block, Mapping)
            else int(levels_per_block)
        )
        roots = min(block.dimension, roots)
        dense = block.dimension <= matrix_free_threshold or roots == block.dimension
        full_matrix = (
            (
                assemble_calpgm_coupled_block(block.coupled_block, parameters)
                if block.coupled_block is not None
                else assemble_operator(block.operator_model, parameters)
            )
            if dense
            else None
        )
        sparse_matrix = (
            assemble_calpgm_coupled_sparse(block.coupled_block, parameters)
            if not dense and block.coupled_block is not None
            else None
        )
        if block.coupled_block is not None and block.coupled_block.symmetry_block_ids:
            ids = np.asarray(block.coupled_block.symmetry_block_ids, dtype=int)
            sectors = tuple(
                (value, np.flatnonzero(ids == value))
                for value in sorted(set(ids.tolist()))
            )
        else:
            sectors = ((None, np.arange(block.dimension, dtype=int)),)
        full_diagonal = (
            np.real(np.diag(full_matrix))
            if full_matrix is not None
            else np.real(sparse_matrix.diagonal())
            if sparse_matrix is not None
            else calpgm_block_diagonal(block, parameters)
        )
        for sector_number, (sector_id, indices) in enumerate(sectors):
            sector_roots = min(len(indices), roots)
            sector_matrix = (
                None if full_matrix is None else full_matrix[np.ix_(indices, indices)]
            )
            sector_sparse = (
                None
                if sparse_matrix is None
                else sparse_matrix[indices, :][:, indices].tocsr()
            )

            def sector_matvec(
                vector: np.ndarray,
                *,
                current: CALPGMCompiledBlock = block,
                selected_indices: np.ndarray = indices,
                selected_sector: int | None = sector_id,
                sparse_operator=sector_sparse,
            ) -> np.ndarray:
                if sparse_operator is not None:
                    return np.asarray(sparse_operator @ vector).reshape(-1)
                embedded = np.zeros(current.dimension, dtype=complex)
                embedded[selected_indices] = vector
                if current.coupled_block is not None:
                    applied = apply_calpgm_coupled_block(
                        current.coupled_block,
                        embedded,
                        parameters,
                        symmetry_block_id=selected_sector,
                    )
                else:
                    applied = apply_operator(
                        current.operator_model,
                        embedded,
                        parameters,
                        backend=factorized_backend,
                    )
                return applied[selected_indices]

            problems.append(EigenproblemBlock(
                key=("calpgm", block.block_quantum_number, J, sector_number),
                dimension=len(indices),
                roots=sector_roots,
                matrix=sector_matrix,
                matvec=None if sector_matrix is not None else sector_matvec,
                diagonal=full_diagonal[indices],
            ))
            problem_metadata.append((J, indices))
    selected_policy = policy or AdaptiveSolverPolicy(
        dense_max_dimension=matrix_free_threshold
    )
    thread_limit = blas_threads if blas_threads is not None else (1 if workers > 1 else None)
    try:
        from threadpoolctl import threadpool_limits
        thread_context = (
            threadpool_limits(limits=thread_limit, user_api="blas")
            if thread_limit is not None else nullcontext()
        )
    except ImportError:
        thread_context = nullcontext()
    with thread_context:
        solved = solve_eigenblocks(
            problems,
            policy=selected_policy,
            workspace=workspace,
            workers=workers,
        )
    grouped: dict[int, list[tuple[np.ndarray, Any]]] = {J: [] for J in requested}
    for metadata, result in zip(problem_metadata, solved, strict=True):
        J, indices = metadata
        grouped[J].append((indices, result))
    merged_blocks = []
    for J in requested:
        dimension = compiled[J].dimension
        values = []
        vectors = []
        residuals = []
        backends = []
        reused = True
        for indices, result in grouped[J]:
            embedded = np.zeros((dimension, result.eigenvectors.shape[1]), dtype=complex)
            embedded[indices, :] = result.eigenvectors
            values.append(np.asarray(result.eigenvalues, dtype=float))
            vectors.append(embedded)
            residuals.append(np.asarray(result.residual_norms, dtype=float))
            backends.append(result.backend)
            reused = reused and result.reused_subspace
        all_values = np.concatenate(values)
        all_vectors = np.column_stack(vectors)
        all_residuals = np.concatenate(residuals)
        keep = (
            dimension
            if levels_per_block is None
            else int(levels_per_block[J])
            if isinstance(levels_per_block, Mapping)
            else int(levels_per_block)
        )
        order = np.argsort(all_values, kind="stable")[:min(dimension, keep)]
        merged_blocks.append(CALPGMSolvedBlock(
            J=J,
            energies_MHz=all_values[order],
            vectors=all_vectors[:, order],
            residual_norms_MHz=all_residuals[order],
            backend="symmetry[" + ",".join(dict.fromkeys(backends)) + "]",
            reused_subspace=reused,
            dimension=dimension,
            block_quantum_number=compiled[J].block_quantum_number,
        ))
    blocks = tuple(merged_blocks)
    return CALPGMNativeResult(
        model=model,
        compiled_blocks=compiled,
        blocks=blocks,
        matrix_free_threshold=matrix_free_threshold,
        factorized_backend=factorized_backend,
    )


def calpgm_level_jacobians(
    result: CALPGMNativeResult,
    *,
    backend: str = "auto",
    workers: int = 1,
    parameter_combinations: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[int, tuple[tuple[str, ...], np.ndarray]]:
    """Return exact Hellmann--Feynman derivatives for every retained level."""

    if isinstance(workers, bool) or int(workers) != workers or workers < 1:
        raise ValueError("CALPGM Jacobian workers must be a positive integer")

    def evaluate(solved):
        compiled = result.compiled_blocks[solved.J]
        if compiled.coupled_block is not None:
            native_names, native_jacobian = calpgm_coupled_operator_expectations(
                compiled.coupled_block, solved.vectors
            )
            if parameter_combinations is None:
                value = native_names, native_jacobian
            else:
                native_index = {
                    name: column for column, name in enumerate(native_names)
                }
                names = tuple(parameter_combinations)
                reduced = np.zeros(
                    (native_jacobian.shape[0], len(names)), dtype=float
                )
                for column, name in enumerate(names):
                    for parameter, weight in parameter_combinations[name].items():
                        if parameter in native_index:
                            reduced[:, column] += (
                                float(weight)
                                * native_jacobian[:, native_index[parameter]]
                            )
                value = names, reduced
        else:
            value = hellmann_feynman_factorized_jacobian(
                compiled.operator_model,
                solved.vectors,
                backend=backend,
                parameter_combinations=parameter_combinations,
            )
        return solved.J, value

    if workers == 1 or len(result.blocks) < 2:
        rows = map(evaluate, result.blocks)
    else:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=int(workers))
        try:
            rows = tuple(executor.map(evaluate, result.blocks))
        finally:
            executor.shutdown(wait=True)
    return dict(rows)


def track_calpgm_solution(
    reference: CALPGMNativeResult,
    current: CALPGMNativeResult,
    *,
    degeneracy_tolerance: float = 1.0e-8,
) -> tuple[CALPGMNativeResult, dict[int, Any]]:
    """Reorder current blocks by overlap, retaining subspace diagnostics."""

    reference_blocks = {block.J: block for block in reference.blocks}
    diagnostics = {}
    reordered = []
    for block in current.blocks:
        left = reference_blocks.get(block.J)
        if left is None:
            raise ValueError(f"reference CALPGM solution lacks J={block.J}")
        mapping = track_eigenstates_by_subspace(
            left.energies_MHz,
            left.vectors,
            block.energies_MHz,
            block.vectors,
            degeneracy_tolerance=degeneracy_tolerance,
        )
        permutation = np.asarray(mapping.permutation, dtype=int)
        reordered.append(
            CALPGMSolvedBlock(
                J=block.J,
                energies_MHz=block.energies_MHz[permutation],
                vectors=block.vectors[:, permutation],
                residual_norms_MHz=block.residual_norms_MHz[permutation],
                backend=block.backend,
                reused_subspace=block.reused_subspace,
                dimension=block.dimension,
                block_quantum_number=block.block_quantum_number,
            )
        )
        diagnostics[block.J] = mapping
    return (
        CALPGMNativeResult(
            model=current.model,
            compiled_blocks=current.compiled_blocks,
            blocks=tuple(reordered),
            matrix_free_threshold=current.matrix_free_threshold,
            factorized_backend=current.factorized_backend,
        ),
        diagnostics,
    )


def calpgm_transition_catalog(
    result: CALPGMNativeResult,
    *,
    dipole_debye: Sequence[float],
    temperature_K: float,
    intensity_cut: float = 0.0,
    vibrational_dipoles: Sequence[np.ndarray] | None = None,
    partition_function: float | None = None,
) -> list[dict[str, Any]]:
    """Generate hyperfine-resolved CALPGM strengths and SPCAT intensities.

    ``intensity`` is the integrated absorption intensity in nm2 MHz used by
    the JPL/CDMS catalog convention.  A supplied partition function permits
    direct comparison with an external SPCAT ``.int`` calculation; otherwise
    it is evaluated from the solved blocks.
    """

    dipole = np.asarray(dipole_debye, dtype=float)
    if dipole.shape != (3,) or temperature_K <= 0.0:
        raise ValueError("three dipoles and a positive temperature are required")
    coupled = any(block.coupled_block is not None for block in result.compiled_blocks.values())
    state_count = next(iter(result.compiled_blocks.values())).state_count
    if vibrational_dipoles is None:
        vibration = (np.eye(state_count, dtype=complex),) * 3
    else:
        vibration = tuple(np.asarray(value, dtype=complex) for value in vibrational_dipoles)
        if len(vibration) != 3 or any(value.shape != (state_count, state_count) for value in vibration):
            raise ValueError("vibrational dipoles require three state_count square matrices")
    blocks = {block.J: block for block in result.blocks}
    ground = min(float(np.min(block.energies_MHz)) for block in blocks.values())
    beta = MHZ_TO_KELVIN / temperature_K
    state_weights: dict[tuple[int, int], float] = {}
    for quantum_number, solved in blocks.items():
        compiled = result.compiled_blocks[quantum_number]
        if coupled and compiled.coupled_block is not None:
            weights = np.asarray([
                float(label.get("statistical_weight", 1.0))
                for label in compiled.coupled_block.basis_labels
            ])
            mixed = np.real(np.sum(np.abs(solved.vectors) ** 2 * weights[:, None], axis=0))
            for index, weight in enumerate(mixed):
                state_weights[(quantum_number, index)] = float(weight)
        else:
            for index in range(len(solved.energies_MHz)):
                state_weights[(quantum_number, index)] = 1.0
    calculated_partition = math.fsum(
        state_weights[(block.J, index)]
        * (2 * block.J + 1)
        * math.exp(-(float(energy) - ground) * beta)
        for block in blocks.values()
        for index, energy in enumerate(block.energies_MHz)
    )
    if partition_function is None:
        partition = calculated_partition
    else:
        partition = float(partition_function)
        if not math.isfinite(partition) or partition <= 0.0:
            raise ValueError("partition_function must be positive and finite")
    lines = []
    for lower_J, lower in blocks.items():
        for upper_J, upper in blocks.items():
            if upper_J < lower_J or abs(upper_J - lower_J) > 1:
                continue
            if coupled:
                lower_compiled = result.compiled_blocks[lower_J].coupled_block
                upper_compiled = result.compiled_blocks[upper_J].coupled_block
                primitive = calpgm_coupled_dipole_matrix(
                    lower_compiled, upper_compiled, dipole, vibration
                )
                amplitudes = lower.vectors.conj().T @ primitive @ upper.vectors
            else:
                spin_dimension = math.prod(
                    value + 1 for value in result.compiled_blocks[lower_J].spin_twice
                )
                spin_identity = np.eye(spin_dimension, dtype=complex)
                amplitudes = np.zeros(
                    (len(lower.energies_MHz), len(upper.energies_MHz)), dtype=complex
                )
                for axis, name in enumerate(("a", "b", "c")):
                    if dipole[axis] == 0.0:
                        continue
                    rotation = dipole_transition_matrix(lower_J, upper_J, name)
                    primitive = np.kron(vibration[axis], np.kron(rotation, spin_identity))
                    amplitudes += dipole[axis] * (lower.vectors.conj().T @ primitive @ upper.vectors)
            for lower_index, lower_energy in enumerate(lower.energies_MHz):
                for upper_index, upper_energy in enumerate(upper.energies_MHz):
                    frequency = float(upper_energy - lower_energy)
                    strength = float(abs(amplitudes[lower_index, upper_index]) ** 2)
                    if frequency <= 0.0 or strength <= 1.0e-24:
                        continue
                    degeneracy = (
                        state_weights[(lower_J, lower_index)]
                        if coupled else (2 * lower_J + 1)
                    )
                    population = degeneracy * math.exp(
                        -(float(lower_energy) - ground) * beta
                    ) / partition
                    intensity = (
                        4.16231e-5
                        * frequency
                        * strength
                        * population
                        * (-math.expm1(-frequency * beta))
                    )
                    if intensity < intensity_cut:
                        continue
                    lines.append(
                        {
                            "lower_J": lower_J,
                            "lower_level": lower_index,
                            "upper_J": upper_J,
                            "upper_level": upper_index,
                            "frequency_MHz": frequency,
                            "line_strength_D2": strength,
                            "intensity": intensity,
                            "log10_intensity_nm2_MHz": math.log10(intensity),
                            "partition_function": partition,
                            "calculated_partition_function": calculated_partition,
                        }
                    )
    maximum = max((line["intensity"] for line in lines), default=1.0)
    for line in lines:
        line["relative_intensity"] = min(
            100.0, 100.0 * line["intensity"] / maximum
        )
    return sorted(lines, key=lambda item: item["frequency_MHz"])


def calpgm_qm_parameter_priors(
    model: Mapping[str, Any],
    seed: Mapping[str, Any],
) -> dict[str, tuple[float, float]]:
    """Map general QM constants, surfaces and tensors onto CALPGM parameters."""

    validate_qm_seed(seed)
    parameters = tuple(model.get("parameters", ()))
    keys = _parameter_keys(parameters)
    by_identifier = {
        str(abs(int(parameter["identifier"]))): key
        for key, parameter in zip(keys, parameters, strict=True)
    }
    by_label = {str(parameter.get("label", "")): key for key, parameter in zip(keys, parameters, strict=True)}
    result = {}
    for item in seed["parameters"]:
        if not bool(item.get("fit", True)) or not bool(item.get("use_as_prior", True)):
            continue
        path = str(item["path"])
        token = path.rsplit(".", 1)[-1]
        key = by_label.get(token) or by_identifier.get(token)
        if key is None:
            continue
        result[key] = (float(item["value"]), float(item["sigma"]))
    return result


def calpgm_qm_parameter_constraints(
    model: Mapping[str, Any],
    seed: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expand direct QM priors and MORPHEUS-style predicates onto CALPGM keys."""

    validate_qm_seed(seed)
    parameters = tuple(model.get("parameters", ()))
    keys = _parameter_keys(parameters)
    lookup = {
        token: key
        for key, parameter in zip(keys, parameters, strict=True)
        for token in (str(parameter.get("label", "")), str(abs(int(parameter["identifier"]))))
        if token
    }

    def mapped_key(item: Mapping[str, Any]) -> str | None:
        path = str(item["path"])
        return lookup.get(str(item.get("name", ""))) or lookup.get(path.rsplit(".", 1)[-1])

    constraints = []
    for item in seed["parameters"]:
        key = mapped_key(item)
        if key is None or not bool(item.get("fit", True)):
            continue
        if bool(item.get("use_as_prior", True)):
            constraints.append(
                {
                    "name": f"qm-prior:{item['path']}",
                    "parameter": key,
                    "value": float(item["value"]),
                    "sigma": float(item["sigma"]),
                    "kind": "prior",
                }
            )
    for index, predicate in enumerate(seed.get("predicates", ())):
        pattern = str(predicate["path_pattern"])
        matches = [
            item
            for item in seed["parameters"]
            if bool(item.get("fit", True))
            and (
                fnmatchcase(str(item["path"]), pattern)
                or fnmatchcase(str(item.get("name", item["path"])), pattern)
            )
        ]
        mapped = [(item, mapped_key(item)) for item in matches]
        mapped = [(item, key) for item, key in mapped if key is not None]
        if bool(predicate.get("required", True)) and not mapped:
            raise ValueError(f"QM predicate {pattern!r} matches no CALPGM parameter")
        for item, key in mapped:
            constraints.append(
                {
                    "name": str(predicate.get("name", f"qm-predicate-{index + 1}")),
                    "parameter": key,
                    "value": float(predicate.get("value", item["value"])),
                    "sigma": float(predicate["sigma"]),
                    "kind": "predicate",
                    "path_pattern": pattern,
                }
            )
    return constraints


def _fit_component(payload: Mapping[str, Any]) -> tuple[int, int, int, int, float]:
    lower, upper = payload["lower"], payload["upper"]
    lower_block = lower.get("F", lower.get("J"))
    upper_block = upper.get("F", upper.get("J"))
    if lower_block is None or upper_block is None:
        raise ValueError("native CALPGM levels require F (spinful) or J (spin-free)")
    result = (
        int(lower_block),
        int(lower["level"]),
        int(upper_block),
        int(upper["level"]),
        float(payload.get("weight", 1.0)),
    )
    if min(result[:4]) < 0 or not math.isfinite(result[4]) or result[4] <= 0.0:
        raise ValueError("invalid native CALPGM fit component")
    return result


def _fit_observation(
    payload: Mapping[str, Any],
) -> tuple[tuple[tuple[int, int, int, int, float], ...], float, float]:
    raw_components = payload.get("components")
    if raw_components is None:
        raw_components = (payload,)
    if not raw_components:
        raise ValueError("native CALPGM blend requires at least one component")
    components = tuple(_fit_component(item) for item in raw_components)
    total_weight = sum(item[4] for item in components)
    components = tuple((*item[:4], item[4] / total_weight) for item in components)
    frequency = float(payload["frequency_MHz"])
    sigma = float(payload["uncertainty_MHz"])
    if not math.isfinite(frequency) or not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("invalid native CALPGM fit observation")
    return components, frequency, sigma


def _calpgm_ratio_relations(
    parameters: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, tuple[str, float]]:
    """Build general multiplicative dependencies for constrained parameters."""

    defaults = {
        key: float(item["value"]) for key, item in zip(keys, parameters, strict=True)
    }
    relations: dict[str, tuple[str, float]] = {}
    for index, (key, item) in enumerate(zip(keys, parameters, strict=True)):
        if not bool(item.get("ratio_constrained", False)):
            continue
        if index == 0:
            raise ValueError("the first CALPGM parameter cannot be ratio constrained")
        master = str(item.get("ratio_master", keys[index - 1]))
        if master not in defaults or defaults[master] == 0.0:
            raise ValueError(f"invalid ratio master for CALPGM parameter {key!r}")
        relations[key] = (master, defaults[key] / defaults[master])
    for dependent, specification in (overrides or {}).items():
        if dependent not in defaults:
            raise ValueError(f"unknown ratio-constrained parameter: {dependent!r}")
        if not isinstance(specification, Mapping):
            raise ValueError("a ratio constraint must contain master and factor")
        master = str(specification["master"])
        factor = float(specification["factor"])
        if master not in defaults or not math.isfinite(factor):
            raise ValueError(f"invalid ratio constraint for {dependent!r}")
        relations[str(dependent)] = (master, factor)

    def visit(name: str, active: set[str]) -> None:
        if name not in relations:
            return
        if name in active:
            raise ValueError("cyclic CALPGM ratio constraints")
        visit(relations[name][0], {*active, name})

    for name in relations:
        visit(name, set())
    return relations


def _expand_calpgm_ratio_parameters(
    values: Mapping[str, float], relations: Mapping[str, tuple[str, float]]
) -> dict[str, float]:
    expanded = {str(name): float(value) for name, value in values.items()}

    def resolve(name: str) -> float:
        if name not in relations:
            return expanded[name]
        master, factor = relations[name]
        expanded[name] = factor * resolve(master)
        return expanded[name]

    for dependent in relations:
        resolve(dependent)
    return expanded


def _calpgm_ratio_response(
    parameter: str,
    variable: str,
    relations: Mapping[str, tuple[str, float]],
) -> float:
    if parameter == variable:
        return 1.0
    if parameter not in relations:
        return 0.0
    master, factor = relations[parameter]
    return factor * _calpgm_ratio_response(master, variable, relations)


def fit_calpgm_native(
    model: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    checkpoint: Path | str | None = None,
    aggregate_map: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit assigned native level indices with analytic tracked derivatives."""

    from scipy.optimize import least_squares

    if request.get("schema") != CALPGM_NATIVE_FIT_SCHEMA:
        raise ValueError(f"unsupported native CALPGM fit schema: {request.get('schema')!r}")
    observations = tuple(_fit_observation(item) for item in request.get("observations", ()))
    if not observations:
        raise ValueError("native CALPGM fitting requires observations")
    if any(calpgm_spin_twice_from_model(model)) and aggregate_map is None:
        raise ValueError("spinful native CALPGM fitting requires an aggregate map")
    all_parameters = tuple(model.get("parameters", ()))
    keys = _parameter_keys(all_parameters)
    defaults = {key: float(item["value"]) for key, item in zip(keys, all_parameters, strict=True)}
    ratio_relations = _calpgm_ratio_relations(
        all_parameters, keys, request.get("ratio_constraints")
    )
    default_variables = tuple(
        key for key, item in zip(keys, all_parameters, strict=True)
        if key not in ratio_relations
        and abs(float(item.get("uncertainty_or_constraint", 1.0e35))) >= 1.0e20
    )
    variables = tuple(str(value) for value in request.get("variables", default_variables))
    if not variables:
        raise ValueError("native CALPGM fitting requires at least one variable")
    unknown = set(variables) - set(keys)
    if unknown:
        raise ValueError(f"unknown native CALPGM fit variables: {sorted(unknown)}")
    constrained = set(ratio_relations)
    if constrained & set(variables):
        raise ValueError(
            "ratio-constrained CALPGM parameters cannot be independent variables"
        )
    scales = np.asarray(
        [float(request.get("scales", {}).get(name, max(abs(defaults[name]), 1.0))) for name in variables]
    )
    if np.any(scales <= 0.0):
        raise ValueError("native CALPGM fit scales must be positive")
    J_values = sorted({
        block
        for components, _frequency, _sigma in observations
        for lower_J, _lower_level, upper_J, _upper_level, _weight in components
        for block in (lower_J, upper_J)
    })
    requested_roots = {
        J: 1 + max(
            level
            for components, _frequency, _sigma in observations
            for lower_J, lower_level, upper_J, upper_level, _weight in components
            for block_J, level in ((lower_J, lower_level), (upper_J, upper_level))
            if block_J == J
        )
        for J in J_values
    }
    tracking_buffer = int(request.get("tracking_buffer", 4))
    if tracking_buffer < 0:
        raise ValueError("native CALPGM tracking buffer must be nonnegative")
    roots = {J: count + tracking_buffer for J, count in requested_roots.items()}
    workers = int(request.get("workers", 1))
    blas_threads = request.get("blas_threads")
    blas_threads = None if blas_threads is None else int(blas_threads)
    threshold = int(request.get("matrix_free_threshold", 1024))
    backend = str(request.get("factorized_backend", "auto"))
    workspace = EigensolverWorkspace()
    initial_parameters = defaults.copy()
    requested_initial = request.get("initial_parameters", {})
    if set(requested_initial) - set(defaults):
        raise ValueError("native CALPGM initial_parameters contains unknown parameters")
    initial_parameters.update({
        str(name): float(value) for name, value in requested_initial.items()
    })
    initial_parameters = _expand_calpgm_ratio_parameters(
        initial_parameters, ratio_relations
    )
    resumed = False
    checkpoint_path = Path(checkpoint) if checkpoint is not None else None
    if checkpoint_path is not None and checkpoint_path.is_file():
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if saved.get("schema") != "matrix.lock.calpgm_native_fit_checkpoint.v1":
            raise ValueError("unsupported native CALPGM fit checkpoint schema")
        if tuple(saved.get("variables", ())) != variables:
            raise ValueError("native CALPGM checkpoint variables do not match the request")
        saved_parameters = saved.get("parameters", {})
        if set(saved_parameters) - set(defaults):
            raise ValueError("native CALPGM checkpoint contains unknown parameters")
        initial_parameters.update(
            {name: float(value) for name, value in saved_parameters.items()}
        )
        initial_parameters = _expand_calpgm_ratio_parameters(
            initial_parameters, ratio_relations
        )
        workspace_path = checkpoint_path.parent / str(saved["workspace"])
        if workspace_path.is_file():
            workspace = read_eigensolver_workspace(workspace_path)
        resumed = True
    solve_blocks = (
        {"F_values": J_values, "aggregate_map": aggregate_map}
        if any(calpgm_spin_twice_from_model(model))
        else {"J_values": J_values}
    )
    reference = solve_calpgm_native(
        model,
        **solve_blocks,
        levels_per_block=roots,
        parameters=initial_parameters,
        workers=workers,
        matrix_free_threshold=threshold,
        factorized_backend=backend,
        workspace=workspace,
        blas_threads=blas_threads,
    )
    compiled_cache = reference.compiled_blocks
    constraints = [
        {
            "name": f"request-prior:{name}",
            "parameter": str(name),
            "value": float(value[0]),
            "sigma": float(value[1]),
            "kind": "prior",
        }
        for name, value in request.get("priors", {}).items()
    ]
    if "qm_seed" in request:
        constraints.extend(calpgm_qm_parameter_constraints(model, request["qm_seed"]))
    invalid_constraints = [
        item["parameter"] for item in constraints if item["parameter"] not in defaults
    ]
    if invalid_constraints:
        raise ValueError(
            f"unknown native CALPGM theoretical parameters: {sorted(set(invalid_constraints))}"
        )
    best = {"cost": float("inf"), "parameters": initial_parameters.copy()}
    cache: dict[str, Any] = {}
    jacobian_combinations = {
        variable: {
            parameter: response
            for parameter in defaults
            if (
                response := _calpgm_ratio_response(
                    parameter, variable, ratio_relations
                )
            ) != 0.0
        }
        for variable in variables
    }

    def evaluate(scaled: np.ndarray):
        scaled = np.asarray(scaled, dtype=float)
        previous = cache.get("scaled")
        if previous is not None and np.array_equal(previous, scaled):
            return cache["residual"], cache["result"]
        parameters = defaults.copy()
        parameters.update(
            {name: defaults[name] + scales[index] * scaled[index] for index, name in enumerate(variables)}
        )
        parameters = _expand_calpgm_ratio_parameters(parameters, ratio_relations)
        current = solve_calpgm_native(
            model,
            **solve_blocks,
            levels_per_block=roots,
            parameters=parameters,
            workers=workers,
            matrix_free_threshold=threshold,
            factorized_backend=backend,
            workspace=workspace,
            compiled_blocks=compiled_cache,
            blas_threads=blas_threads,
        )
        tracked, tracking = track_calpgm_solution(reference, current)
        by_J = {block.J: block for block in tracked.blocks}
        residual = np.asarray(
            [
                (
                    frequency
                    - sum(
                        weight * (
                            by_J[upper_J].energies_MHz[upper_level]
                            - by_J[lower_J].energies_MHz[lower_level]
                        )
                        for lower_J, lower_level, upper_J, upper_level, weight
                        in components
                    )
                )
                / sigma
                for components, frequency, sigma in observations
            ],
            dtype=float,
        )
        prior_rows = [
            (parameters[item["parameter"]] - item["value"]) / item["sigma"]
            for item in constraints
        ]
        total = np.concatenate((residual, np.asarray(prior_rows, dtype=float)))
        cost = float(total @ total)
        if cost < best["cost"]:
            best.update(cost=cost, parameters=parameters.copy(), result=tracked, tracking=tracking)
            if checkpoint is not None:
                _write_calpgm_fit_checkpoint(checkpoint, variables, parameters, cost, workspace)
        cache.update(scaled=scaled.copy(), residual=total, result=tracked)
        return total, tracked

    def residual(scaled: np.ndarray) -> np.ndarray:
        return evaluate(scaled)[0]

    def jacobian(scaled: np.ndarray) -> np.ndarray:
        _residual, solved = evaluate(scaled)
        derivatives = calpgm_level_jacobians(
            solved,
            backend=backend,
            workers=workers,
            parameter_combinations=jacobian_combinations,
        )
        rows = []
        for components, _frequency, sigma in observations:
            reference_names = derivatives[components[0][0]][0]
            index = {name: column for column, name in enumerate(reference_names)}
            rows.append(
                [
                    -sum(
                        weight * (
                            derivatives[upper_J][1][upper_level, index[name]]
                            - derivatives[lower_J][1][lower_level, index[name]]
                        )
                        for lower_J, lower_level, upper_J, upper_level, weight
                        in components
                    )
                    * scales[column]
                    / sigma
                    for column, name in enumerate(variables)
                ]
            )
        for item in constraints:
            name, sigma = item["parameter"], item["sigma"]
            row = np.zeros(len(variables))
            for column, variable in enumerate(variables):
                row[column] = (
                    _calpgm_ratio_response(name, variable, ratio_relations)
                    * scales[column] / sigma
                )
            rows.append(row.tolist())
        return np.asarray(rows, dtype=float)

    initial_scaled = np.asarray(
        [(initial_parameters[name] - defaults[name]) / scales[index]
         for index, name in enumerate(variables)],
        dtype=float,
    )
    initial_jacobian = jacobian(initial_scaled)
    column_norms = np.linalg.norm(initial_jacobian, axis=0)
    largest_column_norm = float(np.max(column_norms, initial=0.0))
    inactive_tolerance = float(
        request.get("unobservable_derivative_tolerance", 1.0e-11)
    )
    if not 0.0 <= inactive_tolerance < 1.0:
        raise ValueError(
            "unobservable_derivative_tolerance must lie in [0, 1)"
        )
    active_columns = np.asarray([
        index for index, norm in enumerate(column_norms)
        if largest_column_norm == 0.0
        or norm > inactive_tolerance * largest_column_norm
    ], dtype=int)
    inactive_columns = tuple(
        index for index in range(len(variables)) if index not in set(active_columns)
    )
    if not len(active_columns):
        raise ValueError("all native CALPGM fit variables are unobservable")

    def expanded_scaled(active: np.ndarray) -> np.ndarray:
        full = initial_scaled.copy()
        full[active_columns] = np.asarray(active, dtype=float)
        return full

    optimized = least_squares(
        lambda active: residual(expanded_scaled(active)),
        initial_scaled[active_columns],
        jac=lambda active: jacobian(expanded_scaled(active))[:, active_columns],
        x_scale="jac",
        loss=str(request.get("loss", "soft_l1")),
        max_nfev=int(request.get("maximum_evaluations", 100)),
        ftol=float(request.get("function_tolerance", 1.0e-8)),
        xtol=float(request.get("step_tolerance", 1.0e-8)),
        gtol=float(request.get("gradient_tolerance", 1.0e-8)),
    )
    optimized_scaled = expanded_scaled(np.asarray(optimized.x))
    final_residual, final = evaluate(optimized_scaled)
    fitted = defaults.copy()
    fitted.update(
        {
            name: defaults[name] + scales[index] * optimized_scaled[index]
            for index, name in enumerate(variables)
        }
    )
    fitted = _expand_calpgm_ratio_parameters(fitted, ratio_relations)
    by_block = {block.J: block for block in final.blocks}
    comparisons = []
    residuals_MHz = []
    for observation_index, (components, frequency, sigma) in enumerate(observations):
        calculated = sum(
            weight * (
                by_block[upper_J].energies_MHz[upper_level]
                - by_block[lower_J].energies_MHz[lower_level]
            )
            for lower_J, lower_level, upper_J, upper_level, weight in components
        )
        difference = float(frequency - calculated)
        residuals_MHz.append(difference)
        comparisons.append({
            "observation": observation_index,
            "component_count": len(components),
            "observed_MHz": frequency,
            "calculated_MHz": float(calculated),
            "observed_minus_calculated_MHz": difference,
            "uncertainty_MHz": sigma,
            "standardized_residual": difference / sigma,
        })
    residual_values = np.asarray(residuals_MHz, dtype=float)
    final_jacobian = jacobian(optimized_scaled)
    data_jacobian = final_jacobian[:len(observations)]
    _augmented_u, augmented_singular, augmented_right_vectors = np.linalg.svd(
        final_jacobian, full_matrices=False
    )
    _u, singular_values, right_vectors = np.linalg.svd(
        data_jacobian, full_matrices=False
    )
    relative_rank_tolerance = float(request.get("relative_rank_tolerance", 1.0e-10))
    if not 0.0 < relative_rank_tolerance < 1.0:
        raise ValueError("relative_rank_tolerance must lie between zero and one")
    cutoff = (
        relative_rank_tolerance * singular_values[0]
        if singular_values.size else float("inf")
    )
    rank = int(np.count_nonzero(singular_values > cutoff))
    augmented_cutoff = (
        relative_rank_tolerance * augmented_singular[0]
        if augmented_singular.size else float("inf")
    )
    augmented_rank = int(np.count_nonzero(augmented_singular > augmented_cutoff))
    if augmented_rank:
        retained = augmented_right_vectors[:augmented_rank]
        covariance_scaled = (
            retained.T / augmented_singular[:augmented_rank] ** 2
        ) @ retained
    else:
        covariance_scaled = np.zeros((len(variables), len(variables)))
    covariance = covariance_scaled * scales[:, None] * scales[None, :]
    standard_errors = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    denominator = standard_errors[:, None] * standard_errors[None, :]
    correlation = np.divide(
        covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0.0
    )
    weak_count = min(5, len(variables))
    weak_combinations = []
    for singular_index in range(len(singular_values) - weak_count, len(singular_values)):
        vector = right_vectors[singular_index]
        scale = float(np.max(np.abs(vector), initial=0.0))
        weak_combinations.append({
            "singular_value": float(singular_values[singular_index]),
            "relative_singular_value": float(
                singular_values[singular_index] / singular_values[0]
            ),
            "coefficients": {
                name: float(value / scale)
                for name, value in zip(variables, vector, strict=True)
                if abs(value / scale) >= 0.05
            },
        })
    augmented_null_basis = augmented_right_vectors[augmented_rank:]
    parameter_uncertainties = {}
    unidentifiable_parameters = []
    for parameter in keys:
        response = np.asarray([
            _calpgm_ratio_response(parameter, variable, ratio_relations)
            for variable in variables
        ])
        response_scaled = response * scales
        null_projection = (
            float(np.linalg.norm(augmented_null_basis @ response_scaled))
            if augmented_null_basis.size else 0.0
        )
        explicitly_frozen = (
            parameter in variables
            and variables.index(parameter) in inactive_columns
        )
        if explicitly_frozen or null_projection > relative_rank_tolerance * max(
                float(np.linalg.norm(response_scaled)), 1.0):
            parameter_uncertainties[parameter] = None
            unidentifiable_parameters.append(parameter)
        else:
            variance = float(response @ covariance @ response)
            parameter_uncertainties[parameter] = math.sqrt(max(0.0, variance))
    data_standardized = final_residual[:len(observations)]
    chi_square = float(data_standardized @ data_standardized)
    degrees_of_freedom = max(0, len(observations) - rank)
    if checkpoint is not None:
        _write_calpgm_fit_checkpoint(
            checkpoint,
            variables,
            fitted,
            float(final_residual @ final_residual),
            workspace,
        )
    return {
        "schema": CALPGM_NATIVE_FIT_RESULT_SCHEMA,
        "status": "converged" if optimized.success else "stopped",
        "message": optimized.message,
        "variables": list(variables),
        "parameters": fitted,
        "observation_count": len(observations),
        "component_count": sum(len(row[0]) for row in observations),
        "blend_count": sum(len(row[0]) > 1 for row in observations),
        "prior_count": sum(item["kind"] == "prior" for item in constraints),
        "predicate_count": sum(item["kind"] == "predicate" for item in constraints),
        "normalized_rms": float(np.sqrt(np.mean(data_standardized * data_standardized))),
        "rms_residual_MHz": float(np.sqrt(np.mean(residual_values * residual_values))),
        "mean_residual_MHz": float(np.mean(residual_values)),
        "maximum_absolute_residual_MHz": float(np.max(np.abs(residual_values))),
        "chi_square": chi_square,
        "reduced_chi_square": (
            chi_square / degrees_of_freedom if degrees_of_freedom else None
        ),
        "degrees_of_freedom": degrees_of_freedom,
        "cost": float(final_residual @ final_residual),
        "evaluations": int(optimized.nfev),
        "jacobian_evaluations": int(optimized.njev or 0),
        "resumed_from_checkpoint": resumed,
        "analytic_jacobian": True,
        "tracking": "Hungarian plus degenerate-subspace SVD",
        "factorized_backend": backend,
        "compiled_block_cache": True,
        "frozen_unobservable_variables": [variables[index] for index in inactive_columns],
        "blas_threads": blas_threads if blas_threads is not None else (1 if workers > 1 else None),
        "ratio_constraints": {
            dependent: {"master": master, "factor": factor}
            for dependent, (master, factor) in ratio_relations.items()
        },
        "parameter_standard_errors": parameter_uncertainties,
        "unidentifiable_parameters": unidentifiable_parameters,
        "identifiability": {
            "rank": rank,
            "augmented_rank": augmented_rank,
            "variable_count": len(variables),
            "relative_rank_tolerance": relative_rank_tolerance,
            "singular_values": [float(value) for value in singular_values],
            "condition_number": (
                float(singular_values[0] / singular_values[rank - 1])
                if rank else None
            ),
            "weak_parameter_combinations": weak_combinations,
        },
        "covariance": covariance.tolist(),
        "correlation": correlation.tolist(),
        "comparisons": comparisons,
        "blocks": final.to_dict()["blocks"],
        "theoretical_constraints": [
            {
                **item,
                "standardized_residual": (
                    fitted[item["parameter"]] - item["value"]
                ) / item["sigma"],
            }
            for item in constraints
        ],
    }


def _write_calpgm_fit_checkpoint(
    path: Path | str,
    variables: Sequence[str],
    parameters: Mapping[str, float],
    cost: float,
    workspace: EigensolverWorkspace,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    vectors = target.with_suffix(target.suffix + ".npz")
    write_eigensolver_workspace(vectors, workspace)
    payload = {
        "schema": "matrix.lock.calpgm_native_fit_checkpoint.v1",
        "variables": list(variables),
        "parameters": dict(parameters),
        "cost": float(cost),
        "workspace": vectors.name,
    }
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def write_calpgm_native_result(path: Path | str, result: CALPGMNativeResult) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target


def write_calpgm_native_catalog(
    path: Path | str,
    transitions: Sequence[Mapping[str, Any]],
    *,
    dipole_debye: Sequence[float],
    temperature_K: float,
) -> Path:
    """Write a versioned native CALPGM intensity catalog."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CALPGM_NATIVE_CATALOG_SCHEMA,
        "temperature_K": float(temperature_K),
        "dipole_debye": [float(value) for value in dipole_debye],
        "transitions": [dict(value) for value in transitions],
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


__all__ = [
    "CALPGM_NATIVE_FIT_RESULT_SCHEMA",
    "CALPGM_NATIVE_FIT_SCHEMA",
    "CALPGM_NATIVE_CATALOG_SCHEMA",
    "CALPGM_NATIVE_RESULT_SCHEMA",
    "CALPGMCompiledBlock",
    "CALPGMNativeResult",
    "CALPGMSolvedBlock",
    "calpgm_block_diagonal",
    "calpgm_level_jacobians",
    "calpgm_qm_parameter_priors",
    "calpgm_qm_parameter_constraints",
    "calpgm_transition_catalog",
    "compile_calpgm_block",
    "fit_calpgm_native",
    "solve_calpgm_native",
    "track_calpgm_solution",
    "write_calpgm_native_result",
    "write_calpgm_native_catalog",
]
