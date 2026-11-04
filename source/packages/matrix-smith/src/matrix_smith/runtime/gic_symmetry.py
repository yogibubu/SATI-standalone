from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import json

import numpy as np

from matrix_chem.symmetry import cartesian_operation_matrix
from matrix_numerics import numerical_matrix_rank

from matrix_chem.geometry_io import read_xyz_atoms_coords
from matrix_chem.topology.elements import atomic_number, atomic_symbol
from matrix_chem.topology.pipeline import build_topology_objects
from matrix_smith.coordinate_kernel.pipeline import b_matrix_analytic
from matrix_smith.coordinate_kernel.primitives import Primitive
from matrix_smith.coordinate_kernel.symmetry_classifier import group_label as _group_label
from matrix_smith.coordinate_kernel.symmetry_detector import orient_coords, symmetry_elements_from_geometry
from matrix_smith.coordinate_kernel.symmetry_global import (
    irrep_characters_for_operations,
    primitive_permutation,
)


SYMM_TOL = 1.0e-2
SYMM_INERTIA_TOL = 1.0e-3
ZERO_TOL = 1.0e-8
RANK_TOL = 1.0e-7
FIT_TOL = 1.0e-4
PRINT_TOL = 1.0e-6
PROVOUT_SYMM_START = " Symmetrized GIC summary from GICSYM"
PROVOUT_SYMM_END = " End Symmetrized GIC summary from GICSYM"


@dataclass(frozen=True)
class GICLine:
    name: str
    terms: tuple[tuple[float, Primitive], ...]


def write_gic_symmetry_files(
    workdir: Path,
    *,
    symmetrize_gics: bool | None = None,
    symmetry_backend: str | None = None,
) -> None:
    run_dir = Path(workdir)
    gauin = run_dir / "gauin"
    raw_gauin = run_dir / "gauin.raw"
    source_gauin = raw_gauin if raw_gauin.exists() else gauin
    xyzin = run_dir / "xyzin"
    if not xyzin.exists():
        return
    atoms, coords, _comment = read_xyz_atoms_coords(xyzin)
    if sycart_requested(run_dir):
        _write_sycart_files(run_dir, atoms, coords)
    if symmetrize_gics is None:
        symmetrize_gics = True
    if not symmetrize_gics or not source_gauin.exists():
        return
    gics = _parse_gauin_gics(source_gauin)
    if not gics:
        return
    prims, u_matrix = _primitive_basis(gics)
    oriented = _oriented_coords(atoms, coords)
    op_data = _operation_data(atoms, oriented, prims, already_oriented=True)
    backend = _requested_symmetry_backend(run_dir, symmetry_backend)
    python_point_group = _group_label(
        [(label, rotation, 0.0) for label, rotation, _mapping, _primitive_op in op_data]
    )
    fortran_point_group = _fortran_point_group(run_dir / "provout")
    point_group = (
        fortran_point_group
        if backend == "fortran" and fortran_point_group != "UNKNOWN"
        else python_point_group
    )
    raw_class_targets = _class_counts(u_matrix, prims)
    sym_gics, class_targets = _symmetry_adapted_gics(
        atoms, gics, prims, u_matrix, op_data, oriented, point_group=point_group
    )
    _write_gicsym(run_dir / "gicsym", sym_gics)
    _write_gic_symmetry_diagnostics(
        run_dir / "gic_symmetry_diagnostics.json",
        sym_gics,
        op_data,
        len(coords),
        class_targets,
        prims,
        oriented,
        raw_class_targets,
        symmetry_backend=backend,
        point_group=point_group,
        python_point_group=python_point_group,
        fortran_point_group=fortran_point_group,
    )
    _write_symmetrized_gauin(source_gauin, run_dir / "gauin.symm", sym_gics, prims)
    if gicsym_requested(run_dir):
        _promote_symmetrized_gauin(gauin, run_dir / "gauin.symm")
        _append_symmetrized_provout(run_dir / "provout", sym_gics, prims)


def _parse_gauin_gics(gauin: Path) -> list[GICLine]:
    out: list[GICLine] = []
    for raw in gauin.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = _parse_gic_line(raw)
        if parsed is not None:
            out.append(parsed)
    return out


def _parse_gic_line(line: str) -> GICLine | None:
    stripped = line.strip()
    if not stripped or "=" not in stripped:
        return None
    name, rhs = stripped.split("=", 1)
    name = name.replace("(Inactive)", "").strip()
    if name.startswith(("QPck", "PhiP")):
        return None
    number = r"[+-]?\s*(?:\d+(?:\.\d*)?|\.\d+)(?:[EDed][+-]?\d+)?"
    terms: list[tuple[float, Primitive]] = []
    for match in re.finditer(rf"({number})\s*\*\s*([RADLUH])\(([^)]*)\)", rhs):
        coeff = float(match.group(1).replace(" ", "").replace("D", "E").replace("d", "e"))
        terms.append((coeff, _primitive(match.group(2), match.group(3))))
    if not terms:
        simple = re.search(r"\b([RADLUH])\(([^)]*)\)", rhs)
        if simple:
            terms.append((1.0, _primitive(simple.group(1), simple.group(2))))
    if not terms:
        return None
    return GICLine(name=name, terms=tuple(terms))


def _primitive(kind: str, atoms_text: str) -> Primitive:
    values = tuple(int(item.strip()) for item in atoms_text.split(",") if item.strip())
    atoms = tuple(value - 1 for value in values)
    if kind == "R" and len(atoms) == 2:
        return Primitive("bond", atoms)
    if kind == "A" and len(atoms) == 3:
        return Primitive("angle", atoms)
    if kind == "D" and len(atoms) == 4:
        return Primitive("dihedral", atoms)
    if kind == "U" and len(atoms) == 4:
        return Primitive("out_of_plane", atoms)
    if kind == "H" and len(atoms) == 4:
        return Primitive("out_of_plane_height", atoms)
    if kind == "L" and len(values) == 5:
        mode = values[4] if values[4] in {-1, -2} else -1
        ref = (values[3] - 1,) if values[3] > 0 else ()
        return Primitive("linear_bend", atoms[:3], mode=mode, ref=ref)
    raise ValueError(f"Unsupported GIC primitive {kind}({atoms_text})")


def _primitive_basis(gics: list[GICLine]) -> tuple[list[Primitive], np.ndarray]:
    prims: list[Primitive] = []
    index: dict[Primitive, int] = {}
    columns: list[np.ndarray] = []
    for gic in gics:
        col = np.zeros(len(prims), dtype=float)
        for coeff, prim in gic.terms:
            if prim not in index:
                index[prim] = len(prims)
                prims.append(prim)
                col = np.pad(col, (0, 1))
                for i, existing in enumerate(columns):
                    columns[i] = np.pad(existing, (0, 1))
            col[index[prim]] += coeff
        columns.append(col)
    return prims, np.column_stack(columns)


def _oriented_coords(atoms: list[str], coords: np.ndarray) -> np.ndarray:
    z_numbers = np.array([atomic_number(atom) for atom in atoms], dtype=int)
    return orient_coords(coords, weights=z_numbers)


def _operation_data(
    atoms: list[str], coords: np.ndarray, prims: list[Primitive], already_oriented: bool = False
):
    z_numbers = np.array([atomic_number(atom) for atom in atoms], dtype=int)
    symbols = [atomic_symbol(int(z)) for z in z_numbers]
    oriented = coords if already_oriented else orient_coords(coords, weights=z_numbers)
    elements, _classes, permutations = symmetry_elements_from_geometry(
        symbols,
        oriented,
        tol=SYMM_TOL,
        max_n=6,
        tol_H=SYMM_TOL,
        ignore_isotopes=True,
        auto_max_n=True,
        inertia_tol=SYMM_INERTIA_TOL,
    )
    unique = []
    seen = set()
    for element, permutation in zip(elements, permutations):
        mapped = tuple(int(item) for item in permutation)
        # Planar Cs molecules can have E and sigma with the same atom
        # permutation.  The rotation/reflection matrix is therefore part of
        # the identity of the symmetry operation.
        op_key = (mapped, tuple(np.round(np.asarray(element[1], dtype=float).reshape(-1), 8)))
        if op_key in seen:
            continue
        seen.add(op_key)
        unique.append((element[0], element[1], mapped, primitive_permutation(prims, mapped)))
    identity = tuple(range(len(atoms)))
    op_data = unique or [("E", np.eye(3), identity, primitive_permutation(prims, identity))]
    return _canonical_operation_order(op_data)


def _symmetry_adapted_gics(
    atoms, gics, prims, u_matrix, op_data, coords: np.ndarray, *, point_group: str | None = None
):
    irreps = _irrep_characters([item[0] for item in op_data], point_group=point_group)
    if not irreps:
        return [
            (gic.name, "A", "input", u_matrix[:, idx]) for idx, gic in enumerate(gics)
        ], _class_counts(u_matrix, prims)
    targets = _vibrational_irrep_counts(op_data, irreps, len(coords))
    b_primitive = b_matrix_analytic(prims, coords)
    source_rows = u_matrix.T @ b_primitive
    vib_projector = _vibrational_projector(coords)
    cart_ops = [
        _cartesian_operation(rotation, mapping, len(coords))
        for _label, rotation, mapping, _prim_op in op_data
    ]
    projection_blocks = _projection_blocks(atoms, coords, prims)
    class_targets = _class_counts(u_matrix, prims)
    irrep_order = {irrep: idx for idx, (irrep, _chars) in enumerate(irreps)}
    class_order = {
        "bond": 0,
        "angle": 1,
        "linear_bend": 2,
        "dihedral": 3,
        "out_of_plane": 4,
        "out_of_plane_height": 4,
    }
    candidates = []
    for col, source_row in enumerate(source_rows):
        kind = _dominant_kind(u_matrix[:, col], prims)
        for irrep, chars in irreps:
            projected_row_raw = _project_cartesian_row(source_row, chars, cart_ops)
            projected_row = projected_row_raw @ vib_projector
            score = float(np.linalg.norm(projected_row))
            if score > ZERO_TOL:
                candidates.append(
                    (
                        score,
                        class_order.get(kind, 9),
                        col,
                        irrep_order[irrep],
                        irrep,
                        chars,
                        kind,
                        projected_row_raw,
                        projected_row,
                    )
                )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    resolved_candidates = []
    for (
        score,
        class_idx,
        col,
        irrep_idx,
        irrep,
        chars,
        kind,
        projected_row_raw,
        projected_row,
    ) in candidates:
        coeff = _project_column_to_irrep(u_matrix[:, col], chars, op_data)
        coeff_norm = np.linalg.norm(coeff)
        source = "primitive_projection"
        if coeff_norm < ZERO_TOL:
            coeff, source = _fit_projected_coeff(
                projected_row_raw,
                projected_row,
                b_primitive,
                u_matrix[:, col],
                prims,
                projection_blocks,
                irrep not in {"A1", "A", "Ag", "A'"},
            )
            coeff_norm = np.linalg.norm(coeff) if coeff is not None else 0.0
        if coeff_norm < ZERO_TOL:
            continue
        output_row = coeff @ b_primitive @ vib_projector
        if np.linalg.norm(output_row) < RANK_TOL:
            continue
        resolved_candidates.append(
            (score, class_idx, col, irrep_idx, irrep, kind, source, coeff, output_row)
        )
    class_targets = _rank_limited_class_targets(
        class_targets, resolved_candidates, sum(targets.values())
    )
    capacities: dict[tuple[str, str], int] = {}
    for irrep, _chars in irreps:
        for kind in class_targets:
            rows = [
                entry[8] for entry in resolved_candidates if entry[4] == irrep and entry[5] == kind
            ]
            capacities[(irrep, kind)] = _rank_capacity(rows)
    block_targets = _allocate_symmetry_class_counts(
        targets,
        class_targets,
        capacities,
        [irrep for irrep, _chars in irreps],
        sorted(class_targets, key=lambda item: class_order.get(item, 9)),
    )
    chosen = []
    used_names: dict[str, int] = {}
    selected_rows: dict[str, list[np.ndarray]] = {irrep: [] for irrep, _chars in irreps}
    selected_class_rows: dict[tuple[str, str], list[np.ndarray]] = {}
    selected_class_coeffs: dict[tuple[str, str], list[np.ndarray]] = {}
    selected_classes: dict[str, int] = {kind: 0 for kind in class_targets}
    selected_global: list[np.ndarray] = []
    selected_blocks: dict[tuple[str, str], int] = {key: 0 for key in block_targets}
    fallback_used = False
    for (
        _score,
        _class_idx,
        col,
        _irrep_idx,
        irrep,
        kind,
        source,
        coeff,
        _output_row,
    ) in resolved_candidates:
        class_key = (irrep, kind)
        if selected_blocks.get(class_key, 0) >= block_targets.get(class_key, 0):
            continue
        class_rows = selected_class_rows.setdefault(class_key, [])
        class_coeffs = selected_class_coeffs.setdefault(class_key, [])
        output_row = coeff @ b_primitive @ vib_projector
        coeff_residual = coeff.astype(float, copy=True)
        for basis_row, basis_coeff in zip(class_rows, class_coeffs):
            coeff_residual -= np.dot(basis_row, output_row) * basis_coeff
        output_residual = coeff_residual @ b_primitive @ vib_projector
        row_norm = np.linalg.norm(output_residual)
        if row_norm < RANK_TOL:
            continue
        irrep_residual = _orthogonal_residual(output_residual / row_norm, selected_rows[irrep])
        irrep_norm = np.linalg.norm(irrep_residual)
        if irrep_norm < RANK_TOL:
            continue
        coeff = coeff_residual / row_norm
        output_unit = output_residual / row_norm
        global_residual = _orthogonal_residual(output_unit, selected_global)
        global_norm = np.linalg.norm(global_residual)
        if global_norm < RANK_TOL:
            continue
        class_rows.append(output_unit)
        class_coeffs.append(coeff)
        selected_rows[irrep].append(irrep_residual / irrep_norm)
        selected_global.append(global_residual / global_norm)
        selected_classes[kind] = selected_classes.get(kind, 0) + 1
        selected_blocks[class_key] = selected_blocks.get(class_key, 0) + 1
        chosen.append((irrep_order[irrep], col, irrep, kind, source, coeff))
        if (
            all(len(selected_rows[name]) == targets.get(name, 0) for name, _chars in irreps)
            and selected_classes == class_targets
            and selected_blocks == block_targets
        ):
            break
    counts = {irrep: len(rows) for irrep, rows in selected_rows.items()}
    if counts != targets:
        for (
            _score,
            _class_idx,
            col,
            _irrep_idx,
            irrep,
            kind,
            source,
            coeff,
            _output_row,
        ) in resolved_candidates:
            if len(selected_rows[irrep]) >= targets.get(irrep, 0):
                continue
            output_row = coeff @ b_primitive @ vib_projector
            output_residual = output_row
            residual = _orthogonal_residual(output_residual, selected_rows[irrep])
            norm = np.linalg.norm(residual)
            if norm < RANK_TOL:
                continue
            row_norm = np.linalg.norm(output_residual)
            if row_norm < RANK_TOL:
                continue
            selected_rows[irrep].append(residual / norm)
            selected_global.append(residual / norm)
            selected_classes[kind] = selected_classes.get(kind, 0) + 1
            chosen.append(
                (
                    irrep_order[irrep],
                    col,
                    irrep,
                    kind,
                    f"{source}_rank_completion",
                    coeff / row_norm,
                )
            )
            fallback_used = True
            counts = {name: len(rows) for name, rows in selected_rows.items()}
            if counts == targets:
                break
    if counts != targets:
        raise RuntimeError(f"GIC symmetry reduction count mismatch: {counts}; expected {targets}")
    if selected_classes != class_targets and not fallback_used:
        raise RuntimeError(
            f"GIC class count mismatch: {selected_classes}; expected {class_targets}"
        )
    adapted = []
    for _irrep_idx, _col, irrep, kind, source, coeff in sorted(
        chosen, key=lambda item: (item[0], item[1])
    ):
        adapted.append((_next_name(irrep, kind, used_names), irrep, source, coeff))
    return adapted, class_targets


def _class_counts(u_matrix: np.ndarray, prims: list[Primitive]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for col in range(u_matrix.shape[1]):
        kind = _dominant_kind(u_matrix[:, col], prims)
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _rank_capacity(rows: list[np.ndarray]) -> int:
    if not rows:
        return 0
    return numerical_matrix_rank(
        np.vstack(rows),
        absolute_tolerance=RANK_TOL,
    )


def _rank_limited_class_targets(
    class_targets: dict[str, int],
    resolved_candidates: list[tuple[float, int, int, int, str, str, str, np.ndarray, np.ndarray]],
    target_total: int,
) -> dict[str, int]:
    """Limit family targets to the global vibrational rank without mixing types."""
    current_total = sum(class_targets.values())
    if current_total == target_total:
        return class_targets
    if current_total < target_total:
        selected_rows: list[np.ndarray] = []
        counts = {kind: 0 for kind in class_targets}
        initial_total = current_total
        for (
            _score,
            _class_idx,
            _col,
            _irrep_idx,
            _irrep,
            kind,
            _source,
            _coeff,
            output_row,
        ) in resolved_candidates:
            if sum(counts.values()) < initial_total and counts.get(kind, 0) >= class_targets.get(
                kind, 0
            ):
                continue
            residual = _orthogonal_residual(output_row, selected_rows)
            norm = float(np.linalg.norm(residual))
            if norm < RANK_TOL:
                continue
            selected_rows.append(residual / norm)
            counts[kind] = counts.get(kind, 0) + 1
            if sum(counts.values()) == target_total:
                return {kind: count for kind, count in counts.items() if count}
        raise RuntimeError(
            f"Unable to expand GIC class targets {class_targets} to vibrational rank {target_total}"
        )
    selected_rows: list[np.ndarray] = []
    counts = {kind: 0 for kind in class_targets}
    for (
        _score,
        _class_idx,
        _col,
        _irrep_idx,
        _irrep,
        kind,
        _source,
        _coeff,
        output_row,
    ) in resolved_candidates:
        if counts.get(kind, 0) >= class_targets.get(kind, 0):
            continue
        residual = _orthogonal_residual(output_row, selected_rows)
        norm = float(np.linalg.norm(residual))
        if norm < RANK_TOL:
            continue
        selected_rows.append(residual / norm)
        counts[kind] = counts.get(kind, 0) + 1
        if sum(counts.values()) == target_total:
            return {kind: count for kind, count in counts.items() if count}
    raise RuntimeError(
        f"Unable to reduce GIC class targets {class_targets} to vibrational rank {target_total}"
    )


def _allocate_symmetry_class_counts(
    targets: dict[str, int],
    class_targets: dict[str, int],
    capacities: dict[tuple[str, str], int],
    irrep_order: list[str],
    kind_order: list[str],
) -> dict[tuple[str, str], int]:
    """Allocate final coordinates by irrep and physical coordinate family."""
    remaining_classes = dict(class_targets)
    assignments: dict[tuple[str, str], int] = {}

    def backtrack(row_index: int) -> bool:
        if row_index == len(irrep_order):
            return all(value == 0 for value in remaining_classes.values())
        irrep = irrep_order[row_index]
        for row_assignment in _row_count_assignments(
            irrep, targets[irrep], remaining_classes, capacities, kind_order
        ):
            for kind, count in row_assignment.items():
                remaining_classes[kind] -= count
                assignments[(irrep, kind)] = count
            if _remaining_capacity_sufficient(
                row_index + 1, irrep_order, kind_order, remaining_classes, capacities
            ) and backtrack(row_index + 1):
                return True
            for kind, count in row_assignment.items():
                remaining_classes[kind] += count
                assignments.pop((irrep, kind), None)
        return False

    if backtrack(0):
        return {key: value for key, value in assignments.items() if value}
    capacity_payload = {
        f"{irrep}:{kind}": int(capacities.get((irrep, kind), 0))
        for irrep in irrep_order
        for kind in kind_order
    }
    raise RuntimeError(
        "GIC symmetry class allocation failed: "
        f"targets={targets}; class_targets={class_targets}; capacities={capacity_payload}"
    )


def _row_count_assignments(
    irrep: str,
    target: int,
    remaining_classes: dict[str, int],
    capacities: dict[tuple[str, str], int],
    kind_order: list[str],
) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    current: dict[str, int] = {}

    def rec(kind_index: int, remaining: int) -> None:
        if kind_index == len(kind_order):
            if remaining == 0:
                out.append(dict(current))
            return
        kind = kind_order[kind_index]
        max_count = min(remaining, remaining_classes.get(kind, 0), capacities.get((irrep, kind), 0))
        for count in range(max_count, -1, -1):
            if count:
                current[kind] = count
            else:
                current.pop(kind, None)
            rest_capacity = 0
            for next_kind in kind_order[kind_index + 1 :]:
                rest_capacity += min(
                    remaining_classes.get(next_kind, 0), capacities.get((irrep, next_kind), 0)
                )
            if remaining - count <= rest_capacity:
                rec(kind_index + 1, remaining - count)
        current.pop(kind, None)

    rec(0, target)
    # Prefer chemically transparent A1 coordinates and scarce non-totally
    # symmetric families.  The sort is deterministic and only chooses among
    # allocations already compatible with the exact row/column counts.
    out.sort(key=lambda item: tuple(-item.get(kind, 0) for kind in kind_order))
    return out


def _remaining_capacity_sufficient(
    next_row: int,
    irrep_order: list[str],
    kind_order: list[str],
    remaining_classes: dict[str, int],
    capacities: dict[tuple[str, str], int],
) -> bool:
    for kind in kind_order:
        capacity = sum(capacities.get((irrep, kind), 0) for irrep in irrep_order[next_row:])
        if remaining_classes.get(kind, 0) > capacity:
            return False
    return True




def _projection_blocks(
    atoms: list[str], coords: np.ndarray, prims: list[Primitive]
) -> list[tuple[str, set[int]]]:
    blocks: list[tuple[str, set[int]]] = []
    z_numbers = [atomic_number(atom) for atom in atoms]
    try:
        _continuous, graph, ringset, _synthons, _aromaticity = build_topology_objects(
            coords, z_numbers
        )
    except Exception:
        return blocks
    adjacency = [set(neigh) for neigh in graph.adjacency]
    seen: set[tuple[str, tuple[int, ...]]] = set()

    for ring in ringset.rings:
        ring_atoms = set(ring.atoms)
        idxs = {
            idx
            for idx, prim in enumerate(prims)
            if prim.kind in {"angle", "dihedral", "out_of_plane"}
            and _ring_mixed_member(prim, ring_atoms, adjacency)
        }
        _append_block(blocks, seen, f"ring_mixed_{ring.index + 1}", idxs)

    for center in range(len(atoms)):
        local_atoms = set(adjacency[center])
        local_atoms.add(center)
        idxs = {
            idx
            for idx, prim in enumerate(prims)
            if prim.kind in {"dihedral", "out_of_plane", "out_of_plane_height"}
            and _oop_local_member(prim, center, local_atoms, adjacency)
        }
        _append_block(blocks, seen, f"oop_local_{center + 1}", idxs)
    return blocks


def _append_block(
    blocks: list[tuple[str, set[int]]],
    seen: set[tuple[str, tuple[int, ...]]],
    name: str,
    idxs: set[int],
) -> None:
    if len(idxs) < 2:
        return
    key = (name.split("_", 1)[0], tuple(sorted(idxs)))
    if key in seen:
        return
    seen.add(key)
    blocks.append((name, idxs))


def _ring_mixed_member(prim: Primitive, ring_atoms: set[int], adjacency: list[set[int]]) -> bool:
    atoms = set(prim.atoms)
    if atoms.issubset(ring_atoms):
        return True
    if prim.kind not in {"dihedral", "out_of_plane", "out_of_plane_height"}:
        return False
    if len(atoms & ring_atoms) < 3:
        return False
    external = atoms - ring_atoms
    return all(any(neigh in ring_atoms for neigh in adjacency[atom]) for atom in external)


def _oop_local_member(
    prim: Primitive, center: int, local_atoms: set[int], adjacency: list[set[int]]
) -> bool:
    atoms = set(prim.atoms)
    if not atoms.issubset(local_atoms):
        return False
    if prim.kind in {"out_of_plane", "out_of_plane_height"}:
        return len(prim.atoms) >= 1 and prim.atoms[0] == center
    if prim.kind != "dihedral":
        return False
    return (
        center in atoms
        and sum(1 for atom in atoms if atom != center and atom in adjacency[center]) >= 3
    )


def _canonical_operation_order(op_data):
    labels = [item[0] for item in op_data]
    if (
        len(op_data) == 4
        and any(label.startswith("C2") for label in labels)
        and sum(label.startswith("sigma") for label in labels) == 2
    ):
        order = {"E": 0, "C2": 1, "sigma_xz": 2, "sigma_xy": 3}

        def key(item):
            label = item[0]
            if label.startswith("C2"):
                return (order["C2"], label)
            return (order.get(label, 99), label)

        return sorted(op_data, key=key)
    return sorted(op_data, key=lambda item: (0 if item[0] == "E" else 1, item[0]))


def _cartesian_operation(rotation: np.ndarray, mapping: tuple[int, ...], natoms: int) -> np.ndarray:
    # The detector returns i -> j such that x_i matches R x_j; for row
    # gradients this block form applies the same operation in the oriented
    # Cartesian frame.
    return cartesian_operation_matrix(rotation, mapping, natoms=natoms)


def _project_cartesian_row(
    row: np.ndarray, chars: np.ndarray, cart_ops: list[np.ndarray]
) -> np.ndarray:
    projected = np.zeros_like(row)
    for op_index, op_matrix in enumerate(cart_ops):
        projected += chars[op_index] * (row @ op_matrix)
    return projected / float(len(cart_ops))


def _vibrational_projector(coords: np.ndarray) -> np.ndarray:
    natoms = len(coords)
    basis = []
    for axis in range(3):
        vec = np.zeros(3 * natoms, dtype=float)
        vec[axis::3] = 1.0
        basis.append(vec)
    for axis in np.eye(3):
        vec = np.array(
            [component for coord in coords for component in np.cross(axis, coord)], dtype=float
        )
        basis.append(vec)
    ortho: list[np.ndarray] = []
    for vec in basis:
        residual = _orthogonal_residual(vec, ortho)
        norm = np.linalg.norm(residual)
        if norm > 1.0e-10:
            ortho.append(residual / norm)
    if not ortho:
        return np.eye(3 * natoms, dtype=float)
    q_matrix = np.vstack(ortho).T
    return np.eye(3 * natoms, dtype=float) - q_matrix @ q_matrix.T


def _fit_projected_coeff(
    raw_row: np.ndarray,
    vib_row: np.ndarray,
    b_primitive: np.ndarray,
    source_coeff: np.ndarray,
    prims: list[Primitive],
    projection_blocks: list[tuple[str, set[int]]],
    allow_mixed_blocks: bool,
) -> tuple[np.ndarray | None, str]:
    kind = _dominant_kind(source_coeff, prims)
    support = {idx for idx, coeff in enumerate(source_coeff) if abs(coeff) > ZERO_TOL}
    same_type = {idx for idx, prim in enumerate(prims) if prim.kind == kind}
    candidates: list[tuple[str, set[int]]] = [(f"{kind}_type_projection", same_type)]
    if allow_mixed_blocks:
        for block_name, block_idxs in projection_blocks:
            if support and support.issubset(block_idxs):
                candidates.append((block_name, block_idxs))
        candidates.append(("full_pruned_gic_space", set(range(len(prims)))))

    for block_name, idxs in candidates:
        for row_name, row in (("cartesian", raw_row), ("vibrational", vib_row)):
            coeff = _least_squares_coeff(row, b_primitive, sorted(idxs))
            if _fit_residual(coeff, b_primitive, row) <= FIT_TOL:
                return coeff, f"{block_name}_{row_name}"
    return None, "unresolved"


def _fit_residual(coeff: np.ndarray, b_primitive: np.ndarray, row: np.ndarray) -> float:
    return float(np.linalg.norm(coeff @ b_primitive - row) / max(1.0, np.linalg.norm(row)))


def _least_squares_coeff(row: np.ndarray, b_primitive: np.ndarray, idxs: list[int]) -> np.ndarray:
    coeff = np.zeros(b_primitive.shape[0], dtype=float)
    if not idxs:
        return coeff
    sub_b = b_primitive[np.array(idxs, dtype=int), :]
    values, *_ = np.linalg.lstsq(sub_b.T, row.T, rcond=1.0e-10)
    coeff[np.array(idxs, dtype=int)] = values
    return coeff


def _project_column_to_irrep(source: np.ndarray, chars: np.ndarray, op_data) -> np.ndarray:
    projected = np.zeros_like(source)
    for op_index, (_label, _rotation, _mapping, (perm_idx, sign)) in enumerate(op_data):
        transformed = np.zeros_like(source)
        for src, dst in enumerate(perm_idx):
            transformed[dst] += sign[src] * source[src]
        projected += chars[op_index] * transformed
    return projected / float(len(op_data))


def _dominant_kind(column: np.ndarray, prims: list[Primitive]) -> str:
    weights: dict[str, float] = {}
    for coeff, prim in zip(column, prims):
        weights[prim.kind] = weights.get(prim.kind, 0.0) + float(coeff * coeff)
    if not weights:
        return "gic"
    return max(weights.items(), key=lambda item: item[1])[0]


def _orthogonal_residual(vector: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    residual = vector.astype(float, copy=True)
    for item in basis:
        residual -= np.dot(item, residual) * item
    return residual


def _irrep_characters(
    labels: list[str], point_group: str | None = None
) -> list[tuple[str, np.ndarray]]:
    return irrep_characters_for_operations(labels, point_group=point_group)


def _vibrational_irrep_counts(
    op_data, irreps: list[tuple[str, np.ndarray]], natoms: int
) -> dict[str, int]:
    gamma_3n = []
    gamma_trans = []
    gamma_rot = []
    for _label, rotation, mapping, _primitive_op in op_data:
        fixed = sum(1 for i, j in enumerate(mapping) if i == j)
        trace = float(np.trace(rotation))
        gamma_3n.append(fixed * trace)
        gamma_trans.append(trace)
        gamma_rot.append(float(np.linalg.det(rotation) * trace))
    gamma_vib = np.array(gamma_3n) - np.array(gamma_trans) - np.array(gamma_rot)
    counts = {}
    group_order = float(len(op_data))
    for irrep, chars in irreps:
        value = int(round(float(np.dot(gamma_vib, chars)) / group_order))
        counts[irrep] = max(value, 0)
    if sum(counts.values()) != 3 * natoms - 6:
        raise RuntimeError(
            f"Vibrational irrep count mismatch: {counts} sums to {sum(counts.values())}, expected {3 * natoms - 6}"
        )
    return counts


def _next_name(irrep: str, kind: str, used: dict[str, int]) -> str:
    prefix = {
        "bond": "Str",
        "angle": "Ang",
        "linear_bend": "Lin",
        "dihedral": "Tor",
        "out_of_plane": "Oop",
    }.get(kind, "Gic")
    key = f"{irrep}{prefix}"
    used[key] = used.get(key, 0) + 1
    return f"{irrep}{prefix}{used[key]:04d}"


def _write_gicsym(path: Path, sym_gics) -> None:
    lines = ["name,irrep,source"]
    for name, irrep, source, _column in sym_gics:
        lines.append(f"{name},{irrep},{source}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gic_symmetry_diagnostics(
    path: Path,
    sym_gics,
    op_data,
    natoms: int,
    class_targets: dict[str, int],
    prims: list[Primitive] | None = None,
    coords: np.ndarray | None = None,
    raw_class_targets: dict[str, int] | None = None,
    symmetry_backend: str = "python",
    point_group: str = "UNKNOWN",
    python_point_group: str = "UNKNOWN",
    fortran_point_group: str = "UNKNOWN",
) -> None:
    irreps = _irrep_characters([item[0] for item in op_data], point_group=point_group)
    targets = _vibrational_irrep_counts(op_data, irreps, natoms) if irreps else {"A": len(sym_gics)}
    counts: dict[str, int] = {irrep: 0 for irrep in targets}
    class_counts: dict[str, int] = {}
    sources: dict[str, int] = {}
    for name, irrep, source, _column in sym_gics:
        counts[irrep] = counts.get(irrep, 0) + 1
        class_name = _class_from_name(name)
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        sources[source] = sources.get(source, 0) + 1
    b_ranks: dict[str, int] = {}
    if prims is not None and coords is not None and sym_gics:
        b_primitive = b_matrix_analytic(prims, coords)
        vib_projector = _vibrational_projector(coords)
        for irrep in targets:
            rows = [
                column @ b_primitive @ vib_projector
                for _name, row_irrep, _source, column in sym_gics
                if row_irrep == irrep
            ]
            b_ranks[irrep] = (
                numerical_matrix_rank(
                    np.array(rows),
                    absolute_tolerance=RANK_TOL,
                )
                if rows
                else 0
            )
    payload = {
        "schema": "oracle.gic_symmetry.v1",
        "symmetry_backend": symmetry_backend,
        "point_group": point_group,
        "python_point_group": python_point_group,
        "fortran_point_group": fortran_point_group,
        "operation_order": [item[0] for item in op_data],
        "irreps": [name for name, _chars in irreps],
        "targets": targets,
        "counts": counts,
        "b_ranks": b_ranks,
        "class_targets": class_targets,
        "class_counts": class_counts,
        "raw_class_targets": raw_class_targets or class_targets,
        "sources": sources,
        "strict_clean": all(not source.startswith(("global_", "unresolved")) for source in sources),
        "tolerances": {
            "fit": FIT_TOL,
            "symmetry": SYMM_TOL,
            "inertia": SYMM_INERTIA_TOL,
            "zero": ZERO_TOL,
            "rank": RANK_TOL,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _class_from_name(name: str) -> str:
    if "Str" in name:
        return "bond"
    if "Ang" in name:
        return "angle"
    if "Lin" in name:
        return "linear_bend"
    if "Tor" in name:
        return "dihedral"
    if "Oop" in name:
        return "out_of_plane"
    return "gic"


def _write_symmetrized_gauin(source: Path, target: Path, sym_gics, prims: list[Primitive]) -> None:
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    first_gic = next(
        (i for i, line in enumerate(lines) if _parse_gic_line(line) is not None), len(lines)
    )
    prefix = [_explicit_gic_route(line) for line in lines[:first_gic]]
    out = list(prefix)
    a1 = [item for item in sym_gics if item[1] in {"A1", "A", "Ag", "A'"}]
    other = [item for item in sym_gics if item not in a1]
    for name, _irrep, _source, column in a1:
        out.append(_format_gic_line(name, column, prims))
    if other:
        out.append("")
    for name, _irrep, _source, column in other:
        out.append(_format_gic_line(name, column, prims))
    target.write_text("\n".join(out) + "\n", encoding="utf-8")


def _provin_text(run_dir: Path) -> str:
    provin = run_dir / "provin"
    if not provin.exists():
        return ""
    return provin.read_text(encoding="utf-8", errors="replace").upper()


def gicsym_requested(run_dir: Path) -> bool:
    text = _provin_text(run_dir)
    return "GICSYM" in text or "SYMMALL" in text


def _requested_symmetry_backend(run_dir: Path, explicit: str | None = None) -> str:
    if explicit is not None:
        value = explicit.strip().lower()
    else:
        text = _provin_text(run_dir)
        if "GICSYMPY" in text:
            value = "python"
        elif "GICSYMFT" in text or "GICSYMFORTRAN" in text:
            value = "fortran"
        else:
            value = "python"
    if value not in {"python", "fortran"}:
        raise ValueError(f"Unsupported GICSYM symmetry backend: {explicit!r}")
    return value


def _fortran_point_group(provout: Path) -> str:
    if not provout.exists():
        return "UNKNOWN"
    match = re.search(
        r"Point Group from symm\.f:\s*([A-Za-z0-9]+)",
        provout.read_text(encoding="utf-8", errors="replace"),
    )
    return match.group(1) if match else "UNKNOWN"


def sycart_requested(run_dir: Path) -> bool:
    return "SYCART" in _provin_text(run_dir)


def symmetry_postprocess_requested(run_dir: Path) -> bool:
    return gicsym_requested(run_dir) or sycart_requested(run_dir)


def _promote_symmetrized_gauin(gauin: Path, gauin_symm: Path) -> None:
    if not gauin.exists() or not gauin_symm.exists():
        return
    raw = gauin.with_name("gauin.raw")
    if not raw.exists():
        raw.write_text(gauin.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    gauin.write_text(gauin_symm.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")


def _append_symmetrized_provout(path: Path, sym_gics, prims: list[Primitive]) -> None:
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    clean: list[str] = []
    in_old_block = False
    for line in lines:
        if line.strip() == PROVOUT_SYMM_START.strip():
            in_old_block = True
            continue
        if line.strip() == PROVOUT_SYMM_END.strip():
            in_old_block = False
            continue
        if not in_old_block:
            clean.append(line)
    while clean and not clean[-1].strip():
        clean.pop()
    block = [
        "",
        PROVOUT_SYMM_START,
        _symmetrized_count_summary(sym_gics),
        " Name          Irrep    Source                      Coordinate",
    ]
    for name, irrep, source, column in sym_gics:
        block.append(
            f" {name:<13s} {irrep:<8s} {source:<27s} {_format_gic_line(name, column, prims).strip()}"
        )
    block.append(PROVOUT_SYMM_END)
    path.write_text("\n".join(clean + block) + "\n", encoding="utf-8")


def _symmetrized_count_summary(sym_gics) -> str:
    counts = {"Str": 0, "Ang": 0, "Lin": 0, "Tor": 0, "Oop": 0}
    for name, _irrep, _source, _column in sym_gics:
        for marker in counts:
            if marker in name:
                counts[marker] += 1
                break
    return (
        " Symmetrized coordinate counts:"
        f" Stretch={counts['Str']:5d}"
        f" Bend={counts['Ang']:5d}"
        f" Linear={counts['Lin']:5d}"
        f" Torsion={counts['Tor']:5d}"
        f" Out-of-plane={counts['Oop']:5d}"
        f" Total={sum(counts.values()):5d}"
    )


def _explicit_gic_route(line: str) -> str:
    if not line.lstrip().startswith("#"):
        return line
    return re.sub(
        r"geom=\(\s*readallgic\s*,\s*gic(?:all)?symm\s*\)",
        "geom=readallgic",
        line,
        flags=re.IGNORECASE,
    )


def _orient_with_frame(
    coords: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    com = np.sum(coords * weights[:, None], axis=0) / total
    centered = coords - com
    inertia = np.zeros((3, 3), dtype=float)
    for weight, vector in zip(weights, centered):
        inertia += weight * ((float(np.dot(vector, vector)) * np.eye(3)) - np.outer(vector, vector))
    _evals, evecs = np.linalg.eigh(inertia)
    order = np.argsort(_evals)
    frame = evecs[:, order]
    if np.linalg.det(frame) < 0.0:
        frame[:, -1] *= -1.0
    return centered @ frame, com, frame


def _write_sycart_files(run_dir: Path, atoms: list[str], coords: np.ndarray) -> None:
    weights = np.array([atomic_number(atom) for atom in atoms], dtype=float)
    oriented, com, frame = _orient_with_frame(coords, weights)
    elements, _classes, permutations = symmetry_elements_from_geometry(
        atoms,
        oriented,
        tol=SYMM_TOL,
        max_n=8,
        auto_max_n=True,
        inertia_tol=SYMM_INERTIA_TOL,
    )
    if not elements or not permutations:
        return
    sym_oriented = np.zeros_like(oriented)
    for element, mapping in zip(elements, permutations):
        rotation = element[1]
        transformed = oriented @ rotation.T
        permuted = np.zeros_like(transformed)
        for atom_index, mapped_index in enumerate(mapping):
            permuted[atom_index] = transformed[mapped_index]
        sym_oriented += permuted
    sym_oriented /= float(len(elements))
    sym_coords = np.round(sym_oriented @ frame.T + com[None, :], decimals=8)
    _write_xyz(
        run_dir / "sycart.xyz",
        atoms,
        sym_coords,
        "GICForge SyCart symmetrized Cartesian coordinates",
    )
    _write_xyz(
        run_dir / "symmetrized.xyz",
        atoms,
        sym_coords,
        "GICForge SyCart symmetrized Cartesian coordinates",
    )


def _write_xyz(path: Path, atoms: list[str], coords: np.ndarray, comment: str) -> None:
    lines = [str(len(atoms)), comment]
    for atom, (x, y, z) in zip(atoms, coords):
        lines.append(f"{atom:>2s} {x:16.8f} {y:16.8f} {z:16.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_gic_line(name: str, column: np.ndarray, prims: list[Primitive]) -> str:
    parts = []
    for coeff, primitive in zip(column, prims):
        if abs(coeff) < PRINT_TOL:
            continue
        parts.append((coeff, _primitive_expression(primitive)))
    expr = _join_terms(parts)
    return f" {name}=[ {expr}]"


def _primitive_expression(primitive: Primitive) -> str:
    atoms = tuple(atom + 1 for atom in primitive.atoms)
    if primitive.kind == "bond":
        return f"R({atoms[0]:3d},{atoms[1]:3d})"
    if primitive.kind == "angle":
        return f"A({atoms[0]:3d},{atoms[1]:3d},{atoms[2]:3d})"
    if primitive.kind == "dihedral":
        return f"D({atoms[0]:3d},{atoms[1]:3d},{atoms[2]:3d},{atoms[3]:3d})"
    if primitive.kind == "out_of_plane":
        return f"U({atoms[0]:3d},{atoms[1]:3d},{atoms[2]:3d},{atoms[3]:3d})"
    if primitive.kind == "out_of_plane_height":
        return f"H({atoms[0]:3d},{atoms[1]:3d},{atoms[2]:3d},{atoms[3]:3d})"
    if primitive.kind == "linear_bend":
        reference = primitive.ref[0] + 1 if len(primitive.ref) == 1 else 0
        return (
            f"L({atoms[0]:3d},{atoms[1]:3d},{atoms[2]:3d},"
            f"{reference:3d},{primitive.mode:3d})"
        )
    raise ValueError(f"Unsupported primitive kind: {primitive.kind}")


def _join_terms(parts: list[tuple[float, str]]) -> str:
    text = ""
    for idx, (coeff, expr) in enumerate(parts):
        sign = "-" if coeff < 0.0 else "+"
        body = f"{abs(coeff):.8f}*{expr}"
        if idx == 0:
            text += f"-{body}" if sign == "-" else body
        else:
            text += sign + body
    return text
