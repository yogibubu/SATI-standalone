from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations, permutations
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
from matrix_numerics import (
    normalized_matrix_condition,
    numerical_matrix_rank,
    select_rank_revealing_rows,
    singular_spectrum,
    spectrum_rank,
)
from matrix_chem import linear_bend_reference_atom

from matrix_chem.local_perception import (
    LOCAL_COORDINATION_TEMPLATES,
    LOCAL_TEMPLATE_MIN_MARGIN,
    LOCAL_TEMPLATE_RMS_THRESHOLD,
    LocalCoordinationMatch,
    LocalCoordinationTemplate,
    LocalPerceptionSettings,
    infer_local_pseudogroup,
    ligand_pair_cosine,
    local_coordination_match,
    local_ligand_equivalence_classes,
    local_ligand_unit_vectors,
    nearest_cosine_class,
    ring_local_pseudogroup,
    sorted_pair_cosines,
    template_pair_cosine_classes,
)
from matrix_smith.coordinate_kernel.geometry import angle
from matrix_smith.coordinate_kernel.pipeline import b_matrix_analytic
from matrix_smith.analytic_salc import (
    cyclic_out_of_plane_coefficients,
    cyclic_out_of_plane_source_terms,
)
from matrix_smith.coordinate_kernel.primitives import Primitive, eval_primitive
from matrix_chem.topology.covalent_radii import covalent_radius
from matrix_chem.topology.elements import atomic_number
from matrix_chem.topology.pipeline import build_topology_objects
from matrix_chem.topology.ringset import RingSet
from matrix_smith.policy import LINEAR_ANGLE_DEGREES

# Fullerene cage metric experiment: inverse powers of distance are tested
# explicitly because a cage with LJ-like radial interactions is closer to
# quadratic in R^{-p} than in R.  Keep this as a single auditable policy
# switch; p=1 is the default and p=3 is the comparison protocol.
FULLERENE_DISTANCE_POWER = 1
FULLERENE_RADIAL_VARIABLE = "inverse"

from ..policy import (
    AROMATIC_LOCAL_MODEL_DIAGNOSTIC,
    CANONICAL_SALC_DIAGNOSTIC,
    MAX_NORMALIZED_SONIC_CONDITION,
    SONIC_CONSTRUCTION_POLICY,
    normalize_ring_puckering_model,
)
from ..fallback_ledger import make_fallback_event, merge_fallback_events
from ..models import FallbackEvent
from .model import (
    GICDefinition,
    _definition_coordinate_kind_counts,
    _gicforge_cartesian_from_gauin,
    _primitive_signature,
    define_gics_from_cartesian,
)


LINEAR_THRESHOLD_RAD = np.deg2rad(LINEAR_ANGLE_DEGREES)
COLLAPSED_BOND_THRESHOLD_ANGSTROM = 0.2
LOCAL_TORSION_STABILITY_THRESHOLD = 1.0e-4
# Backward-compatible public name.  The numerical policy now lives in the
# shared kernel; ORACLE is the semantic producer in the frozen contract.
LocalSALCSettings = LocalPerceptionSettings


@dataclass(frozen=True)
class GICForgePythonCoordinate:
    name: str
    block: str
    terms: tuple[tuple[float, Primitive], ...]
    type_index: int = 0
    diagnostic: str = ""
    fallback_events: tuple[FallbackEvent, ...] = ()

    @property
    def dominant_kind(self) -> str:
        return self.terms[0][1].kind if self.terms else "unknown"


@dataclass(frozen=True)
class _CoordinateBasisAudit:
    """One typed numerical snapshot used by SONIC basis selection."""

    rows: np.ndarray
    normalized_rows: np.ndarray
    rank: int | None
    condition: float | None


@dataclass(frozen=True)
class _RingCoordinateDomain:
    index: int
    atoms: tuple[int, ...]
    local_group: str
    confidence: str
    operation_count: int
    aromatic: bool
    diagnostic_suffix: str


@dataclass(frozen=True)
class GICForgePythonModel:
    atom_symbols: tuple[str, ...]
    atomic_numbers: tuple[int, ...]
    coordinates_angstrom: tuple[tuple[float, float, float], ...]
    primitive_candidates: tuple[GICForgePythonCoordinate, ...]
    coordinates: tuple[GICForgePythonCoordinate, ...]
    target_rank: int
    primitive_fallback: bool
    diagnostics: dict[str, object]
    fallback_events: tuple[FallbackEvent, ...] = ()

    def to_definition(self, *, workdir: Path | None = None) -> GICDefinition:
        primitive_basis = _primitive_basis(self.coordinates)
        row_index = {primitive: index for index, primitive in enumerate(primitive_basis)}
        u_matrix = np.zeros((len(primitive_basis), len(self.coordinates)), dtype=float)
        labels: list[str] = []
        names: list[str] = []
        for column, coordinate in enumerate(self.coordinates):
            names.append(coordinate.name)
            for coefficient, primitive in coordinate.terms:
                u_matrix[row_index[primitive], column] += float(coefficient)
            labels.append(
                f"GIC{column + 1:03d} GICForgePython {coordinate.name} "
                f"irrep=UNK {_format_terms(coordinate.terms)}"
            )
        definition = GICDefinition(
            atom_symbols=self.atom_symbols,
            atomic_numbers=self.atomic_numbers,
            reference_coordinates_angstrom=self.coordinates_angstrom,
            primitives=primitive_basis,
            u_matrix=u_matrix,
            labels=tuple(labels),
            names=tuple(names),
            irreps=tuple("UNK" for _ in names),
            point_group="UNKNOWN",
            symmetrized=False,
            symmetry_source="none",
            gaussian_input="\n".join(
                _format_readgic(name, coord.terms) for name, coord in zip(names, self.coordinates)
            )
            + "\n",
            source="gicforge-python",
            generation_workdir=str(workdir) if workdir is not None else None,
            provenance={
                "backend": "gicforge-python",
                "target_vibrational_rank": str(self.target_rank),
                "primitive_fallback": str(self.primitive_fallback).lower(),
                "svd_local": str(self.diagnostics.get("svd_local", False)).lower(),
                "local_salc": str(self.diagnostics.get("local_salc", False)).lower(),
            },
        )
        if workdir is not None:
            Path(workdir).mkdir(parents=True, exist_ok=True)
            (Path(workdir) / "gicforge_python_diagnostics.json").write_text(
                json.dumps(self.diagnostics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return definition


def _validated_topology_bonds(
    bonds: Iterable[tuple[int, int]],
    *,
    natoms: int,
) -> tuple[tuple[int, int], ...]:
    """Return a canonical zero-based copy of an ORACLE-owned bond graph."""

    canonical: set[tuple[int, int]] = set()
    for pair in bonds:
        if len(pair) != 2:
            raise ValueError("topology bonds must contain exactly two atom indices")
        first, second = (int(value) for value in pair)
        if first == second or first < 0 or second < 0 or first >= natoms or second >= natoms:
            raise ValueError("topology bond contains an invalid atom index")
        canonical.add(tuple(sorted((first, second))))
    return tuple(sorted(canonical))


def build_gicforge_python_model(
    atom_symbols: Iterable[str],
    coordinates_angstrom: np.ndarray,
    *,
    topology_bonds: Iterable[tuple[int, int]] | None = None,
    frozen_rings: tuple[tuple[int, ...], ...] | None = None,
    frozen_aromatic_atoms: frozenset[int] | None = None,
    impdih: bool = False,
    onedih: bool = True,
    svd_local: bool = False,
    local_salc: bool = False,
    xy3_torsions: bool = False,
    xy2_torsions: bool = False,
    separate_exocyclic_torsions: bool = False,
    max_linear_angle_pairs_per_center: int = 3,
    linear_threshold: float = LINEAR_THRESHOLD_RAD,
    primitive_fallback: bool = True,
    local_salc_settings: LocalSALCSettings | None = None,
    ring_puckering_model: str = "triangular_flap",
) -> GICForgePythonModel:
    # IMPDIH is retained in the low-level signature only for source
    # compatibility.  The native default is the triangular-flap chart;
    # Gaussian conversion of its analytic dihedral kernel is performed only
    # while writing a G16 input file.
    impdih = False
    model_fallback_events: list[FallbackEvent] = []
    atoms = tuple(str(atom).strip() for atom in atom_symbols)
    coords = np.asarray(coordinates_angstrom, dtype=float)
    if coords.shape != (len(atoms), 3):
        raise ValueError(f"Expected coordinate shape ({len(atoms)}, 3), got {coords.shape}")
    atomic_numbers = tuple(atomic_number(atom) for atom in atoms)
    local_settings = local_salc_settings or LocalSALCSettings()
    ring_model = normalize_ring_puckering_model(ring_puckering_model)
    graph, ringset, aromatic_atoms = _gicforge_topology_context(
        atoms,
        atomic_numbers,
        coords,
        topology_bonds=topology_bonds,
        frozen_rings=frozen_rings,
        frozen_aromatic_atoms=frozen_aromatic_atoms,
    )
    primitive_blocks = _fortran_like_primitive_blocks(
        graph,
        coords,
        atomic_numbers=atomic_numbers,
        ringset=ringset,
        aromatic_atoms=aromatic_atoms,
        impdih=impdih,
        onedih=onedih,
        svd_local=svd_local,
        local_salc=local_salc,
        xy3_torsions=xy3_torsions,
        xy2_torsions=xy2_torsions,
        separate_exocyclic_torsions=separate_exocyclic_torsions,
        max_linear_angle_pairs_per_center=max_linear_angle_pairs_per_center,
        linear_threshold=linear_threshold,
        local_salc_settings=local_settings,
        ring_puckering_model=ring_model,
    )
    target = _target_rank(coords, graph)
    candidates, coordinates, initial_events = _initial_gicforge_selection(
        tuple(coord for block in primitive_blocks for coord in block),
        graph=graph,
        ringset=ringset,
        coords=coords,
        atomic_numbers=atomic_numbers,
        target=target,
        primitive_fallback=primitive_fallback,
        impdih=impdih,
        linear_threshold=linear_threshold,
        svd_local=svd_local,
    )
    model_fallback_events.extend(initial_events)
    used_augmented_fallback = False
    if primitive_fallback and (
        len(coordinates) < target or _coordinate_b_rank(coordinates, coords) < target
    ):
        used_augmented_fallback = True
        selected_rank_before_augmentation = _coordinate_b_rank(coordinates, coords)
        fallback_blocks = _primitive_fallback_blocks(
            graph,
            coords,
            atomic_numbers=atomic_numbers,
            ringset=ringset,
            impdih=impdih,
            linear_threshold=linear_threshold,
        )
        fallback_candidates = _without_polyhedral_axis_coordinates(
            tuple(coordinate for block in fallback_blocks for coordinate in block),
            polyhedral_centers=_polyhedral_catalog_centers(candidates),
        )
        augmented_candidates = tuple(dict.fromkeys((*candidates, *fallback_candidates)))
        if len(augmented_candidates) < target:
            raise ValueError(
                "Augmented primitive candidates below vibrational rank "
                f"({len(augmented_candidates)} < {target})"
            )
        augmented_rows = _coordinate_b_matrix(augmented_candidates, coords)
        row_norms = np.linalg.norm(augmented_rows, axis=1)
        normalized_rows = np.vstack(
            [
                row / norm if norm > 1.0e-12 else row
                for row, norm in zip(augmented_rows, row_norms, strict=True)
            ]
        )
        selection = select_rank_revealing_rows(
            normalized_rows,
            target_rank=target,
            tolerance=1.0e-10,
            priorities=tuple(
                0 if _is_analytic_salc(coordinate) else 1 for coordinate in augmented_candidates
            ),
            tie_tolerance=1.0e-12,
        )
        candidates = augmented_candidates
        coordinates = tuple(augmented_candidates[index] for index in selection.indices)
        if selection.rank < target or _coordinate_b_rank(coordinates, coords) < target:
            raise ValueError(
                "GICForge Python augmented fallback did not reach vibrational rank "
                f"({len(coordinates)} coordinates for target {target})"
            )
        model_fallback_events.append(
            make_fallback_event(
                stage="SMITH_RANK_SELECTION",
                algorithm_id="AUGMENTED_PRIMITIVE_RANK_RECOVERY",
                trigger="TYPE_LOCAL_SELECTION_BELOW_TARGET_RANK",
                rank_before=selected_rank_before_augmentation,
                rank_after=selection.rank,
            )
        )
    if used_augmented_fallback:
        coordinates = _conditioned_coordinate_basis(
            candidates,
            coordinates,
            coords,
            target_rank=target,
            preserve_special=False,
        )
    elif len(coordinates) == target and _coordinate_b_rank(coordinates, coords) == target:
        original_coordinates = coordinates
        try:
            conditioned_coordinates = _conditioned_coordinate_basis(
                candidates,
                coordinates,
                coords,
                target_rank=target,
                preserve_special=True,
            )
            coordinates = (
                conditioned_coordinates
                if _cage_chart_semantics_preserved(
                    original_coordinates,
                    conditioned_coordinates,
                )
                else original_coordinates
            )
        except ValueError:
            if not primitive_fallback:
                raise
            fallback_blocks = _primitive_fallback_blocks(
                graph,
                coords,
                atomic_numbers=atomic_numbers,
                ringset=ringset,
                impdih=impdih,
                linear_threshold=linear_threshold,
            )
            fallback_candidates = _without_polyhedral_axis_coordinates(
                tuple(coordinate for block in fallback_blocks for coordinate in block),
                polyhedral_centers=_polyhedral_catalog_centers(candidates),
            )
            augmented_candidates = tuple(dict.fromkeys((*candidates, *fallback_candidates)))
            conditioned_coordinates = _conditioned_coordinate_basis(
                augmented_candidates,
                coordinates,
                coords,
                target_rank=target,
                preserve_special=True,
            )
            if _cage_chart_semantics_preserved(
                original_coordinates,
                conditioned_coordinates,
            ):
                coordinates = conditioned_coordinates
                candidates = augmented_candidates
                model_fallback_events.append(
                    make_fallback_event(
                        stage="SMITH_CONDITIONING",
                        algorithm_id="AUGMENTED_CONDITIONING_RECOVERY",
                        trigger="PRIMARY_CONDITIONING_COULD_NOT_PRESERVE_EXACT_CHART",
                        rank_before=target,
                        rank_after=target,
                    )
                )
            else:
                coordinates = original_coordinates
    diagnostics = _python_model_diagnostics(
        candidates,
        coordinates,
        coords,
        target_rank=target,
        svd_local=svd_local,
        local_salc=local_salc,
        xy3_torsions=xy3_torsions,
        xy2_torsions=xy2_torsions,
        separate_exocyclic_torsions=separate_exocyclic_torsions,
        onedih=onedih,
        max_linear_angle_pairs_per_center=max_linear_angle_pairs_per_center,
        local_salc_settings=local_settings,
        ring_puckering_model=ring_model,
    )
    return GICForgePythonModel(
        atom_symbols=atoms,
        atomic_numbers=atomic_numbers,
        coordinates_angstrom=tuple(tuple(float(value) for value in row) for row in coords),
        primitive_candidates=candidates,
        coordinates=coordinates,
        target_rank=target,
        primitive_fallback=primitive_fallback,
        diagnostics=diagnostics,
        fallback_events=merge_fallback_events(
            tuple(model_fallback_events),
            tuple(
                event
                for coordinate in coordinates
                for event in coordinate.fallback_events
            ),
        ),
    )


def _gicforge_topology_context(
    atoms: tuple[str, ...],
    atomic_numbers: tuple[int, ...],
    coords: np.ndarray,
    *,
    topology_bonds: Iterable[tuple[int, int]] | None,
    frozen_rings: tuple[tuple[int, ...], ...] | None = None,
    frozen_aromatic_atoms: frozenset[int] | None = None,
):
    """Build the frozen or perceived topology consumed by primitive generation."""

    if frozen_rings is not None:
        if topology_bonds is None or frozen_aromatic_atoms is None:
            raise ValueError("frozen ORACLE topology needs bonds, rings and aromatic atoms")
        frozen_bonds = _validated_topology_bonds(topology_bonds, natoms=len(atoms))
        adjacency = [set() for _ in atoms]
        for left, right in frozen_bonds:
            adjacency[left].add(right)
            adjacency[right].add(left)
        graph = SimpleNamespace(natoms=len(atoms), Z=np.asarray(atomic_numbers), coords=coords,
            bonds=list(frozen_bonds), adjacency=adjacency,
            neighbors=lambda atom: tuple(sorted(adjacency[int(atom)])),
            bonds_are_canonical_sorted=True, hydrogen_bridges=())
        ringset = SimpleNamespace(rings=tuple(SimpleNamespace(index=i, atoms=ring)
                                            for i, ring in enumerate(frozen_rings)))
        return graph, ringset, frozen_aromatic_atoms
    _cg, perceived_graph, perceived_ringset, _synthons, aromaticity = build_topology_objects(
        coords, np.asarray(atomic_numbers)
    )
    frozen_bonds = (
        _validated_topology_bonds(topology_bonds, natoms=len(atoms))
        if topology_bonds is not None
        else None
    )
    if frozen_bonds is None:
        graph = perceived_graph
        ringset = perceived_ringset
    else:
        adjacency = [set() for _atom in atoms]
        for first, second in frozen_bonds:
            adjacency[first].add(second)
            adjacency[second].add(first)
        graph = SimpleNamespace(
            natoms=len(atoms),
            Z=np.asarray(atomic_numbers, dtype=int),
            coords=coords,
            bonds=list(frozen_bonds),
            adjacency=adjacency,
            neighbors=lambda atom: tuple(sorted(adjacency[int(atom)])),
            bonds_are_canonical_sorted=True,
            hydrogen_bridges=(),
        )
        ringset = RingSet(graph, coords=coords)
    _validate_no_spurious_hh_contacts(coords, atomic_numbers, graph.bonds)
    if frozen_bonds is None and _remove_collapsed_bonds(graph, coords):
        ringset = RingSet(graph, coords=coords)
    return (
        graph,
        ringset,
        frozenset(int(atom) for atom in aromaticity.aromatic_atoms),
    )


def _initial_gicforge_selection(
    primitive_candidates: tuple[GICForgePythonCoordinate, ...],
    *,
    graph,
    ringset,
    coords: np.ndarray,
    atomic_numbers: tuple[int, ...],
    target: int,
    primitive_fallback: bool,
    impdih: bool,
    linear_threshold: float,
    svd_local: bool,
) -> tuple[
    tuple[GICForgePythonCoordinate, ...],
    tuple[GICForgePythonCoordinate, ...],
    tuple[FallbackEvent, ...],
]:
    """Perform primary candidate completion and exact-rank selection."""

    events: list[FallbackEvent] = []
    candidates = _without_polyhedral_axis_coordinates(primitive_candidates)
    if primitive_fallback and len(candidates) < target:
        initial_count = len(candidates)
        primitive_blocks = _primitive_fallback_blocks(
            graph,
            coords,
            atomic_numbers=atomic_numbers,
            ringset=ringset,
            impdih=impdih,
            linear_threshold=linear_threshold,
        )
        fallback_candidates = tuple(
            coordinate for block in primitive_blocks for coordinate in block
        )
        candidates = _without_polyhedral_axis_coordinates(
            tuple(dict.fromkeys((*candidates, *fallback_candidates)))
        )
        events.append(
            make_fallback_event(
                stage="SMITH_PRIMITIVE_GENERATION",
                algorithm_id="PRIMITIVE_FALLBACK_BLOCKS",
                trigger="PRIMARY_CANDIDATE_POOL_BELOW_TARGET_RANK",
                rank_before=initial_count,
                rank_after=len(candidates),
            )
        )
    if not primitive_fallback and len(candidates) < target:
        raise ValueError(
            f"GICForge Python candidates below vibrational rank ({len(candidates)} < {target})"
        )
    if len(candidates) < target:
        raise ValueError(
            f"Primitive candidates below vibrational rank ({len(candidates)} < {target})"
        )
    coordinates = _prune_type_local(
        candidates,
        coords,
        target_rank=target,
        block_pruning=svd_local,
    )
    if len(coordinates) != target:
        return candidates, coordinates, tuple(events)
    selected_condition = singular_spectrum(
        _coordinate_b_matrix(tuple(coordinates), coords),
        absolute_tolerance=0.0,
    ).condition_number
    if not np.isfinite(selected_condition) or selected_condition <= 1.0e4:
        return candidates, coordinates, tuple(events)
    candidates_for_rank = tuple(
        coordinate
        for coordinate in candidates
        if not (
            coordinate.dominant_kind == "linear_bend"
            and any(
                len(graph.adjacency[primitive.atoms[1]]) >= 3
                for _coefficient, primitive in coordinate.terms
                if primitive.kind == "linear_bend"
            )
        )
    )
    selection = select_rank_revealing_rows(
        _coordinate_b_matrix(candidates_for_rank, coords),
        target_rank=target,
        tolerance=1.0e-10,
        priorities=tuple(
            0 if coordinate.dominant_kind in {"linear_bend", "ring"} else 1
            for coordinate in candidates_for_rank
        ),
        tie_tolerance=1.0e-12,
    )
    if selection.rank == target:
        coordinates = tuple(candidates_for_rank[index] for index in selection.indices)
        events.append(
            make_fallback_event(
                stage="SMITH_RANK_SELECTION",
                algorithm_id="GLOBAL_RANK_REVEALING_RECOVERY",
                trigger="SELECTED_CHART_CONDITION_EXCEEDED_1E4",
                rank_before=target,
                rank_after=selection.rank,
                condition_before=selected_condition,
            )
        )
    return candidates, coordinates, tuple(events)


def _validate_no_spurious_hh_contacts(
    coords: np.ndarray,
    atomic_numbers: tuple[int, ...],
    bonds: Iterable[tuple[int, int]],
) -> None:
    bonded = {tuple(sorted((int(i), int(j)))) for i, j in bonds}
    contacts: list[str] = []
    for i, zi in enumerate(atomic_numbers):
        if zi != 1:
            continue
        ri = covalent_radius(zi)
        if ri is None:
            continue
        for j in range(i + 1, len(atomic_numbers)):
            if atomic_numbers[j] != 1 or (i, j) in bonded:
                continue
            rj = covalent_radius(atomic_numbers[j])
            if rj is None:
                continue
            distance = float(np.linalg.norm(coords[i] - coords[j]))
            if distance <= 1.25 * (float(ri) + float(rj)):
                contacts.append(f"{i + 1}-{j + 1} ({distance:.3f} A)")
    if contacts:
        preview = ", ".join(contacts[:8])
        extra = f"; {len(contacts) - 8} additional H-H contacts" if len(contacts) > 8 else ""
        raise ValueError(
            f"GICForge Python input topology validation failed: spurious nonbonded H-H contact {preview}{extra}"
        )


def compare_gicforge_python_to_fortran(
    atom_symbols: Iterable[str],
    coordinates_angstrom: np.ndarray,
    *,
    workdir: Path,
    executable: Path | None = None,
    impdih: bool = False,
    onedih: bool = True,
    svd_local: bool = False,
    local_salc: bool = False,
    xy3_torsions: bool = False,
    xy2_torsions: bool = False,
    ring_puckering_model: str = "triangular_flap",
) -> dict[str, object]:
    workdir = Path(workdir)
    fortran_dir = workdir / "fortran"
    python_model = build_gicforge_python_model(
        atom_symbols,
        coordinates_angstrom,
        impdih=impdih,
        onedih=onedih,
        svd_local=svd_local,
        local_salc=local_salc,
        xy3_torsions=xy3_torsions,
        xy2_torsions=xy2_torsions,
        ring_puckering_model=ring_puckering_model,
    )
    extra_keywords = []
    if impdih:
        extra_keywords.append("IMPDIH")
    if not onedih:
        extra_keywords.append("NOONEDIH")
    if svd_local:
        extra_keywords.append("LOCSVD")
    if local_salc:
        extra_keywords.append("LOCSALC")
    if xy3_torsions:
        extra_keywords.append("XY3")
    if xy2_torsions:
        extra_keywords.append("XY2")
    normalized_ring_model = normalize_ring_puckering_model(ring_puckering_model)
    if normalized_ring_model == "local_out_of_plane":
        extra_keywords.append("RINGU")
    elif normalized_ring_model == "endocyclic_dihedral":
        extra_keywords.append("RINGD")
    elif normalized_ring_model == "charm":
        extra_keywords.append("RINGH")
    fortran_definition = define_gics_from_cartesian(
        tuple(atom_symbols),
        np.asarray(coordinates_angstrom, dtype=float),
        workdir=fortran_dir,
        executable=executable,
        symmetrize=False,
        extra_keywords=tuple(extra_keywords),
    )
    raw_coords = _gicforge_cartesian_from_gauin(
        fortran_dir / "gauin", len(fortran_definition.atom_symbols)
    )
    fortran_signatures = tuple(
        _primitive_signature(primitive) for primitive in fortran_definition.primitives
    )
    python_candidates: list[tuple[str, GICDefinition]] = [
        ("input", python_model.to_definition(workdir=workdir / "python-input"))
    ]
    try:
        python_candidates.append(
            (
                "fortran-frame",
                build_gicforge_python_model(
                    atom_symbols,
                    raw_coords,
                    impdih=impdih,
                    onedih=onedih,
                    svd_local=svd_local,
                    local_salc=local_salc,
                    xy3_torsions=xy3_torsions,
                    xy2_torsions=xy2_torsions,
                    ring_puckering_model=ring_puckering_model,
                ).to_definition(workdir=workdir / "python-fortran-frame"),
            )
        )
    except Exception:
        pass
    selected_frame, python_definition = python_candidates[0]
    for candidate_frame, candidate_definition in python_candidates:
        candidate_signatures = tuple(
            _primitive_signature(primitive) for primitive in candidate_definition.primitives
        )
        if (
            len(candidate_definition.names) == len(fortran_definition.names)
            and _definition_coordinate_kind_counts(candidate_definition)
            == _definition_coordinate_kind_counts(fortran_definition)
            and candidate_signatures == fortran_signatures
        ):
            selected_frame = candidate_frame
            python_definition = candidate_definition
            break
    python_signatures = tuple(
        _primitive_signature(primitive) for primitive in python_definition.primitives
    )
    same_ordered_primitives = fortran_signatures == python_signatures
    b_max_abs_diff = None
    b_row_space_residual = None
    if len(fortran_definition.names) == len(python_definition.names):
        fortran_b = fortran_definition.u_matrix.T @ b_matrix_analytic(
            fortran_definition.primitives, raw_coords
        )
        python_b = python_definition.u_matrix.T @ b_matrix_analytic(
            python_definition.primitives,
            raw_coords,
        )
        if same_ordered_primitives and fortran_b.shape == python_b.shape:
            b_max_abs_diff = (
                float(np.max(np.abs(python_b - fortran_b))) if python_b.size else 0.0
            )
        if python_b.size:
            fortran_q = np.linalg.svd(fortran_b, full_matrices=False)[2]
            python_q = np.linalg.svd(python_b, full_matrices=False)[2]
            rank = min(
                int(np.linalg.matrix_rank(fortran_b)),
                int(np.linalg.matrix_rank(python_b)),
            )
            if rank:
                fortran_projector = fortran_q[:rank].T @ fortran_q[:rank]
                python_projector = python_q[:rank].T @ python_q[:rank]
                b_row_space_residual = float(
                    np.linalg.norm(fortran_projector - python_projector, ord=2)
                )
            else:
                b_row_space_residual = 0.0
    return {
        "passed": (
            len(python_definition.names) == len(fortran_definition.names)
            and (
                (
                    _definition_coordinate_kind_counts(python_definition)
                    == _definition_coordinate_kind_counts(fortran_definition)
                    and same_ordered_primitives
                    and b_max_abs_diff is not None
                    and b_max_abs_diff <= 1.0e-7
                )
                or (
                    b_row_space_residual is not None
                    and b_row_space_residual <= 1.0e-7
                )
            )
        ),
        "target_rank": python_model.target_rank,
        "python_gic_count": len(python_definition.names),
        "fortran_gic_count": len(fortran_definition.names),
        "python_kind_counts": _definition_coordinate_kind_counts(python_definition),
        "fortran_kind_counts": _definition_coordinate_kind_counts(fortran_definition),
        "python_primitive_count": len(python_definition.primitives),
        "fortran_primitive_count": len(fortran_definition.primitives),
        "same_ordered_primitives": same_ordered_primitives,
        "b_max_abs_diff": b_max_abs_diff,
        "b_row_space_residual": b_row_space_residual,
        "python_names": list(python_definition.names),
        "fortran_names": list(fortran_definition.names),
        "python_comparison_frame": selected_frame,
        "python_workdir": str(workdir / f"python-{selected_frame}"),
        "fortran_workdir": str(fortran_dir),
    }


def _ring_coordinate_domains(
    selected_rings: list[tuple[int, ...]],
    *,
    aromatic_atoms: frozenset[int],
    effective_atomic_numbers: tuple[float, ...],
    coords: np.ndarray,
    settings: LocalSALCSettings,
) -> tuple[_RingCoordinateDomain, ...]:
    aromatic_blocks = _aromatic_ring_block_labels(selected_rings, aromatic_atoms)
    domains: list[_RingCoordinateDomain] = []
    for ring_index, ring in enumerate(selected_rings, start=1):
        local_group, confidence, operation_count = _ring_local_pseudogroup(
            ring,
            effective_atomic_numbers=effective_atomic_numbers,
            coords=coords,
            settings=settings,
        )
        aromatic = bool(ring) and all(atom in aromatic_atoms for atom in ring)
        diagnostic_suffix = (
            f" AROMATIC_LOCAL_SALC=YES AROMATIC_BLOCK={aromatic_blocks[ring_index - 1]}"
            if aromatic
            else ""
        )
        domains.append(
            _RingCoordinateDomain(
                index=ring_index,
                atoms=ring,
                local_group=local_group,
                confidence=confidence,
                operation_count=operation_count,
                aromatic=aromatic,
                diagnostic_suffix=diagnostic_suffix,
            )
        )
    return tuple(domains)


def _ring_bend_block(
    domains: tuple[_RingCoordinateDomain, ...],
    *,
    coords: np.ndarray,
    svd_local: bool,
) -> list[GICForgePythonCoordinate]:
    coordinates: list[GICForgePythonCoordinate] = []
    claimed_sources: set[tuple[int, int, int]] = set()
    for domain in domains:
        diagnostic = (
            f"LOCAL_SALC KIND=RING_ANGLE DOMAIN=RING:{domain.index} "
            f"GROUP={domain.local_group} CONFIDENCE={domain.confidence} "
            f"OPERATIONS={domain.operation_count}{domain.diagnostic_suffix}"
        )
        builder = _cyclic_svd_coordinates if svd_local else _cyclic_coordinates
        kwargs = {"coords": coords} if svd_local else {}
        ring_coordinates = builder(
            domain.atoms,
            valence_angle=True,
            prefix="RDef",
            start=len(coordinates) + 1,
            diagnostic=diagnostic,
            **kwargs,
        )
        ring_coordinates = _condensed_ring_bend_complement(
            ring_coordinates,
            ring=domain.atoms,
            claimed_sources=claimed_sources,
        )
        coordinates.extend(ring_coordinates)
        claimed_sources.update(
            _canonical_angle_source(primitive)
            for coordinate in ring_coordinates
            for _coefficient, primitive in coordinate.terms
            if primitive.kind == "angle"
        )
    return coordinates


def _canonical_angle_source(primitive: Primitive) -> tuple[int, int, int]:
    """Return an orientation-independent key for one valence angle."""

    first, center, second = primitive.atoms
    left, right = sorted((int(first), int(second)))
    return left, int(center), right


def _condensed_ring_bend_complement(
    coordinates: list[GICForgePythonCoordinate],
    *,
    ring: tuple[int, ...],
    claimed_sources: set[tuple[int, int, int]],
) -> list[GICForgePythonCoordinate]:
    """Remove duplicated internal-angle sources from condensed-ring modes.

    Minimum-cycle-basis rings that share a path of at least three bonds also
    share one or more complete valence-angle primitives.  Building an
    independent Fourier chart for every cycle then places the same primitive
    in multiple protected RDef blocks and can make the molecular SONIC chart
    nearly singular.  Each shared source is therefore owned by the first
    (deterministically ordered) cycle.  Later cycles construct their N-3
    modes in the topology-only complement whenever that complement has full
    dimension.  The QR is confined to one small ring and never touches the
    Cartesian molecular matrix.

    If an unusually highly condensed cycle has too few exclusive sources,
    the minimum number of shared sources is restored in ring order.  This
    preserves the exact N-3 dimension while still minimizing overlap.
    """

    if not coordinates or not claimed_sources:
        return coordinates

    primitives = _cyclic_primitives_legacy_order(ring, valence_angle=True)
    source_keys = tuple(_canonical_angle_source(primitive) for primitive in primitives)
    shared_indices = [
        index for index, source_key in enumerate(source_keys) if source_key in claimed_sources
    ]
    if not shared_indices:
        return coordinates

    source_index = {source_key: index for index, source_key in enumerate(source_keys)}
    coefficients = np.zeros((len(primitives), len(coordinates)), dtype=float)
    for mode, coordinate in enumerate(coordinates):
        for coefficient, primitive in coordinate.terms:
            index = source_index.get(_canonical_angle_source(primitive))
            if index is not None:
                coefficients[index, mode] += float(coefficient)

    allowed = [index for index in range(len(primitives)) if index not in shared_indices]
    required_rank = len(coordinates)
    restored = 0
    for index in shared_indices:
        if np.linalg.matrix_rank(coefficients[allowed], tol=1.0e-12) >= required_rank:
            break
        allowed.append(index)
        restored += 1
    allowed.sort()
    restricted = coefficients[allowed]
    if np.linalg.matrix_rank(restricted, tol=1.0e-12) < required_rank:
        return coordinates

    orthonormal, _upper = np.linalg.qr(restricted, mode="reduced")
    orthonormal = orthonormal[:, :required_rank]
    # Fix QR's arbitrary column signs against the corresponding projected
    # Fourier modes so serialized definitions remain deterministic.
    for mode in range(required_rank):
        if float(np.dot(orthonormal[:, mode], restricted[:, mode])) < 0.0:
            orthonormal[:, mode] *= -1.0

    shared_total = len(shared_indices)
    excluded = shared_total - restored
    rebuilt: list[GICForgePythonCoordinate] = []
    for mode, coordinate in enumerate(coordinates):
        terms = tuple(
            (float(orthonormal[row, mode]), primitives[index])
            for row, index in enumerate(allowed)
            if abs(float(orthonormal[row, mode])) > 1.0e-12
        )
        rebuilt.append(
            replace(
                coordinate,
                terms=terms,
                diagnostic=(
                    f"{coordinate.diagnostic} BASIS=CONDENSED_RING_COMPLEMENT "
                    f"SHARED_ANGLE_SOURCES={shared_total} "
                    f"EXCLUDED_SHARED_SOURCES={excluded} "
                    f"RESTORED_SHARED_SOURCES={restored}"
                ).strip(),
            )
        )
    return rebuilt


def _ring_puckering_block(
    domains: tuple[_RingCoordinateDomain, ...],
    *,
    coords: np.ndarray,
    atomic_numbers: tuple[int, ...],
    requested_model: str,
) -> list[GICForgePythonCoordinate]:
    coordinates: list[GICForgePythonCoordinate] = []
    for domain in domains:
        diagnostic = (
            f"LOCAL_SALC KIND=RING_TORSION DOMAIN=RING:{domain.index} "
            f"GROUP={domain.local_group} CONFIDENCE={domain.confidence} "
            f"OPERATIONS={domain.operation_count}{domain.diagnostic_suffix}"
        )
        # Minimum optimization uses the discrete ORACLE aromaticity policy;
        # the geometry is never reclassified during a run.  Explicit models
        # remain available to TS/exploration callers through their dedicated
        # entry points.
        effective_model = SONIC_CONSTRUCTION_POLICY.discrete_minimum_ring_model(
            aromatic=domain.aromatic
        )
        aromatic_default = domain.aromatic and effective_model != requested_model
        if effective_model == "charm":
            ring_coordinates = _cyclic_charm_coordinates(
                domain.atoms,
                coords=coords,
                prefix="RPck",
                start=len(coordinates) + 1,
                diagnostic=diagnostic.replace("KIND=RING_TORSION", "KIND=RING_CHARM"),
            )
        elif effective_model == "local_out_of_plane":
            ring_kind = (
                "KIND=AROMATIC_BLOCK_OUT_OF_PLANE"
                if aromatic_default
                else "KIND=RING_OUT_OF_PLANE"
            )
            model_suffix = f" {AROMATIC_LOCAL_MODEL_DIAGNOSTIC}" if aromatic_default else ""
            ring_coordinates = _cyclic_out_of_plane_coordinates(
                domain.atoms,
                coords=coords,
                prefix="RPck",
                start=len(coordinates) + 1,
                diagnostic=diagnostic.replace("KIND=RING_TORSION", ring_kind) + model_suffix,
            )
        elif effective_model == "triangular_flap":
            ring_coordinates = _cyclic_triangular_flap_coordinates(
                domain.atoms,
                coords=coords,
                prefix="RPck",
                start=len(coordinates) + 1,
                diagnostic=diagnostic.replace(
                    "KIND=RING_TORSION",
                    "KIND=RING_TRIANGULAR_FLAP",
                ),
            )
        else:
            ring_coordinates = _legacy_ring_puckering_coordinates(
                domain,
                coords=coords,
                atomic_numbers=atomic_numbers,
                start=len(coordinates) + 1,
                diagnostic=diagnostic,
            )
        if not ring_coordinates and len(domain.atoms) > 3:
            raise ValueError(
                f"cannot construct {requested_model} coordinates for ring "
                f"{domain.index} with atoms {tuple(atom + 1 for atom in domain.atoms)}"
            )
        coordinates.extend(ring_coordinates)
    return coordinates


def _legacy_ring_puckering_coordinates(
    domain: _RingCoordinateDomain,
    *,
    coords: np.ndarray,
    atomic_numbers: tuple[int, ...],
    start: int,
    diagnostic: str,
) -> list[GICForgePythonCoordinate]:
    coordinates = _cyclic_coordinates(
        domain.atoms,
        valence_angle=False,
        prefix="RPck",
        start=start,
        coords=coords,
        atomic_numbers=atomic_numbers,
        diagnostic=(
            f"{diagnostic} MODEL=ENDOCYCLIC_DIHEDRAL STATUS=LEGACY_DEPRECATED"
        ),
    )
    expected_rank = len(domain.atoms) - 3
    legacy_rank = _coordinate_b_rank(tuple(coordinates), coords)
    if legacy_rank == expected_rank:
        return coordinates
    warnings.warn(
        "legacy endocyclic-dihedral ring coordinates are rank deficient "
        f"for ring {domain.index} ({legacy_rank}/{expected_rank}); selecting "
        "the triangular-flap chart for this new construction",
        RuntimeWarning,
        stacklevel=3,
    )
    source = (
        f"{diagnostic.replace('KIND=RING_TORSION', 'KIND=RING_TRIANGULAR_FLAP')} "
        "REQUESTED_MODEL=ENDOCYCLIC_DIHEDRAL "
        f"REQUESTED_RANK={legacy_rank}/{expected_rank} "
        "STATUS=LEGACY_RANK_DEFICIENT FALLBACK=TRIANGULAR_FLAP "
        "RESTART_POLICY=UNCHANGED"
    )
    event = make_fallback_event(
        stage="SMITH_RING_CHART",
        algorithm_id="TRIANGULAR_FLAP",
        trigger="LEGACY_ENDOCYCLIC_DIHEDRAL_RANK_DEFICIENT",
        domain=f"RING:{domain.index}",
        macrofamily="RING_PUCKER_COMPONENT",
        rank_before=legacy_rank,
        rank_after=expected_rank,
        source=source,
    )
    return [
        replace(coordinate, fallback_events=(event,))
        for coordinate in _cyclic_triangular_flap_coordinates(
            domain.atoms,
            coords=coords,
            prefix="RPck",
            start=start,
            diagnostic=source,
        )
    ]


def _fortran_like_primitive_blocks(
    graph,
    coords: np.ndarray,
    *,
    atomic_numbers: tuple[int, ...],
    ringset,
    aromatic_atoms: frozenset[int],
    impdih: bool,
    onedih: bool,
    svd_local: bool,
    local_salc: bool,
    xy3_torsions: bool,
    xy2_torsions: bool,
    separate_exocyclic_torsions: bool,
    max_linear_angle_pairs_per_center: int,
    linear_threshold: float,
    local_salc_settings: LocalSALCSettings,
    ring_puckering_model: str,
):
    bond_primitives: list[GICForgePythonCoordinate] = []
    bends: list[GICForgePythonCoordinate] = []
    linears: list[GICForgePythonCoordinate] = []
    neighbors = [sorted(graph.adjacency[index]) for index in range(graph.natoms)]
    effective_atomic_numbers = _effective_atomic_numbers(graph, coords, atomic_numbers, neighbors)
    selected_rings = _minimum_cycle_basis(
        graph,
        ringset,
        effective_atomic_numbers=effective_atomic_numbers,
        neighbors=neighbors,
    )
    from matrix_smith.primitive_generation import (
        _adjacency,
        _complete_single_missing_fullerene_face,
        _is_fullerene_cage_topology,
    )

    one_based_bonds = tuple(
        (int(left) + 1, int(right) + 1) for left, right in graph.bonds
    )
    one_based_rings = tuple(
        (index, tuple(int(atom) + 1 for atom in ring))
        for index, ring in enumerate(selected_rings, start=1)
    )
    one_based_adjacency = _adjacency(one_based_bonds, natoms=graph.natoms)
    one_based_rings = _complete_single_missing_fullerene_face(
        one_based_bonds,
        rings=one_based_rings,
        adjacency=one_based_adjacency,
        atom_symbols=tuple("C" if number == 6 else str(number) for number in atomic_numbers),
        natoms=graph.natoms,
    )
    fullerene_cage = _is_fullerene_cage_topology(
        one_based_bonds,
        rings=one_based_rings,
        adjacency=one_based_adjacency,
        atom_symbols=tuple("C" if number == 6 else str(number) for number in atomic_numbers),
        natoms=graph.natoms,
    )
    if fullerene_cage:
        selected_rings = tuple(
            tuple(atom - 1 for atom in ring) for _index, ring in one_based_rings
        )
    atom_ring = _atom_ring_map_from_rings(selected_rings, graph.natoms)
    ring_counts = _atom_selected_ring_counts(selected_rings, graph.natoms)
    ring_bonds = {
        tuple(sorted((ring[i], ring[(i + 1) % len(ring)])))
        for ring in selected_rings
        for i in range(len(ring))
    }
    bridge_bonds = _bridge_bonds(selected_rings)
    xy2_domains: dict[int, tuple[int, tuple[int, int]]] = {}

    for center in range(graph.natoms):
        neigh = neighbors[center]
        for first in neigh:
            if first < center:
                continue
            bond_primitives.append(
                _primitive_coordinate(
                    "Stre",
                    len(bond_primitives) + 1,
                    Primitive("bond", (center, first)),
                )
            )
        if fullerene_cage:
            continue
        # The analytic C3v angular SALCs and the optional collective XY3
        # torsion are independent choices.  Always use the existing WXY3
        # angular chart when ORACLE recognizes Z-X(Y)3; ``xy3_torsions`` only
        # controls the torsion family below.  Coupling these switches forced
        # default builds through a geometry-fitted tetrahedral angle basis,
        # which need not remain independent as the local geometry evolves.
        xy3_domain = _xy3_angle_domain(
            center,
            neigh,
            atomic_numbers=atomic_numbers,
            effective_atomic_numbers=effective_atomic_numbers,
            neighbors=neighbors,
            atom_ring=atom_ring,
        )
        if xy3_domain is not None:
            z_atom, y_atoms = xy3_domain
            frozen = {
                atom: atom_ring[atom] != 0 and atom_ring[center] != 0 for atom in (z_atom, *y_atoms)
            }
            bends.extend(
                _wxy3_coordinates(
                    center,
                    z_atom,
                    y_atoms[0],
                    y_atoms[1],
                    y_atoms[2],
                    frozen=frozen,
                    start=len(bends) + 1,
                )
            )
            continue
        xy2_domain = (
            _xy2_angle_domain(
                center,
                neigh,
                atomic_numbers=atomic_numbers,
                effective_atomic_numbers=effective_atomic_numbers,
                neighbors=neighbors,
                atom_ring=atom_ring,
            )
            if xy2_torsions
            else None
        )
        if xy2_domain is not None:
            z_atom, y_atoms = xy2_domain
            xy2_domains[center] = xy2_domain
            bends.extend(
                _xy2_angle_coordinates(
                    center,
                    z_atom,
                    y_atoms[0],
                    y_atoms[1],
                    start=len(bends) + 1,
                )
            )
            continue
        template_match = (
            _local_coordination_match(
                center,
                neigh,
                coords=coords,
                max_rms_cosine_error=local_salc_settings.template_rms_threshold,
                min_score_margin=local_salc_settings.template_min_margin,
            )
            if len(neigh) >= 4
            else None
        )
        use_template_catalog = (
            template_match is not None
            and template_match.template is not None
            and (
                atom_ring[center] == 0
                or len(neigh) >= 5
                or template_match.template.name == "SQUARE_PLANAR"
            )
        )
        if use_template_catalog:
            template_coordinates = _template_angle_salc_coordinates(
                center,
                neigh,
                template=template_match.template,
                match=template_match,
                coords=coords,
                settings=local_salc_settings,
                start=len(bends) + 1,
            )
            if template_coordinates is not None:
                bends.extend(template_coordinates)
                continue
        if local_salc and len(neigh) > 1:
            exo_primitives, exo_linears = _exocyclic_angle_primitives(
                center,
                neigh,
                selected_rings=selected_rings,
                coords=coords,
                linear_threshold=linear_threshold,
            )
            for diagnostic, primitives in _local_angle_group_records(
                center,
                neigh,
                exo_primitives,
                effective_atomic_numbers=effective_atomic_numbers,
                coords=coords,
                settings=local_salc_settings,
            ):
                use_geometry_svd_fallback = len(neigh) >= 5 or (
                    len(neigh) == 4 and atom_ring[center] == 0
                )
                fallback_source = (
                    f"{diagnostic} CANONICAL_CATALOG=NO FALLBACK=ACTUAL_GEOMETRY_SVD"
                )
                builder = (
                    _svd_local_coordinates if use_geometry_svd_fallback else _local_salc_coordinates
                )
                bends.extend(
                    builder(
                        list(primitives),
                        prefix="XAng",
                        start=len(bends) + 1,
                        kind_type_index=0,
                        diagnostic=(
                            fallback_source
                            if use_geometry_svd_fallback
                            else diagnostic
                        ),
                        **(
                            {
                                "coords": coords,
                                "fallback_events": (
                                    make_fallback_event(
                                        stage="SMITH_LOCAL_SALC",
                                        algorithm_id="ACTUAL_GEOMETRY_SVD",
                                        trigger="NO_CANONICAL_HIGH_COORDINATION_CATALOG",
                                        domain=f"CENTER:{center + 1}",
                                        macrofamily="BEND",
                                        source=fallback_source,
                                    ),
                                ),
                            }
                            if use_geometry_svd_fallback
                            else {}
                        ),
                    )
                )
            exo_linears = (
                exo_linears[: max(0, max_linear_angle_pairs_per_center)] if len(neigh) == 2 else []
            )
            for primitive in exo_linears:
                linears.append(_primitive_coordinate("LAng", len(linears) + 1, primitive))
                linears.append(
                    _primitive_coordinate(
                        "LAng",
                        len(linears) + 1,
                        Primitive(
                            "linear_bend",
                            primitive.atoms,
                            mode=-2,
                            ref=primitive.ref,
                        ),
                    )
                )
        elif svd_local and len(neigh) > 1:
            exo_primitives, exo_linears = _exocyclic_angle_primitives(
                center,
                neigh,
                selected_rings=selected_rings,
                coords=coords,
                linear_threshold=linear_threshold,
            )
            if exo_primitives:
                bends.extend(
                    _svd_local_coordinates(
                        exo_primitives,
                        coords=coords,
                        prefix="XAng",
                        start=len(bends) + 1,
                        kind_type_index=0,
                    )
                )
            exo_linears = (
                exo_linears[: max(0, max_linear_angle_pairs_per_center)] if len(neigh) == 2 else []
            )
            for primitive in exo_linears:
                linears.append(_primitive_coordinate("LAng", len(linears) + 1, primitive))
                linears.append(
                    _primitive_coordinate(
                        "LAng",
                        len(linears) + 1,
                        Primitive(
                            "linear_bend",
                            primitive.atoms,
                            mode=-2,
                            ref=primitive.ref,
                        ),
                    )
                )
        elif len(neigh) == 3:
            bends.extend(
                _c2v3_angle_coordinates(
                    center,
                    neigh,
                    atomic_numbers=atomic_numbers,
                    effective_atomic_numbers=effective_atomic_numbers,
                    neighbors=neighbors,
                    atom_ring=atom_ring,
                    coords=coords,
                    start=len(bends) + 1,
                )
            )
        elif len(neigh) > 1:
            if len(neigh) == 2 and atom_ring[center] != 0:
                continue
            if len(neigh) == 4 and _is_spiro_center(center, neigh, ring_counts):
                bends.extend(
                    _spiro_angle_coordinates(
                        center,
                        neigh,
                        selected_rings=selected_rings,
                        start=len(bends) + 1,
                    )
                )
                continue
            if len(neigh) == 4 and not _has_linear_pair(center, neigh, coords, linear_threshold):
                four_atom_coordinates = _four_atom_angle_coordinates(
                    center,
                    neigh,
                    atomic_numbers=atomic_numbers,
                    effective_atomic_numbers=effective_atomic_numbers,
                    neighbors=neighbors,
                    atom_ring=atom_ring,
                    coords=coords,
                    force_xy3_salc=False,
                    start=len(bends) + 1,
                )
                bends.extend(four_atom_coordinates)
                continue
            if len(neigh) > 4:
                high_bends, high_linears = _high_coord_angle_coordinates(
                    center,
                    neigh,
                    effective_atomic_numbers=effective_atomic_numbers,
                    coords=coords,
                    linear_threshold=linear_threshold,
                    angle_start=len(bends) + 1,
                    linear_start=len(linears) + 1,
                    settings=local_salc_settings,
                )
                bends.extend(high_bends)
                linears.extend(high_linears)
                continue
            for ib, first_angle in enumerate(neigh[:-1]):
                for second_angle in neigh[ib + 1 :]:
                    value = angle(first_angle, center, second_angle, coords)
                    left, right = sorted((first_angle, second_angle))
                    if value < linear_threshold:
                        bends.append(
                            _primitive_coordinate(
                                "Bend", len(bends) + 1, Primitive("angle", (left, center, right))
                            )
                        )
                    else:
                        if len(neigh) >= 3 or _has_redundant_linear_pairs(center, neigh, coords):
                            continue
                        linears.append(
                            _primitive_coordinate(
                                "LAng",
                                len(linears) + 1,
                                _linear_bend_primitive((left, center, right), coords, mode=-1),
                            )
                        )
                        linears.append(
                            _primitive_coordinate(
                                "LAng",
                                len(linears) + 1,
                                _linear_bend_primitive((left, center, right), coords, mode=-2),
                            )
                        )

    ring_domains = _ring_coordinate_domains(
        selected_rings,
        aromatic_atoms=aromatic_atoms,
        effective_atomic_numbers=effective_atomic_numbers,
        coords=coords,
        settings=local_salc_settings,
    )
    if fullerene_cage:
        bonds = _bond_length_coordinates(
            bond_primitives,
            effective_atomic_numbers=effective_atomic_numbers,
            coords=coords,
            neighbors=neighbors,
            selected_rings=selected_rings,
            local_salc=local_salc,
            settings=local_salc_settings,
        )
        # Triangulate the polyhedral surface intrinsically.  Together with
        # all cage edges, the n-3 diagonals of every n-membered face give
        # exactly 3*N-6 distances by Euler's formula.  Unlike independent
        # face flaps, these rows are homogeneous, non-periodic and constrain
        # the shared surface metric rather than treating each face as an
        # isolated ring.
        face_diagonals: list[GICForgePythonCoordinate] = []
        for domain in ring_domains:
            for _parent, left, right, _child in _triangular_flap_stencil(domain.atoms):
                face_diagonals.append(
                    GICForgePythonCoordinate(
                        name=f"FCDg{len(face_diagonals) + 1:04d}",
                        block="FCDg",
                        terms=(
                            (
                                1.0,
                                Primitive(
                                    "inverse_cubic_bond"
                                    if FULLERENE_RADIAL_VARIABLE == "inverse_cubic"
                                    else "exponential_bond"
                                    if FULLERENE_RADIAL_VARIABLE == "exponential"
                                    else "bond"
                                    if FULLERENE_RADIAL_VARIABLE == "ordinary"
                                    else "inverse_bond",
                                    (left, right),
                                ),
                            ),
                        ),
                        diagnostic=(
                            f"LOCAL_SALC KIND=FULLERENE_FACE_DIAGONAL "
                            f"DOMAIN=RING:{domain.index} GROUP={domain.local_group} "
                            f"CONFIDENCE={domain.confidence} "
                            f"OPERATIONS={domain.operation_count}{domain.diagnostic_suffix} "
                            "MODEL=BALANCED_FACE_TRIANGULATION "
                            "FULLERENE_CAGE_FACE_DIAGONAL=YES "
                            "BASIS=EDGE_DISTANCE_PLUS_FACE_DIAGONAL "
                            f"RADIAL_VARIABLE={'EXP(-R)' if FULLERENE_RADIAL_VARIABLE == 'exponential' else 'R^-3' if FULLERENE_RADIAL_VARIABLE == 'inverse_cubic' else 'R' if FULLERENE_RADIAL_VARIABLE == 'ordinary' else 'R^-1'}"
                        ),
                    )
                )
        return bonds, face_diagonals
    ring_bends = _ring_bend_block(
        ring_domains,
        coords=coords,
        svd_local=svd_local,
    )

    # Ring deformations are the independent coordinates that enforce cyclic
    # closure.  Keep them ahead of exocyclic angle complements during the
    # rank-revealing prune; otherwise a substituted C1 ring can lose every
    # explicit ring-deformation coordinate merely because its local primitive
    # angles were enumerated first.
    xy3_bends = [coordinate for coordinate in bends if "KIND=XY3_ANGLE" in coordinate.diagnostic]
    xy2_bends = [coordinate for coordinate in bends if "KIND=XY2_ANGLE" in coordinate.diagnostic]
    xy2_out_of_plane = [
        GICForgePythonCoordinate(
            name=f"X2OP{index:04d}",
            block="X2OP",
            type_index=2,
            # Gaussian U(center,plane1,plane2,out): the equivalent Y pair
            # defines the plane and the distinct Z ligand is displaced.  The
            # former (center,Z,Y1,Y2) ordering selected one Y as ``out`` and
            # its analytic Wilson row was not covariant away from planarity.
            terms=((1.0, Primitive("out_of_plane", (center, *y_atoms, z_atom))),),
            diagnostic=(
                f"LOCAL_SALC KIND=XY2_OUT_OF_PLANE DOMAIN=CENTER:{center + 1} "
                f"Z={z_atom + 1} Y={y_atoms[0] + 1},{y_atoms[1] + 1} "
                "OWNERSHIP=XY2 NORMALIZATION=ANALYTIC"
            ),
        )
        for index, (center, (z_atom, y_atoms)) in enumerate(sorted(xy2_domains.items()), start=1)
    ]
    ordinary_bends = [
        coordinate
        for coordinate in bends
        if "KIND=XY3_ANGLE" not in coordinate.diagnostic
        and "KIND=XY2_ANGLE" not in coordinate.diagnostic
    ]
    # Domain ownership order: stretches are emitted first by the returned
    # block order; protected ring and XY3 coordinates precede all remaining
    # ordinary bends so the rank reduction never decides ownership by chance.
    special_bends = ring_bends + xy3_bends + xy2_bends + xy2_out_of_plane

    special_torsions, ordinary_torsions = _fortran_like_torsion_blocks(
        bond_primitives,
        ring_domains=ring_domains,
        bridge_bonds=bridge_bonds,
        ring_bonds=ring_bonds,
        selected_rings=selected_rings,
        neighbors=neighbors,
        atomic_numbers=atomic_numbers,
        effective_atomic_numbers=effective_atomic_numbers,
        atom_ring=atom_ring,
        ring_counts=ring_counts,
        coords=coords,
        linear_threshold=linear_threshold,
        ring_puckering_model=ring_puckering_model,
        xy3_torsions=xy3_torsions,
        xy2_torsions=xy2_torsions,
        separate_exocyclic_torsions=separate_exocyclic_torsions,
        local_salc=local_salc,
        onedih=onedih,
    )
    oops = _fortran_like_out_of_plane_block(
        graph.natoms,
        neighbors=neighbors,
        xy2_domains=xy2_domains,
        atom_ring=atom_ring,
        ring_bonds=ring_bonds,
        atomic_numbers=atomic_numbers,
        effective_atomic_numbers=effective_atomic_numbers,
        coords=coords,
        impdih=impdih,
    )

    bonds = _bond_length_coordinates(
        bond_primitives,
        effective_atomic_numbers=effective_atomic_numbers,
        coords=coords,
        neighbors=neighbors,
        selected_rings=selected_rings,
        local_salc=local_salc,
        settings=local_salc_settings,
    )
    assembled = (
        bonds,
        special_bends,
        special_torsions,
        linears,
        oops,
        ordinary_bends,
        ordinary_torsions,
    )
    ordinary_torsions.extend(
        _linear_case_completeness_torsions(
            tuple(coordinate for block in assembled for coordinate in block),
            graph,
            coords,
            neighbors=neighbors,
            linear_threshold=linear_threshold,
            start=len(ordinary_torsions) + 1,
            excluded_central_bonds=frozenset(ring_bonds),
        )
    )
    # All protected/special domains precede the residual standard chart.
    return (
        bonds,
        special_bends,
        special_torsions,
        linears,
        oops,
        ordinary_bends,
        ordinary_torsions,
    )


def _fortran_like_torsion_blocks(
    bond_primitives,
    *,
    ring_domains,
    bridge_bonds,
    ring_bonds,
    selected_rings,
    neighbors,
    atomic_numbers,
    effective_atomic_numbers,
    atom_ring,
    ring_counts,
    coords,
    linear_threshold,
    ring_puckering_model,
    xy3_torsions,
    xy2_torsions,
    separate_exocyclic_torsions,
    local_salc,
    onedih,
):
    """Build ring-protected and ordinary torsion blocks."""

    collective_xy3 = []
    collective_xy2 = []
    ordinary = []
    for bond in bond_primitives:
        _coefficient, primitive = bond.terms[0]
        center, right = primitive.atoms
        index = len(collective_xy3) + len(collective_xy2) + len(ordinary) + 1
        if tuple(sorted((center, right))) in bridge_bonds:
            butterfly = _butterfly_coordinate(
                center,
                right,
                neighbors=neighbors,
                atom_ring=atom_ring,
                selected_rings=selected_rings,
                coords=coords,
                linear_threshold=linear_threshold,
                index=index,
            )
            if butterfly is not None:
                ordinary.append(butterfly)
            continue
        if tuple(sorted((center, right))) in ring_bonds:
            continue
        if len(neighbors[center]) == 1 or len(neighbors[right]) == 1:
            continue
        common = {
            "neighbors": neighbors,
            "atomic_numbers": atomic_numbers,
            "effective_atomic_numbers": effective_atomic_numbers,
            "atom_ring": atom_ring,
            "ring_counts": ring_counts,
            "coords": coords,
            "linear_threshold": linear_threshold,
            "index": index,
        }
        if xy3_torsions and not separate_exocyclic_torsions:
            coordinate = _xy3_collective_torsion_coordinate(center, right, **common)
            if coordinate is not None:
                collective_xy3.append(coordinate)
                continue
        if xy2_torsions and not separate_exocyclic_torsions:
            coordinate = _xy2_collective_torsion_coordinate(center, right, **common)
            if coordinate is not None:
                collective_xy2.append(coordinate)
                continue
        factory = (
            _single_dihedral_torsion_coordinate
            if separate_exocyclic_torsions
            else (
                _local_salc_torsion_coordinate
                if local_salc
                else (_onedih_torsion_coordinate if onedih else _torsion_coordinate)
            )
        )
        coordinate = factory(center, right, **common)
        if coordinate is not None:
            ordinary.append(coordinate)
    ring_torsions = _ring_puckering_block(
        ring_domains,
        coords=coords,
        atomic_numbers=atomic_numbers,
        requested_model=ring_puckering_model,
    )
    return ring_torsions + collective_xy3 + collective_xy2, ordinary


def _fortran_like_out_of_plane_block(
    natoms,
    *,
    neighbors,
    xy2_domains,
    atom_ring,
    ring_bonds,
    atomic_numbers,
    effective_atomic_numbers,
    coords,
    impdih,
):
    """Build the tricoordinate out-of-plane block."""

    output = []
    prefix = "ImpD" if impdih else "OuPl"
    for center in range(natoms):
        neigh = neighbors[center]
        if len(neigh) != 3 or center in xy2_domains:
            continue
        first, second, third = neigh
        if impdih:
            primitive = Primitive("dihedral", (first, center, third, second))
            output.append(_primitive_coordinate(prefix, len(output) + 1, primitive))
            continue
        endocyclic = tuple(
            atom for atom in neigh
            if tuple(sorted((center, atom))) in ring_bonds
        )
        if atom_ring[center] != 0 and len(endocyclic) == 3:
            continue
        if atom_ring[center] != 0 and len(endocyclic) == 2:
            moving = next(atom for atom in neigh if atom not in endocyclic)
            primitive = Primitive("out_of_plane", (center, *endocyclic, moving))
            output.append(_primitive_coordinate(prefix, len(output) + 1, primitive))
            continue
        atom_orders = _tricoordinate_out_of_plane_atom_orders(
            center,
            (first, second, third),
            atomic_numbers=atomic_numbers,
            effective_atomic_numbers=effective_atomic_numbers,
            neighbors=neighbors,
            atom_ring=atom_ring,
            coords=coords,
        )
        primitives = tuple(Primitive("out_of_plane", atoms) for atoms in atom_orders)
        if len(primitives) == 1:
            output.append(_primitive_coordinate(prefix, len(output) + 1, primitives[0]))
            continue
        coefficient = 1.0 / np.sqrt(float(len(primitives)))
        output.append(
            GICForgePythonCoordinate(
                name=f"{prefix}{len(output) + 1:04d}",
                block=prefix,
                type_index=2,
                terms=tuple((coefficient, primitive) for primitive in primitives),
                diagnostic=(
                    f"LOCAL_SALC KIND=XY3_OUT_OF_PLANE DOMAIN=CENTER:{center + 1} "
                    "MODEL=C3V_LIKE_NORMALIZED_CYCLIC_MEAN NORMALIZATION=ANALYTIC"
                ),
            )
        )
    return output


def _primitive_fallback_blocks(
    graph,
    coords: np.ndarray,
    *,
    atomic_numbers: tuple[int, ...],
    ringset,
    impdih: bool,
    linear_threshold: float,
):
    bond_primitives: list[GICForgePythonCoordinate] = []
    bends: list[GICForgePythonCoordinate] = []
    linears: list[GICForgePythonCoordinate] = []
    torsions: list[GICForgePythonCoordinate] = []
    oops: list[GICForgePythonCoordinate] = []
    neighbors = [sorted(graph.adjacency[index]) for index in range(graph.natoms)]
    effective_atomic_numbers = _effective_atomic_numbers(graph, coords, atomic_numbers, neighbors)
    selected_rings = _minimum_cycle_basis(
        graph,
        ringset,
        effective_atomic_numbers=effective_atomic_numbers,
        neighbors=neighbors,
    )
    atom_ring = _atom_ring_map_from_rings(selected_rings, graph.natoms)
    ring_bonds = {
        tuple(sorted((ring[index], ring[(index + 1) % len(ring)])))
        for ring in selected_rings
        for index in range(len(ring))
    }

    for center in range(graph.natoms):
        neigh = neighbors[center]
        for first in neigh:
            if first < center:
                continue
            bond_primitives.append(
                _primitive_coordinate(
                    "Stre",
                    len(bond_primitives) + 1,
                    Primitive("bond", (center, first)),
                )
            )
        for ib, first_angle in enumerate(neigh[:-1]):
            for second_angle in neigh[ib + 1 :]:
                left, right = sorted((first_angle, second_angle))
                primitive = Primitive("angle", (left, center, right))
                if angle(first_angle, center, second_angle, coords) < linear_threshold:
                    bends.append(_primitive_coordinate("Bend", len(bends) + 1, primitive))
                else:
                    # The fallback pool must obey the same recognized-template
                    # policy as the primary construction.  Otherwise rank
                    # completion can reintroduce redundant trans bends that
                    # were intentionally suppressed for linear-pair templates.
                    if len(neigh) >= 3 or _has_redundant_linear_pairs(center, neigh, coords):
                        continue
                    linears.append(
                        _primitive_coordinate(
                            "LAng",
                            len(linears) + 1,
                            Primitive(
                                "linear_bend",
                                primitive.atoms,
                                mode=-1,
                                ref=primitive.ref,
                            ),
                        )
                    )
                    linears.append(
                        _primitive_coordinate(
                            "LAng",
                            len(linears) + 1,
                            Primitive(
                                "linear_bend",
                                primitive.atoms,
                                mode=-2,
                                ref=primitive.ref,
                            ),
                        )
                    )

    for bond in bond_primitives:
        _coef, primitive = bond.terms[0]
        center, right = primitive.atoms
        # Ring closure is represented by the dedicated RDef/RPck (or
        # cage-specific) chart.  A primitive fallback may complete the
        # remaining molecular rank, but it must never reintroduce a torsion
        # whose central bond belongs to an explicit ring.
        if tuple(sorted((center, right))) in ring_bonds:
            continue
        if len(neighbors[center]) == 1 or len(neighbors[right]) == 1:
            continue
        for left in neighbors[center]:
            if left == right:
                continue
            if angle(left, center, right, coords) > linear_threshold:
                continue
            for far in neighbors[right]:
                if far == center or far == left:
                    continue
                if angle(center, right, far, coords) > linear_threshold:
                    continue
                torsions.append(
                    _primitive_coordinate(
                        "Dihe",
                        len(torsions) + 1,
                        Primitive("dihedral", (left, center, right, far)),
                    )
                )

    oop_prefix = "ImpD" if impdih else "OuPl"
    for center in range(graph.natoms):
        neigh = neighbors[center]
        if len(neigh) != 3:
            continue
        first, second, third = neigh
        if all(atom_ring[atom] != 0 for atom in (center, first, second, third)):
            continue
        if impdih:
            primitive = Primitive("dihedral", (first, center, third, second))
        else:
            primitive = Primitive("out_of_plane", (center, first, second, third))
        oops.append(_primitive_coordinate(oop_prefix, len(oops) + 1, primitive))

    bonds = _bond_length_coordinates(
        bond_primitives,
        effective_atomic_numbers=effective_atomic_numbers,
        coords=coords,
    )
    assembled = (bonds, bends, linears, torsions, oops)
    torsions.extend(
        _linear_case_completeness_torsions(
            tuple(coordinate for block in assembled for coordinate in block),
            graph,
            coords,
            neighbors=neighbors,
            linear_threshold=linear_threshold,
            start=len(torsions) + 1,
            excluded_central_bonds=frozenset(ring_bonds),
        )
    )
    return bonds, bends, linears, torsions, oops


def _linear_case_completeness_torsions(
    coordinates: tuple[GICForgePythonCoordinate, ...],
    graph,
    coords: np.ndarray,
    *,
    neighbors: list[list[int]],
    linear_threshold: float,
    start: int,
    excluded_central_bonds: frozenset[tuple[int, int]] = frozenset(),
) -> list[GICForgePythonCoordinate]:
    """Add only regular collapsed-center torsions needed to close the rank.

    An ordinary dihedral containing a linear triple has a singular derivative.
    Collapsing that triple to its two endpoints gives the regular equivalent
    used by Gaussian DefRed.  Candidate acceptance is global and rank-revealing,
    so this rule applies to every linear sequence without molecule-specific
    branches and leaves already complete coordinate charts unchanged.
    """

    target_rank = _target_rank(coords, graph)
    current = list(coordinates)
    current_rank = _coordinate_b_rank(tuple(current), coords)
    if current_rank >= target_rank:
        return []

    seen = {
        min(primitive.atoms, tuple(reversed(primitive.atoms)))
        for coordinate in current
        for _coefficient, primitive in coordinate.terms
        if primitive.kind == "dihedral" and len(primitive.atoms) == 4
    }
    additions: list[GICForgePythonCoordinate] = []
    for linear_center in range(graph.natoms):
        center_neighbors = neighbors[linear_center]
        for index, left_endpoint in enumerate(center_neighbors[:-1]):
            for right_endpoint in center_neighbors[index + 1 :]:
                if angle(left_endpoint, linear_center, right_endpoint, coords) < linear_threshold:
                    continue
                for left in neighbors[left_endpoint]:
                    if left == linear_center:
                        continue
                    for right in neighbors[right_endpoint]:
                        if right == linear_center:
                            continue
                        if tuple(sorted((left_endpoint, right_endpoint))) in (
                            excluded_central_bonds
                        ):
                            continue
                        atoms = (left, left_endpoint, right_endpoint, right)
                        if len(set(atoms)) != 4:
                            continue
                        canonical = min(atoms, tuple(reversed(atoms)))
                        if canonical in seen:
                            continue
                        seen.add(canonical)
                        candidate = _primitive_coordinate(
                            "Dihe",
                            start + len(additions),
                            Primitive("dihedral", canonical),
                        )
                        trial = tuple((*current, candidate))
                        trial_rank = _coordinate_b_rank(trial, coords)
                        if trial_rank <= current_rank:
                            continue
                        additions.append(
                            GICForgePythonCoordinate(
                                name=candidate.name,
                                block=candidate.block,
                                terms=candidate.terms,
                                type_index=candidate.type_index,
                                diagnostic=(
                                    "SPECIAL_LINEAR_CASE=YES "
                                    f"COLLAPSED_CENTER={linear_center + 1} "
                                    f"ENDPOINTS={left_endpoint + 1}-{right_endpoint + 1} "
                                    "SELECTION=GLOBAL_RANK_REVEALING"
                                ),
                            )
                        )
                        current.append(additions[-1])
                        current_rank = trial_rank
                        if current_rank >= target_rank:
                            return additions
    return additions


def _has_linear_pair(
    center: int, neigh: list[int], coords: np.ndarray, linear_threshold: float
) -> bool:
    for ib, first in enumerate(neigh[:-1]):
        for second in neigh[ib + 1 :]:
            if angle(first, center, second, coords) >= linear_threshold:
                return True
    return False


def _remove_collapsed_bonds(graph, coords: np.ndarray) -> bool:
    removed = False
    kept_bonds = []
    for first, second in graph.bonds:
        distance = float(np.linalg.norm(coords[first] - coords[second]))
        if distance < COLLAPSED_BOND_THRESHOLD_ANGSTROM:
            graph.adjacency[first].discard(second)
            graph.adjacency[second].discard(first)
            removed = True
        else:
            kept_bonds.append((first, second))
    if removed:
        graph.bonds = kept_bonds
    return removed


def _minimum_cycle_basis(
    graph,
    ringset,
    *,
    effective_atomic_numbers: tuple[float, ...],
    neighbors: list[list[int]],
) -> list[tuple[int, ...]]:
    if ringset is None:
        return []
    diagnostics = getattr(ringset, "cycle_basis_diagnostics", None)
    if diagnostics is not None and not diagnostics.complete:
        raise ValueError(
            "ORACLE ring contract is incomplete; remove the explicit ring-size truncation"
        )
    # ORACLE has already selected the independent minimum cycle basis. SMITH
    # changes only the traversal origin/direction used to define coordinate
    # phases; it must not perform a second cycle-space reduction.
    return [
        _orient_ring_for_gicforge(
            tuple(int(atom) for atom in ring.atoms),
            effective_atomic_numbers=effective_atomic_numbers,
            neighbors=neighbors,
        )
        for ring in sorted(ringset.rings, key=lambda item: (len(item.atoms), item.index))
    ]


def _orient_ring_for_gicforge(
    atoms: tuple[int, ...],
    *,
    effective_atomic_numbers: tuple[float, ...],
    neighbors: list[list[int]],
) -> tuple[int, ...]:
    if len(atoms) <= 3:
        return atoms

    traversal = _connected_ring_traversal(atoms, neighbors)
    candidates = []
    for seq in (traversal, tuple(reversed(traversal))):
        for index in range(len(seq)):
            candidates.append(seq[index:] + seq[:index])
    return max(
        candidates,
        key=lambda seq: _ring_priority_key(
            seq,
            effective_atomic_numbers=effective_atomic_numbers,
            neighbors=neighbors,
        ),
    )


def _connected_ring_traversal(
    atoms: tuple[int, ...], neighbors: list[list[int]]
) -> tuple[int, ...]:
    remaining = list(atoms)
    first = min(remaining)
    start = remaining.index(first)
    remaining[0], remaining[start] = remaining[start], remaining[0]
    for index in range(len(remaining) - 1):
        current = remaining[index]
        next_index = index + 1
        for candidate_index in range(index + 1, len(remaining)):
            if remaining[candidate_index] in neighbors[current]:
                next_index = candidate_index
                break
        remaining[index + 1], remaining[next_index] = remaining[next_index], remaining[index + 1]
    return tuple(remaining)


def _ring_priority_key(
    ring: tuple[int, ...],
    *,
    effective_atomic_numbers: tuple[float, ...],
    neighbors: list[list[int]],
) -> tuple[tuple[float, int, tuple[float, ...], int], ...]:
    return tuple(
        (
            round(effective_atomic_numbers[atom], 12),
            len(neighbors[atom]),
            tuple(_exocyclic_neighbor_priorities(atom, ring, effective_atomic_numbers, neighbors)),
            -atom,
        )
        for atom in ring
    )


def _exocyclic_neighbor_priorities(
    atom: int,
    ring: tuple[int, ...],
    effective_atomic_numbers: tuple[float, ...],
    neighbors: list[list[int]],
) -> tuple[float, ...]:
    ring_atoms = set(ring)
    return tuple(
        sorted(
            (
                round(effective_atomic_numbers[neighbor], 12)
                for neighbor in neighbors[atom]
                if neighbor not in ring_atoms
            ),
            reverse=True,
        )
    )


def _effective_atomic_numbers(
    graph,
    coords: np.ndarray,
    atomic_numbers: tuple[int, ...],
    neighbors: list[list[int]],
) -> tuple[float, ...]:
    synthons: list[float] = []
    for center in range(graph.natoms):
        neigh = neighbors[center]
        if not neigh:
            synthons.append(0.0)
            continue
        z_center = atomic_numbers[center]
        nval = z_center
        if z_center > 2:
            nval -= 2
        if z_center > 10:
            nval -= 8
        nmax = 8 - nval
        neff = nmax - len(neigh) + 1
        if nval == 1:
            t0 = 180.0
        else:
            t0_values = {1: 109.47, 2: 120.0, 3: 180.0}
            t0 = t0_values.get(neff, 109.47)

        delocalization = 1.0
        coordination = 0.0
        rigidity = 0.0
        angle_count = 0
        for pos, neighbor in enumerate(neigh):
            distance = float(np.linalg.norm(coords[center] - coords[neighbor]))
            radius_center = covalent_radius(z_center) or 0.0
            radius_neighbor = covalent_radius(atomic_numbers[neighbor]) or 0.0
            bond_order = float(np.exp(((radius_center + radius_neighbor) - distance) / 0.3))
            delocalization *= bond_order
            coordination += float(atomic_numbers[neighbor]) * bond_order
            if len(neigh) == 1:
                continue
            for other in neigh[pos:]:
                if other == neighbor:
                    continue
                angle_count += 1
                rigidity += abs(np.sin(angle(neighbor, center, other, coords) - np.deg2rad(t0)))
        if angle_count == 0:
            angle_count = 1
        synthons.append(
            coordination / len(neigh) + delocalization / len(neigh) + rigidity / angle_count
        )

    synmax = max(synthons) if synthons else 0.0
    denominator = synmax + 0.1
    return tuple(
        float(z) - 0.495 + synthon / denominator for z, synthon in zip(atomic_numbers, synthons)
    )




def _bridge_bonds(rings: list[tuple[int, ...]]) -> set[tuple[int, int]]:
    counts: dict[tuple[int, int], int] = {}
    for ring in rings:
        for index, atom in enumerate(ring):
            edge = tuple(sorted((atom, ring[(index + 1) % len(ring)])))
            counts[edge] = counts.get(edge, 0) + 1
    return {edge for edge, count in counts.items() if count >= 2}


def _butterfly_coordinate(
    center: int,
    right: int,
    *,
    neighbors: list[list[int]],
    atom_ring: list[int],
    selected_rings: list[tuple[int, ...]],
    coords: np.ndarray,
    linear_threshold: float,
    index: int,
) -> GICForgePythonCoordinate | None:
    terms: list[tuple[float, Primitive]] = []
    for left in neighbors[center]:
        if left == right:
            continue
        if angle(left, center, right, coords) > linear_threshold:
            continue
        if atom_ring[left] == 0:
            continue
        for far in neighbors[right]:
            if far == center or far == left:
                continue
            if angle(center, right, far, coords) > linear_threshold:
                continue
            if atom_ring[far] == 0:
                continue
            if _atoms_share_selected_ring(left, far, selected_rings):
                continue
            coefficient = 1.0 if not terms else -1.0
            terms.append((coefficient, Primitive("dihedral", (left, center, right, far))))
    if not terms:
        return None
    norm = np.sqrt(float(len(terms)))
    return GICForgePythonCoordinate(
        name=f"BtFl{index:04d}",
        block="BtFl",
        type_index=2,
        terms=tuple((coefficient / norm, primitive) for coefficient, primitive in terms),
    )


def _atoms_share_selected_ring(first: int, second: int, rings: list[tuple[int, ...]]) -> bool:
    for ring in rings:
        if first in ring and second in ring:
            return True
    return False


def _exocyclic_angle_primitives(
    center: int,
    neigh: list[int],
    *,
    selected_rings: list[tuple[int, ...]],
    coords: np.ndarray,
    linear_threshold: float,
) -> tuple[list[Primitive], list[Primitive]]:
    angles: list[Primitive] = []
    linears: list[Primitive] = []
    suppress_redundant_trans = len(neigh) >= 3 or _has_redundant_linear_pairs(center, neigh, coords)
    for index, first in enumerate(neigh[:-1]):
        for second in neigh[index + 1 :]:
            left, right = sorted((first, second))
            if _is_endocyclic_angle(left, center, right, selected_rings):
                continue
            primitive = Primitive("angle", (left, center, right))
            if angle(left, center, right, coords) < linear_threshold:
                angles.append(primitive)
            else:
                if suppress_redundant_trans:
                    continue
                linears.append(_linear_bend_primitive((left, center, right), coords, mode=-1))
    return angles, linears


def _has_redundant_linear_pairs(center: int, neigh: list[int], coords: np.ndarray) -> bool:
    """Return whether a recognized template contains redundant trans pairs."""

    match = _local_coordination_match(center, neigh, coords=coords)
    if match.template is None:
        return False
    ideal_cosines = _sorted_pair_cosines(np.asarray(match.template.directions, dtype=float))
    return bool(np.any(ideal_cosines <= -1.0 + 1.0e-8))


def _is_endocyclic_angle(left: int, center: int, right: int, rings: list[tuple[int, ...]]) -> bool:
    for ring in rings:
        size = len(ring)
        for index, atom in enumerate(ring):
            if atom != center:
                continue
            if {left, right} == {ring[(index - 1) % size], ring[(index + 1) % size]}:
                return True
    return False


def _local_angle_group_records(
    center: int,
    neigh: list[int],
    primitives: list[Primitive],
    *,
    effective_atomic_numbers: tuple[float, ...],
    coords: np.ndarray,
    settings: LocalSALCSettings,
) -> tuple[tuple[str, tuple[Primitive, ...]], ...]:
    classes = _local_ligand_equivalence_classes(
        center,
        neigh,
        effective_atomic_numbers=effective_atomic_numbers,
        coords=coords,
        zeff_tolerance=settings.zeff_tolerance,
        distance_tolerance=settings.distance_tolerance_angstrom,
    )
    class_by_atom = {
        atom: class_index for class_index, atoms in enumerate(classes) for atom in atoms
    }
    match = _local_coordination_match(
        center,
        neigh,
        coords=coords,
        max_rms_cosine_error=settings.template_rms_threshold,
        min_score_margin=settings.template_min_margin,
    )
    ideal_cosines = (
        _template_pair_cosine_classes(
            match.template,
            tolerance=settings.angle_class_tolerance,
        )
        if len(neigh) >= 5 and match.template is not None
        else ()
    )
    grouped: dict[tuple[int, ...], list[Primitive]] = {}
    for primitive in primitives:
        left, _center, right = primitive.atoms
        key: tuple[int, ...] = tuple(sorted((class_by_atom[left], class_by_atom[right])))
        if ideal_cosines:
            key += (
                _nearest_cosine_class(
                    _ligand_pair_cosine(center, left, right, coords),
                    ideal_cosines,
                ),
            )
        grouped.setdefault(key, []).append(primitive)
    group, confidence = _infer_local_pseudogroup(
        center,
        neigh,
        classes,
        coords=coords,
        settings=settings,
        match=match,
    )
    template_diagnostic = _local_template_diagnostic(match) if len(neigh) >= 5 else ""
    return tuple(
        (
            "LOCAL_SALC KIND=ANGLE "
            f"DOMAIN=CENTER:{center + 1} GROUP={group} CONFIDENCE={confidence} "
            f"LIGAND_CLASSES={_ligand_class_token(classes)} "
            f"KEY={'-'.join(str(item) for item in key)} "
            f"SIZE={len(grouped[key])} ZEFF_TOL={settings.zeff_tolerance:.1e} "
            f"DIST_TOL={settings.distance_tolerance_angstrom:.1e} "
            f"TEMPLATE_RMS_TOL={settings.template_rms_threshold:.6g} "
            f"TEMPLATE_MARGIN_TOL={settings.template_min_margin:.6g} "
            f"{template_diagnostic} "
            f"ATOMS={_primitive_atom_group_token(tuple(grouped[key]))}",
            tuple(grouped[key]),
        )
        for key in sorted(grouped)
    )


def _ligand_class_token(classes: tuple[tuple[int, ...], ...]) -> str:
    return ";".join("-".join(str(atom + 1) for atom in group) for group in classes) or "NONE"


def _infer_local_pseudogroup(
    center: int,
    neigh: list[int],
    classes: tuple[tuple[int, ...], ...],
    *,
    coords: np.ndarray,
    settings: LocalSALCSettings | None = None,
    match: LocalCoordinationMatch | None = None,
) -> tuple[str, str]:
    return infer_local_pseudogroup(
        center,
        neigh,
        classes,
        coordinates_angstrom=coords,
        settings=settings,
        match=match,
    )


def _ring_local_pseudogroup(
    ring: tuple[int, ...],
    *,
    effective_atomic_numbers: tuple[float, ...],
    coords: np.ndarray,
    settings: LocalSALCSettings | None = None,
) -> tuple[str, str, int]:
    return ring_local_pseudogroup(
        ring,
        effective_atomic_numbers=effective_atomic_numbers,
        coordinates_angstrom=coords,
        settings=settings,
    )


def _aromatic_ring_block_labels(
    rings: list[tuple[int, ...]],
    aromatic_atoms: frozenset[int],
) -> dict[int, int]:
    """Label fused aromatic components without changing ring ownership.

    Two aromatic rings belong to the same local block when they share an edge.
    The labels are deterministic in ring order and are used only as semantic
    diagnostics; each ring still supplies its own analytic local SALCs.
    """

    aromatic_indices = [
        index
        for index, ring in enumerate(rings)
        if ring and all(atom in aromatic_atoms for atom in ring)
    ]
    memberships = {index: frozenset(rings[index]) for index in aromatic_indices}
    adjacency = {index: set() for index in aromatic_indices}
    for offset, left in enumerate(aromatic_indices):
        for right in aromatic_indices[offset + 1 :]:
            if len(memberships[left] & memberships[right]) >= 2:
                adjacency[left].add(right)
                adjacency[right].add(left)

    labels: dict[int, int] = {}
    for start in aromatic_indices:
        if start in labels:
            continue
        block = len(set(labels.values())) + 1
        pending = [start]
        while pending:
            index = pending.pop()
            if index in labels:
                continue
            labels[index] = block
            pending.extend(sorted(adjacency[index] - labels.keys(), reverse=True))
    return labels


def _torsion_coordinate(
    center: int,
    right: int,
    *,
    neighbors: list[list[int]],
    atomic_numbers: tuple[int, ...],
    effective_atomic_numbers: tuple[float, ...],
    atom_ring: list[int],
    ring_counts: list[int],
    coords: np.ndarray,
    linear_threshold: float,
    index: int,
) -> GICForgePythonCoordinate | None:
    candidates: list[tuple[int, int]] = []
    for left in neighbors[center]:
        if left == right:
            continue
        if angle(left, center, right, coords) > linear_threshold:
            continue
        for far in neighbors[right]:
            if far == center or far == left:
                continue
            if angle(center, right, far, coords) > linear_threshold:
                continue
            if ring_counts[left] >= 2 or ring_counts[far] >= 2:
                continue
            candidates.append((left, far))
    if not candidates:
        return None
    coefficient = 1.0 / np.sqrt(float(len(candidates)))
    return GICForgePythonCoordinate(
        name=f"Tors{index:04d}",
        block="Tors",
        type_index=-1,
        terms=tuple(
            (coefficient, Primitive("dihedral", (left, center, right, far)))
            for left, far in candidates
        ),
    )


def _onedih_torsion_coordinate(
    center: int,
    right: int,
    *,
    neighbors: list[list[int]],
    atomic_numbers: tuple[int, ...],
    effective_atomic_numbers: tuple[float, ...],
    atom_ring: list[int],
    ring_counts: list[int],
    coords: np.ndarray,
    linear_threshold: float,
    index: int,
) -> GICForgePythonCoordinate | None:
    candidates: list[tuple[int, int]] = []
    for left in neighbors[center]:
        if left == right:
            continue
        if angle(left, center, right, coords) > linear_threshold:
            continue
        if ring_counts[left] >= 2:
            continue
        for far in neighbors[right]:
            if far == center or far == left:
                continue
            if angle(center, right, far, coords) > linear_threshold:
                continue
            if ring_counts[far] >= 2:
                continue
            candidates.append((left, far))
    if not candidates:
        return None

    threshold = 5.0e-4
    selected_left, selected_far = max(
        candidates,
        key=lambda pair: (
            round(effective_atomic_numbers[pair[0]] / threshold) * threshold,
            round(effective_atomic_numbers[pair[1]] / threshold) * threshold,
            len(neighbors[center]),
            len(neighbors[right]),
            -pair[0],
            -pair[1],
        ),
    )
    return GICForgePythonCoordinate(
        name=f"Tors{index:04d}",
        block="Tors",
        type_index=-1,
        terms=((1.0, Primitive("dihedral", (selected_left, center, right, selected_far))),),
        diagnostic=(
            f"ONEDIH KIND=TORSION DOMAIN=BOND:{center + 1}-{right + 1} "
            f"SELECTED={selected_left + 1}-{center + 1}-{right + 1}-{selected_far + 1} "
            "POLICY=DETERMINISTIC_SINGLE_DIHEDRAL"
        ),
    )


def _xy3_collective_torsion_coordinate(
    center: int,
    right: int,
    *,
    neighbors: list[list[int]],
    atomic_numbers: tuple[int, ...],
    effective_atomic_numbers: tuple[float, ...],
    atom_ring: list[int],
    ring_counts: list[int],
    coords: np.ndarray,
    linear_threshold: float,
    index: int,
) -> GICForgePythonCoordinate | None:
    """Return the normalized collective rotor coordinate for an X(Y1,Y2,Y3)-Z domain.

    Gaussian's XY3 model spans three combinations: two differential branch
    coordinates and their common rotation.  SMITH already represents the two
    branch-deformation directions in its angle block, so the nonredundant
    SONIC chart retains only the common torsional direction.  Averaging over
    every admissible W-Z anchor makes the coordinate independent of a single
    skeleton atom; normalization leaves the Wilson-row scale well conditioned.
    """

    domains: list[tuple[bool, int, int, tuple[int, int, int], tuple[int, ...]]] = []
    threshold = 5.0e-4
    for x_atom, z_atom in ((center, right), (right, center)):
        y_atoms = tuple(atom for atom in neighbors[x_atom] if atom != z_atom)
        if len(y_atoms) != 3:
            continue
        if any(
            angle(y_atom, x_atom, z_atom, coords) > linear_threshold or ring_counts[y_atom] >= 2
            for y_atom in y_atoms
        ):
            continue
        anchors = tuple(
            atom
            for atom in neighbors[z_atom]
            if atom != x_atom
            and atom not in y_atoms
            and angle(atom, z_atom, x_atom, coords) <= linear_threshold
            and ring_counts[atom] < 2
        )
        if not anchors:
            continue
        equivalent = _branches_are_equivalent(
            y_atoms,
            excluded_center=x_atom,
            atomic_numbers=atomic_numbers,
            effective_atomic_numbers=effective_atomic_numbers,
            neighbors=neighbors,
            atom_ring=atom_ring,
        )
        if not equivalent:
            continue
        ordered_y = tuple(
            sorted(
                y_atoms,
                key=lambda atom: (
                    -round(effective_atomic_numbers[atom] / threshold),
                    -atomic_numbers[atom],
                    -len(neighbors[atom]),
                    atom,
                ),
            )
        )
        ordered_anchors = tuple(
            sorted(
                anchors,
                key=lambda atom: (
                    -round(effective_atomic_numbers[atom] / threshold),
                    -atomic_numbers[atom],
                    -len(neighbors[atom]),
                    atom,
                ),
            )
        )
        domains.append((equivalent, x_atom, z_atom, ordered_y, ordered_anchors))
    if not domains:
        return None

    equivalent, x_atom, z_atom, y_atoms, anchors = max(
        domains,
        key=lambda item: (
            item[0],
            round(effective_atomic_numbers[item[1]] / threshold),
            atomic_numbers[item[1]],
            -item[1],
            -item[2],
        ),
    )
    term_count = len(y_atoms) * len(anchors)
    coefficient = 1.0 / np.sqrt(float(term_count))
    return GICForgePythonCoordinate(
        name=f"Tors{index:04d}",
        block="Tors",
        type_index=-3 if equivalent else -1,
        terms=tuple(
            (coefficient, Primitive("dihedral", (anchor, z_atom, x_atom, y_atom)))
            for anchor in anchors
            for y_atom in y_atoms
        ),
        diagnostic=(
            f"XY3 KIND=COLLECTIVE_TORSION DOMAIN=X:{x_atom + 1}-Z:{z_atom + 1} "
            f"Y={','.join(str(atom + 1) for atom in y_atoms)} "
            f"ANCHORS={','.join(str(atom + 1) for atom in anchors)} "
            f"TERMS={term_count} NORMALIZATION=UNIT_L2 "
            f"EQUIVALENCE={'PSEUDOSYMMETRIC' if equivalent else 'HETERO'} "
            f"PERIODICITY={3 if equivalent else 1} "
            "GAUSSIAN_XY3_SPAN=COMMON_ROTATION "
            "DIFFERENTIAL_BRANCH_MODES=ANGLE_BLOCK"
        ),
    )


def _xy2_collective_torsion_coordinate(
    center: int,
    right: int,
    *,
    neighbors: list[list[int]],
    atomic_numbers: tuple[int, ...],
    effective_atomic_numbers: tuple[float, ...],
    atom_ring: list[int],
    ring_counts: list[int],
    coords: np.ndarray,
    linear_threshold: float,
    index: int,
) -> GICForgePythonCoordinate | None:
    domains = []
    for x_atom, z_atom in ((center, right), (right, center)):
        if atom_ring[x_atom] != 0:
            continue
        y_atoms = tuple(sorted(atom for atom in neighbors[x_atom] if atom != z_atom))
        if len(y_atoms) != 2:
            continue
        if not _branches_are_equivalent(
            y_atoms,
            excluded_center=x_atom,
            atomic_numbers=atomic_numbers,
            effective_atomic_numbers=effective_atomic_numbers,
            neighbors=neighbors,
            atom_ring=atom_ring,
        ):
            continue
        if any(angle(y, x_atom, z_atom, coords) > linear_threshold for y in y_atoms):
            continue
        anchors = tuple(
            sorted(
                atom
                for atom in neighbors[z_atom]
                if atom != x_atom
                and angle(atom, z_atom, x_atom, coords) <= linear_threshold
                and ring_counts[atom] < 2
            )
        )
        if not anchors:
            continue
        domains.append((x_atom, z_atom, y_atoms, anchors))
    if not domains:
        return None
    threshold = 5.0e-4
    x_atom, z_atom, y_atoms, anchors = max(
        domains,
        key=lambda item: (
            round(effective_atomic_numbers[item[0]] / threshold),
            atomic_numbers[item[0]],
            -item[0],
        ),
    )
    term_count = len(y_atoms) * len(anchors)
    coefficient = 1.0 / np.sqrt(float(term_count))
    return GICForgePythonCoordinate(
        name=f"Tors{index:04d}",
        block="Tors",
        type_index=-2,
        terms=tuple(
            (coefficient, Primitive("dihedral", (anchor, z_atom, x_atom, y_atom)))
            for anchor in anchors
            for y_atom in y_atoms
        ),
        diagnostic=(
            f"XY2 KIND=COLLECTIVE_TORSION DOMAIN=X:{x_atom + 1}-Z:{z_atom + 1} "
            f"Y={y_atoms[0] + 1},{y_atoms[1] + 1} "
            f"ANCHORS={','.join(str(atom + 1) for atom in anchors)} "
            f"TERMS={term_count} NORMALIZATION=UNIT_L2 PERIODICITY=2 "
            "DIFFERENTIAL_BRANCH_MODES=ANGLE_AND_OUT_OF_PLANE_BLOCK"
        ),
    )


def _single_dihedral_torsion_coordinate(
    center: int,
    right: int,
    **kwargs: object,
) -> GICForgePythonCoordinate | None:
    """Return one physical exocyclic dihedral for a PES scan variable."""

    candidate = _onedih_torsion_coordinate(center, right, **kwargs)
    if candidate is None:
        return None
    _coefficient, primitive = candidate.terms[0]
    return GICForgePythonCoordinate(
        name=candidate.name,
        block=candidate.block,
        type_index=candidate.type_index,
        terms=((1.0, primitive),),
        diagnostic=(
            f"PES_EXPLORATION KIND=TORSION DOMAIN=BOND:{center + 1}-{right + 1} "
            "POLICY=SINGLE_DIHEDRAL NO_COMBINATION=TRUE"
        ),
    )


def _local_salc_torsion_coordinate(
    center: int,
    right: int,
    **kwargs: object,
) -> GICForgePythonCoordinate | None:
    candidate = _torsion_coordinate(center, right, **kwargs)
    if candidate is None:
        fallback = _onedih_torsion_coordinate(center, right, **kwargs)
        if fallback is None:
            return None
        source = (
            f"LOCAL_SALC KIND=TORSION DOMAIN=BOND:{center + 1}-{right + 1} "
            "GROUP=C1 LOCAL_IRREP=A1 FALLBACK=ONEDIH "
            "REASON=NO_COMPLETE_ORBIT STABILITY=0 RANK_TEST=FAILED "
            "CONTINUITY=LOCAL_ANALYTIC CONFIDENCE=LOW"
        )
        return GICForgePythonCoordinate(
            name=fallback.name,
            block=fallback.block,
            terms=fallback.terms,
            type_index=fallback.type_index,
            diagnostic=source,
            fallback_events=(
                make_fallback_event(
                    stage="SMITH_LOCAL_SALC",
                    algorithm_id="ONEDIH",
                    trigger="NO_COMPLETE_TORSION_ORBIT",
                    domain=f"BOND:{center + 1}-{right + 1}",
                    macrofamily="TORSION",
                    source=source,
                ),
            ),
        )
    # Use the complete locally equivalent orbit only when its analytic Wilson
    # row survives signed cancellation.  The dimensionless ratio is invariant
    # to Cartesian orientation and exposes the numerical decision explicitly.
    coords = np.asarray(kwargs["coords"], dtype=float)
    stability, row_norm = _torsion_orbit_stability(candidate, coords)
    if row_norm <= 1.0e-10 or stability < LOCAL_TORSION_STABILITY_THRESHOLD:
        fallback = _onedih_torsion_coordinate(center, right, **kwargs)
        if fallback is None:
            return None
        source = (
            f"LOCAL_SALC KIND=TORSION DOMAIN=BOND:{center + 1}-{right + 1} "
            "GROUP=C1 LOCAL_IRREP=A1 FALLBACK=ONEDIH "
            f"REASON=UNSTABLE_ORBIT STABILITY={stability:.6g} "
            "RANK_TEST=FAILED CONTINUITY=LOCAL_ANALYTIC CONFIDENCE=LOW"
        )
        return GICForgePythonCoordinate(
            name=fallback.name,
            block=fallback.block,
            terms=fallback.terms,
            type_index=fallback.type_index,
            diagnostic=source,
            fallback_events=(
                make_fallback_event(
                    stage="SMITH_LOCAL_SALC",
                    algorithm_id="ONEDIH",
                    trigger="UNSTABLE_TORSION_ORBIT",
                    domain=f"BOND:{center + 1}-{right + 1}",
                    macrofamily="TORSION",
                    source=source,
                ),
            ),
        )
    return GICForgePythonCoordinate(
        name=candidate.name,
        block=candidate.block,
        terms=candidate.terms,
        type_index=candidate.type_index,
        diagnostic=(
            f"LOCAL_SALC KIND=TORSION DOMAIN=BOND:{center + 1}-{right + 1} "
            f"GROUP=LOCAL_ROTOR LOCAL_IRREP=A1 ORBIT_SIZE={len(candidate.terms)} "
            f"FALLBACK=NONE SELECTION=FULL_ORBIT STABILITY={stability:.6g} "
            "RANK_TEST=PASSED CONTINUITY=LOCAL_ANALYTIC CONFIDENCE=HIGH"
        ),
    )


def _torsion_orbit_stability(
    coordinate: GICForgePythonCoordinate,
    coords: np.ndarray,
) -> tuple[float, float]:
    aggregate = np.zeros(coords.size, dtype=float)
    scale = 0.0
    for coefficient, primitive in coordinate.terms:
        primitive_row = b_matrix_analytic((primitive,), coords)[0]
        aggregate += float(coefficient) * primitive_row
        scale += abs(float(coefficient)) * float(np.linalg.norm(primitive_row))
    norm = float(np.linalg.norm(aggregate))
    return (norm / scale if scale > 1.0e-15 else 0.0), norm


from matrix_chem.ring_primitives import _balanced_ring_anchors, _triangular_flap_stencil


def _flap_quaternion(
    atoms: tuple[int, int, int, int], coords: np.ndarray
) -> tuple[float, np.ndarray, float]:
    """Return canonical hinge quaternion and its signed angle.

    The scalar part is kept non-negative, fixing the Q == -Q ambiguity for
    incidence angles in the principal interval.  ``atan2`` in the primitive
    dihedral kernel retains the sign and remains continuous through zero.
    """

    angle_value = float(eval_primitive(Primitive("dihedral", atoms), coords))
    edge = np.asarray(coords[atoms[2]] - coords[atoms[1]], dtype=float)
    edge_norm = float(np.linalg.norm(edge))
    if edge_norm <= 1.0e-14:
        raise ValueError("collapsed triangular-flap hinge")
    axis = edge / edge_norm
    scalar = float(np.cos(0.5 * angle_value))
    vector = axis * float(np.sin(0.5 * angle_value))
    if scalar < 0.0:
        scalar = -scalar
        vector = -vector
    recovered = float(2.0 * np.arctan2(np.dot(axis, vector), scalar))
    return scalar, vector, recovered


def _cyclic_triangular_flap_coordinates(
    ring: tuple[int, ...],
    *,
    coords: np.ndarray,
    prefix: str,
    start: int,
    diagnostic: str = "",
) -> list[GICForgePythonCoordinate]:
    """Build the default minimal curvilinear ring-puckering chart."""

    stencils = _triangular_flap_stencil(ring)
    if not stencils:
        return []
    primitives = tuple(Primitive("dihedral", atoms) for atoms in stencils)
    flap_b = b_matrix_analytic(primitives, coords)
    singular_values = np.linalg.svd(flap_b, compute_uv=False)
    expected_rank = len(ring) - 3
    rank = _svd_rank(singular_values)
    if rank != expected_rank or singular_values[-1] <= 1.0e-12:
        return []
    condition = float(singular_values[0] / singular_values[-1])
    if not np.isfinite(condition) or condition > 1.0e10:
        return []
    anchors = _balanced_ring_anchors(len(ring))
    anchor_atoms = ",".join(str(ring[index] + 1) for index in anchors)
    coordinates: list[GICForgePythonCoordinate] = []
    for mode, primitive in enumerate(primitives, start=1):
        scalar, vector, recovered = _flap_quaternion(primitive.atoms, coords)
        quaternion = ",".join(
            f"{value:.10g}" for value in (scalar, vector[0], vector[1], vector[2])
        )
        coordinates.append(
            GICForgePythonCoordinate(
                name=f"{prefix}{start + len(coordinates):04d}",
                block=prefix,
                type_index=15,
                terms=((1.0, primitive),),
                diagnostic=(
                    f"{diagnostic} MODEL=TRIANGULAR_FLAP "
                    f"PRIMITIVE=SIGNED_PLANE_INCIDENCE KERNEL=ATAN2_DIHEDRAL "
                    f"REFERENCE_TRIANGLE={anchor_atoms} TREE_EDGE={mode} "
                    f"FLAP_RANK={rank}/{expected_rank} "
                    f"FLAP_CONDITION={condition:.3e} "
                    f"VALUE_DEG={np.rad2deg(recovered):.10g} "
                    f"QUATERNION={quaternion} QUATERNION_SIGN=SCALAR_NONNEGATIVE"
                ).strip(),
            )
        )
    return coordinates


def _cyclic_out_of_plane_coordinates(
    ring: tuple[int, ...],
    *,
    coords: np.ndarray,
    prefix: str,
    start: int,
    diagnostic: str = "",
) -> list[GICForgePythonCoordinate]:
    """Build ring Fourier modes from native out-of-plane primitives.

    A four-atom ring dihedral mixes the displacement of several trigonal
    centers, just as an improper dihedral does at one trigonal center.  For a
    ring, rewrite each ordered ring torsion as its native out-of-plane
    counterpart U(b,a,d,c) and form the same n-3 Fourier subspace.  The
    construction is also attempted for nonplanar rings; it is accepted only
    when the analytic U-based B rows retain full rank and a finite condition
    number.  Failure is reported explicitly; the selected model is never
    replaced silently by endocyclic dihedrals.
    """

    ncyc = len(ring)
    if ncyc <= 3:
        return []
    ring_xyz = np.asarray([coords[atom] for atom in ring], dtype=float)
    centered = ring_xyz - np.mean(ring_xyz, axis=0)
    planarity = float(np.linalg.svd(centered, compute_uv=False)[-1])
    source_terms = tuple(
        tuple((coefficient, Primitive("out_of_plane", atoms)) for coefficient, atoms in terms)
        for terms in cyclic_out_of_plane_source_terms(ring)
    )

    coefficients = _cyclic_reference_coefficients(ncyc, valence_angle=False)
    source_b = np.vstack(
        [
            sum(
                coefficient * b_matrix_analytic((primitive,), coords)[0]
                for coefficient, primitive in terms
            )
            for terms in source_terms
        ]
    )
    mode_b = coefficients.T @ source_b
    singular_values = np.linalg.svd(mode_b, compute_uv=False)
    rank = _svd_rank(singular_values)
    expected_rank = ncyc - 3
    if rank != expected_rank or singular_values[-1] <= 1.0e-12:
        return []
    condition = float(singular_values[0] / singular_values[-1])
    if not np.isfinite(condition) or condition > 1.0e10:
        return []
    geometry_kind = "PLANAR" if planarity <= 5.0e-2 else "NONPLANAR"
    coordinates: list[GICForgePythonCoordinate] = []
    for mode in range(coefficients.shape[1]):
        terms = tuple(
            (float(mode_coefficient) * float(source_coefficient), primitive)
            for mode_coefficient, source in zip(coefficients[:, mode], source_terms)
            if abs(float(mode_coefficient)) > 1.0e-12
            for source_coefficient, primitive in source
        )
        if not terms:
            continue
        coordinates.append(
            GICForgePythonCoordinate(
                name=f"{prefix}{start + len(coordinates):04d}",
                block=prefix,
                type_index=1,
                terms=terms,
                diagnostic=(
                    f"{diagnostic} PRIMITIVE=OUT_OF_PLANE "
                    f"ORIENTATION_PAIRING=FORWARD_MINUS_REVERSE "
                    f"RING_GEOMETRY={geometry_kind} PLANARITY={planarity:.3e} "
                    f"U_RANK={rank}/{expected_rank} U_CONDITION={condition:.3e} "
                    f"LOCAL_IRREP=FOURIER_{mode + 1} MODE={mode + 1}"
                ).strip(),
            )
        )
    return coordinates


from matrix_chem.ring_primitives import _charm_height_source_primitives


def _cyclic_charm_coordinates(
    ring: tuple[int, ...],
    *,
    coords: np.ndarray,
    prefix: str,
    start: int,
    diagnostic: str = "",
) -> list[GICForgePythonCoordinate]:
    """Build generic Cyclic Height-Averaged Ring Modes (CHARM)."""

    ncyc = len(ring)
    sources = _charm_height_source_primitives(ring)
    if len(sources) != ncyc:
        return []
    source_terms = tuple(
        tuple((1.0 / len(primitives), primitive) for primitive in primitives)
        for primitives in sources
    )
    source_b = np.vstack(
        [
            sum(
                coefficient * b_matrix_analytic((primitive,), coords)[0]
                for coefficient, primitive in terms
            )
            for terms in source_terms
        ]
    )
    coefficients = _cyclic_reference_coefficients(ncyc, valence_angle=False)
    mode_b = coefficients.T @ source_b
    singular_values = np.linalg.svd(mode_b, compute_uv=False)
    expected_rank = ncyc - 3
    rank = _svd_rank(singular_values)
    if rank != expected_rank or singular_values[-1] <= 1.0e-12:
        return []
    condition = float(singular_values[0] / singular_values[-1])
    if not np.isfinite(condition) or condition > 1.0e10:
        return []
    coordinates: list[GICForgePythonCoordinate] = []
    for mode in range(coefficients.shape[1]):
        terms = tuple(
            (float(source_coefficient) * float(height_coefficient), primitive)
            for source_coefficient, height_terms in zip(coefficients[:, mode], source_terms)
            if abs(float(source_coefficient)) > 1.0e-12
            for height_coefficient, primitive in height_terms
        )
        coordinates.append(
            GICForgePythonCoordinate(
                name=f"{prefix}{start + len(coordinates):04d}",
                block=prefix,
                type_index=15,
                terms=terms,
                diagnostic=(
                    f"{diagnostic} MODEL=CHARM "
                    f"PRIMITIVE=OUT_OF_PLANE_HEIGHT KERNEL=GAUSSIAN_H "
                    f"SOURCE_HEIGHTS={len(sources)} "
                    f"HEIGHT_TERMS_MIN={min(map(len, sources))} "
                    f"HEIGHT_TERMS_MAX={max(map(len, sources))} "
                    f"H_RANK={rank}/{expected_rank} H_CONDITION={condition:.3e} "
                    f"LOCAL_IRREP=FOURIER_{mode + 1} MODE={mode + 1}"
                ).strip(),
            )
        )
    return coordinates


def build_charm_ring_coordinates(
    ring: tuple[int, ...] | list[int],
    coordinates_angstrom: np.ndarray,
    *,
    prefix: str = "RPck",
    start: int = 1,
) -> tuple[GICForgePythonCoordinate, ...]:
    """Return SMITH's canonical CHARM rows for one ORACLE-ordered cycle.

    This is the public in-memory entry point for consumers such as ARCHITECT.
    It deliberately delegates to the same implementation used by GICForge so
    no second ring-coordinate definition can arise.
    """

    atoms = tuple(int(atom) for atom in ring)
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if len(atoms) < 5 or len(set(atoms)) != len(atoms):
        raise ValueError("CHARM requires an ordered cycle of at least five distinct atoms")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or np.any(~np.isfinite(xyz)):
        raise ValueError("CHARM coordinates require a finite (natoms, 3) geometry")
    if min(atoms) < 0 or max(atoms) >= len(xyz):
        raise ValueError("CHARM ring atom lies outside the geometry")
    rows = _cyclic_charm_coordinates(
        atoms,
        coords=xyz,
        prefix=str(prefix),
        start=int(start),
        diagnostic="ORACLE_ORDERED_CYCLE DIRECT_ARCHITECT_CONSUMER",
    )
    expected = len(atoms) - 3
    if len(rows) != expected:
        raise ValueError(f"SMITH CHARM construction is rank deficient ({len(rows)}/{expected})")
    return tuple(rows)


def _cyclic_coordinates(
    ring: tuple[int, ...],
    *,
    valence_angle: bool,
    prefix: str,
    start: int,
    coords: np.ndarray | None = None,
    atomic_numbers: tuple[int, ...] = (),
    diagnostic: str = "",
) -> list[GICForgePythonCoordinate]:
    ncyc = len(ring)
    if ncyc == 3:
        return []
    istart = 2 if valence_angle else ncyc
    if valence_angle:
        if ncyc == 6:
            istart = 2
        elif ncyc == 7:
            istart = 4
        elif ncyc == 8:
            istart = 2
        else:
            istart = 3
    vnorm = np.sqrt(2.0 / float(ncyc))
    vnorm1 = np.sqrt(1.0 / float(ncyc))
    coordinates: list[GICForgePythonCoordinate] = []
    flexibilities = (
        _ring_dihedral_flexibilities(ring, coords=coords, atomic_numbers=atomic_numbers)
        if not valence_angle
        else tuple(1.0 for _atom in ring)
    )
    for ivar in range(1, ncyc - 2):
        even = ivar == 2 * (ivar // 2)
        terms = []
        for iterm in range(1, ncyc + 1):
            iang1 = _cyclic_index(iterm + istart - 1, ncyc)
            iang2 = _cyclic_index(iterm + istart, ncyc)
            iang3 = _cyclic_index(iterm + istart + 1, ncyc)
            iang4 = _cyclic_index(iterm + istart + 2, ncyc)
            # The N-3 cyclic-deformation rows are the sine/cosine pairs for
            # harmonics k=2,3,..., plus the alternating mode for even N.  The
            # former case-by-case map covered only the first three pairs and
            # generated an identically zero sine row once N reached 16.
            harmonic = (ivar + 1) // 2 + 1
            snum = float(2 * harmonic * (iterm - 1))
            value = np.pi * snum / float(ncyc)
            if even:
                coefficient = vnorm * np.sin(value)
            elif ivar < ncyc - 3:
                coefficient = vnorm * np.cos(value)
            else:
                coefficient = vnorm1 * np.cos(float(iterm - 1) * np.pi)
            if not valence_angle:
                coefficient *= flexibilities[iterm - 1]
            if abs(coefficient) < 1.0e-14:
                coefficient = 0.0
            if valence_angle:
                primitive = Primitive("angle", (ring[iang1], ring[iang2], ring[iang3]))
            else:
                primitive = Primitive(
                    "dihedral", (ring[iang1], ring[iang2], ring[iang3], ring[iang4])
                )
            terms.append((float(coefficient), primitive))
        if not valence_angle:
            terms = _normalize_coordinate_terms(terms)
        coordinates.append(
            GICForgePythonCoordinate(
                name=f"{prefix}{start + len(coordinates):04d}",
                block=prefix,
                type_index=14 if valence_angle else 1,
                terms=tuple(terms),
                diagnostic=(
                    f"{diagnostic} LOCAL_IRREP=FOURIER_{ivar} MODE={ivar}" if diagnostic else ""
                ),
            )
        )
    return coordinates


def _cyclic_svd_coordinates(
    ring: tuple[int, ...],
    *,
    valence_angle: bool,
    coords: np.ndarray,
    atomic_numbers: tuple[int, ...] = (),
    prefix: str,
    start: int,
    diagnostic: str = "",
) -> list[GICForgePythonCoordinate]:
    primitives = _cyclic_primitives_legacy_order(ring, valence_angle=valence_angle)
    if not primitives:
        return []
    reference = _cyclic_reference_coefficients(len(ring), valence_angle=valence_angle)
    primitive_b = b_matrix_analytic(tuple(primitives), coords)
    u_matrix, singular_values, _vh = np.linalg.svd(primitive_b, full_matrices=False)
    rank = min(_svd_rank(singular_values), max(0, len(ring) - 3))
    if rank == 0:
        return []
    coefficients = _align_svd_modes_to_reference(u_matrix[:, :rank], reference[:, :rank])
    coordinates: list[GICForgePythonCoordinate] = []
    flexibilities = (
        _ring_dihedral_flexibilities(ring, coords=coords, atomic_numbers=atomic_numbers)
        if not valence_angle
        else tuple(1.0 for _primitive in primitives)
    )
    for mode in range(rank):
        coeffs = coefficients[:, mode].astype(float)
        if not valence_angle:
            for idx, flexibility in enumerate(flexibilities):
                coeffs[idx] *= flexibility
            coeffs = _normalize_coefficients(coeffs)
        coeffs[np.abs(coeffs) < 1.0e-14] = 0.0
        terms = tuple(
            (float(coefficient), primitive)
            for coefficient, primitive in zip(coeffs, primitives)
            if abs(float(coefficient)) > 1.0e-12
        )
        if not terms:
            continue
        coordinates.append(
            GICForgePythonCoordinate(
                name=f"{prefix}{start + len(coordinates):04d}",
                block=prefix,
                type_index=14 if valence_angle else 1,
                terms=terms,
                diagnostic=(
                    f"{diagnostic} LOCAL_IRREP=FOURIER_{mode + 1} MODE={mode + 1}"
                    if diagnostic
                    else ""
                ),
            )
        )
    return coordinates


def _ring_dihedral_flexibilities(
    ring: tuple[int, ...],
    *,
    coords: np.ndarray | None,
    atomic_numbers: tuple[int, ...],
    contrast_tolerance: float = 0.50,
) -> tuple[float, ...]:
    if coords is None or not atomic_numbers:
        return tuple(1.0 for _atom in ring)
    orders: list[float | None] = []
    ncyc = len(ring)
    for iterm in range(1, ncyc + 1):
        iang2 = _cyclic_index(iterm + ncyc, ncyc)
        iang3 = _cyclic_index(iterm + ncyc + 1, ncyc)
        orders.append(
            _geometric_bond_order(
                ring[iang2], ring[iang3], coords=coords, atomic_numbers=atomic_numbers
            )
        )
    finite = [order for order in orders if order is not None and order > 1.0e-12]
    if len(finite) != len(orders):
        return tuple(1.0 for _order in orders)
    reference = min(float(order) for order in finite)
    maximum = max(float(order) for order in finite)
    if reference <= 0.0 or maximum / reference <= 1.0 + float(contrast_tolerance):
        return tuple(1.0 for _order in orders)
    return tuple(float(np.sqrt(reference / float(order))) for order in finite)


def _geometric_bond_order(
    left: int,
    right: int,
    *,
    coords: np.ndarray,
    atomic_numbers: tuple[int, ...],
) -> float | None:
    if left < 0 or right < 0 or left >= len(atomic_numbers) or right >= len(atomic_numbers):
        return None
    radius_left = covalent_radius(atomic_numbers[left]) or 0.0
    radius_right = covalent_radius(atomic_numbers[right]) or 0.0
    reference = float(radius_left) + float(radius_right)
    distance = float(
        np.linalg.norm(
            np.asarray(coords[right], dtype=float) - np.asarray(coords[left], dtype=float)
        )
    )
    if reference <= 0.0 or distance <= 1.0e-12:
        return None
    return float(np.exp((reference - distance) / 0.30))


def _normalize_coordinate_terms(
    terms: list[tuple[float, Primitive]],
) -> list[tuple[float, Primitive]]:
    coefficients = np.asarray([coefficient for coefficient, _primitive in terms], dtype=float)
    coefficients = _normalize_coefficients(coefficients)
    return [
        (float(coefficient), primitive)
        for coefficient, (_old_coefficient, primitive) in zip(coefficients, terms)
    ]


def _normalize_coefficients(coefficients: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(coefficients))
    if norm <= 1.0e-14:
        return coefficients
    return coefficients / norm


def _cyclic_primitives_legacy_order(
    ring: tuple[int, ...], *, valence_angle: bool
) -> list[Primitive]:
    ncyc = len(ring)
    if ncyc == 3:
        return []
    istart = _cyclic_legacy_start(ncyc, valence_angle=valence_angle)
    primitives: list[Primitive] = []
    for iterm in range(1, ncyc + 1):
        iang1 = _cyclic_index(iterm + istart - 1, ncyc)
        iang2 = _cyclic_index(iterm + istart, ncyc)
        iang3 = _cyclic_index(iterm + istart + 1, ncyc)
        iang4 = _cyclic_index(iterm + istart + 2, ncyc)
        if valence_angle:
            primitives.append(Primitive("angle", (ring[iang1], ring[iang2], ring[iang3])))
        else:
            primitives.append(
                Primitive("dihedral", (ring[iang1], ring[iang2], ring[iang3], ring[iang4]))
            )
    return primitives


def _cyclic_reference_coefficients(ncyc: int, *, valence_angle: bool) -> np.ndarray:
    """Return the topology-only real Fourier basis for cyclic deformations.

    In the linearized height representation of a regular planar reference,
    translation of the ring plane and its two rigid tilts occupy Fourier
    wavenumbers zero and one.  The remaining modes therefore start at m=2.
    For nonlinear local-U sources this is a topology-only reference basis, not
    an exact finite-geometry separation; the actual projected B rows are rank
    checked before use.  Cosine/sine pairs are followed, for even rings, by the
    unpaired crown mode m=N/2.  The construction supplies exactly N-3
    orthonormal columns for any N >= 4 and has no chemical weighting.

    ``valence_angle`` remains part of the private call contract because the
    same reference basis is used to align both cyclic bend and puckering
    source spaces.
    """

    del valence_angle
    return cyclic_out_of_plane_coefficients(ncyc)


def _align_svd_modes_to_reference(u_matrix: np.ndarray, reference: np.ndarray) -> np.ndarray:
    aligned = np.zeros_like(reference)
    used: set[int] = set()
    for mode in range(reference.shape[1]):
        best_index = -1
        best_dot = 0.0
        best_score = -1.0
        for candidate in range(u_matrix.shape[1]):
            if candidate in used:
                continue
            dot = float(np.dot(reference[:, mode], u_matrix[:, candidate]))
            score = abs(dot)
            if score > best_score:
                best_index = candidate
                best_dot = dot
                best_score = score
        if best_index < 0:
            continue
        used.add(best_index)
        sign = -1.0 if best_dot < 0.0 else 1.0
        aligned[:, mode] = sign * u_matrix[:, best_index]
    return aligned


def _cyclic_legacy_start(ncyc: int, *, valence_angle: bool) -> int:
    if not valence_angle:
        return ncyc
    if ncyc == 6:
        return 2
    if ncyc == 7:
        return 4
    if ncyc == 8:
        return 2
    return 3


def _svd_local_coordinates(
    primitives: list[Primitive],
    *,
    coords: np.ndarray,
    prefix: str,
    start: int,
    kind_type_index: int,
    max_modes: int | None = None,
    diagnostic: str = "",
    fallback_events: tuple[FallbackEvent, ...] = (),
) -> list[GICForgePythonCoordinate]:
    if not primitives:
        return []
    primitive_b = b_matrix_analytic(tuple(primitives), coords)
    u_matrix, singular_values, _vh = np.linalg.svd(primitive_b, full_matrices=False)
    rank = _svd_rank(singular_values)
    if max_modes is not None:
        rank = min(rank, max_modes)
    coordinates: list[GICForgePythonCoordinate] = []
    for mode in range(rank):
        coeffs = u_matrix[:, mode].astype(float)
        coeffs = _canonical_svd_coefficients(coeffs)
        terms = tuple(
            (float(coefficient), primitive)
            for coefficient, primitive in zip(coeffs, primitives)
            if abs(float(coefficient)) > 1.0e-12
        )
        if not terms:
            continue
        coordinates.append(
            GICForgePythonCoordinate(
                name=f"{prefix}{start + len(coordinates):04d}",
                block=prefix,
                type_index=kind_type_index,
                terms=terms,
                diagnostic=f"{diagnostic} MODE={mode + 1}" if diagnostic else "",
                fallback_events=fallback_events,
            )
        )
    return coordinates


def _canonical_local_salc_coefficients(size: int) -> np.ndarray:
    """Return A1 followed by a deterministic orthonormal Helmert basis."""
    if size <= 0:
        return np.zeros((0, 0), dtype=float)
    coefficients = np.zeros((size, size), dtype=float)
    coefficients[0, :] = 1.0 / np.sqrt(float(size))
    for mode in range(1, size):
        denominator = np.sqrt(float(mode * (mode + 1)))
        coefficients[mode, :mode] = 1.0 / denominator
        coefficients[mode, mode] = -float(mode) / denominator
    return coefficients


def _local_salc_coordinates(
    primitives: list[Primitive],
    *,
    prefix: str,
    start: int,
    kind_type_index: int,
    diagnostic: str,
) -> list[GICForgePythonCoordinate]:
    coefficients = _canonical_local_salc_coefficients(len(primitives))
    coordinates: list[GICForgePythonCoordinate] = []
    for mode, row in enumerate(coefficients):
        irrep = "A1" if mode == 0 else f"NON_A1_{mode}"
        terms = tuple(
            (float(coefficient), primitive)
            for coefficient, primitive in zip(row, primitives)
            if abs(float(coefficient)) > 1.0e-12
        )
        coordinates.append(
            GICForgePythonCoordinate(
                name=f"{prefix}{start + mode:04d}",
                block=prefix,
                type_index=kind_type_index,
                terms=terms,
                diagnostic=f"{diagnostic} LOCAL_IRREP={irrep} MODE={mode + 1}",
            )
        )
    return coordinates


def _svd_rank(singular_values: np.ndarray) -> int:
    return spectrum_rank(
        singular_values,
        absolute_tolerance=1.0e-10,
        relative_tolerance=1.0e-8,
    )


def _canonical_svd_coefficients(coefficients: np.ndarray) -> np.ndarray:
    if coefficients.size == 0:
        return coefficients
    dominant = int(np.argmax(np.abs(coefficients)))
    if coefficients[dominant] < 0.0:
        coefficients = -coefficients
    coefficients[np.abs(coefficients) < 1.0e-14] = 0.0
    return coefficients


def _bond_length_coordinates(
    bond_primitives: list[GICForgePythonCoordinate],
    *,
    effective_atomic_numbers: tuple[float, ...],
    coords: np.ndarray,
    neighbors: list[list[int]] | None = None,
    selected_rings: list[tuple[int, ...]] | None = None,
    local_salc: bool = True,
    settings: LocalSALCSettings | None = None,
) -> list[GICForgePythonCoordinate]:
    local_settings = settings or LocalSALCSettings()
    if not local_salc:
        # The legacy Fortran chart emits primitive stretches in canonical
        # bond order.  Do the same here: local equivalence must not make the
        # persistent coordinate identity depend on geometry or atom traversal.
        return [
            GICForgePythonCoordinate(
                name=f"Stre{index:04d}",
                block="Stre",
                type_index=0,
                terms=((1.0, primitive),),
                diagnostic=(
                    "LOCAL_EQUIVALENCE KIND=STRETCH GROUP=C1 "
                    "CANONICAL_PRIMITIVE=YES "
                    f"BOND={primitive.atoms[0] + 1}-{primitive.atoms[1] + 1}"
                ),
            )
            for index, primitive in enumerate(
                sorted(
                    (coordinate.terms[0][1] for coordinate in bond_primitives),
                    key=lambda item: tuple(sorted(item.atoms)),
                ),
                start=1,
            )
        ]
    if local_salc and neighbors is not None and selected_rings is not None:
        records = _bond_primitive_domain_records(
            bond_primitives,
            effective_atomic_numbers=effective_atomic_numbers,
            coords=coords,
            neighbors=neighbors,
            selected_rings=selected_rings,
            settings=local_settings,
        )
    else:
        records = tuple(
            (
                "LOCAL_EQUIVALENCE KIND=STRETCH DOMAIN=MOLECULE GROUP=C1",
                primitives,
            )
            for primitives in _bond_primitives_by_equivalence(
                bond_primitives,
                effective_atomic_numbers=effective_atomic_numbers,
                coords=coords,
                zeff_tolerance=local_settings.zeff_tolerance,
                distance_tolerance=local_settings.distance_tolerance_angstrom,
            )
        )
    coordinates: list[GICForgePythonCoordinate] = []
    for diagnostic, primitives in records:
        if len(primitives) == 1:
            coordinate = _primitive_coordinate("Stre", len(coordinates) + 1, primitives[0])
            coordinates.append(
                GICForgePythonCoordinate(
                    name=coordinate.name,
                    block=coordinate.block,
                    terms=coordinate.terms,
                    type_index=coordinate.type_index,
                    diagnostic=f"{diagnostic} LOCAL_IRREP=A1 MODE=1",
                )
            )
            continue
        builder = _local_salc_coordinates if local_salc else _svd_local_coordinates
        coordinates.extend(
            builder(
                list(primitives),
                prefix="Stre",
                start=len(coordinates) + 1,
                kind_type_index=0,
                diagnostic=diagnostic,
                **({} if local_salc else {"coords": coords}),
            )
        )
    return coordinates


def _bond_primitive_domain_records(
    bond_coordinates: list[GICForgePythonCoordinate],
    *,
    effective_atomic_numbers: tuple[float, ...],
    coords: np.ndarray,
    neighbors: list[list[int]],
    selected_rings: list[tuple[int, ...]],
    settings: LocalSALCSettings,
) -> tuple[tuple[str, tuple[Primitive, ...]], ...]:
    primitives = [coordinate.terms[0][1] for coordinate in bond_coordinates]
    by_bond = {tuple(sorted(primitive.atoms)): primitive for primitive in primitives}
    assigned: set[tuple[int, int]] = set()
    records: list[tuple[str, tuple[Primitive, ...]]] = []

    for ring_index, ring in enumerate(selected_rings, start=1):
        ring_primitives = []
        for index, atom in enumerate(ring):
            key = tuple(sorted((atom, ring[(index + 1) % len(ring)])))
            primitive = by_bond.get(key)
            if primitive is not None and key not in assigned:
                ring_primitives.append(primitive)
                assigned.add(key)
        if ring_primitives:
            shared_edges = len(ring) - len(ring_primitives)
            if shared_edges:
                group, confidence, operations = "C1", "MEDIUM", 1
            else:
                group, confidence, operations = _ring_local_pseudogroup(
                    ring,
                    effective_atomic_numbers=effective_atomic_numbers,
                    coords=coords,
                    settings=settings,
                )
            if group == "C1":
                # C1 has no non-trivial local operation and hence no justified
                # equivalence orbit.  Combining the entire ring here would be
                # a delocalized basis rotation, not a local SALC, and creates
                # artificial off-diagonal force-field couplings.
                for primitive in ring_primitives:
                    records.append(
                        (
                            "LOCAL_SALC KIND=STRETCH "
                            f"DOMAIN=RING:{ring_index} GROUP=C1 "
                            f"CONFIDENCE={confidence} OPERATIONS={operations} "
                            f"SHARED_EDGES={shared_edges} SIZE=1 "
                            f"ATOMS={_primitive_atom_group_token((primitive,))}",
                            (primitive,),
                        )
                    )
            else:
                records.append(
                    (
                        "LOCAL_SALC KIND=STRETCH "
                        f"DOMAIN=RING:{ring_index} GROUP={group} CONFIDENCE={confidence} "
                        f"OPERATIONS={operations} SHARED_EDGES={shared_edges} "
                        f"SIZE={len(ring_primitives)} "
                        f"ATOMS={_primitive_atom_group_token(tuple(ring_primitives))}",
                        tuple(ring_primitives),
                    )
                )

    for center in range(len(neighbors)):
        incident: list[Primitive] = []
        for ligand in neighbors[center]:
            key = tuple(sorted((center, ligand)))
            if key in assigned or key not in by_bond:
                continue
            # A non-ring bond belongs to the more highly coordinated end;
            # atom number is the stable tie breaker.
            owner = center
            if (len(neighbors[ligand]), -ligand) > (len(neighbors[center]), -center):
                owner = ligand
            if owner != center:
                continue
            incident.append(by_bond[key])
            assigned.add(key)
        if not incident:
            continue
        for class_index, equivalent in enumerate(
            _bond_primitives_by_equivalence(
                [
                    GICForgePythonCoordinate("", "Stre", ((1.0, primitive),))
                    for primitive in incident
                ],
                effective_atomic_numbers=effective_atomic_numbers,
                coords=coords,
                zeff_tolerance=settings.zeff_tolerance,
                distance_tolerance=settings.distance_tolerance_angstrom,
            ),
            start=1,
        ):
            ligands = [
                next(atom for atom in primitive.atoms if atom != center) for primitive in equivalent
            ]
            classes = _local_ligand_equivalence_classes(
                center,
                ligands,
                effective_atomic_numbers=effective_atomic_numbers,
                coords=coords,
                zeff_tolerance=settings.zeff_tolerance,
                distance_tolerance=settings.distance_tolerance_angstrom,
            )
            match = _local_coordination_match(
                center,
                ligands,
                coords=coords,
                max_rms_cosine_error=settings.template_rms_threshold,
                min_score_margin=settings.template_min_margin,
            )
            group, confidence = _infer_local_pseudogroup(
                center,
                ligands,
                classes,
                coords=coords,
                settings=settings,
                match=match,
            )
            template_diagnostic = _local_template_diagnostic(match) if len(ligands) >= 5 else ""
            records.append(
                (
                    "LOCAL_SALC KIND=STRETCH "
                    f"DOMAIN=CENTER:{center + 1} CLASS={class_index} GROUP={group} "
                    f"CONFIDENCE={confidence} SIZE={len(equivalent)} "
                    f"ZEFF_TOL={settings.zeff_tolerance:.1e} "
                    f"DIST_TOL={settings.distance_tolerance_angstrom:.1e} "
                    f"{template_diagnostic} "
                    f"ATOMS={_primitive_atom_group_token(equivalent)}",
                    equivalent,
                )
            )

    for key, primitive in sorted(by_bond.items()):
        if key in assigned:
            continue
        records.append(
            (
                "LOCAL_SALC KIND=STRETCH DOMAIN=BOND:"
                f"{key[0] + 1}-{key[1] + 1} GROUP=C1 CONFIDENCE=HIGH SIZE=1 "
                f"ATOMS={_primitive_atom_group_token((primitive,))}",
                (primitive,),
            )
        )
    return tuple(records)


def _bond_primitives_by_equivalence(
    bond_coordinates: list[GICForgePythonCoordinate],
    *,
    effective_atomic_numbers: tuple[float, ...],
    coords: np.ndarray,
    zeff_tolerance: float = 5.0e-4,
    distance_tolerance: float = 1.0e-3,
) -> tuple[tuple[Primitive, ...], ...]:
    groups: list[list[Primitive]] = []
    keys: list[tuple[float, float, float]] = []
    for coordinate in bond_coordinates:
        _coefficient, primitive = coordinate.terms[0]
        first, second = primitive.atoms
        endpoint_key = sorted(
            (
                float(effective_atomic_numbers[int(first)]),
                float(effective_atomic_numbers[int(second)]),
            )
        )
        distance = float(np.linalg.norm(coords[int(first)] - coords[int(second)]))
        key = (endpoint_key[0], endpoint_key[1], distance)
        match = next(
            (
                index
                for index, other in enumerate(keys)
                if abs(key[0] - other[0]) <= zeff_tolerance
                and abs(key[1] - other[1]) <= zeff_tolerance
                and abs(key[2] - other[2]) <= distance_tolerance
            ),
            None,
        )
        if match is None:
            keys.append(key)
            groups.append([primitive])
            continue
        groups[match].append(primitive)
    return tuple(tuple(group) for _key, group in sorted(zip(keys, groups)))


def _primitive_atom_group_token(primitives: tuple[Primitive, ...] | list[Primitive]) -> str:
    return "+".join(
        "-".join(str(int(atom) + 1) for atom in primitive.atoms) for primitive in primitives
    )


def _cyclic_index(index_1based: int, ncyc: int) -> int:
    while index_1based > ncyc:
        index_1based -= ncyc
    while index_1based <= 0:
        index_1based += ncyc
    return index_1based - 1


def _c2v3_angle_coordinates(
    center: int,
    neigh: list[int],
    *,
    atomic_numbers: tuple[int, ...],
    effective_atomic_numbers: tuple[float, ...],
    neighbors: list[list[int]],
    atom_ring: list[int],
    coords: np.ndarray,
    start: int,
) -> list[GICForgePythonCoordinate]:
    first, second, third = neigh
    classes = _local_ligand_equivalence_classes(
        center,
        neigh,
        effective_atomic_numbers=effective_atomic_numbers,
        coords=coords,
    )
    class_sizes = {atom: len(group) for group in classes for atom in group}
    singleton_atoms = [atom for atom in neigh if class_sizes[atom] == 1]
    if len(singleton_atoms) == 1:
        different = singleton_atoms[0]
    elif len(singleton_atoms) == 3:
        different = first
        if atomic_numbers[second] == 1:
            different = second
        elif atomic_numbers[third] == 1:
            different = third
        elif len(neighbors[second]) == 1:
            different = second
        elif len(neighbors[third]) == 1:
            different = third
    elif not singleton_atoms:
        different = first
    else:
        different = singleton_atoms[0]

    if atom_ring[center] != 0:
        if atom_ring[first] != 0 and atom_ring[second] != 0:
            different = third
        if atom_ring[first] != 0 and atom_ring[third] != 0:
            different = second
        if atom_ring[second] != 0 and atom_ring[third] != 0:
            different = first

    if different == first:
        jat, kat, lat = first, second, third
    elif different == second:
        jat, kat, lat = second, first, third
    else:
        jat, kat, lat = third, second, first

    if atom_ring[center] != 0 and atom_ring[jat] != 0:
        return []

    coords: list[GICForgePythonCoordinate] = []
    if atom_ring[center] == 0:
        den = np.sqrt(6.0)
        coords.append(
            GICForgePythonCoordinate(
                name=f"SymD{start:04d}",
                block="SymD",
                type_index=1,
                terms=(
                    (2.0 / den, Primitive("angle", (kat, center, lat))),
                    (-1.0 / den, Primitive("angle", (jat, center, kat))),
                    (-1.0 / den, Primitive("angle", (jat, center, lat))),
                ),
            )
        )
        start += 1
    den = np.sqrt(2.0)
    coords.append(
        GICForgePythonCoordinate(
            name=f"Rock{start:04d}",
            block="Rock",
            type_index=2,
            terms=(
                (1.0 / den, Primitive("angle", (jat, center, kat))),
                (-1.0 / den, Primitive("angle", (jat, center, lat))),
            ),
        )
    )
    return coords


def _four_atom_angle_coordinates(
    center: int,
    neigh: list[int],
    *,
    atomic_numbers: tuple[int, ...],
    effective_atomic_numbers: tuple[float, ...],
    neighbors: list[list[int]],
    atom_ring: list[int],
    coords: np.ndarray,
    force_xy3_salc: bool = False,
    start: int,
) -> list[GICForgePythonCoordinate]:
    if force_xy3_salc:
        xy3_domains = []
        for z_atom in neigh:
            y_atoms = tuple(atom for atom in neigh if atom != z_atom)
            if (
                len({atomic_numbers[atom] for atom in y_atoms}) == 1
                and len({len(neighbors[atom]) for atom in y_atoms}) == 1
            ):
                xy3_domains.append((z_atom, tuple(sorted(y_atoms))))
        if xy3_domains:
            z_atom, y_atoms = max(
                xy3_domains,
                key=lambda item: (
                    len({atomic_numbers[atom] for atom in item[1]}),
                    round(effective_atomic_numbers[item[0]] / 5.0e-4),
                    -item[0],
                ),
            )
            frozen = {
                atom: atom_ring[atom] != 0 and atom_ring[center] != 0 for atom in (z_atom, *y_atoms)
            }
            return _wxy3_coordinates(
                center,
                z_atom,
                y_atoms[0],
                y_atoms[1],
                y_atoms[2],
                frozen=frozen,
                start=start,
            )
    first, second, third, fourth = _order_four_atom_neighbors(
        tuple(neigh),
        center=center,
        effective_atomic_numbers=effective_atomic_numbers,
        atom_ring=atom_ring,
        coords=coords,
    )
    frozen = {
        atom: atom_ring[atom] != 0 and atom_ring[center] != 0
        for atom in (first, second, third, fourth)
    }
    equal_count = _four_atom_equal_count((first, second, third, fourth), effective_atomic_numbers)
    pivot_count = sum(1 for atom in (first, second, third, fourth) if frozen[atom])
    if equal_count == 4:
        return _td_four_atom_coordinates(
            center,
            first,
            second,
            third,
            fourth,
            frozen=frozen,
            start=start,
        )
    if equal_count == 3 or pivot_count in {1, 3}:
        return _wxy3_coordinates(
            center,
            first,
            second,
            third,
            fourth,
            frozen=frozen,
            start=start,
        )
    return _w2xy2_coordinates(
        center,
        first,
        second,
        third,
        fourth,
        equal_count=equal_count,
        frozen=frozen,
        start=start,
    )


def _xy3_angle_domain(
    center: int,
    neigh: list[int],
    *,
    atomic_numbers: tuple[int, ...],
    effective_atomic_numbers: tuple[float, ...],
    neighbors: list[list[int]],
    atom_ring: list[int],
) -> tuple[int, tuple[int, int, int]] | None:
    if len(neigh) != 4 or atom_ring[center] != 0:
        return None
    candidates = []
    for z_atom in neigh:
        y_atoms = tuple(sorted(atom for atom in neigh if atom != z_atom))
        if _branches_are_equivalent(
            y_atoms,
            excluded_center=center,
            atomic_numbers=atomic_numbers,
            effective_atomic_numbers=effective_atomic_numbers,
            neighbors=neighbors,
            atom_ring=atom_ring,
        ) and not _branches_are_equivalent(
            (*y_atoms, z_atom),
            excluded_center=center,
            atomic_numbers=atomic_numbers,
            effective_atomic_numbers=effective_atomic_numbers,
            neighbors=neighbors,
            atom_ring=atom_ring,
        ):
            candidates.append((z_atom, y_atoms))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            round(effective_atomic_numbers[item[0]] / 5.0e-4),
            atomic_numbers[item[0]],
            -item[0],
        ),
    )


def _xy2_angle_domain(
    center: int,
    neigh: list[int],
    *,
    atomic_numbers: tuple[int, ...],
    effective_atomic_numbers: tuple[float, ...],
    neighbors: list[list[int]],
    atom_ring: list[int],
) -> tuple[int, tuple[int, int]] | None:
    if len(neigh) != 3 or atom_ring[center] != 0:
        return None
    candidates = []
    for z_atom in neigh:
        y_atoms = tuple(sorted(atom for atom in neigh if atom != z_atom))
        if _branches_are_equivalent(
            y_atoms,
            excluded_center=center,
            atomic_numbers=atomic_numbers,
            effective_atomic_numbers=effective_atomic_numbers,
            neighbors=neighbors,
            atom_ring=atom_ring,
        ) and not _branches_are_equivalent(
            (*y_atoms, z_atom),
            excluded_center=center,
            atomic_numbers=atomic_numbers,
            effective_atomic_numbers=effective_atomic_numbers,
            neighbors=neighbors,
            atom_ring=atom_ring,
        ):
            candidates.append((z_atom, y_atoms))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            round(effective_atomic_numbers[item[0]] / 5.0e-4),
            atomic_numbers[item[0]],
            -item[0],
        ),
    )


def _tricoordinate_out_of_plane_atom_orders(
    center: int,
    substituents: tuple[int, int, int],
    *,
    atomic_numbers: tuple[int, ...],
    effective_atomic_numbers: tuple[float, ...],
    neighbors: list[list[int]],
    atom_ring: list[int],
    coords: np.ndarray,
) -> tuple[tuple[int, int, int, int], ...]:
    """Choose a continuous native-U chart for a tricoordinate center."""

    ordered = tuple(sorted(substituents))
    ring_substituents = tuple(
        atom
        for atom in substituents
        if atom_ring[atom] != 0 and atom_ring[atom] == atom_ring[center]
    )
    if len(ring_substituents) == 2:
        exocyclic = next(atom for atom in ordered if atom not in ring_substituents)
        return ((center, *ring_substituents, exocyclic),)
    all_equivalent = _branches_are_equivalent(
        ordered,
        excluded_center=center,
        atomic_numbers=atomic_numbers,
        effective_atomic_numbers=effective_atomic_numbers,
        neighbors=neighbors,
        atom_ring=atom_ring,
    )
    if all_equivalent:
        first, second, third = ordered
        return (
            (center, first, second, third),
            (center, second, third, first),
            (center, third, first, second),
        )

    equivalent_pairs = []
    for first, second in combinations(ordered, 2):
        if _branches_are_equivalent(
            (first, second),
            excluded_center=center,
            atomic_numbers=atomic_numbers,
            effective_atomic_numbers=effective_atomic_numbers,
            neighbors=neighbors,
            atom_ring=atom_ring,
        ):
            unique = next(atom for atom in ordered if atom not in {first, second})
            equivalent_pairs.append((first, second, unique))
    if len(equivalent_pairs) == 1:
        first, second, unique = equivalent_pairs[0]
        return ((center, first, second, unique),)

    c2v_order = _c2v_like_tricoordinate_order(center, ordered, coords=coords)
    if c2v_order is not None:
        return (c2v_order,)
    first, second, third = ordered
    return (
        (center, first, second, third),
        (center, second, third, first),
        (center, third, first, second),
    )


def _c2v_like_tricoordinate_order(
    center: int,
    substituents: tuple[int, int, int],
    *,
    coords: np.ndarray,
    pair_margin: float = 0.45,
    absolute_tolerance: float = 2.0e-2,
) -> tuple[int, int, int, int] | None:
    vectors = {
        atom: np.asarray(coords[atom] - coords[center], dtype=float) for atom in substituents
    }
    lengths = {atom: float(np.linalg.norm(vector)) for atom, vector in vectors.items()}
    if any(length <= 1.0e-12 for length in lengths.values()):
        return None

    def local_angle(first: int, second: int) -> float:
        cosine = float(np.dot(vectors[first], vectors[second]) / (lengths[first] * lengths[second]))
        return float(np.arccos(np.clip(cosine, -1.0, 1.0)))

    scored = []
    for first, second in combinations(substituents, 2):
        unique = next(atom for atom in substituents if atom not in {first, second})
        length_scale = 0.5 * (lengths[first] + lengths[second])
        length_mismatch = abs(lengths[first] - lengths[second]) / length_scale
        angle_mismatch = abs(local_angle(first, unique) - local_angle(second, unique))
        scored.append((float(np.hypot(length_mismatch, angle_mismatch)), first, second, unique))
    scored.sort()
    best, runner_up = scored[0], scored[1]
    if best[0] > absolute_tolerance or best[0] > pair_margin * max(runner_up[0], 1.0e-12):
        return None
    return (center, best[1], best[2], best[3])


def _branches_are_equivalent(
    roots: tuple[int, ...],
    *,
    excluded_center: int,
    atomic_numbers: tuple[int, ...],
    effective_atomic_numbers: tuple[float, ...],
    neighbors: list[list[int]],
    atom_ring: list[int],
) -> bool:
    """Conservatively recognize equivalent substituent branches.

    Terminal identical atoms are exactly permutable even in a distorted
    geometry.  Nonterminal branches must additionally have the same rooted
    graph environment after removing the special-domain center.  The
    refinement is atom-number invariant and deliberately ignores distances,
    so a persistent domain cannot disappear during an optimization merely
    because its geometry becomes asymmetric.
    """

    if len(roots) < 2:
        return False
    if len({atomic_numbers[root] for root in roots}) != 1:
        return False
    if all(len(neighbors[root]) == 1 for root in roots):
        return True

    active = tuple(atom for atom in range(len(neighbors)) if atom != excluded_center)
    labels: dict[int, object] = {
        atom: (
            atomic_numbers[atom],
            round(effective_atomic_numbers[atom] / 5.0e-4),
            atom_ring[atom] != 0,
            sum(1 for other in neighbors[atom] if other != excluded_center),
        )
        for atom in active
    }
    # One-dimensional Weisfeiler-Lehman refinement distinguishes local branch
    # environments without introducing atom labels into the signature.
    for _iteration in range(len(active)):
        raw = {
            atom: (
                labels[atom],
                tuple(
                    sorted(labels[other] for other in neighbors[atom] if other != excluded_center)
                ),
            )
            for atom in active
        }
        unique = {value: index for index, value in enumerate(sorted(set(raw.values())))}
        refined = {atom: unique[value] for atom, value in raw.items()}
        if all(refined[atom] == labels[atom] for atom in active):
            break
        labels = refined
    return len({labels[root] for root in roots}) == 1


def _xy2_angle_coordinates(
    center: int,
    z_atom: int,
    first_y: int,
    second_y: int,
    *,
    start: int,
) -> list[GICForgePythonCoordinate]:
    diagnostic = (
        f"LOCAL_SALC KIND=XY2_ANGLE DOMAIN=CENTER:{center + 1} "
        f"Z={z_atom + 1} Y={first_y + 1},{second_y + 1} "
        "GROUP=C2v BASIS=A1_PLUS_B NORMALIZATION=ORTHONORMAL"
    )
    return [
        GICForgePythonCoordinate(
            name=f"SymD{start:04d}",
            block="SymD",
            type_index=1,
            terms=(
                (2.0 / np.sqrt(6.0), Primitive("angle", (first_y, center, second_y))),
                (-1.0 / np.sqrt(6.0), Primitive("angle", (z_atom, center, first_y))),
                (-1.0 / np.sqrt(6.0), Primitive("angle", (z_atom, center, second_y))),
            ),
            diagnostic=diagnostic + " IRREP=A1",
        ),
        GICForgePythonCoordinate(
            name=f"Rock{start + 1:04d}",
            block="Rock",
            type_index=2,
            terms=(
                (1.0 / np.sqrt(2.0), Primitive("angle", (z_atom, center, first_y))),
                (-1.0 / np.sqrt(2.0), Primitive("angle", (z_atom, center, second_y))),
            ),
            diagnostic=diagnostic + " IRREP=B",
        ),
    ]


def _order_four_atom_neighbors(
    atoms: tuple[int, int, int, int],
    *,
    center: int,
    effective_atomic_numbers: tuple[float, ...],
    atom_ring: list[int],
    coords: np.ndarray,
) -> tuple[int, int, int, int]:
    jat, kat, lat, mat = atoms
    threshold = 5.0e-4
    classes = _local_ligand_equivalence_classes(
        center,
        list(atoms),
        effective_atomic_numbers=effective_atomic_numbers,
        coords=coords,
    )
    ordered_classes = sorted(classes, key=lambda group: (-len(group), group))
    if len(ordered_classes) == 1:
        jat, kat, lat, mat = ordered_classes[0]
    elif len(ordered_classes) == 2:
        first_class, second_class = ordered_classes
        if len(first_class) == 3:
            kat, lat, mat = first_class
            jat = second_class[0]
        elif len(first_class) == 2 and len(second_class) == 2:
            jat, kat = first_class
            lat, mat = second_class
        else:
            jat, kat, lat = first_class
            mat = second_class[0]
    elif len(ordered_classes) == 3:
        pair = next(group for group in ordered_classes if len(group) == 2)
        singles = [atom for group in ordered_classes if len(group) == 1 for atom in group]
        jat, kat = pair
        lat, mat = singles
    else:
        jat, kat, lat, mat = atoms

    pivot_count = 0
    if atom_ring[center] != 0:
        pivot_count = sum(1 for atom in (jat, kat, lat, mat) if atom_ring[atom] != 0)
    if pivot_count == 1:
        if atom_ring[jat] != 0:
            return jat, kat, lat, mat
        if atom_ring[kat] != 0:
            return kat, jat, lat, mat
        if atom_ring[lat] != 0:
            return lat, kat, jat, mat
        if atom_ring[mat] != 0:
            return mat, kat, lat, jat
    if pivot_count == 3:
        if atom_ring[jat] == 0:
            ordered = (jat, kat, lat, mat)
        elif atom_ring[kat] == 0:
            ordered = (kat, jat, lat, mat)
        elif atom_ring[lat] == 0:
            ordered = (lat, kat, jat, mat)
        else:
            ordered = (mat, kat, lat, jat)
        jat, kat, lat, mat = ordered
        if abs(effective_atomic_numbers[kat] - effective_atomic_numbers[lat]) < threshold:
            if abs(effective_atomic_numbers[lat] - effective_atomic_numbers[mat]) >= threshold:
                kat, mat = mat, kat
        elif abs(effective_atomic_numbers[lat] - effective_atomic_numbers[mat]) >= threshold:
            kat, lat = lat, kat
    return jat, kat, lat, mat


def _four_atom_equal_count(
    atoms: tuple[int, int, int, int], effective_atomic_numbers: tuple[float, ...]
) -> int:
    jat, kat, lat, mat = atoms
    threshold = 5.0e-4

    def equivalent(first: int, second: int) -> bool:
        return abs(effective_atomic_numbers[first] - effective_atomic_numbers[second]) < threshold

    neq = 0
    if equivalent(jat, kat):
        neq += 1
        if equivalent(jat, lat):
            neq += 2
            if equivalent(jat, mat):
                neq += 1
        elif equivalent(jat, mat):
            neq += 2
        elif equivalent(lat, mat):
            neq += 1
    elif equivalent(jat, lat):
        neq += 1
        if equivalent(jat, mat):
            neq += 2
        elif equivalent(kat, mat):
            neq += 1
    elif equivalent(jat, mat):
        neq += 1
        if equivalent(kat, lat):
            neq += 1
    elif equivalent(kat, lat):
        neq += 1
        if equivalent(lat, mat):
            neq += 2
    elif equivalent(kat, mat):
        neq += 1
    elif equivalent(lat, mat):
        neq += 1
    return neq


def _is_spiro_center(center: int, neighbors: list[int], ring_counts: list[int]) -> bool:
    return (
        len(neighbors) == 4
        and ring_counts[center] == 2
        and all(ring_counts[neighbor] == 1 for neighbor in neighbors)
    )


def _spiro_angle_coordinates(
    center: int,
    neigh: list[int],
    *,
    selected_rings: list[tuple[int, ...]],
    start: int,
) -> list[GICForgePythonCoordinate]:
    first, second, third, fourth = _order_spiro_neighbors(center, tuple(neigh), selected_rings)
    den = np.sqrt(2.0)
    cross_angles = (
        Primitive("angle", (first, center, third)),
        Primitive("angle", (first, center, fourth)),
        Primitive("angle", (second, center, third)),
        Primitive("angle", (second, center, fourth)),
    )
    patterns = (
        (1.0, 1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0, -1.0),
    )
    return [
        GICForgePythonCoordinate(
            name=f"Spir{start + offset:04d}",
            block="Spir",
            type_index=16,
            terms=tuple(
                (coefficient / den, primitive)
                for coefficient, primitive in zip(pattern, cross_angles)
            ),
        )
        for offset, pattern in enumerate(patterns)
    ]


def _order_spiro_neighbors(
    center: int,
    neighbors: tuple[int, int, int, int],
    selected_rings: list[tuple[int, ...]],
) -> tuple[int, int, int, int]:
    membership: dict[int, int] = {}
    for ring_index, ring in enumerate(selected_rings, start=1):
        if center not in ring:
            continue
        for atom in ring:
            if atom != center:
                membership.setdefault(atom, ring_index)

    first, second, third, fourth = neighbors
    first_ring = membership.get(first, 0)
    if membership.get(second, -1) == first_ring:
        return first, second, third, fourth
    if membership.get(third, -1) == first_ring:
        return first, third, second, fourth
    return first, fourth, second, third


def _w2xy2_coordinates(
    center: int,
    jat: int,
    kat: int,
    lat: int,
    mat: int,
    *,
    equal_count: int,
    frozen: dict[int, bool],
    start: int,
) -> list[GICForgePythonCoordinate]:
    if all(frozen[atom] for atom in (jat, kat, lat, mat)):
        return []
    inot1, inot2 = jat, kat
    iyes1, iyes2 = lat, mat
    if frozen[jat] and frozen[lat]:
        inot1, inot2 = jat, lat
        iyes1, iyes2 = kat, mat
    elif frozen[jat] and frozen[mat]:
        inot1, inot2 = jat, mat
        iyes1, iyes2 = kat, lat
    elif frozen[kat] and frozen[lat]:
        inot1, inot2 = kat, lat
        iyes1, iyes2 = jat, mat
    elif frozen[kat] and frozen[mat]:
        inot1, inot2 = kat, mat
        iyes1, iyes2 = jat, lat
    elif frozen[lat] and frozen[mat]:
        inot1, inot2 = lat, mat
        iyes1, iyes2 = jat, kat

    den_sym = np.sqrt(6.0)
    den_rock = np.sqrt(2.0)
    coordinates = [
        GICForgePythonCoordinate(
            name=f"SymD{start:04d}",
            block="SymD",
            type_index=1,
            terms=(
                (2.0 / den_sym, Primitive("angle", (iyes1, center, iyes2))),
                (-1.0 / den_sym, Primitive("angle", (inot1, center, iyes1))),
                (-1.0 / den_sym, Primitive("angle", (inot1, center, iyes2))),
            ),
        ),
        GICForgePythonCoordinate(
            name=f"Rock{start + 1:04d}",
            block="Rock",
            type_index=2,
            terms=(
                (1.0 / den_rock, Primitive("angle", (inot1, center, iyes1))),
                (-1.0 / den_rock, Primitive("angle", (inot1, center, iyes2))),
            ),
        ),
        GICForgePythonCoordinate(
            name=f"SymD{start + 2:04d}",
            block="SymD",
            type_index=1,
            terms=(
                (2.0 / den_sym, Primitive("angle", (iyes1, center, iyes2))),
                (-1.0 / den_sym, Primitive("angle", (inot2, center, iyes1))),
                (-1.0 / den_sym, Primitive("angle", (inot2, center, iyes2))),
            ),
        ),
        GICForgePythonCoordinate(
            name=f"Rock{start + 3:04d}",
            block="Rock",
            type_index=2,
            terms=(
                (1.0 / den_rock, Primitive("angle", (inot2, center, iyes1))),
                (-1.0 / den_rock, Primitive("angle", (inot2, center, iyes2))),
            ),
        ),
    ]
    if not any(frozen.values()):
        coordinates.append(
            _primitive_coordinate(
                "Bend" if equal_count != 2 else "Bend",
                start + 4,
                Primitive("angle", (inot1, center, inot2)),
            )
        )
    return coordinates


def _square_planar_out_of_plane_coordinates(
    center: int,
    cyclic: tuple[int, int, int, int],
    *,
    start: int,
) -> list[GICForgePythonCoordinate]:
    """Return the two canonical D4h out-of-plane SALCs for a square plane."""

    if len(cyclic) != 4:
        raise ValueError("square-planar mean height requires exactly four ligands")
    primitives = tuple(
        Primitive(
            "out_of_plane_height",
            (center, cyclic[(index + 1) % 4], cyclic[(index + 2) % 4], atom),
        )
        for index, atom in enumerate(cyclic)
    )
    coefficient_rows = (
        (0.5, 0.5, 0.5, 0.5),
        (0.5, -0.5, 0.5, -0.5),
    )
    irreps = ("A2U", "B2U")
    return [
        GICForgePythonCoordinate(
            name=f"OopH{start + mode:04d}",
            block="OopH",
            type_index=12,
            terms=tuple(zip(coefficients, primitives, strict=True)),
            diagnostic=(
                "LOCAL_SALC DOMAIN=CENTER:"
                f"{center + 1} KIND=OUT_OF_PLANE TEMPLATE=SQUARE_PLANAR "
                f"LOCAL_IRREP={irreps[mode]} CANONICAL_CATALOG=YES"
            ),
        )
        for mode, coefficients in enumerate(coefficient_rows)
    ]


@lru_cache(maxsize=None)
def _template_angle_salc_catalog(
    template: LocalCoordinationTemplate,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[float, ...], ...]]:
    """Return the fixed angular SALC basis of an ideal coordination template.

    The Gram operator is built only from the immutable ideal polyhedron.  Its
    nonzero eigenspace is consequently a template property, not a fit to the
    molecular geometry.  Degenerate eigenvectors are legitimate partner
    functions of the same local representation; canonical signs make the
    stored ordering reproducible.
    """

    directions = np.asarray(template.directions, dtype=float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    pairs = tuple(
        (first, second)
        for first, second in combinations(range(len(directions)), 2)
        if float(np.dot(directions[first], directions[second])) > -1.0 + 1.0e-8
    )
    ideal_coords = np.vstack((np.zeros((1, 3), dtype=float), directions))
    primitives = tuple(Primitive("angle", (first + 1, 0, second + 1)) for first, second in pairs)
    primitive_b = b_matrix_analytic(primitives, ideal_coords)
    gram = primitive_b @ primitive_b.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    scale = max(0.0, float(eigenvalues[-1])) if eigenvalues.size else 0.0
    threshold = max(1.0e-10, 1.0e-8 * scale)
    selected = [
        index
        for index in range(len(eigenvalues) - 1, -1, -1)
        if float(eigenvalues[index]) > threshold
    ]
    expected_rank = 2 * len(directions) - 3
    if template.name == "SQUARE_PLANAR":
        expected_rank -= 2
    if len(selected) != expected_rank:
        raise ValueError(
            f"ideal angular SALC catalog for {template.name} has rank "
            f"{len(selected)}, expected {expected_rank}"
        )
    rows = []
    for index in selected:
        row = _canonical_svd_coefficients(eigenvectors[:, index].astype(float))
        rows.append(tuple(float(value) for value in row))
    return pairs, tuple(rows)


def _template_ligand_assignment(
    center: int,
    neigh: list[int],
    *,
    template: LocalCoordinationTemplate,
    coords: np.ndarray,
) -> tuple[int, ...] | None:
    """Map ideal template slots to ligands using the frozen cosine graph."""

    directions = np.asarray(template.directions, dtype=float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    actual = _local_ligand_unit_vectors(center, neigh, coords)
    size = len(neigh)
    if directions.shape != (size, 3):
        return None
    cosine_classes = _template_pair_cosine_classes(template)
    ideal_class = np.full((size, size), -1, dtype=int)
    actual_class = np.full((size, size), -1, dtype=int)
    for first, second in combinations(range(size), 2):
        ideal_class[first, second] = ideal_class[second, first] = _nearest_cosine_class(
            float(np.dot(directions[first], directions[second])), cosine_classes
        )
        actual_class[first, second] = actual_class[second, first] = _nearest_cosine_class(
            float(np.dot(actual[first], actual[second])), cosine_classes
        )
    ideal_signatures = tuple(
        tuple(sorted(int(ideal_class[first, second]) for second in range(size) if second != first))
        for first in range(size)
    )
    actual_signatures = tuple(
        tuple(sorted(int(actual_class[first, second]) for second in range(size) if second != first))
        for first in range(size)
    )
    assigned = [-1] * size
    used: set[int] = set()

    def assign(slot: int) -> bool:
        if slot == size:
            return True
        for actual_index in range(size):
            if actual_index in used or ideal_signatures[slot] != actual_signatures[actual_index]:
                continue
            if any(
                ideal_class[slot, previous] != actual_class[actual_index, assigned[previous]]
                for previous in range(slot)
            ):
                continue
            assigned[slot] = actual_index
            used.add(actual_index)
            if assign(slot + 1):
                return True
            used.remove(actual_index)
            assigned[slot] = -1
        return False

    if assign(0):
        return tuple(neigh[index] for index in assigned)

    # Near a cosine-class boundary the discrete graph can change even though
    # the template decision remains valid.  Recover the same frozen slot map
    # by aligning one maximally independent ideal triad and solving the finite
    # assignment problem.  This SVD aligns frames; it never constructs or
    # selects a SALC.
    from scipy.optimize import linear_sum_assignment

    anchor = max(
        combinations(range(size), 3),
        key=lambda triple: abs(float(np.linalg.det(directions[list(triple)]))),
    )
    best_score = float("inf")
    best_assignment: tuple[int, ...] | None = None
    for actual_anchor in permutations(range(size), 3):
        left = directions[list(anchor)]
        right = actual[list(actual_anchor)]
        u_matrix, _singular, vh = np.linalg.svd(left.T @ right)
        rotation = u_matrix @ vh
        cost = 1.0 - directions @ rotation @ actual.T
        rows, columns = linear_sum_assignment(cost)
        trial = np.empty(size, dtype=int)
        trial[rows] = columns
        difference = directions @ directions.T - actual[trial] @ actual[trial].T
        upper = difference[np.triu_indices(size, k=1)]
        score = float(np.sqrt(np.mean(upper * upper)))
        trial_atoms = tuple(neigh[index] for index in trial)
        if score < best_score - 1.0e-14 or (
            abs(score - best_score) <= 1.0e-14
            and (best_assignment is None or trial_atoms < best_assignment)
        ):
            best_score = score
            best_assignment = trial_atoms
    return best_assignment


def _template_angle_salc_coordinates(
    center: int,
    neigh: list[int],
    *,
    template: LocalCoordinationTemplate,
    match: LocalCoordinationMatch,
    coords: np.ndarray,
    settings: LocalSALCSettings | None = None,
    start: int,
) -> list[GICForgePythonCoordinate] | None:
    """Instantiate the fixed ideal-template angular SALCs on one center."""

    assignment = _template_ligand_assignment(
        center,
        neigh,
        template=template,
        coords=coords,
    )
    if assignment is None:
        return None
    local_settings = settings or LocalSALCSettings()
    pairs, coefficient_rows = _template_angle_salc_catalog(template)
    primitives = tuple(
        Primitive(
            "angle",
            (
                min(assignment[first], assignment[second]),
                center,
                max(assignment[first], assignment[second]),
            ),
        )
        for first, second in pairs
    )
    group, confidence = _infer_local_pseudogroup(
        center,
        neigh,
        (tuple(neigh),),
        coords=coords,
        match=match,
    )
    coordinates = [
        GICForgePythonCoordinate(
            name=f"HCAn{start + mode:04d}",
            block="HCAn",
            type_index=17,
            terms=tuple(
                (float(coefficient), primitive)
                for coefficient, primitive in zip(coefficients, primitives, strict=True)
                if abs(float(coefficient)) > 1.0e-12
            ),
            diagnostic=(
                f"LOCAL_SALC KIND=ANGLE DOMAIN=CENTER:{center + 1} "
                f"GROUP={group} CONFIDENCE={confidence} SOURCE=TEMPLATE "
                f"ZEFF_TOL={local_settings.zeff_tolerance:.1e} "
                f"DIST_TOL={local_settings.distance_tolerance_angstrom:.1e} "
                f"TEMPLATE_RMS_TOL={local_settings.template_rms_threshold:.6g} "
                f"TEMPLATE_MARGIN_TOL={local_settings.template_min_margin:.6g} "
                f"{_local_template_diagnostic(match)} "
                f"LOCAL_IRREP={group}_BEND_{mode + 1} MODE={mode + 1} "
                "CANONICAL_CATALOG=YES CATALOG_SOURCE=IDEAL_TEMPLATE_GRAM"
            ),
        )
        for mode, coefficients in enumerate(coefficient_rows)
    ]
    if template.name == "SQUARE_PLANAR":
        cyclic = (assignment[0], assignment[2], assignment[1], assignment[3])
        coordinates.extend(
            _square_planar_out_of_plane_coordinates(
                center,
                cyclic,
                start=start + len(coordinates),
            )
        )
    return coordinates


def _wxy3_coordinates(
    center: int,
    jat: int,
    kat: int,
    lat: int,
    mat: int,
    *,
    frozen: dict[int, bool],
    start: int,
) -> list[GICForgePythonCoordinate]:
    if all(frozen[atom] for atom in (jat, kat, lat, mat)):
        return []
    if not any(frozen.values()):
        den_e1 = np.sqrt(6.0)
        den_e2 = np.sqrt(2.0)
        den_a1 = np.sqrt(3.0)
        diagnostic = (
            f"LOCAL_SALC KIND=XY3_ANGLE DOMAIN=CENTER:{center + 1} "
            f"Z={jat + 1} Y={kat + 1},{lat + 1},{mat + 1} "
            "GROUP=C3v BASIS=A1_PLUS_2E NORMALIZATION=ORTHONORMAL"
        )
        zxy_e1 = GICForgePythonCoordinate(
            name=f"Rock{start:04d}",
            block="Rock",
            type_index=2,
            terms=(
                (2.0 / den_e1, Primitive("angle", (jat, center, kat))),
                (-1.0 / den_e1, Primitive("angle", (jat, center, lat))),
                (-1.0 / den_e1, Primitive("angle", (jat, center, mat))),
            ),
            diagnostic=diagnostic + " IRREP=E COMPONENT=1 SOURCE=ZXY",
        )
        zxy_e2 = GICForgePythonCoordinate(
            name=f"Rock{start + 1:04d}",
            block="Rock",
            type_index=2,
            terms=(
                (1.0 / den_e2, Primitive("angle", (jat, center, lat))),
                (-1.0 / den_e2, Primitive("angle", (jat, center, mat))),
            ),
            diagnostic=diagnostic + " IRREP=E COMPONENT=2 SOURCE=ZXY",
        )
        zxy_a1 = GICForgePythonCoordinate(
            name=f"SymD{start + 2:04d}",
            block="SymD",
            type_index=1,
            terms=(
                (1.0 / den_a1, Primitive("angle", (jat, center, kat))),
                (1.0 / den_a1, Primitive("angle", (jat, center, lat))),
                (1.0 / den_a1, Primitive("angle", (jat, center, mat))),
            ),
            diagnostic=(
                diagnostic
                + " IRREP=A1 SOURCE=ZXY ROLE=UMBRELLA CHART=GLOBAL_REGULAR"
            ),
        )
        yxy_e1 = GICForgePythonCoordinate(
            name=f"Rock{start + 3:04d}",
            block="Rock",
            type_index=2,
            terms=(
                (2.0 / den_e1, Primitive("angle", (kat, center, lat))),
                (-1.0 / den_e1, Primitive("angle", (kat, center, mat))),
                (-1.0 / den_e1, Primitive("angle", (lat, center, mat))),
            ),
            diagnostic=diagnostic + " IRREP=E COMPONENT=1 SOURCE=YXY",
        )
        yxy_e2 = GICForgePythonCoordinate(
            name=f"Rock{start + 4:04d}",
            block="Rock",
            type_index=2,
            terms=(
                (1.0 / den_e2, Primitive("angle", (kat, center, mat))),
                (-1.0 / den_e2, Primitive("angle", (lat, center, mat))),
            ),
            diagnostic=diagnostic + " IRREP=E COMPONENT=2 SOURCE=YXY",
        )

        # Use the symmetric Z-X-Y umbrella at every true Z-X(Y)3 center.
        # The alternative symmetric sum of the three Y-X-Y angles becomes
        # identically 2*pi at planarity and therefore has a vanishing Wilson
        # row.  A geometry-independent chart avoids that singularity without
        # a local SVD, branch, or discontinuous coordinate redefinition.
        return [zxy_e1, zxy_e2, zxy_a1, yxy_e1, yxy_e2]

    den = np.sqrt(2.0)
    coordinates = [
        GICForgePythonCoordinate(
            name=f"Rock{start:04d}",
            block="Rock",
            type_index=2,
            terms=(
                (0.5, Primitive("angle", (jat, center, kat))),
                (-0.25, Primitive("angle", (jat, center, lat))),
                (-0.25, Primitive("angle", (jat, center, mat))),
            ),
        ),
        GICForgePythonCoordinate(
            name=f"Rock{start + 1:04d}",
            block="Rock",
            type_index=2,
            terms=(
                (1.0 / den, Primitive("angle", (jat, center, lat))),
                (-1.0 / den, Primitive("angle", (jat, center, mat))),
            ),
        ),
    ]
    if sum(1 for value in frozen.values() if value) == 3:
        return coordinates
    coordinates.append(
        _primitive_coordinate("Bend", start + 2, Primitive("angle", (lat, center, mat)))
    )
    if sum(1 for value in frozen.values() if value) == 2:
        return coordinates
    coordinates.append(
        _primitive_coordinate("Bend", start + 3, Primitive("angle", (kat, center, lat)))
    )
    coordinates.append(
        _primitive_coordinate("Bend", start + 4, Primitive("angle", (kat, center, mat)))
    )
    return coordinates


def _td_four_atom_coordinates(
    center: int,
    jat: int,
    kat: int,
    lat: int,
    mat: int,
    *,
    frozen: dict[int, bool],
    start: int,
) -> list[GICForgePythonCoordinate]:
    den_ea = np.sqrt(12.0)
    den_eb = 2.0
    den_t2 = np.sqrt(2.0)
    return [
        GICForgePythonCoordinate(
            name=f"EEee{start:04d}",
            block="EEee",
            type_index=8,
            terms=(
                (2.0 / den_ea, Primitive("angle", (jat, center, kat))),
                (-1.0 / den_ea, Primitive("angle", (jat, center, lat))),
                (-1.0 / den_ea, Primitive("angle", (jat, center, mat))),
                (-1.0 / den_ea, Primitive("angle", (kat, center, lat))),
                (-1.0 / den_ea, Primitive("angle", (kat, center, mat))),
                (2.0 / den_ea, Primitive("angle", (lat, center, mat))),
            ),
        ),
        GICForgePythonCoordinate(
            name=f"EEee{start + 1:04d}",
            block="EEee",
            type_index=8,
            terms=(
                (1.0 / den_eb, Primitive("angle", (jat, center, lat))),
                (-1.0 / den_eb, Primitive("angle", (jat, center, mat))),
                (-1.0 / den_eb, Primitive("angle", (kat, center, lat))),
                (1.0 / den_eb, Primitive("angle", (kat, center, mat))),
            ),
        ),
        GICForgePythonCoordinate(
            name=f"T2xx{start + 2:04d}",
            block="T2xx",
            type_index=9,
            terms=(
                (1.0 / den_t2, Primitive("angle", (jat, center, lat))),
                (-1.0 / den_t2, Primitive("angle", (kat, center, mat))),
            ),
        ),
        GICForgePythonCoordinate(
            name=f"T2yy{start + 3:04d}",
            block="T2yy",
            type_index=10,
            terms=(
                (1.0 / den_t2, Primitive("angle", (kat, center, lat))),
                (-1.0 / den_t2, Primitive("angle", (jat, center, mat))),
            ),
        ),
        GICForgePythonCoordinate(
            name=f"T2zz{start + 4:04d}",
            block="T2zz",
            type_index=11,
            terms=(
                (1.0 / den_t2, Primitive("angle", (jat, center, kat))),
                (-1.0 / den_t2, Primitive("angle", (lat, center, mat))),
            ),
        ),
    ]


def _high_coord_angle_coordinates(
    center: int,
    neigh: list[int],
    *,
    effective_atomic_numbers: tuple[float, ...],
    coords: np.ndarray,
    linear_threshold: float,
    angle_start: int,
    linear_start: int,
    settings: LocalSALCSettings | None = None,
) -> tuple[list[GICForgePythonCoordinate], list[GICForgePythonCoordinate]]:
    local_settings = settings or LocalSALCSettings()
    angle_primitives: list[Primitive] = []
    linears: list[GICForgePythonCoordinate] = []
    suppress_redundant_trans = len(neigh) >= 3 or _has_redundant_linear_pairs(center, neigh, coords)
    for ib, first in enumerate(neigh[:-1]):
        for second in neigh[ib + 1 :]:
            left, right = sorted((first, second))
            value = angle(left, center, right, coords)
            if value >= linear_threshold:
                if suppress_redundant_trans:
                    continue
                linears.append(
                    _primitive_coordinate(
                        "LAng",
                        linear_start + len(linears),
                        _linear_bend_primitive((left, center, right), coords, mode=-1),
                    )
                )
                linears.append(
                    _primitive_coordinate(
                        "LAng",
                        linear_start + len(linears),
                        _linear_bend_primitive((left, center, right), coords, mode=-2),
                    )
                )
            else:
                angle_primitives.append(Primitive("angle", (left, center, right)))
    coordinates: list[GICForgePythonCoordinate] = []
    if len(neigh) >= 5:
        angle_group_records = _high_coord_angle_group_records_by_template_or_equivalence(
            center,
            neigh,
            effective_atomic_numbers=effective_atomic_numbers,
            coords=coords,
            linear_threshold=linear_threshold,
            settings=local_settings,
        )
        for diagnostic, primitives in angle_group_records:
            coordinates.extend(
                _svd_local_coordinates(
                    primitives,
                    coords=coords,
                    prefix="HCAn",
                    start=angle_start + len(coordinates),
                    kind_type_index=17,
                    diagnostic=diagnostic,
                )
            )
    else:
        for primitive in angle_primitives:
            coordinates.append(
                _primitive_coordinate("HCAn", angle_start + len(coordinates), primitive)
            )
    return coordinates, linears


def _high_coord_angle_group_records_by_template_or_equivalence(
    center: int,
    neigh: list[int],
    *,
    effective_atomic_numbers: tuple[float, ...],
    coords: np.ndarray,
    linear_threshold: float,
    settings: LocalSALCSettings | None = None,
) -> tuple[tuple[str, tuple[Primitive, ...]], ...]:
    local_settings = settings or LocalSALCSettings()
    match = _local_coordination_match(
        center,
        neigh,
        coords=coords,
        max_rms_cosine_error=local_settings.template_rms_threshold,
        min_score_margin=local_settings.template_min_margin,
    )
    if match.template is None:
        return _high_coord_angle_group_records_by_ligand_equivalence(
            center,
            neigh,
            effective_atomic_numbers=effective_atomic_numbers,
            coords=coords,
            linear_threshold=linear_threshold,
            settings=local_settings,
            match=match,
        )
    return _high_coord_angle_group_records_by_template(
        center,
        neigh,
        template=match.template,
        effective_atomic_numbers=effective_atomic_numbers,
        coords=coords,
        linear_threshold=linear_threshold,
        settings=local_settings,
        match=match,
    )


def _high_coord_angle_primitives_by_template_or_equivalence(
    center: int,
    neigh: list[int],
    *,
    effective_atomic_numbers: tuple[float, ...],
    coords: np.ndarray,
    linear_threshold: float,
    settings: LocalSALCSettings | None = None,
) -> tuple[tuple[Primitive, ...], ...]:
    return tuple(
        primitives
        for _diagnostic, primitives in _high_coord_angle_group_records_by_template_or_equivalence(
            center,
            neigh,
            effective_atomic_numbers=effective_atomic_numbers,
            coords=coords,
            linear_threshold=linear_threshold,
            settings=settings,
        )
    )


def _high_coord_angle_group_records_by_template(
    center: int,
    neigh: list[int],
    *,
    template: LocalCoordinationTemplate,
    effective_atomic_numbers: tuple[float, ...],
    coords: np.ndarray,
    linear_threshold: float,
    settings: LocalSALCSettings | None = None,
    match: LocalCoordinationMatch | None = None,
) -> tuple[tuple[str, tuple[Primitive, ...]], ...]:
    local_settings = settings or LocalSALCSettings()
    local_match = match or _local_coordination_match(
        center,
        neigh,
        coords=coords,
        max_rms_cosine_error=local_settings.template_rms_threshold,
        min_score_margin=local_settings.template_min_margin,
    )
    grouped = _high_coord_angle_grouped_by_template(
        center,
        neigh,
        template=template,
        effective_atomic_numbers=effective_atomic_numbers,
        coords=coords,
        linear_threshold=linear_threshold,
        settings=local_settings,
    )
    return tuple(
        (
            "LOCAL_EQUIVALENCE KIND=ANGLE "
            f"CENTER={center + 1} SOURCE=TEMPLATE "
            f"{_local_template_diagnostic(local_match)} "
            f"KEY={key[0]}-{key[1]}-{key[2]} SIZE={len(primitives)} "
            f"ATOMS={_primitive_atom_group_token(primitives)}",
            primitives,
        )
        for key, primitives in grouped
    )




def _high_coord_angle_grouped_by_template(
    center: int,
    neigh: list[int],
    *,
    template: LocalCoordinationTemplate,
    effective_atomic_numbers: tuple[float, ...],
    coords: np.ndarray,
    linear_threshold: float,
    settings: LocalSALCSettings | None = None,
) -> tuple[tuple[tuple[int, int, int], tuple[Primitive, ...]], ...]:
    local_settings = settings or LocalSALCSettings()
    classes = _local_ligand_equivalence_classes(
        center,
        neigh,
        effective_atomic_numbers=effective_atomic_numbers,
        coords=coords,
        zeff_tolerance=local_settings.zeff_tolerance,
        distance_tolerance=local_settings.distance_tolerance_angstrom,
    )
    class_by_atom = {
        atom: class_index for class_index, atoms in enumerate(classes) for atom in atoms
    }
    ideal_cosines = _template_pair_cosine_classes(
        template,
        tolerance=local_settings.angle_class_tolerance,
    )
    grouped: dict[tuple[int, int, int], list[Primitive]] = {}
    for ib, first in enumerate(neigh[:-1]):
        for second in neigh[ib + 1 :]:
            left, right = sorted((first, second))
            if angle(left, center, right, coords) >= linear_threshold:
                continue
            first_class = class_by_atom[first]
            second_class = class_by_atom[second]
            angle_class = _nearest_cosine_class(
                _ligand_pair_cosine(center, left, right, coords),
                ideal_cosines,
            )
            key = (*sorted((first_class, second_class)), angle_class)
            grouped.setdefault(key, []).append(Primitive("angle", (left, center, right)))
    return tuple((key, tuple(grouped[key])) for key in sorted(grouped))


def _high_coord_angle_group_records_by_ligand_equivalence(
    center: int,
    neigh: list[int],
    *,
    effective_atomic_numbers: tuple[float, ...],
    coords: np.ndarray,
    linear_threshold: float,
    settings: LocalSALCSettings | None = None,
    match: LocalCoordinationMatch | None = None,
) -> tuple[tuple[str, tuple[Primitive, ...]], ...]:
    local_settings = settings or LocalSALCSettings()
    local_match = match or _local_coordination_match(
        center,
        neigh,
        coords=coords,
        max_rms_cosine_error=local_settings.template_rms_threshold,
        min_score_margin=local_settings.template_min_margin,
    )
    grouped = _high_coord_angle_grouped_by_ligand_equivalence(
        center,
        neigh,
        effective_atomic_numbers=effective_atomic_numbers,
        coords=coords,
        linear_threshold=linear_threshold,
        settings=local_settings,
    )
    return tuple(
        (
            "LOCAL_EQUIVALENCE KIND=ANGLE "
            f"CENTER={center + 1} SOURCE=EQUIVALENCE "
            f"{_local_template_diagnostic(local_match)} "
            f"KEY={key[0]}-{key[1]} SIZE={len(primitives)} "
            f"ATOMS={_primitive_atom_group_token(primitives)}",
            primitives,
        )
        for key, primitives in grouped
    )


def _high_coord_angle_primitives_by_ligand_equivalence(
    center: int,
    neigh: list[int],
    *,
    effective_atomic_numbers: tuple[float, ...],
    coords: np.ndarray,
    linear_threshold: float,
    settings: LocalSALCSettings | None = None,
) -> tuple[tuple[Primitive, ...], ...]:
    return tuple(
        primitives
        for _key, primitives in _high_coord_angle_grouped_by_ligand_equivalence(
            center,
            neigh,
            effective_atomic_numbers=effective_atomic_numbers,
            coords=coords,
            linear_threshold=linear_threshold,
            settings=settings,
        )
    )


def _high_coord_angle_grouped_by_ligand_equivalence(
    center: int,
    neigh: list[int],
    *,
    effective_atomic_numbers: tuple[float, ...],
    coords: np.ndarray,
    linear_threshold: float,
    settings: LocalSALCSettings | None = None,
) -> tuple[tuple[tuple[int, int], tuple[Primitive, ...]], ...]:
    local_settings = settings or LocalSALCSettings()
    classes = _local_ligand_equivalence_classes(
        center,
        neigh,
        effective_atomic_numbers=effective_atomic_numbers,
        coords=coords,
        zeff_tolerance=local_settings.zeff_tolerance,
        distance_tolerance=local_settings.distance_tolerance_angstrom,
    )
    class_by_atom = {
        atom: class_index for class_index, atoms in enumerate(classes) for atom in atoms
    }
    grouped: dict[tuple[int, int], list[Primitive]] = {}
    for ib, first in enumerate(neigh[:-1]):
        for second in neigh[ib + 1 :]:
            left, right = sorted((first, second))
            if angle(left, center, right, coords) >= linear_threshold:
                continue
            first_class = class_by_atom[first]
            second_class = class_by_atom[second]
            key = tuple(sorted((first_class, second_class)))
            grouped.setdefault(key, []).append(Primitive("angle", (left, center, right)))
    return tuple((key, tuple(grouped[key])) for key in sorted(grouped))


def _local_ligand_equivalence_classes(
    center: int,
    neigh: list[int],
    *,
    effective_atomic_numbers: tuple[float, ...],
    coords: np.ndarray,
    zeff_tolerance: float = 5.0e-4,
    distance_tolerance: float = 1.0e-3,
) -> tuple[tuple[int, ...], ...]:
    return local_ligand_equivalence_classes(
        center,
        neigh,
        effective_atomic_numbers=effective_atomic_numbers,
        coordinates_angstrom=coords,
        zeff_tolerance=zeff_tolerance,
        distance_tolerance_angstrom=distance_tolerance,
    )


def _recognize_local_coordination_template(
    center: int,
    neigh: list[int],
    *,
    coords: np.ndarray,
    max_rms_cosine_error: float = LOCAL_TEMPLATE_RMS_THRESHOLD,
    min_score_margin: float = LOCAL_TEMPLATE_MIN_MARGIN,
) -> tuple[LocalCoordinationTemplate | None, float]:
    match = _local_coordination_match(
        center,
        neigh,
        coords=coords,
        max_rms_cosine_error=max_rms_cosine_error,
        min_score_margin=min_score_margin,
    )
    return match.template, match.score


def _local_coordination_match(
    center: int,
    neigh: list[int],
    *,
    coords: np.ndarray,
    max_rms_cosine_error: float = LOCAL_TEMPLATE_RMS_THRESHOLD,
    min_score_margin: float = LOCAL_TEMPLATE_MIN_MARGIN,
) -> LocalCoordinationMatch:
    return local_coordination_match(
        center,
        neigh,
        coordinates_angstrom=coords,
        max_rms_cosine_error=max_rms_cosine_error,
        min_score_margin=min_score_margin,
    )




def _local_template_diagnostic(match: LocalCoordinationMatch) -> str:
    best = match.best_template.name if match.best_template is not None else "NONE"
    nearest = match.nearest_template.name if match.nearest_template is not None else "NONE"
    score = "NA" if not np.isfinite(match.score) else f"{match.score:.6g}"
    margin = "NA" if not np.isfinite(match.margin) else f"{match.margin:.6g}"
    selected = match.template.name if match.template is not None else "GENERIC"
    rms_headroom = "NA" if not np.isfinite(match.rms_headroom) else f"{match.rms_headroom:.6g}"
    margin_headroom = (
        "NA" if not np.isfinite(match.margin_headroom) else f"{match.margin_headroom:.6g}"
    )
    return (
        f"TEMPLATE={selected} BEST_TEMPLATE={best} NEAREST_TEMPLATE={nearest} TEMPLATE_SCORE={score} "
        f"TEMPLATE_MARGIN={margin} TEMPLATE_STATUS={match.status} "
        f"TEMPLATE_RMS_HEADROOM={rms_headroom} "
        f"TEMPLATE_MARGIN_HEADROOM={margin_headroom} "
        f"THRESHOLD_SENSITIVITY={match.sensitivity} ASSIGNMENT=FROZEN"
    )




def _template_pair_cosine_classes(
    template: LocalCoordinationTemplate,
    tolerance: float = 2.0e-2,
) -> tuple[float, ...]:
    return template_pair_cosine_classes(template, tolerance=tolerance)


def _nearest_cosine_class(value: float, classes: tuple[float, ...]) -> int:
    return nearest_cosine_class(value, classes)


def _ligand_pair_cosine(center: int, first: int, second: int, coords: np.ndarray) -> float:
    return ligand_pair_cosine(center, first, second, coords)


def _local_ligand_unit_vectors(center: int, neigh: list[int], coords: np.ndarray) -> np.ndarray:
    return local_ligand_unit_vectors(center, neigh, coords)


def _sorted_pair_cosines(vectors: np.ndarray) -> np.ndarray:
    return sorted_pair_cosines(vectors)


_LOCAL_COORDINATION_TEMPLATES = LOCAL_COORDINATION_TEMPLATES


def _atom_ring_map_from_rings(rings: list[tuple[int, ...]], natoms: int) -> list[int]:
    atom_ring = [0 for _ in range(natoms)]
    for index, ring in enumerate(rings, start=1):
        for atom in ring:
            atom_ring[int(atom)] = index
    return atom_ring


def _atom_selected_ring_counts(rings: list[tuple[int, ...]], natoms: int) -> list[int]:
    counts = [0 for _ in range(natoms)]
    for ring in rings:
        for atom in ring:
            counts[int(atom)] += 1
    return counts


def _primitive_coordinate(
    prefix: str, index: int, primitive: Primitive
) -> GICForgePythonCoordinate:
    return GICForgePythonCoordinate(
        name=f"{prefix}{index:04d}",
        block=prefix,
        terms=((1.0, primitive),),
    )


def _linear_bend_primitive(
    atoms: tuple[int, int, int],
    coords: np.ndarray,
    *,
    mode: int,
) -> Primitive:
    """Construct one Gaussian-compatible local linear-bend primitive."""

    reference = linear_bend_reference_atom(atoms, coords)
    ref = (reference,) if reference is not None else ()
    return Primitive("linear_bend", atoms, mode=mode, ref=ref)


def _python_model_diagnostics(
    candidates: tuple[GICForgePythonCoordinate, ...],
    coordinates: tuple[GICForgePythonCoordinate, ...],
    coords: np.ndarray,
    *,
    target_rank: int,
    svd_local: bool,
    local_salc: bool,
    xy3_torsions: bool,
    xy2_torsions: bool,
    separate_exocyclic_torsions: bool,
    onedih: bool,
    max_linear_angle_pairs_per_center: int,
    local_salc_settings: LocalSALCSettings,
    ring_puckering_model: str,
) -> dict[str, object]:
    candidate_counts = _coordinate_block_counts(candidates)
    kept_counts = _coordinate_block_counts(coordinates)
    removed_counts = {
        block: int(candidate_counts.get(block, 0) - kept_counts.get(block, 0))
        for block in sorted(set(candidate_counts) | set(kept_counts))
    }
    candidate_rank = _coordinate_b_rank(candidates, coords)
    final_rank = _coordinate_b_rank(coordinates, coords)
    normalized_condition = _normalized_coordinate_b_condition(coordinates, coords)
    selected_local_records = {
        (coordinate.name, coordinate.diagnostic)
        for coordinate in coordinates
        if coordinate.diagnostic.startswith("LOCAL_")
    }
    candidate_local_records = {
        (coordinate.name, coordinate.diagnostic)
        for coordinate in candidates
        if coordinate.diagnostic.startswith("LOCAL_")
    }
    return {
        "backend": "gicforge-python",
        "svd_local": bool(svd_local),
        "local_salc": bool(local_salc),
        "xy3_torsions": bool(xy3_torsions),
        "xy2_torsions": bool(xy2_torsions),
        "separate_exocyclic_torsions": bool(separate_exocyclic_torsions),
        "onedih": bool(onedih),
        "ring_puckering_model": ring_puckering_model,
        "target_rank": int(target_rank),
        "candidate_count": int(len(candidates)),
        "final_count": int(len(coordinates)),
        "candidate_rank": int(candidate_rank),
        "final_rank": int(final_rank),
        "rank_complete": bool(final_rank == target_rank),
        "count_complete": bool(len(coordinates) == target_rank),
        "normalized_b_condition_number": normalized_condition,
        "max_normalized_b_condition_number": MAX_NORMALIZED_SONIC_CONDITION,
        "condition_accepted": bool(np.isfinite(normalized_condition)),
        "condition_within_policy": bool(
            np.isfinite(normalized_condition)
            and normalized_condition <= MAX_NORMALIZED_SONIC_CONDITION
        ),
        "max_linear_angle_pairs_per_center": int(max_linear_angle_pairs_per_center),
        "local_salc_thresholds": {
            "zeff_tolerance": local_salc_settings.zeff_tolerance,
            "distance_tolerance_angstrom": local_salc_settings.distance_tolerance_angstrom,
            "template_rms_threshold": local_salc_settings.template_rms_threshold,
            "template_min_margin": local_salc_settings.template_min_margin,
            "angle_class_tolerance": local_salc_settings.angle_class_tolerance,
        },
        "candidate_counts_by_block": candidate_counts,
        "final_counts_by_block": kept_counts,
        "removed_counts_by_block": removed_counts,
        "local_equivalence": tuple(
            sorted(
                f"{record.split(' ', 1)[0]} GIC={name} {record.split(' ', 1)[1]} "
                f"STATUS={'KEPT' if (name, record) in selected_local_records else 'PRUNED'}"
                for name, record in candidate_local_records
            )
        ),
        "threshold_sensitive_local_domains": tuple(
            sorted(
                record
                for _name, record in candidate_local_records
                if "THRESHOLD_SENSITIVITY=NEAR_" in record
            )
        ),
    }


def _coordinate_block_counts(coordinates: tuple[GICForgePythonCoordinate, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for coordinate in coordinates:
        counts[coordinate.block] = counts.get(coordinate.block, 0) + 1
    return dict(sorted(counts.items()))


def _polyhedral_catalog_centers(
    coordinates: tuple[GICForgePythonCoordinate, ...],
) -> frozenset[int]:
    """Return centers owned by an a-priori polyhedral angular catalog."""

    return frozenset(
        primitive.atoms[1]
        for coordinate in coordinates
        if CANONICAL_SALC_DIAGNOSTIC in coordinate.diagnostic
        for _coefficient, primitive in coordinate.terms
        if primitive.kind == "angle" and len(primitive.atoms) == 3
    )


def _without_polyhedral_axis_coordinates(
    coordinates: tuple[GICForgePythonCoordinate, ...],
    *,
    polyhedral_centers: frozenset[int] | None = None,
) -> tuple[GICForgePythonCoordinate, ...]:
    """Remove linear-axis substitutes from recognized polyhedral domains.

    A trans ligand pair is part of a polyhedron, not a covalent linear chain.
    Its angular space is already owned by the catalog SALCs.  Treating that
    pair as a collapsed linear center creates a dihedral whose middle
    "bond" joins the two non-bonded ligands and couples almost singularly to
    the catalog bends and ligand out-of-plane rows.
    """

    centers = (
        _polyhedral_catalog_centers(coordinates)
        if polyhedral_centers is None
        else frozenset(polyhedral_centers)
    )
    if not centers:
        return coordinates

    def owned_by_polyhedral_axis(coordinate: GICForgePythonCoordinate) -> bool:
        if any(
            primitive.kind == "linear_bend"
            and len(primitive.atoms) == 3
            and primitive.atoms[1] in centers
            for _coefficient, primitive in coordinate.terms
        ):
            return True
        return "SPECIAL_LINEAR_CASE=YES" in coordinate.diagnostic and any(
            f"COLLAPSED_CENTER={center + 1}" in coordinate.diagnostic for center in centers
        )

    return tuple(
        coordinate for coordinate in coordinates if not owned_by_polyhedral_axis(coordinate)
    )




def _normalize_coordinate_b_rows(rows: np.ndarray) -> np.ndarray:
    """Normalize B rows without rebuilding or stacking the input matrix."""

    normalized = np.array(rows, dtype=float, copy=True)
    if normalized.ndim != 2 or normalized.shape[0] == 0:
        return normalized
    norms = np.linalg.norm(normalized, axis=1)
    nonzero = norms > 1.0e-12
    normalized[nonzero] /= norms[nonzero, None]
    return normalized


def _cage_chart_semantics_preserved(
    original: tuple[GICForgePythonCoordinate, ...],
    conditioned: tuple[GICForgePythonCoordinate, ...],
) -> bool:
    """Keep a cage chart's defining rows and broad motion-family dimensions."""

    if not any(coordinate.block in {"BtFl", "Spir"} for coordinate in original):
        return True

    special_blocks = {"BtFl", "Spir", "RDef", "RPck"}
    original_special = {coordinate for coordinate in original if coordinate.block in special_blocks}
    conditioned_special = {
        coordinate for coordinate in conditioned if coordinate.block in special_blocks
    }
    if conditioned_special != original_special:
        return False

    def motion_family(coordinate: GICForgePythonCoordinate) -> str:
        kind = coordinate.dominant_kind
        if kind in {"angle", "out_of_plane", "out_of_plane_height"}:
            return "angular"
        if kind == "bond":
            return "stretch"
        if kind == "dihedral":
            return "torsional"
        return kind

    def family_counts(
        coordinates: tuple[GICForgePythonCoordinate, ...],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for coordinate in coordinates:
            family = motion_family(coordinate)
            counts[family] = counts.get(family, 0) + 1
        return counts

    return family_counts(conditioned) == family_counts(original)


def _normalized_coordinate_b_condition(
    coordinates: tuple[GICForgePythonCoordinate, ...],
    coords: np.ndarray,
) -> float:
    if not coordinates:
        return float("inf")
    return normalized_matrix_condition(
        _coordinate_b_matrix(coordinates, coords),
        absolute_tolerance=1.0e-12,
        zero_row_tolerance=1.0e-12,
        required_rank=len(coordinates),
    )


def _normalized_b_condition_from_singular_values(singular: np.ndarray) -> float:
    if not singular.size or singular[-1] <= 1.0e-12:
        return float("inf")
    return float(singular[0] / singular[-1])


def _coordinate_basis_audit(
    rows: np.ndarray,
    *,
    certify_rank: bool = True,
    certify_condition: bool = True,
) -> _CoordinateBasisAudit:
    """Compute the raw-rank and normalized-condition certificates once."""

    matrix = np.asarray(rows, dtype=float)
    normalized = _normalize_coordinate_b_rows(matrix)
    return _CoordinateBasisAudit(
        rows=matrix,
        normalized_rows=normalized,
        rank=(
            numerical_matrix_rank(
                matrix,
                absolute_tolerance=1.0e-10,
                relative_tolerance=1.0e-8,
            )
            if certify_rank
            else None
        ),
        condition=(
            _normalized_b_condition_from_singular_values(
                np.linalg.svd(normalized, compute_uv=False)
            )
            if certify_condition
            else None
        ),
    )


def _is_analytic_salc(coordinate: GICForgePythonCoordinate) -> bool:
    """Return whether rank selection should exhaust this analytic family first.

    Priority is not a requirement that every candidate survive.  Ring and cage
    generators can intentionally emit a redundant analytic spanning set; the
    rank kernel must retain a maximal independent subset before it considers
    primitive completion rows.
    """

    return SONIC_CONSTRUCTION_POLICY.is_analytic_salc(
        block=coordinate.block,
        diagnostic=coordinate.diagnostic,
    )


def _conditioned_coordinate_basis(
    candidates: tuple[GICForgePythonCoordinate, ...],
    selected: tuple[GICForgePythonCoordinate, ...],
    coords: np.ndarray,
    *,
    target_rank: int,
    preserve_special: bool,
) -> tuple[GICForgePythonCoordinate, ...]:
    """Freeze a complete, scale-independent SONIC basis or fail explicitly.

    Canonical coordination-polyhedron SALCs are semantic coordinates and are
    exhausted first.  The shared maximum-residual rank kernel chooses only
    their complementary rows.  No Gaussian behavior or molecule identity is
    part of this decision.
    """

    if len(selected) != target_rank:
        return selected
    selected_audit = _coordinate_basis_audit(_coordinate_b_matrix(selected, coords))
    if selected_audit.rank != target_rank:
        return selected
    if (
        np.isfinite(selected_audit.condition)
        and selected_audit.condition <= MAX_NORMALIZED_SONIC_CONDITION
    ):
        return selected

    protected_special = {
        coordinate
        for coordinate in selected
        if preserve_special
        and (
            coordinate.block in {"BtFl", "Spir", "RDef", "RPck"}
            or "KIND=RING_" in coordinate.diagnostic
        )
    }
    candidate_audit = _coordinate_basis_audit(
        _coordinate_b_matrix(candidates, coords),
        certify_rank=False,
        certify_condition=False,
    )
    rows = candidate_audit.normalized_rows
    selection = select_rank_revealing_rows(
        rows,
        target_rank=target_rank,
        tolerance=1.0e-10,
        priorities=tuple(
            0 if _is_analytic_salc(coordinate) or coordinate in protected_special else 1
            for coordinate in candidates
        ),
        tie_tolerance=1.0e-12,
    )
    if selection.rank != target_rank:
        raise ValueError(
            "GICForge Python conditioned selection did not reach vibrational rank "
            f"({selection.rank}/{target_rank})"
        )
    conditioned = tuple(candidates[index] for index in selection.indices)
    catalog = {
        coordinate
        for coordinate in candidates
        if CANONICAL_SALC_DIAGNOSTIC in coordinate.diagnostic
    }
    if not catalog.issubset(conditioned):
        raise ValueError("canonical polyhedral SALCs are not an independent chart subspace")
    # Do not require the whole preferred ring/cage candidate set to survive:
    # it may be an intentionally redundant analytic span.  Priority above
    # already selects its maximal independent subset.  Cage-specific semantic
    # preservation is checked by _cage_chart_semantics_preserved at the caller.
    conditioned_audit = _coordinate_basis_audit(
        candidate_audit.rows[np.asarray(selection.indices, dtype=int)],
        certify_rank=False,
    )
    if conditioned_audit.condition is None or not np.isfinite(conditioned_audit.condition):
        raise ValueError(
            "GICForge Python could not construct a finite SONIC chart "
            f"(condition={conditioned_audit.condition:.6g})"
        )
    return conditioned


def _coordinate_b_rank(
    coordinates: tuple[GICForgePythonCoordinate, ...], coords: np.ndarray
) -> int:
    if not coordinates:
        return 0
    primitive_basis = _primitive_basis(coordinates)
    row_index = {primitive: index for index, primitive in enumerate(primitive_basis)}
    primitive_b = b_matrix_analytic(primitive_basis, coords)
    b_rows = []
    for coordinate in coordinates:
        row = np.zeros(primitive_b.shape[1], dtype=float)
        for coefficient, primitive in coordinate.terms:
            row += coefficient * primitive_b[row_index[primitive]]
        b_rows.append(row)
    return numerical_matrix_rank(
        np.vstack(b_rows),
        absolute_tolerance=1.0e-10,
        relative_tolerance=1.0e-8,
    )


def _coordinate_b_matrix(
    coordinates: tuple[GICForgePythonCoordinate, ...], coords: np.ndarray
) -> np.ndarray:
    if not coordinates:
        return np.zeros((0, int(np.asarray(coords).size)), dtype=float)
    primitive_basis = _primitive_basis(coordinates)
    row_index = {primitive: index for index, primitive in enumerate(primitive_basis)}
    primitive_b = b_matrix_analytic(primitive_basis, coords)
    rows = []
    for coordinate in coordinates:
        row = np.zeros(primitive_b.shape[1], dtype=float)
        for coefficient, primitive in coordinate.terms:
            row += coefficient * primitive_b[row_index[primitive]]
        rows.append(row)
    return np.vstack(rows)


def _prune_type_local(
    coordinates: tuple[GICForgePythonCoordinate, ...],
    coords: np.ndarray,
    *,
    target_rank: int,
    block_pruning: bool = False,
) -> tuple[GICForgePythonCoordinate, ...]:
    if block_pruning:
        return _prune_block_local(coordinates, coords, target_rank=target_rank)
    cage_special = [
        coordinate
        for coordinate in coordinates
        if coordinate.block in {"BtFl", "Spir"}
        or "KIND=BUTTERFLY" in coordinate.diagnostic
        or "KIND=SPIRO" in coordinate.diagnostic
    ]
    ring_special = [
        coordinate
        for coordinate in coordinates
        if coordinate not in cage_special
        and (coordinate.block in {"RDef", "RPck"} or "KIND=RING_" in coordinate.diagnostic)
    ]
    # Cage-specific collective coordinates are the irreducible special modes
    # of bridged/spiro systems.  When they overlap the ordinary per-ring
    # puckering span, retain the cage mode first; neither class may ever be
    # displaced by an exocyclic angle or torsion.
    special = cage_special + ring_special
    ordinary = [coordinate for coordinate in coordinates if coordinate not in special]
    by_kind = {
        "bond": [coord for coord in ordinary if coord.dominant_kind == "bond"],
        "angle": [coord for coord in ordinary if coord.dominant_kind == "angle"],
        "linear_bend": [coord for coord in ordinary if coord.dominant_kind == "linear_bend"],
        "dihedral": [coord for coord in ordinary if coord.dominant_kind == "dihedral"],
        "out_of_plane": [
            coord
            for coord in ordinary
            if coord.dominant_kind in {"out_of_plane", "out_of_plane_height"}
        ],
    }
    bond_neighbors: dict[int, set[int]] = {}
    for coordinate in by_kind["bond"]:
        for _coefficient, primitive in coordinate.terms:
            if primitive.kind != "bond":
                continue
            first, second = primitive.atoms
            bond_neighbors.setdefault(first, set()).add(second)
            bond_neighbors.setdefault(second, set()).add(first)
    linear_bends = [
        coordinate
        for coordinate in by_kind["linear_bend"]
        if not any(
            len(bond_neighbors.get(center, ())) >= 3
            for _coefficient, primitive in coordinate.terms
            if primitive.kind == "linear_bend"
            for center in (primitive.atoms[1],)
        )
    ]
    if any(coord.block == "HCAn" for coord in by_kind["angle"]):
        linear_bends = _high_coord_linear_pruning_order(linear_bends)
    angular, promoted_out_of_plane = _native_angular_pruning_order(
        by_kind["angle"],
        by_kind["out_of_plane"],
    )
    ordered = (
        by_kind["bond"]
        + special
        + angular
        + linear_bends
        + by_kind["dihedral"]
        + [
            coordinate
            for coordinate in by_kind["out_of_plane"]
            if coordinate not in promoted_out_of_plane
        ]
    )
    if len(ordered) <= target_rank:
        return tuple(ordered)
    primitive_basis = _primitive_basis(ordered)
    row_index = {primitive: index for index, primitive in enumerate(primitive_basis)}
    primitive_b = b_matrix_analytic(primitive_basis, coords)
    b_rows = []
    for coordinate in ordered:
        row = np.zeros(primitive_b.shape[1], dtype=float)
        for coefficient, primitive in coordinate.terms:
            row += coefficient * primitive_b[row_index[primitive]]
        b_rows.append(row)

    basis = _IncrementalRowBasis(max_rows=target_rank)
    keep: list[GICForgePythonCoordinate] = []
    # A fallback candidate pool can contain more bond primitives than the
    # molecular vibrational rank (high-coordination clusters are the common
    # example).  Every retained row must therefore earn an independent B-row;
    # appending all bonds unconditionally can both exceed the target count and
    # prevent later angle/linear rows from completing the rank.
    for index, coordinate in enumerate(ordered):
        if coordinate.dominant_kind != "bond":
            continue
        if len(basis) >= target_rank or len(keep) >= target_rank:
            break
        if basis.try_add(b_rows[index]):
            keep.append(coordinate)
    for index, coordinate in enumerate(ordered):
        if coordinate.dominant_kind == "bond":
            continue
        if len(basis) >= target_rank or len(keep) >= target_rank:
            continue
        if basis.try_add(b_rows[index]):
            keep.append(coordinate)
    local_selection = tuple(keep)
    if (
        len(local_selection) >= target_rank
        and _coordinate_b_rank(local_selection, coords) >= target_rank
    ):
        return local_selection

    # The global recovery is needed for the near-linear primitive family. For
    # ordinary charts, retain the established family-prioritized selection:
    # applying a rank-revealing pivot globally there can replace valid special
    # coordinates even when no linear-bend closure is involved.
    if not linear_bends:
        return local_selection

    # The sequential family-ordered pass above is intentionally conservative,
    # but a greedy first choice can consume a direction needed by a later
    # coordinate.  Reuse the shared rank-revealing kernel as a general fallback
    # on the complete candidate pool.  In this recovery path the numerical
    # pivot quality must decide the representatives: family priorities can
    # select a formally independent but poorly conditioned basis.
    selection = select_rank_revealing_rows(
        np.vstack(b_rows),
        target_rank=target_rank,
        tolerance=1.0e-10,
    )
    global_selection = tuple(ordered[index] for index in selection.indices)
    if (
        len(global_selection) >= target_rank
        and _coordinate_b_rank(global_selection, coords) >= target_rank
    ):
        return global_selection
    return local_selection


def _native_angular_pruning_order(
    angles: list[GICForgePythonCoordinate],
    out_of_plane: list[GICForgePythonCoordinate],
) -> tuple[list[GICForgePythonCoordinate], set[GICForgePythonCoordinate]]:
    """Prefer two in-plane bends plus U at a three-coordinate center.

    Near planarity, the three pair angles obey an almost exact closure relation.
    Keeping all three before the genuine out-of-plane coordinate creates a
    badly conditioned, although formally full-rank, SONIC chart.  Gaussian
    improper dihedrals remain an export choice; the native chart uses U.
    """

    angle_centers = [_single_angle_center(coordinate) for coordinate in angles]
    neighbors_by_center: dict[int, set[int]] = {}
    count_by_center: dict[int, int] = {}
    for coordinate, center in zip(angles, angle_centers, strict=True):
        if center is None:
            continue
        count_by_center[center] = count_by_center.get(center, 0) + 1
        neighbors = neighbors_by_center.setdefault(center, set())
        for _coefficient, primitive in coordinate.terms:
            neighbors.update(atom for atom in primitive.atoms if atom != center)

    oop_by_center: dict[int, GICForgePythonCoordinate] = {}
    for coordinate in out_of_plane:
        center = _single_out_of_plane_center(coordinate)
        if center is not None and center not in oop_by_center:
            oop_by_center[center] = coordinate

    eligible = {
        center
        for center, coordinate_count in count_by_center.items()
        if coordinate_count >= 3
        and len(neighbors_by_center.get(center, ())) == 3
        and center in oop_by_center
    }
    seen: dict[int, int] = {}
    promoted: set[GICForgePythonCoordinate] = set()
    ordered: list[GICForgePythonCoordinate] = []
    for coordinate, center in zip(angles, angle_centers, strict=True):
        ordered.append(coordinate)
        if center not in eligible:
            continue
        seen[center] = seen.get(center, 0) + 1
        if seen[center] == 2:
            oop = oop_by_center[center]
            ordered.append(oop)
            promoted.add(oop)
    return ordered, promoted


def _single_angle_center(coordinate: GICForgePythonCoordinate) -> int | None:
    centers = {
        primitive.atoms[1]
        for _coefficient, primitive in coordinate.terms
        if primitive.kind == "angle" and len(primitive.atoms) == 3
    }
    if len(centers) != 1 or any(
        primitive.kind != "angle" for _coefficient, primitive in coordinate.terms
    ):
        return None
    return next(iter(centers))


def _single_out_of_plane_center(coordinate: GICForgePythonCoordinate) -> int | None:
    centers = {
        primitive.atoms[0]
        for _coefficient, primitive in coordinate.terms
        if primitive.kind == "out_of_plane" and len(primitive.atoms) == 4
    }
    if len(centers) != 1 or any(
        primitive.kind != "out_of_plane" for _coefficient, primitive in coordinate.terms
    ):
        return None
    return next(iter(centers))


def _prune_block_local(
    coordinates: tuple[GICForgePythonCoordinate, ...],
    coords: np.ndarray,
    *,
    target_rank: int,
) -> tuple[GICForgePythonCoordinate, ...]:
    ordered = sorted(coordinates, key=lambda coord: (_block_pruning_priority(coord), coord.name))
    if len(ordered) <= target_rank:
        return tuple(ordered)
    primitive_basis = _primitive_basis(ordered)
    row_index = {primitive: index for index, primitive in enumerate(primitive_basis)}
    primitive_b = b_matrix_analytic(primitive_basis, coords)
    b_rows = []
    for coordinate in ordered:
        row = np.zeros(primitive_b.shape[1], dtype=float)
        for coefficient, primitive in coordinate.terms:
            row += coefficient * primitive_b[row_index[primitive]]
        b_rows.append(row)

    basis = _IncrementalRowBasis(max_rows=target_rank)
    keep: list[GICForgePythonCoordinate] = []
    for index, coordinate in enumerate(ordered):
        if len(basis) >= target_rank or len(keep) >= target_rank:
            break
        if basis.try_add(b_rows[index]):
            keep.append(coordinate)
    return tuple(sorted(keep, key=lambda coord: coordinates.index(coord)))


def _block_pruning_priority(coordinate: GICForgePythonCoordinate) -> int:
    if coordinate.dominant_kind == "bond":
        return 0
    if coordinate.block == "XAng":
        return 1
    if coordinate.block == "Spir":
        return 1
    if coordinate.dominant_kind == "linear_bend":
        return 2
    if coordinate.block in {"Dihe", "Tors"}:
        return 3
    if coordinate.block == "BtFl":
        return 4
    if coordinate.block == "RDef":
        return 5
    if coordinate.block == "RPck":
        return 6
    if coordinate.dominant_kind in {"out_of_plane", "out_of_plane_height"}:
        return 7
    return 8


def _high_coord_linear_pruning_order(
    coordinates: list[GICForgePythonCoordinate],
) -> list[GICForgePythonCoordinate]:
    groups: list[list[GICForgePythonCoordinate]] = []
    index_by_atoms: dict[tuple[int, ...], int] = {}
    for coordinate in coordinates:
        _coefficient, primitive = coordinate.terms[0]
        atoms = primitive.atoms
        if atoms not in index_by_atoms:
            index_by_atoms[atoms] = len(groups)
            groups.append([])
        groups[index_by_atoms[atoms]].append(coordinate)
    if not groups:
        return []

    ordered: list[GICForgePythonCoordinate] = []
    ordered.extend(groups[0])
    for group in groups[1:]:
        ordered.extend(coord for coord in group if coord.terms[0][1].mode == -2)
    for group in groups[1:]:
        ordered.extend(coord for coord in group if coord.terms[0][1].mode != -2)
    seen: set[GICForgePythonCoordinate] = set(ordered)
    ordered.extend(coord for coord in coordinates if coord not in seen)
    return ordered


class _IncrementalRowBasis:
    """Stable incremental QR gate for family-ordered B rows.

    The stored vectors are the rows of Q.  Two modified Gram--Schmidt passes
    avoid the loss of orthogonality that otherwise appears for nearly
    dependent internal coordinates.  This gate only prunes candidates; the
    completed chart is still certified independently by the established SVD
    rank and condition checks.
    """

    __slots__ = (
        "_count",
        "_rows",
        "absolute_tolerance",
        "max_rows",
        "relative_tolerance",
    )

    def __init__(
        self,
        *,
        max_rows: int,
        absolute_tolerance: float = 1.0e-10,
        relative_tolerance: float = 1.0e-8,
    ) -> None:
        self._rows: np.ndarray | None = None
        self._count = 0
        self.max_rows = int(max_rows)
        self.absolute_tolerance = float(absolute_tolerance)
        self.relative_tolerance = float(relative_tolerance)

    def __len__(self) -> int:
        return self._count

    def try_add(self, row: np.ndarray) -> bool:
        candidate = np.asarray(row, dtype=float).copy()
        original_norm = float(np.linalg.norm(candidate))
        if original_norm <= self.absolute_tolerance:
            return False
        if self._rows is None:
            self._rows = np.empty((self.max_rows, candidate.size), dtype=float)
        if candidate.shape != (self._rows.shape[1],):
            raise ValueError("incremental QR row dimension changed")
        basis = self._rows[: self._count]
        for _pass in range(2):
            if self._count:
                candidate -= (candidate @ basis.T) @ basis
        residual_norm = float(np.linalg.norm(candidate))
        threshold = max(
            self.absolute_tolerance,
            self.relative_tolerance * original_norm,
        )
        if residual_norm <= threshold:
            return False
        if self._count >= self.max_rows:
            return False
        self._rows[self._count] = candidate / residual_norm
        self._count += 1
        return True


def _target_rank(coords: np.ndarray, graph) -> int:
    components = _connected_components(graph)
    rank = 0
    for component in components:
        rank += 3 * len(component) - (5 if _is_linear(coords[list(component)]) else 6)
    return rank


def _connected_components(graph) -> list[tuple[int, ...]]:
    seen: set[int] = set()
    components: list[tuple[int, ...]] = []
    for start in range(graph.natoms):
        if start in seen:
            continue
        stack = [start]
        component = []
        seen.add(start)
        while stack:
            atom = stack.pop()
            component.append(atom)
            for neighbor in graph.adjacency[atom]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return components


def _is_linear(coords: np.ndarray) -> bool:
    if coords.shape[0] <= 2:
        return True
    centered = coords - coords.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    return bool(singular_values[1] <= max(1.0e-8, 1.0e-6 * singular_values[0]))


def _primitive_basis(coordinates: Iterable[GICForgePythonCoordinate]) -> tuple[Primitive, ...]:
    basis: list[Primitive] = []
    seen: set[Primitive] = set()
    for coordinate in coordinates:
        for _coefficient, primitive in coordinate.terms:
            if primitive in seen:
                continue
            seen.add(primitive)
            basis.append(primitive)
    return tuple(basis)


def _format_readgic(name: str, terms: tuple[tuple[float, Primitive], ...]) -> str:
    return f"{name}={_format_terms(terms)}"


def _format_terms(terms: tuple[tuple[float, Primitive], ...]) -> str:
    chunks = []
    for coefficient, primitive in terms:
        atom_text = ",".join(str(atom + 1) for atom in primitive.atoms)
        symbol = {
            "bond": "R",
            "angle": "A",
            "linear_bend": "L",
            "dihedral": "D",
            "out_of_plane": "U",
            "out_of_plane_height": "H",
        }[primitive.kind]
        if primitive.kind == "linear_bend":
            reference = primitive.ref[0] + 1 if len(primitive.ref) == 1 else 0
            atom_text = f"{atom_text},{reference},{primitive.mode}"
        chunks.append(f"{coefficient:.10g}*{symbol}({atom_text})")
    return "+".join(chunks)
