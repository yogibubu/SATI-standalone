"""Shared ORACLE PI actions on an explicitly embedded molecular geometry grid.

For row coordinates X, T_g maps grid points and C_g is a proper rotation with
g(X[q]) C_g(q) = X[T_g(q)]. Thus C_(gh)(q) = C_h(q) C_g(T_h(q)).
This cocycle, rather than a point-independent Cartesian rotation, is what
rovibrational consumers must represent. No electronic or spin action is implied.
"""

from dataclasses import dataclass
import hashlib
from typing import Mapping

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from .permutation_inversion import (
    OraclePermutationInversion,
    PermutationInversionOperation,
    PIExpansionLimit,
    _closure,
)


ORACLE_PI_GRID_SCHEMA = "matrix.oracle.pi_geometry_grid.v1"


@dataclass(frozen=True)
class PermutationInversionGridAction:
    contract: OraclePermutationInversion
    targets: np.ndarray
    compensating_rotations: np.ndarray
    maximum_rmsd_angstrom: float = 0.0

    def __post_init__(self):
        raw = np.asarray(self.targets)
        rotations = np.asarray(self.compensating_rotations, dtype=float)
        order = self.contract.order
        if order is None:
            raise ValueError("PI grid action requires a fully expanded molecular group")
        if raw.ndim != 2 or raw.shape[0] != order or raw.shape[1] < 1 or raw.dtype.kind not in "iu":
            raise ValueError("PI grid targets must have integer shape (group_order, grid_size)")
        targets = raw.astype(int, copy=True)
        n = targets.shape[1]
        if not np.all(np.sort(targets, axis=1) == np.arange(n)):
            raise ValueError("every PI operation must bijectively permute the grid")
        if rotations.shape != (order, n, 3, 3) or not np.all(np.isfinite(rotations)):
            raise ValueError("PI compensating rotations must have shape (group_order, grid_size, 3, 3)")
        if not np.allclose(rotations.swapaxes(-1, -2) @ rotations, np.eye(3), atol=1e-9, rtol=0):
            raise ValueError("PI compensation must be orthogonal")
        if not np.allclose(np.linalg.det(rotations), 1, atol=1e-9, rtol=0):
            raise ValueError("PI compensation must be a proper rotation, including for starred operations")
        if not np.array_equal(targets[0], np.arange(n)) or not np.allclose(rotations[0], np.eye(3), atol=1e-9, rtol=0):
            raise ValueError("PI identity must fix grid points and body orientation")
        if not np.isfinite(self.maximum_rmsd_angstrom) or self.maximum_rmsd_angstrom < 0:
            raise ValueError("PI grid residual must be finite and nonnegative")
        lookup = {op: i for i, op in enumerate(self.contract.operations)}
        # Generator edges suffice for all products; avoid O(|G|**2) matrices.
        for left, op in enumerate(self.contract.operations):
            for generator in self.contract.generators:
                right = lookup[generator]
                product = lookup[op.compose(generator)]
                if not np.array_equal(targets[product], targets[left, targets[right]]):
                    raise ValueError("PI grid permutation violates group multiplication")
                expected = rotations[right] @ rotations[left, targets[right]]
                if not np.allclose(rotations[product], expected, atol=1e-8, rtol=0):
                    raise ValueError("PI body-frame rotations violate the composition cocycle")
        rotations = rotations.copy()
        targets.setflags(write=False)
        rotations.setflags(write=False)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "compensating_rotations", rotations)

    def validate_scalar_field(self, values, *, tolerance=1e-9):
        """Reject symmetry-breaking scalar data before an expensive solver run."""
        field = np.asarray(values, dtype=float).reshape(-1)
        if field.shape != (self.targets.shape[1],) or not np.all(np.isfinite(field)):
            raise ValueError("PI scalar field does not match the geometry grid")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("PI field tolerance must be positive and finite")
        scale = max(1., float(np.ptp(field)))
        residual = float(np.max(abs(field[self.targets]-field[None])))
        if residual > tolerance*scale:
            raise ValueError(f"scalar field is not invariant under ORACLE PI: maximum residual {residual:.8g}")
        return residual

    def reframe(self, rotations):
        """Transport the cocycle to X_new[q] = X_old[q] S[q]."""
        field = np.asarray(rotations, dtype=float)
        if field.shape != (self.targets.shape[1], 3, 3) or not np.all(np.isfinite(field)):
            raise ValueError("PI frame rotations must have shape (grid_size, 3, 3)")
        if not np.allclose(field.swapaxes(-1, -2) @ field, np.eye(3), atol=1e-10, rtol=0) or not np.allclose(np.linalg.det(field), 1, atol=1e-10, rtol=0):
            raise ValueError("PI reframing requires proper SO(3) rotations")
        cocycle = field.swapaxes(-1, -2)[None] @ self.compensating_rotations @ field[self.targets]
        return PermutationInversionGridAction(self.contract, self.targets, cocycle, self.maximum_rmsd_angstrom)

    def to_dict(self) -> dict:
        return {
            "schema": ORACLE_PI_GRID_SCHEMA,
            "contract": self.contract.to_dict(),
            "targets": self.targets.tolist(),
            "compensating_rotations": self.compensating_rotations.tolist(),
            "maximum_rmsd_angstrom": self.maximum_rmsd_angstrom,
        }

    @classmethod
    def from_dict(cls, payload: Mapping):
        if payload.get("schema") != ORACLE_PI_GRID_SCHEMA:
            raise ValueError("unsupported ORACLE PI grid-action schema")
        return cls(
            OraclePermutationInversion.from_dict(payload["contract"]),
            np.asarray(payload["targets"]), np.asarray(payload["compensating_rotations"]),
            float(payload["maximum_rmsd_angstrom"]),
        )


def derive_permutation_inversion_grid_action(
    contract: OraclePermutationInversion,
    geometries_angstrom: np.ndarray,
    masses_amu: np.ndarray,
    *,
    tolerance_angstrom: float = 2e-4,
) -> PermutationInversionGridAction:
    """Derive the PI action in the supplied embedding and flattened grid order.

    Only generators require geometric matching. Invariant distance descriptors
    shortlist candidates before proper mass-weighted Kabsch matching. Ambiguous,
    linear, undersampled, isotope-inconsistent or nonclosed grids are errors.
    """
    if contract.order is None:
        raise ValueError("expand the ORACLE PI group before deriving its grid action")
    xyz = np.asarray(geometries_angstrom, dtype=float)
    masses = np.asarray(masses_amu, dtype=float)
    natoms = len(contract.atoms)
    if xyz.ndim != 3 or xyz.shape[1:] != (natoms, 3) or len(xyz) == 0 or not np.all(np.isfinite(xyz)):
        raise ValueError("PI geometry grid must have finite shape (grid_size, natoms, 3)")
    if masses.shape != (natoms,) or np.any(masses <= 0) or not np.all(np.isfinite(masses)):
        raise ValueError("one finite positive mass is required per nucleus")
    if not np.isfinite(tolerance_angstrom) or tolerance_angstrom <= 0:
        raise ValueError("PI grid tolerance must be finite and positive")
    xyz = xyz - np.sum(xyz * masses[None, :, None], axis=1, keepdims=True) / masses.sum()
    singular = np.linalg.svd(xyz * np.sqrt(masses)[None, :, None], compute_uv=False)
    if singular.shape[1] < 2 or np.any(singular[:, 1] <= 1e-10 * np.maximum(singular[:, 0], 1)):
        raise ValueError("linear/degenerate geometry does not determine a unique SO(3) compensation")
    # Distances to three labelled anchors are frame-invariant. With RMSD <= t,
    # each atom residual <= t sqrt(M/min(m)), hence each pair-distance residual
    # <= 2t sqrt(M/min(m)). Infinity-norm tree queries cannot exclude a match.
    anchors = np.unique([0, natoms // 2, natoms - 1])

    def descriptor(frames):
        return np.linalg.norm(frames[:, :, None] - frames[:, None, anchors], axis=-1).reshape(len(frames), -1)

    tree = cKDTree(descriptor(xyz))
    radius = 2 * tolerance_angstrom * np.sqrt(masses.sum() / masses.min())
    generator_actions = []
    max_residual = 0.0
    for generator in contract.generators:
        if not np.allclose(masses, masses[list(generator.permutation)], atol=1e-10, rtol=0):
            raise ValueError("PI generator exchanges different masses in the active isotopologue")
        transformed = generator.apply(xyz)
        candidates = tree.query_ball_point(descriptor(transformed), radius, p=np.inf)
        target = np.empty(len(xyz), dtype=int)
        rotations = np.empty((len(xyz), 3, 3))
        for source, possible in enumerate(candidates):
            matches = []
            for destination in possible:
                covariance = (masses[:, None] * transformed[source]).T @ xyz[destination]
                u, _, vh = np.linalg.svd(covariance)
                sign = np.eye(3)
                sign[-1, -1] = 1 if np.linalg.det(u @ vh) > 0 else -1
                rotation = u @ sign @ vh
                residual = transformed[source] @ rotation - xyz[destination]
                rmsd = float(np.sqrt(np.sum(masses[:, None] * residual**2) / masses.sum()))
                if rmsd <= tolerance_angstrom:
                    matches.append((destination, rotation, rmsd))
            if len(matches) != 1:
                raise ValueError(
                    f"PI generator {generator} has {len(matches)} matches for grid point {source}; "
                    "provide a unique, complete labelled geometry grid"
                )
            target[source], rotations[source], residual = matches[0]
            max_residual = max(max_residual, residual)
        generator_actions.append((target, rotations))
    targets = [np.arange(len(xyz))]
    rotations = [np.broadcast_to(np.eye(3), (len(xyz), 3, 3)).copy()]
    word_index = {(): 0}
    for word in contract.words[1:]:
        parent = word_index[word[:-1]]
        target, rotation = generator_actions[word[-1]]
        targets.append(targets[parent][target])
        rotations.append(rotation @ rotations[parent][target])
        word_index[word] = len(targets) - 1
    # Certify all reconstructed operations against the actual geometry, not
    # just their algebra: composition must not accumulate out-of-band errors.
    for op, target, rotation in zip(contract.operations, targets, rotations, strict=True):
        residual = np.einsum("qia,qab->qib", op.apply(xyz), rotation) - xyz[target]
        rmsd = np.sqrt(np.sum(masses[None, :, None] * residual**2, axis=(1, 2)) / masses.sum())
        max_residual = max(max_residual, float(np.max(rmsd)))
    if max_residual > tolerance_angstrom:
        raise ValueError("composed PI operation leaves the geometry certification tolerance")
    return PermutationInversionGridAction(contract, np.asarray(targets), np.asarray(rotations), max_residual)


def complete_permutation_inversion_from_geometry_grid(
    seed_contract: OraclePermutationInversion,
    geometries_angstrom: np.ndarray,
    masses_amu: np.ndarray,
    *,
    tolerance_angstrom: float = 2e-4,
    max_candidates: int = 100000,
    max_operations: int = 4096,
) -> tuple[OraclePermutationInversion, PermutationInversionGridAction]:
    """Infer the complete finite PI group represented by a labelled grid.

    ORACLE's frozen labelled bond graph supplies only isotope-preserving graph
    automorphism candidates.  A candidate is retained only when its entire
    cyclic subgroup has a unique, proper-rotation match on every geometry in
    the grid.  The retained operations are then closed without assuming that
    they commute.  Consequently cyclic, dihedral, direct-product and other
    finite non-Abelian PI groups follow the same path into LOCK.

    No molecular name, group symbol, character table, coordinate name or
    generator permutation is an input.  The explicit candidate and group
    budgets fail rather than truncate the physical action.
    """
    if not seed_contract.labelled_bonds:
        raise ValueError(
            "complete PI grid inference requires ORACLE's frozen labelled bond graph"
        )
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("PI grid candidate budget must be a positive integer")
    if type(max_operations) is not int or max_operations < 1:
        raise ValueError("PI group expansion budget must be a positive integer")
    xyz = np.asarray(geometries_angstrom, dtype=float)
    masses = np.asarray(masses_amu, dtype=float)
    natoms = len(seed_contract.atoms)
    if (
        xyz.ndim != 3
        or xyz.shape[1:] != (natoms, 3)
        or not len(xyz)
        or not np.all(np.isfinite(xyz))
    ):
        raise ValueError("PI geometry grid must have finite shape (grid_size, natoms, 3)")
    if masses.shape != (natoms,) or np.any(masses <= 0) or not np.all(np.isfinite(masses)):
        raise ValueError("one finite positive mass is required per nucleus")

    graph = nx.Graph()
    for atom, (element, isotope) in enumerate(
        zip(seed_contract.atoms, seed_contract.nuclear_labels, strict=True)
    ):
        graph.add_node(atom, label=(element, isotope))
    for left, right, label in seed_contract.labelled_bonds:
        graph.add_edge(left, right, label=label)
    if not nx.is_connected(graph):
        raise ValueError("complete PI grid inference requires a connected ORACLE graph")
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        graph,
        graph,
        node_match=nx.algorithms.isomorphism.categorical_node_match("label", None),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match("label", None),
    )

    identity = PermutationInversionOperation(tuple(range(natoms)))
    qualified = {identity}
    candidate_count = 0
    for mapping in matcher.isomorphisms_iter():
        candidate_count += 2
        if candidate_count > max_candidates:
            raise PIExpansionLimit(
                f"PI grid candidate count exceeds the explicit {max_candidates}-candidate limit"
            )
        permutation = tuple(mapping[index] for index in range(natoms))
        for inverted in (False, True):
            operation = PermutationInversionOperation(permutation, inverted)
            if operation == identity:
                continue
            operations, words = _closure(natoms, (operation,), max_operations)
            cyclic = OraclePermutationInversion(
                atoms=seed_contract.atoms,
                nuclear_labels=seed_contract.nuclear_labels,
                geometry_sha256=seed_contract.geometry_sha256,
                point_group=seed_contract.point_group,
                generators=(operation,),
                generator_sources=("GRID:GEOMETRY_CERTIFIED",),
                operations=operations,
                words=words,
                rotors=(),
                rotor_diagnostics=(),
                tolerance_angstrom=tolerance_angstrom,
                labelled_bonds=seed_contract.labelled_bonds,
            )
            try:
                derive_permutation_inversion_grid_action(
                    cyclic, xyz, masses, tolerance_angstrom=tolerance_angstrom
                )
            except ValueError:
                continue
            qualified.update(operations)

    generators: list[PermutationInversionOperation] = []
    generated = {identity}
    ordered = sorted(
        qualified - {identity},
        key=lambda operation: (operation.inverted, operation.permutation),
    )
    for operation in ordered:
        if operation in generated:
            continue
        trial = (*generators, operation)
        operations, _words = _closure(natoms, trial, max_operations)
        if any(item not in qualified for item in operations):
            raise ValueError("geometry-certified PI operations are not closed under composition")
        generators.append(operation)
        generated = set(operations)
    operations, words = _closure(natoms, tuple(generators), max_operations)
    if set(operations) != qualified:
        raise ValueError("failed to generate every geometry-certified PI operation")
    digest = hashlib.sha256(np.asarray(xyz, dtype="<f8").tobytes()).hexdigest()
    contract = OraclePermutationInversion(
        atoms=seed_contract.atoms,
        nuclear_labels=seed_contract.nuclear_labels,
        geometry_sha256=digest,
        point_group=seed_contract.point_group,
        generators=tuple(generators),
        generator_sources=("GRID:GEOMETRY_CERTIFIED",) * len(generators),
        operations=operations,
        words=words,
        rotors=tuple(rotor for rotor in seed_contract.rotors if rotor.operation in qualified),
        rotor_diagnostics=seed_contract.rotor_diagnostics,
        tolerance_angstrom=max(seed_contract.tolerance_angstrom, tolerance_angstrom),
        labelled_bonds=seed_contract.labelled_bonds,
    )
    action = derive_permutation_inversion_grid_action(
        contract, xyz, masses, tolerance_angstrom=tolerance_angstrom
    )
    return contract, action
