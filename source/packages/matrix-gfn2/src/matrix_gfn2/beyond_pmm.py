"""Differential tight-binding models for post-MD discrete/continuum analysis.

The module does not run molecular dynamics and does not interpolate a
polarizable force field.  An accurate calculation is performed independently
for one fixed representative of each structural/environmental cluster.  A
resident tight-binding calculation then transfers only the fluctuation from
that representative to the other frames of the same cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np

from .electronic_state import GFN2OrbitalState
from .stda import ClosedShellSTDAResult


def _readonly(value, *, dtype=float) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    if not np.all(np.isfinite(array)):
        raise ValueError("Beyond-PMM arrays must be finite")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ClusterEnvironmentContract:
    """Immutable explicit/continuum environment definition for one cluster.

    ``explicit_atom_mask`` fixes which solute and near-solvent atoms belong to
    the electronic calculation.  A promotion or demotion across that boundary
    requires a new cluster and therefore a new contract.
    """

    cluster_id: str
    atomic_numbers: np.ndarray
    explicit_atom_mask: np.ndarray
    ellipsoid_semi_axes_angstrom: np.ndarray
    far_charge_model: str
    continuum_model: str
    hydrogen_bond_model: str | None = None

    def __post_init__(self) -> None:
        numbers = _readonly(self.atomic_numbers, dtype=np.int64)
        explicit = np.array(self.explicit_atom_mask, dtype=bool, copy=True)
        axes = _readonly(self.ellipsoid_semi_axes_angstrom)
        if numbers.ndim != 1 or explicit.shape != numbers.shape:
            raise ValueError("cluster atoms and explicit-region mask differ")
        if axes.shape != (3,) or np.any(axes <= 0.0):
            raise ValueError("ellipsoid semi-axes must be three positive values")
        if not np.any(explicit):
            raise ValueError("cluster electronic region cannot be empty")
        if not str(self.cluster_id).strip():
            raise ValueError("cluster identifier cannot be empty")
        if not str(self.far_charge_model).strip() or not str(self.continuum_model).strip():
            raise ValueError("far-charge and continuum models must be identified")
        explicit.setflags(write=False)
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "explicit_atom_mask", explicit)
        object.__setattr__(self, "ellipsoid_semi_axes_angstrom", axes)

    @property
    def fingerprint(self) -> str:
        payload = {
            "cluster_id": self.cluster_id,
            "atomic_numbers": self.atomic_numbers.tolist(),
            "explicit_atom_mask": self.explicit_atom_mask.astype(int).tolist(),
            "ellipsoid_semi_axes_angstrom": self.ellipsoid_semi_axes_angstrom.tolist(),
            "far_charge_model": self.far_charge_model,
            "continuum_model": self.continuum_model,
            "hydrogen_bond_model": self.hydrogen_bond_model,
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class ElectronicClusteringPolicy:
    """Immutable qualification limits for adaptive Beyond-PMM clusters."""

    initial_cluster_count: int = 1
    maximum_residual_hartree: float = 1.0e-3
    maximum_rms_residual_hartree: float = 5.0e-4
    minimum_cluster_size: int = 2
    maximum_split_depth: int = 12

    def __post_init__(self) -> None:
        if int(self.initial_cluster_count) < 1:
            raise ValueError("initial geometric cluster count must be positive")
        if (
            not np.isfinite(self.maximum_residual_hartree)
            or self.maximum_residual_hartree <= 0.0
            or not np.isfinite(self.maximum_rms_residual_hartree)
            or self.maximum_rms_residual_hartree <= 0.0
        ):
            raise ValueError("electronic residual thresholds must be finite and positive")
        if int(self.minimum_cluster_size) < 1 or int(self.maximum_split_depth) < 0:
            raise ValueError("cluster size and split depth must be nonnegative integers")


@dataclass(frozen=True)
class ElectronicClusterCertificate:
    """Complete electronic residual certificate for one final cluster."""

    member_indices: np.ndarray
    representative_index: int
    maximum_residual_hartree: float
    rms_residual_hartree: float
    split_depth: int
    qualified: bool

    def __post_init__(self) -> None:
        members = _readonly(self.member_indices, dtype=np.int64)
        if members.ndim != 1 or not len(members):
            raise ValueError("electronic cluster cannot be empty")
        if int(self.representative_index) not in set(members.tolist()):
            raise ValueError("cluster representative must be one of its members")
        if min(self.maximum_residual_hartree, self.rms_residual_hartree) < 0.0:
            raise ValueError("electronic residuals cannot be negative")
        if int(self.split_depth) < 0:
            raise ValueError("cluster split depth cannot be negative")
        object.__setattr__(self, "member_indices", members)


@dataclass(frozen=True)
class AdaptiveElectronicClusteringResult:
    """Final labels and auditable certificates from adaptive refinement."""

    labels: np.ndarray
    representative_indices: np.ndarray
    certificates: tuple[ElectronicClusterCertificate, ...]
    split_count: int
    fully_qualified: bool

    def __post_init__(self) -> None:
        labels = _readonly(self.labels, dtype=np.int64)
        representatives = _readonly(self.representative_indices, dtype=np.int64)
        if labels.ndim != 1 or representatives.shape != (len(self.certificates),):
            raise ValueError("adaptive-clustering result dimensions are inconsistent")
        if len(labels) and (
            np.min(labels) < 0 or np.max(labels) >= len(self.certificates)
        ):
            raise ValueError("adaptive-clustering labels are not contiguous")
        if int(self.split_count) < 0:
            raise ValueError("adaptive-clustering split count cannot be negative")
        if bool(self.fully_qualified) != all(
            certificate.qualified for certificate in self.certificates
        ):
            raise ValueError("aggregate and per-cluster qualification disagree")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "representative_indices", representatives)


def _squared_distances(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    delta = features[:, None, :] - centers[None, :, :]
    return np.einsum("nkd,nkd->nk", delta, delta, optimize=True)


def _deterministic_medoids(features: np.ndarray, count: int) -> np.ndarray:
    """Farthest-first seeds followed by deterministic medoid refinement."""

    samples = len(features)
    count = min(int(count), samples)
    centroid = np.mean(features, axis=0)
    first = int(np.argmin(np.sum((features - centroid) ** 2, axis=1)))
    medoids = [first]
    while len(medoids) < count:
        distance = np.min(_squared_distances(features, features[medoids]), axis=1)
        distance[np.asarray(medoids, dtype=np.int64)] = -1.0
        medoids.append(int(np.argmax(distance)))
    medoids_array = np.asarray(medoids, dtype=np.int64)
    for _ in range(64):
        labels = np.argmin(_squared_distances(features, features[medoids_array]), axis=1)
        updated = medoids_array.copy()
        for cluster in range(count):
            members = np.flatnonzero(labels == cluster)
            if not len(members):
                continue
            pair_distance = _squared_distances(features[members], features[members])
            updated[cluster] = members[int(np.argmin(np.sum(pair_distance, axis=1)))]
        if np.array_equal(updated, medoids_array):
            break
        medoids_array = updated
    return medoids_array


def _electronic_bisection(
    members: np.ndarray, electronic_descriptors: np.ndarray, residuals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    local = electronic_descriptors[members]
    pair_distance = _squared_distances(local, local)
    first, second = np.unravel_index(int(np.argmax(pair_distance)), pair_distance.shape)
    if float(pair_distance[first, second]) <= np.finfo(float).eps:
        order = np.lexsort((members, -residuals[members]))
        midpoint = max(1, len(members) // 2)
        return members[order[:midpoint]], members[order[midpoint:]]
    distance = _squared_distances(local, local[[first, second]])
    assignment = np.argmin(distance, axis=1)
    left = members[assignment == 0]
    right = members[assignment == 1]
    if not len(left) or not len(right):
        order = np.argsort(members, kind="stable")
        midpoint = max(1, len(members) // 2)
        left, right = members[order[:midpoint]], members[order[midpoint:]]
    return left, right


def adaptive_electronic_clustering(
    geometric_features: np.ndarray,
    electronic_descriptors: np.ndarray,
    reference_values_hartree: np.ndarray,
    approximate_values_hartree: np.ndarray,
    *,
    policy: ElectronicClusteringPolicy = ElectronicClusteringPolicy(),
) -> AdaptiveElectronicClusteringResult:
    """Build geometric clusters and refine them until electronic qualification.

    Certification is based on the error in *fluctuations relative to each
    cluster representative*, which is the quantity transferred by Beyond-PMM.
    Every frame must have a finite reference value: sampled validation is not
    silently promoted to a certificate.  A cluster that cannot be split within
    the policy remains in the result with ``qualified=False``.
    """

    geometry = _readonly(geometric_features)
    descriptors = _readonly(electronic_descriptors)
    reference = _readonly(reference_values_hartree)
    approximate = _readonly(approximate_values_hartree)
    if geometry.ndim != 2 or descriptors.ndim != 2 or not len(geometry):
        raise ValueError("geometric and electronic features must be nonempty matrices")
    if len(descriptors) != len(geometry):
        raise ValueError("geometric and electronic descriptor counts differ")
    if reference.ndim == 1:
        reference = reference[:, None]
    if approximate.ndim == 1:
        approximate = approximate[:, None]
    if reference.shape != approximate.shape or reference.shape[0] != len(geometry):
        raise ValueError("reference and approximate electronic values differ")

    initial_medoids = _deterministic_medoids(
        geometry, min(policy.initial_cluster_count, len(geometry))
    )
    initial_labels = np.argmin(
        _squared_distances(geometry, geometry[initial_medoids]), axis=1
    )
    pending = [
        (np.flatnonzero(initial_labels == cluster), 0)
        for cluster in range(len(initial_medoids))
    ]
    pending = [item for item in pending if len(item[0])]
    finished: list[ElectronicClusterCertificate] = []
    split_count = 0
    while pending:
        members, depth = pending.pop(0)
        local_medoid = _deterministic_medoids(geometry[members], 1)[0]
        representative = int(members[local_medoid])
        residual_matrix = (
            reference[members]
            - reference[representative]
            - approximate[members]
            + approximate[representative]
        )
        member_residual = np.max(np.abs(residual_matrix), axis=1)
        maximum = float(np.max(member_residual, initial=0.0))
        rms = float(np.sqrt(np.mean(residual_matrix * residual_matrix)))
        qualified = (
            maximum <= policy.maximum_residual_hartree
            and rms <= policy.maximum_rms_residual_hartree
        )
        can_split = (
            not qualified
            and depth < policy.maximum_split_depth
            and len(members) >= 2 * policy.minimum_cluster_size
        )
        if can_split:
            all_residual = np.zeros(len(geometry), dtype=float)
            all_residual[members] = member_residual
            left, right = _electronic_bisection(members, descriptors, all_residual)
            if len(left) >= policy.minimum_cluster_size and len(right) >= policy.minimum_cluster_size:
                pending.extend(((left, depth + 1), (right, depth + 1)))
                split_count += 1
                continue
        finished.append(
            ElectronicClusterCertificate(
                members,
                representative,
                maximum,
                rms,
                depth,
                qualified,
            )
        )
    finished.sort(key=lambda item: int(np.min(item.member_indices)))
    labels = np.empty(len(geometry), dtype=np.int64)
    for cluster, certificate in enumerate(finished):
        labels[certificate.member_indices] = cluster
    representatives = np.asarray(
        [item.representative_index for item in finished], dtype=np.int64
    )
    return AdaptiveElectronicClusteringResult(
        labels,
        representatives,
        tuple(finished),
        split_count,
        all(item.qualified for item in finished),
    )


@dataclass(frozen=True)
class DifferentialTBReference:
    """Accurate and tight-binding data at one cluster representative."""

    contract: ClusterEnvironmentContract
    coordinates_bohr: np.ndarray
    accurate_energy_hartree: float
    accurate_gradient_hartree_per_bohr: np.ndarray
    tb_energy_hartree: float
    tb_gradient_hartree_per_bohr: np.ndarray

    def __post_init__(self) -> None:
        coordinates = _readonly(self.coordinates_bohr)
        accurate = _readonly(self.accurate_gradient_hartree_per_bohr)
        tb = _readonly(self.tb_gradient_hartree_per_bohr)
        expected = (len(self.contract.atomic_numbers), 3)
        if coordinates.shape != expected or accurate.shape != expected or tb.shape != expected:
            raise ValueError("reference coordinates and gradients have incompatible dimensions")
        if not np.isfinite(self.accurate_energy_hartree) or not np.isfinite(
            self.tb_energy_hartree
        ):
            raise ValueError("reference energies must be finite")
        object.__setattr__(self, "coordinates_bohr", coordinates)
        object.__setattr__(self, "accurate_gradient_hartree_per_bohr", accurate)
        object.__setattr__(self, "tb_gradient_hartree_per_bohr", tb)


@dataclass(frozen=True)
class DifferentialTBResult:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    tb_energy_fluctuation_hartree: float
    contract_fingerprint: str

    def __post_init__(self) -> None:
        gradient = _readonly(self.gradient_hartree_per_bohr)
        if not np.isfinite(self.energy_hartree) or not np.isfinite(
            self.tb_energy_fluctuation_hartree
        ):
            raise ValueError("differential energies must be finite")
        object.__setattr__(self, "gradient_hartree_per_bohr", gradient)


def evaluate_differential_tb(
    reference: DifferentialTBReference,
    coordinates_bohr: np.ndarray,
    tb_energy_hartree: float,
    tb_gradient_hartree_per_bohr: np.ndarray,
    *,
    contract_fingerprint: str,
) -> DifferentialTBResult:
    """Evaluate the conservative accurate-reference plus TB fluctuation model.

    The linear reference terms make the returned gradient exactly the
    derivative of the returned energy at the cluster representative:

    ``E = E_A + g_A.dR + E_TB(R)-E_TB(R0)-g_TB(R0).dR``.
    """

    if contract_fingerprint != reference.contract.fingerprint:
        raise ValueError(
            "frame changed explicit solvent membership or environment contract; "
            "start a new cluster"
        )
    coordinates = np.asarray(coordinates_bohr, dtype=float)
    gradient = np.asarray(tb_gradient_hartree_per_bohr, dtype=float)
    if coordinates.shape != reference.coordinates_bohr.shape or gradient.shape != coordinates.shape:
        raise ValueError("frame coordinates and TB gradient dimensions differ from reference")
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(gradient)):
        raise ValueError("frame coordinates and TB gradient must be finite")
    displacement = coordinates - reference.coordinates_bohr
    tb_fluctuation = float(tb_energy_hartree) - float(reference.tb_energy_hartree)
    energy = (
        float(reference.accurate_energy_hartree)
        + float(np.sum(reference.accurate_gradient_hartree_per_bohr * displacement))
        + tb_fluctuation
        - float(np.sum(reference.tb_gradient_hartree_per_bohr * displacement))
    )
    corrected_gradient = (
        reference.accurate_gradient_hartree_per_bohr
        + gradient
        - reference.tb_gradient_hartree_per_bohr
    )
    return DifferentialTBResult(
        energy,
        corrected_gradient,
        tb_fluctuation,
        contract_fingerprint,
    )


@dataclass(frozen=True)
class DiabaticManifold:
    """Small projected Hamiltonian and its nuclear derivative."""

    hamiltonian_hartree: np.ndarray
    derivative_hartree_per_bohr: np.ndarray
    state_descriptors: np.ndarray

    def __post_init__(self) -> None:
        hamiltonian = _readonly(self.hamiltonian_hartree)
        derivative = _readonly(self.derivative_hartree_per_bohr)
        descriptors = _readonly(self.state_descriptors)
        if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
            raise ValueError("diabatic Hamiltonian must be square")
        states = hamiltonian.shape[0]
        if derivative.ndim != 4 or derivative.shape[-2:] != (states, states):
            raise ValueError("diabatic derivative must have natom x 3 x nstate x nstate shape")
        if derivative.shape[1] != 3 or descriptors.ndim != 2 or descriptors.shape[0] != states:
            raise ValueError("diabatic derivative or descriptors have incompatible dimensions")
        if not np.allclose(hamiltonian, hamiltonian.T, atol=2.0e-12, rtol=0.0):
            raise ValueError("diabatic Hamiltonian must be symmetric")
        if not np.allclose(
            derivative, derivative.swapaxes(-1, -2), atol=2.0e-12, rtol=0.0
        ):
            raise ValueError("diabatic Hamiltonian derivative must be symmetric")
        norms = np.linalg.norm(descriptors, axis=1)
        if np.any(norms <= np.finfo(float).tiny):
            raise ValueError("every state descriptor must have nonzero norm")
        object.__setattr__(self, "hamiltonian_hartree", hamiltonian)
        object.__setattr__(self, "derivative_hartree_per_bohr", derivative)
        object.__setattr__(self, "state_descriptors", descriptors / norms[:, None])

    @property
    def nstates(self) -> int:
        return int(self.hamiltonian_hartree.shape[0])


@dataclass(frozen=True)
class DifferentialStateReference:
    contract: ClusterEnvironmentContract
    accurate: DiabaticManifold
    tight_binding: DiabaticManifold

    def __post_init__(self) -> None:
        if self.accurate.hamiltonian_hartree.shape != self.tight_binding.hamiltonian_hartree.shape:
            raise ValueError("accurate and TB reference manifolds have different sizes")
        if self.accurate.derivative_hartree_per_bohr.shape != self.tight_binding.derivative_hartree_per_bohr.shape:
            raise ValueError("accurate and TB reference derivatives have different dimensions")


@dataclass(frozen=True)
class DifferentialStateResult:
    energies_hartree: np.ndarray
    gradients_hartree_per_bohr: np.ndarray
    eigenvectors: np.ndarray
    state_assignment: np.ndarray
    descriptor_overlaps: np.ndarray
    minimum_energy_gaps_hartree: np.ndarray
    contract_fingerprint: str

    def __post_init__(self) -> None:
        for name, dtype in (
            ("energies_hartree", float),
            ("gradients_hartree_per_bohr", float),
            ("eigenvectors", float),
            ("state_assignment", np.int64),
            ("descriptor_overlaps", float),
            ("minimum_energy_gaps_hartree", float),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype=dtype))


@dataclass(frozen=True)
class ProjectedChargeTDAReference:
    """Frozen TDA root space for repeated external-charge configurations.

    The stored operators are the derivative of the retained-root Hamiltonian
    with respect to the electrostatic potential at every QM atom.  Thus a new
    solvent frame never rebuilds the AO integrals, SCF, or TDA kernel.  The
    approximation is explicit: the QM geometry, orbitals, configuration space,
    and retained TDA roots belong to one immutable cluster representative.
    """

    coordinates_bohr: np.ndarray
    point_charges_electron: np.ndarray
    reference_point_coordinates_bohr: np.ndarray
    reference_atomic_potential_hartree_per_electron: np.ndarray
    excitation_energies_hartree: np.ndarray
    atomic_root_operators: np.ndarray

    def __post_init__(self) -> None:
        coordinates = _readonly(self.coordinates_bohr)
        charges = _readonly(self.point_charges_electron)
        points = _readonly(self.reference_point_coordinates_bohr)
        potential = _readonly(self.reference_atomic_potential_hartree_per_electron)
        energies = _readonly(self.excitation_energies_hartree)
        operators = _readonly(self.atomic_root_operators)
        natoms = len(coordinates)
        roots = len(energies)
        if coordinates.ndim != 2 or coordinates.shape[1:] != (3,):
            raise ValueError("projected-charge QM coordinates must have shape natom x 3")
        if charges.ndim != 1 or points.shape != (len(charges), 3):
            raise ValueError("projected-charge values and coordinates differ")
        if potential.shape != (natoms,):
            raise ValueError("reference atomic potential has the wrong dimension")
        if energies.ndim != 1 or not roots:
            raise ValueError("projected-charge TDA space cannot be empty")
        if operators.shape != (natoms, roots, roots):
            raise ValueError("atomic root operators have incompatible dimensions")
        if not np.allclose(
            operators, operators.swapaxes(-1, -2), atol=2.0e-12, rtol=0.0
        ):
            raise ValueError("atomic root operators must be symmetric")
        object.__setattr__(self, "coordinates_bohr", coordinates)
        object.__setattr__(self, "point_charges_electron", charges)
        object.__setattr__(self, "reference_point_coordinates_bohr", points)
        object.__setattr__(
            self, "reference_atomic_potential_hartree_per_electron", potential
        )
        object.__setattr__(self, "excitation_energies_hartree", energies)
        object.__setattr__(self, "atomic_root_operators", operators)


@dataclass(frozen=True)
class ProjectedChargeTDAResult:
    excitation_energies_hartree: np.ndarray
    eigenvectors: np.ndarray
    point_charge_gradients_hartree_per_bohr: np.ndarray
    atomic_potential_change_hartree_per_electron: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "excitation_energies_hartree",
            "eigenvectors",
            "point_charge_gradients_hartree_per_bohr",
            "atomic_potential_change_hartree_per_electron",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


def _atomic_point_charge_potential(
    coordinates_bohr: np.ndarray,
    point_coordinates_bohr: np.ndarray,
    point_charges_electron: np.ndarray,
    *,
    derivatives: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    displacement = coordinates_bohr[:, None, :] - point_coordinates_bohr[None, :, :]
    distance = np.linalg.norm(displacement, axis=2)
    if np.any(distance < 1.0e-8):
        raise ValueError("point charge coincides with a QM atom")
    inverse = 1.0 / distance
    potential = inverse @ point_charges_electron
    if not derivatives:
        return potential, None
    point_derivative = (
        point_charges_electron[None, :, None]
        * displacement
        / distance[:, :, None] ** 3
    ).transpose(1, 0, 2)
    return potential, point_derivative


def build_projected_charge_tda_reference(
    state: GFN2OrbitalState,
    tda: ClosedShellSTDAResult,
    point_charges_electron: np.ndarray,
    point_charge_coordinates_bohr: np.ndarray,
) -> ProjectedChargeTDAReference:
    """Compile the fixed-orbital charge response in a retained TDA root space."""

    charges = np.asarray(point_charges_electron, dtype=float)
    points = np.asarray(point_charge_coordinates_bohr, dtype=float)
    if charges.ndim != 1 or points.shape != (len(charges), 3):
        raise ValueError("point-charge values and coordinates must have matching shapes")
    amplitudes = np.asarray(tda.amplitudes, dtype=float)
    occupied = np.asarray(tda.occupied_orbitals, dtype=np.int64)
    virtual = np.asarray(tda.virtual_orbitals, dtype=np.int64)
    if amplitudes.ndim != 2 or amplitudes.shape[1] != len(occupied):
        raise ValueError("TDA amplitudes and selected configurations differ")
    if virtual.shape != occupied.shape:
        raise ValueError("TDA occupied and virtual configuration arrays differ")
    if np.any(occupied < 0) or np.any(virtual < 0) or (
        np.max(np.concatenate((occupied, virtual)), initial=-1) >= state.nmo
    ):
        raise ValueError("TDA configuration orbital lies outside its reference state")

    used = np.unique(np.concatenate((occupied, virtual)))
    inverse_orbital = np.full(state.nmo, -1, dtype=np.int64)
    inverse_orbital[used] = np.arange(len(used))
    occupied_local = inverse_orbital[occupied]
    virtual_local = inverse_orbital[virtual]
    coefficients = np.asarray(state.coefficients[:, used], dtype=float)
    overlap = np.asarray(state.overlap, dtype=float)
    ao_atom = np.asarray(state.ao_atom, dtype=np.int64)
    nconfig = len(occupied)
    roots = amplitudes.shape[0]
    operators = np.empty((len(state.atomic_numbers), roots, roots), dtype=float)
    for atom in range(len(state.atomic_numbers)):
        selector = (ao_atom == atom).astype(float)
        ao_operator = -0.5 * overlap * (selector[:, None] + selector[None, :])
        mo_operator = coefficients.T @ ao_operator @ coefficients
        configuration_operator = np.zeros((nconfig, nconfig), dtype=float)
        same_occupied = occupied[:, None] == occupied[None, :]
        same_virtual = virtual[:, None] == virtual[None, :]
        configuration_operator[same_occupied] += mo_operator[
            virtual_local[:, None], virtual_local[None, :]
        ][same_occupied]
        configuration_operator[same_virtual] -= mo_operator[
            occupied_local[None, :], occupied_local[:, None]
        ][same_virtual]
        operators[atom] = amplitudes @ configuration_operator @ amplitudes.T
    operators = 0.5 * (operators + operators.swapaxes(-1, -2))
    potential, _ = _atomic_point_charge_potential(
        np.asarray(state.coordinates_bohr), points, charges, derivatives=False
    )
    return ProjectedChargeTDAReference(
        state.coordinates_bohr,
        charges,
        points,
        potential,
        tda.excitation_energies_hartree,
        operators,
    )


def evaluate_projected_charge_tda(
    reference: ProjectedChargeTDAReference,
    point_charge_coordinates_bohr: np.ndarray,
    *,
    point_charges_electron: np.ndarray | None = None,
) -> ProjectedChargeTDAResult:
    """Rediagonalize the retained TDA space for one solvent-charge frame.

    Returned gradients are derivatives with respect to the point-charge
    coordinates.  The QM nuclei are fixed by this post-MD protocol; their
    relaxed analytic gradients belong to the full differential-TB path.
    """

    points = np.asarray(point_charge_coordinates_bohr, dtype=float)
    charges = np.asarray(
        reference.point_charges_electron
        if point_charges_electron is None
        else point_charges_electron,
        dtype=float,
    )
    if charges.shape != reference.point_charges_electron.shape or points.shape != (
        len(charges),
        3,
    ):
        raise ValueError("frame point-charge set differs from its reference")
    potential, point_derivative = _atomic_point_charge_potential(
        reference.coordinates_bohr, points, charges, derivatives=True
    )
    delta_potential = potential - reference.reference_atomic_potential_hartree_per_electron
    hamiltonian = np.diag(reference.excitation_energies_hartree) + np.einsum(
        "a,aij->ij", delta_potential, reference.atomic_root_operators, optimize=True
    )
    energies, vectors = np.linalg.eigh(hamiltonian)
    derivative = np.einsum(
        "pax,aij->pxij", point_derivative, reference.atomic_root_operators, optimize=True
    )
    gradients = np.einsum(
        "ir,pxij,jr->rpx", vectors, derivative, vectors, optimize=True
    )
    return ProjectedChargeTDAResult(energies, vectors, gradients, delta_potential)


def _maximum_overlap_assignment(overlap: np.ndarray) -> np.ndarray:
    """Exact small-manifold assignment by dynamic programming, O(m 2**m)."""

    weights = np.asarray(overlap, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("state-overlap matrix must be square")
    size = len(weights)
    if size > 16:
        raise ValueError("retained Beyond-PMM manifold is too large for exact assignment")
    scores = {0: (0.0, ())}
    for row in range(size):
        updated: dict[int, tuple[float, tuple[int, ...]]] = {}
        for mask, (score, assignment) in scores.items():
            for column in range(size):
                bit = 1 << column
                if mask & bit:
                    continue
                candidate = (score + abs(float(weights[row, column])), assignment + (column,))
                key = mask | bit
                if key not in updated or candidate[0] > updated[key][0]:
                    updated[key] = candidate
        scores = updated
    return np.asarray(scores[(1 << size) - 1][1], dtype=np.int64)


def evaluate_differential_states(
    reference: DifferentialStateReference,
    frame_tight_binding: DiabaticManifold,
    *,
    contract_fingerprint: str,
    minimum_descriptor_overlap: float = 0.5,
) -> DifferentialStateResult:
    """Diagonalize ``H_A(ref) + H_TB(frame) - H_TB(ref)`` and its gradient."""

    if contract_fingerprint != reference.contract.fingerprint:
        raise ValueError("frame environment differs from its cluster reference")
    if frame_tight_binding.nstates != reference.tight_binding.nstates:
        raise ValueError("frame and reference retained manifolds have different sizes")
    overlap = reference.tight_binding.state_descriptors @ frame_tight_binding.state_descriptors.T
    assignment = _maximum_overlap_assignment(overlap)
    matched = overlap[np.arange(len(assignment)), assignment]
    if np.any(np.abs(matched) < float(minimum_descriptor_overlap)):
        raise ValueError("electronic character is discontinuous; split or expand the cluster")
    phases = np.where(matched < 0.0, -1.0, 1.0)
    transform = np.zeros((len(assignment), len(assignment)), dtype=float)
    transform[assignment, np.arange(len(assignment))] = phases
    frame_hamiltonian = transform.T @ frame_tight_binding.hamiltonian_hartree @ transform
    frame_derivative = np.einsum(
        "pi,...pq,qj->...ij",
        transform,
        frame_tight_binding.derivative_hartree_per_bohr,
        transform,
        optimize=True,
    )
    corrected_hamiltonian = (
        reference.accurate.hamiltonian_hartree
        + frame_hamiltonian
        - reference.tight_binding.hamiltonian_hartree
    )
    corrected_derivative = (
        reference.accurate.derivative_hartree_per_bohr
        + frame_derivative
        - reference.tight_binding.derivative_hartree_per_bohr
    )
    energies, vectors = np.linalg.eigh(corrected_hamiltonian)
    gradients = np.einsum(
        "ir,...ij,jr->r...", vectors, corrected_derivative, vectors, optimize=True
    )
    gaps = np.abs(energies[:, None] - energies[None, :])
    np.fill_diagonal(gaps, np.inf)
    minimum_gaps = np.min(gaps, axis=1) if len(energies) > 1 else np.asarray((np.inf,))
    return DifferentialStateResult(
        energies,
        gradients,
        vectors,
        assignment,
        np.abs(matched),
        minimum_gaps,
        contract_fingerprint,
    )


__all__ = [
    "AdaptiveElectronicClusteringResult",
    "ClusterEnvironmentContract",
    "DiabaticManifold",
    "ElectronicClusterCertificate",
    "ElectronicClusteringPolicy",
    "DifferentialStateReference",
    "DifferentialStateResult",
    "DifferentialTBReference",
    "DifferentialTBResult",
    "ProjectedChargeTDAReference",
    "ProjectedChargeTDAResult",
    "build_projected_charge_tda_reference",
    "adaptive_electronic_clustering",
    "evaluate_differential_states",
    "evaluate_differential_tb",
    "evaluate_projected_charge_tda",
]
