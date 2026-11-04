"""Dissociative ZAFF ring potentials in balanced triangular-flap coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from .bonded import BondOrderRadialFactor, InternalDerivatives


ZAFF_FLAP_RING_SCHEMA = "matrix.zaff.flap_ring.v1"


def balanced_flap_stencils(
    ring_atoms: Sequence[int],
) -> tuple[tuple[int, int, int, int], ...]:
    """Return the deterministic balanced ``N-3`` flap tree used by SMITH.

    The three anchor gaps differ by at most one.  Every returned tuple is
    ``(parent_third, hinge_start, hinge_end, child_third)``; the hinge can be
    a triangulation diagonal and is intentionally independent of the cyclic
    covalent-bond list used by the dissociation envelope.
    """

    ring = tuple(int(atom) for atom in ring_atoms)
    size = len(ring)
    if size < 4 or len(set(ring)) != size or min(ring) < 0:
        raise ValueError("balanced flap stencils require at least four distinct atoms")
    quotient, remainder = divmod(size, 3)
    gaps = tuple(quotient + (1 if index < remainder else 0) for index in range(3))
    anchors = (0, gaps[0], gaps[0] + gaps[1])
    stencils: list[tuple[int, int, int, int]] = []
    for arc in range(3):
        start = anchors[arc]
        end = anchors[(arc + 1) % 3]
        parent = anchors[(arc + 2) % 3]
        interior: list[int] = []
        index = (start + 1) % size
        while index != end:
            interior.append(index)
            index = (index + 1) % size
        if not interior:
            continue
        stencils.append((ring[parent], ring[start], ring[end], ring[interior[-1]]))
        for position in range(len(interior) - 1, 0, -1):
            stencils.append(
                (ring[end], ring[start], ring[interior[position]], ring[interior[position - 1]])
            )
    if len(stencils) != size - 3:
        raise RuntimeError("balanced triangulation did not produce N-3 flap coordinates")
    return tuple(stencils)


@dataclass(frozen=True)
class ZaffDissociativeFlapRingPotential:
    r"""A local, periodic and dissociative ``N-3`` flap-ring potential.

    The angular surface is

    ``sum_i K_ii (1-cos(d_i)) + sum_i<j K_ij sin(d_i) sin(d_j)``

    plus optional positive ``(1-cos(d_i))**2`` stabilizers.  The complete
    surface is multiplied by the product of the reference-normalized ORACLE
    bond-order factors for *all real cyclic bonds*.  Triangulation diagonals
    therefore define flap coordinates but never masquerade as covalent bonds.

    At the reference geometry the energy and gradient vanish and the flap
    Hessian is exactly the sparse matrix defined by
    ``flap_force_constants_hartree`` and ``flap_couplings_hartree``.
    Stretching any one
    cyclic bond to dissociation removes the complete ring term smoothly.
    """

    ring_atoms: tuple[int, ...]
    flap_atoms: tuple[tuple[int, int, int, int], ...]
    reference_flap_angles_radian: tuple[float, ...]
    flap_force_constants_hartree: tuple[float, ...]
    ring_bond_factors: tuple[BondOrderRadialFactor, ...]
    flap_couplings_hartree: tuple[tuple[int, int, float], ...] = ()
    quartic_stabilizers_hartree: tuple[float, ...] = ()
    parameter_source: str = "ZAFF_LOCAL_RING"
    schema: str = ZAFF_FLAP_RING_SCHEMA

    def __post_init__(self) -> None:
        ring = tuple(int(atom) for atom in self.ring_atoms)
        flaps = tuple(tuple(int(atom) for atom in item) for item in self.flap_atoms)
        reference = tuple(float(value) for value in self.reference_flap_angles_radian)
        factors = tuple(self.ring_bond_factors)
        dimension = len(ring) - 3
        if len(ring) < 4 or len(set(ring)) != len(ring) or min(ring) < 0:
            raise ValueError("a flap-ring potential requires at least four distinct atoms")
        if len(flaps) != dimension or len(reference) != dimension:
            raise ValueError("a ring requires exactly N-3 flap coordinates")
        if any(len(item) != 4 or len(set(item)) != 4 for item in flaps):
            raise ValueError("each flap must contain four distinct atoms")
        if any(atom not in ring for item in flaps for atom in item):
            raise ValueError("flap atoms must belong to the ring")
        if len(factors) != len(ring) or any(
            not isinstance(item, BondOrderRadialFactor) for item in factors
        ):
            raise ValueError("one bond-order factor is required for every cyclic ring bond")
        if any(not isfinite(value) for value in reference):
            raise ValueError("reference flap angles must be finite")
        diagonal = tuple(float(value) for value in self.flap_force_constants_hartree)
        if len(diagonal) != dimension or any(
            not isfinite(value) or value < 0.0 for value in diagonal
        ):
            raise ValueError("one non-negative force constant is required per flap")
        couplings = tuple(
            (int(left), int(right), float(value))
            for left, right, value in self.flap_couplings_hartree
        )
        if any(
            left < 0
            or right <= left
            or right >= dimension
            or not isfinite(value)
            for left, right, value in couplings
        ) or len({(left, right) for left, right, _value in couplings}) != len(couplings):
            raise ValueError("flap couplings require unique ordered in-range pairs")
        absolute_coupling_sum = np.zeros(dimension, dtype=float)
        for left, right, value in couplings:
            absolute_coupling_sum[left] += abs(value)
            absolute_coupling_sum[right] += abs(value)
        if np.any(np.asarray(diagonal) + 1.0e-12 < absolute_coupling_sum):
            raise ValueError(
                "flap force constants must satisfy the sparse positive-semidefinite "
                "diagonal-dominance contract"
            )
        stabilizers = (
            (0.0,) * dimension
            if not self.quartic_stabilizers_hartree
            else tuple(float(value) for value in self.quartic_stabilizers_hartree)
        )
        if len(stabilizers) != dimension or any(
            not isfinite(value) or value < 0.0 for value in stabilizers
        ):
            raise ValueError("flap quartic stabilizers must be finite and non-negative")
        if not str(self.parameter_source).strip() or self.schema != ZAFF_FLAP_RING_SCHEMA:
            raise ValueError("flap-ring provenance or schema is invalid")
        object.__setattr__(self, "ring_atoms", ring)
        object.__setattr__(self, "flap_atoms", flaps)
        object.__setattr__(self, "reference_flap_angles_radian", reference)
        object.__setattr__(self, "flap_force_constants_hartree", diagonal)
        object.__setattr__(self, "flap_couplings_hartree", couplings)
        object.__setattr__(self, "ring_bond_factors", factors)
        object.__setattr__(self, "quartic_stabilizers_hartree", stabilizers)

    @property
    def ring_bonds(self) -> tuple[tuple[int, int], ...]:
        """Return the real cyclic bonds, distinct from flap triangulation hinges."""

        return tuple(
            (self.ring_atoms[index], self.ring_atoms[(index + 1) % len(self.ring_atoms)])
            for index in range(len(self.ring_atoms))
        )

    def derivatives(
        self,
        ring_bond_distances_angstrom: Sequence[float],
        flap_angles_radian: Sequence[float],
    ) -> InternalDerivatives:
        """Return E, gradient and a diagnostic dense Hessian.

        Production optimizers should use :meth:`hessian_vector_product`, whose
        cost and storage are linear in the ring size plus the sparse coupling
        count.  This explicit Hessian entry point is retained for small-block
        validation and conventional force-field consumers.
        """

        distances = np.asarray(ring_bond_distances_angstrom, dtype=float).reshape(-1)
        angles = np.asarray(flap_angles_radian, dtype=float).reshape(-1)
        bond_count = len(self.ring_atoms)
        flap_count = bond_count - 3
        if distances.shape != (bond_count,) or np.any(~np.isfinite(distances)):
            raise ValueError("ring-bond distances have invalid dimensions")
        if np.any(distances <= 0.0):
            raise ValueError("ring-bond distances must be positive")
        if angles.shape != (flap_count,) or np.any(~np.isfinite(angles)):
            raise ValueError("flap angles have invalid dimensions")

        switches = tuple(
            factor.derivatives(float(distance))
            for factor, distance in zip(self.ring_bond_factors, distances, strict=True)
        )
        switch_value, switch_gradient, switch_hessian = _product_derivatives(switches)
        angular_value, angular_gradient, angular_hessian = self._angular_derivatives(angles)

        size = bond_count + flap_count
        gradient = np.empty(size, dtype=float)
        gradient[:bond_count] = angular_value * switch_gradient
        gradient[bond_count:] = switch_value * angular_gradient
        hessian = np.empty((size, size), dtype=float)
        hessian[:bond_count, :bond_count] = angular_value * switch_hessian
        cross = np.outer(switch_gradient, angular_gradient)
        hessian[:bond_count, bond_count:] = cross
        hessian[bond_count:, :bond_count] = cross.T
        hessian[bond_count:, bond_count:] = switch_value * angular_hessian
        return InternalDerivatives(
            energy=float(switch_value * angular_value),
            gradient=gradient,
            hessian=hessian,
        )

    def hessian_vector_product(
        self,
        ring_bond_distances_angstrom: Sequence[float],
        flap_angles_radian: Sequence[float],
        direction: Sequence[float],
    ) -> np.ndarray:
        """Apply the exact internal Hessian without constructing a dense block."""

        return self.energy_gradient_hessian_vector(
            ring_bond_distances_angstrom, flap_angles_radian, direction
        )[2]

    def energy_gradient_hessian_vector(
        self,
        ring_bond_distances_angstrom: Sequence[float],
        flap_angles_radian: Sequence[float],
        direction: Sequence[float] | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Return E/G/Hv directly in ring variables, without a dense Hessian.

        An omitted direction gives a zero product, allowing E/G consumers
        to share the same linear-storage switch and sparse-coupling kernels.
        """

        distances = np.asarray(ring_bond_distances_angstrom, dtype=float).reshape(-1)
        angles = np.asarray(flap_angles_radian, dtype=float).reshape(-1)
        bond_count = len(self.ring_atoms)
        flap_count = bond_count - 3
        vector = (np.zeros(bond_count + flap_count) if direction is None
                  else np.asarray(direction, dtype=float).reshape(-1))
        if distances.shape != (bond_count,) or np.any(~np.isfinite(distances)):
            raise ValueError("ring-bond distances have invalid dimensions")
        if np.any(distances <= 0.0):
            raise ValueError("ring-bond distances must be positive")
        if angles.shape != (flap_count,) or np.any(~np.isfinite(angles)):
            raise ValueError("flap angles have invalid dimensions")
        if vector.shape != (bond_count + flap_count,) or np.any(~np.isfinite(vector)):
            raise ValueError("flap-ring Hessian direction has invalid dimensions")
        switches = tuple(
            factor.derivatives(float(distance))
            for factor, distance in zip(self.ring_bond_factors, distances, strict=True)
        )
        switch_value, switch_gradient, switch_hvp = _product_hessian_vector(
            switches, vector[:bond_count]
        )
        angular_value, angular_gradient, angular_hvp = self._angular_hessian_vector(
            angles, vector[bond_count:]
        )
        radial_dot = float(np.dot(switch_gradient, vector[:bond_count]))
        angular_dot = float(np.dot(angular_gradient, vector[bond_count:]))
        product = np.concatenate(
            (
                angular_value * switch_hvp + angular_dot * switch_gradient,
                radial_dot * angular_gradient + switch_value * angular_hvp,
            )
        )
        gradient = np.concatenate((angular_value * switch_gradient, switch_value * angular_gradient))
        return float(switch_value * angular_value), gradient, product

    def energy(
        self,
        ring_bond_distances_angstrom: Sequence[float],
        flap_angles_radian: Sequence[float],
    ) -> float:
        distances = np.asarray(ring_bond_distances_angstrom, dtype=float).reshape(-1)
        angles = np.asarray(flap_angles_radian, dtype=float).reshape(-1)
        if distances.shape != (len(self.ring_atoms),) or np.any(~np.isfinite(distances)):
            raise ValueError("ring-bond distances have invalid dimensions")
        if np.any(distances <= 0.0):
            raise ValueError("ring-bond distances must be positive")
        if angles.shape != (len(self.flap_atoms),) or np.any(~np.isfinite(angles)):
            raise ValueError("flap angles have invalid dimensions")
        switch = 1.0
        for factor, distance in zip(self.ring_bond_factors, distances, strict=True):
            switch *= factor.value(float(distance))
        delta = angles - np.asarray(self.reference_flap_angles_radian, dtype=float)
        sine = np.sin(delta)
        one_minus_cosine = 1.0 - np.cos(delta)
        angular = float(
            np.dot(self.flap_force_constants_hartree, one_minus_cosine)
            + np.dot(self.quartic_stabilizers_hartree, one_minus_cosine**2)
        )
        for left, right, coupling in self.flap_couplings_hartree:
            angular += coupling * sine[left] * sine[right]
        return float(switch * angular)

    def _angular_derivatives(
        self, angles: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        delta = angles - np.asarray(self.reference_flap_angles_radian, dtype=float)
        sine = np.sin(delta)
        cosine = np.cos(delta)
        one_minus_cosine = 1.0 - cosine
        diagonal = np.asarray(self.flap_force_constants_hartree, dtype=float)
        stabilizers = np.asarray(self.quartic_stabilizers_hartree, dtype=float)
        value = float(np.dot(diagonal, one_minus_cosine) + np.dot(stabilizers, one_minus_cosine**2))
        neighbor_sine = np.zeros(len(delta), dtype=float)
        hessian = np.zeros((len(delta), len(delta)), dtype=float)
        for left, right, coupling in self.flap_couplings_hartree:
            value += coupling * sine[left] * sine[right]
            neighbor_sine[left] += coupling * sine[right]
            neighbor_sine[right] += coupling * sine[left]
            hessian[left, right] = hessian[right, left] = (
                coupling * cosine[left] * cosine[right]
            )
        gradient = diagonal * sine + cosine * neighbor_sine + 2.0 * stabilizers * one_minus_cosine * sine
        hessian[np.diag_indices(len(delta))] = (
            diagonal * cosine
            - sine * neighbor_sine
            + 2.0
            * stabilizers
            * (sine**2 + one_minus_cosine * cosine)
        )
        return value, gradient, hessian

    def _angular_hessian_vector(
        self, angles: np.ndarray, direction: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        delta = angles - np.asarray(self.reference_flap_angles_radian, dtype=float)
        sine = np.sin(delta)
        cosine = np.cos(delta)
        one_minus_cosine = 1.0 - cosine
        diagonal = np.asarray(self.flap_force_constants_hartree, dtype=float)
        stabilizers = np.asarray(self.quartic_stabilizers_hartree, dtype=float)
        value = float(
            np.dot(diagonal, one_minus_cosine)
            + np.dot(stabilizers, one_minus_cosine**2)
        )
        neighbor_sine = np.zeros(len(delta), dtype=float)
        product = np.zeros(len(delta), dtype=float)
        for left, right, coupling in self.flap_couplings_hartree:
            value += coupling * sine[left] * sine[right]
            neighbor_sine[left] += coupling * sine[right]
            neighbor_sine[right] += coupling * sine[left]
            off = coupling * cosine[left] * cosine[right]
            product[left] += off * direction[right]
            product[right] += off * direction[left]
        gradient = (
            diagonal * sine
            + cosine * neighbor_sine
            + 2.0 * stabilizers * one_minus_cosine * sine
        )
        local_diagonal = (
            diagonal * cosine
            - sine * neighbor_sine
            + 2.0 * stabilizers * (sine**2 + one_minus_cosine * cosine)
        )
        product += local_diagonal * direction
        return value, gradient, product

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "ring_atoms": list(self.ring_atoms),
            "ring_bonds": [list(item) for item in self.ring_bonds],
            "flap_atoms": [list(item) for item in self.flap_atoms],
            "reference_flap_angles_radian": list(self.reference_flap_angles_radian),
            "flap_force_constants_hartree": list(self.flap_force_constants_hartree),
            "flap_couplings_hartree": [list(item) for item in self.flap_couplings_hartree],
            "quartic_stabilizers_hartree": list(self.quartic_stabilizers_hartree),
            "ring_bond_factors": [_factor_to_dict(item) for item in self.ring_bond_factors],
            "parameter_source": self.parameter_source,
            "functional_form": "BOND_ORDER_PRODUCT_TIMES_PERIODIC_BALANCED_FLAPS",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ZaffDissociativeFlapRingPotential":
        if payload.get("schema") != ZAFF_FLAP_RING_SCHEMA:
            raise ValueError("unsupported ZAFF flap-ring schema")
        return cls(
            ring_atoms=tuple(int(value) for value in payload["ring_atoms"]),
            flap_atoms=tuple(
                tuple(int(value) for value in item) for item in payload["flap_atoms"]
            ),
            reference_flap_angles_radian=tuple(
                float(value) for value in payload["reference_flap_angles_radian"]
            ),
            flap_force_constants_hartree=tuple(
                float(value) for value in payload["flap_force_constants_hartree"]
            ),
            ring_bond_factors=tuple(
                _factor_from_dict(item) for item in payload["ring_bond_factors"]
            ),
            flap_couplings_hartree=tuple(
                (int(item[0]), int(item[1]), float(item[2]))
                for item in payload.get("flap_couplings_hartree", ())
            ),
            quartic_stabilizers_hartree=tuple(
                float(value) for value in payload.get("quartic_stabilizers_hartree", ())
            ),
            parameter_source=str(payload.get("parameter_source", "ZAFF_LOCAL_RING")),
        )


def _product_derivatives(
    factors: Sequence[Any],
) -> tuple[float, np.ndarray, np.ndarray]:
    count = len(factors)
    prefix = np.ones(count + 1, dtype=float)
    suffix = np.ones(count + 1, dtype=float)
    for index, factor in enumerate(factors):
        prefix[index + 1] = prefix[index] * float(factor.value)
    for index in range(count - 1, -1, -1):
        suffix[index] = suffix[index + 1] * float(factors[index].value)
    gradient = np.asarray(
        [float(factors[index].first) * prefix[index] * suffix[index + 1] for index in range(count)]
    )
    hessian = np.zeros((count, count), dtype=float)
    for index, factor in enumerate(factors):
        hessian[index, index] = (
            float(factor.second) * prefix[index] * suffix[index + 1]
        )
        middle = 1.0
        for second in range(index + 1, count):
            value = (
                float(factor.first)
                * float(factors[second].first)
                * prefix[index]
                * middle
                * suffix[second + 1]
            )
            hessian[index, second] = hessian[second, index] = value
            middle *= float(factors[second].value)
    return float(prefix[-1]), gradient, hessian


def _product_hessian_vector(
    factors: Sequence[Any], direction: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Product value, gradient and Hessian-vector in strict linear work."""

    count = len(factors)
    prefix_value = np.ones(count + 1, dtype=float)
    prefix_dot = np.zeros(count + 1, dtype=float)
    suffix_value = np.ones(count + 1, dtype=float)
    suffix_dot = np.zeros(count + 1, dtype=float)
    for index, factor in enumerate(factors):
        value = float(factor.value)
        derivative_dot = float(factor.first) * float(direction[index])
        prefix_value[index + 1] = prefix_value[index] * value
        prefix_dot[index + 1] = prefix_dot[index] * value + prefix_value[index] * derivative_dot
    for index in range(count - 1, -1, -1):
        value = float(factors[index].value)
        derivative_dot = float(factors[index].first) * float(direction[index])
        suffix_value[index] = value * suffix_value[index + 1]
        suffix_dot[index] = derivative_dot * suffix_value[index + 1] + value * suffix_dot[index + 1]
    gradient = np.empty(count, dtype=float)
    product = np.empty(count, dtype=float)
    for index, factor in enumerate(factors):
        excluded_value = prefix_value[index] * suffix_value[index + 1]
        excluded_dot = (
            prefix_dot[index] * suffix_value[index + 1]
            + prefix_value[index] * suffix_dot[index + 1]
        )
        gradient[index] = float(factor.first) * excluded_value
        product[index] = (
            float(factor.second) * float(direction[index]) * excluded_value
            + float(factor.first) * excluded_dot
        )
    return float(prefix_value[-1]), gradient, product


def _factor_to_dict(factor: BondOrderRadialFactor) -> dict[str, float]:
    return {
        "reference_distance_angstrom": float(factor.reference_distance),
        "covalent_radius_sum_angstrom": float(factor.covalent_radius_sum),
        "strong_decay": float(factor.strong_decay),
        "weak_decay": float(factor.weak_decay),
        "switch_sharpness": float(factor.switch_sharpness),
    }


def _factor_from_dict(payload: Mapping[str, Any]) -> BondOrderRadialFactor:
    return BondOrderRadialFactor(
        reference_distance=float(payload["reference_distance_angstrom"]),
        covalent_radius_sum=float(payload["covalent_radius_sum_angstrom"]),
        strong_decay=float(payload["strong_decay"]),
        weak_decay=float(payload["weak_decay"]),
        switch_sharpness=float(payload["switch_sharpness"]),
    )


__all__ = [
    "ZAFF_FLAP_RING_SCHEMA",
    "ZaffDissociativeFlapRingPotential",
    "balanced_flap_stencils",
]
