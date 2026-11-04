"""Dependency-light contract for resident and external potential backends.

The contract deliberately separates energy, energy-plus-gradient and
energy-plus-gradient-plus-Hessian execution.  A backend must never obtain a
lower-order result by silently running a more expensive higher-order path, nor
manufacture an unavailable derivative numerically.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np


POTENTIAL_BACKEND_CONTRACT_SCHEMA = "matrix.potential_backend.contract.v1"
POTENTIAL_EVALUATION_SCHEMA = "matrix.potential_backend.evaluation.v1"


class DerivativeOrder(IntEnum):
    """Highest Cartesian derivative requested from a potential backend."""

    ENERGY = 0
    GRADIENT = 1
    HESSIAN = 2


@dataclass(frozen=True)
class PotentialSystem:
    """Geometry-independent molecular identity supplied when preparing a backend."""

    atoms: tuple[str, ...]
    charge: int = 0
    multiplicity: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        atoms = tuple(str(atom).strip() for atom in self.atoms)
        if not atoms or any(not atom for atom in atoms):
            raise ValueError("potential system atoms must be nonempty")
        if int(self.multiplicity) <= 0:
            raise ValueError("potential system multiplicity must be positive")
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "charge", int(self.charge))
        object.__setattr__(self, "multiplicity", int(self.multiplicity))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata or {})),
        )

    @property
    def natoms(self) -> int:
        return len(self.atoms)


@dataclass(frozen=True)
class PotentialCapabilities:
    """Auditable capabilities of one prepared potential session."""

    maximum_derivative_order: DerivativeOrder
    batch: bool = False
    resident: bool = False
    persistent_neighbor_list: bool = False
    persistent_fmm: bool = False
    polarization: bool = False
    reaction_field: bool = False
    thread_safe: bool = False
    devices: tuple[str, ...] = ("cpu",)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = POTENTIAL_BACKEND_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        order = DerivativeOrder(self.maximum_derivative_order)
        devices = tuple(dict.fromkeys(str(item).strip().lower() for item in self.devices))
        if not devices or any(not item for item in devices):
            raise ValueError("potential backend must declare at least one device")
        if self.schema != POTENTIAL_BACKEND_CONTRACT_SCHEMA:
            raise ValueError(f"unsupported potential capability schema: {self.schema}")
        object.__setattr__(self, "maximum_derivative_order", order)
        object.__setattr__(self, "devices", devices)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata or {})),
        )

    def supports(self, derivative_order: DerivativeOrder | int) -> bool:
        return DerivativeOrder(derivative_order) <= self.maximum_derivative_order


@dataclass(frozen=True)
class PotentialEvaluation:
    """Canonical atomic-unit result returned by every potential backend."""

    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray | None = None
    hessian_hartree_per_bohr2: np.ndarray | None = None
    derivative_order: DerivativeOrder = DerivativeOrder.ENERGY
    backend: str = ""
    model: str = ""
    execution: Mapping[str, Any] = field(default_factory=dict)
    schema: str = POTENTIAL_EVALUATION_SCHEMA

    def __post_init__(self) -> None:
        order = DerivativeOrder(self.derivative_order)
        energy = float(self.energy_hartree)
        gradient = (
            None
            if self.gradient_hartree_per_bohr is None
            else np.asarray(self.gradient_hartree_per_bohr, dtype=float).reshape(-1)
        )
        hessian = (
            None
            if self.hessian_hartree_per_bohr2 is None
            else np.asarray(self.hessian_hartree_per_bohr2, dtype=float)
        )
        if not np.isfinite(energy):
            raise ValueError("potential energy must be finite")
        if order >= DerivativeOrder.GRADIENT:
            if gradient is None or np.any(~np.isfinite(gradient)):
                raise ValueError("gradient path must return a finite Cartesian gradient")
        elif gradient is not None:
            raise ValueError("energy-only path must not return a gradient")
        if order >= DerivativeOrder.HESSIAN:
            if (
                gradient is None
                or hessian is None
                or hessian.shape != (len(gradient), len(gradient))
                or np.any(~np.isfinite(hessian))
            ):
                raise ValueError("Hessian path must return a finite square Cartesian Hessian")
            if not np.allclose(hessian, hessian.T, atol=1.0e-11, rtol=1.0e-9):
                raise ValueError("Cartesian Hessian must be symmetric")
        elif hessian is not None:
            raise ValueError("non-Hessian path must not return a Hessian")
        if self.schema != POTENTIAL_EVALUATION_SCHEMA:
            raise ValueError(f"unsupported potential evaluation schema: {self.schema}")
        object.__setattr__(self, "energy_hartree", energy)
        object.__setattr__(self, "gradient_hartree_per_bohr", gradient)
        object.__setattr__(
            self,
            "hessian_hartree_per_bohr2",
            None if hessian is None else 0.5 * (hessian + hessian.T),
        )
        object.__setattr__(self, "derivative_order", order)
        object.__setattr__(self, "backend", str(self.backend).strip())
        object.__setattr__(self, "model", str(self.model).strip())
        object.__setattr__(
            self,
            "execution",
            MappingProxyType(dict(self.execution or {})),
        )

    def validate_for_system(self, system: PotentialSystem) -> "PotentialEvaluation":
        dimension = 3 * system.natoms
        if (
            self.gradient_hartree_per_bohr is not None
            and self.gradient_hartree_per_bohr.shape != (dimension,)
        ):
            raise ValueError("potential gradient dimension does not match the system")
        if (
            self.hessian_hartree_per_bohr2 is not None
            and self.hessian_hartree_per_bohr2.shape != (dimension, dimension)
        ):
            raise ValueError("potential Hessian dimension does not match the system")
        return self


@runtime_checkable
class PotentialSession(Protocol):
    """Prepared, reusable potential with three explicit derivative paths."""

    @property
    def backend_name(self) -> str: ...

    @property
    def model_identifier(self) -> str: ...

    @property
    def system(self) -> PotentialSystem: ...

    @property
    def capabilities(self) -> PotentialCapabilities: ...

    def energy(self, coordinates_angstrom: np.ndarray) -> PotentialEvaluation: ...

    def energy_gradient(self, coordinates_angstrom: np.ndarray) -> PotentialEvaluation: ...

    def energy_gradient_hessian(
        self,
        coordinates_angstrom: np.ndarray,
    ) -> PotentialEvaluation: ...

    def evaluate_batch(
        self,
        geometries_angstrom: Sequence[np.ndarray],
        *,
        derivative_order: DerivativeOrder = DerivativeOrder.ENERGY,
    ) -> tuple[PotentialEvaluation, ...]: ...


@runtime_checkable
class PotentialBackend(Protocol):
    """Factory that prepares a reusable potential session."""

    @property
    def name(self) -> str: ...

    def prepare(
        self,
        system: PotentialSystem,
        *,
        model: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> PotentialSession: ...


PotentialBackendFactory = Callable[[], PotentialBackend]


class PotentialBackendRegistry:
    """Small lazy registry shared by resident and external backends."""

    def __init__(self) -> None:
        self._factories: dict[str, PotentialBackendFactory] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        name: str,
        factory: PotentialBackendFactory,
        *,
        aliases: Sequence[str] = (),
    ) -> None:
        canonical = _backend_key(name)
        keys = (canonical, *(_backend_key(alias) for alias in aliases))
        if canonical in self._factories or any(key in self._aliases for key in keys):
            raise ValueError(f"potential backend name is already registered: {name}")
        self._factories[canonical] = factory
        for key in keys:
            if key != canonical:
                if key in self._factories or key in self._aliases:
                    raise ValueError(f"potential backend alias is already registered: {key}")
                self._aliases[key] = canonical

    def canonical_name(self, name: str) -> str:
        key = _backend_key(name)
        canonical = self._aliases.get(key, key)
        if canonical not in self._factories:
            raise KeyError(f"unknown potential backend: {name}")
        return canonical

    def create(self, name: str) -> PotentialBackend:
        canonical = self.canonical_name(name)
        backend = self._factories[canonical]()
        if _backend_key(backend.name) != canonical:
            raise ValueError("potential backend factory returned a different canonical name")
        return backend

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def evaluate_session(
    session: PotentialSession,
    coordinates_angstrom: np.ndarray,
    *,
    derivative_order: DerivativeOrder | int,
) -> PotentialEvaluation:
    """Dispatch to the exact requested path without promoting derivative order."""

    order = DerivativeOrder(derivative_order)
    if not session.capabilities.supports(order):
        raise ValueError(
            f"backend {session.backend_name!r} does not support derivative order {int(order)}"
        )
    coordinates = _validated_geometry(coordinates_angstrom, session.system)
    if order is DerivativeOrder.ENERGY:
        result = session.energy(coordinates)
    elif order is DerivativeOrder.GRADIENT:
        result = session.energy_gradient(coordinates)
    else:
        result = session.energy_gradient_hessian(coordinates)
    if result.derivative_order is not order:
        raise ValueError("potential backend returned a different derivative order")
    return result.validate_for_system(session.system)


def _validated_geometry(
    coordinates_angstrom: np.ndarray,
    system: PotentialSystem,
) -> np.ndarray:
    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.shape != (system.natoms, 3):
        raise ValueError("potential geometry must have shape (natoms, 3)")
    if np.any(~np.isfinite(coordinates)):
        raise ValueError("potential geometry contains non-finite values")
    return coordinates


def _backend_key(value: str) -> str:
    key = str(value).strip().lower().replace("_", "-")
    if not key:
        raise ValueError("potential backend name must be nonempty")
    return key


__all__ = [
    "DerivativeOrder",
    "POTENTIAL_BACKEND_CONTRACT_SCHEMA",
    "POTENTIAL_EVALUATION_SCHEMA",
    "PotentialBackend",
    "PotentialBackendRegistry",
    "PotentialCapabilities",
    "PotentialEvaluation",
    "PotentialSession",
    "PotentialSystem",
    "evaluate_session",
]
