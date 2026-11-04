"""Backend-neutral SCF-state reuse owned by APOC.

APOC owns the accepted-state lifecycle and the representation-independent
density, Fock, orbital, and overlap contracts. Backend adapters provide only
the representation-specific import, projection, and export operations.
Finite-difference branches are intentionally independent and never update the
parent history with an intermediate displaced point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Protocol

import numpy as np


SCF_STATE_SCHEMA = "matrix.apoc.scf_state.v1"
SCF_GUESS_SCHEMA = "matrix.apoc.scf_guess.v1"
_LEGACY_SCF_STATE_SCHEMA = "matrix.link.scf_state.v1"


class SCFStateContractError(ValueError):
    """An SCF state cannot be used for the requested electronic problem."""


@dataclass(frozen=True)
class SCFStateCapabilities:
    density_projection: bool = False
    density_extrapolation: bool = False
    fock_projection: bool = False
    fock_extrapolation: bool = False
    orbital_restart: bool = False
    in_process: bool = False
    maximum_history: int = 4

    def __post_init__(self) -> None:
        if self.maximum_history < 1:
            raise ValueError("SCF state history must retain at least one state")


@dataclass(frozen=True)
class SCFBackendRequest:
    backend: str
    method: str
    basis: str
    charge: int
    multiplicity: int
    electronic_state: int
    coordinates: np.ndarray
    electron_count: float | None = None
    spin: str = ""
    symmetry: str = ""
    orbital_order: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("SCF request coordinates must have shape natoms x 3")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("SCF request coordinates contain non-finite values")
        if self.charge != int(self.charge) or self.multiplicity < 1:
            raise ValueError("invalid SCF charge or multiplicity")
        object.__setattr__(self, "coordinates", coordinates.copy())
        object.__setattr__(self, "backend", str(self.backend).strip())
        object.__setattr__(self, "method", str(self.method).strip())
        object.__setattr__(self, "basis", str(self.basis).strip())
        object.__setattr__(self, "spin", str(self.spin).strip())
        object.__setattr__(self, "symmetry", str(self.symmetry).strip())
        object.__setattr__(self, "orbital_order", tuple(int(i) for i in self.orbital_order))


@dataclass(frozen=True)
class SCFState:
    backend: str
    method: str
    basis: str
    charge: int
    multiplicity: int
    electronic_state: int
    coordinates: np.ndarray
    density_alpha: np.ndarray | None = None
    density_beta: np.ndarray | None = None
    fock_alpha: np.ndarray | None = None
    fock_beta: np.ndarray | None = None
    orbitals: np.ndarray | None = None
    occupations: np.ndarray | None = None
    overlap: np.ndarray | None = None
    electron_count: float | None = None
    spin: str = ""
    symmetry: str = ""
    orbital_order: tuple[int, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SCF_STATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema not in {SCF_STATE_SCHEMA, _LEGACY_SCF_STATE_SCHEMA}:
            raise SCFStateContractError("unsupported SCF state schema")
        coordinates = np.asarray(self.coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise SCFStateContractError("SCF state coordinates must have shape natoms x 3")
        if not np.all(np.isfinite(coordinates)):
            raise SCFStateContractError("SCF state coordinates are non-finite")
        object.__setattr__(self, "coordinates", coordinates.copy())
        for name in (
            "density_alpha",
            "density_beta",
            "fock_alpha",
            "fock_beta",
            "orbitals",
            "occupations",
            "overlap",
        ):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=float)
                if not np.all(np.isfinite(array)):
                    raise SCFStateContractError(f"SCF state {name} contains non-finite values")
                object.__setattr__(self, name, array.copy())
        object.__setattr__(self, "backend", str(self.backend).strip())
        object.__setattr__(self, "method", str(self.method).strip())
        object.__setattr__(self, "basis", str(self.basis).strip())
        object.__setattr__(self, "spin", str(self.spin).strip())
        object.__setattr__(self, "symmetry", str(self.symmetry).strip())
        object.__setattr__(self, "orbital_order", tuple(int(i) for i in self.orbital_order))
        validate_scf_state(self)

    def compatibility_key(self) -> tuple[Any, ...]:
        return (
            self.backend,
            self.method,
            self.basis,
            self.charge,
            self.multiplicity,
            self.electronic_state,
            self.spin,
            self.symmetry,
            self.orbital_order,
        )


@dataclass(frozen=True)
class SCFGuess:
    density_alpha: np.ndarray | None = None
    density_beta: np.ndarray | None = None
    fock_alpha: np.ndarray | None = None
    fock_beta: np.ndarray | None = None
    orbitals: np.ndarray | None = None
    occupations: np.ndarray | None = None
    status: str = "standard_guess"
    reason: str = ""
    source_state_count: int = 0
    restart_artifact: str | None = None
    native_projection: bool = False
    schema: str = SCF_GUESS_SCHEMA


class SCFStateAdapter(Protocol):
    """Required adapter contract for backend SCF-state reuse."""

    def capabilities(self) -> SCFStateCapabilities: ...

    def export_state(self, result: Any) -> SCFState: ...

    def project_state(
        self, state: SCFState, target_geometry: np.ndarray, target_basis: str
    ) -> SCFGuess: ...

    def extrapolate_state(
        self, history: Sequence[SCFState], target_geometry: np.ndarray, target_basis: str
    ) -> SCFGuess: ...

    def evaluate(self, request: SCFBackendRequest, guess: SCFGuess | None = None) -> Any: ...


def project_scf_state(
    state: SCFState,
    target_geometry: np.ndarray,
    target_basis: str,
    basis_transform: Callable[[SCFState, np.ndarray, str], np.ndarray],
) -> SCFGuess:
    """Project an SCF state with a backend-supplied AO basis transformation.

    ``basis_transform`` maps source AO coefficients to target AO coefficients.
    No rounding or text serialization is performed.  A non-orthogonal
    transform is accepted, but the resulting density is checked before use.
    """
    transform = np.asarray(basis_transform(state, target_geometry, target_basis), dtype=float)
    if (
        transform.ndim != 2
        or state.density_alpha is not None
        and transform.shape[1] != state.density_alpha.shape[0]
    ):
        raise SCFStateContractError("SCF basis transformation has incompatible dimensions")

    def transform_density(matrix: np.ndarray | None) -> np.ndarray | None:
        if matrix is None:
            return None
        return 0.5 * (transform @ matrix @ transform.T + transform @ matrix.T @ transform.T)

    def transform_operator(matrix: np.ndarray | None) -> np.ndarray | None:
        if matrix is None:
            return None
        inverse = np.linalg.pinv(transform, rcond=1.0e-11)
        value = inverse.T @ matrix @ inverse
        return 0.5 * (value + value.T)

    density_alpha = transform_density(state.density_alpha)
    density_beta = transform_density(state.density_beta)
    for label, density in (("alpha", density_alpha), ("beta", density_beta)):
        if density is not None and np.min(np.linalg.eigvalsh(density)) < -1.0e-8:
            raise SCFStateContractError(f"projected {label} density is not positive semidefinite")
    return SCFGuess(
        density_alpha=density_alpha,
        density_beta=density_beta,
        # A density is contravariant under an AO coefficient map, whereas an
        # effective Hamiltonian/Fock operator is covariant.  Applying the
        # density congruence to both silently changes <C|F|C>.
        fock_alpha=transform_operator(state.fock_alpha),
        fock_beta=transform_operator(state.fock_beta),
        orbitals=(None if state.orbitals is None else transform @ state.orbitals),
        occupations=state.occupations,
        status="projected",
        reason="backend_basis_projection",
    )


def extrapolate_scf_state(
    history: Sequence[SCFState],
    target_geometry: np.ndarray,
    target_basis: str,
    project: Callable[[SCFState, np.ndarray, str], SCFGuess],
) -> SCFGuess:
    """Extrapolate two projected states to a target geometry without chaining FD points."""
    if len(history) < 2:
        raise SCFStateContractError("SCF extrapolation requires two converged states")
    latest = history[-1]
    q1 = np.asarray(latest.coordinates, dtype=float)
    qt = np.asarray(target_geometry, dtype=float)
    direction = np.asarray(latest.coordinates, dtype=float) - np.asarray(
        history[-2].coordinates, dtype=float
    )
    step = float(np.linalg.norm(direction))
    if step <= 1.0e-14:
        raise SCFStateContractError("SCF extrapolation needs distinct parent geometries")
    direction /= step
    parameters = np.asarray(
        [
            float(np.sum((np.asarray(item.coordinates, dtype=float) - q1) * direction))
            for item in history
        ]
    )
    target_parameter = float(np.sum((qt - q1) * direction))
    if np.unique(parameters).size < 2:
        raise SCFStateContractError("SCF extrapolation needs distinct parent geometries")
    projected = [project(item, target_geometry, target_basis) for item in history]
    design = np.column_stack((np.ones(len(parameters)), parameters))
    target_vector = np.asarray((1.0, target_parameter), dtype=float)
    # One set of linear weights is shared by density and Hamiltonian blocks.
    # Solving it once avoids four identical least-squares factorizations and
    # guarantees that the coupled matrices follow exactly the same predictor.
    weights = np.linalg.lstsq(design.T, target_vector, rcond=None)[0]

    def extrapolate(values: Sequence[np.ndarray | None]) -> np.ndarray | None:
        if any(value is None for value in values):
            return values[-1]
        arrays = [np.asarray(value, dtype=float) for value in values]
        shape = arrays[-1].shape
        observations = np.asarray(arrays).reshape(len(arrays), -1)
        value = (weights @ observations).reshape(shape)
        return 0.5 * (value + value.T)

    density_alpha = extrapolate([item.density_alpha for item in projected])
    density_beta = extrapolate([item.density_beta for item in projected])
    for label, density in (("alpha", density_alpha), ("beta", density_beta)):
        if density is not None and np.min(np.linalg.eigvalsh(density)) < -1.0e-8:
            raise SCFStateContractError(
                f"extrapolated {label} density is not positive semidefinite"
            )
    return SCFGuess(
        density_alpha=density_alpha,
        density_beta=density_beta,
        fock_alpha=extrapolate([item.fock_alpha for item in projected]),
        fock_beta=extrapolate([item.fock_beta for item in projected]),
        orbitals=projected[-1].orbitals,
        occupations=projected[-1].occupations,
        status="extrapolated",
        reason="backend_basis_projection_and_geometry_extrapolation",
    )


def reuse_same_basis_state(
    state: SCFState, target_geometry: np.ndarray, target_basis: str
) -> SCFGuess:
    """Reuse matrices directly when AO basis identity and ordering are unchanged."""
    return SCFGuess(
        density_alpha=None
        if state.density_alpha is None
        else np.array(state.density_alpha, copy=True),
        density_beta=None
        if state.density_beta is None
        else np.array(state.density_beta, copy=True),
        fock_alpha=None if state.fock_alpha is None else np.array(state.fock_alpha, copy=True),
        fock_beta=None if state.fock_beta is None else np.array(state.fock_beta, copy=True),
        orbitals=None if state.orbitals is None else np.array(state.orbitals, copy=True),
        occupations=None if state.occupations is None else np.array(state.occupations, copy=True),
        status="same_basis_reuse",
        reason="unchanged_basis_and_orbital_order",
    )


def extrapolate_same_basis_state(
    history: Sequence[SCFState], target_geometry: np.ndarray, target_basis: str
) -> SCFGuess:
    """Extrapolate SCF matrices directly in a fixed AO representation."""
    if len(history) < 2:
        raise SCFStateContractError("same-basis extrapolation requires two converged states")
    return extrapolate_scf_state(
        history,
        target_geometry,
        target_basis,
        lambda state, geometry, basis: reuse_same_basis_state(state, geometry, basis),
    )


def validate_scf_state(state: SCFState) -> None:
    """Check AO dimensions, spin channels, Hermiticity, PSD, and electron count."""
    matrices = {
        name: getattr(state, name)
        for name in ("density_alpha", "density_beta", "fock_alpha", "fock_beta", "overlap")
        if getattr(state, name) is not None
    }
    dimensions = set()
    for name, matrix in matrices.items():
        array = np.asarray(matrix, dtype=float)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            if name != "overlap":
                raise SCFStateContractError(f"SCF state {name} must be a square matrix")
        dimensions.add(array.shape)
        if name.startswith(("density_", "fock_", "overlap")) and array.ndim == 2:
            if not np.allclose(array, array.T, rtol=0.0, atol=1.0e-12):
                raise SCFStateContractError(f"SCF state {name} is not symmetric")
        if name.startswith("density_"):
            eigenvalues = np.linalg.eigvalsh(0.5 * (array + array.T))
            if float(np.min(eigenvalues)) < -1.0e-9:
                raise SCFStateContractError(f"SCF state {name} is not positive semidefinite")
    if len(dimensions) > 1:
        raise SCFStateContractError("SCF state matrix dimensions are inconsistent")
    if state.density_beta is not None and state.density_alpha is None:
        raise SCFStateContractError("beta density requires an alpha density channel")
    if state.fock_beta is not None and state.fock_alpha is None:
        raise SCFStateContractError("beta Fock matrix requires an alpha Fock channel")
    if state.density_beta is not None and (
        (state.fock_alpha is None) != (state.fock_beta is None)
    ):
        raise SCFStateContractError("unrestricted density and Fock channels are inconsistent")
    if (
        state.electron_count is not None
        and state.overlap is not None
        and state.density_alpha is not None
    ):
        overlap = np.asarray(state.overlap, dtype=float)
        measured = float(np.trace(np.asarray(state.density_alpha) @ overlap))
        if state.density_beta is not None:
            measured += float(np.trace(np.asarray(state.density_beta) @ overlap))
        if not np.isclose(
            measured,
            float(state.electron_count),
            rtol=0.0,
            atol=2.0e-6,
        ):
            raise SCFStateContractError(
                "SCF density and AO overlap do not preserve the declared electron count"
            )
    if state.orbitals is not None:
        orbitals = np.asarray(state.orbitals)
        if orbitals.ndim != 2 or dimensions and orbitals.shape[0] != next(iter(dimensions))[0]:
            raise SCFStateContractError("SCF state orbitals have incompatible dimensions")
        if state.orbital_order and (
            len(state.orbital_order) != orbitals.shape[1]
            or len(set(state.orbital_order)) != len(state.orbital_order)
        ):
            raise SCFStateContractError("SCF state orbital ordering is invalid")
    if state.occupations is not None:
        occupations = np.asarray(state.occupations)
        if occupations.ndim not in (1, 2):
            raise SCFStateContractError("SCF state occupations must be one- or two-dimensional")
        if occupations.ndim == 2 and occupations.shape[0] not in {1, 2}:
            raise SCFStateContractError("SCF occupations have an invalid spin-channel count")
        expected_channels = 2 if state.density_beta is not None else 1
        actual_channels = 1 if occupations.ndim == 1 else occupations.shape[0]
        if actual_channels != expected_channels:
            raise SCFStateContractError("SCF occupations and density spin channels differ")


def validate_state_for_request(
    state: SCFState, request: SCFBackendRequest, *, allow_basis_change: bool = False
) -> None:
    """Reject method, basis, charge, spin, or state mismatches before projection."""
    expected = (
        request.backend,
        request.method,
        request.basis,
        request.charge,
        request.multiplicity,
        request.electronic_state,
        request.spin,
        request.symmetry,
        request.orbital_order,
    )
    actual = state.compatibility_key()
    if allow_basis_change:
        actual = (actual[0], actual[1], expected[2], *actual[3:])
    if actual != expected:
        raise SCFStateContractError("SCF state is incompatible with backend request")
    if request.electron_count is not None and state.electron_count is not None:
        if not np.isclose(state.electron_count, request.electron_count, rtol=0.0, atol=1.0e-10):
            raise SCFStateContractError("SCF state electron count differs from backend request")


class SCFStateManager:
    """Maintain independent converged-state histories for client branches."""

    def __init__(self, adapter: SCFStateAdapter, *, maximum_history: int | None = None) -> None:
        self.adapter = adapter
        advertised = adapter.capabilities()
        self.maximum_history = int(maximum_history or advertised.maximum_history)
        if self.maximum_history < 1:
            raise ValueError("SCF state history must retain at least one state")
        self._histories: dict[str, list[SCFState]] = {}
        self.events: list[dict[str, Any]] = []

    def prepare_guess(self, request: SCFBackendRequest, *, branch: str = "root") -> SCFGuess:
        history = self._histories.get(str(branch), [])
        if not history:
            guess = SCFGuess(reason="no_converged_parent_state")
        else:
            try:
                caps = self.adapter.capabilities()
                for state in history:
                    validate_state_for_request(
                        state,
                        request,
                        allow_basis_change=(
                            state.basis != request.basis and caps.density_projection
                        ),
                    )
                if len(history) >= 2 and (caps.density_extrapolation or caps.fock_extrapolation):
                    guess = self.adapter.extrapolate_state(
                        tuple(history), request.coordinates, request.basis
                    )
                elif caps.density_projection or caps.fock_projection or caps.orbital_restart:
                    guess = self.adapter.project_state(
                        history[-1], request.coordinates, request.basis
                    )
                else:
                    raise SCFStateContractError("adapter does not advertise state reuse")
                guess = _tag_guess(guess, len(history))
            except (SCFStateContractError, ValueError, RuntimeError) as exc:
                reason = f"state_reuse_rejected:{type(exc).__name__}:{exc}"
                self.events.append(
                    {"event": "standard_guess_fallback", "branch": str(branch), "reason": reason}
                )
                guess = SCFGuess(
                    status="standard_guess_fallback", reason=reason, source_state_count=len(history)
                )
        self.events.append(
            {
                "event": "guess_prepared",
                "branch": str(branch),
                "status": guess.status,
                "source_state_count": guess.source_state_count,
                "reason": guess.reason,
            }
        )
        return guess

    def record_converged(self, result: Any, *, branch: str = "root") -> SCFState:
        state = self.adapter.export_state(result)
        history = self._histories.setdefault(str(branch), [])
        history.append(state)
        del history[: -self.maximum_history]
        self.events.append(
            {"event": "state_recorded", "branch": str(branch), "history_size": len(history)}
        )
        return state

    def seed_history(self, states: Sequence[SCFState], *, branch: str = "root") -> None:
        """Seed an independent branch from already validated converged states."""
        cloned = [_clone_state(state) for state in states]
        self._histories[str(branch)] = cloned[-self.maximum_history :]
        self.events.append(
            {
                "event": "history_seeded",
                "branch": str(branch),
                "history_size": len(self._histories[str(branch)]),
            }
        )

    def evaluate_and_record(
        self, request: SCFBackendRequest, *, branch: str = "root"
    ) -> tuple[Any, SCFGuess, SCFState]:
        """Evaluate one point with the reusable guess and record only its result."""
        guess = self.prepare_guess(request, branch=branch)
        result = self.adapter.evaluate(request, guess=guess)
        state = self.record_converged(result, branch=branch)
        return result, guess, state

    def fork(self, parent: str = "root", child: str | None = None) -> str:
        """Create a branch sharing the parent snapshot but not its future updates."""
        child_name = str(child or f"{parent}:branch-{len(self._histories)}")
        self._histories[child_name] = [
            _clone_state(state) for state in self._histories.get(str(parent), ())
        ]
        self.events.append({"event": "branch_created", "parent": str(parent), "branch": child_name})
        return child_name


def _clone_state(state: SCFState) -> SCFState:
    """Copy arrays so a displaced finite-difference branch cannot mutate its parent."""
    values = {
        name: getattr(state, name)
        for name in (
            "density_alpha",
            "density_beta",
            "fock_alpha",
            "fock_beta",
            "orbitals",
            "occupations",
            "overlap",
        )
    }
    values = {
        name: None if value is None else np.array(value, copy=True)
        for name, value in values.items()
    }
    return SCFState(
        backend=state.backend,
        method=state.method,
        basis=state.basis,
        charge=state.charge,
        multiplicity=state.multiplicity,
        electronic_state=state.electronic_state,
        coordinates=np.array(state.coordinates, copy=True),
        electron_count=state.electron_count,
        spin=state.spin,
        symmetry=state.symmetry,
        orbital_order=state.orbital_order,
        provenance=dict(state.provenance),
        **values,
    )


def _tag_guess(guess: SCFGuess, source_count: int) -> SCFGuess:
    return SCFGuess(
        density_alpha=guess.density_alpha,
        density_beta=guess.density_beta,
        fock_alpha=guess.fock_alpha,
        fock_beta=guess.fock_beta,
        orbitals=guess.orbitals,
        occupations=guess.occupations,
        status=guess.status,
        reason=guess.reason,
        source_state_count=source_count,
        restart_artifact=guess.restart_artifact,
        native_projection=guess.native_projection,
    )


class APOCStateHistory:
    """Own the accepted electronic-state history used by geometry drivers."""

    def __init__(self, *, maximum_history: int = 4) -> None:
        if int(maximum_history) < 1:
            raise ValueError("APOC state history length must be positive")
        self.maximum_history = int(maximum_history)
        self._states: list[SCFState] = []

    def record(self, state: SCFState) -> None:
        validate_scf_state(state)
        if self._states and self._states[-1] is state:
            return
        self._states.append(state)
        del self._states[: -self.maximum_history]

    def snapshot(self) -> tuple[SCFState, ...]:
        return tuple(self._states)

    def clear(self) -> None:
        self._states.clear()


__all__ = [
    "SCF_STATE_SCHEMA",
    "SCF_GUESS_SCHEMA",
    "SCFBackendRequest",
    "SCFGuess",
    "SCFState",
    "SCFStateAdapter",
    "SCFStateCapabilities",
    "SCFStateContractError",
    "SCFStateManager",
    "APOCStateHistory",
    "validate_scf_state",
    "validate_state_for_request",
    "project_scf_state",
    "extrapolate_scf_state",
    "reuse_same_basis_state",
    "extrapolate_same_basis_state",
]
