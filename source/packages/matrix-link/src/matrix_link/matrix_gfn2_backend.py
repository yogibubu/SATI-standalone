"""Resident LINK adapter for the autonomous MATRIX-GFN2 calculator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


SUPPORTED_SOLVERS = frozenset({"dense", "linear", "auto"})
SF_TDA_TRACKING_CHECKPOINT_SCHEMA = "matrix.link.sf_tda_tracking_checkpoint.v1"


@dataclass(frozen=True)
class MatrixGFN2Evaluation:
    """One immutable LINK-facing MATRIX-GFN2 evaluation."""

    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    charges: np.ndarray | None = None
    bond_orders: np.ndarray | None = None
    cache_hit: bool = False
    telemetry: object | None = None
    dipole_au: np.ndarray | None = None
    hessian_hartree_per_bohr2: np.ndarray | None = None


@dataclass(frozen=True)
class MatrixGFN2SFTDAEvaluation:
    """One complete resident SF-TDA manifold for LINK/APOC."""

    target_state_id: int
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    ground_energy_hartree: float
    excitation_energies_hartree: np.ndarray
    transition_density_vectors: np.ndarray
    difference_density_vectors: np.ndarray
    cache_hit: bool = False

    def __post_init__(self) -> None:
        excitation = np.array(self.excitation_energies_hartree, dtype=float, copy=True)
        transition = np.array(self.transition_density_vectors, dtype=float, copy=True)
        difference = np.array(self.difference_density_vectors, dtype=float, copy=True)
        gradient = np.array(self.gradient_hartree_per_bohr, dtype=float, copy=True)
        if excitation.ndim != 1 or transition.ndim != 2:
            raise ValueError("resident SF-TDA manifold has inconsistent dimensions")
        if transition.shape[0] != len(excitation) or difference.shape != transition.shape:
            raise ValueError("resident SF-TDA state fingerprints are inconsistent")
        if not 1 <= int(self.target_state_id) <= len(excitation):
            raise ValueError("resident SF-TDA target state lies outside the manifold")
        if not all(
            np.all(np.isfinite(item))
            for item in (excitation, transition, difference, gradient)
        ):
            raise ValueError("resident SF-TDA evaluation must be finite")
        for item in (excitation, transition, difference, gradient):
            item.setflags(write=False)
        object.__setattr__(self, "excitation_energies_hartree", excitation)
        object.__setattr__(self, "transition_density_vectors", transition)
        object.__setattr__(self, "difference_density_vectors", difference)
        object.__setattr__(self, "gradient_hartree_per_bohr", gradient)


@dataclass(frozen=True)
class MatrixGFN2SFTDACouplingEvaluation:
    """Two resident SF-TDA surfaces and their analytic interstate vector."""

    state_ids: tuple[int, int]
    energies_hartree: np.ndarray
    gradients_hartree_per_bohr: np.ndarray
    interstate_numerator_hartree_per_bohr: np.ndarray
    derivative_coupling_per_bohr: np.ndarray | None
    energy_gap_hartree: float
    excitation_energies_hartree: np.ndarray
    transition_density_vectors: np.ndarray
    subspace_match: object | None = None
    cache_hit: bool = False

    def __post_init__(self) -> None:
        energies = np.array(self.energies_hartree, dtype=float, copy=True)
        gradients = np.array(self.gradients_hartree_per_bohr, dtype=float, copy=True)
        numerator = np.array(
            self.interstate_numerator_hartree_per_bohr, dtype=float, copy=True
        )
        excitation = np.array(self.excitation_energies_hartree, dtype=float, copy=True)
        fingerprints = np.array(self.transition_density_vectors, dtype=float, copy=True)
        coupling = self.derivative_coupling_per_bohr
        if len(self.state_ids) != 2 or self.state_ids[0] == self.state_ids[1]:
            raise ValueError("resident SF-TDA coupling requires two distinct states")
        if energies.shape != (2,) or gradients.ndim != 2 or gradients.shape[0] != 2:
            raise ValueError("resident SF-TDA coupling surfaces are inconsistent")
        if numerator.shape != gradients.shape[1:]:
            raise ValueError("resident SF-TDA interstate vector has the wrong shape")
        if fingerprints.ndim != 2 or fingerprints.shape[0] != len(excitation):
            raise ValueError("resident SF-TDA coupling fingerprints are inconsistent")
        if coupling is not None:
            coupling = np.array(coupling, dtype=float, copy=True)
            if coupling.shape != numerator.shape or not np.all(np.isfinite(coupling)):
                raise ValueError("resident SF-TDA derivative coupling is invalid")
            coupling.setflags(write=False)
        if not all(
            np.all(np.isfinite(value))
            for value in (energies, gradients, numerator, excitation, fingerprints)
        ):
            raise ValueError("resident SF-TDA coupling evaluation must be finite")
        for value in (energies, gradients, numerator, excitation, fingerprints):
            value.setflags(write=False)
        object.__setattr__(self, "energies_hartree", energies)
        object.__setattr__(self, "gradients_hartree_per_bohr", gradients)
        object.__setattr__(self, "interstate_numerator_hartree_per_bohr", numerator)
        object.__setattr__(self, "derivative_coupling_per_bohr", coupling)
        object.__setattr__(self, "excitation_energies_hartree", excitation)
        object.__setattr__(self, "transition_density_vectors", fingerprints)


@dataclass(frozen=True)
class MatrixGFN2TwoPhotonEvaluation:
    """LINK-facing TDA two-photon strength and analytic Cartesian gradient."""

    target_state_id: int
    strength_au: float
    strength_gradient_au_per_bohr: np.ndarray
    transition_tensor_au: np.ndarray
    maximum_response_residual: float
    cache_hit: bool = False

    def __post_init__(self) -> None:
        gradient = np.array(self.strength_gradient_au_per_bohr, dtype=float, copy=True)
        tensor = np.array(self.transition_tensor_au, copy=True)
        if gradient.ndim != 1 or tensor.shape != (3, 3):
            raise ValueError("resident TDA two-photon property dimensions are inconsistent")
        if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(tensor)):
            raise ValueError("resident TDA two-photon property must be finite")
        gradient.setflags(write=False)
        tensor.setflags(write=False)
        object.__setattr__(self, "strength_gradient_au_per_bohr", gradient)
        object.__setattr__(self, "transition_tensor_au", tensor)


@dataclass(frozen=True)
class MatrixGFN2Statistics:
    calculator_creations: int
    backend_evaluations: int
    cache_hits: int
    scc_retries: int
    scc_iterations: int


def matrix_gfn2_available() -> bool:
    """Return whether the qualified autonomous implementation is installed."""

    try:
        from matrix_gfn2 import qualification_status
    except ImportError:
        return False
    try:
        qualification_status().require_production_ready()
    except RuntimeError:
        return False
    return True


class MatrixGFN2Session:
    """Persistent, cache-aware MATRIX-GFN2 session for one molecular state."""

    MAX_PROCESSORS = 64

    def __init__(
        self,
        atoms: Sequence[str],
        *,
        charge: int = 0,
        multiplicity: int = 1,
        method: str = "MATRIX-GFN2",
        processors: int = 1,
        solver_mode: str = "auto",
        linear_scaling_threshold: int = 128,
        alpb_solvent: str | None = None,
        population_analysis: str = "initial",
        symmetry_contract: object | None = None,
        scf_energy_tolerance_hartree: float | None = None,
        scf_wavefunction_tolerance_electron: float = 1.0e-6,
        scf_mixer_tolerance: float = 2.0e-5,
    ) -> None:
        from matrix_gfn2 import GFN2Calculator, qualification_status

        qualification_status().require_production_ready()
        self.atoms = tuple(str(atom) for atom in atoms)
        if not self.atoms:
            raise ValueError("MATRIX-GFN2 requires at least one atom")
        normalized_method = str(method).strip().casefold().replace("-", "")
        if normalized_method not in {"", "matrixgfn2", "gfn2xtb", "gfn2"}:
            raise ValueError("the resident autonomous calculator supports MATRIX-GFN2 only")
        self.method = "MATRIX-GFN2"
        self.alpb_solvent = alpb_solvent
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        self.processors = int(processors)
        if not 1 <= self.processors <= self.MAX_PROCESSORS:
            raise ValueError(
                f"MATRIX-GFN2 processors must be between 1 and {self.MAX_PROCESSORS}"
            )
        self.effective_processors = self.processors
        normalized_solver = str(solver_mode).strip().casefold().replace("_", "-")
        if normalized_solver == "linear-scaling":
            normalized_solver = "linear"
        if normalized_solver not in SUPPORTED_SOLVERS:
            raise ValueError("MATRIX-GFN2 solver_mode must be dense, linear, or auto")
        self.solver_mode = normalized_solver
        self.linear_scaling_threshold = int(linear_scaling_threshold)
        if self.linear_scaling_threshold < 1:
            raise ValueError("MATRIX-GFN2 linear_scaling_threshold must be positive")
        population = str(population_analysis).strip().casefold().replace("_", "-")
        if population not in {"initial", "every-point", "bond-orders", "scc", "never"}:
            raise ValueError(
                "MATRIX-GFN2 population_analysis must be initial, every-point, bond-orders, scc, or never"
            )
        self.population_analysis = population
        density_solver = {
            "dense": "diagonalization",
            "linear": "auto",
            "auto": "auto",
        }[normalized_solver]
        sparse_threshold = (
            self.linear_scaling_threshold
            if normalized_solver == "linear"
            else max(4096, self.linear_scaling_threshold)
        )
        self._calculator = GFN2Calculator(
            self.atoms,
            charge=self.charge,
            multiplicity=self.multiplicity,
            maximum_scf_iterations=500,
            scf_energy_tolerance_hartree=scf_energy_tolerance_hartree,
            scf_wavefunction_tolerance_electron=scf_wavefunction_tolerance_electron,
            scf_mixer_tolerance=scf_mixer_tolerance,
            density_solver=density_solver,
            dense_sp2_min_orbitals=self.linear_scaling_threshold,
            sparse_sp2_min_orbitals=sparse_threshold,
            symmetry_contract=symmetry_contract,
            alpb_solvent=alpb_solvent,
        )
        self._population_analysis_pending = population in {"initial", "every-point", "bond-orders"}
        self._last_coordinates: np.ndarray | None = None
        self._last_evaluation: MatrixGFN2Evaluation | None = None
        self._last_sf_key: tuple[bytes, int, int] | None = None
        self._last_sf_evaluation: MatrixGFN2SFTDAEvaluation | None = None
        self._last_sf_coupling_key: tuple[bytes, int, int, int] | None = None
        self._last_sf_coupling: MatrixGFN2SFTDACouplingEvaluation | None = None
        self._last_sf_coupling_amplitudes: np.ndarray | None = None
        self._last_sf_subspace_reference: tuple[object, ...] | None = None
        self._last_two_photon_key: tuple[bytes, int, float] | None = None
        self._last_two_photon_evaluation: MatrixGFN2TwoPhotonEvaluation | None = None
        self._backend_evaluations = 0
        self._cache_hits = 0
        self._scc_iterations = 0

    @property
    def statistics(self) -> MatrixGFN2Statistics:
        return MatrixGFN2Statistics(
            calculator_creations=1,
            backend_evaluations=self._backend_evaluations,
            cache_hits=self._cache_hits,
            scc_retries=0,
            scc_iterations=self._scc_iterations,
        )

    def restart_from(self, other: "MatrixGFN2Session") -> None:
        if not isinstance(other, MatrixGFN2Session):
            raise TypeError("MATRIX-GFN2 restart source must be a MatrixGFN2Session")
        state = (
            self.atoms, self.charge, self.multiplicity,
            self.solver_mode, self.alpb_solvent,
        )
        other_state = (
            other.atoms,
            other.charge,
            other.multiplicity,
            other.solver_mode,
            other.alpb_solvent,
        )
        if state != other_state:
            raise ValueError("MATRIX-GFN2 restart sessions must use the same model and state")
        if (
            other._last_evaluation is None
            and other._last_sf_subspace_reference is None
        ):
            raise ValueError("source MATRIX-GFN2 session has no converged state")
        # The calculator currently restarts deterministically from its reference
        # density.  Do not misrepresent a cached geometry as an electronic restart.
        self._last_coordinates = None
        self._last_evaluation = None
        self._last_sf_coupling_amplitudes = self._copy(
            other._last_sf_coupling_amplitudes
        )
        self._last_sf_subspace_reference = other._last_sf_subspace_reference

    def write_sf_tda_tracking_checkpoint(self, path: Path | str) -> Path:
        """Persist only the APOC state-character data needed for a restart."""

        if (
            self._last_sf_subspace_reference is None
            or self._last_sf_coupling_amplitudes is None
        ):
            raise ValueError("no SF-TDA state-subspace tracking state is available")
        payload = {
            "schema": SF_TDA_TRACKING_CHECKPOINT_SCHEMA,
            "model": "SF-TDA-MATRIX-GFN2-R.v1",
            "atoms": list(self.atoms),
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "fingerprints": [
                {
                    "state_id": int(item.state_id),
                    "representation": str(item.representation),
                    "energy_hartree": item.energy_hartree,
                    "label": str(item.label),
                    "source": str(item.source),
                    "vector": np.asarray(item.vector, dtype=float).tolist(),
                }
                for item in self._last_sf_subspace_reference
            ],
            "amplitudes": np.asarray(
                self._last_sf_coupling_amplitudes, dtype=float
            ).tolist(),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        return target

    def load_sf_tda_tracking_checkpoint(self, path: Path | str) -> None:
        """Restore an APOC tracking subspace without reusing an SCF density."""

        from matrix_apoc import StateFingerprint

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != SF_TDA_TRACKING_CHECKPOINT_SCHEMA:
            raise ValueError("unsupported SF-TDA tracking checkpoint")
        identity = (
            tuple(payload.get("atoms", ())),
            int(payload.get("charge", 0)),
            int(payload.get("multiplicity", 0)),
        )
        if identity != (self.atoms, self.charge, self.multiplicity):
            raise ValueError("SF-TDA tracking checkpoint belongs to another molecular state")
        records = payload.get("fingerprints")
        if not isinstance(records, list) or len(records) < 2:
            raise ValueError("SF-TDA tracking checkpoint has no state subspace")
        fingerprints = tuple(
            StateFingerprint(
                int(item["state_id"]),
                np.asarray(item["vector"], dtype=float),
                str(item["representation"]),
                energy_hartree=item.get("energy_hartree"),
                label=str(item.get("label", "")),
                source=str(item.get("source", "")),
            )
            for item in records
        )
        amplitudes = np.asarray(payload.get("amplitudes"), dtype=float)
        if amplitudes.ndim != 2 or not np.all(np.isfinite(amplitudes)):
            raise ValueError("SF-TDA tracking checkpoint amplitudes are invalid")
        self._last_sf_subspace_reference = fingerprints
        self._last_sf_coupling_amplitudes = amplitudes.copy()
        self._last_sf_coupling_key = None
        self._last_sf_coupling = None

    @staticmethod
    def _copy(values: np.ndarray | None) -> np.ndarray | None:
        return None if values is None else np.asarray(values, dtype=float).copy()

    def evaluate(
        self, coordinates_angstrom: np.ndarray, *, compute_hessian: bool = False
    ) -> MatrixGFN2Evaluation:
        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atoms), 3):
            raise ValueError(
                "MATRIX-GFN2 resident coordinates must have shape natoms x 3"
            )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("MATRIX-GFN2 resident coordinates must be finite")
        if self._last_coordinates is not None and np.array_equal(
            coordinates, self._last_coordinates
        ) and (
            not compute_hessian
            or self._last_evaluation.hessian_hartree_per_bohr2 is not None
        ):
            self._cache_hits += 1
            assert self._last_evaluation is not None
            cached = self._last_evaluation
            return MatrixGFN2Evaluation(
                cached.energy_hartree,
                cached.gradient_hartree_per_bohr.copy(),
                self._copy(cached.charges),
                self._copy(cached.bond_orders),
                True,
                cached.telemetry,
                self._copy(cached.dipole_au),
                self._copy(cached.hessian_hartree_per_bohr2) if compute_hessian else None,
            )
        include_population = self._population_analysis_pending
        result = self._calculator.evaluate(
            coordinates,
            compute_cm5=False,
            compute_bond_orders=include_population,
            **({"retain_orbitals": True} if compute_hessian else {}),
        )
        hessian = None
        if compute_hessian:
            hessian = np.asarray(
                self._calculator.ground_state_hessian(
                    result.orbital_state
                ).hessian_hartree_per_bohr2,
                dtype=float,
            )
            if hessian.shape != (coordinates.size, coordinates.size) or not np.all(
                np.isfinite(hessian)
            ):
                raise ValueError("MATRIX-GFN2 returned an invalid Cartesian Hessian")
        self._backend_evaluations += 1
        self._scc_iterations += int(result.iterations)
        dipole_au = None
        if getattr(result, "orbital_state", None) is not None:
            from matrix_gfn2 import molecular_dipole_mo

            state = result.orbital_state
            occupations = state.alpha_occupations + state.beta_occupations
            electronic = np.einsum(
                "i,xii->x", occupations, molecular_dipole_mo(state), optimize=True,
            )
            nuclear = np.einsum(
                "i,ia->a", np.asarray(self._calculator._valence, dtype=float),
                coordinates / 0.529177210903,
            )
            dipole_au = nuclear + electronic
        evaluation = MatrixGFN2Evaluation(
            energy_hartree=float(result.energy_hartree),
            gradient_hartree_per_bohr=np.asarray(
                result.gradient_hartree_per_bohr, dtype=float
            ).reshape(-1),
            charges=(
                np.asarray(result.charges_electron, dtype=float)
                if include_population or self.population_analysis == "scc"
                else None
            ),
            bond_orders=(
                np.asarray(result.bond_orders, dtype=float)
                if include_population
                else None
            ),
            telemetry=getattr(result, "telemetry", None),
            dipole_au=dipole_au,
            hessian_hartree_per_bohr2=self._copy(hessian),
        )
        self._population_analysis_pending = self.population_analysis in {"every-point", "bond-orders"}
        self._last_coordinates = coordinates.copy()
        self._last_evaluation = evaluation
        return MatrixGFN2Evaluation(
            evaluation.energy_hartree,
            evaluation.gradient_hartree_per_bohr.copy(),
            self._copy(evaluation.charges),
            self._copy(evaluation.bond_orders),
            False,
            evaluation.telemetry,
            self._copy(evaluation.dipole_au),
            self._copy(evaluation.hessian_hartree_per_bohr2),
        )

    def write_scaling_observables(
        self, path: Path | str, coordinates_angstrom: np.ndarray
    ) -> Path:
        """Evaluate and serialize SCC/Mayer descriptors at one frozen geometry.

        This is deliberately part of the resident session so a KeyMaker-owned
        optimization can emit its fitting descriptors without launching a
        second, unaudited calculation path.
        """

        from matrix_gfn2 import write_gfn2_scaling_observables

        coordinates = np.asarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atoms), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("MATRIX-GFN2 scaling geometry must have shape natoms x 3")
        result = self._calculator.evaluate(
            coordinates,
            compute_cm5=False,
            compute_bond_orders=True,
        )
        self._backend_evaluations += 1
        self._scc_iterations += int(result.iterations)
        return write_gfn2_scaling_observables(path, self.atoms, coordinates, result)

    def evaluate_sf_tda(
        self,
        coordinates_angstrom: np.ndarray,
        *,
        target_state_id: int,
        roots: int,
    ) -> MatrixGFN2SFTDAEvaluation:
        """Evaluate one target while retaining the full APOC root manifold."""

        coordinates = np.ascontiguousarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atoms), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("resident SF-TDA coordinates must have shape natoms x 3")
        target = int(target_state_id)
        root_count = int(roots)
        if target < 1 or root_count < target:
            raise ValueError("resident SF-TDA roots must include the requested state")
        key = (coordinates.tobytes(), target, root_count)
        if key == self._last_sf_key and self._last_sf_evaluation is not None:
            cached = self._last_sf_evaluation
            return MatrixGFN2SFTDAEvaluation(
                cached.target_state_id,
                cached.energy_hartree,
                cached.gradient_hartree_per_bohr,
                cached.ground_energy_hartree,
                cached.excitation_energies_hartree,
                cached.transition_density_vectors,
                cached.difference_density_vectors,
                True,
            )
        result = self._calculator.evaluate_sf_tda(coordinates, roots=root_count)
        properties = tuple(result.sf_tda_result.transition_properties)
        transition = np.stack(
            tuple(np.asarray(item.transition_density_ao).reshape(-1) for item in properties)
        )
        difference = np.stack(
            tuple(np.asarray(item.difference_density_ao).reshape(-1) for item in properties)
        )
        index = target - 1
        evaluation = MatrixGFN2SFTDAEvaluation(
            target,
            float(result.total_energies_hartree[index]),
            np.asarray(result.total_gradients_hartree_per_bohr[index]).reshape(-1),
            float(result.ground_state.energy_hartree),
            result.excitation_energies_hartree,
            transition,
            difference,
            False,
        )
        self._backend_evaluations += 1
        self._scc_iterations += int(result.ground_state.iterations)
        self._last_sf_key = key
        self._last_sf_evaluation = evaluation
        return evaluation

    def evaluate_tda_two_photon(
        self,
        coordinates_angstrom: np.ndarray,
        *,
        target_state_id: int,
        spectral_range_ev: float = 20.0,
    ) -> MatrixGFN2TwoPhotonEvaluation:
        """Evaluate a one-based closed-shell TDA 2PA target for LINK."""

        coordinates = np.ascontiguousarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atoms), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("resident TDA two-photon coordinates must have shape natoms x 3")
        target = int(target_state_id)
        window = float(spectral_range_ev)
        if target < 1 or window <= 0.0:
            raise ValueError("resident TDA two-photon state and window must be positive")
        if self.multiplicity != 1:
            raise ValueError("ordinary TDA two-photon response requires a closed shell")
        key = (coordinates.tobytes(), target, window)
        if key == self._last_two_photon_key and self._last_two_photon_evaluation is not None:
            cached = self._last_two_photon_evaluation
            return MatrixGFN2TwoPhotonEvaluation(
                cached.target_state_id,
                cached.strength_au,
                cached.strength_gradient_au_per_bohr,
                cached.transition_tensor_au,
                cached.maximum_response_residual,
                True,
            )
        result = self._calculator.evaluate_tda_two_photon_gradient(
            coordinates,
            target_state=target - 1,
            spectral_range_ev=window,
        )
        evaluation = MatrixGFN2TwoPhotonEvaluation(
            target,
            float(result.response.isotropic_strength_au),
            np.asarray(result.strength_gradient_au_per_bohr).reshape(-1),
            result.response.transition_tensor_au,
            float(result.maximum_response_residual),
            False,
        )
        self._backend_evaluations += 1
        self._last_two_photon_key = key
        self._last_two_photon_evaluation = evaluation
        return evaluation

    def evaluate_sf_tda_coupling(
        self,
        coordinates_angstrom: np.ndarray,
        *,
        state_ids: tuple[int, int],
        roots: int,
        gap_tolerance_hartree: float = 1.0e-7,
        minimum_principal_overlap: float = 0.70,
        ambiguity_margin: float = 0.02,
    ) -> MatrixGFN2SFTDACouplingEvaluation:
        """Evaluate two one-based SF-TDA states and their analytic coupling."""

        coordinates = np.ascontiguousarray(coordinates_angstrom, dtype=float)
        if coordinates.shape != (len(self.atoms), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("resident SF-TDA coordinates must have shape natoms x 3")
        first, second = (int(value) for value in state_ids)
        if min(first, second) < 1 or first == second:
            raise ValueError("resident SF-TDA coupling state IDs must be distinct and one-based")
        root_count = max(int(roots), first, second)
        key = (coordinates.tobytes(), first, second, root_count)
        if key == self._last_sf_coupling_key and self._last_sf_coupling is not None:
            cached = self._last_sf_coupling
            return MatrixGFN2SFTDACouplingEvaluation(
                cached.state_ids,
                cached.energies_hartree,
                cached.gradients_hartree_per_bohr,
                cached.interstate_numerator_hartree_per_bohr,
                cached.derivative_coupling_per_bohr,
                cached.energy_gap_hartree,
                cached.excitation_energies_hartree,
                cached.transition_density_vectors,
                cached.subspace_match,
                True,
            )
        from matrix_apoc import StateFingerprint, follow_state_subspace

        manifold = self._calculator.evaluate_sf_tda(coordinates, roots=root_count)
        properties = tuple(manifold.sf_tda_result.transition_properties)
        transition = np.stack(
            tuple(np.asarray(item.transition_density_ao).reshape(-1) for item in properties)
        )
        representation = "matrix-gfn2-sf-tda-transition-density-ao.v1"
        fingerprints = tuple(
            StateFingerprint(
                index + 1,
                vector,
                representation,
                energy_hartree=float(manifold.total_energies_hartree[index]),
            )
            for index, vector in enumerate(transition)
        )
        subspace_match = None
        if self._last_sf_subspace_reference is None:
            selected_ids = (first, second)
        else:
            selected, subspace_match = follow_state_subspace(
                self._last_sf_subspace_reference,
                fingerprints,
                minimum_principal_overlap=float(minimum_principal_overlap),
                ambiguity_margin=float(ambiguity_margin),
                strict=True,
            )
            selected_ids = tuple(int(item.state_id) for item in selected)
        indices = np.asarray(tuple(value - 1 for value in selected_ids), dtype=int)
        result = self._calculator.sf_tda_coupling_from_manifold(
            manifold,
            state_i=int(indices[0]),
            state_j=int(indices[1]),
            gap_tolerance_hartree=float(gap_tolerance_hartree),
            reference_amplitudes=self._last_sf_coupling_amplitudes,
        )
        derivative = result.derivative_coupling_per_bohr
        evaluation = MatrixGFN2SFTDACouplingEvaluation(
            selected_ids,
            manifold.total_energies_hartree[indices],
            manifold.total_gradients_hartree_per_bohr[indices].reshape(2, -1),
            result.interstate_numerator_hartree_per_bohr.reshape(-1),
            None if derivative is None else derivative.reshape(-1),
            result.energy_gap_hartree,
            manifold.excitation_energies_hartree,
            transition,
            subspace_match,
            False,
        )
        self._backend_evaluations += 1
        self._scc_iterations += int(manifold.ground_state.iterations)
        self._last_sf_coupling_key = key
        self._last_sf_coupling = evaluation
        self._last_sf_coupling_amplitudes = np.asarray(
            manifold.sf_tda_result.amplitudes, dtype=float
        ).copy()
        self._last_sf_subspace_reference = tuple(fingerprints[index] for index in indices)
        return evaluation

    def bond_order_matrix(self, coordinates_angstrom: np.ndarray) -> np.ndarray:
        """Return the Mayer bond-order matrix for trajectory event detection."""

        evaluation = self.evaluate(np.asarray(coordinates_angstrom, dtype=float))
        if evaluation.bond_orders is None:
            raise RuntimeError(
                "MATRIX-GFN2 bond orders require population_analysis='bond-orders' or 'every-point'"
            )
        return np.asarray(evaluation.bond_orders, dtype=float).copy()


__all__ = [
    "MatrixGFN2Evaluation",
    "MatrixGFN2SFTDAEvaluation",
    "MatrixGFN2SFTDACouplingEvaluation",
    "MatrixGFN2Session",
    "MatrixGFN2Statistics",
    "matrix_gfn2_available",
]
