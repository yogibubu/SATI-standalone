"""LINK geometry points: backend S0 gradient plus root-followed std2 correction."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading

import numpy as np

from matrix_core import CalculationResources, require_authorized_descendant_calculation
from matrix_apoc import electronic_state_from_std2, follow_excited_state, cross_ao_overlap_from_molden
from .std2 import STD2Settings, prepare_std2_spectrum, evaluate_std2_spectrum

BOHR_TO_ANGSTROM = 0.529177210903


class STD2GeometrySession:
    """A reference advances only on acceptance; derivative branches never advance it."""

    def __init__(self, atoms, backend, run_dir):
        self.atoms = tuple(atoms)
        self.backend = backend
        self.root = Path(run_dir) / "std2-points"
        self.root.mkdir(parents=True, exist_ok=True)
        self.contract = dict(backend.std2_contract or {})
        if set(self.contract) != {"settings", "reference"}:
            raise ValueError("std2 geometry contract requires settings and reference")
        self.settings = STD2Settings(**self.contract["settings"])
        self.settings.arguments()
        self.reference_config = dict(self.contract["reference"])
        if self.reference_config.get("backend") not in {"pyscf", "command"}:
            raise ValueError("std2 reference requires pyscf or an explicit command provider")
        if backend.electronic_state < 1 or backend.state_tracking != "apoc":
            raise ValueError("std2 optimization requires a positive target root and APOC tracking")
        if backend.gradient_mode != "hybrid":
            raise ValueError("std2 geometry gradients require explicit hybrid mode")
        if backend.numerical_gradient_step_bohr <= 0:
            raise ValueError("std2 excitation-gradient step must be positive")
        self.resources = CalculationResources(1, backend.processors, float(backend.memory_gb or 1))
        self.reference = None
        self._counter = 0
        self._lock = threading.RLock()

    def _ground(self, coordinates, directory, gradient):
        """The reference backend supplies S0 and exports the very same orbitals."""
        directory.mkdir(parents=True)
        request = {
            "schema": "matrix.link.std2.ground_request.v1", "atoms": self.atoms,
            "coordinates_angstrom": np.asarray(coordinates).tolist(),
            "charge": self.backend.charge, "multiplicity": self.backend.multiplicity,
            "require_gradient": gradient, "reference": self.reference_config,
        }
        request_path = directory / "request.json"
        request_path.write_text(json.dumps(request, indent=2) + "\n")
        config = self.reference_config
        if config["backend"] == "pyscf":
            from matrix_pyscf import PySCFResidentCalculator
            from pyscf.tools import molden

            require_authorized_descendant_calculation(
                backend="PySCF/std2-reference", input_path=request_path,
                command=("pyscf-resident", str(request_path)), workdir=directory,
            )
            if self.settings.molden_style != 3:
                raise ValueError("PySCF reference export requires std2 Molden style 3")
            calculator = PySCFResidentCalculator(
                self.atoms, coordinates, method=config.get("method", "HF"),
                reference=config.get("reference"), basis=config.get("basis", "sto-3g"),
                charge=self.backend.charge, multiplicity=self.backend.multiplicity,
                density_fitting=bool(config.get("density_fitting", False)),
                processors=self.backend.processors,
            )
            # Reuse the existing reference calculator and its gradient scanner.
            mf = calculator._scanner.base
            mf.conv_tol = float(config.get("conv_tol", 1e-11))
            mf.conv_tol_grad = float(config.get("conv_tol_grad", 1e-8))
            mf.max_memory = self.resources.memory_per_job_gb * 1024
            point = calculator.evaluate(coordinates, include_gradient=gradient)
            mf = calculator._scanner.base
            if not mf.converged:
                raise RuntimeError("std2 ground-state backend did not converge")
            path = directory / "ground.molden"
            molden.from_scf(mf, str(path))
            payload = {
                "schema": "matrix.link.std2.ground_point.v1", "converged": True,
                "energy_hartree": point.energy_hartree,
                "gradient_hartree_per_bohr": None if point.gradient_hartree_per_bohr is None
                else np.asarray(point.gradient_hartree_per_bohr).reshape(-1).tolist(),
                "molden": str(path), "coordinates_angstrom": np.asarray(coordinates).tolist(),
            }
        else:
            arguments = config.get("command")
            if not isinstance(arguments, list) or not arguments:
                raise ValueError("reference command must be a nonempty argument list")
            output = directory / "ground.json"
            command = tuple(str(item).format(request=str(request_path), result=str(output),
                                            workdir=str(directory)) for item in arguments)
            require_authorized_descendant_calculation(
                backend="std2-ground-provider", input_path=request_path,
                command=command, workdir=directory,
            )
            completed = subprocess.run(command, cwd=directory, timeout=self.backend.timeout,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            (directory / "provider.out").write_text(completed.stdout)
            if completed.returncode:
                raise RuntimeError("std2 reference provider failed; inspect provider.out")
            payload = json.loads(output.read_text())
            path = Path(payload["molden"])
            path = path if path.is_absolute() else directory / path
        if payload.get("schema") != "matrix.link.std2.ground_point.v1" or payload.get("converged") is not True:
            raise ValueError("std2 requires a converged ground-state point")
        returned = np.asarray(payload["coordinates_angstrom"], dtype=float)
        if returned.shape != np.asarray(coordinates).shape or not np.allclose(returned, coordinates, atol=1e-9, rtol=0):
            raise ValueError("std2 reference changed the requested Cartesian frame")
        energy = float(payload["energy_hartree"])
        g = payload.get("gradient_hartree_per_bohr")
        g = None if g is None else np.asarray(g, dtype=float).reshape(-1)
        if not np.isfinite(energy) or (gradient and (g is None or g.size != returned.size or not np.all(np.isfinite(g)))):
            raise ValueError("std2 reference lacks finite ground-state energy/gradient")
        return energy, g, path

    def _point(self, coordinates, directory, *, gradient=False, reference=None):
        energy, ground_gradient, molden = self._ground(coordinates, directory / "ground", gradient)
        plan = prepare_std2_spectrum(
            molden, directory / "spectrum", settings=self.settings,
            resources=self.resources, executable=self.backend.executable,
        )
        evaluate_std2_spectrum(plan, launch_authorization=None, timeout=self.backend.timeout or 300)
        state = electronic_state_from_std2(plan.workdir)
        if state.charge != self.backend.charge or state.multiplicity != self.backend.multiplicity:
            raise ValueError("std2 Molden electron/spin count differs from the reference contract")
        from matrix_chem.topology.elements import atomic_number

        if tuple(state.atomic_numbers) != tuple(atomic_number(atom) for atom in self.atoms):
            raise ValueError("std2 Molden changed the requested atom order")
        if not np.allclose(state.coordinates_bohr * BOHR_TO_ANGSTROM, coordinates, atol=1e-7, rtol=0):
            raise ValueError("std2 Molden geometry differs from the backend point")
        match = None
        if reference is None:
            selected = next((s for s in state.excited_states if s.state_id == self.backend.electronic_state), None)
            if selected is None:
                raise ValueError("requested std2 root is outside the computed window")
        else:
            ref_state, ref_selected, ref_dir = reference
            cross = cross_ao_overlap_from_molden(Path(ref_dir) / "reference.molden",
                                                Path(plan.workdir) / "reference.molden")
            selected, match = follow_excited_state(
                ref_selected, state.excited_states, cross,
                overlap_reference=ref_state.overlap_ao, overlap_candidate=state.overlap_ao,
                minimum_overlap=self.backend.state_tracking_minimum_overlap,
                ambiguity_margin=self.backend.state_tracking_ambiguity_margin, strict=True,
            )
        return energy, ground_gradient, (state, selected, Path(plan.workdir)), match

    def evaluate(self, point, requested_properties=()):
        """Differentiate only the std2 excitation; retain the backend S0 gradient."""
        from .scan import PointEvaluationResult
        from .adaptive_finite_difference import (
            LocalModeSet, AdaptiveStepConfig, numerical_gradient_from_energies,
        )
        with self._lock:
            self._counter += 1
            directory = self.root / f"point-{self._counter:06d}"
            if directory.exists():
                raise FileExistsError("std2 session requires a fresh point directory")
            directory.mkdir()
            need_gradient = not requested_properties or "gradient" in requested_properties or "hessian" in requested_properties
            energy, ground_gradient, center, match = self._point(
                point.coordinates_angstrom, directory / "center", gradient=need_gradient,
                reference=self.reference,
            )
            state, selected, spectrum_dir = center
            gradient = None
            branches = []
            if need_gradient:
                branch_index = 0

                def excitation_energy(coordinates_bohr):
                    nonlocal branch_index
                    branch_index += 1
                    _, _, branch, branch_match = self._point(
                        coordinates_bohr * BOHR_TO_ANGSTROM, directory / f"gradient-{branch_index:05d}",
                        reference=center,
                    )
                    branches.append({"index": branch_index, "root": branch[1].state_id,
                                     "overlap": branch_match.overlap, "margin": branch_match.margin})
                    return branch[1].energy_hartree

                size = point.coordinates_angstrom.size
                step = self.backend.numerical_gradient_step_bohr
                fd = numerical_gradient_from_energies(
                    np.asarray(point.coordinates_angstrom) / BOHR_TO_ANGSTROM,
                    selected.energy_hartree,
                    LocalModeSet(np.ones(size), np.eye(size), (), "frozen_cartesian_frame"),
                    excitation_energy, AdaptiveStepConfig(1e-7, minimum_step=step, maximum_step=step),
                    stencil="central",
                )
                gradient = ground_gradient + fd.gradient
            execution = {
                "state_tracking": "APOC", "requested_electronic_state": self.backend.electronic_state,
                "selected_electronic_state": selected.state_id, "gradient_electronic_state": selected.state_id,
                "state_overlap": 1.0 if match is None else match.overlap,
                "state_assignment_margin": 1.0 if match is None else match.margin,
                "state_assignment_continuous": True, "state_assignment_ambiguous": False,
                "state_ids": [s.state_id for s in state.excited_states],
                "excitation_energies_hartree": [s.energy_hartree for s in state.excited_states],
                "ground_energy_hartree": energy,
                "std2_spectrum_directory": str(spectrum_dir),
                "gradient_source": "backend_S0_analytic_plus_APOC_followed_std2_excitation_central_fd",
                "std2_gradient_step_bohr": self.backend.numerical_gradient_step_bohr,
                "std2_gradient_branches": branches,
                "excitation_eigenvalue_precision": "upstream_real4_not_float64_despite_printed_digits",
                "energy_evaluations": 1 + len(branches), "gradient_evaluations": int(need_gradient),
                "hessian_evaluations": 0,
            }
            result = PointEvaluationResult(
                point_index=point.index, displacement=point.displacement,
                coordinate_key=point.coordinate_key, side=point.side,
                energy_hartree=energy + selected.energy_hartree,
                gradient_hartree_per_bohr=gradient,
                backend_coordinates_angstrom=point.coordinates_angstrom,
                execution=execution, source="std2 + external ground-state backend",
            )
            (directory / "point.json").write_text(json.dumps(execution, indent=2) + "\n")
            return result

    def accept(self, result):
        """Promote exactly the point accepted by LINK, including cached points."""
        with self._lock:
            if not result.execution.get("state_assignment_continuous") or result.execution.get("state_assignment_ambiguous"):
                raise ValueError("cannot promote a discontinuous or ambiguous std2 state")
            directory = Path(result.execution["std2_spectrum_directory"])
            state = electronic_state_from_std2(directory)
            selected = next(s for s in state.excited_states
                            if s.state_id == result.execution["selected_electronic_state"])
            self.reference = state, selected, directory


__all__ = ["STD2GeometrySession"]
