"""Complete resident nonperiodic GFN-FF energy/gradient assembly."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .bonded import GFNFFDynamicGaussianBondModel
from .gfnff_d3 import GFNFFD3Model
from .gfnff_eeq import GFNFFEEQModel, _CoordinationState
from .gfnff_nci import GFNFFNCIModel
from .gfnff_repulsion import GFNFFRepulsionModel, GFNFFRepulsionSession
from .gfnff_valence import GFNFFValenceModel
from .native_kernels import dynamic_gaussian_bond_shared_coordination, native_zaff_backend
from .nonbonded import (
    BOHR_TO_ANGSTROM,
    PersistentVerletNeighborList,
    select_mm_runtime_policy,
)


GFNFF_COMPLETE_COMPONENTS = (
    "bond",
    "angle_torsion",
    "repulsion",
    "electrostatics",
    "dispersion",
    "hydrogen_halogen_bonded_atm",
)


@dataclass(frozen=True)
class GFNFFEvaluation:
    """One complete GFN-FF energy/gradient evaluation in atomic units."""

    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    component_energies_hartree: Mapping[str, float]
    eeq_charges: np.ndarray
    coordination_numbers: np.ndarray
    execution: Mapping[str, Any]


class GFNFFDynamicsRuntime:
    """Resident sparse runtime for one fully compiled GFN-FF topology.

    ARCHITECT resolves all topology-dependent lists and parameters once.  At
    each geometry the runtime updates one shared sparse coordination pair
    list, the repulsion and D3 lists, and evaluates the six native component
    kernels without materializing an atom-pair matrix.
    """

    def __init__(
        self,
        model: Mapping[str, Any],
        *,
        neighbor_skin_bohr: float = 2.0,
        profile: bool = False,
    ) -> None:
        migration = model.get("gfnff_migration")
        if not isinstance(migration, Mapping) or migration.get("calculator_ready") is not True:
            missing = () if not isinstance(migration, Mapping) else migration.get(
                "missing_term_families", ()
            )
            suffix = "" if not missing else f": {', '.join(map(str, missing))}"
            raise ValueError(f"native GFN-FF model is incomplete{suffix}")
        self.model = dict(model)
        self.natoms = len(tuple(model.get("atoms", ())))
        if self.natoms < 1:
            raise ValueError("compiled GFN-FF model has no atoms")

        self.coordination_record = self._record("gfnff_coordination", "GFNFF_LOG_ERF")
        self.eeq_record = self._record("gfnff_eeq", "GFNFF_EEQ_GAUSSIAN")
        self.d3_record = self._record("gfnff_d3_two_body", "GFNFF_D3_BJ_TWO_BODY")
        repulsion_record = self._record("gfnff_repulsion", "GFNFF_REPULSION_R32")
        valence_record = self._record("gfnff_valence", "GFNFF_EXACT_LOCAL_VALENCE")
        nci_record = self._record("gfnff_nci", "GFNFF_EXACT_NCI")

        if nci_record.get("coverage_certified") is not True:
            raise ValueError("compiled GFN-FF NCI coverage is not certified")
        self.bond_records = tuple(dict(record) for record in model.get("bonds", ()))
        if not self.bond_records or any(
            record.get("functional_form") != "GFNFF_GAUSSIAN"
            or record.get("reference_response") != "DYNAMIC_GFNFF_LOG_CN"
            for record in self.bond_records
        ):
            raise ValueError("complete GFN-FF requires dynamic-CN Gaussian bonds")

        cutoff_angstrom = float(self.coordination_record["cutoff_angstrom"])
        self.scaling_policy = select_mm_runtime_policy(
            np.zeros((self.natoms, 3)),
            cutoff_angstrom=cutoff_angstrom,
            neighbor_list_minimum_atoms=1,
            fmm_minimum_atoms=int(self.eeq_record.get("fmm_minimum_atoms", 256)),
            fmm_precision=float(self.eeq_record.get("fmm_precision", 1.0e-10)),
            materialize_neighbor_pairs=False,
            require_linear_scaling=True,
        )
        self.eeq_record["backend"] = self.scaling_policy.electrostatic_backend
        self.coordination_cutoff_bohr = cutoff_angstrom / BOHR_TO_ANGSTROM
        self.coordination_neighbors = PersistentVerletNeighborList(
            cutoff_bohr=self.coordination_cutoff_bohr,
            skin_bohr=float(neighbor_skin_bohr),
        )
        self.dispersion_cutoff_bohr = float(self.d3_record["cutoff_bohr"])
        self.dispersion_neighbors = PersistentVerletNeighborList(
            cutoff_bohr=self.dispersion_cutoff_bohr,
            skin_bohr=float(neighbor_skin_bohr),
        )
        self.repulsion = GFNFFRepulsionSession(
            GFNFFRepulsionModel.from_record(repulsion_record),
            neighbor_skin_bohr=neighbor_skin_bohr,
        )
        self.valence = GFNFFValenceModel.from_record(valence_record)
        self.nci = GFNFFNCIModel.from_record(nci_record)
        self.evaluation_count = 0
        self.profile = bool(profile)
        self._prepared_bond: GFNFFDynamicGaussianBondModel | None = None
        self._prepared_eeq: GFNFFEEQModel | None = None
        self._prepared_d3: GFNFFD3Model | None = None
        self.model_rebuild_count = 0

    def _prepare_components(
        self, coordination_pairs: np.ndarray, dispersion_pairs: np.ndarray
    ) -> tuple[GFNFFDynamicGaussianBondModel, GFNFFEEQModel, GFNFFD3Model]:
        """Rebuild parameters only when an active pair list actually changes.

        Compare the cutoff-filtered lists, not merely the Verlet rebuild
        counter: pairs can cross a cutoff while the skin is still valid.
        Geometry-dependent CN, damping and charges are never cached here.
        """

        cn_changed = self._prepared_bond is None or not np.array_equal(
            coordination_pairs, self._prepared_bond.coordination_pairs
        )
        d3_changed = self._prepared_d3 is None or not np.array_equal(
            dispersion_pairs, self._prepared_d3.dispersion_pairs
        )
        if cn_changed:
            self._prepared_bond = self._bond_model(coordination_pairs)
            self._prepared_eeq = GFNFFEEQModel.from_records(
                self.eeq_record, self.coordination_record, coordination_pairs
            )
        if cn_changed or d3_changed:
            self._prepared_d3 = GFNFFD3Model.from_records(
                self.d3_record, self.coordination_record, coordination_pairs, dispersion_pairs
            )
            self.model_rebuild_count += 1
        return self._prepared_bond, self._prepared_eeq, self._prepared_d3

    def _record(self, name: str, kind: str) -> dict[str, Any]:
        record = self.model.get(name)
        if not isinstance(record, Mapping) or record.get("kind") != kind:
            raise ValueError(f"compiled GFN-FF model lacks {name}")
        return dict(record)

    @staticmethod
    def _filtered_pairs(
        neighbors: PersistentVerletNeighborList,
        coordinates_bohr: np.ndarray,
        cutoff_bohr: float,
    ) -> np.ndarray:
        candidates = neighbors.native_candidates(coordinates_bohr)
        if not len(candidates):
            return candidates
        delta = coordinates_bohr[candidates[:, 0]] - coordinates_bohr[candidates[:, 1]]
        return candidates[np.sum(delta * delta, axis=1) <= cutoff_bohr**2]

    def _bond_model(self, coordination_pairs: np.ndarray) -> GFNFFDynamicGaussianBondModel:
        radii = np.asarray(
            self.coordination_record["covalent_radii_angstrom"], dtype=float
        )
        bonds = np.asarray(
            [record["atoms"] for record in self.bond_records], dtype=np.intp
        ).reshape(-1, 2)
        return GFNFFDynamicGaussianBondModel(
            coordination_pairs=coordination_pairs,
            coordination_radii_angstrom=(
                radii[coordination_pairs[:, 0]] + radii[coordination_pairs[:, 1]]
            ),
            bonds=bonds,
            amplitudes_hartree=np.asarray(
                [record["amplitude_hartree"] for record in self.bond_records]
            ),
            exponents_per_angstrom2=np.asarray(
                [record["exponent_per_angstrom2"] for record in self.bond_records]
            ),
            reference_bases_angstrom=np.asarray(
                [record["reference_base_angstrom"] for record in self.bond_records]
            ),
            reference_cn_coefficients_angstrom=np.asarray(
                [
                    record["reference_cn_coefficients_angstrom"]
                    for record in self.bond_records
                ]
            ),
            cn_max=float(self.coordination_record.get("cn_max", 4.4)),
            steepness=float(self.coordination_record.get("steepness", 7.5)),
        )

    def evaluate(self, coordinates_bohr: np.ndarray) -> GFNFFEvaluation:
        timings: dict[str, float] = {}
        started = perf_counter() if self.profile else 0.0

        def mark(name: str) -> None:
            nonlocal started
            if self.profile:
                now = perf_counter()
                timings[name] = now - started
                started = now

        xyz = np.asarray(coordinates_bohr, dtype=float)
        if xyz.shape != (self.natoms, 3) or np.any(~np.isfinite(xyz)):
            raise ValueError("GFN-FF coordinates have wrong dimensions")
        coordination_pairs = self._filtered_pairs(
            self.coordination_neighbors, xyz, self.coordination_cutoff_bohr
        )
        dispersion_pairs = self._filtered_pairs(
            self.dispersion_neighbors, xyz, self.dispersion_cutoff_bohr
        )

        bond_model, eeq_model, d3_model = self._prepare_components(
            coordination_pairs, dispersion_pairs
        )
        mark("neighbor_lists_and_parameters")
        shared_native = native_zaff_backend(
            max(len(coordination_pairs), len(bond_model.bonds))
        ).accelerated
        if shared_native:
            bond_energy, bond_gradient, atom_state, pair_state = (
                dynamic_gaussian_bond_shared_coordination(
                    xyz * BOHR_TO_ANGSTROM,
                    bond_model.coordination_pairs,
                    bond_model.coordination_radii_angstrom,
                    bond_model.bonds,
                    bond_model.amplitudes_hartree,
                    bond_model.exponents_per_angstrom2,
                    bond_model.reference_bases_angstrom,
                    bond_model.reference_cn_coefficients_angstrom,
                    bond_model.cn_max,
                    bond_model.steepness,
                )
            )
            # Native bond derivatives use Angstrom internally and return
            # the gradient in atomic units. EEQ/D3 need the CN derivatives
            # in bohr; atom-state columns are dimensionless views.
            state = _CoordinationState(
                atom_state[:, 0], atom_state[:, 1], atom_state[:, 2], atom_state[:, 3],
                pair_state[:, 0] / BOHR_TO_ANGSTROM, pair_state[:, 1:4],
                pair_state[:, 4] * BOHR_TO_ANGSTROM,
                pair_state[:, 5] * BOHR_TO_ANGSTROM**2,
            )
            bond_cn = state.values
        else:
            bond_energy, bond_gradient_angstrom, _product, bond_cn = bond_model.evaluate(
                xyz * BOHR_TO_ANGSTROM
            )
            bond_gradient = bond_gradient_angstrom * BOHR_TO_ANGSTROM
            state = eeq_model._coordination(xyz)
        mark("bond_and_coordination")
        eeq = eeq_model._evaluate_with_coordination(xyz, state)
        mark("eeq")
        d3 = d3_model._evaluate_with_coordination(xyz, state)
        mark("dispersion")
        repulsion = self.repulsion.evaluate(xyz)
        mark("repulsion")
        valence = self.valence.evaluate(xyz)
        mark("angle_torsion")
        nci = self.nci.evaluate(xyz)
        mark("nci")

        # In native mode all three components consume the very same state.
        # Only the independent NumPy bond fallback needs a consistency check.
        if not shared_native and not np.allclose(
            bond_cn, state.values, atol=2.0e-14, rtol=0.0
        ):
            raise RuntimeError("GFN-FF component coordination states disagree")
        components = {
            "bond": float(bond_energy),
            "angle_torsion": float(valence.energy_hartree),
            "repulsion": float(repulsion.energy_hartree),
            "electrostatics": float(eeq.energy_hartree),
            "dispersion": float(d3.energy_hartree),
            "hydrogen_halogen_bonded_atm": float(nci.energy_hartree),
        }
        gradients = (
            bond_gradient.reshape(-1),
            valence.gradient_hartree_per_bohr,
            repulsion.gradient_hartree_per_bohr,
            eeq.gradient_hartree_per_bohr,
            d3.gradient_hartree_per_bohr,
            nci.gradient_hartree_per_bohr,
        )
        gradient = gradients[0].copy()
        for contribution in gradients[1:]:
            gradient += contribution
        energy = float(sum(components.values()))
        if not np.isfinite(energy) or np.any(~np.isfinite(gradient)):
            raise FloatingPointError("non-finite complete GFN-FF result")
        gradient.setflags(write=False)
        charges = np.asarray(eeq.charges, dtype=float).copy()
        charges.setflags(write=False)
        coordination = np.asarray(bond_cn, dtype=float).copy()
        coordination.setflags(write=False)
        self.evaluation_count += 1
        mark("assembly")
        execution = MappingProxyType(
            {
                "runtime": "GFNFF_COMPLETE_RESIDENT",
                "matrix_storage": "MATRIX_FREE",
                "pair_storage": "SPARSE_PERSISTENT_VERLET",
                "component_order": GFNFF_COMPLETE_COMPONENTS,
                "coordination_pair_count": len(coordination_pairs),
                "dispersion_pair_count": len(dispersion_pairs),
                "repulsion_pair_count": (
                    repulsion.bonded_pair_count + repulsion.nonbonded_pair_count
                ),
                "valence_term_count": self.valence.term_count,
                "nci_term_count": self.nci.term_count,
                "coordination_rebuild_count": self.coordination_neighbors.rebuild_count,
                "coordination_reuse_count": self.coordination_neighbors.reuse_count,
                "dispersion_rebuild_count": self.dispersion_neighbors.rebuild_count,
                "dispersion_reuse_count": self.dispersion_neighbors.reuse_count,
                "evaluation_count": self.evaluation_count,
                "parameter_rebuild_count": self.model_rebuild_count,
                "coordination_evaluations": 1 if shared_native else 2,
                "coordination_backend": "CPP_SHARED" if shared_native else "NUMPY_SHARED_EEQ_D3",
                "eeq_iterations": eeq.iterations,
                "eeq_residual_norm": eeq.residual_norm,
                **({"component_seconds": timings} if self.profile else {}),
                "long_range_backend": self.scaling_policy.electrostatic_backend,
                "linear_scaling_required": True,
                "linear_scaling_certified": (
                    self.scaling_policy.linear_scaling_certified
                ),
            }
        )
        return GFNFFEvaluation(
            energy_hartree=energy,
            gradient_hartree_per_bohr=gradient,
            component_energies_hartree=MappingProxyType(components),
            eeq_charges=charges,
            coordination_numbers=coordination,
            execution=execution,
        )


__all__ = [
    "GFNFF_COMPLETE_COMPONENTS",
    "GFNFFDynamicsRuntime",
    "GFNFFEvaluation",
]
