"""Warm-up calibration of the ZAFF mean-field confinement potential."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from matrix_epoc.confinement import (
    EllipsoidalVdwConfinement,
    attach_vdw_confinement,
    vdw_confinement_to_record,
)


ZAFF_VDW_WARMUP_SCHEMA = "matrix.zaff.vdw_warmup.v1"
ZAFF_VDW_CALIBRATION_SCHEMA = "matrix.zaff.vdw_calibration.v1"
K_BOLTZMANN_HARTREE_PER_KELVIN = 3.166811563e-6


def _density_profile(
    value: float | Sequence[float], shell_count: int
) -> np.ndarray:
    raw = np.asarray(value, dtype=float)
    if raw.ndim == 0:
        return np.full(shell_count, float(raw))
    return raw.reshape(shell_count)


@dataclass(frozen=True)
class VdwWarmupConfig:
    """Controls iterative-Boltzmann shell updates and final basis selection."""

    target_number_density_bohr3: Mapping[str, float | Sequence[float]]
    shell_edges_rho: Sequence[float]
    temperature_kelvin: float
    target_total_number_density_bohr3: float | Sequence[float] | None = None
    mixture_fractions: Mapping[str, float] | None = None
    update_damping: float = 0.25
    pseudocount_fraction: float = 1.0e-3
    minimum_blocks: int = 3
    gaussian_count_candidates: Sequence[int] = (2, 3, 4)
    gaussian_width_scale_candidates: Sequence[float] = (0.7, 1.0, 1.4)
    morse_minimum_fraction_candidates: Sequence[float] = (0.08, 0.18, 0.32)
    morse_alpha_layer_candidates: Sequence[float] = (2.0, 4.0, 8.0)
    ridge_candidates: Sequence[float] = (1.0e-10, 1.0e-7, 1.0e-4, 1.0e-2)
    validation_stride: int = 4
    morse_well_depth_prior_hartree: Mapping[str, float] | None = None
    morse_prior_strength: float = 1.0e-4
    gaussian_amplitude_bound_factor: float = 10.0
    fit_parallel_workers: int = 1

    def __post_init__(self) -> None:
        edges = tuple(float(value) for value in self.shell_edges_rho)
        shell_count = len(edges) - 1
        targets = {
            str(label): _density_profile(value, shell_count)
            for label, value in self.target_number_density_bohr3.items()
        }
        total_target = (
            None
            if self.target_total_number_density_bohr3 is None
            else _density_profile(
                self.target_total_number_density_bohr3, shell_count
            )
        )
        fractions = (
            None
            if self.mixture_fractions is None
            else {
                str(label): float(value)
                for label, value in self.mixture_fractions.items()
            }
        )
        counts = tuple(int(value) for value in self.gaussian_count_candidates)
        widths = tuple(float(value) for value in self.gaussian_width_scale_candidates)
        minima = tuple(
            float(value) for value in self.morse_minimum_fraction_candidates
        )
        alphas = tuple(float(value) for value in self.morse_alpha_layer_candidates)
        ridges = tuple(float(value) for value in self.ridge_candidates)
        priors = (
            None
            if self.morse_well_depth_prior_hartree is None
            else {
                str(label): float(value)
                for label, value in self.morse_well_depth_prior_hartree.items()
            }
        )
        if (
            not targets
            or any(
                np.any(~np.isfinite(value)) or np.any(value <= 0.0)
                for value in targets.values()
            )
            or len(edges) < 5
            or any(not math.isfinite(value) for value in edges)
            or not 0.0 <= edges[0] < edges[-1] <= 1.0
            or np.any(np.diff(edges) <= 0.0)
        ):
            raise ValueError("UvdW warm-up needs positive targets and ordered rho shells")
        if total_target is not None and (
            np.any(~np.isfinite(total_target)) or np.any(total_target <= 0.0)
        ):
            raise ValueError("total UvdW shell target must be positive")
        if (total_target is None) != (fractions is None):
            raise ValueError(
                "total UvdW target and mixture fractions must be supplied together"
            )
        if fractions is not None and (
            set(fractions) != set(targets)
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in fractions.values()
            )
            or not math.isclose(sum(fractions.values()), 1.0, abs_tol=1.0e-10)
        ):
            raise ValueError(
                "UvdW mixture fractions must be positive and sum to one"
            )
        if total_target is not None and fractions is not None and any(
            not np.allclose(
                targets[label],
                total_target * fractions[label],
                rtol=1.0e-8,
                atol=1.0e-14,
            )
            for label in targets
        ):
            raise ValueError(
                "species profiles must equal total UvdW profile times mixture fractions"
            )
        if (
            not math.isfinite(float(self.temperature_kelvin))
            or float(self.temperature_kelvin) <= 0.0
            or not 0.0 < float(self.update_damping) <= 1.0
            or not 0.0 < float(self.pseudocount_fraction) < 1.0
            or int(self.minimum_blocks) < 2
            or int(self.validation_stride) < 2
            or float(self.morse_prior_strength) < 0.0
            or float(self.gaussian_amplitude_bound_factor) <= 0.0
            or int(self.fit_parallel_workers) <= 0
        ):
            raise ValueError("invalid UvdW warm-up thermodynamic controls")
        if (
            not counts
            or any(value <= 0 or value + 2 >= len(edges) for value in counts)
            or not widths
            or any(not math.isfinite(value) or value <= 0.0 for value in widths)
            or not minima
            or any(not 0.0 <= value < 1.0 for value in minima)
            or not alphas
            or any(not math.isfinite(value) or value <= 0.0 for value in alphas)
            or not ridges
            or any(not math.isfinite(value) or value < 0.0 for value in ridges)
        ):
            raise ValueError("invalid UvdW Morse--Gaussian model-selection grid")
        if priors is not None and (
            set(priors) != set(targets)
            or any(
                not math.isfinite(value) or value < 0.0
                for value in priors.values()
            )
        ):
            raise ValueError("UvdW Morse priors must cover all explicit types")
        object.__setattr__(self, "target_number_density_bohr3", targets)
        object.__setattr__(
            self, "target_total_number_density_bohr3", total_target
        )
        object.__setattr__(self, "mixture_fractions", fractions)
        object.__setattr__(self, "shell_edges_rho", edges)
        object.__setattr__(self, "gaussian_count_candidates", counts)
        object.__setattr__(self, "gaussian_width_scale_candidates", widths)
        object.__setattr__(self, "morse_minimum_fraction_candidates", minima)
        object.__setattr__(self, "morse_alpha_layer_candidates", alphas)
        object.__setattr__(self, "ridge_candidates", ridges)
        object.__setattr__(self, "morse_well_depth_prior_hartree", priors)

    def to_record(self) -> dict[str, Any]:
        return {
            "target_number_density_bohr3": {
                label: np.asarray(values).tolist()
                for label, values in self.target_number_density_bohr3.items()
            },
            "target_total_number_density_bohr3": (
                None
                if self.target_total_number_density_bohr3 is None
                else np.asarray(
                    self.target_total_number_density_bohr3
                ).tolist()
            ),
            "mixture_fractions": (
                None
                if self.mixture_fractions is None
                else dict(self.mixture_fractions)
            ),
            "shell_edges_rho": list(self.shell_edges_rho),
            "temperature_kelvin": float(self.temperature_kelvin),
            "update_damping": float(self.update_damping),
            "pseudocount_fraction": float(self.pseudocount_fraction),
            "minimum_blocks": int(self.minimum_blocks),
            "gaussian_count_candidates": list(self.gaussian_count_candidates),
            "gaussian_width_scale_candidates": list(
                self.gaussian_width_scale_candidates
            ),
            "morse_minimum_fraction_candidates": list(
                self.morse_minimum_fraction_candidates
            ),
            "morse_alpha_layer_candidates": list(
                self.morse_alpha_layer_candidates
            ),
            "ridge_candidates": list(self.ridge_candidates),
            "validation_stride": int(self.validation_stride),
            "morse_well_depth_prior_hartree": (
                None
                if self.morse_well_depth_prior_hartree is None
                else dict(self.morse_well_depth_prior_hartree)
            ),
            "morse_prior_strength": float(self.morse_prior_strength),
            "gaussian_amplitude_bound_factor": float(
                self.gaussian_amplitude_bound_factor
            ),
            "fit_parallel_workers": int(self.fit_parallel_workers),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "VdwWarmupConfig":
        return cls(**dict(record))


@dataclass(frozen=True)
class VdwWarmupBlockResult:
    """One accepted-geometry block update and its provisional field."""

    block_index: int
    sample_count: int
    density_by_type: Mapping[str, np.ndarray]
    bias_by_type_hartree: Mapping[str, np.ndarray]
    provisional_confinement: EllipsoidalVdwConfinement
    fit_report: Mapping[str, Any]


@dataclass(frozen=True)
class VdwSiteTypeDefinition:
    """Auditable chemical identity of one UvdW density component."""

    label: str
    element: str
    chemical_environment: str
    united_atom: bool = False

    def __post_init__(self) -> None:
        if not all(
            str(value).strip()
            for value in (self.label, self.element, self.chemical_environment)
        ):
            raise ValueError("UvdW site-type fields cannot be empty")

    def to_record(self) -> dict[str, Any]:
        return {
            "label": str(self.label),
            "element": str(self.element),
            "chemical_environment": str(self.chemical_environment),
            "united_atom": bool(self.united_atom),
        }


@dataclass(frozen=True)
class VdwCalibrationResult:
    """Frozen production confinement and complete calibration audit."""

    confinement: EllipsoidalVdwConfinement
    report: Mapping[str, Any]
    schema: str = ZAFF_VDW_CALIBRATION_SCHEMA


class VdwWarmupCalibrator:
    """Restartable iterative-Boltzmann calibration on homothetic shells.

    ``observe_block`` is valid only during warm-up.  ``freeze`` irreversibly
    closes the object and returns the production field.  This fail-closed
    state transition prevents density adaptation from leaking into production
    Monte Carlo.
    """

    def __init__(
        self,
        center_bohr: np.ndarray,
        semiaxes_bohr: np.ndarray,
        rotation: np.ndarray,
        site_types: Sequence[str],
        config: VdwWarmupConfig,
        site_type_definitions: Mapping[str, VdwSiteTypeDefinition] | None = None,
    ) -> None:
        self.center_bohr = np.asarray(center_bohr, dtype=float).reshape(3)
        self.semiaxes_bohr = np.asarray(semiaxes_bohr, dtype=float).reshape(3)
        self.rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
        self.site_types = tuple(str(value) for value in site_types)
        self.config = config
        definitions = (
            {
                label: VdwSiteTypeDefinition(label, "X", "unspecified")
                for label in sorted(set(self.site_types))
            }
            if site_type_definitions is None
            else dict(site_type_definitions)
        )
        if (
            not self.site_types
            or np.any(~np.isfinite(self.center_bohr))
            or np.any(~np.isfinite(self.semiaxes_bohr))
            or np.any(self.semiaxes_bohr <= 0.0)
            or not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1.0e-12)
            or np.linalg.det(self.rotation) < 0.0
            or any(
                label not in self.config.target_number_density_bohr3
                for label in self.site_types
            )
            or set(definitions) != set(self.site_types)
            or any(key != value.label for key, value in definitions.items())
        ):
            raise ValueError("UvdW warm-up geometry or site types are inconsistent")
        self.site_type_definitions = definitions
        shell_count = len(self.config.shell_edges_rho) - 1
        self.bias_by_type_hartree = {
            label: np.zeros(shell_count)
            for label in sorted(set(self.site_types))
        }
        self.block_count = 0
        self.sample_count = 0
        self.history: list[dict[str, Any]] = []
        self.latest_density_by_type: dict[str, np.ndarray] = {}
        self.latest_total_density = np.zeros(shell_count)
        self._frozen_result: VdwCalibrationResult | None = None

    @property
    def frozen(self) -> bool:
        return self._frozen_result is not None

    @property
    def volume_equivalent_radius_bohr(self) -> float:
        return float(np.prod(self.semiaxes_bohr) ** (1.0 / 3.0))

    @property
    def layer_depth_bohr(self) -> float:
        return self.volume_equivalent_radius_bohr * (
            1.0 - float(self.config.shell_edges_rho[0])
        )

    def observe_block(self, geometries_bohr: np.ndarray) -> VdwWarmupBlockResult:
        """Update shell biases from one SENTINEL warm-up geometry block."""

        if self.frozen:
            raise RuntimeError("UvdW calibration is frozen; production adaptation is forbidden")
        geometries = self._validated_geometries(geometries_bohr)
        density = self._shell_densities(geometries)
        self.block_count += 1
        self.sample_count += len(geometries)
        learning_rate = float(self.config.update_damping) / math.sqrt(self.block_count)
        thermal_energy = (
            K_BOLTZMANN_HARTREE_PER_KELVIN
            * float(self.config.temperature_kelvin)
        )
        for label, observed in density.items():
            target = self._target_profile(label)
            pseudocount = float(self.config.pseudocount_fraction) * target
            update = learning_rate * thermal_energy * np.log(
                (observed + pseudocount) / (target + pseudocount)
            )
            self.bias_by_type_hartree[label] += update
            # Only energy differences across the layer matter.  Anchoring the
            # innermost shell removes the unidentifiable constant.
            self.bias_by_type_hartree[label] -= self.bias_by_type_hartree[label][0]
        self.latest_density_by_type = {
            label: values.copy() for label, values in density.items()
        }
        self.latest_total_density = np.sum(
            np.stack(tuple(density.values())), axis=0
        )
        field, fit_report = self._fit_confinement()
        block_report = {
            "block_index": self.block_count,
            "sample_count": len(geometries),
            "learning_rate": learning_rate,
            "density_by_type": {
                label: values.tolist() for label, values in density.items()
            },
            "bias_by_type_hartree": {
                label: values.tolist()
                for label, values in self.bias_by_type_hartree.items()
            },
            "fit": fit_report,
        }
        self.history.append(block_report)
        return VdwWarmupBlockResult(
            block_index=self.block_count,
            sample_count=len(geometries),
            density_by_type={
                label: values.copy() for label, values in density.items()
            },
            bias_by_type_hartree={
                label: values.copy()
                for label, values in self.bias_by_type_hartree.items()
            },
            provisional_confinement=field,
            fit_report=fit_report,
        )

    def freeze(self) -> VdwCalibrationResult:
        """Fit and irreversibly freeze the production UvdW field."""

        if self._frozen_result is not None:
            return self._frozen_result
        if self.block_count < int(self.config.minimum_blocks):
            raise RuntimeError(
                "UvdW calibration cannot freeze before the required warm-up blocks"
            )
        field, fit_report = self._fit_confinement()
        report = {
            "schema": ZAFF_VDW_CALIBRATION_SCHEMA,
            "status": "FROZEN_FOR_PRODUCTION",
            "adaptation_contract": "WARMUP_ONLY_NO_PRODUCTION_UPDATES",
            "block_count": self.block_count,
            "sample_count": self.sample_count,
            "site_types": list(self.site_types),
            "site_type_definitions": {
                label: definition.to_record()
                for label, definition in self.site_type_definitions.items()
            },
            "config": self.config.to_record(),
            "shell_volumes_bohr3": self._shell_volumes().tolist(),
            "final_density_by_type": {
                label: values.tolist()
                for label, values in self.latest_density_by_type.items()
            },
            "final_total_density_bohr3": self.latest_total_density.tolist(),
            "final_bias_by_type_hartree": {
                label: values.tolist()
                for label, values in self.bias_by_type_hartree.items()
            },
            "fit": fit_report,
            "confinement": vdw_confinement_to_record(field),
            "history": list(self.history),
        }
        self._frozen_result = VdwCalibrationResult(field, report)
        return self._frozen_result

    def to_record(self) -> dict[str, Any]:
        """Serialize a restartable warm-up or frozen calibration state."""

        return {
            "schema": ZAFF_VDW_WARMUP_SCHEMA,
            "status": "FROZEN" if self.frozen else "WARMUP",
            "center_bohr": self.center_bohr.tolist(),
            "semiaxes_bohr": self.semiaxes_bohr.tolist(),
            "rotation": self.rotation.tolist(),
            "site_types": list(self.site_types),
            "site_type_definitions": {
                label: definition.to_record()
                for label, definition in self.site_type_definitions.items()
            },
            "config": self.config.to_record(),
            "block_count": self.block_count,
            "sample_count": self.sample_count,
            "bias_by_type_hartree": {
                label: values.tolist()
                for label, values in self.bias_by_type_hartree.items()
            },
            "latest_density_by_type": {
                label: values.tolist()
                for label, values in self.latest_density_by_type.items()
            },
            "latest_total_density_bohr3": self.latest_total_density.tolist(),
            "history": list(self.history),
            "frozen_report": (
                None if self._frozen_result is None else dict(self._frozen_result.report)
            ),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "VdwWarmupCalibrator":
        """Restore a checkpoint while retaining the production freeze guard."""

        if str(record.get("schema", "")) != ZAFF_VDW_WARMUP_SCHEMA:
            raise ValueError("unsupported UvdW warm-up checkpoint")
        calibrator = cls(
            np.asarray(record["center_bohr"], dtype=float),
            np.asarray(record["semiaxes_bohr"], dtype=float),
            np.asarray(record["rotation"], dtype=float),
            tuple(str(value) for value in record["site_types"]),
            VdwWarmupConfig.from_record(dict(record["config"])),
            {
                str(label): VdwSiteTypeDefinition(**dict(value))
                for label, value in dict(
                    record.get("site_type_definitions", {})
                ).items()
            }
            or None,
        )
        calibrator.block_count = int(record.get("block_count", 0))
        calibrator.sample_count = int(record.get("sample_count", 0))
        shell_count = len(calibrator.config.shell_edges_rho) - 1
        calibrator.bias_by_type_hartree = {
            str(label): np.asarray(values, dtype=float).reshape(shell_count)
            for label, values in dict(record["bias_by_type_hartree"]).items()
        }
        calibrator.latest_density_by_type = {
            str(label): np.asarray(values, dtype=float).reshape(shell_count)
            for label, values in dict(record.get("latest_density_by_type", {})).items()
        }
        calibrator.latest_total_density = np.asarray(
            record.get(
                "latest_total_density_bohr3",
                np.sum(
                    np.stack(tuple(calibrator.latest_density_by_type.values())),
                    axis=0,
                )
                if calibrator.latest_density_by_type
                else np.zeros(shell_count),
            ),
            dtype=float,
        ).reshape(shell_count)
        calibrator.history = [
            dict(item) for item in record.get("history", [])
        ]
        frozen_report = record.get("frozen_report")
        if frozen_report is not None:
            if str(dict(frozen_report).get("status", "")) != "FROZEN_FOR_PRODUCTION":
                raise ValueError("invalid frozen UvdW warm-up checkpoint")
            field, _fit = calibrator._fit_confinement()
            calibrator._frozen_result = VdwCalibrationResult(
                field, dict(frozen_report)
            )
        return calibrator

    def _validated_geometries(self, geometries_bohr: np.ndarray) -> np.ndarray:
        geometries = np.asarray(geometries_bohr, dtype=float)
        if (
            geometries.ndim != 3
            or geometries.shape[1:] != (len(self.site_types), 3)
            or len(geometries) == 0
            or np.any(~np.isfinite(geometries))
        ):
            raise ValueError("UvdW warm-up block must have shape (nframe, nsite, 3)")
        local = (
            geometries - self.center_bohr[None, None, :]
        ) @ self.rotation
        rho = np.linalg.norm(
            local / self.semiaxes_bohr[None, None, :], axis=2
        )
        if np.any(rho >= 1.0):
            raise ValueError("every UvdW warm-up site must lie inside confinement")
        return geometries

    def _shell_volumes(self) -> np.ndarray:
        edges = np.asarray(self.config.shell_edges_rho, dtype=float)
        return (
            (4.0 * math.pi / 3.0)
            * float(np.prod(self.semiaxes_bohr))
            * (edges[1:] ** 3 - edges[:-1] ** 3)
        )

    def _target_profile(self, label: str) -> np.ndarray:
        target = np.asarray(
            self.config.target_number_density_bohr3[label], dtype=float
        )
        if (
            self.config.target_total_number_density_bohr3 is not None
            and self.config.mixture_fractions is not None
        ):
            return np.asarray(
                self.config.target_total_number_density_bohr3, dtype=float
            ) * float(self.config.mixture_fractions[label])
        return target

    def _shell_densities(
        self, geometries_bohr: np.ndarray
    ) -> dict[str, np.ndarray]:
        local = (
            geometries_bohr - self.center_bohr[None, None, :]
        ) @ self.rotation
        rho = np.linalg.norm(
            local / self.semiaxes_bohr[None, None, :], axis=2
        )
        volumes = self._shell_volumes()
        densities: dict[str, np.ndarray] = {}
        labels = np.asarray(self.site_types, dtype=object)
        for label in self.bias_by_type_hartree:
            selected = rho[:, labels == label].reshape(-1)
            counts, _edges = np.histogram(
                selected, bins=np.asarray(self.config.shell_edges_rho)
            )
            densities[label] = counts.astype(float) / (
                len(geometries_bohr) * volumes
            )
        return densities

    def _fit_confinement(
        self,
    ) -> tuple[EllipsoidalVdwConfinement, dict[str, Any]]:
        morse: dict[str, tuple[float, float, float]] = {}
        gaussians: dict[str, tuple[tuple[float, float, float], ...]] = {}
        reports: dict[str, Any] = {}
        edges = np.asarray(self.config.shell_edges_rho, dtype=float)
        rho_centers = 0.5 * (edges[:-1] + edges[1:])
        depths = self.volume_equivalent_radius_bohr * (1.0 - rho_centers)
        shell_weights = self._shell_volumes()
        shell_weights /= float(np.mean(shell_weights))
        for label, bias in self.bias_by_type_hartree.items():
            selected = _select_morse_gaussian_model(
                label,
                depths,
                bias,
                shell_weights,
                self.layer_depth_bohr,
                self.config,
            )
            morse[label] = selected["morse_parameters"]
            gaussians[label] = selected["gaussian_terms"]
            reports[label] = selected["report"]
        field = EllipsoidalVdwConfinement(
            self.center_bohr,
            self.semiaxes_bohr,
            self.rotation,
            morse,
            gaussians,
            self.layer_depth_bohr,
        )
        return field, {"by_type": reports}


def calibrate_vdw_from_warmup_blocks(
    calibrator: VdwWarmupCalibrator,
    geometry_blocks_bohr: Iterable[np.ndarray],
) -> VdwCalibrationResult:
    """Consume SENTINEL warm-up blocks and return one frozen production field."""

    for block in geometry_blocks_bohr:
        calibrator.observe_block(block)
    return calibrator.freeze()


def attach_calibrated_vdw_confinement(
    model: Mapping[str, Any], result: VdwCalibrationResult
) -> dict[str, Any]:
    """Attach a frozen field and its calibration audit to a ZAFF model."""

    if (
        result.schema != ZAFF_VDW_CALIBRATION_SCHEMA
        or str(result.report.get("status", "")) != "FROZEN_FOR_PRODUCTION"
    ):
        raise ValueError("only a frozen UvdW calibration may enter a ZAFF model")
    payload = attach_vdw_confinement(model, result.confinement)
    payload["vdw_calibration"] = dict(result.report)
    return payload


def _select_morse_gaussian_model(
    label: str,
    depths: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    layer_depth: float,
    config: VdwWarmupConfig,
) -> dict[str, Any]:
    from scipy.optimize import lsq_linear

    count = len(depths)
    validation = np.arange(count) % int(config.validation_stride) == (
        int(config.validation_stride) - 1
    )
    if np.count_nonzero(~validation) < max(config.gaussian_count_candidates) + 1:
        validation[:] = False
        validation[-1] = True
    training = ~validation
    target_scale = max(float(np.max(np.abs(target))), 1.0e-10)
    amplitude_bound = (
        float(config.gaussian_amplitude_bound_factor) * target_scale
    )
    morse_prior = (
        target_scale
        if config.morse_well_depth_prior_hartree is None
        else float(config.morse_well_depth_prior_hartree[label])
    )
    specifications = []
    for gaussian_count in config.gaussian_count_candidates:
        spacing = layer_depth / float(gaussian_count)
        centers = (np.arange(gaussian_count, dtype=float) + 0.5) * spacing
        for width_scale in config.gaussian_width_scale_candidates:
            sigma = max(1.0e-8, float(width_scale) * spacing)
            exponent = 1.0 / (2.0 * sigma**2)
            for minimum_fraction in config.morse_minimum_fraction_candidates:
                minimum = float(minimum_fraction) * layer_depth
                for alpha_layer in config.morse_alpha_layer_candidates:
                    alpha = float(alpha_layer) / layer_depth
                    for ridge in config.ridge_candidates:
                        specifications.append(
                            (
                                gaussian_count,
                                centers,
                                exponent,
                                minimum,
                                alpha,
                                float(ridge),
                            )
                        )

    def evaluate_candidate(specification):
        gaussian_count, centers, exponent, minimum, alpha, ridge = specification
        design = _morse_gaussian_design(
            depths,
            layer_depth,
            minimum,
            alpha,
            centers,
            exponent,
        )
        coefficients = _bounded_ridge_fit(
            design[training],
            target[training],
            weights[training],
            ridge,
            lsq_linear,
            morse_prior=morse_prior,
            morse_prior_strength=float(config.morse_prior_strength),
            gaussian_amplitude_bound=amplitude_bound,
        )
        residual = design[validation] @ coefficients - target[validation]
        validation_rmse = float(
            np.sqrt(np.average(residual**2, weights=weights[validation]))
        )
        return {
            "score": (validation_rmse, gaussian_count, ridge),
            "gaussian_count": gaussian_count,
            "centers": centers,
            "exponent": exponent,
            "minimum": minimum,
            "alpha": alpha,
            "ridge": ridge,
            "validation_rmse": validation_rmse,
        }

    workers = min(int(config.fit_parallel_workers), len(specifications))
    if workers == 1:
        candidates = tuple(map(evaluate_candidate, specifications))
    else:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="architect-vdw-fit",
        ) as executor:
            candidates = tuple(executor.map(evaluate_candidate, specifications))
    best = min(candidates, key=lambda candidate: candidate["score"])
    final_design = _morse_gaussian_design(
        depths,
        layer_depth,
        best["minimum"],
        best["alpha"],
        best["centers"],
        best["exponent"],
    )
    coefficients = _bounded_ridge_fit(
        final_design,
        target,
        weights,
        best["ridge"],
        lsq_linear,
        morse_prior=morse_prior,
        morse_prior_strength=float(config.morse_prior_strength),
        gaussian_amplitude_bound=amplitude_bound,
    )
    fitted = final_design @ coefficients
    weighted_rmse = float(
        np.sqrt(np.average((fitted - target) ** 2, weights=weights))
    )
    gaussian_terms = tuple(
        (
            float(amplitude),
            float(center),
            float(best["exponent"]),
        )
        for amplitude, center in zip(
            coefficients[1:], best["centers"], strict=True
        )
    )
    uncertainty = _linear_fit_uncertainty(
        final_design,
        target,
        weights,
        coefficients,
        float(best["ridge"]),
    )
    return {
        "morse_parameters": (
            float(coefficients[0]),
            float(best["minimum"]),
            float(best["alpha"]),
        ),
        "gaussian_terms": gaussian_terms,
        "report": {
            "selection": "DETERMINISTIC_HELD_OUT_GRID",
            "gaussian_count": int(best["gaussian_count"]),
            "gaussian_width_scale": float(
                math.sqrt(1.0 / (2.0 * best["exponent"]))
                / (layer_depth / best["gaussian_count"])
            ),
            "morse_minimum_depth_bohr": float(best["minimum"]),
            "morse_alpha_per_bohr": float(best["alpha"]),
            "ridge": float(best["ridge"]),
            "validation_rmse_hartree": float(best["validation_rmse"]),
            "full_weighted_rmse_hartree": weighted_rmse,
            "target_bias_hartree": target.tolist(),
            "fitted_bias_hartree": fitted.tolist(),
            "physical_constraints": {
                "morse_well_depth_nonnegative": True,
                "gaussian_amplitude_bound_hartree": amplitude_bound,
                "compact_C2_asymptote": True,
            },
            "morse_prior_hartree": morse_prior,
            "morse_prior_strength": float(config.morse_prior_strength),
            "parameter_standard_errors": uncertainty["standard_errors"],
            "normal_matrix_condition_number": uncertainty[
                "condition_number"
            ],
        },
    }


def _morse_gaussian_design(
    depths: np.ndarray,
    layer_depth: float,
    minimum: float,
    alpha: float,
    centers: np.ndarray,
    exponent: float,
) -> np.ndarray:
    x = depths / layer_depth
    switch = 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5
    exponential = np.exp(-alpha * (depths - minimum))
    morse_unit = exponential**2 - 2.0 * exponential
    gaussian = np.exp(
        -float(exponent) * (depths[:, None] - centers[None, :]) ** 2
    )
    return switch[:, None] * np.column_stack((morse_unit, gaussian))


def _bounded_ridge_fit(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    ridge: float,
    solver: Any,
    *,
    morse_prior: float,
    morse_prior_strength: float,
    gaussian_amplitude_bound: float,
) -> np.ndarray:
    scale = np.sqrt(np.asarray(weights, dtype=float))
    weighted_design = np.asarray(design, dtype=float) * scale[:, None]
    weighted_target = np.asarray(target, dtype=float) * scale
    column_norm = np.linalg.norm(weighted_design, axis=0)
    column_norm = np.maximum(column_norm, 1.0e-14)
    normalized = weighted_design / column_norm[None, :]
    if ridge > 0.0:
        normalized = np.vstack(
            (normalized, math.sqrt(ridge) * np.eye(normalized.shape[1]))
        )
        weighted_target = np.concatenate(
            (weighted_target, np.zeros(normalized.shape[1]))
        )
    if morse_prior_strength > 0.0:
        prior_row = np.zeros((1, normalized.shape[1]))
        prior_row[0, 0] = math.sqrt(morse_prior_strength) / column_norm[0]
        normalized = np.vstack((normalized, prior_row))
        weighted_target = np.concatenate(
            (weighted_target, [math.sqrt(morse_prior_strength) * morse_prior])
        )
    lower = np.full(normalized.shape[1], -np.inf)
    upper = np.full(normalized.shape[1], np.inf)
    lower[0] = 0.0
    lower[1:] = -float(gaussian_amplitude_bound) * column_norm[1:]
    upper[1:] = float(gaussian_amplitude_bound) * column_norm[1:]
    unconstrained, *_ = np.linalg.lstsq(
        normalized,
        weighted_target,
        rcond=1.0e-12,
    )
    finite_lower = np.where(np.isfinite(lower), np.abs(lower), 0.0)
    finite_upper = np.where(np.isfinite(upper), np.abs(upper), 0.0)
    feasibility_tolerance = 1.0e-12 * np.maximum(
        1.0,
        np.maximum(finite_lower, finite_upper),
    )
    feasible = np.all(unconstrained >= lower - feasibility_tolerance) and np.all(
        unconstrained <= upper + feasibility_tolerance
    )
    if feasible:
        return np.asarray(unconstrained, dtype=float) / column_norm
    result = solver(
        normalized,
        weighted_target,
        bounds=(lower, upper),
        method="trf",
        lsmr_tol="auto",
    )
    if not result.success:
        raise RuntimeError(f"UvdW Morse--Gaussian fit failed: {result.message}")
    return np.asarray(result.x, dtype=float) / column_norm


def _linear_fit_uncertainty(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    coefficients: np.ndarray,
    ridge: float,
) -> dict[str, Any]:
    weighted = np.asarray(design, dtype=float) * np.sqrt(weights)[:, None]
    residual = (design @ coefficients - target) * np.sqrt(weights)
    degrees = max(1, len(target) - design.shape[1])
    variance = float(np.dot(residual, residual) / degrees)
    normal = weighted.T @ weighted + float(ridge) * np.eye(design.shape[1])
    covariance = variance * np.linalg.pinv(normal, rcond=1.0e-12)
    return {
        "standard_errors": np.sqrt(
            np.maximum(0.0, np.diag(covariance))
        ).tolist(),
        "condition_number": float(np.linalg.cond(normal)),
    }


__all__ = [
    "ZAFF_VDW_CALIBRATION_SCHEMA",
    "ZAFF_VDW_WARMUP_SCHEMA",
    "K_BOLTZMANN_HARTREE_PER_KELVIN",
    "VdwCalibrationResult",
    "VdwSiteTypeDefinition",
    "VdwWarmupBlockResult",
    "VdwWarmupCalibrator",
    "VdwWarmupConfig",
    "attach_calibrated_vdw_confinement",
    "calibrate_vdw_from_warmup_blocks",
]
