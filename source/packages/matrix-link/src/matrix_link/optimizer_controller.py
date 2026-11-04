"""Multi-backend trust-region controller used by LINK.

This module contains no molecular-geometry or quantum-chemistry code.  It is
the single contract for deciding whether a proposed model step is usable and
for updating the trust region after the actual surface evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ControllerSettings:
    acceptance_threshold: float = 0.10
    expand_threshold: float = 0.75
    shrink_threshold: float = 0.10
    # Fletcher-style trust-region factors used by the GDIIS/RFO controller.
    # The previous 0.5 contraction was unnecessarily abrupt for an accepted
    # reference step; 0.7/1.2 are the conservative values commonly used in
    # electronic-structure geometry optimization.
    expansion_factor: float = 1.2
    contraction_factor: float = 0.7
    min_radius: float = 1.0e-4
    max_radius: float = 0.3
    energy_noise: float = 1.0e-7
    energy_tolerance: float = 1.0e-6


@dataclass(frozen=True)
class TrialDecision:
    accepted: bool
    ratio: float
    predicted_reduction: float
    actual_reduction: float
    reason: str


class TrustRegionController:
    """Numerically explicit trust-region acceptance and radius policy.

    The ratio is meaningful only when the model predicts a nonzero reduction.
    Energy noise is treated as an uncertainty band, not as permission to
    accept arbitrarily poor model steps.  The radius and boundary fraction are
    defined by the aligned Cartesian atomic-RMS displacement obtained after
    the authoritative nonlinear back-transform, independently of the native
    optimizer-coordinate representation.
    """

    def __init__(self, settings: ControllerSettings) -> None:
        if not 0.0 < settings.acceptance_threshold:
            raise ValueError("invalid trust-region acceptance thresholds")
        if not 0.0 < settings.expand_threshold:
            raise ValueError("expand_threshold must be positive")
        if not 0.0 < settings.contraction_factor < 1.0:
            raise ValueError("contraction_factor must lie in (0, 1)")
        if settings.expansion_factor <= 1.0:
            raise ValueError("expansion_factor must exceed one")
        if settings.min_radius <= 0.0 or settings.max_radius < settings.min_radius:
            raise ValueError("invalid trust-region radius bounds")
        self.settings = settings

    @property
    def noise_floor(self) -> float:
        return max(
            5.0 * float(self.settings.energy_noise),
            0.1 * float(self.settings.energy_tolerance),
        )

    def assess(
        self,
        predicted_reduction: float,
        actual_reduction: float,
        *,
        stationary_point: str = "minimum",
        current_stationarity_norm: float | None = None,
        trial_stationarity_norm: float | None = None,
        stationarity_resolution: float = 0.0,
        near_stationary_step: bool = False,
    ) -> TrialDecision:
        predicted = float(predicted_reduction)
        actual = float(actual_reduction)
        if not math.isfinite(predicted) or not math.isfinite(actual):
            return TrialDecision(False, -math.inf, predicted, actual, "nonfinite_reduction")
        if stationary_point == "transition_state":
            # l103.F does not reject an otherwise valid DXRFO geometry from
            # the predicted/actual energy ratio.  UpdDXM uses that raw ratio
            # only to set the *next* trust radius.  Keep the unused generic
            # stationarity arguments in the public signature for minimum/TS
            # call-site compatibility.
            del current_stationarity_norm, trial_stationarity_norm
            del stationarity_resolution, near_stationary_step
            ratio = math.nan if predicted == 0.0 else actual / predicted
            return TrialDecision(True, ratio, predicted, actual, "gdv_dxrfo_step")
        if abs(actual) <= float(self.settings.energy_tolerance):
            ratio = actual / predicted if predicted > 0.0 else 0.0
            return TrialDecision(
                True,
                ratio,
                predicted,
                actual,
                "accepted_energy_plateau",
            )
        if predicted <= 0.0:
            return TrialDecision(False, -math.inf, predicted, actual, "nonpositive_prediction")
        ratio = actual / predicted
        if actual < -self.noise_floor:
            return TrialDecision(False, ratio, predicted, actual, "energy_increase")
        accepted = ratio >= self.settings.acceptance_threshold
        return TrialDecision(
            accepted, ratio, predicted, actual,
            "accepted" if accepted else "low_model_quality",
        )

    def radius_after(
        self,
        radius: float,
        decision: TrialDecision,
        *,
        step_fraction: float,
    ) -> float:
        current = float(radius)
        fraction = float(step_fraction)
        if not decision.accepted or decision.ratio < self.settings.shrink_threshold:
            current *= self.settings.contraction_factor
        elif decision.ratio >= self.settings.expand_threshold and fraction >= 0.8:
            current *= self.settings.expansion_factor
        return min(self.settings.max_radius, max(self.settings.min_radius, current))

    def radius_after_rejection(
        self,
        radius: float,
        *,
        realized_step: float,
    ) -> float:
        """Contract from the smaller of the radius and rejected realized step.

        This is the rejection model used by geomeTRIC: a trial that did not
        reach the trust boundary must still reduce the next radius below the
        step that was actually rejected.  Otherwise a nonlinear optimizer can
        regenerate the same rejected geometry after merely halving a larger
        inactive radius.
        """

        current = float(radius)
        realized = float(realized_step)
        contraction_base = current
        if math.isfinite(realized) and realized > 0.0:
            contraction_base = min(current, realized)
        contracted = self.settings.contraction_factor * contraction_base
        return min(
            self.settings.max_radius,
            max(self.settings.min_radius, contracted),
        )
