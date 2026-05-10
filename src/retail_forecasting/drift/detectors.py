from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DriftResult:
    """Container for drift detection results."""

    is_drift: bool
    score: float
    threshold: float
    detected_at_index: int | None = None


@dataclass(slots=True)
class PageHinkleyDetector:
    """Implementation of the Page-Hinkley test for concept drift detection.

    The Page-Hinkley (PH) test is a sequential analysis technique used for
    monitoring change point detection. It is designed to detect a change in
    the mean of a Gaussian signal.

    In the context of forecasting, we use it to monitor the model error.
    An increase in the cumulative deviation of the error suggests that
    the model's performance is degrading (Concept Drift).
    """

    threshold: float = 30.0
    delta: float = 0.005
    min_instances: int = 30
    cumulative_error: float = field(default=0.0, init=False)
    observations_seen: int = field(default=0, init=False)
    cumulative_deviation: float = field(default=0.0, init=False)
    min_cumulative_deviation: float = field(default=0.0, init=False)
    current_mean_error: float = field(default=0.0, init=False)
    last_deviation: float = field(default=0.0, init=False)
    drift_detected: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset the detector state."""
        self.cumulative_error = 0.0
        self.observations_seen = 0
        self.cumulative_deviation = 0.0
        self.min_cumulative_deviation = 0.0
        self.current_mean_error = 0.0
        self.last_deviation = 0.0
        self.drift_detected = False

    def update(self, error: float) -> DriftResult:
        """Update the detector with a new error observation.

        Args:
            error: The current error metric (e.g., MAE or Pinball loss).

        Returns:
            A DriftResult object indicating if drift was detected.
        """
        self.observations_seen += 1
        self.cumulative_error += error
        self.current_mean_error = self.cumulative_error / self.observations_seen

        # Simplified PH logic for a degrading metric:
        # deviations above the running mean accumulate evidence for drift.
        self.last_deviation = error - self.current_mean_error - self.delta
        self.cumulative_deviation += self.last_deviation
        if self.cumulative_deviation < self.min_cumulative_deviation:
            self.min_cumulative_deviation = self.cumulative_deviation

        ph_stat = self.cumulative_deviation - self.min_cumulative_deviation

        if self.observations_seen > self.min_instances and ph_stat > self.threshold:
            self.drift_detected = True
            return DriftResult(
                is_drift=True,
                score=ph_stat,
                threshold=self.threshold,
                detected_at_index=self.observations_seen,
            )

        return DriftResult(is_drift=False, score=ph_stat, threshold=self.threshold)
