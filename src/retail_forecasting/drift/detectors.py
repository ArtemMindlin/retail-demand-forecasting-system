from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DriftResult:
    """Container for drift detection results."""

    is_drift: bool
    score: float
    threshold: float
    detected_at_index: int | None = None


class PageHinkleyDetector:
    """Implementation of the Page-Hinkley test for concept drift detection.

    The Page-Hinkley (PH) test is a sequential analysis technique used for
    monitoring change point detection. It is designed to detect a change in
    the mean of a Gaussian signal.

    In the context of forecasting, we use it to monitor the model error.
    An increase in the cumulative deviation of the error suggests that
    the model's performance is degrading (Concept Drift).
    """

    def __init__(
        self, threshold: float = 30.0, delta: float = 0.005, min_instances: int = 30
    ):
        """Initialize the detector.

        Args:
            threshold: Cumulative deviation threshold (lambda). Higher values
                make the detector more robust to noise but slower to detect.
            delta: Magnitude of changes that are allowed.
            min_instances: Minimum number of samples before detecting drift.
        """
        self.threshold = threshold
        self.delta = delta
        self.min_instances = min_instances

        self.reset()

    def reset(self):
        """Reset the detector state."""
        self.sum_errors = 0.0
        self.n = 0
        self.min_sum_errors = 0.0
        self.is_drift = False

    def update(self, error: float) -> DriftResult:
        """Update the detector with a new error observation.

        Args:
            error: The current error metric (e.g., MAE or Pinball loss).

        Returns:
            A DriftResult object indicating if drift was detected.
        """
        self.n += 1

        # Calculate the average error so far
        avg_error = self.sum_errors / self.n if self.n > 0 else 0.0
        self.sum_errors += error

        # Cumulative deviation
        # We look for an increase in the mean error
        self.sum_errors_diff = error - avg_error - self.delta
        # In PH, we track the cumulative sum and its minimum
        # This is a simplified version focusing on performance degradation
        # (increasing errors).

        # Simplified PH logic for increasing signal:
        # m_n = sum_{i=1}^n (x_i - mean_i - delta)
        # M_n = min(m_1 ... m_n)
        # PH_n = m_n - M_n
        # if PH_n > threshold -> Drift!

        # For simplicity in this implementation, we'll use a cumulative
        # absolute deviation approach often used in drift monitoring.
        self.sum_errors += error - avg_error - self.delta
        if self.sum_errors < self.min_sum_errors:
            self.min_sum_errors = self.sum_errors

        ph_stat = self.sum_errors - self.min_sum_errors

        if self.n > self.min_instances and ph_stat > self.threshold:
            self.is_drift = True
            return DriftResult(
                is_drift=True,
                score=ph_stat,
                threshold=self.threshold,
                detected_at_index=self.n,
            )

        return DriftResult(is_drift=False, score=ph_stat, threshold=self.threshold)
