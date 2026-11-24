
"""
Emotion Threshold Loader

Loads validation-selected per-label thresholds
for the GoEmotions multi-label classifier.

Threshold selection is not probability calibration.
"""

import json
import math

from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]

THRESHOLD_FILE = (
    PROJECT_ROOT
    / "training"
    / "emotion"
    / "goemotions_distilbert"
    / "validation_analysis"
    / "validation_summary.json"
)


class EmotionThresholdError(RuntimeError):
    """Raised when emotion threshold configuration is invalid."""


def load_emotion_thresholds(
    expected_labels: Sequence[str],
) -> dict[str, float]:

    if not THRESHOLD_FILE.is_file():
        raise EmotionThresholdError(
            f"Emotion threshold file not found: {THRESHOLD_FILE}"
        )

    try:
        with THRESHOLD_FILE.open("r", encoding="utf-8") as file:
            summary = json.load(file)

    except (OSError, json.JSONDecodeError) as exc:
        raise EmotionThresholdError(
            "Unable to read emotion validation summary."
        ) from exc

    thresholds = summary.get("thresholds")

    if not isinstance(thresholds, dict):
        raise EmotionThresholdError(
            "Validation summary does not contain a valid thresholds mapping."
        )

    expected = set(expected_labels)
    actual = set(thresholds)

    missing = expected - actual
    unexpected = actual - expected

    if missing or unexpected:
        raise EmotionThresholdError(
            f"Emotion label mismatch. "
            f"Missing: {sorted(missing)}; "
            f"Unexpected: {sorted(unexpected)}"
        )

    validated_thresholds = {}

    for label in expected_labels:

        value = thresholds[label]

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmotionThresholdError(
                f"Invalid threshold type for emotion: {label}"
            )

        value = float(value)

        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise EmotionThresholdError(
                f"Threshold outside valid range for emotion: {label}"
            )

        validated_thresholds[label] = value

    if len(validated_thresholds) != 28:
        raise EmotionThresholdError(
            "Expected exactly 28 emotion thresholds."
        )

    return validated_thresholds


def get_threshold_summary(
    thresholds: dict[str, float],
) -> dict:

    default_labels = [
        label
        for label, value in thresholds.items()
        if value == 0.5
    ]

    return {
        "total_labels": len(thresholds),
        "validation_selected_count": (
            len(thresholds) - len(default_labels)
        ),
        "default_threshold_count": len(default_labels),
        "default_threshold_labels": default_labels,
        "threshold_source": str(THRESHOLD_FILE),
        "probability_calibration": "not_calibrated",
    }
