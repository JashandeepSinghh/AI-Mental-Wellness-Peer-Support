
"""
Confidence and Uncertainty Configuration
=========================================

Central configuration for confidence estimation across
the eight analytical modules.

This module calculates descriptive model confidence and
uncertainty indicators. It does not establish clinical
certainty or guarantee prediction correctness.
"""

from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "confidence"
)


# ============================================================
# 2. MODULE INFORMATION
# ============================================================

MODULE_NAME = "Confidence and Uncertainty Estimation"

MODULE_VERSION = "1.0.0"


# ============================================================
# 3. SUPPORTED ANALYTICAL MODULES
# ============================================================

SUPPORTED_MODULES = {

    "emotion": {

        "model": "DistilBERT",

        "output_type": "multi_label",

        "activation": "sigmoid"
    },

    "sentiment": {

        "model": "DistilBERT",

        "output_type": "multi_class",

        "activation": "softmax"
    },

    "sentiment_bilstm": {

        "model": "BiLSTM",

        "output_type": "multi_class",

        "activation": "softmax"
    },

    "stress": {

        "model": "DistilBERT",

        "output_type": "multi_class",

        "activation": "softmax"
    },

    "loneliness": {

        "model": "DistilBERT",

        "output_type": "multi_class",

        "activation": "softmax"
    },

    "rumination": {

        "model": "Rule-based",

        "output_type": "rule_based",

        "activation": None
    },

    "risk": {

        "model": "Context-aware rules",

        "output_type": "rule_based",

        "activation": None
    },

    "vad": {

        "model": "NRC VAD Lexicon",

        "output_type": "lexicon_based",

        "activation": None
    }
}


# ============================================================
# 4. PROBABILITY VALIDATION
# ============================================================

MIN_PROBABILITY = 0.0

MAX_PROBABILITY = 1.0

PROBABILITY_TOLERANCE = 1e-5


# ============================================================
# 5. MULTI-CLASS CONFIDENCE
# ============================================================

ENABLE_MAX_PROBABILITY = True

ENABLE_PREDICTION_MARGIN = True

ENABLE_ENTROPY = True

ENABLE_NORMALIZED_ENTROPY = True

ENABLE_TOP_K_PREDICTIONS = True

DEFAULT_TOP_K = 3


# ============================================================
# 6. MULTI-LABEL CONFIDENCE
# ============================================================

ENABLE_PER_LABEL_CONFIDENCE = True

ENABLE_BINARY_ENTROPY = True

ENABLE_MULTI_LABEL_UNCERTAINTY = True

MULTI_LABEL_THRESHOLD = 0.5


# ============================================================
# 7. ENTROPY CONFIGURATION
# ============================================================

ENTROPY_LOG_BASE = 2

ENTROPY_EPSILON = 1e-12

NORMALIZE_ENTROPY = True


# ============================================================
# 8. CONFIDENCE INTERPRETATION BANDS
# ============================================================

# These are descriptive engineering bands, not
# statistically calibrated probability guarantees.

CONFIDENCE_BANDS = {

    "low": {

        "minimum": 0.0,

        "maximum": 0.50
    },

    "moderate": {

        "minimum": 0.50,

        "maximum": 0.75
    },

    "high": {

        "minimum": 0.75,

        "maximum": 1.0
    }
}


# ============================================================
# 9. UNCERTAINTY INTERPRETATION BANDS
# ============================================================

UNCERTAINTY_BANDS = {

    "low": {

        "minimum": 0.0,

        "maximum": 0.25
    },

    "moderate": {

        "minimum": 0.25,

        "maximum": 0.60
    },

    "high": {

        "minimum": 0.60,

        "maximum": 1.0
    }
}


# ============================================================
# 10. RULE-BASED UNCERTAINTY
# ============================================================

ENABLE_RULE_EVIDENCE_REPORTING = True

ENABLE_CONTEXTUAL_AMBIGUITY_REPORTING = True

ENABLE_VAD_COVERAGE_REPORTING = True

# Rule-based outputs must not be assigned invented
# probability scores.

RULE_BASED_PROBABILITY_AVAILABLE = False


# ============================================================
# 11. CALIBRATION STATUS
# ============================================================

CALIBRATION_STATUS = "not_calibrated"

CALIBRATED_PROBABILITY_CLAIM = False

ENABLE_CALIBRATION_METRICS = False

# Calibration evaluation will be a later dedicated
# improvement task, after project completion.


# ============================================================
# 12. SAFETY SETTINGS
# ============================================================

CLINICAL_CERTAINTY_ESTABLISHED = False

ACTUAL_RISK_CONFIRMED = False

SAFETY_PROBABILITY_AVAILABLE = False

RISK_CONFIDENCE_MUST_NOT_OVERRIDE_HUMAN_REVIEW = True

NO_PREDICTION_MUST_NOT_MEAN_SAFE = True


# ============================================================
# 13. TEXT AND INPUT VALIDATION
# ============================================================

MIN_CLASS_COUNT = 2

MAX_CLASS_COUNT = 1000

MIN_TEXT_LENGTH = 1

MAX_TEXT_LENGTH = 20000

ROUNDING_PRECISION = 6


# ============================================================
# 14. OUTPUT CONFIGURATION
# ============================================================

OUTPUT_FIELDS = [

    "module_name",

    "module_version",

    "output_type",

    "confidence_available",

    "confidence_metrics",

    "uncertainty_metrics",

    "interpretation",

    "calibration_status",

    "warnings"
]


# ============================================================
# 15. DISCLAIMER
# ============================================================

DISCLAIMER = (

    "Confidence and uncertainty values describe the behavior "
    "of analytical models or the availability of linguistic "
    "evidence. Raw model probabilities are not necessarily "
    "calibrated. These values do not establish clinical "
    "certainty, diagnosis, actual psychological state or "
    "student safety."
)


# ============================================================
# 16. CONFIGURATION VALIDATION
# ============================================================

def validate_config():

    errors = []

    # --------------------------------------------------------
    # A. Validate module registry
    # --------------------------------------------------------

    if not SUPPORTED_MODULES:

        errors.append(
            "Supported module registry cannot be empty."
        )

    required_modules = {

        "emotion",

        "sentiment",

        "sentiment_bilstm",

        "stress",

        "loneliness",

        "rumination",

        "risk",

        "vad"
    }

    if set(SUPPORTED_MODULES) != required_modules:

        errors.append(
            "Supported module registry is incomplete or invalid."
        )

    # --------------------------------------------------------
    # B. Validate probability range
    # --------------------------------------------------------

    if MIN_PROBABILITY != 0.0:

        errors.append(
            "Minimum probability must be zero."
        )

    if MAX_PROBABILITY != 1.0:

        errors.append(
            "Maximum probability must be one."
        )

    if MIN_PROBABILITY >= MAX_PROBABILITY:

        errors.append(
            "Invalid probability range."
        )

    # --------------------------------------------------------
    # C. Validate top-K
    # --------------------------------------------------------

    if DEFAULT_TOP_K < 1:

        errors.append(
            "Default top-K must be at least one."
        )

    # --------------------------------------------------------
    # D. Validate entropy
    # --------------------------------------------------------

    if ENTROPY_EPSILON <= 0:

        errors.append(
            "Entropy epsilon must be positive."
        )

    if ENTROPY_LOG_BASE != 2:

        errors.append(
            "Entropy logarithm base must be two."
        )

    # --------------------------------------------------------
    # E. Validate confidence bands
    # --------------------------------------------------------

    if CONFIDENCE_BANDS["low"]["minimum"] != 0.0:

        errors.append(
            "Confidence bands must start at zero."
        )

    if CONFIDENCE_BANDS["high"]["maximum"] != 1.0:

        errors.append(
            "Confidence bands must end at one."
        )

    # --------------------------------------------------------
    # F. Validate uncertainty bands
    # --------------------------------------------------------

    if UNCERTAINTY_BANDS["low"]["minimum"] != 0.0:

        errors.append(
            "Uncertainty bands must start at zero."
        )

    if UNCERTAINTY_BANDS["high"]["maximum"] != 1.0:

        errors.append(
            "Uncertainty bands must end at one."
        )

    # --------------------------------------------------------
    # G. Validate calibration policy
    # --------------------------------------------------------

    if CALIBRATION_STATUS != "not_calibrated":

        errors.append(
            "Current calibration status must be not_calibrated."
        )

    if CALIBRATED_PROBABILITY_CLAIM:

        errors.append(
            "Unvalidated calibrated probability claims are prohibited."
        )

    # --------------------------------------------------------
    # H. Validate safety policy
    # --------------------------------------------------------

    if SAFETY_PROBABILITY_AVAILABLE:

        errors.append(
            "A safety probability must not be exposed."
        )

    if not RISK_CONFIDENCE_MUST_NOT_OVERRIDE_HUMAN_REVIEW:

        errors.append(
            "Risk confidence must never override human review."
        )

    if not NO_PREDICTION_MUST_NOT_MEAN_SAFE:

        errors.append(
            "Absence of prediction must not establish safety."
        )

    # --------------------------------------------------------
    # I. Validate output directory
    # --------------------------------------------------------

    if not isinstance(OUTPUT_DIR, Path):

        errors.append(
            "Output directory must be a pathlib Path."
        )

    if errors:

        raise ValueError(
            "\n".join(errors)
        )

    return True


# ============================================================
# 17. CONFIGURATION SUMMARY
# ============================================================

def get_config_summary():

    return {

        "module_name": MODULE_NAME,

        "module_version": MODULE_VERSION,

        "supported_modules": list(
            SUPPORTED_MODULES.keys()
        ),

        "multi_class_metrics": [

            "max_probability",

            "prediction_margin",

            "entropy",

            "normalized_entropy"
        ],

        "multi_label_metrics": [

            "per_label_probability",

            "binary_entropy",

            "multi_label_uncertainty"
        ],

        "calibration_status": CALIBRATION_STATUS,

        "calibrated_probability_claim": (
            CALIBRATED_PROBABILITY_CLAIM
        ),

        "rule_based_probability_available": (
            RULE_BASED_PROBABILITY_AVAILABLE
        ),

        "safety_probability_available": (
            SAFETY_PROBABILITY_AVAILABLE
        ),

        "clinical_certainty_established": (
            CLINICAL_CERTAINTY_ESTABLISHED
        )
    }


# ============================================================
# 18. MAIN
# ============================================================

if __name__ == "__main__":

    validate_config()

    print("=" * 70)

    print("CONFIDENCE CONFIGURATION VALIDATION: PASSED")

    print("=" * 70)

    for key, value in get_config_summary().items():

        print(f"{key}: {value}")

    print("\nConfidence configuration is valid.")
