
"""
VAD Analysis Configuration
==========================

NRC VAD Lexicon v2.1
Valence, Arousal and Dominance analysis.

This module measures linguistic signals from text.
It does not measure a person's actual psychological
or physiological state.
"""

from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "nrc_vad"
    / "standardized"
    / "nrc_vad.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "vad"
)

LEXICON_SUMMARY_PATH = (
    OUTPUT_DIR
    / "lexicon_validation_summary.json"
)


# ============================================================
# 2. MODULE INFORMATION
# ============================================================

MODULE_NAME = "Valence-Arousal-Dominance Analysis"

MODULE_VERSION = "1.0.0"

LEXICON_NAME = "NRC VAD Lexicon"

LEXICON_VERSION = "2.1"


# ============================================================
# 3. LEXICON SCHEMA
# ============================================================

REQUIRED_COLUMNS = [

    "term",

    "term_type",

    "valence",

    "arousal",

    "dominance"
]

TERM_COLUMN = "term"

TERM_TYPE_COLUMN = "term_type"

VALENCE_COLUMN = "valence"

AROUSAL_COLUMN = "arousal"

DOMINANCE_COLUMN = "dominance"


# ============================================================
# 4. VAD SCORE CONFIGURATION
# ============================================================

MIN_VAD_SCORE = -1.0

MAX_VAD_SCORE = 1.0

VAD_DIMENSIONS = {

    "valence": {

        "description": (
            "Linguistic positivity or negativity."
        ),

        "minimum": -1.0,

        "maximum": 1.0
    },

    "arousal": {

        "description": (
            "Linguistic activation or intensity."
        ),

        "minimum": -1.0,

        "maximum": 1.0
    },

    "dominance": {

        "description": (
            "Linguistic control or power."
        ),

        "minimum": -1.0,

        "maximum": 1.0
    }
}


# ============================================================
# 5. TEXT PREPROCESSING
# ============================================================

MIN_TEXT_LENGTH = 1

MAX_TEXT_LENGTH = 20000

LOWERCASE_TERMS = True

NORMALIZE_WHITESPACE = True

PRESERVE_APOSTROPHES = True

PRESERVE_NEGATION_WORDS = True


# ============================================================
# 6. TERM MATCHING
# ============================================================

ENABLE_MULTIWORD_MATCHING = True

ENABLE_LONGEST_PHRASE_MATCHING = True

ALLOW_OVERLAPPING_TERM_MATCHES = False

CASE_INSENSITIVE_MATCHING = True


# ============================================================
# 7. AGGREGATION CONFIGURATION
# ============================================================

# Each matched lexicon entry contributes to the
# corresponding dimension.

AGGREGATION_METHOD = "mean"

MINIMUM_MATCHED_TERMS = 1

ROUNDING_PRECISION = 4

# VAD values represent the average scores of matched
# lexical entries, not calibrated psychological scales.


# ============================================================
# 8. COVERAGE / UNCERTAINTY CONFIGURATION
# ============================================================

REPORT_MATCHED_TERM_COUNT = True

REPORT_UNMATCHED_TOKEN_COUNT = True

REPORT_LEXICON_COVERAGE = True

REPORT_MATCHED_TERMS = True

REPORT_UNCERTAINTY = True

# Low lexical coverage should produce a warning.
# It must not be interpreted as neutral emotional state.

LOW_COVERAGE_WARNING_THRESHOLD = 0.10


# ============================================================
# 9. OUTPUT CONFIGURATION
# ============================================================

OUTPUT_DIMENSIONS = [

    "valence",

    "arousal",

    "dominance"
]

INCLUDE_MATCHED_TERMS = True

INCLUDE_MATCHED_TERM_SCORES = True

INCLUDE_COVERAGE = True

INCLUDE_WARNINGS = True

CLINICAL_DIAGNOSIS = False

ACTUAL_PSYCHOLOGICAL_STATE_CONFIRMED = False


# ============================================================
# 10. DATASET VALIDATION
# ============================================================

EXPECTED_LEXICON_ROWS = 54800

REQUIRE_UNIQUE_TERMS = True

REQUIRE_NON_EMPTY_TERMS = True

REQUIRE_NUMERIC_SCORES = True

REQUIRE_VALID_SCORE_RANGE = True


# ============================================================
# 11. SAFETY AND INTERPRETATION
# ============================================================

DISCLAIMER = (

    "VAD results represent linguistic signals derived from "
    "matched words and phrases in the NRC VAD Lexicon. "
    "They do not establish a student's actual emotional, "
    "psychological or physiological state. Results must "
    "not be used for diagnosis or as a standalone safety "
    "assessment."
)


# ============================================================
# 12. CONFIGURATION VALIDATION
# ============================================================

def validate_config():

    errors = []

    # --------------------------------------------------------
    # A. Validate dataset path
    # --------------------------------------------------------

    if not DATASET_PATH.exists():

        errors.append(
            f"VAD lexicon not found: {DATASET_PATH}"
        )

    # --------------------------------------------------------
    # B. Validate required columns
    # --------------------------------------------------------

    if not REQUIRED_COLUMNS:

        errors.append(
            "Required dataset columns cannot be empty."
        )

    # --------------------------------------------------------
    # C. Validate score range
    # --------------------------------------------------------

    if MIN_VAD_SCORE >= MAX_VAD_SCORE:

        errors.append(
            "Invalid VAD score range."
        )

    if MIN_VAD_SCORE != -1.0 or MAX_VAD_SCORE != 1.0:

        errors.append(
            "NRC VAD score range must be [-1.0, 1.0]."
        )

    # --------------------------------------------------------
    # D. Validate dimensions
    # --------------------------------------------------------

    expected_dimensions = {

        "valence",

        "arousal",

        "dominance"
    }

    if set(VAD_DIMENSIONS) != expected_dimensions:

        errors.append(
            "VAD dimensions must contain valence, arousal "
            "and dominance."
        )

    # --------------------------------------------------------
    # E. Validate aggregation
    # --------------------------------------------------------

    if AGGREGATION_METHOD != "mean":

        errors.append(
            "Unsupported aggregation method."
        )

    if MINIMUM_MATCHED_TERMS < 1:

        errors.append(
            "Minimum matched terms must be at least 1."
        )

    # --------------------------------------------------------
    # F. Validate text limits
    # --------------------------------------------------------

    if MIN_TEXT_LENGTH < 1:

        errors.append(
            "Minimum text length must be positive."
        )

    if MAX_TEXT_LENGTH < MIN_TEXT_LENGTH:

        errors.append(
            "Maximum text length must be greater than "
            "or equal to minimum text length."
        )

    # --------------------------------------------------------
    # G. Validate coverage threshold
    # --------------------------------------------------------

    if not 0.0 <= LOW_COVERAGE_WARNING_THRESHOLD <= 1.0:

        errors.append(
            "Coverage warning threshold must be between 0 and 1."
        )

    # --------------------------------------------------------
    # H. Validate expected dataset size
    # --------------------------------------------------------

    if EXPECTED_LEXICON_ROWS <= 0:

        errors.append(
            "Expected lexicon row count must be positive."
        )

    if errors:

        raise ValueError(
            "\n".join(errors)
        )

    return True


# ============================================================
# 13. CONFIGURATION SUMMARY
# ============================================================

def get_config_summary():

    return {

        "module_name": MODULE_NAME,

        "module_version": MODULE_VERSION,

        "lexicon_name": LEXICON_NAME,

        "lexicon_version": LEXICON_VERSION,

        "dataset_path": str(DATASET_PATH),

        "output_directory": str(OUTPUT_DIR),

        "dimensions": OUTPUT_DIMENSIONS,

        "aggregation_method": AGGREGATION_METHOD,

        "minimum_score": MIN_VAD_SCORE,

        "maximum_score": MAX_VAD_SCORE,

        "expected_lexicon_rows": EXPECTED_LEXICON_ROWS,

        "clinical_diagnosis": CLINICAL_DIAGNOSIS,

        "actual_psychological_state_confirmed": (
            ACTUAL_PSYCHOLOGICAL_STATE_CONFIRMED
        )
    }


# ============================================================
# 14. MAIN
# ============================================================

if __name__ == "__main__":

    validate_config()

    print("=" * 65)

    print("VAD CONFIGURATION VALIDATION: PASSED")

    print("=" * 65)

    for key, value in get_config_summary().items():

        print(f"{key}: {value}")

    print("\nVAD configuration is valid.")
