
"""
Risk-Language Detection Configuration
======================================

Central configuration for the rule-based risk-language detection
module.

Important:
- This module does not diagnose mental-health conditions.
- A detected linguistic signal is not proof of actual intent.
- No detected signal does not establish that a person is safe.
- Urgent language requires appropriate human attention.
"""

from pathlib import Path


# ============================================================
# 1. MODULE INFORMATION
# ============================================================

MODULE_NAME = "risk_language_detection"

MODULE_VERSION = "1.0.0"

MODULE_DESCRIPTION = (
    "Context-aware detection of potentially concerning "
    "and urgent risk-related language."
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RISK_MODULE_DIR = Path(__file__).resolve().parent

TEST_DIR = RISK_MODULE_DIR / "tests"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "risk"
    / "rule_based"
)


# ============================================================
# 3. RISK CATEGORIES
# ============================================================

RISK_CATEGORIES = {
    "NO_EXPLICIT_RISK_SIGNAL": "no_explicit_risk_signal",

    "POTENTIAL_CONCERN": "potential_concern",

    "URGENT_LANGUAGE_SIGNAL": "urgent_language_signal",

    "UNCERTAIN_OR_AMBIGUOUS": "uncertain_or_ambiguous"
}


CATEGORY_DESCRIPTIONS = {
    RISK_CATEGORIES["NO_EXPLICIT_RISK_SIGNAL"]: (
        "No explicit risk-related language was detected. "
        "This does not establish that the user is safe."
    ),

    RISK_CATEGORIES["POTENTIAL_CONCERN"]: (
        "Language contains potentially concerning indicators "
        "that may require supportive follow-up."
    ),

    RISK_CATEGORIES["URGENT_LANGUAGE_SIGNAL"]: (
        "Language contains explicit indicators of possible "
        "immediate danger requiring prompt human attention."
    ),

    RISK_CATEGORIES["UNCERTAIN_OR_AMBIGUOUS"]: (
        "Available language or context is insufficient, "
        "contradictory, or ambiguous."
    )
}


# ============================================================
# 4. DETECTION DIMENSIONS
# ============================================================

DETECTION_DIMENSIONS = {
    "self_harm_language": True,
    "suicidal_language": True,
    "immediate_danger_language": True,
    "hopelessness_language": True,
    "severe_distress_language": True,
    "threat_language": True
}


# ============================================================
# 5. CONTEXT ANALYSIS SETTINGS
# ============================================================

CONTEXT_SETTINGS = {

    # Negation handling:
    "ENABLE_NEGATION_ANALYSIS": True,

    # Distinguish current statements from historical experiences.
    "ENABLE_TEMPORAL_ANALYSIS": True,

    # Identify hypothetical statements.
    "ENABLE_HYPOTHETICAL_ANALYSIS": True,

    # Identify statements referring to another person.
    "ENABLE_THIRD_PERSON_ANALYSIS": True,

    # Identify quoted or reported language.
    "ENABLE_QUOTATION_ANALYSIS": True,

    # Examine surrounding conversational messages.
    "ENABLE_CONVERSATION_CONTEXT": True,

    # Preserve uncertainty when contextual interpretation is unclear.
    "PRESERVE_AMBIGUITY": True
}


# ============================================================
# 6. TEXT VALIDATION SETTINGS
# ============================================================

MIN_TEXT_LENGTH = 1

MAX_TEXT_LENGTH = 20000

MIN_CONVERSATION_MESSAGES = 1

MAX_CONVERSATION_MESSAGES = 100


# ============================================================
# 7. CONTEXT WINDOW SETTINGS
# ============================================================

CONTEXT_WINDOW = {

    # Number of tokens/words to inspect around a detected phrase.
    "LEFT_CONTEXT_WORDS": 12,

    "RIGHT_CONTEXT_WORDS": 12,

    # Number of preceding messages available for contextual review.
    "PREVIOUS_MESSAGES": 5,

    # Number of following messages available when supplied.
    "FOLLOWING_MESSAGES": 2
}


# ============================================================
# 8. DETECTION POLICY
# ============================================================

DETECTION_POLICY = {

    # Individual keyword matches cannot independently establish risk.
    "KEYWORD_MATCH_ALONE_IS_INSUFFICIENT": True,

    # Negative sentiment alone must not trigger urgent classification.
    "NEGATIVE_SENTIMENT_ALONE_IS_INSUFFICIENT": True,

    # Emotion classification alone must not trigger urgent classification.
    "EMOTION_LABEL_ALONE_IS_INSUFFICIENT": True,

    # Absence of detected language is not proof of safety.
    "NO_SIGNAL_DOES_NOT_MEAN_SAFE": True,

    # Ambiguous language must preserve uncertainty.
    "AMBIGUOUS_LANGUAGE_REQUIRES_CAUTION": True,

    # Clinical diagnosis is outside the system's scope.
    "CLINICAL_DIAGNOSIS_ENABLED": False,

    # Human review is required for urgent signals.
    "URGENT_SIGNAL_REQUIRES_HUMAN_REVIEW": True,

    # Human review is recommended for potential concerns.
    "POTENTIAL_CONCERN_REQUIRES_REVIEW": True
}


# ============================================================
# 9. ESCALATION SETTINGS
# ============================================================

ESCALATION_LEVELS = {

    "NONE": "none",

    "SUPPORTIVE_FOLLOW_UP": "supportive_follow_up",

    "HUMAN_REVIEW": "human_review",

    "PROMPT_HUMAN_REVIEW": "prompt_human_review"
}


ESCALATION_MAPPING = {

    RISK_CATEGORIES["NO_EXPLICIT_RISK_SIGNAL"]:
        ESCALATION_LEVELS["NONE"],

    RISK_CATEGORIES["POTENTIAL_CONCERN"]:
        ESCALATION_LEVELS["SUPPORTIVE_FOLLOW_UP"],

    RISK_CATEGORIES["URGENT_LANGUAGE_SIGNAL"]:
        ESCALATION_LEVELS["PROMPT_HUMAN_REVIEW"],

    RISK_CATEGORIES["UNCERTAIN_OR_AMBIGUOUS"]:
        ESCALATION_LEVELS["HUMAN_REVIEW"]
}


# ============================================================
# 10. STRUCTURED OUTPUT CONFIGURATION
# ============================================================

OUTPUT_FIELDS = [

    "module_name",

    "module_version",

    "risk_category",

    "escalation_level",

    "detected_indicators",

    "context_flags",

    "evidence",

    "explanation",

    "requires_human_review",

    "uncertainty_notes",

    "disclaimer"
]


DEFAULT_DISCLAIMER = (
    "This output identifies linguistic indicators only. "
    "It is not a clinical diagnosis or a determination of actual risk. "
    "Absence of detected indicators does not establish safety."
)


# ============================================================
# 11. VALIDATION
# ============================================================

def validate_config():

    """
    Validate configuration consistency before detector execution.
    """

    assert MODULE_NAME
    assert MODULE_VERSION

    assert MIN_TEXT_LENGTH > 0

    assert MAX_TEXT_LENGTH >= MIN_TEXT_LENGTH

    assert MIN_CONVERSATION_MESSAGES > 0

    assert (
        MAX_CONVERSATION_MESSAGES
        >= MIN_CONVERSATION_MESSAGES
    )

    assert len(RISK_CATEGORIES) == 4

    assert len(set(RISK_CATEGORIES.values())) == 4

    assert all(
        category in CATEGORY_DESCRIPTIONS
        for category in RISK_CATEGORIES.values()
    )

    assert all(
        category in ESCALATION_MAPPING
        for category in RISK_CATEGORIES.values()
    )

    assert (
        DETECTION_POLICY[
            "KEYWORD_MATCH_ALONE_IS_INSUFFICIENT"
        ]
        is True
    )

    assert (
        DETECTION_POLICY[
            "NEGATIVE_SENTIMENT_ALONE_IS_INSUFFICIENT"
        ]
        is True
    )

    assert (
        DETECTION_POLICY[
            "CLINICAL_DIAGNOSIS_ENABLED"
        ]
        is False
    )

    assert len(OUTPUT_FIELDS) == len(set(OUTPUT_FIELDS))

    assert OUTPUT_DIR.is_absolute()

    return True


# ============================================================
# 12. CONFIGURATION SUMMARY
# ============================================================

def get_config_summary():

    """
    Return a structured overview of the active configuration.
    """

    return {

        "module_name": MODULE_NAME,

        "module_version": MODULE_VERSION,

        "risk_categories": list(
            RISK_CATEGORIES.values()
        ),

        "detection_dimensions": list(
            DETECTION_DIMENSIONS.keys()
        ),

        "context_analysis_enabled": [
            key
            for key, value in CONTEXT_SETTINGS.items()
            if value is True
        ],

        "clinical_diagnosis_enabled": (
            DETECTION_POLICY[
                "CLINICAL_DIAGNOSIS_ENABLED"
            ]
        ),

        "output_fields": OUTPUT_FIELDS.copy()
    }


if __name__ == "__main__":

    validate_config()

    print("=" * 65)
    print("RISK-LANGUAGE DETECTION CONFIGURATION")
    print("=" * 65)

    print(f"Module       : {MODULE_NAME}")
    print(f"Version      : {MODULE_VERSION}")
    print(f"Project root : {PROJECT_ROOT}")

    print("\nRisk categories:")

    for category in RISK_CATEGORIES.values():
        print(f"  - {category}")

    print("\nConfiguration validation: PASSED")

    print("=" * 65)
