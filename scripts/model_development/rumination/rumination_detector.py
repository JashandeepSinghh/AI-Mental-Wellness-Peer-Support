
import re
from typing import Any

from .rumination_config import (
    MIN_TEXT_LENGTH,
    MAX_TEXT_LENGTH,
    MIN_PATTERN_MATCHES,
    CATEGORY_SCORE_CAP,
    SCORE_PRECISION,
    RUMINATION_CATEGORIES,
)

from .rumination_patterns import (
    RUMINATION_PATTERNS,
    FLAGS,
)


# ============================================================
# ADDITIONAL LINGUISTIC PATTERNS
# ============================================================

ADDITIONAL_PATTERNS = {

    "repetitive_thoughts": [],

    "past_focused": [
        {
            "pattern_id": "PF005",
            "pattern": (
                r"\bi\s+keep\s+replaying\s+"
                r"(?:that|the|this)\s+"
                r"(?:conversation|memory|moment|incident|"
                r"event|situation)\b"
            ),
            "description": "Replaying a specific past experience",
            "weight": 1.0,
        },
    ],

    "future_worry": [],

    "self_blame": [
        {
            "pattern_id": "SB005",
            "pattern": (
                r"\beverything\s+that\s+happened\s+"
                r"was\s+(?:all\s+)?my\s+fault\b"
            ),
            "description": "Taking personal blame for an event",
            "weight": 1.0,
        },
    ],

    "unresolved_loops": [
        {
            "pattern_id": "UL005",
            "pattern": (
                r"\bi\s+(?:cannot|can't)\s+get\s+"
                r"(?:that|the|this)\s+"
                r"(?:incident|memory|conversation|"
                r"moment|event|thought|experience)\s+"
                r"out\s+of\s+my\s+head\b"
            ),
            "description": "Difficulty disengaging from a specific thought",
            "weight": 1.0,
        },
    ],
}


# ============================================================
# FIRST-PERSON PERSPECTIVE VALIDATION
# ============================================================

FIRST_PERSON_PATTERN = re.compile(
    r"\b(?:i|i'm|my|myself|me)\b",
    re.IGNORECASE,
)


def has_first_person_perspective(text: str) -> bool:
    """
    Check whether the text contains a first-person reference.

    This is a lightweight safeguard, not full speaker attribution.
    """

    return bool(
        FIRST_PERSON_PATTERN.search(text)
    )


# ============================================================
# NEGATION DETECTION
# ============================================================

NEGATION_SUFFIX_PATTERN = re.compile(
    r"\b(?:do|does|did|am|is|are|was|were|"
    r"have|has|had|can|could|will|would|"
    r"should)\s+not\s+(?:\w+\s+){0,3}$"
    r"|\b(?:don't|doesn't|didn't|"
    r"never|no\s+longer)\s+(?:\w+\s+){0,3}$",
    re.IGNORECASE,
)


def is_negated_match(
    text: str,
    match_start: int,
) -> bool:
    """
    Check for a nearby negation immediately before a match.

    The check is limited to the current sentence.
    """

    prefix = text[:match_start]

    # Restrict analysis to the current sentence.
    sentence_start = max(
        prefix.rfind("."),
        prefix.rfind("!"),
        prefix.rfind("?"),
        prefix.rfind(";"),
    )

    sentence_prefix = prefix[sentence_start + 1:]

    return bool(
        NEGATION_SUFFIX_PATTERN.search(
            sentence_prefix
        )
    )


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def normalize_text(text: str) -> str:

    if not isinstance(text, str):
        raise TypeError(
            "Input text must be a string."
        )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_text(text: str) -> str:

    text = normalize_text(text)

    if len(text) < MIN_TEXT_LENGTH:
        raise ValueError(
            f"Text must contain at least {MIN_TEXT_LENGTH} characters."
        )

    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(
            f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters."
        )

    return text


# ============================================================
# GET CATEGORY PATTERNS
# ============================================================

def get_category_patterns(category: str):

    if category not in RUMINATION_PATTERNS:
        raise ValueError(
            f"Unknown rumination category: {category}"
        )

    return (
        RUMINATION_PATTERNS[category]
        + ADDITIONAL_PATTERNS.get(category, [])
    )


# ============================================================
# PATTERN MATCHING
# ============================================================

def detect_category_patterns(
    text: str,
    category: str,
) -> list[dict[str, Any]]:

    matches = []

    patterns = get_category_patterns(
        category
    )

    for item in patterns:

        compiled_pattern = re.compile(
            item["pattern"],
            FLAGS,
        )

        for match in compiled_pattern.finditer(text):

            evidence = match.group(0).strip()

            if not evidence:
                continue

            # --------------------------------------------
            # NEGATION FILTER
            # --------------------------------------------

            if is_negated_match(
                text,
                match.start(),
            ):
                continue

            # --------------------------------------------
            # FIRST-PERSON FILTER
            # --------------------------------------------

            if not has_first_person_perspective(text):
                continue

            matches.append({

                "pattern_id": item["pattern_id"],

                "description": item["description"],

                "evidence": evidence,

                "start": match.start(),

                "end": match.end(),

                "weight": item["weight"],

            })

    return matches


# ============================================================
# CATEGORY SCORING
# ============================================================

def calculate_category_score(
    matches: list[dict[str, Any]],
) -> float:

    if len(matches) < MIN_PATTERN_MATCHES:
        return 0.0

    unique_pattern_weights = {}

    for match in matches:

        pattern_id = match["pattern_id"]

        unique_pattern_weights[pattern_id] = max(
            unique_pattern_weights.get(pattern_id, 0.0),
            match["weight"],
        )

    total_weight = sum(
        unique_pattern_weights.values()
    )

    score = min(
        total_weight,
        CATEGORY_SCORE_CAP,
    )

    return round(
        score,
        SCORE_PRECISION,
    )


# ============================================================
# SINGLE-TEXT RUMINATION DETECTION
# ============================================================

def detect_rumination(text: str) -> dict[str, Any]:

    text = validate_text(text)

    category_results = {}

    total_matches = 0

    detected_categories = []

    for category, category_name in RUMINATION_CATEGORIES.items():

        matches = detect_category_patterns(
            text,
            category,
        )

        score = calculate_category_score(
            matches
        )

        detected = (
            len(matches) >= MIN_PATTERN_MATCHES
        )

        total_matches += len(matches)

        if detected:
            detected_categories.append(category)

        category_results[category] = {

            "category_name": category_name,

            "detected": detected,

            "heuristic_score": score,

            "match_count": len(matches),

            "evidence": matches,

        }

    any_indicator_detected = (
        len(detected_categories) > 0
    )

    return {

        "module": "Rumination Detection",

        "version": "1.1.0",

        "analysis_type": "single_text",

        "input_text": text,

        "any_indicator_detected": any_indicator_detected,

        "detected_category_count": len(
            detected_categories
        ),

        "detected_categories": detected_categories,

        "total_pattern_matches": total_matches,

        "categories": category_results,

        "interpretation": (
            "Possible rumination-related linguistic patterns "
            "were detected. This is not a clinical diagnosis."
            if any_indicator_detected
            else
            "No configured rumination-related linguistic "
            "patterns were detected. This does not establish "
            "the absence of rumination."
        ),

        "limitations": [
            "Pattern matching cannot establish a person's mental state.",
            "A single message cannot establish persistence over time.",
            "First-person detection is a heuristic, not full speaker attribution.",
            "Negation detection does not understand every sentence structure.",
            "Patterns may occur in ordinary conversation.",
            "Absence of a pattern does not prove absence of rumination.",
        ],

    }


# ============================================================
# CONSOLE OUTPUT
# ============================================================

def print_detection_result(
    result: dict[str, Any],
) -> None:

    print("\n" + "=" * 65)
    print("RUMINATION DETECTION RESULT")
    print("=" * 65)

    print(
        f"Indicator detected: "
        f"{result['any_indicator_detected']}"
    )

    print(
        f"Detected categories: "
        f"{result['detected_category_count']}"
    )

    print(
        f"Total pattern matches: "
        f"{result['total_pattern_matches']}"
    )

    print("\nCATEGORY ANALYSIS")

    for category, details in result["categories"].items():

        print(f"\n{details['category_name']}")

        print(
            f"Detected: {details['detected']}"
        )

        print(
            f"Heuristic score: "
            f"{details['heuristic_score']}"
        )

        print(
            f"Matches: {details['match_count']}"
        )

        for evidence in details["evidence"]:

            print(
                f"  [{evidence['pattern_id']}] "
                f"{evidence['evidence']}"
            )

    print("\nINTERPRETATION")
    print(result["interpretation"])

    print("\n" + "=" * 65)


# ============================================================
# MANUAL EXECUTION
# ============================================================

if __name__ == "__main__":

    sample_text = (
        "I keep thinking about what happened yesterday. "
        "I cannot stop thinking about it."
    )

    result = detect_rumination(
        sample_text
    )

    print_detection_result(
        result
    )
