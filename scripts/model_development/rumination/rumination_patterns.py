
import re


# ============================================================
# REUSABLE REGEX FLAGS
# ============================================================

FLAGS = re.IGNORECASE


# ============================================================
# RUMINATION LINGUISTIC PATTERN DICTIONARY
# ============================================================

RUMINATION_PATTERNS = {

    # --------------------------------------------------------
    # 1. REPETITIVE THOUGHT PATTERNS
    # --------------------------------------------------------

    "repetitive_thoughts": [

        {
            "pattern_id": "RT001",
            "pattern": (
                r"\b(?:i\s+)?(?:keep|constantly|always)\s+"
                r"(?:thinking|wondering|going\s+over)\s+"
                r"(?:about\s+)?"
            ),
            "description": "Repeated thinking expression",
            "weight": 1.0,
        },

        {
            "pattern_id": "RT002",
            "pattern": (
                r"\b(?:the\s+same\s+thoughts?|"
                r"same\s+thing)\s+"
                r"(?:keep\s+)?(?:coming|running)\s+"
                r"(?:back|through)\b"
            ),
            "description": "Recurring thought expression",
            "weight": 1.0,
        },

        {
            "pattern_id": "RT003",
            "pattern": (
                r"\b(?:can't|cannot|can\s+not)\s+"
                r"(?:stop|quit)\s+"
                r"(?:thinking|wondering|worrying)\b"
            ),
            "description": "Difficulty disengaging from thoughts",
            "weight": 1.0,
        },

        {
            "pattern_id": "RT004",
            "pattern": (
                r"\b(?:over\s+and\s+over\s+again|"
                r"time\s+and\s+time\s+again)\b"
            ),
            "description": "Explicit repetition expression",
            "weight": 0.8,
        },
    ],


    # --------------------------------------------------------
    # 2. PAST-FOCUSED RUMINATION
    # --------------------------------------------------------

    "past_focused": [

        {
            "pattern_id": "PF001",
            "pattern": (
                r"\b(?:i\s+)?(?:keep|can't|cannot)\s+"
                r"(?:thinking|dwelling|looking\s+back)\s+"
                r"(?:about|on|at)\s+"
                r"(?:what\s+happened|the\s+past|"
                r"that\s+day|that\s+moment)\b"
            ),
            "description": "Repeated focus on past events",
            "weight": 1.0,
        },

        {
            "pattern_id": "PF002",
            "pattern": (
                r"\b(?:if\s+only)\s+"
                r"(?:i|we|he|she|they)\s+"
                r"(?:had|hadn't|would|wouldn't)\b"
            ),
            "description": "Counterfactual regret expression",
            "weight": 0.9,
        },

        {
            "pattern_id": "PF003",
            "pattern": (
                r"\b(?:i\s+)?(?:keep\s+)?"
                r"(?:replaying|reliving)\s+"
                r"(?:the\s+)?(?:memory|moment|incident|"
                r"conversation|event|situation)\b"
            ),
            "description": "Replaying past experiences",
            "weight": 1.0,
        },

        {
            "pattern_id": "PF004",
            "pattern": (
                r"\b(?:i\s+)?(?:should|could|would)\s+"
                r"have\s+(?:done|said|acted|handled)\s+"
                r"(?:things?\s+)?(?:differently|better)\b"
            ),
            "description": "Regret about past actions",
            "weight": 0.8,
        },
    ],


    # --------------------------------------------------------
    # 3. FUTURE-ORIENTED WORRY
    # --------------------------------------------------------

    "future_worry": [

        {
            "pattern_id": "FW001",
            "pattern": (
                r"\b(?:i\s+)?(?:keep|constantly)\s+"
                r"worrying\s+about\s+"
                r"(?:what\s+will\s+happen|the\s+future|"
                r"what\s+might\s+happen)\b"
            ),
            "description": "Repeated future-oriented worry",
            "weight": 1.0,
        },

        {
            "pattern_id": "FW002",
            "pattern": (
                r"\b(?:i\s+)?(?:can't|cannot)\s+"
                r"stop\s+worrying\s+about\b"
            ),
            "description": "Difficulty disengaging from worry",
            "weight": 1.0,
        },

        {
            "pattern_id": "FW003",
            "pattern": (
                r"\b(?:i\s+)?(?:keep\s+)?"
                r"(?:imagining|expecting|fearing)\s+"
                r"the\s+worst\b"
            ),
            "description": "Repeated negative future expectation",
            "weight": 0.9,
        },

        {
            "pattern_id": "FW004",
            "pattern": (
                r"\b(?:what\s+if)\s+"
                r"(?:everything|something|things|"
                r"it|i|we)\s+"
                r"(?:goes?|went|turns?|turn)\s+"
                r"(?:wrong|badly)\b"
            ),
            "description": "Negative hypothetical future thinking",
            "weight": 0.8,
        },
    ],


    # --------------------------------------------------------
    # 4. SELF-BLAMING THOUGHT PATTERNS
    # --------------------------------------------------------

    "self_blame": [

        {
            "pattern_id": "SB001",
            "pattern": (
                r"\b(?:everything|all\s+of\s+this|"
                r"what\s+happened)\s+"
                r"(?:is|was)\s+"
                r"(?:all\s+)?my\s+fault\b"
            ),
            "description": "Explicit self-blame",
            "weight": 1.0,
        },

        {
            "pattern_id": "SB002",
            "pattern": (
                r"\b(?:i\s+)?(?:keep|always)\s+"
                r"blaming\s+myself\b"
            ),
            "description": "Repeated self-blaming expression",
            "weight": 1.0,
        },

        {
            "pattern_id": "SB003",
            "pattern": (
                r"\b(?:i\s+am|i'm)\s+"
                r"(?:such\s+a\s+)?"
                r"(?:failure|burden|terrible\s+person)\b"
            ),
            "description": "Negative self-evaluation",
            "weight": 0.8,
        },

        {
            "pattern_id": "SB004",
            "pattern": (
                r"\b(?:i\s+)?(?:should|could)\s+"
                r"have\s+(?:known|prevented|stopped|"
                r"realized)\s+(?:better|it|that)\b"
            ),
            "description": "Self-directed responsibility and regret",
            "weight": 0.8,
        },
    ],


    # --------------------------------------------------------
    # 5. UNRESOLVED THOUGHT LOOPS
    # --------------------------------------------------------

    "unresolved_loops": [

        {
            "pattern_id": "UL001",
            "pattern": (
                r"\b(?:no\s+matter\s+what\s+i\s+do|"
                r"whatever\s+i\s+do),?\s+"
                r"(?:i\s+)?(?:keep\s+)?"
                r"(?:coming\s+back\s+to|thinking\s+about)\b"
            ),
            "description": "Recurring unresolved concern",
            "weight": 1.0,
        },

        {
            "pattern_id": "UL002",
            "pattern": (
                r"\b(?:i\s+)?(?:can't|cannot)\s+"
                r"(?:get\s+it|this|that)\s+"
                r"(?:out\s+of\s+my\s+head|"
                r"off\s+my\s+mind)\b"
            ),
            "description": "Persistent unresolved thought",
            "weight": 1.0,
        },

        {
            "pattern_id": "UL003",
            "pattern": (
                r"\b(?:i\s+)?(?:keep|always)\s+"
                r"(?:going\s+in\s+circles|"
                r"coming\s+back\s+to\s+the\s+same\s+thing)\b"
            ),
            "description": "Repeated unresolved thinking cycle",
            "weight": 0.9,
        },

        {
            "pattern_id": "UL004",
            "pattern": (
                r"\b(?:i\s+)?(?:can't|cannot)\s+"
                r"(?:find|reach|get\s+to)\s+"
                r"(?:an?\s+)?(?:answer|solution|closure)\b"
            ),
            "description": "Unresolved concern or lack of closure",
            "weight": 0.7,
        },
    ],
}


# ============================================================
# PATTERN VALIDATION
# ============================================================

def validate_patterns():

    expected_categories = {
        "repetitive_thoughts",
        "past_focused",
        "future_worry",
        "self_blame",
        "unresolved_loops",
    }

    if set(RUMINATION_PATTERNS) != expected_categories:
        raise ValueError(
            "Rumination pattern categories do not match configuration."
        )

    pattern_ids = []

    for category, patterns in RUMINATION_PATTERNS.items():

        if not patterns:
            raise ValueError(
                f"No patterns configured for {category}"
            )

        for item in patterns:

            required_fields = {
                "pattern_id",
                "pattern",
                "description",
                "weight",
            }

            if not required_fields.issubset(item):
                raise ValueError(
                    f"Incomplete pattern definition: {item}"
                )

            re.compile(
                item["pattern"],
                FLAGS,
            )

            if not 0 < item["weight"] <= 1:
                raise ValueError(
                    f"Invalid weight in {item['pattern_id']}"
                )

            pattern_ids.append(item["pattern_id"])

    if len(pattern_ids) != len(set(pattern_ids)):
        raise ValueError("Duplicate pattern IDs detected.")

    return True


if __name__ == "__main__":

    validate_patterns()

    print("=" * 60)
    print("RUMINATION LINGUISTIC PATTERN DICTIONARY")
    print("=" * 60)

    total_patterns = 0

    for category, patterns in RUMINATION_PATTERNS.items():

        print(f"\n{category}: {len(patterns)} patterns")

        total_patterns += len(patterns)

        for item in patterns:
            print(
                f"  {item['pattern_id']} - "
                f"{item['description']}"
            )

    print(f"\nTotal patterns: {total_patterns}")
    print("Regex compilation: PASSED")
    print("Unique pattern IDs: PASSED")
    print("Pattern validation: PASSED")
