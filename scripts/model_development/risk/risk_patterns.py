
"""
Risk-Language Detection Pattern Library
========================================

Identifies candidate risk-related expressions and contextual language.

Patterns are linguistic indicators only. They are not diagnoses,
proof of intent, or confirmation of actual danger.
"""

import re

from scripts.model_development.risk import risk_config as config


# ============================================================
# 1. RISK PATTERNS
# ============================================================

RISK_PATTERNS = {

    "self_harm_language": {

        "potential_concern": [
            r"\b(?:hurt|harm|injure)\s+myself\b",
            r"\bself[- ]harm(?:ing)?\b",
            r"\b(?:cutting|hurting)\s+myself\b",
            r"\b(?:i\s+have|i've)\s+been\s+hurting\s+myself\b",
            r"\b(?:i\s+want|i\s+feel\s+like)\s+to\s+hurt\s+myself\b",

            # Third-person / indirect expressions
            r"\b(?:he|she|they|someone|somebody|anyone|a person)\s+"
            r"(?:want|wants|wanted)\s+to\s+"
            r"(?:hurt|harm|injure)\s+"
            r"(?:himself|herself|themselves|themself)\b"
        ],

        "urgent_language_signal": [
            r"\b(?:i\s+am|i'm)\s+going\s+to\s+hurt\s+myself\b",
            r"\b(?:i\s+will|i'll)\s+hurt\s+myself\b",
            r"\b(?:i\s+am|i'm)\s+about\s+to\s+hurt\s+myself\b",
            r"\b(?:i\s+just|i've\s+just)\s+hurt\s+myself\b"
        ]
    },

    "suicidal_language": {

        "potential_concern": [
            r"\b(?:i\s+feel|i\s+think)\s+like\s+ending\s+my\s+life\b",
            r"\b(?:i\s+don't|i\s+do\s+not)\s+want\s+to\s+be\s+alive\b",
            r"\b(?:i\s+wish)\s+i\s+(?:were|was)\s+dead\b",
            r"\b(?:i\s+can't|i\s+cannot)\s+go\s+on\s+living\b",
            r"\b(?:life|living)\s+(?:is|feels)\s+not\s+worth\s+it\b",

            # Broader first-person, third-person and indirect expressions
            r"\b(?:i|he|she|they|someone|somebody|anyone|a person)\s+"
            r"(?:want|wants|wanted|would want)\s+to\s+"
            r"(?:kill|end)\s+"
            r"(?:myself|himself|herself|themselves|themself|"
            r"their life|his life|her life)\b",

            r"\b(?:thought|thinking|thoughts)\s+about\s+suicide\b",
            r"\b(?:considering|considered)\s+suicide\b"
        ],

        "urgent_language_signal": [
            r"\b(?:i\s+am|i'm)\s+going\s+to\s+kill\s+myself\b",
            r"\b(?:i\s+will|i'll)\s+kill\s+myself\b",
            r"\b(?:i\s+plan|i've\s+planned)\s+to\s+end\s+my\s+life\b",
            r"\b(?:i\s+am|i'm)\s+about\s+to\s+end\s+my\s+life\b",
            r"\b(?:i\s+have|i've)\s+decided\s+to\s+end\s+my\s+life\b"
        ]
    },

    "immediate_danger_language": {

        "potential_concern": [
            r"\b(?:i\s+don't|i\s+do\s+not)\s+feel\s+safe\b",
            r"\b(?:someone|somebody)\s+is\s+threatening\s+me\b",
            r"\b(?:i\s+am|i'm)\s+afraid\s+for\s+my\s+safety\b",
            r"\b(?:i\s+am|i'm)\s+in\s+danger\b"
        ],

        "urgent_language_signal": [
            r"\b(?:someone|somebody)\s+is\s+trying\s+to\s+hurt\s+me\s+right\s+now\b",
            r"\b(?:i\s+am|i'm)\s+being\s+attacked\b",
            r"\b(?:i\s+need)\s+immediate\s+help\b",
            r"\b(?:i\s+am|i'm)\s+in\s+immediate\s+danger\b",
            r"\b(?:they\s+are|they're)\s+coming\s+to\s+hurt\s+me\s+right\s+now\b"
        ]
    },

    "hopelessness_language": {

        "potential_concern": [
            r"\bthere\s+is\s+no\s+hope\b",
            r"\b(?:nothing|no\s+one)\s+can\s+help\s+me\b",
            r"\bthings\s+will\s+never\s+get\s+better\b",
            r"\b(?:i\s+feel|i'm)\s+completely\s+hopeless\b",
            r"\bi\s+feel\s+hopeless\b",
            r"\b(?:i\s+see)\s+no\s+way\s+forward\b",
            r"\beverything\s+feels\s+pointless\b"
        ],

        "urgent_language_signal": []
    },

    "severe_distress_language": {

        "potential_concern": [
            r"\b(?:i\s+can't|i\s+cannot)\s+cope\s+anymore\b",
            r"\b(?:i\s+am|i'm)\s+completely\s+overwhelmed\b",
            r"\b(?:i\s+feel|i'm)\s+emotionally\s+broken\b",
            r"\b(?:i\s+can't|i\s+cannot)\s+handle\s+anything\s+anymore\b",
            r"\b(?:i\s+feel|i'm)\s+trapped\s+and\s+helpless\b",
            r"\b(?:i\s+am|i'm)\s+at\s+my\s+breaking\s+point\b"
        ],

        "urgent_language_signal": []
    },

    "threat_language": {

        "potential_concern": [
            r"\b(?:someone|somebody)\s+threatened\s+to\s+hurt\s+me\b",
            r"\b(?:i\s+am|i'm)\s+being\s+threatened\b",
            r"\b(?:someone|somebody)\s+is\s+threatening\s+my\s+family\b"
        ],

        "urgent_language_signal": [
            r"\b(?:someone|somebody)\s+is\s+threatening\s+to\s+attack\s+me\s+right\s+now\b",
            r"\b(?:someone|somebody)\s+is\s+trying\s+to\s+attack\s+me\s+right\s+now\b",
            r"\b(?:i\s+am|i'm)\s+under\s+attack\s+right\s+now\b"
        ]
    }
}


# ============================================================
# 2. CONTEXT PATTERNS
# ============================================================

NEGATION_PATTERNS = [
    r"\b(?:not|never|no longer)\b",
    r"\b(?:don't|doesn't|didn't|can't|cannot|won't|wouldn't)\b",
    r"\b(?:haven't|hasn't|hadn't|isn't|aren't|wasn't|weren't)\b",
    r"\b(?:without|neither|nor)\b"
]


HISTORICAL_PATTERNS = [
    r"\b(?:years|months|weeks|days)\s+ago\b",
    r"\bwhen\s+i\s+was\s+(?:younger|a\s+child|a\s+teenager)\b",
    r"\b(?:used\s+to|previously|formerly)\b",
    r"\b(?:in\s+the\s+past|back\s+then|long\s+ago)\b",
    r"\b(?:i\s+once|there\s+was\s+a\s+time)\b",
    r"\b(?:when\s+i\s+was\s+younger|as\s+a\s+child)\b"
]


HYPOTHETICAL_PATTERNS = [
    r"\bwhat\s+if\b",
    r"\bif\s+i\s+(?:were|was|had|could|would)\b",
    r"\bimagine\s+if\b",
    r"\bsuppose\s+that\b",
    r"\bhypothetically\b",
    r"\b(?:in\s+a\s+fictional\s+story|for\s+a\s+story)\b",

    # General and third-person hypothetical questions
    r"\bwhat\s+should\b.{0,150}\bif\b",
    r"\bif\s+(?:someone|somebody|anyone|a\s+person)\b",
    r"\bwhat\s+would\s+happen\s+if\b"
]


THIRD_PERSON_PATTERNS = [
    r"\b(?:my\s+friend|my\s+roommate|my\s+classmate)\b",
    r"\b(?:my\s+brother|my\s+sister|my\s+parent)\b",
    r"\b(?:he|she|they)\s+(?:said|says|feels|felt|wants|wanted)\b",
    r"\b(?:someone|somebody)\s+(?:else|i\s+know)\b",
    r"\b(?:someone|somebody|anyone|a\s+person)\b"
]


QUOTATION_PATTERNS = [
    r"""["“][^"”]+["”]""",
    r"\b(?:he|she|they)\s+(?:said|told\s+me|mentioned)\b",
    r"\b(?:my\s+friend|my\s+roommate|my\s+classmate)\s+"
    r"(?:said|told\s+me|mentioned)\b",
    r"\b(?:the\s+message|the\s+post|the\s+comment)\s+said\b"
]


INTENT_PATTERNS = [
    r"\b(?:i\s+intend\s+to|i\s+plan\s+to)\b",
    r"\b(?:i\s+have\s+decided\s+to|i\s+am\s+going\s+to)\b",
    r"\b(?:i\s+will|i'll)\b",
    r"\b(?:i\s+am\s+about\s+to|i'm\s+about\s+to)\b"
]


CURRENT_TIME_PATTERNS = [
    r"\b(?:right\s+now|currently|at\s+this\s+moment)\b",
    r"\b(?:today|tonight|this\s+evening)\b",
    r"\b(?:about\s+to|immediately|just\s+now)\b"
]


FUTURE_TIME_PATTERNS = [
    r"\b(?:tomorrow|next\s+week|next\s+month)\b",
    r"\b(?:soon|in\s+the\s+future|later\s+today)\b"
]


AMBIGUITY_PATTERNS = [
    r"\b(?:i\s+don't\s+know|i'm\s+not\s+sure)\b",
    r"\b(?:maybe|perhaps|possibly)\b",
    r"\b(?:i\s+guess|sort\s+of|kind\s+of)\b",
    r"\b(?:it\s+feels\s+like|sometimes\s+i\s+think)\b"
]


# ============================================================
# 3. PATTERN COMPILATION
# ============================================================

def compile_patterns(pattern_list):

    return [
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in pattern_list
    ]


COMPILED_RISK_PATTERNS = {
    category: {
        severity: compile_patterns(patterns)
        for severity, patterns in severity_groups.items()
    }
    for category, severity_groups in RISK_PATTERNS.items()
}


COMPILED_CONTEXT_PATTERNS = {
    "negation": compile_patterns(NEGATION_PATTERNS),
    "historical": compile_patterns(HISTORICAL_PATTERNS),
    "hypothetical": compile_patterns(HYPOTHETICAL_PATTERNS),
    "third_person": compile_patterns(THIRD_PERSON_PATTERNS),
    "quotation": compile_patterns(QUOTATION_PATTERNS),
    "intent": compile_patterns(INTENT_PATTERNS),
    "current_time": compile_patterns(CURRENT_TIME_PATTERNS),
    "future_time": compile_patterns(FUTURE_TIME_PATTERNS),
    "ambiguity": compile_patterns(AMBIGUITY_PATTERNS)
}


# ============================================================
# 4. MATCHING UTILITIES
# ============================================================

def find_matches(text, compiled_patterns):

    matches = []

    if not isinstance(text, str):
        return matches

    for pattern in compiled_patterns:

        for match in pattern.finditer(text):

            matches.append({
                "matched_text": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "pattern": pattern.pattern
            })

    return sorted(
        matches,
        key=lambda item: (item["start"], item["end"])
    )


def get_risk_matches(text):

    results = {}

    for category, severity_groups in COMPILED_RISK_PATTERNS.items():

        results[category] = {}

        for severity, patterns in severity_groups.items():

            results[category][severity] = find_matches(
                text,
                patterns
            )

    return results


def get_context_matches(text):

    return {
        context_name: find_matches(text, patterns)
        for context_name, patterns in COMPILED_CONTEXT_PATTERNS.items()
    }


# ============================================================
# 5. VALIDATION
# ============================================================

def validate_patterns():

    expected_categories = set(config.DETECTION_DIMENSIONS.keys())
    actual_categories = set(RISK_PATTERNS.keys())

    assert actual_categories == expected_categories

    for severity_groups in RISK_PATTERNS.values():

        assert "potential_concern" in severity_groups
        assert "urgent_language_signal" in severity_groups

        for patterns in severity_groups.values():

            for pattern in patterns:
                re.compile(pattern, flags=re.IGNORECASE)

    assert len(COMPILED_RISK_PATTERNS) == 6
    assert len(COMPILED_CONTEXT_PATTERNS) == 9

    # Targeted coverage checks
    assert get_risk_matches(
        "What should someone do if they want to kill themselves?"
    )["suicidal_language"]["potential_concern"]

    assert get_risk_matches(
        "My friend says he wants to hurt himself."
    )["self_harm_language"]["potential_concern"]

    assert get_risk_matches(
        'My friend said, "I want to kill myself."'
    )["suicidal_language"]["potential_concern"]

    return True


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":

    validate_patterns()

    print("=" * 65)
    print("RISK-LANGUAGE PATTERN LIBRARY")
    print("=" * 65)

    for category, severity_groups in RISK_PATTERNS.items():

        print(f"\n{category}")

        for severity, patterns in severity_groups.items():
            print(f"  {severity}: {len(patterns)} patterns")

    print("\nContext pattern groups:", len(COMPILED_CONTEXT_PATTERNS))
    print("Pattern validation: PASSED")
    print("=" * 65)
