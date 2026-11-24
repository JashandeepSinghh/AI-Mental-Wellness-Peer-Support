
"""
Risk Context Analyzer
=====================

Analyzes contextual indicators surrounding candidate risk-language
matches.

Contextual indicators are evidence for human interpretation.
They never automatically suppress candidate risk signals.
"""

from scripts.model_development.risk import risk_config as config

from scripts.model_development.risk.risk_patterns import (
    COMPILED_CONTEXT_PATTERNS,
    get_context_matches,
    get_risk_matches
)


# ============================================================
# 1. TEXT VALIDATION
# ============================================================

def validate_text(text):

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("Input text cannot be empty.")

    if len(text) < config.MIN_TEXT_LENGTH:
        raise ValueError("Input text is too short.")

    if len(text) > config.MAX_TEXT_LENGTH:
        raise ValueError(
            f"Input exceeds maximum length of "
            f"{config.MAX_TEXT_LENGTH} characters."
        )

    return text


# ============================================================
# 2. EXTRACT MATCH CONTEXT
# ============================================================

def extract_match_context(
    text,
    match,
    left_words=None,
    right_words=None
):

    if left_words is None:
        left_words = config.CONTEXT_WINDOW["LEFT_CONTEXT_WORDS"]

    if right_words is None:
        right_words = config.CONTEXT_WINDOW["RIGHT_CONTEXT_WORDS"]

    start = match["start"]
    end = match["end"]

    preceding_words = text[:start].split()
    following_words = text[end:].split()

    left_context = " ".join(preceding_words[-left_words:])
    right_context = " ".join(following_words[:right_words])

    return {
        "matched_text": match["matched_text"],
        "start": start,
        "end": end,
        "left_context": left_context,
        "right_context": right_context,
        "full_context": (
            f"{left_context} {match['matched_text']} {right_context}"
        ).strip()
    }


# ============================================================
# 3. MATCH CONTEXTUAL INDICATORS
# ============================================================

def find_contextual_indicators(text):

    matches = get_context_matches(text)

    return {
        context_name: [
            item["matched_text"]
            for item in context_matches
        ]
        for context_name, context_matches in matches.items()
    }


# ============================================================
# 4. NEGATION ANALYSIS
# ============================================================

def analyze_negation(text, match=None):
    """
    Identify possible negation in the message or before a match.

    Negation is evidence only. It does not suppress a risk match.
    """

    if match is not None:

        context = extract_match_context(text, match)
        search_text = context["left_context"]

    else:

        search_text = text

    patterns = COMPILED_CONTEXT_PATTERNS["negation"]

    matched_negations = []

    for pattern in patterns:

        for found in pattern.finditer(search_text):
            matched_negations.append(found.group(0))

    matched_negations = list(dict.fromkeys(matched_negations))

    return {
        "negation_detected": bool(matched_negations),
        "negation_terms": matched_negations,
        "interpretation": (
            "Possible negation detected. Its scope and target "
            "require contextual review; the expression is not "
            "automatically dismissed."
            if matched_negations
            else "No negation pattern detected in the examined context."
        )
    }


# ============================================================
# 5. TEMPORAL ANALYSIS
# ============================================================

def analyze_temporal_context(text):

    matches = get_context_matches(text)

    historical = matches["historical"]
    current = matches["current_time"]
    future = matches["future_time"]

    return {
        "historical_reference_detected": bool(historical),

        "historical_terms": [
            item["matched_text"] for item in historical
        ],

        "current_time_reference_detected": bool(current),

        "current_time_terms": [
            item["matched_text"] for item in current
        ],

        "future_time_reference_detected": bool(future),

        "future_time_terms": [
            item["matched_text"] for item in future
        ],

        "interpretation": (
            "Temporal expressions are indicators, not proof "
            "that a risk is current or resolved."
        )
    }


# ============================================================
# 6. HYPOTHETICAL ANALYSIS
# ============================================================

def analyze_hypothetical_context(text):

    matches = get_context_matches(text)["hypothetical"]

    return {
        "hypothetical_detected": bool(matches),

        "matched_terms": [
            item["matched_text"] for item in matches
        ],

        "interpretation": (
            "Hypothetical framing detected. It does not establish "
            "that the statement is harmless or unrelated to the speaker."
        )
    }


# ============================================================
# 7. THIRD-PERSON ANALYSIS
# ============================================================

def analyze_third_person_context(text):

    matches = get_context_matches(text)["third_person"]

    return {
        "third_person_reference_detected": bool(matches),

        "matched_terms": [
            item["matched_text"] for item in matches
        ],

        "interpretation": (
            "The message may concern another person. "
            "Indirect self-reference must not be excluded."
        )
    }


# ============================================================
# 8. QUOTATION ANALYSIS
# ============================================================

def analyze_quotation_context(text):

    matches = get_context_matches(text)["quotation"]

    return {
        "quotation_detected": bool(matches),

        "matched_terms": [
            item["matched_text"] for item in matches
        ],

        "interpretation": (
            "Quoted or reported language detected. "
            "The original speaker and current relevance "
            "require contextual interpretation."
        )
    }


# ============================================================
# 9. INTENT ANALYSIS
# ============================================================

def analyze_intent_context(text):

    matches = get_context_matches(text)["intent"]

    return {
        "intent_language_detected": bool(matches),

        "matched_terms": [
            item["matched_text"] for item in matches
        ],

        "interpretation": (
            "Intent-related language detected. "
            "Meaning and target require contextual review."
        )
    }


# ============================================================
# 10. AMBIGUITY ANALYSIS
# ============================================================

def analyze_ambiguity_context(text):

    matches = get_context_matches(text)["ambiguity"]

    return {
        "ambiguity_detected": bool(matches),

        "matched_terms": [
            item["matched_text"] for item in matches
        ]
    }


# ============================================================
# 11. ANALYZE INDIVIDUAL RISK MATCHES
# ============================================================

def analyze_risk_match_context(text, risk_matches):

    analyzed_matches = []

    for category, severity_groups in risk_matches.items():

        for severity, matches in severity_groups.items():

            for match in matches:

                context = extract_match_context(text, match)

                negation = analyze_negation(text, match)

                analyzed_matches.append({

                    "category": category,

                    "candidate_severity": severity,

                    "matched_text": match["matched_text"],

                    "character_start": match["start"],

                    "character_end": match["end"],

                    "context": context,

                    "negation": negation,

                    "requires_contextual_review": True,

                    "automatically_suppressed": False
                })

    return analyzed_matches


# ============================================================
# 12. COMPLETE CONTEXT ANALYSIS
# ============================================================

def analyze_context(text):
    """
    Execute contextual and candidate-pattern analysis.

    Does not assign the final risk category.
    """

    text = validate_text(text)

    risk_matches = get_risk_matches(text)

    contextual_indicators = find_contextual_indicators(text)

    analyzed_risk_matches = analyze_risk_match_context(
        text,
        risk_matches
    )

    context_flags = {

        "negation": analyze_negation(text),

        "temporal": analyze_temporal_context(text),

        "hypothetical": analyze_hypothetical_context(text),

        "third_person": analyze_third_person_context(text),

        "quotation": analyze_quotation_context(text),

        "intent": analyze_intent_context(text),

        "ambiguity": analyze_ambiguity_context(text)
    }

    return {

        "text_length": len(text),

        "risk_matches": analyzed_risk_matches,

        "context_flags": context_flags,

        "contextual_indicators": contextual_indicators,

        "final_classification_assigned": False,

        "clinical_interpretation_performed": False
    }


# ============================================================
# 13. VALIDATION
# ============================================================

def validate_context_analyzer():

    sample = (
        "I do not want to hurt myself, "
        "but I feel overwhelmed."
    )

    result = analyze_context(sample)

    assert "risk_matches" in result
    assert "context_flags" in result

    assert result["context_flags"]["negation"][
        "negation_detected"
    ] is True

    assert "not" in result["context_flags"]["negation"][
        "negation_terms"
    ]

    self_harm_matches = [
        item
        for item in result["risk_matches"]
        if item["category"] == "self_harm_language"
    ]

    assert len(self_harm_matches) >= 1

    assert self_harm_matches[0]["negation"][
        "negation_detected"
    ] is True

    assert self_harm_matches[0][
        "automatically_suppressed"
    ] is False

    assert result["final_classification_assigned"] is False
    assert result["clinical_interpretation_performed"] is False

    # Hypothetical language must be retained.
    hypothetical = analyze_context(
        "What should someone do if they want to kill themselves?"
    )

    assert hypothetical["context_flags"]["hypothetical"][
        "hypothetical_detected"
    ] is True

    assert len(hypothetical["risk_matches"]) >= 1

    # Third-person risk language must be retained.
    third_person = analyze_context(
        "My friend says he wants to hurt himself."
    )

    assert third_person["context_flags"]["third_person"][
        "third_person_reference_detected"
    ] is True

    assert len(third_person["risk_matches"]) >= 1

    # Quoted risk language must be retained.
    quoted = analyze_context(
        'My friend said, "I want to kill myself."'
    )

    assert quoted["context_flags"]["quotation"][
        "quotation_detected"
    ] is True

    assert len(quoted["risk_matches"]) >= 1

    return True


# ============================================================
# 14. MAIN
# ============================================================

if __name__ == "__main__":

    validate_context_analyzer()

    print("=" * 65)
    print("RISK CONTEXT ANALYZER")
    print("=" * 65)

    sample_text = (
        "I do not want to hurt myself, "
        "but I feel overwhelmed."
    )

    result = analyze_context(sample_text)

    print("\nSample text:")
    print(sample_text)

    print("\nRisk matches:")

    for match in result["risk_matches"]:

        print(
            f"  {match['category']} | "
            f"{match['candidate_severity']} | "
            f"{match['matched_text']}"
        )

        print(
            "    Local negation:",
            match["negation"]["negation_detected"]
        )

        print(
            "    Suppressed:",
            match["automatically_suppressed"]
        )

    print("\nOverall negation:")
    print(result["context_flags"]["negation"])

    print("\nOther context flags:")

    for name, value in result["context_flags"].items():

        if name != "negation":
            print(f"  {name}: {value}")

    print("\nContext analyzer validation: PASSED")
    print("=" * 65)
