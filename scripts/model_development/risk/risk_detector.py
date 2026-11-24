
"""
Risk-Language Detection Engine
==============================

Integrates:
1. Risk configuration
2. Linguistic risk patterns
3. Sentence-level context analysis
4. Risk-category assignment
5. Escalation recommendations
6. Structured output

Experimental linguistic signal detector.

It does not diagnose, establish intent, or determine actual safety.
Human review is required for concerning or uncertain signals.
"""

import re
from datetime import datetime, timezone

from scripts.model_development.risk import risk_config as config

from scripts.model_development.risk.risk_context_analyzer import (
    analyze_context,
    validate_text,
)

from scripts.model_development.risk.risk_patterns import (
    get_context_matches,
)


# ============================================================
# 1. MODULE CONSTANTS
# ============================================================

CATEGORY = config.RISK_CATEGORIES

IDENTITY_OR_TIME_CONTEXTS = (
    "historical",
    "hypothetical",
    "third_person",
    "quotation",
)


# ============================================================
# 2. EXTRACT SENTENCE-LEVEL CONTEXT
# ============================================================

def extract_sentence_context(text, match):
    """
    Extract the sentence containing a particular risk expression.

    Context from separate sentences must not automatically
    change the classification of this expression.
    """

    match_start = match["character_start"]
    match_end = match["character_end"]

    separators = list(
        re.finditer(r"[.!?\n]+", text)
    )

    sentence_start = 0
    sentence_end = len(text)

    for separator in separators:

        if separator.end() <= match_start:
            sentence_start = separator.end()

        elif separator.start() >= match_end:
            sentence_end = separator.start()
            break

    return text[sentence_start:sentence_end].strip()


def analyze_match_context(text, match):
    """
    Evaluate contextual indicators within the sentence
    containing an individual risk expression.
    """

    sentence = extract_sentence_context(text, match)

    matches = get_context_matches(sentence)

    local_flags = {
        "historical": bool(matches["historical"]),
        "hypothetical": bool(matches["hypothetical"]),
        "third_person": bool(matches["third_person"]),
        "quotation": bool(matches["quotation"]),
    }

    match["sentence_context"] = sentence

    match["local_context_flags"] = local_flags

    match["local_context_uncertain"] = any(
        local_flags.values()
    )

    return match


def annotate_match_contexts(text, risk_matches):
    """
    Attach sentence-level context to every detected expression.
    """

    annotated_matches = []

    for match in risk_matches:

        annotated = analyze_match_context(
            text,
            match.copy(),
        )

        annotated_matches.append(annotated)

    return annotated_matches


# ============================================================
# 3. EXTRACT DETECTED INDICATORS
# ============================================================

def extract_detected_indicators(risk_matches):
    """
    Group detected expressions by linguistic category.
    """

    indicators = {}

    for match in risk_matches:

        category = match["category"]

        if category not in indicators:
            indicators[category] = []

        indicators[category].append({

            "matched_text": match["matched_text"],

            "candidate_severity": match["candidate_severity"],

            "character_start": match["character_start"],

            "character_end": match["character_end"],

            "negation_detected": (
                match["negation"]["negation_detected"]
            ),

            "automatically_suppressed": (
                match["automatically_suppressed"]
            ),

            "sentence_context": match["sentence_context"],

            "local_context_flags": (
                match["local_context_flags"]
            ),

            "local_context_uncertain": (
                match["local_context_uncertain"]
            ),
        })

    return indicators


# ============================================================
# 4. DETERMINE CANDIDATE SEVERITY
# ============================================================

def determine_candidate_severity(risk_matches):
    """
    Determine the strongest matched linguistic severity.

    This is not a clinical severity assessment.
    """

    has_urgent = any(
        match["candidate_severity"] == "urgent_language_signal"
        for match in risk_matches
    )

    has_potential = any(
        match["candidate_severity"] == "potential_concern"
        for match in risk_matches
    )

    if has_urgent:
        return "urgent_language_signal"

    if has_potential:
        return "potential_concern"

    return None


# ============================================================
# 5. IDENTIFY GLOBAL CONTEXTUAL UNCERTAINTY
# ============================================================

def identify_contextual_uncertainty(context_flags):
    """
    Identify contextual indicators in the complete message.

    These indicators are retained for reporting.
    They do not automatically downgrade unrelated matches.
    """

    uncertainty_reasons = []

    if context_flags["negation"]["negation_detected"]:

        uncertainty_reasons.append(
            "Negation language detected; scope and target require review."
        )

    if context_flags["temporal"]["historical_reference_detected"]:

        uncertainty_reasons.append(
            "Historical reference detected; current relevance is unknown."
        )

    if context_flags["hypothetical"]["hypothetical_detected"]:

        uncertainty_reasons.append(
            "Hypothetical framing detected; real-world relevance is unknown."
        )

    if context_flags["third_person"]["third_person_reference_detected"]:

        uncertainty_reasons.append(
            "Third-person reference detected; the person concerned is unclear."
        )

    if context_flags["quotation"]["quotation_detected"]:

        uncertainty_reasons.append(
            "Quoted or reported language detected; original speaker is unclear."
        )

    if context_flags["ambiguity"]["ambiguity_detected"]:

        uncertainty_reasons.append(
            "Ambiguous linguistic expressions detected."
        )

    return uncertainty_reasons


# ============================================================
# 6. CHECK INDIVIDUAL MATCH UNCERTAINTY
# ============================================================

def has_identity_or_time_uncertainty(context_flags):
    """
    Check contextual uncertainty for an individual expression.
    """

    for context_name in IDENTITY_OR_TIME_CONTEXTS:

        if context_flags.get(context_name, False):
            return True

    return False


# ============================================================
# 7. ASSIGN RISK CATEGORY
# ============================================================

def assign_risk_category(
    candidate_severity,
    contextual_uncertainty,
    context_flags=None,
    risk_matches=None,
):
    """
    Assign one of the four configured output categories.

    Classification policy:

    1. A direct urgent expression remains urgent.
    2. An urgent expression with local attribution uncertainty
       is classified as uncertain and requires human review.
    3. An unrelated sentence cannot downgrade a direct match.
    4. Potential concerns preserve their own local context.
    5. Negation alone never automatically suppresses a match.
    6. Contextual uncertainty without a risk match is retained.
    """

    # --------------------------------------------------------
    # A. Match-level classification
    # --------------------------------------------------------

    if risk_matches:

        urgent_direct = False
        urgent_uncertain = False

        potential_direct = False
        potential_uncertain = False

        for match in risk_matches:

            severity = match["candidate_severity"]

            local_uncertain = (
                match["local_context_uncertain"]
            )

            if severity == "urgent_language_signal":

                if local_uncertain:
                    urgent_uncertain = True
                else:
                    urgent_direct = True

            elif severity == "potential_concern":

                if local_uncertain:
                    potential_uncertain = True
                else:
                    potential_direct = True

        # A direct urgent signal takes precedence.
        if urgent_direct:

            return CATEGORY["URGENT_LANGUAGE_SIGNAL"]

        # An urgent expression with uncertain attribution
        # must still receive human review.
        if urgent_uncertain:

            return CATEGORY["UNCERTAIN_OR_AMBIGUOUS"]

        # A direct potential concern is preserved even when
        # unrelated sentences contain contextual indicators.
        if potential_direct:

            return CATEGORY["POTENTIAL_CONCERN"]

        if potential_uncertain:

            return CATEGORY["UNCERTAIN_OR_AMBIGUOUS"]

    # --------------------------------------------------------
    # B. Backward-compatible fallback
    # --------------------------------------------------------

    if risk_matches is None and context_flags is not None:

        attribution_uncertain = any(
            context_flags.get(name, {}).get(
                {
                    "historical": "historical_reference_detected",
                    "hypothetical": "hypothetical_detected",
                    "third_person": "third_person_reference_detected",
                    "quotation": "quotation_detected",
                }[name],
                False,
            )
            for name in IDENTITY_OR_TIME_CONTEXTS
        )

        if (
            candidate_severity == "urgent_language_signal"
            and attribution_uncertain
        ):
            return CATEGORY["UNCERTAIN_OR_AMBIGUOUS"]

        if candidate_severity == "urgent_language_signal":
            return CATEGORY["URGENT_LANGUAGE_SIGNAL"]

        if (
            candidate_severity == "potential_concern"
            and attribution_uncertain
        ):
            return CATEGORY["UNCERTAIN_OR_AMBIGUOUS"]

        if candidate_severity == "potential_concern":
            return CATEGORY["POTENTIAL_CONCERN"]

    # --------------------------------------------------------
    # C. No risk expression detected
    # --------------------------------------------------------

    if contextual_uncertainty:

        return CATEGORY["UNCERTAIN_OR_AMBIGUOUS"]

    return CATEGORY["NO_EXPLICIT_RISK_SIGNAL"]


# ============================================================
# 8. DETERMINE ESCALATION
# ============================================================

def determine_escalation(risk_category):

    return config.ESCALATION_MAPPING[risk_category]


# ============================================================
# 9. DETERMINE HUMAN REVIEW REQUIREMENT
# ============================================================

def requires_human_review(risk_category):

    return risk_category in {

        CATEGORY["POTENTIAL_CONCERN"],

        CATEGORY["URGENT_LANGUAGE_SIGNAL"],

        CATEGORY["UNCERTAIN_OR_AMBIGUOUS"],
    }


# ============================================================
# 10. GENERATE EXPLANATION
# ============================================================

def generate_explanation(
    risk_category,
    detected_indicators,
    uncertainty_reasons,
):
    """
    Generate a transparent explanation of the assigned category.
    """

    total_matches = sum(
        len(matches)
        for matches in detected_indicators.values()
    )

    if risk_category == CATEGORY["URGENT_LANGUAGE_SIGNAL"]:

        explanation = (
            "One or more expressions matched configured "
            "urgent-language patterns without detected "
            "speaker or temporal attribution uncertainty "
            "within the matched expression's sentence. "
            "This is a linguistic signal requiring prompt "
            "human attention, not confirmation of actual "
            "intent or immediate danger."
        )

    elif risk_category == CATEGORY["POTENTIAL_CONCERN"]:

        explanation = (
            "One or more expressions matched potential-concern "
            "patterns. Supportive follow-up and contextual review "
            "are recommended."
        )

    elif risk_category == CATEGORY["UNCERTAIN_OR_AMBIGUOUS"]:

        explanation = (
            "The message contains risk-related or ambiguous "
            "contextual language. The available information "
            "is insufficient to confidently attribute the "
            "expression to a current risk involving the speaker. "
            "Human review is recommended."
        )

    else:

        explanation = (
            "No explicit configured risk-language pattern was "
            "detected. This does not establish that the user is safe."
        )

    return {

        "summary": explanation,

        "matched_expression_count": total_matches,

        "contextual_uncertainty_count": len(
            uncertainty_reasons
        ),

        "contextual_uncertainty": uncertainty_reasons,
    }


# ============================================================
# 11. BUILD STRUCTURED OUTPUT
# ============================================================

def build_detection_output(
    risk_category,
    escalation_level,
    detected_indicators,
    context_flags,
    explanation,
):
    """
    Build a consistent JSON-compatible result.
    """

    return {

        "module_name": config.MODULE_NAME,

        "module_version": config.MODULE_VERSION,

        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "risk_category": risk_category,

        "escalation_level": escalation_level,

        "detected_indicators": detected_indicators,

        "context_flags": context_flags,

        "evidence": [
            match
            for matches in detected_indicators.values()
            for match in matches
        ],

        "explanation": explanation,

        "requires_human_review": requires_human_review(
            risk_category
        ),

        "uncertainty_notes": explanation[
            "contextual_uncertainty"
        ],

        "disclaimer": config.DEFAULT_DISCLAIMER,

        "clinical_diagnosis": False,

        "actual_risk_confirmed": False,
    }


# ============================================================
# 12. MAIN DETECTION FUNCTION
# ============================================================

def detect_risk_language(text):
    """
    Execute the complete risk-language detection pipeline.

    Returns structured linguistic risk-detection output.
    """

    config.validate_config()

    text = validate_text(text)

    # --------------------------------------------------------
    # A. Context and pattern analysis
    # --------------------------------------------------------

    context_result = analyze_context(text)

    risk_matches = context_result["risk_matches"]

    context_flags = context_result["context_flags"]

    # --------------------------------------------------------
    # B. Attach individual sentence-level context
    # --------------------------------------------------------

    risk_matches = annotate_match_contexts(
        text,
        risk_matches,
    )

    # --------------------------------------------------------
    # C. Extract detected indicators
    # --------------------------------------------------------

    detected_indicators = extract_detected_indicators(
        risk_matches
    )

    # --------------------------------------------------------
    # D. Determine strongest candidate severity
    # --------------------------------------------------------

    candidate_severity = determine_candidate_severity(
        risk_matches
    )

    # --------------------------------------------------------
    # E. Identify global contextual uncertainty for reporting
    # --------------------------------------------------------

    contextual_uncertainty = identify_contextual_uncertainty(
        context_flags
    )

    # --------------------------------------------------------
    # F. Assign category using individual match context
    # --------------------------------------------------------

    risk_category = assign_risk_category(
        candidate_severity=candidate_severity,
        contextual_uncertainty=contextual_uncertainty,
        context_flags=context_flags,
        risk_matches=risk_matches,
    )

    # --------------------------------------------------------
    # G. Determine escalation
    # --------------------------------------------------------

    escalation_level = determine_escalation(
        risk_category
    )

    # --------------------------------------------------------
    # H. Generate explanation
    # --------------------------------------------------------

    explanation = generate_explanation(
        risk_category,
        detected_indicators,
        contextual_uncertainty,
    )

    # --------------------------------------------------------
    # I. Build structured output
    # --------------------------------------------------------

    return build_detection_output(
        risk_category=risk_category,
        escalation_level=escalation_level,
        detected_indicators=detected_indicators,
        context_flags=context_flags,
        explanation=explanation,
    )


# ============================================================
# 13. BASIC VALIDATION AND REGRESSION TESTS
# ============================================================

def validate_detector():

    # Test 1: No explicit risk expression.
    result = detect_risk_language(
        "I enjoyed studying with my friends today."
    )

    assert result["risk_category"] == (
        CATEGORY["NO_EXPLICIT_RISK_SIGNAL"]
    )

    # Test 2: Potential concern.
    result = detect_risk_language(
        "I feel completely hopeless."
    )

    assert result["risk_category"] == (
        CATEGORY["POTENTIAL_CONCERN"]
    )

    # Test 3: Direct urgent-language match.
    result = detect_risk_language(
        "I am going to kill myself."
    )

    assert result["risk_category"] == (
        CATEGORY["URGENT_LANGUAGE_SIGNAL"]
    )

    # Test 4: Negation must not suppress the expression.
    result = detect_risk_language(
        "I do not want to hurt myself."
    )

    assert result["risk_category"] == (
        CATEGORY["POTENTIAL_CONCERN"]
    )

    assert result["requires_human_review"] is True

    # Test 5: Ambiguous language.
    result = detect_risk_language(
        "Maybe things will change someday."
    )

    assert result["risk_category"] == (
        CATEGORY["UNCERTAIN_OR_AMBIGUOUS"]
    )

    # Test 6: Hypothetical urgent-language expression.
    result = detect_risk_language(
        "What should someone do if they want to kill themselves?"
    )

    assert result["risk_category"] == (
        CATEGORY["UNCERTAIN_OR_AMBIGUOUS"]
    )

    assert result["requires_human_review"] is True

    # Test 7: Third-person risk expression.
    result = detect_risk_language(
        "My friend says he wants to hurt himself."
    )

    assert result["risk_category"] == (
        CATEGORY["UNCERTAIN_OR_AMBIGUOUS"]
    )

    assert result["requires_human_review"] is True

    # Test 8: Quoted risk expression.
    result = detect_risk_language(
        'My friend said, "I want to kill myself."'
    )

    assert result["risk_category"] == (
        CATEGORY["UNCERTAIN_OR_AMBIGUOUS"]
    )

    assert result["requires_human_review"] is True

    # Test 9: Reassuring language must not override
    # a direct urgent expression.
    result = detect_risk_language(
        "I am fine, but I am going to kill myself."
    )

    assert result["risk_category"] == (
        CATEGORY["URGENT_LANGUAGE_SIGNAL"]
    )

    # Test 10: Historical risk language requires review.
    result = detect_risk_language(
        "I thought about suicide when I was younger."
    )

    assert result["risk_category"] == (
        CATEGORY["UNCERTAIN_OR_AMBIGUOUS"]
    )

    assert result["requires_human_review"] is True

    # Test 11: Clinical diagnosis must never be assigned.
    assert result["clinical_diagnosis"] is False

    assert result["actual_risk_confirmed"] is False

    # Test 12: Required output fields.
    for field in config.OUTPUT_FIELDS:

        assert field in result, (
            f"Missing output field: {field}"
        )

    # --------------------------------------------------------
    # NEW REGRESSION TESTS: MIXED SENTENCE CONTEXT
    # --------------------------------------------------------

    mixed_context_samples = [

        (
            "I am going to kill myself. "
            "What should someone do if they want to kill themselves?"
        ),

        (
            "I am going to kill myself. "
            "I remember that I felt hopeless years ago."
        ),

        (
            "I am going to kill myself. "
            "My friend once said he felt hopeless."
        ),

    ]

    for sample in mixed_context_samples:

        result = detect_risk_language(sample)

        assert result["risk_category"] == (
            CATEGORY["URGENT_LANGUAGE_SIGNAL"]
        ), (
            "Unrelated sentence context downgraded "
            f"a direct urgent expression: {sample}"
        )

        assert result["requires_human_review"] is True

    # --------------------------------------------------------
    # Local context must still be respected.
    # --------------------------------------------------------

    local_hypothetical = detect_risk_language(
        "Hypothetically, I am going to kill myself."
    )

    assert local_hypothetical["risk_category"] == (
        CATEGORY["UNCERTAIN_OR_AMBIGUOUS"]
    )

    assert local_hypothetical["requires_human_review"] is True

    # Historical context in the same sentence.
    local_historical = detect_risk_language(
        "Years ago, I was going to kill myself."
    )

    assert local_historical["risk_category"] == (
        CATEGORY["UNCERTAIN_OR_AMBIGUOUS"]
    )

    # --------------------------------------------------------
    # Output safeguards
    # --------------------------------------------------------

    for sample in mixed_context_samples:

        result = detect_risk_language(sample)

        assert result["clinical_diagnosis"] is False

        assert result["actual_risk_confirmed"] is False

    return True


# ============================================================
# 14. MAIN
# ============================================================

if __name__ == "__main__":

    validate_detector()

    print("=" * 70)
    print("RISK-LANGUAGE DETECTION ENGINE")
    print("=" * 70)

    samples = [

        "I enjoyed studying with my friends today.",

        "I feel completely hopeless.",

        "I am going to kill myself.",

        "I do not want to hurt myself.",

        "Maybe things will change someday.",

        "What should someone do if they want to kill themselves?",

        "My friend says he wants to hurt himself.",

        'My friend said, "I want to kill myself."',

        "I am fine, but I am going to kill myself.",

        (
            "I am going to kill myself. "
            "What should someone do if they want to kill themselves?"
        ),

        (
            "I am going to kill myself. "
            "I remember that I felt hopeless years ago."
        ),

        (
            "I am going to kill myself. "
            "My friend once said he felt hopeless."
        ),
    ]

    for index, sample in enumerate(samples, start=1):

        result = detect_risk_language(sample)

        print(f"\nTEST {index}")
        print("-" * 70)

        print("Input:", sample)

        print("Risk category:", result["risk_category"])

        print("Escalation:", result["escalation_level"])

        print("Human review:", result["requires_human_review"])

        print(
            "Detected indicators:",
            list(result["detected_indicators"].keys())
        )

        print(
            "Matched expressions:",
            result["explanation"]["matched_expression_count"]
        )

    print("\nDetector validation: PASSED")
    print("=" * 70)
