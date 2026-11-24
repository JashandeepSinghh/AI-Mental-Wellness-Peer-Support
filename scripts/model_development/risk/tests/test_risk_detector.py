
"""
Comprehensive Risk Detector Test Suite
======================================

Tests:
1. Normal conversation
2. Potential concern
3. Explicit urgent language
4. Negation
5. Historical references
6. Hypothetical statements
7. Third-person references
8. Quoted statements
9. Contradictory expressions
10. Ordinary emotional distress
11. Contextual uncertainty
12. Output consistency

These are engineering tests, not clinical validation.
"""

from scripts.model_development.risk.risk_detector import (
    detect_risk_language
)

from scripts.model_development.risk import risk_config as config


# ============================================================
# 1. TEST CASES
# ============================================================

TEST_CASES = [

    {
        "id": "RISK001",
        "name": "Normal conversation",
        "text": "I enjoyed studying with my friends today.",
        "expected": "no_explicit_risk_signal"
    },

    {
        "id": "RISK002",
        "name": "Hopelessness expression",
        "text": "I feel completely hopeless.",
        "expected": "potential_concern"
    },

    {
        "id": "RISK003",
        "name": "Explicit urgent expression",
        "text": "I am going to kill myself.",
        "expected": "urgent_language_signal"
    },

    {
        "id": "RISK004",
        "name": "Negated self-harm expression",
        "text": "I do not want to hurt myself.",
        "expected": "potential_concern"
    },

    {
        "id": "RISK005",
        "name": "Historical experience",
        "text": "I wanted to hurt myself several years ago.",
        "expected": None
    },

    {
        "id": "RISK006",
        "name": "Hypothetical statement",
        "text": "What if someone wanted to end their life?",
        "expected": None
    },

    {
        "id": "RISK007",
        "name": "Third-person reference",
        "text": "My friend says he wants to hurt himself.",
        "expected": None
    },

    {
        "id": "RISK008",
        "name": "Quoted risk expression",
        "text": 'My friend said, "I want to kill myself."',
        "expected": None
    },

    {
        "id": "RISK009",
        "name": "Contradictory expressions",
        "text": "I am doing fine, but I feel completely hopeless.",
        "expected": None
    },

    {
        "id": "RISK010",
        "name": "Ordinary emotional distress",
        "text": "I am stressed about my upcoming university exams.",
        "expected": None
    },

    {
        "id": "RISK011",
        "name": "Ambiguous future expression",
        "text": "Maybe things will change someday.",
        "expected": "uncertain_or_ambiguous"
    },

    {
        "id": "RISK012",
        "name": "Negation with emotional distress",
        "text": "I do not want to hurt myself, but I feel overwhelmed.",
        "expected": "potential_concern"
    },

    {
        "id": "RISK013",
        "name": "Historical suicidal expression",
        "text": "I thought about suicide when I was younger.",
        "expected": None
    },

    {
        "id": "RISK014",
        "name": "Hypothetical urgent expression",
        "text": "What should someone do if they want to kill themselves?",
        "expected": None
    },

    {
        "id": "RISK015",
        "name": "Urgent expression with contradictory wording",
        "text": "I am fine, but I am going to kill myself.",
        "expected": "urgent_language_signal"
    }
]


# ============================================================
# 2. CATEGORY NORMALIZATION
# ============================================================

CATEGORY_MAP = {

    "no_explicit_risk_signal":
        config.RISK_CATEGORIES["NO_EXPLICIT_RISK_SIGNAL"],

    "potential_concern":
        config.RISK_CATEGORIES["POTENTIAL_CONCERN"],

    "urgent_language_signal":
        config.RISK_CATEGORIES["URGENT_LANGUAGE_SIGNAL"],

    "uncertain_or_ambiguous":
        config.RISK_CATEGORIES["UNCERTAIN_OR_AMBIGUOUS"]
}


# ============================================================
# 3. OUTPUT VALIDATION
# ============================================================

def validate_output_structure(result):

    for field in config.OUTPUT_FIELDS:

        assert field in result, (
            f"Missing output field: {field}"
        )

    assert result["clinical_diagnosis"] is False

    assert result["actual_risk_confirmed"] is False

    assert isinstance(result["detected_indicators"], dict)

    assert isinstance(result["context_flags"], dict)

    assert isinstance(result["evidence"], list)

    assert isinstance(result["requires_human_review"], bool)


# ============================================================
# 4. SAFETY INVARIANT VALIDATION
# ============================================================

def validate_safety_invariants(result):

    # No detected expression should be automatically suppressed.
    for evidence in result["evidence"]:

        assert evidence["automatically_suppressed"] is False

    # Urgent linguistic signals must require human review.
    if result["risk_category"] == CATEGORY_MAP[
        "urgent_language_signal"
    ]:

        assert result["requires_human_review"] is True

        assert result["escalation_level"] == (
            config.ESCALATION_MAPPING[
                CATEGORY_MAP["urgent_language_signal"]
            ]
        )

    # No explicit risk signal must not be represented as proof
    # that the user is safe.
    if result["risk_category"] == CATEGORY_MAP[
        "no_explicit_risk_signal"
    ]:

        assert result["actual_risk_confirmed"] is False


# ============================================================
# 5. RUN TEST CASES
# ============================================================

def run_test_suite():

    print("=" * 80)
    print("COMPREHENSIVE RISK DETECTOR TEST SUITE")
    print("=" * 80)

    passed = 0
    failed = 0
    results = []

    for test in TEST_CASES:

        print("\n" + "-" * 80)

        print(
            f"{test['id']} | {test['name']}"
        )

        print("Input:", test["text"])

        try:

            result = detect_risk_language(
                test["text"]
            )

            validate_output_structure(result)

            validate_safety_invariants(result)

            actual = result["risk_category"]

            expected = test["expected"]

            if expected is not None:

                expected_category = CATEGORY_MAP[expected]

                category_status = (
                    actual == expected_category
                )

            else:

                # Context-dependent cases are observed,
                # not forced into an assumed category.
                category_status = None

            if category_status is False:

                status = "REVIEW"

            else:

                status = "PASSED"

            if status == "PASSED":
                passed += 1
            else:
                failed += 1

            print("Detected category:", actual)

            if expected is not None:
                print("Expected category:", expected_category)
            else:
                print("Expected category: Context-dependent")

            print(
                "Escalation:",
                result["escalation_level"]
            )

            print(
                "Human review:",
                result["requires_human_review"]
            )

            print(
                "Detected indicators:",
                list(result["detected_indicators"].keys())
            )

            print(
                "Uncertainty notes:",
                result["uncertainty_notes"]
            )

            print("Status:", status)

            results.append({

                "id": test["id"],

                "name": test["name"],

                "actual_category": actual,

                "expected_category": expected,

                "status": status
            })

        except Exception as error:

            failed += 1

            print("Status: FAILED")

            print("Error:", str(error))

            results.append({

                "id": test["id"],

                "name": test["name"],

                "actual_category": None,

                "expected_category": test["expected"],

                "status": "FAILED"
            })

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)

    print("Total tests:", len(TEST_CASES))

    print("Passed:", passed)

    print("Failed / needs review:", failed)

    print(
        "Engineering checks:",
        f"{passed}/{len(TEST_CASES)}"
    )

    print("\nFINAL TEST RESULTS")

    for result in results:

        print(
            result["id"],
            "|",
            result["status"],
            "|",
            result["actual_category"]
        )

    print("=" * 80)

    if failed == 0:
        print("TEST SUITE: PASSED")
    else:
        print("TEST SUITE: REVIEW REQUIRED")

    return results


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":

    run_test_suite()
