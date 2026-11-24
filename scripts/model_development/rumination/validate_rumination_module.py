
"""
Complete Rumination Module Validation
-------------------------------------
Validates:
1. Single-message rumination detection
2. Conversation-level analysis
3. Semantic-analysis integration
4. Output structure and consistency
5. Invalid-input handling

This script uses constructed test inputs.
It is an engineering validation suite, not clinical validation.
"""

import json
import traceback

from pathlib import Path
from datetime import datetime
from typing import Any

from scripts.model_development.rumination.rumination_detector import (
    detect_rumination,
)

from scripts.model_development.rumination.rumination_conversation_analyzer import (
    analyze_conversation,
)


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path(
    "training/rumination/rule_based/module_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TEST_RESULTS: list[dict[str, Any]] = []


# ============================================================
# TEST RESULT MANAGEMENT
# ============================================================

def record_result(
    test_id: str,
    category: str,
    description: str,
    passed: bool,
    details: str = "",
) -> None:

    TEST_RESULTS.append({
        "test_id": test_id,
        "category": category,
        "description": description,
        "status": "PASS" if passed else "FAIL",
        "details": details,
    })

    print(
        f"{test_id:<12}"
        f"{category:<25}"
        f"{'PASS' if passed else 'FAIL':<10}"
        f"{description}"
    )


# ============================================================
# 1. SINGLE-MESSAGE DETECTION TESTS
# ============================================================

def test_single_message_detection() -> None:

    print("\n" + "=" * 85)
    print("1. SINGLE-MESSAGE DETECTION")
    print("=" * 85)

    test_cases = [

        {
            "id": "SINGLE001",
            "description": "Repetitive-thinking input",
            "text": (
                "I keep thinking about the same mistake "
                "and cannot stop replaying it."
            ),
        },

        {
            "id": "SINGLE002",
            "description": "Ordinary neutral input",
            "text": (
                "I went to the university library "
                "and studied computer science."
            ),
        },

        {
            "id": "SINGLE003",
            "description": "Past-focused input",
            "text": (
                "I keep replaying that conversation "
                "in my head every day."
            ),
        },

        {
            "id": "SINGLE004",
            "description": "Future-worry input",
            "text": (
                "I keep worrying about what might "
                "happen in the future."
            ),
        },

    ]

    for case in test_cases:

        try:

            result = detect_rumination(
                case["text"]
            )

            required_keys = {
                "module",
                "analysis_type",
                "any_indicator_detected",
                "detected_categories",
                "total_pattern_matches",
                "categories",
                "interpretation",
                "limitations",
            }

            valid_structure = required_keys.issubset(
                result.keys()
            )

            valid_types = (
                isinstance(
                    result["any_indicator_detected"],
                    bool,
                )
                and isinstance(
                    result["detected_categories"],
                    list,
                )
                and isinstance(
                    result["total_pattern_matches"],
                    int,
                )
                and isinstance(
                    result["categories"],
                    dict,
                )
                and isinstance(
                    result["limitations"],
                    list,
                )
            )

            passed = (
                valid_structure
                and valid_types
            )

            record_result(
                case["id"],
                "Single-message",
                case["description"],
                passed,
                "Output structure and data types validated."
                if passed
                else "Unexpected output structure or data types.",
            )

        except Exception as error:

            record_result(
                case["id"],
                "Single-message",
                case["description"],
                False,
                str(error),
            )


# ============================================================
# 2. CONVERSATION-LEVEL TESTS
# ============================================================

def test_conversation_analysis() -> None:

    print("\n" + "=" * 85)
    print("2. CONVERSATION-LEVEL ANALYSIS")
    print("=" * 85)

    conversation = [

        {
            "message_id": "MSG001",
            "text": (
                "I made a mistake during my examination "
                "and I feel bad about it."
            ),
        },

        {
            "message_id": "MSG002",
            "text": (
                "I keep remembering what happened "
                "during that examination."
            ),
        },

        {
            "message_id": "MSG003",
            "text": (
                "I cannot stop thinking about that mistake."
            ),
        },

    ]

    try:

        result = analyze_conversation(
            conversation
        )

        required_keys = {
            "module",
            "analysis_type",
            "message_count",
            "messages_with_indicators",
            "message_results",
            "repeated_concerns",
            "topic_alignment",
            "category_frequency",
            "recurring_rumination_signal",
            "interpretation",
            "limitations",
            "semantic_analysis",
        }

        valid_structure = required_keys.issubset(
            result.keys()
        )

        valid_message_count = (
            result["message_count"]
            == len(conversation)
        )

        valid_message_results = (
            isinstance(
                result["message_results"],
                list,
            )
            and len(result["message_results"])
            == len(conversation)
        )

        valid_signal = isinstance(
            result["recurring_rumination_signal"],
            bool,
        )

        valid_semantic_output = isinstance(
            result["semantic_analysis"],
            dict,
        )

        passed = all([
            valid_structure,
            valid_message_count,
            valid_message_results,
            valid_signal,
            valid_semantic_output,
        ])

        record_result(
            "CONV001",
            "Conversation",
            "Complete conversation output validation",
            passed,
            (
                "Message count, result structure, "
                "signal type and semantic output validated."
                if passed
                else "One or more conversation output checks failed."
            ),
        )

        # Save a compact analysis report.
        report = {
            "message_count": result["message_count"],
            "messages_with_indicators": (
                result["messages_with_indicators"]
            ),
            "recurring_rumination_signal": (
                result["recurring_rumination_signal"]
            ),
            "semantic_analysis": (
                result["semantic_analysis"]
            ),
            "interpretation": result["interpretation"],
        }

        with open(
            OUTPUT_DIR / "conversation_validation.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
                ensure_ascii=False,
                default=str,
            )

    except Exception as error:

        record_result(
            "CONV001",
            "Conversation",
            "Complete conversation output validation",
            False,
            str(error),
        )

        traceback.print_exc()


# ============================================================
# 3. SEMANTIC INTEGRATION TEST
# ============================================================

def test_semantic_integration() -> None:

    print("\n" + "=" * 85)
    print("3. SEMANTIC INTEGRATION")
    print("=" * 85)

    messages = [

        {
            "message_id": "SEM001",
            "text": (
                "I made a serious mistake in my examination "
                "and I feel upset."
            ),
        },

        {
            "message_id": "SEM002",
            "text": (
                "I keep replaying that exam incident "
                "in my mind."
            ),
        },

        {
            "message_id": "SEM003",
            "text": (
                "I cannot stop thinking about what happened "
                "during the test."
            ),
        },

    ]

    try:

        result = analyze_conversation(
            messages,
            enable_semantic_analysis=True,
        )

        semantic_result = result.get(
            "semantic_analysis"
        )

        passed = (
            isinstance(semantic_result, dict)
            and len(semantic_result) > 0
        )

        record_result(
            "SEM001",
            "Semantic",
            "Semantic analysis returns structured output",
            passed,
            (
                "Semantic analysis returned a non-empty dictionary."
                if passed
                else "Semantic analysis output was missing or empty."
            ),
        )

    except Exception as error:

        record_result(
            "SEM001",
            "Semantic",
            "Semantic analysis returns structured output",
            False,
            str(error),
        )

        traceback.print_exc()


# ============================================================
# 4. INVALID INPUT TESTS
# ============================================================

def test_invalid_inputs() -> None:

    print("\n" + "=" * 85)
    print("4. INVALID INPUT HANDLING")
    print("=" * 85)

    invalid_cases = [

        {
            "id": "INPUT001",
            "description": "Non-string input",
            "function": lambda: detect_rumination(12345),
            "expected": TypeError,
        },

        {
            "id": "INPUT002",
            "description": "Text shorter than minimum length",
            "function": lambda: detect_rumination("a"),
            "expected": ValueError,
        },

        {
            "id": "INPUT003",
            "description": "Text exceeding maximum length",
            "function": lambda: detect_rumination("x" * 20001),
            "expected": ValueError,
        },

        {
            "id": "INPUT004",
            "description": "Conversation with fewer than two messages",
            "function": lambda: analyze_conversation([
                {
                    "message_id": "ONE",
                    "text": "I am feeling worried today.",
                }
            ]),
            "expected": ValueError,
        },

        {
            "id": "INPUT005",
            "description": "Conversation with non-list input",
            "function": lambda: analyze_conversation("invalid"),
            "expected": TypeError,
        },

        {
            "id": "INPUT006",
            "description": "Conversation with empty message",
            "function": lambda: analyze_conversation([
                {
                    "message_id": "EMPTY",
                    "text": "",
                },
                {
                    "message_id": "VALID",
                    "text": "I am worried about my examination.",
                },
            ]),
            "expected": ValueError,
        },

    ]

    for case in invalid_cases:

        try:

            case["function"]()

            record_result(
                case["id"],
                "Input handling",
                case["description"],
                False,
                "Expected exception was not raised.",
            )

        except case["expected"]:

            record_result(
                case["id"],
                "Input handling",
                case["description"],
                True,
                "Expected exception raised correctly.",
            )

        except Exception as error:

            record_result(
                case["id"],
                "Input handling",
                case["description"],
                False,
                (
                    f"Wrong exception: "
                    f"{type(error).__name__}: {error}"
                ),
            )


# ============================================================
# 5. OUTPUT CONSISTENCY
# ============================================================

def test_output_consistency() -> None:

    print("\n" + "=" * 85)
    print("5. OUTPUT CONSISTENCY")
    print("=" * 85)

    messages = [

        {
            "message_id": "CHECK001",
            "text": (
                "I keep thinking about the mistake "
                "I made during my examination."
            ),
        },

        {
            "message_id": "CHECK002",
            "text": (
                "That examination mistake keeps "
                "coming back into my thoughts."
            ),
        },

    ]

    try:

        first_result = analyze_conversation(
            messages
        )

        second_result = analyze_conversation(
            messages
        )

        first_signal = first_result[
            "recurring_rumination_signal"
        ]

        second_signal = second_result[
            "recurring_rumination_signal"
        ]

        first_count = first_result[
            "message_count"
        ]

        second_count = second_result[
            "message_count"
        ]

        passed = (
            first_signal == second_signal
            and first_count == second_count
        )

        record_result(
            "CONS001",
            "Consistency",
            "Repeated identical input gives consistent outputs",
            passed,
            (
                "Signal and message count remained consistent."
                if passed
                else "Repeated input produced inconsistent results."
            ),
        )

    except Exception as error:

        record_result(
            "CONS001",
            "Consistency",
            "Repeated identical input gives consistent outputs",
            False,
            str(error),
        )


# ============================================================
# 6. FINAL SUMMARY
# ============================================================

def generate_final_summary() -> None:

    total = len(TEST_RESULTS)

    passed = sum(
        item["status"] == "PASS"
        for item in TEST_RESULTS
    )

    failed = total - passed

    summary = {

        "module": "Rumination Detection",

        "validation_type": "Engineering module validation",

        "timestamp": datetime.now().isoformat(),

        "total_tests": total,

        "passed": passed,

        "failed": failed,

        "pass_percentage": round(
            (passed / total) * 100,
            2,
        ) if total else 0,

        "results": TEST_RESULTS,

        "qualification": (
            "Constructed test cases validate implementation "
            "behavior only. They do not establish clinical "
            "validity or real-world detection accuracy."
        ),

    }

    output_file = (
        OUTPUT_DIR / "rumination_module_validation_summary.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 85)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 85)

    print(f"Total tests : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")

    print(
        f"Pass rate   : {summary['pass_percentage']}%"
    )

    print(f"\nSaved report: {output_file.resolve()}")

    print("=" * 85)

    if failed:
        print("RESULT: VALIDATION FAILED")
    else:
        print("RESULT: ALL VALIDATION CHECKS PASSED")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main() -> None:

    print("\n" + "=" * 85)
    print("COMPLETE RUMINATION MODULE VALIDATION")
    print("=" * 85)

    test_single_message_detection()

    test_conversation_analysis()

    test_semantic_integration()

    test_invalid_inputs()

    test_output_consistency()

    generate_final_summary()


if __name__ == "__main__":

    main()
