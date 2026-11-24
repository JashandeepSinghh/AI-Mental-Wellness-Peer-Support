
import json
from pathlib import Path

from scripts.model_development.rumination.rumination_conversation_analyzer import (
    analyze_conversation,
)

from scripts.model_development.rumination.rumination_config import RUMINATION_OUTPUT_DIR


# ============================================================
# CONVERSATION TEST DATASET
# ============================================================

TEST_CASES = [

    # --------------------------------------------------------
    # 1. REPEATED CONCERN + RUMINATION LANGUAGE
    # --------------------------------------------------------

    {
        "id": "CONV001",
        "group": "repeated_rumination",
        "expected": True,

        "messages": [
            {
                "message_id": "m1",
                "text": "I made a mistake during my examination."
            },
            {
                "message_id": "m2",
                "text": "I keep remembering what happened in that examination."
            },
            {
                "message_id": "m3",
                "text": "I cannot stop thinking about that mistake."
            },
        ],
    },

    # --------------------------------------------------------
    # 2. REPEATED TOPIC WITHOUT RUMINATION LANGUAGE
    # --------------------------------------------------------

    {
        "id": "CONV002",
        "group": "ordinary_topic_repetition",
        "expected": False,

        "messages": [
            {
                "message_id": "m1",
                "text": "My examination is scheduled for Monday."
            },
            {
                "message_id": "m2",
                "text": "I need to prepare for my examination."
            },
            {
                "message_id": "m3",
                "text": "The examination syllabus has five units."
            },
        ],
    },

    # --------------------------------------------------------
    # 3. RUMINATION LANGUAGE WITHOUT REPEATED CONCERN
    # --------------------------------------------------------

    {
        "id": "CONV003",
        "group": "isolated_rumination_language",
        "expected": False,

        "messages": [
            {
                "message_id": "m1",
                "text": "I keep thinking about yesterday."
            },
            {
                "message_id": "m2",
                "text": "I attended my computer science lecture."
            },
        ],
    },

    # --------------------------------------------------------
    # 4. UNRELATED MESSAGES WITHOUT RUMINATION INDICATORS
    # --------------------------------------------------------

    {
        "id": "CONV004",
        "group": "unrelated_conversation",
        "expected": False,

        "messages": [
            {
                "message_id": "m1",
                "text": "I played football with my friends."
            },
            {
                "message_id": "m2",
                "text": "We watched a movie yesterday."
            },
            {
                "message_id": "m3",
                "text": "I completed my programming assignment."
            },
        ],
    },

    # --------------------------------------------------------
    # 5. MULTIPLE RUMINATION INDICATORS ACROSS TURNS
    # --------------------------------------------------------

    {
        "id": "CONV005",
        "group": "multiple_rumination_indicators",
        "expected": True,

        "messages": [
            {
                "message_id": "m1",
                "text": "I keep thinking about my mistake."
            },
            {
                "message_id": "m2",
                "text": "I keep blaming myself for that mistake."
            },
            {
                "message_id": "m3",
                "text": "I cannot stop thinking about my mistake."
            },
        ],
    },

    # --------------------------------------------------------
    # 6. REPEATED THIRD-PERSON TOPIC
    # --------------------------------------------------------

    {
        "id": "CONV006",
        "group": "third_person_conversation",
        "expected": False,

        "messages": [
            {
                "message_id": "m1",
                "text": "My friend keeps thinking about his examination."
            },
            {
                "message_id": "m2",
                "text": "He keeps worrying about his examination."
            },
        ],
    },

    # --------------------------------------------------------
    # 7. NEGATED RUMINATION EXPRESSIONS
    # --------------------------------------------------------

    {
        "id": "CONV007",
        "group": "negated_expressions",
        "expected": False,

        "messages": [
            {
                "message_id": "m1",
                "text": "I do not keep thinking about my examination."
            },
            {
                "message_id": "m2",
                "text": "I am not constantly worrying about my examination."
            },
        ],
    },

]


# ============================================================
# EXECUTE CONVERSATION TESTS
# ============================================================

def run_tests():

    results = []

    for case in TEST_CASES:

        try:

            analysis = analyze_conversation(
                case["messages"]
            )

            predicted = analysis[
                "recurring_rumination_signal"
            ]

            message_count = analysis[
                "messages_with_indicators"
            ]

            repeated_keyword_count = analysis[
                "repeated_concerns"
            ]["repeated_keyword_count"]

            repeated_pair_count = analysis[
                "repeated_concerns"
            ]["repeated_pair_count"]

            error = None

        except Exception as exception:

            predicted = None
            message_count = 0
            repeated_keyword_count = 0
            repeated_pair_count = 0
            error = str(exception)

        if predicted is None:

            status = "ERROR"

        elif predicted == case["expected"]:

            status = "PASS"

        else:

            status = "FAIL"

        results.append({

            "id": case["id"],

            "group": case["group"],

            "expected": case["expected"],

            "predicted": predicted,

            "status": status,

            "message_count": len(
                case["messages"]
            ),

            "messages_with_indicators": message_count,

            "repeated_keyword_count": repeated_keyword_count,

            "repeated_pair_count": repeated_pair_count,

            "error": error,

        })

    return results


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(results):

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for item in results:

        expected = item["expected"]
        predicted = item["predicted"]

        if predicted is None:
            continue

        if expected and predicted:
            tp += 1

        elif not expected and not predicted:
            tn += 1

        elif not expected and predicted:
            fp += 1

        elif expected and not predicted:
            fn += 1

    precision = (
        tp / (tp + fp)
        if tp + fp else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
        if tp + tn + fp + fn else 0.0
    )

    return {

        "true_positive": tp,

        "true_negative": tn,

        "false_positive": fp,

        "false_negative": fn,

        "accuracy": round(accuracy, 4),

        "precision": round(precision, 4),

        "recall": round(recall, 4),

        "f1_score": round(f1, 4),

    }


# ============================================================
# SAVE TEST RESULTS
# ============================================================

def save_results(results, metrics):

    output_dir = (
        RUMINATION_OUTPUT_DIR
        / "conversation_test_suite"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {

        "module": "Rumination Detection",

        "analysis_type": "conversation_level",

        "total_tests": len(results),

        "metrics": metrics,

        "results": results,

        "limitations": [

            "The test cases are manually constructed.",

            "The test set is small and not representative of real users.",

            "Lexical overlap does not establish semantic equivalence.",

            "Passing these tests does not establish clinical validity.",

        ],

    }

    output_path = (
        output_dir / "conversation_test_summary.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    return output_path


# ============================================================
# DISPLAY RESULTS
# ============================================================

def print_results(results, metrics):

    print("\n" + "=" * 85)
    print("CONVERSATION-LEVEL RUMINATION TEST SUITE")
    print("=" * 85)

    print(
        f"{'ID':<15}"
        f"{'GROUP':<35}"
        f"{'EXPECTED':<12}"
        f"{'PREDICTED':<12}"
        f"{'STATUS':<10}"
    )

    print("-" * 85)

    for item in results:

        print(
            f"{item['id']:<15}"
            f"{item['group']:<35}"
            f"{str(item['expected']):<12}"
            f"{str(item['predicted']):<12}"
            f"{item['status']:<10}"
        )

        if item["status"] in {"FAIL", "ERROR"}:

            print(
                f"  Messages with indicators: "
                f"{item['messages_with_indicators']}"
            )

            print(
                f"  Repeated keywords: "
                f"{item['repeated_keyword_count']}"
            )

            print(
                f"  Repeated pairs: "
                f"{item['repeated_pair_count']}"
            )

            print(
                f"  Error: {item['error']}"
            )

    print("\n" + "=" * 85)
    print("CONFUSION MATRIX METRICS")
    print("=" * 85)

    for name, value in metrics.items():

        print(
            f"{name.replace('_', ' ').title()}: {value}"
        )

    passed = sum(
        item["status"] == "PASS"
        for item in results
    )

    failed = sum(
        item["status"] == "FAIL"
        for item in results
    )

    errors = sum(
        item["status"] == "ERROR"
        for item in results
    )

    print("\nTEST SUMMARY")

    print(f"Total: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")

    print("=" * 85)


# ============================================================
# MAIN EXECUTION
# ============================================================

def test_rumination_conversation_suite():
    results = run_tests()
    failed = [r for r in results if r["status"] == "FAIL"]
    assert len(failed) == 0, f"Failed tests: {failed}"


if __name__ == "__main__":

    results = run_tests()

    metrics = calculate_metrics(
        results
    )

    output_path = save_results(
        results,
        metrics,
    )

    print_results(
        results,
        metrics,
    )

    print(
        f"\nSaved summary: {output_path}"
    )
