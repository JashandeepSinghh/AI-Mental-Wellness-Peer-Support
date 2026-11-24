
import csv
import json
from pathlib import Path

from scripts.model_development.rumination.rumination_detector import detect_rumination
from scripts.model_development.rumination.rumination_config import RUMINATION_OUTPUT_DIR


# ============================================================
# MANUALLY LABELLED TEST DATASET
# ============================================================

TEST_CASES = [

    # --------------------------------------------------------
    # 1. POSITIVE EXAMPLES
    # --------------------------------------------------------

    {
        "id": "POS001",
        "group": "positive",
        "expected": True,
        "text": "I keep thinking about what happened yesterday."
    },

    {
        "id": "POS002",
        "group": "positive",
        "expected": True,
        "text": "I cannot stop thinking about my past mistakes."
    },

    {
        "id": "POS003",
        "group": "positive",
        "expected": True,
        "text": "I keep replaying that conversation in my head."
    },

    {
        "id": "POS004",
        "group": "positive",
        "expected": True,
        "text": "Everything that happened was all my fault."
    },

    {
        "id": "POS005",
        "group": "positive",
        "expected": True,
        "text": "I keep worrying about what might happen in the future."
    },

    {
        "id": "POS006",
        "group": "positive",
        "expected": True,
        "text": "No matter what I do, I keep coming back to the same problem."
    },

    {
        "id": "POS007",
        "group": "positive",
        "expected": True,
        "text": "I keep blaming myself for everything."
    },

    {
        "id": "POS008",
        "group": "positive",
        "expected": True,
        "text": "I cannot get that incident out of my head."
    },


    # --------------------------------------------------------
    # 2. NEGATIVE EXAMPLES
    # --------------------------------------------------------

    {
        "id": "NEG001",
        "group": "negative",
        "expected": False,
        "text": "I am feeling sad today."
    },

    {
        "id": "NEG002",
        "group": "negative",
        "expected": False,
        "text": "My examination is scheduled for tomorrow."
    },

    {
        "id": "NEG003",
        "group": "negative",
        "expected": False,
        "text": "I went to college and attended my lectures."
    },

    {
        "id": "NEG004",
        "group": "negative",
        "expected": False,
        "text": "I am happy because I completed my assignment."
    },

    {
        "id": "NEG005",
        "group": "negative",
        "expected": False,
        "text": "Yesterday I watched a movie with my friends."
    },

    {
        "id": "NEG006",
        "group": "negative",
        "expected": False,
        "text": "I made a mistake, but I learned from it and moved forward."
    },


    # --------------------------------------------------------
    # 3. NEGATION EXAMPLES
    # --------------------------------------------------------

    {
        "id": "NEGATION001",
        "group": "negation",
        "expected": False,
        "text": "I do not keep thinking about what happened."
    },

    {
        "id": "NEGATION002",
        "group": "negation",
        "expected": False,
        "text": "I can stop thinking about my past mistakes."
    },

    {
        "id": "NEGATION003",
        "group": "negation",
        "expected": False,
        "text": "I am not constantly worrying about the future."
    },

    {
        "id": "NEGATION004",
        "group": "negation",
        "expected": False,
        "text": "I do not blame myself for what happened."
    },


    # --------------------------------------------------------
    # 4. THIRD-PERSON EXAMPLES
    # --------------------------------------------------------

    {
        "id": "THIRD001",
        "group": "third_person",
        "expected": False,
        "text": "My friend keeps thinking about what happened yesterday."
    },

    {
        "id": "THIRD002",
        "group": "third_person",
        "expected": False,
        "text": "She cannot stop thinking about her past mistakes."
    },

    {
        "id": "THIRD003",
        "group": "third_person",
        "expected": False,
        "text": "He keeps blaming himself for everything."
    },

    {
        "id": "THIRD004",
        "group": "third_person",
        "expected": False,
        "text": "They keep worrying about what might happen."
    },


    # --------------------------------------------------------
    # 5. AMBIGUOUS EXAMPLES
    # Excluded from binary classification metrics.
    # --------------------------------------------------------

    {
        "id": "AMB001",
        "group": "ambiguous",
        "expected": None,
        "text": "I was thinking about my examination yesterday."
    },

    {
        "id": "AMB002",
        "group": "ambiguous",
        "expected": None,
        "text": "What if I fail my examination?"
    },

    {
        "id": "AMB003",
        "group": "ambiguous",
        "expected": None,
        "text": "I should have prepared better for the examination."
    },

    {
        "id": "AMB004",
        "group": "ambiguous",
        "expected": None,
        "text": "I keep thinking about the new project."
    },

]


# ============================================================
# CONFUSION MATRIX METRICS
# ============================================================

def calculate_metrics(results):

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for item in results:

        expected = item["expected"]
        predicted = item["predicted"]

        if expected is None:
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
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
        if (tp + tn + fp + fn) > 0
        else 0.0
    )

    return {

        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,

        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),

    }


# ============================================================
# RUN ALL TEST CASES
# ============================================================

def run_test_suite():

    results = []

    for case in TEST_CASES:

        try:

            detection = detect_rumination(
                case["text"]
            )

            predicted = detection[
                "any_indicator_detected"
            ]

            detected_categories = detection[
                "detected_categories"
            ]

            total_matches = detection[
                "total_pattern_matches"
            ]

            error = None

        except Exception as exception:

            predicted = None
            detected_categories = []
            total_matches = 0
            error = str(exception)

        if case["expected"] is None:

            status = "AMBIGUOUS"

        elif predicted is None:

            status = "ERROR"

        elif predicted == case["expected"]:

            status = "PASS"

        else:

            status = "FAIL"

        results.append({

            "id": case["id"],
            "group": case["group"],
            "text": case["text"],
            "expected": case["expected"],
            "predicted": predicted,
            "status": status,
            "detected_categories": detected_categories,
            "total_matches": total_matches,
            "error": error,

        })

    return results


# ============================================================
# GROUP-WISE ANALYSIS
# ============================================================

def calculate_group_results(results):

    groups = {}

    for item in results:

        group = item["group"]

        groups.setdefault(
            group,
            {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0,
            }
        )

        groups[group]["total"] += 1

        if item["status"] == "PASS":
            groups[group]["passed"] += 1

        elif item["status"] == "FAIL":
            groups[group]["failed"] += 1

        elif item["status"] == "ERROR":
            groups[group]["errors"] += 1

    return groups


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results, metrics, group_results):

    output_dir = RUMINATION_OUTPUT_DIR / "test_suite"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # CSV RESULTS
    # --------------------------------------------------------

    csv_path = output_dir / "test_results.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "group",
                "text",
                "expected",
                "predicted",
                "status",
                "detected_categories",
                "total_matches",
                "error",
            ],
        )

        writer.writeheader()

        for item in results:

            row = item.copy()

            row["detected_categories"] = json.dumps(
                row["detected_categories"]
            )

            writer.writerow(row)

    # --------------------------------------------------------
    # JSON SUMMARY
    # --------------------------------------------------------

    summary = {

        "module": "Rumination Detection",

        "test_count": len(results),

        "metrics": metrics,

        "group_results": group_results,

        "limitations": [
            "Test examples are manually constructed.",
            "Metrics do not establish clinical validity.",
            "Ambiguous examples are excluded from binary metrics.",
            "The test set is small and not representative of real users."
        ],

    }

    json_path = output_dir / "test_summary.json"

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    return csv_path, json_path


# ============================================================
# DISPLAY TEST RESULTS
# ============================================================

def print_results(results, metrics, group_results):

    print("\n" + "=" * 75)
    print("RUMINATION DETECTOR TEST SUITE")
    print("=" * 75)

    print(
        f"{'ID':<15}"
        f"{'GROUP':<18}"
        f"{'EXPECTED':<12}"
        f"{'PREDICTED':<12}"
        f"{'STATUS':<12}"
    )

    print("-" * 75)

    for item in results:

        print(
            f"{item['id']:<15}"
            f"{item['group']:<18}"
            f"{str(item['expected']):<12}"
            f"{str(item['predicted']):<12}"
            f"{item['status']:<12}"
        )

        if item["status"] in {"FAIL", "ERROR"}:

            print(
                f"  Text: {item['text']}"
            )

            print(
                f"  Detected: {item['detected_categories']}"
            )

    print("\n" + "=" * 75)
    print("GROUP-WISE RESULTS")
    print("=" * 75)

    for group, details in group_results.items():

        print(f"\n{group.upper()}")

        print(f"Total: {details['total']}")
        print(f"Passed: {details['passed']}")
        print(f"Failed: {details['failed']}")
        print(f"Errors: {details['errors']}")

    print("\n" + "=" * 75)
    print("BINARY CLASSIFICATION METRICS")
    print("=" * 75)

    for name, value in metrics.items():

        print(
            f"{name.replace('_', ' ').title()}: {value}"
        )

    print("\n" + "=" * 75)

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

    ambiguous = sum(
        item["status"] == "AMBIGUOUS"
        for item in results
    )

    print(f"Total examples: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Ambiguous: {ambiguous}")

    print("=" * 75)


# ============================================================
# MAIN EXECUTION
# ============================================================

def test_rumination_detector_suite():
    results = run_test_suite()
    failed = [r for r in results if r["status"] == "FAIL"]
    assert len(failed) == 0, f"Failed tests: {failed}"


if __name__ == "__main__":

    results = run_test_suite()

    metrics = calculate_metrics(
        results
    )

    group_results = calculate_group_results(
        results
    )

    csv_path, json_path = save_results(
        results,
        metrics,
        group_results,
    )

    print_results(
        results,
        metrics,
        group_results,
    )

    print("\nSAVED OUTPUTS")

    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
