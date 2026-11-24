
import json
from pprint import pprint

from backend.app.inference.inference_pipeline import InferencePipeline


def print_section(title, data):
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    pprint(data, sort_dicts=False, width=120)


def main():

    pipeline = InferencePipeline()

    text = (
        "I feel happy and grateful today, "
        "although I have been thinking about my future."
    )

    print("\n========== INFERENCE OUTPUT INSPECTION ==========")

    result = pipeline.analyze_text(text)

    report = result["confidence_report"]
    modules = result["modules"]

    # 1. Input
    print_section("INPUT TEXT", text)

    # 2. Overall confidence report
    print_section("CONFIDENCE REPORT SUMMARY", {
        "module_count": report["module_count"],
        "evaluated_module_count": report["evaluated_module_count"],
        "not_evaluated_module_count": report["not_evaluated_module_count"],
        "overall_confidence": report["overall_confidence"],
        "overall_uncertainty": report["overall_uncertainty"],
        "overall_confidence_calculated": report[
            "overall_confidence_calculated"
        ],
        "human_review_required": report["human_review_required"],
        "calibration_status": report["calibration_status"],
    })

    # 3. Individual analytical module outputs
    for module_name, module_output in modules.items():
        print_section(
            f"{module_name.upper()} MODULE OUTPUT",
            module_output,
        )

    # 4. Individual confidence report entries
    for module_name, module_report in report["modules"].items():
        print_section(
            f"{module_name.upper()} CONFIDENCE REPORT",
            module_report,
        )

    # 5. Structural verification
    assert report["module_count"] == 8
    assert len(report["modules"]) == 8
    assert report["evaluated_module_count"] == 8
    assert report["not_evaluated_module_count"] == 0

    assert (
        report["modules"]["sentiment_bilstm"]["status"]
        == "evaluated"
    )

    assert (
        report["modules"]["risk"]["status"]
        == "evaluated"
    )

    assert isinstance(report["human_review_required"], bool)

    assert report["overall_confidence"] is None
    assert report["overall_uncertainty"] is None
    assert report["overall_confidence_calculated"] is False

    assert (
        report["modules"]["risk"]["probability_available"]
        is False
    )

    # 6. Emotion threshold verification
    emotion = modules["emotion"]

    assert emotion["thresholds_applied"] is True

    assert len(
        emotion["confidence_analysis"]["label_results"]
    ) == 28

    # 7. Calibration verification
    for module_name in (
        "emotion",
        "sentiment",
        "sentiment_bilstm",
        "stress",
        "loneliness",
    ):
        assert (
            report["modules"][module_name]["is_calibrated"]
            is False
        )

    print("\n========== STRUCTURAL VALIDATION ==========")
    print("All eight modules registered: PASSED")
    print("All eight modules evaluated: PASSED")
    print("Zero modules not evaluated: PASSED")
    print("Risk review propagation: PASSED")
    print("No artificial overall confidence: PASSED")
    print("No safety probability: PASSED")
    print("All 28 emotion thresholds: PASSED")
    print("Calibration safeguards: PASSED")

    print("\n========== INFERENCE OUTPUT INSPECTION: PASSED ==========")


if __name__ == "__main__":
    main()
