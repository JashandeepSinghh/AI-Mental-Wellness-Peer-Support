
"""
Confidence Module End-to-End Validation
========================================

Validates:
1. Configuration
2. Mathematical confidence engine
3. Multi-class analysis
4. Multi-label analysis
5. Cross-module aggregation
6. Missing-module handling
7. Risk safety requirements
8. Calibration declarations
9. Output consistency
"""

import json

from pathlib import Path

from scripts.model_development.confidence.confidence_config import (
    PROJECT_ROOT,
    SUPPORTED_MODULES,
    CALIBRATION_STATUS,
    SAFETY_PROBABILITY_AVAILABLE,
    CLINICAL_CERTAINTY_ESTABLISHED,
    validate_config,
)

from scripts.model_development.confidence.confidence_metrics import (
    validate_confidence_metrics,
    analyze_multiclass_logits,
    analyze_multilabel_logits,
)

from scripts.model_development.confidence.confidence_aggregator import (
    validate_aggregator,
    aggregate_all_modules,
)


# ============================================================
# 1. VALIDATION STATE
# ============================================================

validation_results = []


def record_check(name, passed, detail=""):

    validation_results.append({

        "check": name,

        "passed": bool(passed),

        "detail": str(detail)
    })

    status = "PASSED" if passed else "FAILED"

    print(f"[{status}] {name}")

    if detail:

        print(f"         {detail}")


# ============================================================
# 2. CONFIGURATION VALIDATION
# ============================================================

def validate_configuration():

    try:

        validate_config()

        record_check(
            "Configuration validation",
            True
        )

    except Exception as error:

        record_check(
            "Configuration validation",
            False,
            error
        )

    record_check(
        "Eight analytical modules registered",
        len(SUPPORTED_MODULES) == 8,
        f"Registered modules: {len(SUPPORTED_MODULES)}"
    )

    record_check(
        "Calibration correctly marked unavailable",
        CALIBRATION_STATUS == "not_calibrated",
        f"Status: {CALIBRATION_STATUS}"
    )


# ============================================================
# 3. MATHEMATICAL ENGINE VALIDATION
# ============================================================

def validate_metrics_engine():

    try:

        validate_confidence_metrics()

        record_check(
            "Mathematical engine smoke validation",
            True
        )

    except Exception as error:

        record_check(
            "Mathematical engine smoke validation",
            False,
            error
        )

    # --------------------------------------------------------
    # Multi-class analysis
    # --------------------------------------------------------

    try:

        result = analyze_multiclass_logits(

            logits=[2.0, 1.0, 0.1],

            class_names=[
                "negative",
                "neutral",
                "positive"
            ]
        )

        passed = (

            result["output_type"] == "multi_class"

            and result["is_calibrated"] is False

            and 0.0 <= result["normalized_entropy"] <= 1.0

        )

        record_check(
            "Multi-class confidence calculation",
            passed,
            f"Prediction: {result['predicted_class']}"
        )

    except Exception as error:

        record_check(
            "Multi-class confidence calculation",
            False,
            error
        )

    # --------------------------------------------------------
    # Multi-label analysis
    # --------------------------------------------------------

    try:

        result = analyze_multilabel_logits(

            logits=[2.0, -1.0, 0.0],

            label_names=[
                "joy",
                "anger",
                "sadness"
            ]
        )

        passed = (

            result["output_type"] == "multi_label"

            and result["is_calibrated"] is False

            and len(result["label_results"]) == 3

        )

        record_check(
            "Multi-label confidence calculation",
            passed,
            f"Labels analyzed: {result['label_count']}"
        )

    except Exception as error:

        record_check(
            "Multi-label confidence calculation",
            False,
            error
        )


# ============================================================
# 4. AGGREGATOR VALIDATION
# ============================================================

def validate_cross_module_aggregation():

    try:

        validate_aggregator()

        record_check(
            "Aggregator smoke validation",
            True
        )

    except Exception as error:

        record_check(
            "Aggregator smoke validation",
            False,
            error
        )

    # --------------------------------------------------------
    # Representative outputs
    # --------------------------------------------------------

    example_outputs = {

        "emotion": {

            "output_type": "multi_label",

            "mean_label_confidence": 0.78,

            "normalized_uncertainty": 0.42,

            "uncertainty_band": "moderate",

            "label_results": [

                {"label_name": "joy", "probability": 0.8}

            ],

            "selected_labels": ["joy"],

            "calibration_status": "not_calibrated"
        },

        "sentiment": {

            "output_type": "multi_class",

            "predicted_class": "negative",

            "confidence": 0.80,

            "normalized_entropy": 0.35,

            "prediction_margin": 0.60,

            "uncertainty_band": "moderate",

            "calibration_status": "not_calibrated"
        },

        "sentiment_bilstm": {

            "output_type": "multi_class",

            "predicted_class": "neutral",

            "confidence": 0.72,

            "normalized_entropy": 0.48,

            "prediction_margin": 0.31,

            "uncertainty_band": "moderate",

            "calibration_status": "not_calibrated"
        },

        "stress": {

            "output_type": "multi_class",

            "predicted_class": "stress",

            "confidence": 0.85,

            "normalized_entropy": 0.28,

            "prediction_margin": 0.70,

            "uncertainty_band": "moderate",

            "calibration_status": "not_calibrated"
        },

        "loneliness": {

            "output_type": "multi_class",

            "predicted_class": "non_lonely",

            "confidence": 0.90,

            "normalized_entropy": 0.20,

            "prediction_margin": 0.80,

            "uncertainty_band": "low",

            "calibration_status": "not_calibrated"
        },

        "rumination": {

            "detected_indicators": [],

            "ambiguity_flags": []
        },

        "risk": {

            "category": "no_explicit_risk_signal",

            "detected_indicators": [],

            "requires_human_review": False
        },

        "vad": {

            "coverage": 0.75,

            "scoring_term_count": 6
        }
    }

    try:

        result = aggregate_all_modules(
            example_outputs
        )

        record_check(
            "All eight modules aggregated",
            result["evaluated_module_count"] == 8,
            f"Evaluated: {result['evaluated_module_count']}/8"
        )

        record_check(
            "No artificial overall confidence",
            result["overall_confidence"] is None,
            "Overall confidence remains unavailable."
        )

        record_check(
            "All modules marked evaluated",
            result["not_evaluated_module_count"] == 0
        )

        record_check(
            "Risk review status propagated",
            result["human_review_required"] is False
        )

    except Exception as error:

        record_check(
            "Cross-module integration",
            False,
            error
        )


# ============================================================
# 5. SAFETY VALIDATION
# ============================================================

def validate_safety_requirements():

    record_check(
        "No safety probability exposed",
        SAFETY_PROBABILITY_AVAILABLE is False
    )

    record_check(
        "No clinical certainty claim",
        CLINICAL_CERTAINTY_ESTABLISHED is False
    )

    # --------------------------------------------------------
    # Missing risk evaluation
    # --------------------------------------------------------

    try:

        result = aggregate_all_modules({

            "sentiment": {

                "output_type": "multi_class",

                "predicted_class": "negative",

                "confidence": 0.8,

                "normalized_entropy": 0.3,

                "prediction_margin": 0.5,

                "calibration_status": "not_calibrated"
            }
        })

        risk_result = result["modules"]["risk"]

        passed = (

            risk_result["status"] == "not_evaluated"

            and result["risk_evaluation_status"] == "not_evaluated"

            and "does not establish" in (
                risk_result["safety_interpretation"]
            )

        )

        record_check(
            "Missing risk evaluation does not imply safety",
            passed
        )

    except Exception as error:

        record_check(
            "Missing risk evaluation does not imply safety",
            False,
            error
        )


# ============================================================
# 6. VALIDATION REPORT
# ============================================================

def save_validation_report():

    output_dir = (

        PROJECT_ROOT

        / "training"

        / "confidence"

        / "module_validation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    total = len(validation_results)

    passed = sum(
        item["passed"]
        for item in validation_results
    )

    failed = total - passed

    summary = {

        "module_name": "Confidence and Uncertainty Estimation",

        "module_version": "1.0.0",

        "total_checks": total,

        "passed": passed,

        "failed": failed,

        "status": "PASSED" if failed == 0 else "FAILED",

        "calibration_status": CALIBRATION_STATUS,

        "clinical_certainty_established": (
            CLINICAL_CERTAINTY_ESTABLISHED
        ),

        "safety_probability_available": (
            SAFETY_PROBABILITY_AVAILABLE
        ),

        "validation_results": validation_results
    }

    report_path = (

        output_dir

        / "confidence_module_validation_summary.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print("\n" + "=" * 70)

    print("VALIDATION SUMMARY")

    print("=" * 70)

    print(f"Total checks : {total}")

    print(f"Passed       : {passed}")

    print(f"Failed       : {failed}")

    print(f"Status       : {summary['status']}")

    print(f"\nValidation report:\n{report_path}")

    print("=" * 70)

    return failed == 0


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print("=" * 70)

    print("CONFIDENCE MODULE END-TO-END VALIDATION")

    print("=" * 70)

    validate_configuration()

    validate_metrics_engine()

    validate_cross_module_aggregation()

    validate_safety_requirements()

    success = save_validation_report()

    if success:

        print("\nCONFIDENCE MODULE VALIDATION: PASSED")

    else:

        print("\nCONFIDENCE MODULE VALIDATION: FAILED")

        raise SystemExit(1)


if __name__ == "__main__":

    main()
