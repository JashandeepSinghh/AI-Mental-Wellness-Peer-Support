
"""
Cross-Module Confidence and Uncertainty Aggregator
==================================================

Aggregates outputs from all eight analytical modules.

Classification:
    emotion, sentiment, sentiment_bilstm, stress, loneliness

Rule-based:
    rumination, risk

Lexicon-based:
    vad

No overall confidence or safety probability is calculated.
Missing modules remain not_evaluated.
Missing risk evaluation is represented as unknown.
"""

from datetime import datetime, timezone

from .confidence_config import (
    SUPPORTED_MODULES,
    ROUNDING_PRECISION,
    CALIBRATION_STATUS,
    validate_config,
)


# ============================================================
# 1. MODULE GROUPS
# ============================================================

CLASSIFICATION_MODULES = {
    "emotion",
    "sentiment",
    "sentiment_bilstm",
    "stress",
    "loneliness",
}

RULE_BASED_MODULES = {
    "rumination",
    "risk",
}

LEXICON_MODULES = {
    "vad",
}


# ============================================================
# 2. VALIDATION HELPERS
# ============================================================

def validate_module_name(module_name):

    if not isinstance(module_name, str):
        raise ValueError("Module name must be a string.")

    if module_name not in SUPPORTED_MODULES:
        raise ValueError(
            f"Unsupported analytical module: {module_name}"
        )

    return module_name


def validate_module_output(output):

    if not isinstance(output, dict):
        raise ValueError(
            "Each module output must be a dictionary."
        )

    return output


def validate_unit_interval(value, field_name):

    if isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be numeric."
        )

    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} must be numeric."
        ) from error

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{field_name} must be between zero and one."
        )

    return value


# ============================================================
# 3. CLASSIFICATION OUTPUT AGGREGATION
# ============================================================

def aggregate_classification_output(module_name, output):

    output = validate_module_output(output)

    output_type = output.get("output_type")

    # --------------------------------------------------------
    # Multi-class classification
    # --------------------------------------------------------

    if output_type == "multi_class":

        required_fields = {
            "confidence",
            "normalized_entropy",
            "prediction_margin",
            "calibration_status",
        }

        missing = required_fields - set(output)

        if missing:
            raise ValueError(
                f"{module_name}: missing fields {sorted(missing)}"
            )

        confidence = validate_unit_interval(
            output["confidence"],
            "confidence",
        )

        uncertainty = validate_unit_interval(
            output["normalized_entropy"],
            "normalized_entropy",
        )

        margin = validate_unit_interval(
            output["prediction_margin"],
            "prediction_margin",
        )

        return {
            "module_name": module_name,
            "method": "multi_class_classification",
            "status": "evaluated",
            "predicted_class": output.get("predicted_class"),
            "confidence": round(
                confidence,
                ROUNDING_PRECISION,
            ),
            "uncertainty": round(
                uncertainty,
                ROUNDING_PRECISION,
            ),
            "prediction_margin": round(
                margin,
                ROUNDING_PRECISION,
            ),
            "uncertainty_band": output.get(
                "uncertainty_band"
            ),
            "calibration_status": output[
                "calibration_status"
            ],
            "is_calibrated": False,
        }

    # --------------------------------------------------------
    # Multi-label classification
    # --------------------------------------------------------

    if output_type == "multi_label":

        required_fields = {
            "mean_label_confidence",
            "normalized_uncertainty",
            "label_results",
            "calibration_status",
        }

        missing = required_fields - set(output)

        if missing:
            raise ValueError(
                f"{module_name}: missing fields {sorted(missing)}"
            )

        confidence = validate_unit_interval(
            output["mean_label_confidence"],
            "mean_label_confidence",
        )

        uncertainty = validate_unit_interval(
            output["normalized_uncertainty"],
            "normalized_uncertainty",
        )

        if not isinstance(output["label_results"], list):
            raise ValueError(
                "label_results must be a list."
            )

        return {
            "module_name": module_name,
            "method": "multi_label_classification",
            "status": "evaluated",
            "confidence": round(
                confidence,
                ROUNDING_PRECISION,
            ),
            "uncertainty": round(
                uncertainty,
                ROUNDING_PRECISION,
            ),
            "uncertainty_band": output.get(
                "uncertainty_band"
            ),
            "label_count": len(output["label_results"]),
            "selected_labels": output.get(
                "selected_labels",
                [],
            ),
            "calibration_status": output[
                "calibration_status"
            ],
            "is_calibrated": False,
        }

    raise ValueError(
        f"{module_name}: unsupported output type '{output_type}'"
    )


# ============================================================
# 4. RULE-BASED OUTPUT AGGREGATION
# ============================================================

def aggregate_rule_based_output(module_name, output):

    output = validate_module_output(output)

    # --------------------------------------------------------
    # Risk module
    # --------------------------------------------------------

    if module_name == "risk":

        category = output.get(
            "category",
            output.get("risk_category"),
        )

        human_review = output.get(
            "requires_human_review"
        )

        if not isinstance(human_review, bool):
            raise ValueError(
                "Evaluated risk output must include a boolean "
                "requires_human_review field."
            )

        indicators = output.get(
            "detected_indicators",
            output.get("indicators", []),
        )

        # The actual risk detector may return a dictionary.
        # Preserve that structure rather than discarding evidence.
        if not isinstance(indicators, (list, dict)):
            raise ValueError(
                "Risk indicators must be a list or dictionary."
            )

        return {
            "module_name": "risk",
            "method": "context_aware_rules",
            "status": "evaluated",
            "risk_category": category,
            "detected_indicators": indicators,
            "requires_human_review": human_review,
            "confidence": None,
            "uncertainty": None,
            "probability_available": False,
            "clinical_certainty": False,
        }

    # --------------------------------------------------------
    # Rumination module
    # --------------------------------------------------------

    indicators = output.get(
        "detected_indicators",
        output.get("indicators", []),
    )

    if not isinstance(indicators, list):
        raise ValueError(
            "Rumination indicators must be a list."
        )

    ambiguity_flags = output.get(
        "ambiguity_flags",
        [],
    )

    if not isinstance(ambiguity_flags, list):
        raise ValueError(
            "Ambiguity flags must be a list."
        )

    return {
        "module_name": "rumination",
        "method": "linguistic_rule_analysis",
        "status": "evaluated",
        "detected_indicators": indicators,
        "ambiguity_flags": ambiguity_flags,
        "confidence": None,
        "uncertainty": None,
        "probability_available": False,
        "diagnosis_established": False,
    }


# ============================================================
# 5. VAD OUTPUT AGGREGATION
# ============================================================

def aggregate_vad_output(output):

    output = validate_module_output(output)

    if "coverage" not in output:
        raise ValueError(
            "VAD output must contain lexical coverage."
        )

    coverage = validate_unit_interval(
        output["coverage"],
        "coverage",
    )

    scoring_term_count = output.get(
        "scoring_term_count"
    )

    if (
        isinstance(scoring_term_count, bool)
        or not isinstance(scoring_term_count, int)
        or scoring_term_count < 0
    ):
        raise ValueError(
            "VAD scoring_term_count must be a "
            "non-negative integer."
        )

    return {
        "module_name": "vad",
        "method": "lexicon_based_analysis",
        "status": "evaluated",
        "lexical_coverage": round(
            coverage,
            ROUNDING_PRECISION,
        ),
        "scoring_term_count": scoring_term_count,
        "confidence": None,
        "uncertainty": None,
        "probability_available": False,
        "interpretation": (
            "Lexical coverage describes how much of the "
            "input was matched by the VAD lexicon. It is "
            "not a confidence probability."
        ),
    }


# ============================================================
# 6. SINGLE MODULE ROUTER
# ============================================================

def aggregate_single_module(module_name, output):

    validate_module_name(module_name)

    if module_name in CLASSIFICATION_MODULES:
        return aggregate_classification_output(
            module_name,
            output,
        )

    if module_name in RULE_BASED_MODULES:
        return aggregate_rule_based_output(
            module_name,
            output,
        )

    if module_name in LEXICON_MODULES:
        return aggregate_vad_output(output)

    raise ValueError(
        f"No aggregation handler for {module_name}."
    )


# ============================================================
# 7. MISSING MODULE HANDLING
# ============================================================

def build_missing_module_entry(module_name):

    result = {
        "module_name": module_name,
        "status": "not_evaluated",
        "confidence": None,
        "uncertainty": None,
        "probability_available": False,
    }

    if module_name == "risk":

        result["requires_human_review"] = None

        result["safety_interpretation"] = (
            "Not evaluated. This does not establish that "
            "the student is safe."
        )

    return result


# ============================================================
# 8. CROSS-MODULE AGGREGATION
# ============================================================

def aggregate_all_modules(module_outputs):

    if not isinstance(module_outputs, dict):
        raise ValueError(
            "module_outputs must be a dictionary."
        )

    if not module_outputs:
        raise ValueError(
            "At least one module output is required."
        )

    unknown_modules = (
        set(module_outputs) - set(SUPPORTED_MODULES)
    )

    if unknown_modules:
        raise ValueError(
            f"Unknown modules: {sorted(unknown_modules)}"
        )

    module_results = {}

    for module_name, output in module_outputs.items():

        module_results[module_name] = (
            aggregate_single_module(
                module_name,
                output,
            )
        )

    # Add all modules that were not executed.
    missing_modules = [
        name
        for name in SUPPORTED_MODULES
        if name not in module_results
    ]

    for module_name in missing_modules:

        module_results[module_name] = (
            build_missing_module_entry(
                module_name
            )
        )

    evaluated_count = sum(
        result["status"] == "evaluated"
        for result in module_results.values()
    )

    not_evaluated_count = (
        len(module_results) - evaluated_count
    )

    # --------------------------------------------------------
    # Human review propagation
    # --------------------------------------------------------

    risk_result = module_results["risk"]

    if risk_result["status"] == "not_evaluated":

        # Unknown is not equivalent to False.
        human_review_required = None

    else:

        human_review_required = (
            risk_result["requires_human_review"]
        )

    # --------------------------------------------------------
    # Warnings
    # --------------------------------------------------------

    warnings = [

        "Raw model probabilities are not calibrated.",

        "Confidence values are not clinical certainty.",

        "Rule-based outputs do not receive invented "
        "probability scores.",

        "VAD lexical coverage is not a confidence probability.",

        "Missing module outputs are not evidence of safety.",

    ]

    if missing_modules:
        warnings.append(
            "Some analytical modules were not evaluated."
        )

    if risk_result["status"] == "not_evaluated":

        warnings.append(
            "Risk detection was not evaluated. "
            "No safety conclusion can be drawn."
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    return {

        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "module_count": len(SUPPORTED_MODULES),

        "evaluated_module_count": evaluated_count,

        "not_evaluated_module_count": not_evaluated_count,

        "overall_confidence": None,

        "overall_uncertainty": None,

        "overall_confidence_calculated": False,

        "human_review_required": human_review_required,

        "risk_evaluation_status": risk_result["status"],

        "modules": module_results,

        "warnings": warnings,

        "calibration_status": CALIBRATION_STATUS,

        "clinical_diagnosis": False,

        "actual_risk_confirmed": False,

    }


# ============================================================
# 9. SMOKE VALIDATION
# ============================================================

def validate_aggregator():

    # --------------------------------------------------------
    # A. Classification example
    # --------------------------------------------------------

    sentiment_example = {

        "output_type": "multi_class",

        "predicted_class": "negative",

        "confidence": 0.80,

        "normalized_entropy": 0.35,

        "prediction_margin": 0.60,

        "uncertainty_band": "moderate",

        "calibration_status": "not_calibrated",

    }

    sentiment_result = aggregate_single_module(
        "sentiment",
        sentiment_example,
    )

    assert sentiment_result["status"] == "evaluated"

    assert sentiment_result["is_calibrated"] is False

    # --------------------------------------------------------
    # B. Risk example with dictionary indicators
    # --------------------------------------------------------

    risk_example = {

        "category": "uncertain_or_ambiguous",

        "detected_indicators": {
            "suicidal_language": True,
        },

        "requires_human_review": True,

    }

    risk_result = aggregate_single_module(
        "risk",
        risk_example,
    )

    assert risk_result["confidence"] is None

    assert risk_result["requires_human_review"] is True

    assert isinstance(
        risk_result["detected_indicators"],
        dict,
    )

    # --------------------------------------------------------
    # C. VAD example
    # --------------------------------------------------------

    vad_example = {
        "coverage": 0.75,
        "scoring_term_count": 6,
    }

    vad_result = aggregate_single_module(
        "vad",
        vad_example,
    )

    assert vad_result["lexical_coverage"] == 0.75

    assert vad_result["confidence"] is None

    # --------------------------------------------------------
    # D. Cross-module example
    # --------------------------------------------------------

    combined = aggregate_all_modules({

        "sentiment": sentiment_example,

        "risk": risk_example,

        "vad": vad_example,

    })

    assert combined["module_count"] == 8

    assert combined["evaluated_module_count"] == 3

    assert combined["not_evaluated_module_count"] == 5

    assert combined["overall_confidence"] is None

    assert combined["human_review_required"] is True

    assert combined["clinical_diagnosis"] is False

    assert combined["actual_risk_confirmed"] is False

    # --------------------------------------------------------
    # E. Missing risk handling
    # --------------------------------------------------------

    missing_risk = aggregate_all_modules({
        "sentiment": sentiment_example,
    })

    assert (
        missing_risk["risk_evaluation_status"]
        == "not_evaluated"
    )

    assert missing_risk["human_review_required"] is None

    assert (
        missing_risk["modules"]["risk"][
            "requires_human_review"
        ] is None
    )

    assert any(
        "No safety conclusion" in warning
        for warning in missing_risk["warnings"]
    )

    # --------------------------------------------------------
    # F. Evaluated risk with no review required
    # --------------------------------------------------------

    no_review_risk = {
        "category": "no_explicit_risk_signal",
        "detected_indicators": {},
        "requires_human_review": False,
    }

    no_review_report = aggregate_all_modules({
        "risk": no_review_risk,
        "sentiment": sentiment_example,
    })

    assert no_review_report["human_review_required"] is False

    assert (
        no_review_report["risk_evaluation_status"]
        == "evaluated"
    )

    # --------------------------------------------------------
    # G. Rumination example
    # --------------------------------------------------------

    rumination_example = {
        "detected_indicators": ["repetitive_thought"],
        "ambiguity_flags": [],
    }

    rumination_result = aggregate_single_module(
        "rumination",
        rumination_example,
    )

    assert rumination_result["confidence"] is None

    assert rumination_result["status"] == "evaluated"

    return True


# ============================================================
# 10. MAIN
# ============================================================

if __name__ == "__main__":

    validate_config()

    validate_aggregator()

    print("=" * 70)

    print("CONFIDENCE AGGREGATOR: SMOKE TEST PASSED")

    print("=" * 70)

    example_outputs = {

        "sentiment": {

            "output_type": "multi_class",

            "predicted_class": "negative",

            "confidence": 0.80,

            "normalized_entropy": 0.35,

            "prediction_margin": 0.60,

            "uncertainty_band": "moderate",

            "calibration_status": "not_calibrated",

        },

        "risk": {

            "category": "uncertain_or_ambiguous",

            "detected_indicators": {
                "suicidal_language": True,
            },

            "requires_human_review": True,

        },

        "vad": {

            "coverage": 0.75,

            "scoring_term_count": 6,

        },

    }

    report = aggregate_all_modules(
        example_outputs
    )

    print(
        "\nEVALUATED MODULES:",
        report["evaluated_module_count"],
    )

    print(
        "NOT EVALUATED MODULES:",
        report["not_evaluated_module_count"],
    )

    print(
        "OVERALL CONFIDENCE:",
        report["overall_confidence"],
    )

    print(
        "HUMAN REVIEW REQUIRED:",
        report["human_review_required"],
    )

    print(
        "RISK EVALUATION STATUS:",
        report["risk_evaluation_status"],
    )

    print("\nDISCLAIMER:")

    for warning in report["warnings"]:
        print("-", warning)
