
"""
Cross-Module Confidence Aggregator Regression Tests
====================================================

Tests all eight analytical module formats, missing outputs,
invalid inputs, safety handling and output consistency.
"""

import unittest

from scripts.model_development.confidence.confidence_aggregator import (
    aggregate_single_module,
    aggregate_all_modules,
    build_missing_module_entry,
)


# ============================================================
# 1. REUSABLE TEST OUTPUTS
# ============================================================

def multiclass_output():

    return {

        "output_type": "multi_class",

        "predicted_class": "negative",

        "confidence": 0.80,

        "normalized_entropy": 0.35,

        "prediction_margin": 0.60,

        "uncertainty_band": "moderate",

        "calibration_status": "not_calibrated"
    }


def multilabel_output():

    return {

        "output_type": "multi_label",

        "mean_label_confidence": 0.78,

        "normalized_uncertainty": 0.42,

        "uncertainty_band": "moderate",

        "label_count": 28,

        "selected_labels": ["joy", "sadness"],

        "label_results": [

            {"label_name": "joy", "probability": 0.8},

            {"label_name": "sadness", "probability": 0.7}
        ],

        "calibration_status": "not_calibrated"
    }


def rumination_output():

    return {

        "has_rumination_language": True,

        "detected_indicators": [
            "repetitive_thoughts",
            "past_focused"
        ],

        "ambiguity_flags": []
    }


def risk_output():

    return {

        "category": "uncertain_or_ambiguous",

        "detected_indicators": [
            "suicidal_language"
        ],

        "requires_human_review": True
    }


def vad_output():

    return {

        "coverage": 0.75,

        "scoring_term_count": 6,

        "zero_vector_term_count": 3,

        "valence": 0.2088,

        "arousal": 0.2313,

        "dominance": 0.117
    }


# ============================================================
# 2. TEST CLASS
# ============================================================

class TestConfidenceAggregator(unittest.TestCase):

    # ========================================================
    # A. ALL CLASSIFICATION MODULES
    # ========================================================

    def test_emotion_module(self):

        result = aggregate_single_module(
            "emotion",
            multilabel_output()
        )

        self.assertEqual(
            result["method"],
            "multi_label_classification"
        )

        self.assertEqual(
            result["status"],
            "evaluated"
        )

        self.assertFalse(
            result["is_calibrated"]
        )

    def test_sentiment_module(self):

        result = aggregate_single_module(
            "sentiment",
            multiclass_output()
        )

        self.assertEqual(
            result["predicted_class"],
            "negative"
        )

    def test_sentiment_bilstm_module(self):

        result = aggregate_single_module(
            "sentiment_bilstm",
            multiclass_output()
        )

        self.assertEqual(
            result["module_name"],
            "sentiment_bilstm"
        )

    def test_stress_module(self):

        result = aggregate_single_module(
            "stress",
            multiclass_output()
        )

        self.assertEqual(
            result["method"],
            "multi_class_classification"
        )

    def test_loneliness_module(self):

        result = aggregate_single_module(
            "loneliness",
            multiclass_output()
        )

        self.assertEqual(
            result["status"],
            "evaluated"
        )

    # ========================================================
    # B. RULE-BASED MODULES
    # ========================================================

    def test_rumination_module(self):

        result = aggregate_single_module(
            "rumination",
            rumination_output()
        )

        self.assertIsNone(
            result["confidence"]
        )

        self.assertFalse(
            result["probability_available"]
        )

        self.assertEqual(
            len(result["detected_indicators"]),
            2
        )

    def test_risk_module(self):

        result = aggregate_single_module(
            "risk",
            risk_output()
        )

        self.assertTrue(
            result["requires_human_review"]
        )

        self.assertIsNone(
            result["confidence"]
        )

        self.assertFalse(
            result["clinical_certainty"]
        )

    def test_risk_without_human_review(self):

        output = risk_output()

        output["requires_human_review"] = False

        result = aggregate_single_module(
            "risk",
            output
        )

        self.assertFalse(
            result["requires_human_review"]
        )

        self.assertIsNone(
            result["confidence"]
        )

    # ========================================================
    # C. LEXICON MODULE
    # ========================================================

    def test_vad_module(self):

        result = aggregate_single_module(
            "vad",
            vad_output()
        )

        self.assertEqual(
            result["lexical_coverage"],
            0.75
        )

        self.assertEqual(
            result["scoring_term_count"],
            6
        )

        self.assertIsNone(
            result["confidence"]
        )

        self.assertFalse(
            result["probability_available"]
        )

    # ========================================================
    # D. ALL EIGHT MODULES TOGETHER
    # ========================================================

    def test_all_eight_modules(self):

        outputs = {

            "emotion": multilabel_output(),

            "sentiment": multiclass_output(),

            "sentiment_bilstm": multiclass_output(),

            "stress": multiclass_output(),

            "loneliness": multiclass_output(),

            "rumination": rumination_output(),

            "risk": risk_output(),

            "vad": vad_output()
        }

        result = aggregate_all_modules(
            outputs
        )

        self.assertEqual(
            result["module_count"],
            8
        )

        self.assertEqual(
            result["evaluated_module_count"],
            8
        )

        self.assertEqual(
            result["not_evaluated_module_count"],
            0
        )

        self.assertTrue(
            result["human_review_required"]
        )

        self.assertIsNone(
            result["overall_confidence"]
        )

        self.assertFalse(
            result["overall_confidence_calculated"]
        )

    # ========================================================
    # E. PARTIAL MODULE OUTPUTS
    # ========================================================

    def test_missing_modules_are_identified(self):

        result = aggregate_all_modules({

            "sentiment": multiclass_output()
        })

        self.assertEqual(
            result["evaluated_module_count"],
            1
        )

        self.assertEqual(
            result["not_evaluated_module_count"],
            7
        )

        self.assertEqual(
            result["modules"]["risk"]["status"],
            "not_evaluated"
        )

    def test_missing_risk_does_not_establish_safety(self):

        result = aggregate_all_modules({

            "sentiment": multiclass_output()
        })

        risk_result = result["modules"]["risk"]

        self.assertIn(
            "does not establish",
            risk_result["safety_interpretation"]
        )

        self.assertEqual(
            result["risk_evaluation_status"],
            "not_evaluated"
        )

    def test_missing_vad_is_not_zero_confidence(self):

        result = build_missing_module_entry(
            "vad"
        )

        self.assertIsNone(
            result["confidence"]
        )

        self.assertEqual(
            result["status"],
            "not_evaluated"
        )

    # ========================================================
    # F. HUMAN REVIEW PROPAGATION
    # ========================================================

    def test_urgent_or_ambiguous_review_flag_propagates(self):

        result = aggregate_all_modules({

            "risk": risk_output()
        })

        self.assertTrue(
            result["human_review_required"]
        )

    def test_no_review_flag_is_not_a_safety_claim(self):

        output = risk_output()

        output["requires_human_review"] = False

        result = aggregate_all_modules({

            "risk": output
        })

        self.assertFalse(
            result["human_review_required"]
        )

        self.assertFalse(
            result["actual_risk_confirmed"]
        )

        self.assertFalse(
            result["clinical_diagnosis"]
        )

    # ========================================================
    # G. INVALID INPUT TESTS
    # ========================================================

    def test_unknown_module_rejected(self):

        with self.assertRaises(ValueError):

            aggregate_single_module(
                "unknown_module",
                {}
            )

    def test_non_dictionary_module_output_rejected(self):

        with self.assertRaises(ValueError):

            aggregate_single_module(
                "sentiment",
                [0.8, 0.2]
            )

    def test_empty_aggregation_rejected(self):

        with self.assertRaises(ValueError):

            aggregate_all_modules({})

    def test_non_dictionary_aggregation_rejected(self):

        with self.assertRaises(ValueError):

            aggregate_all_modules([])

    def test_missing_classification_fields_rejected(self):

        with self.assertRaises(ValueError):

            aggregate_single_module(
                "sentiment",
                {
                    "output_type": "multi_class"
                }
            )

    def test_invalid_confidence_rejected(self):

        output = multiclass_output()

        output["confidence"] = 1.5

        with self.assertRaises(ValueError):

            aggregate_single_module(
                "sentiment",
                output
            )

    def test_invalid_uncertainty_rejected(self):

        output = multiclass_output()

        output["normalized_entropy"] = -0.2

        with self.assertRaises(ValueError):

            aggregate_single_module(
                "stress",
                output
            )

    def test_missing_risk_review_flag_rejected(self):

        output = risk_output()

        del output["requires_human_review"]

        with self.assertRaises(ValueError):

            aggregate_single_module(
                "risk",
                output
            )

    def test_invalid_vad_coverage_rejected(self):

        output = vad_output()

        output["coverage"] = 1.5

        with self.assertRaises(ValueError):

            aggregate_single_module(
                "vad",
                output
            )

    def test_missing_vad_scoring_count_rejected(self):

        output = vad_output()

        del output["scoring_term_count"]

        with self.assertRaises(ValueError):

            aggregate_single_module(
                "vad",
                output
            )

    # ========================================================
    # H. OUTPUT CONSISTENCY AND SAFETY
    # ========================================================

    def test_aggregation_timestamp_present(self):

        result = aggregate_all_modules({

            "sentiment": multiclass_output()
        })

        self.assertIn(
            "timestamp_utc",
            result
        )

    def test_no_overall_confidence_is_generated(self):

        result = aggregate_all_modules({

            "sentiment": multiclass_output(),

            "emotion": multilabel_output()
        })

        self.assertIsNone(
            result["overall_confidence"]
        )

        self.assertIsNone(
            result["overall_uncertainty"]
        )

    def test_no_clinical_diagnosis_claim(self):

        result = aggregate_all_modules({

            "risk": risk_output()
        })

        self.assertFalse(
            result["clinical_diagnosis"]
        )

    def test_no_actual_risk_confirmation(self):

        result = aggregate_all_modules({

            "risk": risk_output()
        })

        self.assertFalse(
            result["actual_risk_confirmed"]
        )

    def test_missing_module_warning_present(self):

        result = aggregate_all_modules({

            "sentiment": multiclass_output()
        })

        self.assertTrue(
            any(
                "not evaluated" in warning.lower()
                for warning in result["warnings"]
            )
        )


# ============================================================
# 3. TEST EXECUTION
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print("CROSS-MODULE CONFIDENCE AGGREGATOR TESTS")

    print("=" * 70)

    unittest.main(verbosity=2)
