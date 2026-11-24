
"""
Confidence Metrics Regression Tests
====================================

Tests:
1. Softmax and sigmoid
2. Entropy calculations
3. Normalized entropy
4. Prediction margin
5. Multi-class analysis
6. Multi-label analysis
7. Numerical stability
8. Boundary conditions
9. Invalid input handling
10. Calibration and safety declarations
"""

import unittest

import numpy as np

from scripts.model_development.confidence.confidence_metrics import (
    softmax,
    sigmoid,
    calculate_entropy,
    calculate_normalized_entropy,
    calculate_prediction_margin,
    calculate_binary_entropy,
    analyze_multiclass_probabilities,
    analyze_multiclass_logits,
    analyze_multilabel_probabilities,
    analyze_multilabel_logits,
    interpret_confidence,
    interpret_uncertainty,
    get_confidence_disclaimer,
)


class TestConfidenceMetrics(unittest.TestCase):

    # ========================================================
    # 1. SOFTMAX TESTS
    # ========================================================

    def test_softmax_probabilities_sum_to_one(self):

        result = softmax([2.0, 1.0, 0.1])

        self.assertAlmostEqual(
            float(np.sum(result)),
            1.0,
            places=10
        )

    def test_softmax_values_within_probability_range(self):

        result = softmax([-2.0, 0.0, 2.0])

        self.assertTrue(
            np.all(result >= 0.0)
        )

        self.assertTrue(
            np.all(result <= 1.0)
        )

    def test_softmax_equal_logits(self):

        result = softmax([1.0, 1.0, 1.0])

        np.testing.assert_allclose(
            result,
            [1 / 3, 1 / 3, 1 / 3]
        )

    def test_softmax_numerical_stability(self):

        result = softmax([
            10000.0,
            10001.0,
            10002.0
        ])

        self.assertTrue(
            np.all(np.isfinite(result))
        )

        self.assertAlmostEqual(
            float(np.sum(result)),
            1.0,
            places=10
        )

    def test_softmax_rejects_single_class(self):

        with self.assertRaises(ValueError):

            softmax([2.0])

    def test_softmax_rejects_nonfinite_values(self):

        with self.assertRaises(ValueError):

            softmax([1.0, np.nan])

    # ========================================================
    # 2. SIGMOID TESTS
    # ========================================================

    def test_sigmoid_zero_equals_half(self):

        result = sigmoid([0.0])

        self.assertAlmostEqual(
            float(result[0]),
            0.5,
            places=10
        )

    def test_sigmoid_values_within_range(self):

        result = sigmoid([-5.0, 0.0, 5.0])

        self.assertTrue(
            np.all(result >= 0.0)
        )

        self.assertTrue(
            np.all(result <= 1.0)
        )

    def test_sigmoid_numerical_stability(self):

        result = sigmoid([
            -1000.0,
            0.0,
            1000.0
        ])

        self.assertTrue(
            np.all(np.isfinite(result))
        )

        self.assertAlmostEqual(
            float(result[0]),
            0.0,
            places=10
        )

        self.assertAlmostEqual(
            float(result[1]),
            0.5,
            places=10
        )

        self.assertAlmostEqual(
            float(result[2]),
            1.0,
            places=10
        )

    # ========================================================
    # 3. MULTI-CLASS ENTROPY TESTS
    # ========================================================

    def test_entropy_uniform_distribution(self):

        result = calculate_entropy([
            0.5,
            0.5
        ])

        self.assertAlmostEqual(
            result,
            1.0,
            places=10
        )

    def test_entropy_certain_distribution(self):

        result = calculate_entropy([
            1.0,
            0.0
        ])

        self.assertAlmostEqual(
            result,
            0.0,
            places=10
        )

    def test_entropy_three_class_uniform_distribution(self):

        result = calculate_entropy([
            1 / 3,
            1 / 3,
            1 / 3
        ])

        self.assertAlmostEqual(
            result,
            np.log2(3),
            places=10
        )

    def test_normalized_entropy_uniform_distribution(self):

        result = calculate_normalized_entropy([
            0.25,
            0.25,
            0.25,
            0.25
        ])

        self.assertAlmostEqual(
            result,
            1.0,
            places=10
        )

    def test_normalized_entropy_certain_distribution(self):

        result = calculate_normalized_entropy([
            1.0,
            0.0,
            0.0
        ])

        self.assertAlmostEqual(
            result,
            0.0,
            places=10
        )

    # ========================================================
    # 4. PREDICTION MARGIN TESTS
    # ========================================================

    def test_prediction_margin(self):

        result = calculate_prediction_margin([
            0.7,
            0.2,
            0.1
        ])

        self.assertAlmostEqual(
            result,
            0.5,
            places=10
        )

    def test_prediction_margin_equal_top_classes(self):

        result = calculate_prediction_margin([
            0.5,
            0.5,
            0.0
        ])

        self.assertAlmostEqual(
            result,
            0.0,
            places=10
        )

    # ========================================================
    # 5. BINARY ENTROPY TESTS
    # ========================================================

    def test_binary_entropy_at_half(self):

        result = calculate_binary_entropy(0.5)

        self.assertAlmostEqual(
            result,
            1.0,
            places=10
        )

    def test_binary_entropy_at_zero(self):

        result = calculate_binary_entropy(0.0)

        self.assertEqual(
            result,
            0.0
        )

    def test_binary_entropy_at_one(self):

        result = calculate_binary_entropy(1.0)

        self.assertEqual(
            result,
            0.0
        )

    def test_binary_entropy_symmetry(self):

        first = calculate_binary_entropy(0.2)

        second = calculate_binary_entropy(0.8)

        self.assertAlmostEqual(
            first,
            second,
            places=10
        )

    # ========================================================
    # 6. MULTI-CLASS INTEGRATION TESTS
    # ========================================================

    def test_multiclass_prediction(self):

        result = analyze_multiclass_probabilities(

            probabilities=[
                0.7,
                0.2,
                0.1
            ],

            class_names=[
                "negative",
                "neutral",
                "positive"
            ]
        )

        self.assertEqual(
            result["predicted_class"],
            "negative"
        )

    def test_multiclass_top_predictions(self):

        result = analyze_multiclass_probabilities(

            probabilities=[
                0.7,
                0.2,
                0.1
            ],

            class_names=[
                "negative",
                "neutral",
                "positive"
            ],

            top_k=2
        )

        self.assertEqual(
            len(result["top_predictions"]),
            2
        )

    def test_multiclass_entropy_and_margin_present(self):

        result = analyze_multiclass_probabilities([
            0.7,
            0.2,
            0.1
        ])

        self.assertIn(
            "entropy",
            result
        )

        self.assertIn(
            "prediction_margin",
            result
        )

        self.assertIn(
            "normalized_entropy",
            result
        )

    def test_multiclass_logits_integration(self):

        result = analyze_multiclass_logits([
            2.0,
            1.0,
            0.1
        ])

        self.assertEqual(
            result["input_type"],
            "logits"
        )

        self.assertEqual(
            result["output_type"],
            "multi_class"
        )

    # ========================================================
    # 7. MULTI-LABEL INTEGRATION TESTS
    # ========================================================

    def test_multilabel_selected_labels(self):

        result = analyze_multilabel_probabilities(

            probabilities=[
                0.9,
                0.2,
                0.6
            ],

            label_names=[
                "joy",
                "anger",
                "sadness"
            ]
        )

        self.assertEqual(
            result["selected_labels"],
            ["joy", "sadness"]
        )

    def test_multilabel_label_count(self):

        result = analyze_multilabel_probabilities([
            0.9,
            0.2,
            0.6
        ])

        self.assertEqual(
            result["label_count"],
            3
        )

    def test_multilabel_entropy_present(self):

        result = analyze_multilabel_probabilities([
            0.9,
            0.2,
            0.6
        ])

        self.assertIn(
            "mean_binary_entropy",
            result
        )

        self.assertIn(
            "normalized_uncertainty",
            result
        )

    def test_multilabel_logits_integration(self):

        result = analyze_multilabel_logits([
            2.0,
            -1.0,
            0.0
        ])

        self.assertEqual(
            result["input_type"],
            "logits"
        )

        self.assertEqual(
            result["output_type"],
            "multi_label"
        )

    # ========================================================
    # 8. CONFIDENCE INTERPRETATION TESTS
    # ========================================================

    def test_confidence_bands(self):

        self.assertEqual(
            interpret_confidence(0.3),
            "low"
        )

        self.assertEqual(
            interpret_confidence(0.6),
            "moderate"
        )

        self.assertEqual(
            interpret_confidence(0.9),
            "high"
        )

    def test_uncertainty_bands(self):

        self.assertEqual(
            interpret_uncertainty(0.1),
            "low"
        )

        self.assertEqual(
            interpret_uncertainty(0.4),
            "moderate"
        )

        self.assertEqual(
            interpret_uncertainty(0.8),
            "high"
        )

    # ========================================================
    # 9. INVALID INPUT TESTS
    # ========================================================

    def test_invalid_probability_sum(self):

        with self.assertRaises(ValueError):

            analyze_multiclass_probabilities([
                0.2,
                0.2
            ])

    def test_negative_probability_rejected(self):

        with self.assertRaises(ValueError):

            analyze_multiclass_probabilities([
                -0.1,
                1.1
            ])

    def test_probability_above_one_rejected(self):

        with self.assertRaises(ValueError):

            analyze_multilabel_probabilities([
                1.2,
                0.2
            ])

    def test_empty_input_rejected(self):

        with self.assertRaises(ValueError):

            softmax([])

    def test_two_dimensional_input_rejected(self):

        with self.assertRaises(ValueError):

            analyze_multiclass_probabilities([
                [0.5, 0.5],
                [0.2, 0.8]
            ])

    def test_invalid_top_k_rejected(self):

        with self.assertRaises(ValueError):

            analyze_multiclass_probabilities(
                [0.5, 0.5],
                top_k=0
            )

    def test_duplicate_class_names_rejected(self):

        with self.assertRaises(ValueError):

            analyze_multiclass_probabilities(

                [0.5, 0.5],

                class_names=[
                    "same",
                    "same"
                ]
            )

    # ========================================================
    # 10. CALIBRATION AND SAFETY TESTS
    # ========================================================

    def test_raw_probabilities_not_claimed_calibrated(self):

        result = analyze_multiclass_probabilities([
            0.8,
            0.2
        ])

        self.assertFalse(
            result["is_calibrated"]
        )

        self.assertEqual(
            result["calibration_status"],
            "not_calibrated"
        )

    def test_multilabel_not_claimed_calibrated(self):

        result = analyze_multilabel_probabilities([
            0.8,
            0.2
        ])

        self.assertFalse(
            result["is_calibrated"]
        )

    def test_disclaimer_present(self):

        disclaimer = get_confidence_disclaimer()

        self.assertIn(
            "not necessarily calibrated",
            disclaimer
        )

        self.assertIn(
            "student safety",
            disclaimer
        )


# ============================================================
# 11. TEST EXECUTION
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print("CONFIDENCE METRICS REGRESSION TESTS")

    print("=" * 70)

    unittest.main(verbosity=2)
