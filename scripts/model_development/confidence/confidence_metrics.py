
"""
Confidence Metrics Engine
=========================

Mathematical confidence and uncertainty calculations for:

1. Multi-class classification
2. Multi-label classification

Supports raw logits and probability inputs.

Important:
Raw model probabilities are not necessarily calibrated.
These metrics describe model output behavior, not clinical
certainty or actual psychological states.
"""

import numpy as np

from .confidence_config import (
    CALIBRATION_STATUS,
    DEFAULT_TOP_K,
    ENTROPY_EPSILON,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
    MULTI_LABEL_THRESHOLD,
    NORMALIZE_ENTROPY,
    PROBABILITY_TOLERANCE,
    ROUNDING_PRECISION,
    CONFIDENCE_BANDS,
    UNCERTAINTY_BANDS,
    validate_config,
)


# ============================================================
# 1. INPUT VALIDATION
# ============================================================

def validate_numeric_array(values, name="values"):

    try:

        array = np.asarray(values, dtype=np.float64)

    except (TypeError, ValueError) as error:

        raise ValueError(
            f"{name} must contain numeric values."
        ) from error

    if array.size == 0:

        raise ValueError(
            f"{name} cannot be empty."
        )

    if not np.all(np.isfinite(array)):

        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    return array


def validate_class_probabilities(probabilities):

    probabilities = validate_numeric_array(
        probabilities,
        "probabilities"
    )

    if probabilities.ndim != 1:

        raise ValueError(
            "Multi-class probabilities must be one-dimensional."
        )

    if len(probabilities) < 2:

        raise ValueError(
            "At least two classes are required."
        )

    if np.any(probabilities < MIN_PROBABILITY):

        raise ValueError(
            "Probabilities cannot be negative."
        )

    if np.any(probabilities > MAX_PROBABILITY):

        raise ValueError(
            "Probabilities cannot exceed one."
        )

    if not np.isclose(
        probabilities.sum(),
        1.0,
        atol=PROBABILITY_TOLERANCE
    ):

        raise ValueError(
            "Multi-class probabilities must sum to one."
        )

    return probabilities


def validate_multilabel_probabilities(probabilities):

    probabilities = validate_numeric_array(
        probabilities,
        "probabilities"
    )

    if probabilities.ndim != 1:

        raise ValueError(
            "Multi-label probabilities must be one-dimensional."
        )

    if np.any(probabilities < MIN_PROBABILITY):

        raise ValueError(
            "Probabilities cannot be negative."
        )

    if np.any(probabilities > MAX_PROBABILITY):

        raise ValueError(
            "Probabilities cannot exceed one."
        )

    return probabilities


# ============================================================
# 2. SOFTMAX
# ============================================================

def softmax(logits):

    logits = validate_numeric_array(
        logits,
        "logits"
    )

    if logits.ndim != 1:

        raise ValueError(
            "Softmax expects one-dimensional logits."
        )

    if len(logits) < 2:

        raise ValueError(
            "Softmax requires at least two class logits."
        )

    # Subtract maximum for numerical stability.
    shifted_logits = logits - np.max(logits)

    exponentials = np.exp(shifted_logits)

    probabilities = (
        exponentials / np.sum(exponentials)
    )

    return probabilities


# ============================================================
# 3. SIGMOID
# ============================================================

def sigmoid(logits):

    logits = validate_numeric_array(
        logits,
        "logits"
    )

    # Numerically stable sigmoid.
    result = np.empty_like(logits)

    positive_mask = logits >= 0

    result[positive_mask] = (
        1.0 /
        (1.0 + np.exp(-logits[positive_mask]))
    )

    negative_exp = np.exp(
        logits[~positive_mask]
    )

    result[~positive_mask] = (
        negative_exp /
        (1.0 + negative_exp)
    )

    return result


# ============================================================
# 4. SHANNON ENTROPY
# ============================================================

def calculate_entropy(probabilities):

    probabilities = validate_class_probabilities(
        probabilities
    )

    # Ignore zero-probability terms in logarithm.
    nonzero_probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        nonzero_probabilities *
        np.log2(nonzero_probabilities)
    )

    return float(entropy)


# ============================================================
# 5. NORMALIZED ENTROPY
# ============================================================

def calculate_normalized_entropy(
    probabilities,
    entropy=None
):

    probabilities = validate_class_probabilities(
        probabilities
    )

    if entropy is None:

        entropy = calculate_entropy(
            probabilities
        )

    maximum_entropy = np.log2(
        len(probabilities)
    )

    if maximum_entropy == 0:

        return 0.0

    normalized_entropy = (
        entropy / maximum_entropy
    )

    return float(
        np.clip(
            normalized_entropy,
            0.0,
            1.0
        )
    )


# ============================================================
# 6. PREDICTION MARGIN
# ============================================================

def calculate_prediction_margin(probabilities):

    probabilities = validate_class_probabilities(
        probabilities
    )

    sorted_probabilities = np.sort(
        probabilities
    )[::-1]

    margin = (
        sorted_probabilities[0] -
        sorted_probabilities[1]
    )

    return float(margin)


# ============================================================
# 7. CONFIDENCE BAND
# ============================================================

def interpret_confidence(confidence):

    if not 0.0 <= confidence <= 1.0:

        raise ValueError(
            "Confidence must be between zero and one."
        )

    if confidence <= CONFIDENCE_BANDS["low"]["maximum"]:

        return "low"

    if confidence <= CONFIDENCE_BANDS["moderate"]["maximum"]:

        return "moderate"

    return "high"


# ============================================================
# 8. UNCERTAINTY BAND
# ============================================================

def interpret_uncertainty(uncertainty):

    if not 0.0 <= uncertainty <= 1.0:

        raise ValueError(
            "Uncertainty must be between zero and one."
        )

    if uncertainty <= UNCERTAINTY_BANDS["low"]["maximum"]:

        return "low"

    if uncertainty <= UNCERTAINTY_BANDS["moderate"]["maximum"]:

        return "moderate"

    return "high"


# ============================================================
# 9. MULTI-CLASS PROBABILITY ANALYSIS
# ============================================================

def analyze_multiclass_probabilities(
    probabilities,
    class_names=None,
    top_k=DEFAULT_TOP_K
):

    probabilities = validate_class_probabilities(
        probabilities
    )

    class_count = len(probabilities)

    if not isinstance(top_k, int) or top_k < 1:

        raise ValueError(
            "top_k must be a positive integer."
        )

    if class_names is None:

        class_names = [
            f"class_{index}"
            for index in range(class_count)
        ]

    if len(class_names) != class_count:

        raise ValueError(
            "Number of class names must match probability count."
        )

    if len(set(class_names)) != len(class_names):

        raise ValueError(
            "Class names must be unique."
        )

    top_k = min(top_k, class_count)

    predicted_index = int(
        np.argmax(probabilities)
    )

    maximum_probability = float(
        probabilities[predicted_index]
    )

    entropy = calculate_entropy(
        probabilities
    )

    normalized_entropy = calculate_normalized_entropy(
        probabilities,
        entropy
    )

    margin = calculate_prediction_margin(
        probabilities
    )

    top_indices = np.argsort(
        probabilities
    )[::-1][:top_k]

    top_predictions = [

        {
            "class_name": class_names[index],

            "class_index": int(index),

            "probability": round(
                float(probabilities[index]),
                ROUNDING_PRECISION
            )
        }

        for index in top_indices
    ]

    confidence_band = interpret_confidence(
        maximum_probability
    )

    uncertainty_band = interpret_uncertainty(
        normalized_entropy
    )

    return {

        "output_type": "multi_class",

        "predicted_class": class_names[
            predicted_index
        ],

        "predicted_class_index": predicted_index,

        "confidence": round(
            maximum_probability,
            ROUNDING_PRECISION
        ),

        "confidence_band": confidence_band,

        "prediction_margin": round(
            margin,
            ROUNDING_PRECISION
        ),

        "entropy": round(
            entropy,
            ROUNDING_PRECISION
        ),

        "normalized_entropy": round(
            normalized_entropy,
            ROUNDING_PRECISION
        ),

        "uncertainty_band": uncertainty_band,

        "top_predictions": top_predictions,

        "calibration_status": CALIBRATION_STATUS,

        "is_calibrated": False
    }


# ============================================================
# 10. MULTI-CLASS LOGIT ANALYSIS
# ============================================================

def analyze_multiclass_logits(
    logits,
    class_names=None,
    top_k=DEFAULT_TOP_K
):

    probabilities = softmax(logits)

    result = analyze_multiclass_probabilities(
        probabilities=probabilities,
        class_names=class_names,
        top_k=top_k
    )

    result["input_type"] = "logits"

    return result


# ============================================================
# 11. BINARY ENTROPY
# ============================================================

def calculate_binary_entropy(probability):

    probability = float(probability)

    if not np.isfinite(probability):

        raise ValueError(
            "Probability must be finite."
        )

    if not 0.0 <= probability <= 1.0:

        raise ValueError(
            "Probability must be between zero and one."
        )

    # Boundary values have zero entropy.
    if probability == 0.0 or probability == 1.0:

        return 0.0

    complement = 1.0 - probability

    entropy = -(

        probability * np.log2(
            probability
        )

        +

        complement * np.log2(
            complement
        )
    )

    return float(entropy)


# ============================================================
# 12. MULTI-LABEL PROBABILITY ANALYSIS
# ============================================================

def analyze_multilabel_probabilities(
    probabilities,
    label_names=None,
    threshold=MULTI_LABEL_THRESHOLD
):

    probabilities = validate_multilabel_probabilities(
        probabilities
    )

    label_count = len(probabilities)

    if not 0.0 <= threshold <= 1.0:

        raise ValueError(
            "Threshold must be between zero and one."
        )

    if label_names is None:

        label_names = [

            f"label_{index}"

            for index in range(label_count)
        ]

    if len(label_names) != label_count:

        raise ValueError(
            "Number of label names must match probabilities."
        )

    if len(set(label_names)) != len(label_names):

        raise ValueError(
            "Label names must be unique."
        )

    label_results = []

    binary_entropies = []

    selected_labels = []

    for index, probability in enumerate(probabilities):

        probability = float(probability)

        binary_entropy = calculate_binary_entropy(
            probability
        )

        binary_entropies.append(
            binary_entropy
        )

        label_confidence = max(
            probability,
            1.0 - probability
        )

        predicted_positive = (
            probability >= threshold
        )

        if predicted_positive:

            selected_labels.append(
                label_names[index]
            )

        label_results.append({

            "label_name": label_names[index],

            "probability": round(
                probability,
                ROUNDING_PRECISION
            ),

            "label_confidence": round(
                label_confidence,
                ROUNDING_PRECISION
            ),

            "binary_entropy": round(
                binary_entropy,
                ROUNDING_PRECISION
            ),

            "predicted_positive": bool(
                predicted_positive
            )
        })

    mean_binary_entropy = float(
        np.mean(binary_entropies)
    )

    mean_normalized_entropy = (
        mean_binary_entropy
    )

    # Binary entropy has a maximum of one bit.
    mean_uncertainty = float(
        np.clip(
            mean_normalized_entropy,
            0.0,
            1.0
        )
    )

    mean_label_confidence = float(
        np.mean([
            item["label_confidence"]
            for item in label_results
        ])
    )

    return {

        "output_type": "multi_label",

        "label_count": label_count,

        "threshold": threshold,

        "selected_labels": selected_labels,

        "mean_label_confidence": round(
            mean_label_confidence,
            ROUNDING_PRECISION
        ),

        "mean_binary_entropy": round(
            mean_binary_entropy,
            ROUNDING_PRECISION
        ),

        "normalized_uncertainty": round(
            mean_uncertainty,
            ROUNDING_PRECISION
        ),

        "uncertainty_band": interpret_uncertainty(
            mean_uncertainty
        ),

        "label_results": label_results,

        "calibration_status": CALIBRATION_STATUS,

        "is_calibrated": False
    }


# ============================================================
# 13. MULTI-LABEL LOGIT ANALYSIS
# ============================================================

def analyze_multilabel_logits(
    logits,
    label_names=None,
    threshold=MULTI_LABEL_THRESHOLD
):

    probabilities = sigmoid(logits)

    result = analyze_multilabel_probabilities(
        probabilities=probabilities,
        label_names=label_names,
        threshold=threshold
    )

    result["input_type"] = "logits"

    return result


# ============================================================
# 14. GENERAL WARNING
# ============================================================

def get_confidence_disclaimer():

    return (

        "Confidence metrics describe model output behavior. "

        "Raw probabilities are not necessarily calibrated. "

        "They do not establish clinical certainty, diagnosis, "

        "actual psychological state or student safety."
    )


# ============================================================
# 15. SMOKE VALIDATION
# ============================================================

def validate_confidence_metrics():

    # --------------------------------------------------------
    # A. Softmax
    # --------------------------------------------------------

    softmax_result = softmax(
        [2.0, 1.0, 0.1]
    )

    assert np.isclose(
        softmax_result.sum(),
        1.0
    )

    # --------------------------------------------------------
    # B. Sigmoid
    # --------------------------------------------------------

    sigmoid_result = sigmoid(
        [-2.0, 0.0, 2.0]
    )

    assert np.all(
        sigmoid_result >= 0.0
    )

    assert np.all(
        sigmoid_result <= 1.0
    )

    # --------------------------------------------------------
    # C. Entropy
    # --------------------------------------------------------

    entropy_result = calculate_entropy(
        [0.5, 0.5]
    )

    assert np.isclose(
        entropy_result,
        1.0
    )

    # --------------------------------------------------------
    # D. Normalized entropy
    # --------------------------------------------------------

    normalized_result = calculate_normalized_entropy(
        [0.5, 0.5]
    )

    assert np.isclose(
        normalized_result,
        1.0
    )

    # --------------------------------------------------------
    # E. Prediction margin
    # --------------------------------------------------------

    margin_result = calculate_prediction_margin(
        [0.7, 0.2, 0.1]
    )

    assert np.isclose(
        margin_result,
        0.5
    )

    # --------------------------------------------------------
    # F. Multi-class integration
    # --------------------------------------------------------

    multiclass_result = analyze_multiclass_logits(
        logits=[2.0, 1.0, 0.1],
        class_names=[
            "negative",
            "neutral",
            "positive"
        ]
    )

    assert multiclass_result[
        "predicted_class"
    ] == "negative"

    assert multiclass_result[
        "is_calibrated"
    ] is False

    # --------------------------------------------------------
    # G. Multi-label integration
    # --------------------------------------------------------

    multilabel_result = analyze_multilabel_logits(
        logits=[2.0, -1.0, 0.0],
        label_names=[
            "joy",
            "anger",
            "sadness"
        ]
    )

    assert multilabel_result[
        "label_count"
    ] == 3

    assert multilabel_result[
        "is_calibrated"
    ] is False

    # --------------------------------------------------------
    # H. Boundary entropy
    # --------------------------------------------------------

    assert calculate_binary_entropy(
        0.0
    ) == 0.0

    assert calculate_binary_entropy(
        1.0
    ) == 0.0

    # --------------------------------------------------------
    # I. Invalid input
    # --------------------------------------------------------

    try:

        analyze_multiclass_probabilities(
            [0.2, 0.2]
        )

        raise AssertionError(
            "Invalid probability sum was accepted."
        )

    except ValueError:

        pass

    try:

        analyze_multilabel_probabilities(
            [0.2, 1.2]
        )

        raise AssertionError(
            "Invalid probability range was accepted."
        )

    except ValueError:

        pass

    return True


# ============================================================
# 16. MAIN
# ============================================================

if __name__ == "__main__":

    validate_config()

    validate_confidence_metrics()

    print("=" * 70)

    print("CONFIDENCE METRICS ENGINE: SMOKE TEST PASSED")

    print("=" * 70)

    # Multi-class example.
    print("\nMULTI-CLASS EXAMPLE")

    multiclass = analyze_multiclass_logits(

        logits=[2.0, 1.0, 0.1],

        class_names=[
            "negative",
            "neutral",
            "positive"
        ]
    )

    for key, value in multiclass.items():

        print(f"{key}: {value}")

    # Multi-label example.
    print("\nMULTI-LABEL EXAMPLE")

    multilabel = analyze_multilabel_logits(

        logits=[2.0, -1.0, 0.0],

        label_names=[
            "joy",
            "anger",
            "sadness"
        ]
    )

    for key, value in multilabel.items():

        print(f"{key}: {value}")

    print("\nDISCLAIMER:")

    print(get_confidence_disclaimer())
