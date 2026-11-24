
from backend.app.inference.inference_pipeline import InferencePipeline


def main():

    pipeline = InferencePipeline()

    test_text = (
        "I feel very happy, excited and grateful today."
    )

    print("\n========== CONFIDENCE INTEGRATION TEST ==========")

    # --------------------------------------------------
    # 1. Test emotion confidence analysis
    # --------------------------------------------------

    emotion = pipeline.analyze_emotion(test_text)

    emotion_confidence = emotion["confidence_analysis"]

    assert emotion["status"] == "evaluated"
    assert emotion_confidence["output_type"] == "multi_label"
    assert emotion_confidence["label_count"] == 28

    assert len(emotion_confidence["label_results"]) == 28

    assert emotion["thresholds_applied"] is True

    assert emotion_confidence["is_calibrated"] is False

    assert emotion_confidence["calibration_status"] == "not_calibrated"

    # Verify every emotion threshold.
    for item in emotion_confidence["label_results"]:

        label = item["label_name"]

        probability = item["probability"]

        threshold = pipeline.emotion_thresholds[label]

        expected = probability >= threshold

        assert item["threshold"] == threshold

        assert item["predicted_positive"] == expected

    assert set(emotion_confidence["selected_labels"]) == {
        item["label_name"]
        for item in emotion_confidence["label_results"]
        if item["probability"] >= pipeline.emotion_thresholds[
            item["label_name"]
        ]
    }

    print("\nEMOTION CONFIDENCE: PASSED")

    print(
        "Mean label confidence:",
        emotion_confidence["mean_label_confidence"]
    )

    print(
        "Normalized uncertainty:",
        emotion_confidence["normalized_uncertainty"]
    )

    print(
        "Uncertainty band:",
        emotion_confidence["uncertainty_band"]
    )

    print(
        "Selected labels:",
        emotion_confidence["selected_labels"]
    )

    # --------------------------------------------------
    # 2. Test sentiment multiclass confidence
    # --------------------------------------------------

    sentiment = pipeline.analyze_sentiment(test_text)

    sentiment_confidence = sentiment["confidence_analysis"]

    assert sentiment["status"] == "evaluated"

    assert sentiment_confidence["output_type"] == "multi_class"

    assert sentiment_confidence["predicted_class"] == (
        sentiment["predicted_class"]
    )

    assert 0.0 <= sentiment_confidence["confidence"] <= 1.0

    assert 0.0 <= sentiment_confidence["normalized_entropy"] <= 1.0

    assert 0.0 <= sentiment_confidence["prediction_margin"] <= 1.0

    assert sentiment_confidence["is_calibrated"] is False

    assert sentiment_confidence["calibration_status"] == "not_calibrated"

    assert len(sentiment_confidence["top_predictions"]) == 3

    print("\nSENTIMENT CONFIDENCE: PASSED")

    print(
        "Predicted class:",
        sentiment_confidence["predicted_class"]
    )

    print(
        "Confidence:",
        sentiment_confidence["confidence"]
    )

    print(
        "Normalized entropy:",
        sentiment_confidence["normalized_entropy"]
    )

    print(
        "Prediction margin:",
        sentiment_confidence["prediction_margin"]
    )

    print(
        "Uncertainty band:",
        sentiment_confidence["uncertainty_band"]
    )

    # --------------------------------------------------
    # 2b. Test sentiment BiLSTM multiclass confidence
    # --------------------------------------------------

    bilstm = pipeline.analyze_sentiment_bilstm(test_text)

    bilstm_confidence = bilstm["confidence_analysis"]

    assert bilstm["status"] == "evaluated"
    assert bilstm_confidence["output_type"] == "multi_class"
    assert bilstm_confidence["predicted_class"] == bilstm["predicted_class"]
    assert 0.0 <= bilstm_confidence["confidence"] <= 1.0
    assert 0.0 <= bilstm_confidence["normalized_entropy"] <= 1.0
    assert 0.0 <= bilstm_confidence["prediction_margin"] <= 1.0
    assert bilstm_confidence["is_calibrated"] is False
    assert bilstm_confidence["calibration_status"] == "not_calibrated"
    assert len(bilstm_confidence["top_predictions"]) == 3

    print("\nSENTIMENT BILSTM CONFIDENCE: PASSED")

    # --------------------------------------------------
    # 3. Verify no artificial calibration
    # --------------------------------------------------

    assert emotion["is_calibrated"] is False

    assert sentiment["is_calibrated"] is False

    assert bilstm["is_calibrated"] is False

    print("\nCALIBRATION POLICY: PASSED")

    print("\nALL CONFIDENCE INTEGRATION TESTS: PASSED")


if __name__ == "__main__":
    main()
