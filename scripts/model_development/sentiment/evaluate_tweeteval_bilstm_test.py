
import json

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config.tweeteval_bilstm_config import (
    DEVICE,
    TEST_FILE,
    MODEL_DIR,
    OUTPUT_DIR,
    CHECKPOINT_NAME,
    ID2LABEL,
    NUM_CLASSES,
    BATCH_SIZE,
    PAD_INDEX,
)

from scripts.model_development.sentiment.tweeteval_bilstm_dataset import (
    TweetEvalBiLSTMDataset,
    TweetEvalBiLSTMCollator,
)

from scripts.model_development.sentiment.tweeteval_bilstm_model import (
    TweetEvalBiLSTM,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

CLASS_IDS = list(range(NUM_CLASSES))

CLASS_NAMES = [
    ID2LABEL[index]
    for index in CLASS_IDS
]

CHECKPOINT_PATH = MODEL_DIR / CHECKPOINT_NAME

EVALUATION_DIR = (
    OUTPUT_DIR / "final_test_evaluation"
)

EXPECTED_TEST_RECORDS = 12284


# ============================================================
# 2. LOAD AND VALIDATE CHECKPOINT
# ============================================================

def load_best_checkpoint():

    print("=" * 70)
    print("LOADING BEST TWEETEVAL BILSTM CHECKPOINT")
    print("=" * 70)

    if not CHECKPOINT_PATH.is_file():

        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = [
        "model_state_dict",
        "vocabulary",
        "vocabulary_size",
        "epoch",
        "metrics",
        "label_mapping",
        "model_config",
    ]

    for key in required_keys:

        if key not in checkpoint:

            raise ValueError(
                f"Checkpoint missing required key: {key}"
            )

    vocabulary = checkpoint["vocabulary"]

    if len(vocabulary) != checkpoint["vocabulary_size"]:

        raise ValueError(
            "Checkpoint vocabulary size mismatch."
        )

    if checkpoint["label_mapping"] != ID2LABEL:

        raise ValueError(
            "Checkpoint label mapping mismatch."
        )

    model_config = checkpoint["model_config"]

    if model_config["num_classes"] != NUM_CLASSES:

        raise ValueError(
            "Checkpoint class count mismatch."
        )

    model = TweetEvalBiLSTM(
        checkpoint["vocabulary_size"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    print("Checkpoint:", CHECKPOINT_PATH)
    print("Best epoch:", checkpoint["epoch"])

    print(
        "Best validation Macro-F1:",
        checkpoint["metrics"]["macro_f1"],
    )

    print(
        "Vocabulary size:",
        checkpoint["vocabulary_size"],
    )

    print("Device:", DEVICE)

    return model, checkpoint


# ============================================================
# 3. PREPARE OFFICIAL TEST DATA
# ============================================================

def create_test_dataloader(vocabulary):

    print("\n" + "=" * 70)
    print("PREPARING OFFICIAL TEST DATA")
    print("=" * 70)

    if not TEST_FILE.is_file():

        raise FileNotFoundError(
            f"Test dataset not found: {TEST_FILE}"
        )

    # Use the vocabulary saved inside the checkpoint.
    # Do not rebuild or fit a vocabulary on test data.

    test_dataset = TweetEvalBiLSTMDataset(
        TEST_FILE,
        vocabulary,
    )

    if len(test_dataset) != EXPECTED_TEST_RECORDS:

        raise RuntimeError(
            f"Expected {EXPECTED_TEST_RECORDS} test records, "
            f"found {len(test_dataset)}."
        )

    collator = TweetEvalBiLSTMCollator()

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=collator,
    )

    print("Test dataset:", TEST_FILE)
    print("Test records:", len(test_dataset))
    print("Test batches:", len(test_loader))
    print("Vocabulary source: Saved best checkpoint")
    print("Shuffle: False")

    return test_dataset, test_loader


# ============================================================
# 4. RUN TEST INFERENCE
# ============================================================

def run_test_inference(
    model,
    test_loader,
    test_dataset,
):

    print("\n" + "=" * 70)
    print("RUNNING OFFICIAL TEST INFERENCE")
    print("=" * 70)

    all_true = []
    all_predictions = []
    all_probabilities = []

    with torch.inference_mode():

        for batch_index, batch in enumerate(test_loader):

            input_ids = batch["input_ids"].to(
                DEVICE,
                non_blocking=True,
            )

            lengths = batch["lengths"]

            labels = batch["labels"]

            logits = model(
                input_ids,
                lengths,
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            all_true.extend(
                labels.tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_probabilities.extend(
                probabilities.cpu().tolist()
            )

            if (batch_index + 1) % 50 == 0:

                print(
                    f"Processed batches: "
                    f"{batch_index + 1}/{len(test_loader)}"
                )

    y_true = np.asarray(
        all_true,
        dtype=np.int64,
    )

    y_pred = np.asarray(
        all_predictions,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        all_probabilities,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Inference integrity checks.
    # --------------------------------------------------------

    if len(y_true) != len(test_dataset):

        raise RuntimeError(
            "Test prediction count does not match dataset size."
        )

    if len(y_pred) != len(y_true):

        raise RuntimeError(
            "Prediction and label counts differ."
        )

    if probabilities.shape != (
        len(y_true),
        NUM_CLASSES,
    ):

        raise RuntimeError(
            "Unexpected probability matrix shape."
        )

    if not np.isfinite(probabilities).all():

        raise RuntimeError(
            "Non-finite probabilities detected."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-5,
    ):

        raise RuntimeError(
            "Probability rows do not sum to one."
        )

    if not set(np.unique(y_true)).issubset(
        set(CLASS_IDS)
    ):

        raise RuntimeError(
            "Invalid test labels detected."
        )

    print("\nTEST INFERENCE VERIFICATION")
    print("Records evaluated:", len(y_true))
    print("Prediction count: PASSED")
    print("Probability shape: PASSED")
    print("Finite probabilities: PASSED")
    print("Probability sums: PASSED")
    print("Label validation: PASSED")

    return y_true, y_pred, probabilities


# ============================================================
# 5. OVERALL TEST METRICS
# ============================================================

def calculate_overall_metrics(
    y_true,
    y_pred,
    probabilities,
    checkpoint,
):

    confidence = probabilities.max(
        axis=1
    )

    metrics = {

        "dataset": "TweetEval Official Test",

        "records": int(len(y_true)),

        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),

        "best_validation_macro_f1": float(
            checkpoint["metrics"]["macro_f1"]
        ),

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),

        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "mean_confidence": float(
            confidence.mean()
        ),

        "median_confidence": float(
            np.median(confidence)
        ),

        "confidence_p25": float(
            np.percentile(confidence, 25)
        ),

        "confidence_p75": float(
            np.percentile(confidence, 75)
        ),

        "confidence_p95": float(
            np.percentile(confidence, 95)
        ),

    }

    return metrics


# ============================================================
# 6. PER-CLASS METRICS
# ============================================================

def calculate_per_class_metrics(
    y_true,
    y_pred,
):

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    rows = []

    for class_id, class_name in zip(
        CLASS_IDS,
        CLASS_NAMES,
    ):

        class_metrics = report[class_name]

        rows.append({

            "label_id": class_id,

            "sentiment": class_name,

            "precision": class_metrics["precision"],

            "recall": class_metrics["recall"],

            "f1_score": class_metrics["f1-score"],

            "support": int(
                class_metrics["support"]
            ),

            "predicted_count": int(
                np.sum(y_pred == class_id)
            ),

        })

    return pd.DataFrame(rows)


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

def calculate_confusion_matrix(
    y_true,
    y_pred,
):

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_IDS,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"actual_{name}"
            for name in CLASS_NAMES
        ],
        columns=[
            f"predicted_{name}"
            for name in CLASS_NAMES
        ],
    )

    return matrix, matrix_df


# ============================================================
# 8. ACTUAL VS PREDICTED DISTRIBUTION
# ============================================================

def calculate_class_distribution(
    y_true,
    y_pred,
):

    total = len(y_true)

    rows = []

    for class_id, class_name in zip(
        CLASS_IDS,
        CLASS_NAMES,
    ):

        actual_count = int(
            np.sum(y_true == class_id)
        )

        predicted_count = int(
            np.sum(y_pred == class_id)
        )

        rows.append({

            "sentiment": class_name,

            "actual_count": actual_count,

            "predicted_count": predicted_count,

            "actual_percentage": (
                actual_count / total
            ) * 100,

            "predicted_percentage": (
                predicted_count / total
            ) * 100,

        })

    return pd.DataFrame(rows)


# ============================================================
# 9. SAVE TEST EVALUATION ARTIFACTS
# ============================================================

def save_test_artifacts(
    metrics,
    per_class_df,
    confusion_df,
    confusion_array,
    distribution_df,
    y_true,
    y_pred,
    probabilities,
):

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Overall summary.
    # --------------------------------------------------------

    with open(
        EVALUATION_DIR / "final_test_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Per-class metrics.
    # --------------------------------------------------------

    per_class_df.to_csv(
        EVALUATION_DIR / "per_class_test_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Confusion matrix.
    # --------------------------------------------------------

    confusion_df.to_csv(
        EVALUATION_DIR / "test_confusion_matrix.csv"
    )

    # --------------------------------------------------------
    # Class distribution.
    # --------------------------------------------------------

    distribution_df.to_csv(
        EVALUATION_DIR / "test_class_distribution.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Individual predictions.
    # --------------------------------------------------------

    prediction_df = pd.DataFrame({

        "record_index": np.arange(
            len(y_true)
        ),

        "actual_label_id": y_true,

        "actual_sentiment": [
            ID2LABEL[int(value)]
            for value in y_true
        ],

        "predicted_label_id": y_pred,

        "predicted_sentiment": [
            ID2LABEL[int(value)]
            for value in y_pred
        ],

        "confidence": probabilities.max(
            axis=1
        ),

        "probability_negative": probabilities[:, 0],

        "probability_neutral": probabilities[:, 1],

        "probability_positive": probabilities[:, 2],

        "correct_prediction": (
            y_true == y_pred
        ),

    })

    prediction_df.to_csv(
        EVALUATION_DIR / "test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Numerical arrays.
    # --------------------------------------------------------

    np.savez_compressed(
        EVALUATION_DIR / "test_predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        confusion_matrix=confusion_array,
    )

    print("\nTEST ARTIFACTS SAVED")
    print("-" * 70)

    for file_path in sorted(
        EVALUATION_DIR.iterdir()
    ):

        print(file_path.name)


# ============================================================
# 10. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 70)
    print("TWEETEVAL BILSTM FINAL TEST EVALUATION")
    print("=" * 70)

    # Load frozen best checkpoint.
    model, checkpoint = load_best_checkpoint()

    # Prepare official test set using saved vocabulary.
    test_dataset, test_loader = create_test_dataloader(
        checkpoint["vocabulary"]
    )

    # Run inference.
    y_true, y_pred, probabilities = run_test_inference(
        model,
        test_loader,
        test_dataset,
    )

    # Calculate metrics.
    metrics = calculate_overall_metrics(
        y_true,
        y_pred,
        probabilities,
        checkpoint,
    )

    per_class_df = calculate_per_class_metrics(
        y_true,
        y_pred,
    )

    confusion_array, confusion_df = (
        calculate_confusion_matrix(
            y_true,
            y_pred,
        )
    )

    distribution_df = calculate_class_distribution(
        y_true,
        y_pred,
    )

    # Save all artifacts.
    save_test_artifacts(
        metrics,
        per_class_df,
        confusion_df,
        confusion_array,
        distribution_df,
        y_true,
        y_pred,
        probabilities,
    )

    # --------------------------------------------------------
    # Display overall metrics.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    for metric, value in metrics.items():

        print(f"{metric}: {value}")

    # --------------------------------------------------------
    # Display per-class metrics.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-CLASS TEST METRICS")
    print("=" * 70)

    print(
        per_class_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------------
    # Display confusion matrix.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST CONFUSION MATRIX")
    print("=" * 70)

    print(
        confusion_df.to_string()
    )

    # --------------------------------------------------------
    # Display class distribution.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ACTUAL VS PREDICTED CLASS DISTRIBUTION")
    print("=" * 70)

    print(
        distribution_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION COMPLETED")
    print("=" * 70)

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Output directory:", EVALUATION_DIR)

    print(
        "\nIMPORTANT: These test results are for final "
        "independent evaluation only. Do not tune the model "
        "using this test set."
    )


if __name__ == "__main__":

    main()
