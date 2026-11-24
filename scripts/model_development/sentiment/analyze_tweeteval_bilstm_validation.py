
import json

import numpy as np
import pandas as pd
import torch

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
    MODEL_DIR,
    OUTPUT_DIR,
    CHECKPOINT_NAME,
    ID2LABEL,
    NUM_CLASSES,
)

from scripts.model_development.sentiment.tweeteval_bilstm_dataset import (
    create_bilstm_dataloaders,
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

ANALYSIS_DIR = (
    OUTPUT_DIR / "validation_analysis"
)


# ============================================================
# 2. VALIDATE ENVIRONMENT AND CHECKPOINT
# ============================================================

def validate_checkpoint(checkpoint):

    required_keys = [
        "model_state_dict",
        "vocabulary",
        "vocabulary_size",
        "epoch",
        "metrics",
        "label_mapping",
    ]

    for key in required_keys:

        if key not in checkpoint:

            raise ValueError(
                f"Checkpoint is missing required key: {key}"
            )

    if checkpoint["vocabulary_size"] != len(
        checkpoint["vocabulary"]
    ):

        raise ValueError(
            "Checkpoint vocabulary size mismatch."
        )

    if checkpoint["label_mapping"] != ID2LABEL:

        raise ValueError(
            "Checkpoint label mapping does not match configuration."
        )


# ============================================================
# 3. LOAD BEST MODEL
# ============================================================

def load_best_model():

    print("=" * 70)
    print("LOADING BEST TWEETEVAL BILSTM CHECKPOINT")
    print("=" * 70)

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # weights_only=False is required because this checkpoint
    # intentionally contains vocabulary and metadata objects.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    validate_checkpoint(checkpoint)

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
    print("Vocabulary size:", checkpoint["vocabulary_size"])
    print("Device:", DEVICE)

    return model, checkpoint


# ============================================================
# 4. RUN VALIDATION INFERENCE
# ============================================================

def run_validation_inference(
    model,
    checkpoint_vocabulary,
):

    print("\n" + "=" * 70)
    print("RUNNING VALIDATION INFERENCE")
    print("=" * 70)

    (
        current_vocabulary,
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader,
    ) = create_bilstm_dataloaders()

    # Ensure inference uses exactly the vocabulary used
    # by the saved checkpoint.
    if current_vocabulary != checkpoint_vocabulary:

        raise RuntimeError(
            "Rebuilt vocabulary differs from checkpoint vocabulary."
        )

    print("Vocabulary consistency: PASSED")
    print(
        "Validation records:",
        len(validation_dataset),
    )

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.inference_mode():

        for batch in validation_loader:

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

            all_labels.extend(
                labels.tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_probabilities.extend(
                probabilities.cpu().tolist()
            )

    y_true = np.asarray(
        all_labels,
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

    if len(y_true) != len(validation_dataset):

        raise RuntimeError(
            "Validation prediction count mismatch."
        )

    if not np.isfinite(probabilities).all():

        raise RuntimeError(
            "Non-finite prediction probabilities detected."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-5,
    ):

        raise RuntimeError(
            "Softmax probability rows do not sum to one."
        )

    print("Predictions generated:", len(y_pred))
    print("Probability verification: PASSED")

    return (
        y_true,
        y_pred,
        probabilities,
    )


# ============================================================
# 5. CALCULATE OVERALL METRICS
# ============================================================

def calculate_overall_metrics(
    y_true,
    y_pred,
    probabilities,
    checkpoint,
):

    metrics = {

        "records": int(len(y_true)),

        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),

        "checkpoint_validation_macro_f1": float(
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
            probabilities.max(axis=1).mean()
        ),

        "median_confidence": float(
            np.median(
                probabilities.max(axis=1)
            )
        ),

        "confidence_p25": float(
            np.percentile(
                probabilities.max(axis=1),
                25,
            )
        ),

        "confidence_p75": float(
            np.percentile(
                probabilities.max(axis=1),
                75,
            )
        ),

        "confidence_p95": float(
            np.percentile(
                probabilities.max(axis=1),
                95,
            )
        ),
    }

    return metrics


# ============================================================
# 6. PER-CLASS ANALYSIS
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

        class_report = report[class_name]

        rows.append({

            "label_id": class_id,

            "sentiment": class_name,

            "precision": class_report["precision"],

            "recall": class_report["recall"],

            "f1_score": class_report["f1-score"],

            "support": int(
                class_report["support"]
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
# 8. CLASS DISTRIBUTION
# ============================================================

def calculate_class_distribution(
    y_true,
    y_pred,
):

    rows = []

    total = len(y_true)

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
# 9. SAVE ANALYSIS ARTIFACTS
# ============================================================

def save_analysis(
    metrics,
    per_class_df,
    confusion_df,
    confusion_array,
    distribution_df,
    y_true,
    y_pred,
    probabilities,
):

    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Overall metrics.
    summary_path = (
        ANALYSIS_DIR / "validation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # Per-class metrics.
    per_class_df.to_csv(
        ANALYSIS_DIR / "per_class_validation_metrics.csv",
        index=False,
    )

    # Confusion matrix.
    confusion_df.to_csv(
        ANALYSIS_DIR / "validation_confusion_matrix.csv"
    )

    # Actual and predicted class distributions.
    distribution_df.to_csv(
        ANALYSIS_DIR / "validation_class_distribution.csv",
        index=False,
    )

    # Individual predictions and probabilities.
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
        ANALYSIS_DIR / "validation_predictions.csv",
        index=False,
    )

    np.savez_compressed(
        ANALYSIS_DIR / "validation_predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        confusion_matrix=confusion_array,
    )

    print("\nANALYSIS ARTIFACTS SAVED")
    print("-" * 70)

    for file_path in sorted(
        ANALYSIS_DIR.iterdir()
    ):

        print(file_path.name)


# ============================================================
# 10. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 70)
    print("TWEETEVAL BILSTM VALIDATION ANALYSIS")
    print("=" * 70)

    model, checkpoint = load_best_model()

    (
        y_true,
        y_pred,
        probabilities,
    ) = run_validation_inference(
        model,
        checkpoint["vocabulary"],
    )

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

    save_analysis(
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
    # Display overall results.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERALL VALIDATION RESULTS")
    print("=" * 70)

    for metric, value in metrics.items():

        print(f"{metric}: {value}")

    # --------------------------------------------------------
    # Display per-class results.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PER-CLASS VALIDATION METRICS")
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
    print("VALIDATION CONFUSION MATRIX")
    print("=" * 70)

    print(confusion_df.to_string())

    # --------------------------------------------------------
    # Display distribution.
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
    print("VALIDATION ANALYSIS COMPLETED")
    print("=" * 70)

    print("Output directory:", ANALYSIS_DIR)


if __name__ == "__main__":

    main()
