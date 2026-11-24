
import json

import numpy as np
import pandas as pd
import torch

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from config.tweeteval_config import (
    MODEL_DIR,
    OUTPUT_DIR,
    VALIDATION_FILE,
    BATCH_SIZE,
    DEVICE,
    USE_FP16,
    ID2LABEL,
    NUM_LABELS,
)

from scripts.model_development.sentiment.tweeteval_dataset import (
    TweetEvalDataset,
    TweetEvalCollator,
)


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

CHECKPOINT_DIR = MODEL_DIR / "best_checkpoint"

ANALYSIS_DIR = (
    OUTPUT_DIR / "validation_analysis"
)

CLASS_IDS = list(range(NUM_LABELS))

CLASS_NAMES = [
    ID2LABEL[index]
    for index in CLASS_IDS
]


# ============================================================
# 2. VALIDATE INPUT FILES
# ============================================================

def validate_inputs():

    if not CHECKPOINT_DIR.is_dir():
        raise FileNotFoundError(
            f"Best checkpoint not found: {CHECKPOINT_DIR}"
        )

    if not VALIDATION_FILE.is_file():
        raise FileNotFoundError(
            f"Validation dataset not found: {VALIDATION_FILE}"
        )

    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Checkpoint verified.")
    print("Validation dataset verified.")
    print("Output directory:", ANALYSIS_DIR)


# ============================================================
# 3. LOAD VALIDATION DATASET
# ============================================================

def load_validation_data():

    print("\nLoading checkpoint tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_DIR
    )

    print("Tokenizer loaded.")

    print("\nLoading validation dataset...")

    validation_dataset = TweetEvalDataset(
        VALIDATION_FILE,
        tokenizer,
    )

    collator = TweetEvalCollator(
        tokenizer
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=collator,
    )

    print("\nValidation records:", len(validation_dataset))
    print("Validation batches:", len(validation_loader))

    return (
        tokenizer,
        validation_dataset,
        validation_loader,
    )


# ============================================================
# 4. LOAD BEST TRAINED MODEL
# ============================================================

def load_best_model():

    print("\nLoading best trained checkpoint...")

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT_DIR
    )

    if model.config.num_labels != NUM_LABELS:
        raise ValueError(
            "Checkpoint class count does not match configuration."
        )

    model = model.to(DEVICE)

    model.eval()

    print("Checkpoint loaded successfully.")
    print("Model device:", DEVICE)
    print("Number of classes:", model.config.num_labels)

    return model


# ============================================================
# 5. GENERATE VALIDATION PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    validation_dataset,
    validation_loader,
):

    all_true_labels = []
    all_predicted_labels = []
    all_probabilities = []

    print("\nGenerating validation predictions...")

    with torch.inference_mode():

        for batch in tqdm(
            validation_loader,
            desc="Validation inference",
        ):

            input_ids = batch["input_ids"].to(
                DEVICE,
                non_blocking=True,
            )

            attention_mask = batch["attention_mask"].to(
                DEVICE,
                non_blocking=True,
            )

            labels = batch["labels"].to(
                DEVICE,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=USE_FP16,
            ):

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            logits = outputs.logits.float()

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            all_true_labels.extend(
                labels.cpu().numpy().tolist()
            )

            all_predicted_labels.extend(
                predictions.cpu().numpy().tolist()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

    true_labels = np.asarray(
        all_true_labels,
        dtype=np.int64,
    )

    predicted_labels = np.asarray(
        all_predicted_labels,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        all_probabilities,
        dtype=np.float32,
    )

    expected_records = len(validation_dataset)

    if not (
        len(true_labels)
        == len(predicted_labels)
        == len(probabilities)
        == expected_records
    ):
        raise RuntimeError(
            "Prediction count does not match validation dataset size."
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
            "Probability rows do not sum to 1."
        )

    return (
        true_labels,
        predicted_labels,
        probabilities,
    )


# ============================================================
# 6. CALCULATE OVERALL METRICS
# ============================================================

def calculate_overall_metrics(
    true_labels,
    predicted_labels,
    probabilities,
):

    metrics = {
        "validation_records": int(len(true_labels)),

        "accuracy": float(
            accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),

        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),

        "weighted_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="weighted",
                zero_division=0,
            )
        ),

        "macro_precision": float(
            precision_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),

        "macro_recall": float(
            recall_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),

        "mean_prediction_confidence": float(
            probabilities.max(axis=1).mean()
        ),

        "median_prediction_confidence": float(
            np.median(
                probabilities.max(axis=1)
            )
        ),
    }

    confidence_values = probabilities.max(
        axis=1
    )

    metrics["confidence_percentiles"] = {
        "p25": float(
            np.percentile(confidence_values, 25)
        ),

        "p50": float(
            np.percentile(confidence_values, 50)
        ),

        "p75": float(
            np.percentile(confidence_values, 75)
        ),

        "p95": float(
            np.percentile(confidence_values, 95)
        ),
    }

    return metrics


# ============================================================
# 7. PER-CLASS METRICS
# ============================================================

def calculate_class_metrics(
    true_labels,
    predicted_labels,
):

    report = classification_report(
        true_labels,
        predicted_labels,
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

        class_result = report[class_name]

        actual_support = int(
            np.sum(true_labels == class_id)
        )

        predicted_support = int(
            np.sum(predicted_labels == class_id)
        )

        rows.append({
            "label_id": class_id,
            "sentiment": class_name,
            "precision": class_result["precision"],
            "recall": class_result["recall"],
            "f1_score": class_result["f1-score"],
            "support": actual_support,
            "predicted_count": predicted_support,
        })

    return pd.DataFrame(rows)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

def calculate_confusion_matrix(
    true_labels,
    predicted_labels,
):

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=CLASS_IDS,
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=[
            f"Actual_{name}"
            for name in CLASS_NAMES
        ],
        columns=[
            f"Predicted_{name}"
            for name in CLASS_NAMES
        ],
    )

    return confusion_df


# ============================================================
# 9. CLASS DISTRIBUTION
# ============================================================

def calculate_class_distribution(
    true_labels,
    predicted_labels,
):

    rows = []

    total = len(true_labels)

    for class_id, class_name in zip(
        CLASS_IDS,
        CLASS_NAMES,
    ):

        actual_count = int(
            np.sum(true_labels == class_id)
        )

        predicted_count = int(
            np.sum(predicted_labels == class_id)
        )

        rows.append({
            "label_id": class_id,
            "sentiment": class_name,
            "actual_count": actual_count,
            "actual_percentage": (
                actual_count / total * 100
            ),
            "predicted_count": predicted_count,
            "predicted_percentage": (
                predicted_count / total * 100
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# 10. SAVE INDIVIDUAL PREDICTIONS
# ============================================================

def save_prediction_records(
    validation_dataset,
    true_labels,
    predicted_labels,
    probabilities,
):

    rows = []

    for index, text in enumerate(
        validation_dataset.texts
    ):

        predicted_class = int(
            predicted_labels[index]
        )

        true_class = int(
            true_labels[index]
        )

        row = {
            "text": text,
            "actual_label": true_class,
            "actual_sentiment": ID2LABEL[true_class],
            "predicted_label": predicted_class,
            "predicted_sentiment": ID2LABEL[predicted_class],
            "prediction_confidence": float(
                probabilities[index, predicted_class]
            ),
            "correct_prediction": (
                true_class == predicted_class
            ),
        }

        for class_id, class_name in zip(
            CLASS_IDS,
            CLASS_NAMES,
        ):

            row[f"probability_{class_name}"] = float(
                probabilities[index, class_id]
            )

        rows.append(row)

    predictions_df = pd.DataFrame(rows)

    predictions_path = (
        ANALYSIS_DIR / "validation_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    print(
        "Individual predictions saved:",
        predictions_path,
    )


# ============================================================
# 11. SAVE ANALYSIS RESULTS
# ============================================================

def save_analysis(
    metrics,
    class_metrics,
    confusion_df,
    distribution_df,
    true_labels,
    predicted_labels,
    probabilities,
):

    # Overall summary.
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
    class_metrics.to_csv(
        ANALYSIS_DIR / "per_class_validation_metrics.csv",
        index=False,
    )

    # Confusion matrix.
    confusion_df.to_csv(
        ANALYSIS_DIR / "validation_confusion_matrix.csv"
    )

    # Actual vs predicted class distribution.
    distribution_df.to_csv(
        ANALYSIS_DIR / "validation_class_distribution.csv",
        index=False,
    )

    # Numeric arrays for future analysis.
    np.savez_compressed(
        ANALYSIS_DIR / "validation_predictions.npz",
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        probabilities=probabilities,
    )

    print("\nAll analysis files saved.")


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

def display_results(
    metrics,
    class_metrics,
    confusion_df,
    distribution_df,
):

    print("\n" + "=" * 70)
    print("TWEETEVAL VALIDATION ANALYSIS")
    print("=" * 70)

    print("\nOVERALL METRICS")

    for metric_name, value in metrics.items():

        if isinstance(value, dict):
            print(f"\n{metric_name}:")

            for key, item in value.items():
                print(f"  {key}: {item:.6f}")

        elif isinstance(value, float):
            print(f"{metric_name}: {value:.6f}")

        else:
            print(f"{metric_name}: {value}")

    print("\nPER-CLASS METRICS")

    print(
        class_metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nCONFUSION MATRIX")

    print(confusion_df.to_string())

    print("\nACTUAL VS PREDICTED DISTRIBUTION")

    print(
        distribution_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    print("\n" + "=" * 70)


# ============================================================
# 13. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 70)
    print("TWEETEVAL VALIDATION ANALYSIS")
    print("=" * 70)

    validate_inputs()

    (
        tokenizer,
        validation_dataset,
        validation_loader,
    ) = load_validation_data()

    model = load_best_model()

    (
        true_labels,
        predicted_labels,
        probabilities,
    ) = generate_predictions(
        model,
        validation_dataset,
        validation_loader,
    )

    metrics = calculate_overall_metrics(
        true_labels,
        predicted_labels,
        probabilities,
    )

    class_metrics = calculate_class_metrics(
        true_labels,
        predicted_labels,
    )

    confusion_df = calculate_confusion_matrix(
        true_labels,
        predicted_labels,
    )

    distribution_df = calculate_class_distribution(
        true_labels,
        predicted_labels,
    )

    save_prediction_records(
        validation_dataset,
        true_labels,
        predicted_labels,
        probabilities,
    )

    save_analysis(
        metrics,
        class_metrics,
        confusion_df,
        distribution_df,
        true_labels,
        predicted_labels,
        probabilities,
    )

    display_results(
        metrics,
        class_metrics,
        confusion_df,
        distribution_df,
    )

    print("\nVALIDATION ANALYSIS COMPLETED")
    print("Results directory:", ANALYSIS_DIR)


if __name__ == "__main__":

    main()
