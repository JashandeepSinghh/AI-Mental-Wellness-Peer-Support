
"""
FIG-Loneliness Validation Analysis

Evaluates the frozen best checkpoint on the validation split.
Reads original validation text separately from the CSV because
the training dataset object is a PyTorch Dataset, not a DataFrame.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)

from scripts.model_development.loneliness.config import (
    fig_loneliness_config as config
)

from scripts.model_development.loneliness.fig_loneliness_dataset import (
    create_data_pipeline
)

from scripts.model_development.loneliness.fig_loneliness_model import (
    initialize_model
)


# ============================================================
# 1. LOGGING AND PATHS
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

VALIDATION_CSV = Path(
    "datasets/processed/fig_loneliness/standardized/validation.csv"
)


# ============================================================
# 2. LOAD BEST CHECKPOINT
# ============================================================

def load_best_checkpoint(model):

    checkpoint_path = (
        config.CHECKPOINT_DIR / "best_checkpoint.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=config.DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    logger.info(
        "Loaded best checkpoint from epoch %d",
        checkpoint["epoch"]
    )

    logger.info(
        "Checkpoint validation Macro-F1: %.5f",
        checkpoint["best_validation_macro_f1"]
    )

    return checkpoint


# ============================================================
# 3. GENERATE VALIDATION PREDICTIONS
# ============================================================

@torch.no_grad()
def generate_validation_predictions(
    model,
    validation_loader
):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for batch in validation_loader:

        batch = {
            key: value.to(
                config.DEVICE,
                non_blocking=True
            )
            for key, value in batch.items()
        }

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=config.USE_MIXED_PRECISION
        ):

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

        probabilities = torch.softmax(
            outputs.logits.float(),
            dim=-1
        )

        predictions = torch.argmax(
            probabilities,
            dim=-1
        )

        all_labels.extend(
            batch["labels"].cpu().tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_probabilities.extend(
            probabilities.cpu().tolist()
        )

    return (
        np.asarray(all_labels),
        np.asarray(all_predictions),
        np.asarray(all_probabilities)
    )


# ============================================================
# 4. CALCULATE METRICS
# ============================================================

def calculate_metrics(
    labels,
    predictions,
    probabilities
):

    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["non_lonely", "lonely"],
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    )

    confidence = probabilities.max(axis=1)

    return {
        "split": "validation",
        "records": int(len(labels)),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "non_lonely": report["non_lonely"],
        "lonely": report["lonely"],
        "confusion_matrix": matrix.tolist(),
        "confidence": {
            "mean": float(np.mean(confidence)),
            "median": float(np.median(confidence)),
            "p25": float(np.percentile(confidence, 25)),
            "p75": float(np.percentile(confidence, 75)),
            "p95": float(np.percentile(confidence, 95))
        }
    }


# ============================================================
# 5. SAVE VALIDATION ARTIFACTS
# ============================================================

def save_analysis_artifacts(
    validation_dataframe,
    labels,
    predictions,
    probabilities,
    metrics
):

    output_dir = (
        config.TRAINING_OUTPUT_DIR
        / "validation_analysis"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_df = pd.DataFrame({
        "text": validation_dataframe[config.TEXT_COLUMN].tolist(),
        "actual_label": labels,
        "predicted_label": predictions,
        "probability_non_lonely": probabilities[:, 0],
        "probability_lonely": probabilities[:, 1],
        "confidence": probabilities.max(axis=1),
        "correct": labels == predictions
    })

    prediction_df.to_csv(
        output_dir / "validation_predictions.csv",
        index=False
    )

    prediction_df[
        ~prediction_df["correct"]
    ].to_csv(
        output_dir / "validation_misclassified_examples.csv",
        index=False
    )

    class_rows = []

    for class_name in ["non_lonely", "lonely"]:

        result = metrics[class_name]

        class_rows.append({
            "class": class_name,
            "precision": result["precision"],
            "recall": result["recall"],
            "f1_score": result["f1-score"],
            "support": result["support"]
        })

    pd.DataFrame(class_rows).to_csv(
        output_dir / "per_class_validation_metrics.csv",
        index=False
    )

    matrix = np.asarray(
        metrics["confusion_matrix"]
    )

    pd.DataFrame(
        matrix,
        index=["actual_non_lonely", "actual_lonely"],
        columns=["predicted_non_lonely", "predicted_lonely"]
    ).to_csv(
        output_dir / "validation_confusion_matrix.csv"
    )

    with open(
        output_dir / "validation_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    logger.info(
        "Validation artifacts saved to: %s",
        output_dir
    )

    return output_dir


# ============================================================
# 6. PRINT RESULTS
# ============================================================

def print_results(metrics):

    matrix = np.asarray(
        metrics["confusion_matrix"]
    )

    print("\n" + "=" * 70)
    print("FIG-LONELINESS VALIDATION EVALUATION")
    print("=" * 70)

    print(f"Records         : {metrics['records']}")
    print(f"Accuracy        : {metrics['accuracy']:.6f}")
    print(f"Macro Precision : {metrics['macro_precision']:.6f}")
    print(f"Macro Recall    : {metrics['macro_recall']:.6f}")
    print(f"Macro F1        : {metrics['macro_f1']:.6f}")
    print(f"Weighted F1     : {metrics['weighted_f1']:.6f}")

    print("\nPER-CLASS PERFORMANCE")
    print("-" * 70)

    for class_name in ["non_lonely", "lonely"]:

        result = metrics[class_name]

        print(f"\n{class_name.upper()}")
        print(f"Precision : {result['precision']:.6f}")
        print(f"Recall    : {result['recall']:.6f}")
        print(f"F1-score  : {result['f1-score']:.6f}")
        print(f"Support   : {result['support']}")

    print("\nCONFUSION MATRIX")
    print("-" * 70)
    print("                    Predicted Non-lonely | Predicted Lonely")
    print(f"Actual Non-lonely  | {matrix[0, 0]:>20} | {matrix[0, 1]:>16}")
    print(f"Actual Lonely      | {matrix[1, 0]:>20} | {matrix[1, 1]:>16}")

    print("\nCONFIDENCE STATISTICS")
    print("-" * 70)

    for key, value in metrics["confidence"].items():
        print(f"{key:<12}: {value:.6f}")

    print("\nVALIDATION EVALUATION COMPLETED")
    print("=" * 70)


# ============================================================
# 7. MAIN
# ============================================================

def main():

    config.validate_config()

    logger.info("Loading validation dataset and tokenizer.")

    tokenizer, datasets, dataloaders = create_data_pipeline()

    # Read original text from CSV.
    # Do not treat the PyTorch Dataset object as a DataFrame.
    if not VALIDATION_CSV.exists():
        raise FileNotFoundError(
            f"Validation CSV not found: {VALIDATION_CSV.resolve()}"
        )

    validation_dataframe = pd.read_csv(
        VALIDATION_CSV
    )

    if config.TEXT_COLUMN not in validation_dataframe.columns:
        raise KeyError(
            f"Column '{config.TEXT_COLUMN}' not found in validation CSV."
        )

    if len(validation_dataframe) != len(datasets["validation"]):
        raise ValueError(
            "Validation CSV and tokenized validation dataset "
            "have different record counts."
        )

    model = initialize_model()

    checkpoint = load_best_checkpoint(model)

    labels, predictions, probabilities = (
        generate_validation_predictions(
            model=model,
            validation_loader=dataloaders["validation"]
        )
    )

    if len(labels) != len(validation_dataframe):
        raise ValueError(
            "Prediction count does not match validation CSV row count."
        )

    metrics = calculate_metrics(
        labels=labels,
        predictions=predictions,
        probabilities=probabilities
    )

    metrics["checkpoint_epoch"] = int(
        checkpoint["epoch"]
    )

    output_dir = save_analysis_artifacts(
        validation_dataframe=validation_dataframe,
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        metrics=metrics
    )

    print_results(metrics)

    print(f"\nArtifacts directory: {output_dir}")


if __name__ == "__main__":
    main()
