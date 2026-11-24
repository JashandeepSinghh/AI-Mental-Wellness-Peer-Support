
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
)

from config.dreaddit_config import (
    DEVICE,
    USE_FP16,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    VALIDATION_FILE,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    ID2LABEL,
)

from scripts.model_development.stress.dreaddit_dataset import (
    DreadditDataset,
    DreadditCollator,
)

from scripts.model_development.stress.dreaddit_model import (
    DreadditStressModel,
)


# ============================================================
# OUTPUT CONFIGURATION
# ============================================================

ANALYSIS_DIR = (
    OUTPUT_DIR / "validation_analysis"
)

LABEL_NAMES = [
    ID2LABEL[0],
    ID2LABEL[1],
]


# ============================================================
# JSON SAVING
# ============================================================

def save_json(data, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

def load_best_model():

    checkpoint_path = (
        CHECKPOINT_DIR / "best_checkpoint.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {checkpoint_path}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_DIR
    )

    model = DreadditStressModel()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(
        "Best validation Macro-F1: "
        f"{checkpoint['validation_metrics']['macro_f1']:.6f}"
    )

    return model, tokenizer, checkpoint


# ============================================================
# VALIDATION INFERENCE
# ============================================================

def run_validation_inference(model, tokenizer):

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_FILE}"
        )

    dataset = DreadditDataset(
        VALIDATION_FILE,
        tokenizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=DreadditCollator(tokenizer),
    )

    all_labels = []
    all_predictions = []
    all_probabilities = []

    model.eval()

    with torch.inference_mode():

        for batch_index, batch in enumerate(loader):

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
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_FP16,
            ):

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            probabilities = torch.softmax(
                outputs.logits.float(),
                dim=1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

            print(
                f"Inference batch "
                f"{batch_index + 1}/{len(loader)} completed"
            )

    return (
        np.asarray(all_labels, dtype=np.int64),
        np.asarray(all_predictions, dtype=np.int64),
        np.asarray(all_probabilities, dtype=np.float32),
        dataset.texts,
    )


# ============================================================
# METRIC ANALYSIS
# ============================================================

def calculate_analysis(
    actual,
    predicted,
    probabilities,
):

    accuracy = accuracy_score(
        actual,
        predicted,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            actual,
            predicted,
            labels=[0, 1],
            zero_division=0,
        )
    )

    weighted_f1 = (
        precision_recall_fscore_support(
            actual,
            predicted,
            average="weighted",
            zero_division=0,
        )[2]
    )

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    )

    confidence = probabilities.max(axis=1)

    per_class = []

    for class_id in [0, 1]:

        per_class.append({
            "class": LABEL_NAMES[class_id],
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
            "support": int(support[class_id]),
            "predicted_count": int(
                np.sum(predicted == class_id)
            ),
        })

    summary = {
        "records": int(len(actual)),
        "accuracy": float(accuracy),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "non_stress_f1": float(f1[0]),
        "stress_f1": float(f1[1]),
        "confusion_matrix": matrix.tolist(),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "true_negatives": int(matrix[0, 0]),
        "true_positives": int(matrix[1, 1]),
        "mean_confidence": float(np.mean(confidence)),
        "median_confidence": float(np.median(confidence)),
        "confidence_p25": float(
            np.percentile(confidence, 25)
        ),
        "confidence_p75": float(
            np.percentile(confidence, 75)
        ),
        "confidence_p95": float(
            np.percentile(confidence, 95)
        ),
        "per_class": per_class,
    }

    return summary, matrix, per_class, confidence


# ============================================================
# SAVE ANALYSIS ARTIFACTS
# ============================================================

def save_analysis_artifacts(
    actual,
    predicted,
    probabilities,
    texts,
    summary,
    matrix,
    per_class,
    confidence,
):

    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        summary,
        ANALYSIS_DIR / "validation_summary.json",
    )

    per_class_df = pd.DataFrame(per_class)

    per_class_df.to_csv(
        ANALYSIS_DIR / "per_class_validation_metrics.csv",
        index=False,
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=LABEL_NAMES,
        columns=LABEL_NAMES,
    )

    confusion_df.to_csv(
        ANALYSIS_DIR / "validation_confusion_matrix.csv"
    )

    predictions_df = pd.DataFrame({
        "text": texts,
        "actual_label": actual,
        "actual_class": [
            LABEL_NAMES[value] for value in actual
        ],
        "predicted_label": predicted,
        "predicted_class": [
            LABEL_NAMES[value] for value in predicted
        ],
        "probability_non_stress": probabilities[:, 0],
        "probability_stress": probabilities[:, 1],
        "confidence": confidence,
        "correct_prediction": actual == predicted,
    })

    predictions_df.to_csv(
        ANALYSIS_DIR / "validation_predictions.csv",
        index=False,
    )

    predictions_df[
        predictions_df["correct_prediction"] == False
    ].to_csv(
        ANALYSIS_DIR / "validation_misclassified_examples.csv",
        index=False,
    )

    np.savez_compressed(
        ANALYSIS_DIR / "validation_predictions.npz",
        actual=actual,
        predicted=predicted,
        probabilities=probabilities,
        confidence=confidence,
    )

    print("\nAnalysis artifacts saved:")
    print(ANALYSIS_DIR)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DREADDIT VALIDATION ANALYSIS")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print("Evaluation split: VALIDATION ONLY")

    model, tokenizer, checkpoint = load_best_model()

    (
        actual,
        predicted,
        probabilities,
        texts,
    ) = run_validation_inference(
        model,
        tokenizer,
    )

    if len(actual) != len(texts):
        raise RuntimeError(
            "Prediction and text counts do not match."
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
            "Probability normalization failed."
        )

    summary, matrix, per_class, confidence = (
        calculate_analysis(
            actual,
            predicted,
            probabilities,
        )
    )

    save_analysis_artifacts(
        actual,
        predicted,
        probabilities,
        texts,
        summary,
        matrix,
        per_class,
        confidence,
    )

    print("\n" + "=" * 70)
    print("OVERALL VALIDATION METRICS")
    print("=" * 70)

    for key in [
        "records",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
        "non_stress_f1",
        "stress_f1",
    ]:
        print(f"{key}: {summary[key]}")

    print("\n" + "=" * 70)
    print("PER-CLASS METRICS")
    print("=" * 70)

    print(
        pd.DataFrame(per_class).to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        pd.DataFrame(
            matrix,
            index=LABEL_NAMES,
            columns=LABEL_NAMES,
        )
    )

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    print(
        f"False positives: {summary['false_positives']}"
    )

    print(
        f"False negatives: {summary['false_negatives']}"
    )

    print(
        f"Misclassified examples: "
        f"{int(np.sum(actual != predicted))}"
    )

    print("\n" + "=" * 70)
    print("CONFIDENCE DISTRIBUTION")
    print("=" * 70)

    print(
        f"Mean: {summary['mean_confidence']:.6f}"
    )

    print(
        f"Median: {summary['median_confidence']:.6f}"
    )

    print(
        f"25th percentile: {summary['confidence_p25']:.6f}"
    )

    print(
        f"75th percentile: {summary['confidence_p75']:.6f}"
    )

    print(
        f"95th percentile: {summary['confidence_p95']:.6f}"
    )

    print("\nVALIDATION ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
