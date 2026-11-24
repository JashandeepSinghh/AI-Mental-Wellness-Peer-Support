
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
    TEST_FILE,
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

EVALUATION_DIR = (
    OUTPUT_DIR / "final_test_evaluation"
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
# LOAD FROZEN CHECKPOINT
# ============================================================

def load_model():

    checkpoint_path = (
        CHECKPOINT_DIR / "best_checkpoint.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
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
# TEST INFERENCE
# ============================================================

def run_test_inference(model, tokenizer):

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_FILE}"
        )

    dataset = DreadditDataset(
        TEST_FILE,
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

    actual_labels = []
    predicted_labels = []
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

            actual_labels.extend(
                labels.cpu().tolist()
            )

            predicted_labels.extend(
                predictions.cpu().tolist()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

            print(
                f"Test batch {batch_index + 1}/"
                f"{len(loader)} completed"
            )

    return (
        np.asarray(actual_labels, dtype=np.int64),
        np.asarray(predicted_labels, dtype=np.int64),
        np.asarray(all_probabilities, dtype=np.float32),
        dataset.texts,
    )


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(actual, predicted, probabilities):

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
        "dataset": "Dreaddit",
        "evaluation_split": "official_test",
        "records": int(len(actual)),
        "accuracy": float(accuracy),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "non_stress_f1": float(f1[0]),
        "stress_f1": float(f1[1]),
        "confusion_matrix": matrix.tolist(),
        "true_negatives": int(matrix[0, 0]),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
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
# SAVE EVALUATION ARTIFACTS
# ============================================================

def save_artifacts(
    actual,
    predicted,
    probabilities,
    texts,
    summary,
    matrix,
    per_class,
    confidence,
):

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        summary,
        EVALUATION_DIR / "final_test_summary.json",
    )

    pd.DataFrame(per_class).to_csv(
        EVALUATION_DIR / "per_class_test_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        matrix,
        index=LABEL_NAMES,
        columns=LABEL_NAMES,
    ).to_csv(
        EVALUATION_DIR / "test_confusion_matrix.csv"
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
        EVALUATION_DIR / "test_predictions.csv",
        index=False,
    )

    predictions_df[
        predictions_df["correct_prediction"] == False
    ].to_csv(
        EVALUATION_DIR / "test_misclassified_examples.csv",
        index=False,
    )

    np.savez_compressed(
        EVALUATION_DIR / "test_predictions.npz",
        actual=actual,
        predicted=predicted,
        probabilities=probabilities,
        confidence=confidence,
    )

    print(f"\nEvaluation artifacts saved: {EVALUATION_DIR}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DREADDIT FINAL TEST EVALUATION")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print("Evaluation split: OFFICIAL TEST")
    print("Threshold: Standard argmax prediction")

    model, tokenizer, checkpoint = load_model()

    actual, predicted, probabilities, texts = (
        run_test_inference(
            model,
            tokenizer,
        )
    )

    if len(actual) != len(texts):
        raise RuntimeError(
            "Test predictions and texts are misaligned."
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            "Non-finite prediction probabilities."
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
        calculate_metrics(
            actual,
            predicted,
            probabilities,
        )
    )

    save_artifacts(
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
    print("FINAL TEST METRICS")
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
        print(f"{key}: {summary[key]:.6f}" if isinstance(
            summary[key], float
        ) else f"{key}: {summary[key]}")

    print("\nPER-CLASS METRICS")

    print(
        pd.DataFrame(per_class).to_string(
            index=False
        )
    )

    print("\nCONFUSION MATRIX")

    print(
        pd.DataFrame(
            matrix,
            index=LABEL_NAMES,
            columns=LABEL_NAMES,
        )
    )

    print("\nERROR ANALYSIS")
    print(f"False positives: {summary['false_positives']}")
    print(f"False negatives: {summary['false_negatives']}")
    print(
        f"Misclassified records: "
        f"{int(np.sum(actual != predicted))}"
    )

    print("\nCONFIDENCE DISTRIBUTION")
    print(f"Mean: {summary['mean_confidence']:.6f}")
    print(f"Median: {summary['median_confidence']:.6f}")
    print(f"25th percentile: {summary['confidence_p25']:.6f}")
    print(f"75th percentile: {summary['confidence_p75']:.6f}")
    print(f"95th percentile: {summary['confidence_p95']:.6f}")

    print("\nFINAL TEST EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
