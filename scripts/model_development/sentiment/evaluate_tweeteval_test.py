
import json

import numpy as np
import pandas as pd
import torch

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
    TEST_FILE,
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
# 1. CONFIGURATION
# ============================================================

CHECKPOINT_DIR = MODEL_DIR / "best_checkpoint"

EVALUATION_DIR = (
    OUTPUT_DIR / "final_test_evaluation"
)

CLASS_IDS = list(range(NUM_LABELS))

CLASS_NAMES = [
    ID2LABEL[index]
    for index in CLASS_IDS
]


# ============================================================
# 2. INPUT VALIDATION
# ============================================================

def validate_inputs():

    if not CHECKPOINT_DIR.is_dir():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_DIR}"
        )

    if not TEST_FILE.is_file():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_FILE}"
        )

    metadata_path = (
        CHECKPOINT_DIR / "training_metadata.json"
    )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Checkpoint training metadata is missing."
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    if metadata.get("best_metric") != "validation_macro_f1":
        raise ValueError(
            "Checkpoint selection metric is not validation Macro-F1."
        )

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Best checkpoint verified.")
    print("Official test dataset verified.")
    print("Checkpoint epoch:", metadata["epoch"])
    print("Output directory:", EVALUATION_DIR)


# ============================================================
# 3. LOAD TEST DATA
# ============================================================

def load_test_data():

    print("\nLoading checkpoint tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_DIR
    )

    print("Tokenizer loaded.")

    print("\nLoading official test dataset...")

    test_dataset = TweetEvalDataset(
        TEST_FILE,
        tokenizer,
    )

    collator = TweetEvalCollator(
        tokenizer
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=collator,
    )

    print("Test records:", len(test_dataset))
    print("Test batches:", len(test_loader))

    return test_dataset, test_loader


# ============================================================
# 4. LOAD TRAINED CHECKPOINT
# ============================================================

def load_model():

    print("\nLoading trained DistilBERT checkpoint...")

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT_DIR
    )

    if model.config.num_labels != NUM_LABELS:
        raise ValueError(
            "Checkpoint class count mismatch."
        )

    model = model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")
    print("Device:", DEVICE)

    return model


# ============================================================
# 5. GENERATE TEST PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    test_dataset,
    test_loader,
):

    all_true = []
    all_predicted = []
    all_probabilities = []

    print("\nGenerating official test predictions...")

    with torch.inference_mode():

        for batch in tqdm(
            test_loader,
            desc="Test inference",
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

            all_true.extend(
                labels.cpu().tolist()
            )

            all_predicted.extend(
                predictions.cpu().tolist()
            )

            all_probabilities.extend(
                probabilities.cpu().tolist()
            )

    true_labels = np.asarray(
        all_true,
        dtype=np.int64,
    )

    predicted_labels = np.asarray(
        all_predicted,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        all_probabilities,
        dtype=np.float32,
    )

    expected = len(test_dataset)

    if not (
        len(true_labels)
        == len(predicted_labels)
        == len(probabilities)
        == expected
    ):
        raise RuntimeError(
            "Test prediction count mismatch."
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
            "Probability rows do not sum to 1."
        )

    return (
        true_labels,
        predicted_labels,
        probabilities,
    )


# ============================================================
# 6. OVERALL METRICS
# ============================================================

def calculate_metrics(
    true_labels,
    predicted_labels,
    probabilities,
):

    confidence = probabilities.max(
        axis=1
    )

    return {
        "test_records": int(len(true_labels)),

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
            confidence.mean()
        ),

        "median_prediction_confidence": float(
            np.median(confidence)
        ),
    }


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

        result = report[class_name]

        rows.append({
            "label_id": class_id,
            "sentiment": class_name,
            "precision": result["precision"],
            "recall": result["recall"],
            "f1_score": result["f1-score"],
            "support": int(
                np.sum(true_labels == class_id)
            ),
            "predicted_count": int(
                np.sum(predicted_labels == class_id)
            ),
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

    return pd.DataFrame(
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


# ============================================================
# 9. CLASS DISTRIBUTION
# ============================================================

def calculate_distribution(
    true_labels,
    predicted_labels,
):

    total = len(true_labels)
    rows = []

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
            "actual_percentage": actual_count / total * 100,
            "predicted_count": predicted_count,
            "predicted_percentage": predicted_count / total * 100,
        })

    return pd.DataFrame(rows)


# ============================================================
# 10. SAVE INDIVIDUAL PREDICTIONS
# ============================================================

def save_predictions(
    test_dataset,
    true_labels,
    predicted_labels,
    probabilities,
):

    rows = []

    for index, text in enumerate(
        test_dataset.texts
    ):

        true_class = int(true_labels[index])
        predicted_class = int(predicted_labels[index])

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

    predictions_df.to_csv(
        EVALUATION_DIR / "test_predictions.csv",
        index=False,
    )


# ============================================================
# 11. SAVE EVALUATION RESULTS
# ============================================================

def save_results(
    metrics,
    class_metrics,
    confusion_df,
    distribution_df,
    true_labels,
    predicted_labels,
    probabilities,
):

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

    class_metrics.to_csv(
        EVALUATION_DIR / "per_class_test_metrics.csv",
        index=False,
    )

    confusion_df.to_csv(
        EVALUATION_DIR / "test_confusion_matrix.csv"
    )

    distribution_df.to_csv(
        EVALUATION_DIR / "test_class_distribution.csv",
        index=False,
    )

    np.savez_compressed(
        EVALUATION_DIR / "test_predictions.npz",
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        probabilities=probabilities,
    )

    print("\nEvaluation files saved.")


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
    print("FINAL TWEETEVAL TEST EVALUATION")
    print("=" * 70)

    print("\nOVERALL METRICS")

    for name, value in metrics.items():

        if isinstance(value, float):
            print(f"{name}: {value:.6f}")

        else:
            print(f"{name}: {value}")

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
# 13. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL TWEETEVAL TEST EVALUATION")
    print("=" * 70)

    validate_inputs()

    test_dataset, test_loader = load_test_data()

    model = load_model()

    (
        true_labels,
        predicted_labels,
        probabilities,
    ) = generate_predictions(
        model,
        test_dataset,
        test_loader,
    )

    metrics = calculate_metrics(
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

    distribution_df = calculate_distribution(
        true_labels,
        predicted_labels,
    )

    save_predictions(
        test_dataset,
        true_labels,
        predicted_labels,
        probabilities,
    )

    save_results(
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

    print("\nFINAL TEST EVALUATION COMPLETED")
    print("Results directory:", EVALUATION_DIR)


if __name__ == "__main__":

    main()
