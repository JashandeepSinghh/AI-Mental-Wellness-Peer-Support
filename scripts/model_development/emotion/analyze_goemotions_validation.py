
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.goemotions_config import (
    DEVICE,
    MAX_LENGTH,
    BATCH_SIZE,
    USE_FP16,
    LABEL_COLUMNS,
    VALIDATION_FILE,
    MODEL_DIR,
    OUTPUT_DIR,
    RANDOM_SEED,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

CHECKPOINT_DIR = MODEL_DIR / "best_checkpoint"

REPORT_DIR = OUTPUT_DIR / "validation_analysis"

MINIMUM_POSITIVE_SUPPORT = 20

THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95

INFERENCE_BATCH_SIZE = BATCH_SIZE


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. VALIDATION DATASET
# ============================================================

class GoEmotionsValidationDataset(Dataset):

    def __init__(self, csv_path, tokenizer):

        print("\nLoading validation dataset...")

        self.data = pd.read_csv(csv_path)

        required_columns = ["text"] + LABEL_COLUMNS

        missing_columns = [
            column
            for column in required_columns
            if column not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        if self.data[required_columns].isnull().any().any():
            raise ValueError(
                "Validation data contains missing values."
            )

        self.texts = self.data["text"].astype(str).tolist()

        self.labels = self.data[LABEL_COLUMNS].to_numpy(
            dtype=np.float32
        )

        if not np.isin(self.labels, [0.0, 1.0]).all():
            raise ValueError(
                "Emotion labels must contain only 0 and 1."
            )

        self.encodings = []

        print("Tokenizing validation text...")

        chunk_size = 512

        for start in tqdm(
            range(0, len(self.texts), chunk_size),
            desc="Tokenization",
        ):

            batch_texts = self.texts[
                start:start + chunk_size
            ]

            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=False,
            )

            for index in range(len(batch_texts)):

                self.encodings.append({
                    "input_ids": encoded["input_ids"][index],
                    "attention_mask": encoded["attention_mask"][index],
                })

        if len(self.encodings) != len(self.labels):
            raise RuntimeError(
                "Tokenized records do not match label count."
            )

        print("Validation records:", len(self.labels))
        print("Emotion labels:", self.labels.shape[1])


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):

        return {
            "input_ids": self.encodings[index]["input_ids"],
            "attention_mask": self.encodings[index]["attention_mask"],
            "labels": self.labels[index],
        }


# ============================================================
# 4. DYNAMIC PADDING COLLATOR
# ============================================================

class ValidationCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):

        features = [
            {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
            }
            for item in batch
        ]

        padded = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        padded["labels"] = torch.tensor(
            np.stack([
                item["labels"]
                for item in batch
            ]),
            dtype=torch.float32,
        )

        return padded


# ============================================================
# 5. LOAD SAVED MODEL
# ============================================================

def load_model():

    if not CHECKPOINT_DIR.is_dir():
        raise FileNotFoundError(
            f"Best checkpoint not found: {CHECKPOINT_DIR}"
        )

    print("\nLoading saved best checkpoint...")

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_DIR
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT_DIR
    )

    if model.config.num_labels != len(LABEL_COLUMNS):
        raise ValueError(
            "Checkpoint output dimension does not match "
            "the 28 emotion labels."
        )

    model.to(DEVICE)
    model.eval()

    print("Checkpoint loaded successfully.")
    print("Device:", DEVICE)
    print("Output labels:", model.config.num_labels)

    return model, tokenizer


# ============================================================
# 6. GENERATE VALIDATION PREDICTIONS
# ============================================================

def generate_predictions(model, data_loader):

    all_probabilities = []
    all_targets = []

    use_amp = USE_FP16 and DEVICE.type == "cuda"

    print("\nGenerating validation predictions...")

    with torch.inference_mode():

        for batch in tqdm(
            data_loader,
            desc="Inference",
        ):

            input_ids = batch["input_ids"].to(
                DEVICE,
                non_blocking=(DEVICE.type == "cuda"),
            )

            attention_mask = batch["attention_mask"].to(
                DEVICE,
                non_blocking=(DEVICE.type == "cuda"),
            )

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            probabilities = torch.sigmoid(
                outputs.logits.float()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_targets.append(
                batch["labels"].numpy()
            )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    targets = np.concatenate(
        all_targets,
        axis=0,
    )

    if probabilities.shape != targets.shape:
        raise RuntimeError(
            "Prediction and target dimensions do not match."
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            "Non-finite prediction probabilities detected."
        )

    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise RuntimeError(
            "Predicted probabilities are outside [0, 1]."
        )

    return targets, probabilities


# ============================================================
# 7. GLOBAL METRICS
# ============================================================

def calculate_global_metrics(targets, probabilities, thresholds):

    predictions = (
        probabilities >= np.asarray(thresholds)
    ).astype(np.int32)

    return {
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),

        "micro_f1": float(
            f1_score(
                targets,
                predictions,
                average="micro",
                zero_division=0,
            )
        ),

        "macro_precision": float(
            precision_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),

        "macro_recall": float(
            recall_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),

        "no_prediction_rate": float(
            np.mean(
                np.sum(predictions, axis=1) == 0
            )
        ),

        "average_predictions_per_example": float(
            np.mean(
                np.sum(predictions, axis=1)
            )
        ),
    }


# ============================================================
# 8. PER-EMOTION THRESHOLD CALIBRATION
# ============================================================

def find_best_threshold(y_true, y_probability):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        y_probability,
    )

    if len(thresholds) == 0:
        return 0.5, 0.0

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    valid_indices = np.where(
        (thresholds >= THRESHOLD_MIN)
        & (thresholds <= THRESHOLD_MAX)
    )[0]

    if len(valid_indices) == 0:
        return 0.5, 0.0

    valid_f1 = f1_values[valid_indices]

    maximum_f1 = np.max(valid_f1)

    best_indices = valid_indices[
        np.isclose(valid_f1, maximum_f1)
    ]

    # Prefer the higher threshold when F1 is tied.
    best_index = best_indices[
        np.argmax(thresholds[best_indices])
    ]

    return (
        float(thresholds[best_index]),
        float(f1_values[best_index]),
    )


def calibrate_thresholds(targets, probabilities):

    rows = []
    calibrated_thresholds = []

    print("\nCalibrating per-emotion thresholds...")

    for index, emotion in enumerate(LABEL_COLUMNS):

        y_true = targets[:, index]
        y_probability = probabilities[:, index]

        support = int(y_true.sum())

        threshold_05 = 0.5

        predictions_05 = (
            y_probability >= threshold_05
        ).astype(np.int32)

        precision_05 = precision_score(
            y_true,
            predictions_05,
            zero_division=0,
        )

        recall_05 = recall_score(
            y_true,
            predictions_05,
            zero_division=0,
        )

        f1_05 = f1_score(
            y_true,
            predictions_05,
            zero_division=0,
        )

        if support >= MINIMUM_POSITIVE_SUPPORT:

            best_threshold, best_validation_f1 = (
                find_best_threshold(
                    y_true,
                    y_probability,
                )
            )

            status = "calibrated"

        else:

            best_threshold = 0.5
            best_validation_f1 = float(f1_05)

            status = "insufficient_support_default_0.5"

        calibrated_thresholds.append(best_threshold)

        calibrated_predictions = (
            y_probability >= best_threshold
        ).astype(np.int32)

        calibrated_precision = precision_score(
            y_true,
            calibrated_predictions,
            zero_division=0,
        )

        calibrated_recall = recall_score(
            y_true,
            calibrated_predictions,
            zero_division=0,
        )

        calibrated_f1 = f1_score(
            y_true,
            calibrated_predictions,
            zero_division=0,
        )

        rows.append({

            "emotion": emotion,

            "positive_support": support,

            "precision_at_0.5": float(precision_05),

            "recall_at_0.5": float(recall_05),

            "f1_at_0.5": float(f1_05),

            "selected_threshold": float(best_threshold),

            "precision_at_selected_threshold": float(
                calibrated_precision
            ),

            "recall_at_selected_threshold": float(
                calibrated_recall
            ),

            "f1_at_selected_threshold": float(
                calibrated_f1
            ),

            "best_threshold_validation_f1": float(
                best_validation_f1
            ),

            "calibration_status": status,
        })

    report = pd.DataFrame(rows)

    thresholds = np.asarray(
        calibrated_thresholds,
        dtype=np.float32,
    )

    return report, thresholds


# ============================================================
# 9. SAVE VALIDATION ANALYSIS
# ============================================================

def save_results(
    targets,
    probabilities,
    report,
    thresholds,
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save per-emotion metrics.
    report.to_csv(
        REPORT_DIR / "per_emotion_validation_metrics.csv",
        index=False,
    )

    # Save prediction probabilities and ground truth.
    np.savez_compressed(
        REPORT_DIR / "validation_predictions.npz",
        targets=targets,
        probabilities=probabilities,
        labels=np.asarray(LABEL_COLUMNS),
    )

    # Global metrics at default threshold.
    default_metrics = calculate_global_metrics(
        targets,
        probabilities,
        np.full(len(LABEL_COLUMNS), 0.5),
    )

    # Global metrics after threshold calibration.
    calibrated_metrics = calculate_global_metrics(
        targets,
        probabilities,
        thresholds,
    )

    summary = {
        "checkpoint": str(CHECKPOINT_DIR),
        "validation_records": int(len(targets)),
        "number_of_emotions": int(len(LABEL_COLUMNS)),
        "minimum_positive_support": MINIMUM_POSITIVE_SUPPORT,
        "threshold_bounds": [
            THRESHOLD_MIN,
            THRESHOLD_MAX,
        ],
        "default_threshold_metrics": default_metrics,
        "calibrated_threshold_metrics": calibrated_metrics,
        "thresholds": {
            emotion: float(threshold)
            for emotion, threshold in zip(
                LABEL_COLUMNS,
                thresholds,
            )
        },
    }

    with open(
        REPORT_DIR / "validation_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("VALIDATION ANALYSIS RESULTS")
    print("=" * 70)

    print("\nDEFAULT THRESHOLD: 0.5")

    for metric, value in default_metrics.items():
        print(f"{metric}: {value:.6f}")

    print("\nCALIBRATED PER-EMOTION THRESHOLDS")

    for metric, value in calibrated_metrics.items():
        print(f"{metric}: {value:.6f}")

    print("\nPER-EMOTION REPORT")
    print(
        report[
            [
                "emotion",
                "positive_support",
                "f1_at_0.5",
                "selected_threshold",
                "f1_at_selected_threshold",
                "calibration_status",
            ]
        ].to_string(index=False)
    )

    print("\nSaved files:")
    print(REPORT_DIR / "per_emotion_validation_metrics.csv")
    print(REPORT_DIR / "validation_predictions.npz")
    print(REPORT_DIR / "validation_summary.json")

    print("\nVALIDATION ANALYSIS COMPLETED")


# ============================================================
# 10. MAIN EXECUTION
# ============================================================

def main():

    set_seed(RANDOM_SEED)

    print("=" * 70)
    print("GOEMOTIONS VALIDATION ANALYSIS")
    print("=" * 70)

    model, tokenizer = load_model()

    dataset = GoEmotionsValidationDataset(
        VALIDATION_FILE,
        tokenizer,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=ValidationCollator(tokenizer),
    )

    print("Validation batches:", len(data_loader))

    targets, probabilities = generate_predictions(
        model,
        data_loader,
    )

    report, thresholds = calibrate_thresholds(
        targets,
        probabilities,
    )

    save_results(
        targets,
        probabilities,
        report,
        thresholds,
    )


if __name__ == "__main__":

    main()
