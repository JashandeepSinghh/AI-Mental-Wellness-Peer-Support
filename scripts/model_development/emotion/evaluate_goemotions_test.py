
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from config.goemotions_config import (
    DEVICE,
    MAX_LENGTH,
    BATCH_SIZE,
    USE_FP16,
    LABEL_COLUMNS,
    TEST_FILE,
    MODEL_DIR,
    OUTPUT_DIR,
    RANDOM_SEED,
)


# ============================================================
# 1. PATHS AND EVALUATION SETTINGS
# ============================================================

CHECKPOINT_DIR = MODEL_DIR / "best_checkpoint"

VALIDATION_SUMMARY = (
    OUTPUT_DIR
    / "validation_analysis"
    / "validation_summary.json"
)

REPORT_DIR = OUTPUT_DIR / "final_test_evaluation"

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
# 3. LOAD AND VALIDATE FROZEN THRESHOLDS
# ============================================================

def load_frozen_thresholds():

    if not VALIDATION_SUMMARY.is_file():
        raise FileNotFoundError(
            f"Validation summary not found: {VALIDATION_SUMMARY}"
        )

    with open(
        VALIDATION_SUMMARY,
        "r",
        encoding="utf-8",
    ) as file:

        summary = json.load(file)

    threshold_mapping = summary.get("thresholds")

    if not isinstance(threshold_mapping, dict):
        raise ValueError(
            "Validation summary does not contain a valid "
            "threshold mapping."
        )

    if set(threshold_mapping.keys()) != set(LABEL_COLUMNS):
        raise ValueError(
            "Threshold labels do not match the configured "
            "28 emotion labels."
        )

    thresholds = np.asarray(
        [
            threshold_mapping[label]
            for label in LABEL_COLUMNS
        ],
        dtype=np.float32,
    )

    if not np.isfinite(thresholds).all():
        raise ValueError("Non-finite thresholds detected.")

    if not ((thresholds >= 0) & (thresholds <= 1)).all():
        raise ValueError("Thresholds must be within [0, 1].")

    print("Frozen validation thresholds loaded.")
    print("Number of thresholds:", len(thresholds))

    return thresholds


# ============================================================
# 4. LOAD MODEL AND TOKENIZER
# ============================================================

def load_model():

    if not CHECKPOINT_DIR.is_dir():
        raise FileNotFoundError(
            f"Best checkpoint not found: {CHECKPOINT_DIR}"
        )

    print("\nLoading best GoEmotions checkpoint...")

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_DIR
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT_DIR
    )

    if model.config.num_labels != len(LABEL_COLUMNS):
        raise ValueError(
            "Model output dimension does not match "
            "the configured emotion labels."
        )

    checkpoint_labels = model.config.id2label

    if checkpoint_labels:
        expected_labels = {
            index: label
            for index, label in enumerate(LABEL_COLUMNS)
        }

        actual_labels = {
            int(index): label
            for index, label in checkpoint_labels.items()
        }

        if actual_labels != expected_labels:
            raise ValueError(
                "Checkpoint label order does not match "
                "the configured label order."
            )

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")
    print("Device:", DEVICE)
    print("Output labels:", model.config.num_labels)

    return model, tokenizer


# ============================================================
# 5. TEST DATASET
# ============================================================

class GoEmotionsTestDataset(Dataset):

    def __init__(self, csv_path, tokenizer):

        print("\nLoading official test dataset...")

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
                "Test dataset contains missing values."
            )

        self.texts = self.data["text"].astype(str).tolist()

        self.labels = self.data[LABEL_COLUMNS].to_numpy(
            dtype=np.float32
        )

        if not np.isin(self.labels, [0.0, 1.0]).all():
            raise ValueError(
                "Test labels must contain only binary values."
            )

        self.encodings = []

        print("Tokenizing test text...")

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

        print("Test records:", len(self.labels))
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
# 6. DYNAMIC PADDING
# ============================================================

class TestCollator:
    __test__ = False

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
# 7. GENERATE TEST PREDICTIONS
# ============================================================

def generate_predictions(model, data_loader):

    all_probabilities = []
    all_targets = []

    use_amp = USE_FP16 and DEVICE.type == "cuda"

    print("\nGenerating final test predictions...")

    with torch.inference_mode():

        for batch in tqdm(
            data_loader,
            desc="Test inference",
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

    targets = np.concatenate(
        all_targets,
        axis=0,
    )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    if targets.shape != probabilities.shape:
        raise RuntimeError(
            "Test targets and predictions have different shapes."
        )

    if targets.shape[1] != len(LABEL_COLUMNS):
        raise RuntimeError(
            "Unexpected number of emotion outputs."
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            "Non-finite probabilities detected."
        )

    if not (
        (probabilities >= 0)
        & (probabilities <= 1)
    ).all():
        raise RuntimeError(
            "Probabilities outside [0, 1]."
        )

    return targets, probabilities


# ============================================================
# 8. CALCULATE MULTI-LABEL METRICS
# ============================================================

def calculate_metrics(targets, predictions):

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

        "micro_precision": float(
            precision_score(
                targets,
                predictions,
                average="micro",
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

        "micro_recall": float(
            recall_score(
                targets,
                predictions,
                average="micro",
                zero_division=0,
            )
        ),

        "hamming_loss": float(
            hamming_loss(
                targets,
                predictions,
            )
        ),

        "exact_match_accuracy": float(
            accuracy_score(
                targets,
                predictions,
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
# 9. PER-EMOTION METRICS
# ============================================================

def calculate_per_emotion_metrics(
    targets,
    predictions,
    threshold_name,
):

    rows = []

    confusion_matrices = multilabel_confusion_matrix(
        targets,
        predictions,
    )

    for index, emotion in enumerate(LABEL_COLUMNS):

        precision = precision_score(
            targets[:, index],
            predictions[:, index],
            zero_division=0,
        )

        recall = recall_score(
            targets[:, index],
            predictions[:, index],
            zero_division=0,
        )

        f1 = f1_score(
            targets[:, index],
            predictions[:, index],
            zero_division=0,
        )

        tn, fp, fn, tp = confusion_matrices[index].ravel()

        rows.append({

            "emotion": emotion,

            "threshold_type": threshold_name,

            "support": int(targets[:, index].sum()),

            "predicted_positive": int(
                predictions[:, index].sum()
            ),

            "true_positive": int(tp),

            "false_positive": int(fp),

            "false_negative": int(fn),

            "true_negative": int(tn),

            "precision": float(precision),

            "recall": float(recall),

            "f1": float(f1),
        })

    return pd.DataFrame(rows), confusion_matrices


# ============================================================
# 10. SAVE EVALUATION OUTPUTS
# ============================================================

def save_evaluation(
    targets,
    probabilities,
    frozen_thresholds,
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    default_thresholds = np.full(
        len(LABEL_COLUMNS),
        0.5,
        dtype=np.float32,
    )

    default_predictions = (
        probabilities >= default_thresholds
    ).astype(np.int32)

    calibrated_predictions = (
        probabilities >= frozen_thresholds
    ).astype(np.int32)

    default_metrics = calculate_metrics(
        targets,
        default_predictions,
    )

    calibrated_metrics = calculate_metrics(
        targets,
        calibrated_predictions,
    )

    default_report, default_confusion = (
        calculate_per_emotion_metrics(
            targets,
            default_predictions,
            "default_0.5",
        )
    )

    calibrated_report, calibrated_confusion = (
        calculate_per_emotion_metrics(
            targets,
            calibrated_predictions,
            "validation_calibrated",
        )
    )

    combined_report = pd.concat(
        [
            default_report,
            calibrated_report,
        ],
        ignore_index=True,
    )

    combined_report.to_csv(
        REPORT_DIR / "per_emotion_test_metrics.csv",
        index=False,
    )

    # Save multilabel confusion matrices.
    np.savez_compressed(
        REPORT_DIR / "test_confusion_matrices.npz",
        default_0_5=default_confusion,
        validation_calibrated=calibrated_confusion,
        labels=np.asarray(LABEL_COLUMNS),
    )

    # Save raw probabilities and predictions.
    np.savez_compressed(
        REPORT_DIR / "test_predictions.npz",
        targets=targets,
        probabilities=probabilities,
        default_predictions=default_predictions,
        calibrated_predictions=calibrated_predictions,
        thresholds=frozen_thresholds,
        labels=np.asarray(LABEL_COLUMNS),
    )

    summary = {

        "model": "distilbert-base-uncased",

        "checkpoint": str(CHECKPOINT_DIR),

        "evaluation_split": "official_test",

        "test_records": int(len(targets)),

        "number_of_emotions": int(len(LABEL_COLUMNS)),

        "thresholds_source": "validation_summary.json",

        "test_threshold_optimization": False,

        "default_threshold_metrics": default_metrics,

        "validation_calibrated_threshold_metrics": calibrated_metrics,

        "frozen_thresholds": {
            emotion: float(threshold)
            for emotion, threshold in zip(
                LABEL_COLUMNS,
                frozen_thresholds,
            )
        },
    }

    with open(
        REPORT_DIR / "final_test_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\n" + "=" * 75)
    print("FINAL GOEMOTIONS TEST EVALUATION")
    print("=" * 75)

    print("\nDEFAULT THRESHOLD: 0.5")

    for metric, value in default_metrics.items():
        print(f"{metric}: {value:.6f}")

    print("\nFROZEN VALIDATION-CALIBRATED THRESHOLDS")

    for metric, value in calibrated_metrics.items():
        print(f"{metric}: {value:.6f}")

    print("\nPER-EMOTION TEST REPORT")

    print(
        combined_report[
            [
                "emotion",
                "threshold_type",
                "support",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(index=False)
    )

    print("\nSaved evaluation files:")
    print(REPORT_DIR / "per_emotion_test_metrics.csv")
    print(REPORT_DIR / "test_confusion_matrices.npz")
    print(REPORT_DIR / "test_predictions.npz")
    print(REPORT_DIR / "final_test_summary.json")

    print("\nFINAL TEST EVALUATION COMPLETED")


# ============================================================
# 11. MAIN
# ============================================================

def main():

    set_seed(RANDOM_SEED)

    print("=" * 75)
    print("GOEMOTIONS FINAL TEST EVALUATION")
    print("=" * 75)

    frozen_thresholds = load_frozen_thresholds()

    model, tokenizer = load_model()

    dataset = GoEmotionsTestDataset(
        TEST_FILE,
        tokenizer,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=TestCollator(tokenizer),
    )

    print("\nTest batches:", len(data_loader))

    targets, probabilities = generate_predictions(
        model,
        data_loader,
    )

    save_evaluation(
        targets,
        probabilities,
        frozen_thresholds,
    )


if __name__ == "__main__":

    main()
