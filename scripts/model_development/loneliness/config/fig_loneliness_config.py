
"""
FIG-Loneliness DistilBERT Configuration
---------------------------------------
Purpose:
    Central configuration for the Social Isolation / Loneliness
    Detection module.

Model:
    DistilBERT binary sequence classification.

Labels:
    0 = Non-lonely
    1 = Lonely

Important:
    This model identifies expressed loneliness in text.
    It does not diagnose loneliness or objectively measure
    a person's social isolation.
"""

from pathlib import Path
import torch


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Project root: 4AiMentalWellnessPeerSupport
PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "fig_loneliness"
    / "standardized"
)

TRAIN_FILE = DATASET_DIR / "train.csv"
VALIDATION_FILE = DATASET_DIR / "validation.csv"
TEST_FILE = DATASET_DIR / "test.csv"

MODEL_OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "loneliness"
    / "fig_loneliness_distilbert"
)

CHECKPOINT_DIR = MODEL_OUTPUT_DIR / "best_checkpoint"

TRAINING_OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "loneliness"
    / "fig_loneliness_distilbert"
)

TENSORBOARD_DIR = TRAINING_OUTPUT_DIR / "tensorboard"


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

NUM_LABELS = 2

LABEL2ID = {
    "non_lonely": 0,
    "lonely": 1,
}

ID2LABEL = {
    0: "non_lonely",
    1: "lonely",
}

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

MAX_LENGTH = 384


# ============================================================
# 3. TOKENIZER CONFIGURATION
# ============================================================

TOKENIZER_PADDING = "longest"
TOKENIZER_TRUNCATION = True

# Dynamic padding will pad each batch to its longest sequence.
# Sequences longer than MAX_LENGTH will be truncated.


# ============================================================
# 4. TRAINING CONFIGURATION
# ============================================================

NUM_EPOCHS = 5

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

GRADIENT_ACCUMULATION_STEPS = 4

# Effective training batch size:
# 8 physical batch size * 4 accumulation steps = 32

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.10

MAX_GRAD_NORM = 1.0

EARLY_STOPPING_PATIENCE = 2

MINIMUM_IMPROVEMENT = 0.0001


# ============================================================
# 5. OPTIMIZATION CONFIGURATION
# ============================================================

USE_FP16 = True

USE_BF16 = False

PIN_MEMORY = torch.cuda.is_available()

NUM_WORKERS = 0

GRADIENT_CHECKPOINTING = False


# ============================================================
# 6. REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42

DETERMINISTIC = False


# ============================================================
# 7. EVALUATION CONFIGURATION
# ============================================================

PRIMARY_METRIC = "macro_f1"

METRICS = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
]

# Standard binary classification decision threshold.
CLASSIFICATION_THRESHOLD = 0.50

# No threshold optimization will be performed on the official
# test set.


# ============================================================
# 8. DEVICE CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_MIXED_PRECISION = (
    USE_FP16 and DEVICE.type == "cuda"
)


# ============================================================
# 9. DATASET EXPECTATIONS
# ============================================================

EXPECTED_SPLIT_SIZES = {
    "train": 3943,
    "validation": 1126,
    "test": 564,
}

EXPECTED_LABEL_DISTRIBUTION = {
    "train": {
        0: 2103,
        1: 1840,
    },
    "validation": {
        0: 590,
        1: 536,
    },
    "test": {
        0: 307,
        1: 257,
    },
}


# ============================================================
# 10. CONFIGURATION VALIDATION
# ============================================================

def validate_config():
    """
    Validate the configuration before model development.
    """

    errors = []

    if NUM_LABELS != 2:
        errors.append("NUM_LABELS must be 2.")

    if LABEL2ID["non_lonely"] != 0:
        errors.append("Non-lonely label must be 0.")

    if LABEL2ID["lonely"] != 1:
        errors.append("Lonely label must be 1.")

    if MAX_LENGTH <= 0:
        errors.append("MAX_LENGTH must be positive.")

    if TRAIN_BATCH_SIZE <= 0:
        errors.append("TRAIN_BATCH_SIZE must be positive.")

    if EVAL_BATCH_SIZE <= 0:
        errors.append("EVAL_BATCH_SIZE must be positive.")

    if GRADIENT_ACCUMULATION_STEPS <= 0:
        errors.append(
            "GRADIENT_ACCUMULATION_STEPS must be positive."
        )

    if NUM_EPOCHS <= 0:
        errors.append("NUM_EPOCHS must be positive.")

    if LEARNING_RATE <= 0:
        errors.append("LEARNING_RATE must be positive.")

    if not 0 <= WARMUP_RATIO < 1:
        errors.append("WARMUP_RATIO must be between 0 and 1.")

    if EARLY_STOPPING_PATIENCE < 1:
        errors.append(
            "EARLY_STOPPING_PATIENCE must be at least 1."
        )

    if USE_FP16 and USE_BF16:
        errors.append(
            "FP16 and BF16 cannot both be enabled."
        )

    if not 0 < CLASSIFICATION_THRESHOLD < 1:
        errors.append(
            "CLASSIFICATION_THRESHOLD must be between 0 and 1."
        )

    if errors:
        raise ValueError(
            "Configuration validation failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )

    return True


# ============================================================
# 11. CONFIGURATION SUMMARY
# ============================================================

def print_config_summary():

    print("=" * 65)
    print("FIG-LONELINESS DISTILBERT CONFIGURATION")
    print("=" * 65)

    print(f"Project root          : {PROJECT_ROOT}")
    print(f"Model                 : {MODEL_NAME}")
    print(f"Number of classes     : {NUM_LABELS}")
    print(f"Maximum sequence      : {MAX_LENGTH}")

    print("-" * 65)

    print(f"Training batch size   : {TRAIN_BATCH_SIZE}")
    print(f"Evaluation batch size : {EVAL_BATCH_SIZE}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")

    print(
        "Effective batch size : "
        f"{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )

    print(f"Learning rate         : {LEARNING_RATE}")
    print(f"Epochs                : {NUM_EPOCHS}")
    print(f"Weight decay          : {WEIGHT_DECAY}")
    print(f"Warmup ratio          : {WARMUP_RATIO}")

    print("-" * 65)

    print(f"Device                : {DEVICE}")
    print(f"CUDA available        : {torch.cuda.is_available()}")
    print(f"FP16 enabled          : {USE_MIXED_PRECISION}")

    if torch.cuda.is_available():
        print(
            f"GPU                   : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("-" * 65)

    print(f"Primary metric        : {PRIMARY_METRIC}")
    print(f"Classification cutoff : {CLASSIFICATION_THRESHOLD}")

    print("=" * 65)


# ============================================================
# 12. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    validate_config()

    print("\nConfiguration validation: PASSED")

    print_config_summary()
