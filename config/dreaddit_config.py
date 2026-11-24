
from pathlib import Path
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "datasets" / "processed" / "dreaddit" / "final_splits"

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

NUM_LABELS = 2

ID2LABEL = {
    0: "non_stress",
    1: "stress",
}

LABEL2ID = {
    value: key for key, value in ID2LABEL.items()
}

MAX_LENGTH = 256


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

EPOCHS = 5

BATCH_SIZE = 16

GRADIENT_ACCUMULATION_STEPS = 2

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.10

MAX_GRAD_NORM = 1.0

EARLY_STOPPING_PATIENCE = 2

RANDOM_SEED = 42


# ============================================================
# CLASSIFICATION CONFIGURATION
# ============================================================

LOSS_FUNCTION = "CrossEntropyLoss"

NUM_WORKERS = 0

PIN_MEMORY = torch.cuda.is_available()


# ============================================================
# HARDWARE CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_FP16 = DEVICE.type == "cuda"


# ============================================================
# OUTPUT PATHS
# ============================================================

MODEL_DIR = ROOT / "models" / "stress" / "dreaddit_distilbert"

OUTPUT_DIR = ROOT / "training" / "stress" / "dreaddit_distilbert"

CHECKPOINT_DIR = MODEL_DIR / "best_checkpoint"

TENSORBOARD_DIR = OUTPUT_DIR / "tensorboard"

HISTORY_FILE = OUTPUT_DIR / "training_history.json"


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_config():

    required_files = [
        TRAIN_FILE,
        VALIDATION_FILE,
        TEST_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required dataset file not found: {file_path}"
            )

    if NUM_LABELS != 2:
        raise ValueError("Dreaddit requires exactly 2 classes.")

    if MAX_LENGTH <= 0:
        raise ValueError("MAX_LENGTH must be positive.")

    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be positive.")

    if GRADIENT_ACCUMULATION_STEPS <= 0:
        raise ValueError(
            "Gradient accumulation steps must be positive."
        )

    if EPOCHS <= 0:
        raise ValueError("EPOCHS must be positive.")

    if not 0 < WARMUP_RATIO < 1:
        raise ValueError("WARMUP_RATIO must be between 0 and 1.")

    if set(ID2LABEL.keys()) != {0, 1}:
        raise ValueError("Invalid ID2LABEL mapping.")

    if LABEL2ID != {
        "non_stress": 0,
        "stress": 1,
    }:
        raise ValueError("Invalid LABEL2ID mapping.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

    return True


# ============================================================
# CONFIGURATION SUMMARY
# ============================================================

def print_config():

    print("=" * 60)
    print("DREADDIT STRESS DETECTION CONFIGURATION")
    print("=" * 60)

    print(f"Model: {MODEL_NAME}")
    print(f"Classes: {NUM_LABELS}")
    print(f"Label mapping: {ID2LABEL}")
    print(f"Maximum sequence length: {MAX_LENGTH}")

    print(f"\nEpochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(
        f"Gradient accumulation: "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    effective_batch = (
        BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    )

    print(f"Effective batch size: {effective_batch}")

    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Warmup ratio: {WARMUP_RATIO}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")

    print(f"\nDevice: {DEVICE}")
    print(f"FP16 enabled: {USE_FP16}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nModel directory: {MODEL_DIR}")
    print(f"Training output directory: {OUTPUT_DIR}")

    print("\nConfiguration validation: PASSED")


if __name__ == "__main__":

    validate_config()
    print_config()



