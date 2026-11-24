
from pathlib import Path
import torch


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT
    / "datasets"
    / "processed"
    / "tweeteval"
    / "sentiment"
    / "standardized"
)

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

LABEL_MAPPING_FILE = DATA_DIR / "label_mapping.json"


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

NUM_LABELS = 3

MAX_LENGTH = 160

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

LABEL2ID = {
    label: index
    for index, label in ID2LABEL.items()
}


# ============================================================
# 3. TRAINING HYPERPARAMETERS
# ============================================================

EPOCHS = 3

BATCH_SIZE = 16

GRADIENT_ACCUMULATION_STEPS = 2

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.10

MAX_GRAD_NORM = 1.0

RANDOM_SEED = 42

LOSS_FUNCTION = "CrossEntropyLoss"


# ============================================================
# 4. HARDWARE CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_FP16 = DEVICE.type == "cuda"


# ============================================================
# 5. OUTPUT DIRECTORIES
# ============================================================

MODEL_DIR = (
    ROOT
    / "models"
    / "sentiment"
    / "tweeteval_distilbert"
)

OUTPUT_DIR = (
    ROOT
    / "training"
    / "sentiment"
    / "tweeteval_distilbert"
)


# ============================================================
# 6. CONFIGURATION VALIDATION
# ============================================================

def validate_config():

    assert NUM_LABELS == len(ID2LABEL)
    assert NUM_LABELS == len(LABEL2ID)

    assert BATCH_SIZE > 0
    assert GRADIENT_ACCUMULATION_STEPS > 0
    assert EPOCHS > 0
    assert MAX_LENGTH > 0
    assert LEARNING_RATE > 0

    assert 0 <= WARMUP_RATIO < 1

    if set(ID2LABEL.keys()) != {0, 1, 2}:
        raise ValueError("Invalid sentiment label IDs.")

    if set(LABEL2ID.keys()) != {
        "negative",
        "neutral",
        "positive",
    }:
        raise ValueError("Invalid sentiment label mapping.")

    required_files = [
        TRAIN_FILE,
        VALIDATION_FILE,
        TEST_FILE,
        LABEL_MAPPING_FILE,
    ]

    for file_path in required_files:

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# 7. CONFIGURATION SUMMARY
# ============================================================

def print_config():

    print("=" * 65)
    print("TWEETEVAL SENTIMENT CONFIGURATION")
    print("=" * 65)

    print("Model:", MODEL_NAME)
    print("Number of classes:", NUM_LABELS)
    print("Maximum sequence length:", MAX_LENGTH)

    print("Batch size:", BATCH_SIZE)
    print(
        "Gradient accumulation:",
        GRADIENT_ACCUMULATION_STEPS,
    )

    print(
        "Effective batch size:",
        BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
    )

    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Loss function:", LOSS_FUNCTION)

    print("Device:", DEVICE)
    print("FP16:", USE_FP16)

    print("\nLabel mapping:")

    for index, label in ID2LABEL.items():
        print(f"{index}: {label}")

    print("\nDataset files verified.")
    print("Output directories verified.")

    print("\nCONFIGURATION READY")
    print("=" * 65)


# ============================================================
# 8. EXECUTION
# ============================================================

if __name__ == "__main__":

    validate_config()
    print_config()
