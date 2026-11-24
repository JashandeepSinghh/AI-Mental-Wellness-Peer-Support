
from pathlib import Path
import torch


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "datasets" / "processed" / "goemotions" / "cleaned"

TRAIN_FILE = DATA_DIR / "train_cleaned.csv"
VALIDATION_FILE = DATA_DIR / "validation_cleaned.csv"
TEST_FILE = DATA_DIR / "test_cleaned.csv"


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

NUM_LABELS = 28

MAX_LENGTH = 256


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

LOSS_FUNCTION = "BCEWithLogitsLoss"


# ============================================================
# 4. HARDWARE CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_FP16 = DEVICE.type == "cuda"


# ============================================================
# 5. EMOTION LABELS
# ============================================================

LABEL_COLUMNS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


# ============================================================
# 6. OUTPUT DIRECTORIES
# ============================================================

MODEL_DIR = (
    ROOT / "models" / "emotion" / "goemotions_distilbert"
)

OUTPUT_DIR = (
    ROOT / "training" / "emotion" / "goemotions_distilbert"
)


# ============================================================
# 7. CONFIGURATION VALIDATION
# ============================================================

def validate_config():

    assert NUM_LABELS == len(LABEL_COLUMNS), (
        "Number of labels does not match label columns."
    )

    assert BATCH_SIZE > 0
    assert GRADIENT_ACCUMULATION_STEPS > 0
    assert EPOCHS > 0
    assert MAX_LENGTH > 0
    assert LEARNING_RATE > 0

    assert 0 <= WARMUP_RATIO < 1

    for file_path in [
        TRAIN_FILE,
        VALIDATION_FILE,
        TEST_FILE,
    ]:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required dataset file not found: {file_path}"
            )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 8. CONFIGURATION SUMMARY
# ============================================================

def print_config():

    print("=" * 65)
    print("GOEMOTIONS CONFIGURATION")
    print("=" * 65)

    print("Model:", MODEL_NAME)
    print("Number of labels:", NUM_LABELS)
    print("Maximum sequence length:", MAX_LENGTH)
    print("Batch size:", BATCH_SIZE)
    print("Gradient accumulation:", GRADIENT_ACCUMULATION_STEPS)
    print(
        "Effective batch size:",
        BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
    )
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Device:", DEVICE)
    print("FP16:", USE_FP16)

    print("\nDataset files verified.")
    print("Output directories verified.")

    print("\nCONFIGURATION READY")
    print("=" * 65)


# ============================================================  
# 9. EXECUTION
# ============================================================

if __name__ == "__main__":

    validate_config()
    print_config()
