
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

MODEL_DIR = (
    ROOT
    / "models"
    / "sentiment"
    / "tweeteval_bilstm"
)

OUTPUT_DIR = (
    ROOT
    / "training"
    / "sentiment"
    / "tweeteval_bilstm"
)


# ============================================================
# 2. CLASS CONFIGURATION
# ============================================================

NUM_CLASSES = 3

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

LABEL2ID = {
    value: key
    for key, value in ID2LABEL.items()
}


# ============================================================
# 3. TEXT PREPROCESSING
# ============================================================

MAX_VOCAB_SIZE = 30000
MIN_WORD_FREQUENCY = 2

MAX_SEQUENCE_LENGTH = 160

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

PAD_INDEX = 0
UNK_INDEX = 1


# ============================================================
# 4. MODEL ARCHITECTURE
# ============================================================

EMBEDDING_DIM = 200

HIDDEN_DIM = 128

NUM_LAYERS = 2

DROPOUT = 0.30

BIDIRECTIONAL = True


# ============================================================
# 5. TRAINING CONFIGURATION
# ============================================================

EPOCHS = 10

BATCH_SIZE = 64

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

MAX_GRAD_NORM = 1.0

EARLY_STOPPING_PATIENCE = 3

RANDOM_SEED = 42


# ============================================================
# 6. DEVICE CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 7. OUTPUT CONFIGURATION
# ============================================================

CHECKPOINT_NAME = "best_checkpoint.pt"

VOCABULARY_NAME = "vocabulary.json"

HISTORY_NAME = "training_history.json"


# ============================================================
# 8. CONFIGURATION VALIDATION
# ============================================================

def validate_config():

    if NUM_CLASSES != 3:
        raise ValueError(
            "TweetEval requires exactly 3 sentiment classes."
        )

    if set(LABEL2ID.values()) != set(range(NUM_CLASSES)):
        raise ValueError(
            "Invalid sentiment label mapping."
        )

    if not 0 < MIN_WORD_FREQUENCY:
        raise ValueError(
            "Minimum word frequency must be positive."
        )

    if MAX_VOCAB_SIZE <= 2:
        raise ValueError(
            "Vocabulary size must exceed reserved tokens."
        )

    if MAX_SEQUENCE_LENGTH <= 0:
        raise ValueError(
            "Maximum sequence length must be positive."
        )

    if EMBEDDING_DIM <= 0 or HIDDEN_DIM <= 0:
        raise ValueError(
            "Embedding and hidden dimensions must be positive."
        )

    if NUM_LAYERS <= 0:
        raise ValueError(
            "Number of LSTM layers must be positive."
        )

    if not 0 <= DROPOUT < 1:
        raise ValueError(
            "Dropout must be between 0 and 1."
        )

    if EPOCHS <= 0 or BATCH_SIZE <= 0:
        raise ValueError(
            "Epochs and batch size must be positive."
        )

    if LEARNING_RATE <= 0:
        raise ValueError(
            "Learning rate must be positive."
        )

    for path in [
        TRAIN_FILE,
        VALIDATION_FILE,
        TEST_FILE,
    ]:

        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset file not found: {path}"
            )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return True


# ============================================================
# 9. CONFIGURATION SUMMARY
# ============================================================

def print_config():

    print("=" * 65)
    print("TWEETEVAL BILSTM CONFIGURATION")
    print("=" * 65)

    print("Architecture: Bidirectional LSTM")
    print("Number of classes:", NUM_CLASSES)

    print("Vocabulary size limit:", MAX_VOCAB_SIZE)
    print("Minimum word frequency:", MIN_WORD_FREQUENCY)
    print("Maximum sequence length:", MAX_SEQUENCE_LENGTH)

    print("Embedding dimension:", EMBEDDING_DIM)
    print("Hidden dimension:", HIDDEN_DIM)
    print("LSTM layers:", NUM_LAYERS)
    print("Bidirectional:", BIDIRECTIONAL)
    print("Dropout:", DROPOUT)

    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)

    print("Optimizer: AdamW")
    print("Loss function: CrossEntropyLoss")

    print("Device:", DEVICE)

    print("\nLabel mapping:")

    for label_id, label_name in ID2LABEL.items():
        print(f"{label_id}: {label_name}")

    print("\nDataset files verified.")
    print("Output directories verified.")

    print("\nCONFIGURATION READY")
    print("=" * 65)


# ============================================================
# 10. EXECUTION
# ============================================================

if __name__ == "__main__":

    validate_config()

    print_config()
