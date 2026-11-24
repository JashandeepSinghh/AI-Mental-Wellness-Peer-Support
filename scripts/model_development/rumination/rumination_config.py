
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

RUMINATION_OUTPUT_DIR = (
    ROOT / "training" / "rumination" / "rule_based"
)

RUMINATION_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MODULE INFORMATION
# ============================================================

MODULE_NAME = "Rumination Detection"

MODULE_VERSION = "1.2.0"

DETECTION_METHOD = (
    "Rule-Based Linguistic Pattern Analysis "
    "with Semantic Conversation Analysis"
)


# ============================================================
# RUMINATION CATEGORIES
# ============================================================

RUMINATION_CATEGORIES = {
    "repetitive_thoughts": "Repetitive Thought Patterns",

    "past_focused": "Past-Focused Rumination",

    "future_worry": "Future-Oriented Worry",

    "self_blame": "Self-Blaming Thought Patterns",

    "unresolved_loops": "Unresolved Thought Loops",
}


# ============================================================
# DETECTION SETTINGS
# ============================================================

MIN_TEXT_LENGTH = 3

MAX_TEXT_LENGTH = 20000

MIN_PATTERN_MATCHES = 1

ENABLE_CASE_INSENSITIVE_MATCHING = True

ENABLE_EVIDENCE_EXTRACTION = True


# ============================================================
# SCORING SETTINGS
# ============================================================

# These are heuristic indicator scores, not clinical probabilities.

CATEGORY_SCORE_CAP = 1.0

SCORE_PRECISION = 4


# ============================================================
# CONVERSATION SETTINGS
# ============================================================

ENABLE_CROSS_TURN_ANALYSIS = True

MIN_MESSAGES_FOR_CROSS_TURN_ANALYSIS = 2


# ============================================================
# SEMANTIC SIMILARITY SETTINGS
# ============================================================

# Semantic similarity identifies potentially related messages.
# It must NOT independently determine rumination.

ENABLE_SEMANTIC_ANALYSIS = True

SEMANTIC_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

SEMANTIC_EMBEDDING_DIMENSION = 384

SEMANTIC_MAX_SEQUENCE_LENGTH = 256

SEMANTIC_BATCH_SIZE = 16

# None = automatic CUDA/CPU selection.
# Explicit examples: "cuda", "cpu".
SEMANTIC_DEVICE = None


# ============================================================
# SEMANTIC ANALYSIS SAFEGUARDS
# ============================================================

# No fixed cosine-similarity threshold is configured.
# Similarity scores are contextual evidence, not probabilities.

SEMANTIC_SIMILARITY_IS_DIAGNOSTIC = False

SEMANTIC_REQUIRES_LINGUISTIC_CONTEXT = True


# ============================================================
# VALIDATION
# ============================================================

def validate_config():

    if len(RUMINATION_CATEGORIES) != 5:
        raise ValueError(
            "Exactly five rumination categories are required."
        )

    if MIN_TEXT_LENGTH <= 0:
        raise ValueError(
            "MIN_TEXT_LENGTH must be positive."
        )

    if MAX_TEXT_LENGTH < MIN_TEXT_LENGTH:
        raise ValueError(
            "MAX_TEXT_LENGTH must be greater than MIN_TEXT_LENGTH."
        )

    if CATEGORY_SCORE_CAP <= 0:
        raise ValueError(
            "CATEGORY_SCORE_CAP must be positive."
        )

    if SCORE_PRECISION < 0:
        raise ValueError(
            "SCORE_PRECISION cannot be negative."
        )

    if MIN_MESSAGES_FOR_CROSS_TURN_ANALYSIS < 2:
        raise ValueError(
            "Cross-turn analysis requires at least two messages."
        )

    if SEMANTIC_EMBEDDING_DIMENSION != 384:
        raise ValueError(
            "Unexpected semantic embedding dimension."
        )

    if SEMANTIC_MAX_SEQUENCE_LENGTH <= 0:
        raise ValueError(
            "Semantic maximum sequence length must be positive."
        )

    if SEMANTIC_BATCH_SIZE <= 0:
        raise ValueError(
            "Semantic batch size must be positive."
        )

    if not SEMANTIC_MODEL_NAME.strip():
        raise ValueError(
            "Semantic model name cannot be empty."
        )

    if SEMANTIC_DEVICE not in (None, "cuda", "cpu"):
        raise ValueError(
            "SEMANTIC_DEVICE must be None, 'cuda', or 'cpu'."
        )

    if SEMANTIC_SIMILARITY_IS_DIAGNOSTIC:
        raise ValueError(
            "Semantic similarity must never be treated as a diagnosis."
        )

    return True


# ============================================================
# MANUAL EXECUTION
# ============================================================

if __name__ == "__main__":

    validate_config()

    print("=" * 65)
    print("RUMINATION DETECTION CONFIGURATION")
    print("=" * 65)

    print(f"Module: {MODULE_NAME}")
    print(f"Version: {MODULE_VERSION}")
    print(f"Method: {DETECTION_METHOD}")

    print("\nDetection categories:")

    for key, value in RUMINATION_CATEGORIES.items():
        print(f"{key}: {value}")

    print("\nDetection settings:")
    print(f"Minimum text length: {MIN_TEXT_LENGTH}")
    print(f"Maximum text length: {MAX_TEXT_LENGTH}")
    print(f"Minimum pattern matches: {MIN_PATTERN_MATCHES}")

    print("\nConversation settings:")
    print(f"Cross-turn analysis: {ENABLE_CROSS_TURN_ANALYSIS}")
    print(
        "Minimum messages:",
        MIN_MESSAGES_FOR_CROSS_TURN_ANALYSIS
    )

    print("\nSemantic settings:")
    print(f"Enabled: {ENABLE_SEMANTIC_ANALYSIS}")
    print(f"Model: {SEMANTIC_MODEL_NAME}")
    print(f"Embedding dimension: {SEMANTIC_EMBEDDING_DIMENSION}")
    print(f"Maximum sequence length: {SEMANTIC_MAX_SEQUENCE_LENGTH}")
    print(f"Batch size: {SEMANTIC_BATCH_SIZE}")
    print(f"Device: {SEMANTIC_DEVICE or 'Automatic'}")
    print("Similarity treated as diagnosis: False")

    print(f"\nOutput directory: {RUMINATION_OUTPUT_DIR}")

    print("\nCONFIGURATION VALIDATION PASSED")
