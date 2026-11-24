
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

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
# 2. LABEL MAPPING
# ============================================================

LABEL_MAPPING = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# ============================================================
# 3. LOAD DATASET
# ============================================================

def load_dataset(file_path):

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    data = pd.read_csv(file_path)

    required_columns = [
        "text",
        "label",
        "sentiment",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {file_path.name}: "
            f"{missing_columns}"
        )

    return data


# ============================================================
# 4. BASIC DATASET INSPECTION
# ============================================================

def inspect_split(split_name, data):

    print("\n" + "=" * 70)
    print(f"{split_name.upper()} DATASET INSPECTION")
    print("=" * 70)

    print("Records:", len(data))
    print("Columns:", list(data.columns))

    print("\nData types:")
    print(data.dtypes)

    print("\nMissing values:")
    print(data.isnull().sum())

    # Text validation
    missing_text = data["text"].isnull().sum()

    empty_text = (
        data["text"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print("\nTEXT VALIDATION")
    print("Missing text:", missing_text)
    print("Empty text:", empty_text)

    # Label validation
    labels = pd.to_numeric(
        data["label"],
        errors="coerce",
    )

    invalid_labels = (
        labels.isna()
        | ~labels.isin(LABEL_MAPPING.keys())
    )

    print("\nLABEL VALIDATION")
    print("Invalid labels:", int(invalid_labels.sum()))

    # Check sentiment strings against numeric labels.
    expected_sentiments = labels.map(LABEL_MAPPING)

    sentiment_mismatch = (
        expected_sentiments != data["sentiment"]
    )

    print(
        "Sentiment mapping mismatches:",
        int(sentiment_mismatch.sum()),
    )

    # Distribution
    print("\nSENTIMENT DISTRIBUTION")

    distribution = (
        data["sentiment"]
        .value_counts()
        .reindex(
            ["negative", "neutral", "positive"],
            fill_value=0,
        )
    )

    percentages = (
        distribution / len(data) * 100
        if len(data) > 0
        else distribution
    )

    distribution_table = pd.DataFrame({
        "count": distribution,
        "percentage": percentages.round(2),
    })

    print(distribution_table)

    # Text lengths
    text_lengths = (
        data["text"]
        .astype(str)
        .str.len()
    )

    print("\nTEXT LENGTH STATISTICS")
    print("Minimum:", int(text_lengths.min()))
    print("Maximum:", int(text_lengths.max()))
    print("Mean:", round(text_lengths.mean(), 2))
    print("Median:", round(text_lengths.median(), 2))
    print("95th percentile:", round(text_lengths.quantile(0.95), 2))
    print("99th percentile:", round(text_lengths.quantile(0.99), 2))

    # Duplicate text
    duplicate_texts = data["text"].duplicated().sum()

    print("\nDUPLICATE ANALYSIS")
    print("Duplicate text records:", int(duplicate_texts))
    print(
        "Unique text records:",
        int(data["text"].nunique()),
    )

    # Examples
    print("\nSAMPLE RECORDS")

    for index, row in data.head(5).iterrows():

        print(f"\nExample {index + 1}")
        print("Text:", row["text"][:300])
        print("Label:", row["label"])
        print("Sentiment:", row["sentiment"])

    return {
        "records": len(data),
        "missing_text": int(missing_text),
        "empty_text": int(empty_text),
        "invalid_labels": int(invalid_labels.sum()),
        "mapping_mismatches": int(sentiment_mismatch.sum()),
        "duplicate_texts": int(duplicate_texts),
    }


# ============================================================
# 5. CROSS-SPLIT INSPECTION
# ============================================================

def inspect_cross_split(train, validation, test):

    print("\n" + "=" * 70)
    print("CROSS-SPLIT INSPECTION")
    print("=" * 70)

    train_texts = set(train["text"])
    validation_texts = set(validation["text"])
    test_texts = set(test["text"])

    train_validation = (
        train_texts & validation_texts
    )

    train_test = (
        train_texts & test_texts
    )

    validation_test = (
        validation_texts & test_texts
    )

    print(
        "Train-validation exact text overlaps:",
        len(train_validation),
    )

    print(
        "Train-test exact text overlaps:",
        len(train_test),
    )

    print(
        "Validation-test exact text overlaps:",
        len(validation_test),
    )


# ============================================================
# 6. FINAL VALIDATION
# ============================================================

def validate_all(results):

    print("\n" + "=" * 70)
    print("FINAL DATASET VALIDATION")
    print("=" * 70)

    checks = []

    for split_name, result in results.items():

        checks.extend([
            (
                f"{split_name}: missing text",
                result["missing_text"] == 0,
            ),
            (
                f"{split_name}: empty text",
                result["empty_text"] == 0,
            ),
            (
                f"{split_name}: invalid labels",
                result["invalid_labels"] == 0,
            ),
            (
                f"{split_name}: sentiment mapping",
                result["mapping_mismatches"] == 0,
            ),
        ])

    passed = 0
    failed = 0

    for name, status in checks:

        if status:
            passed += 1
            print("[PASS]", name)
        else:
            failed += 1
            print("[FAIL]", name)

    print("\nChecks passed:", passed)
    print("Checks failed:", failed)

    if failed:
        raise RuntimeError(
            "Dataset inspection found validation failures."
        )

    print("\nTWEETEVAL DATA INSPECTION PASSED")


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TWEETEVAL SENTIMENT TRAINING DATA INSPECTION")
    print("=" * 70)

    if LABEL_MAPPING_FILE.is_file():
        print("Label mapping file found.")
    else:
        raise FileNotFoundError(
            f"Label mapping file missing: {LABEL_MAPPING_FILE}"
        )

    train = load_dataset(TRAIN_FILE)
    validation = load_dataset(VALIDATION_FILE)
    test = load_dataset(TEST_FILE)

    results = {}

    results["training"] = inspect_split(
        "training",
        train,
    )

    results["validation"] = inspect_split(
        "validation",
        validation,
    )

    results["testing"] = inspect_split(
        "testing",
        test,
    )

    inspect_cross_split(
        train,
        validation,
        test,
    )

    validate_all(results)

    print("\nEXPECTED SPLIT SIZES")
    print("Training: 45,615")
    print("Validation: 2,000")
    print("Testing: 12,284")

    print("\nACTUAL SPLIT SIZES")
    print("Training:", len(train))
    print("Validation:", len(validation))
    print("Testing:", len(test))

    print("\nTOTAL RECORDS:", len(train) + len(validation) + len(test))


if __name__ == "__main__":

    main()
