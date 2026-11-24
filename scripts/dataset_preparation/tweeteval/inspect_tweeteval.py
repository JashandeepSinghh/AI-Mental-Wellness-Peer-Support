
from pathlib import Path
from collections import Counter
import pandas as pd

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_PATH = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

DATA_PATH = (
    PROJECT_PATH
    / "datasets"
    / "raw"
    / "tweeteval"
    / "sentiment"
)

SPLITS = {
    "Training": ("train_text.txt", "train_labels.txt"),
    "Validation": ("val_text.txt", "val_labels.txt"),
    "Testing": ("test_text.txt", "test_labels.txt")
}

EXPECTED_LABELS = {0, 1, 2}

# --------------------------------------------------
# LOAD MAPPING
# --------------------------------------------------

mapping_file = DATA_PATH / "mapping.txt"

print("=" * 65)
print("TWEETEVAL DATASET INSPECTION")
print("=" * 65)

print("\nOFFICIAL LABEL MAPPING")
print("-" * 45)

mapping_lines = mapping_file.read_text(
    encoding="utf-8"
).splitlines()

for line in mapping_lines:
    print(line)

# --------------------------------------------------
# LOAD AND INSPECT SPLITS
# --------------------------------------------------

datasets = {}

total_records = 0
all_checks_passed = True

for split_name, (text_file, label_file) in SPLITS.items():

    texts = (
        DATA_PATH / text_file
    ).read_text(encoding="utf-8").splitlines()

    labels = (
        DATA_PATH / label_file
    ).read_text(encoding="utf-8").splitlines()

    print("\n" + "=" * 65)
    print(split_name.upper())
    print("=" * 65)

    # Basic counts
    print("Text records:", len(texts))
    print("Label records:", len(labels))

    # Text-label alignment
    alignment = len(texts) == len(labels)

    print("Text-label alignment:", alignment)

    if not alignment:
        all_checks_passed = False
        raise ValueError(
            f"{split_name}: text and label counts do not match."
        )

    # Convert labels to integers
    try:
        numeric_labels = [int(label.strip()) for label in labels]
    except ValueError:
        raise ValueError(
            f"{split_name}: non-numeric label detected."
        )

    # Validate label IDs
    invalid_labels = [
        label for label in numeric_labels
        if label not in EXPECTED_LABELS
    ]

    print("Invalid label IDs:", len(invalid_labels))

    if invalid_labels:
        all_checks_passed = False

    # Missing and empty text
    missing_text = sum(text is None for text in texts)

    empty_text = sum(
        text.strip() == "" for text in texts
    )

    print("Missing text:", missing_text)
    print("Empty text:", empty_text)

    # Duplicate text within split
    duplicate_texts = (
        len(texts) - len(set(texts))
    )

    print("Duplicate text records:", duplicate_texts)

    # Sentiment distribution
    distribution = Counter(numeric_labels)

    print("\nSENTIMENT DISTRIBUTION")

    for label_id, sentiment in [
        (0, "Negative"),
        (1, "Neutral"),
        (2, "Positive")
    ]:

        count = distribution.get(label_id, 0)

        percentage = (
            count / len(numeric_labels) * 100
            if numeric_labels else 0
        )

        print(
            f"{sentiment}: {count:,} "
            f"({percentage:.2f}%)"
        )

    # Store original text and labels
    datasets[split_name] = {
        "texts": texts,
        "labels": numeric_labels
    }

    total_records += len(texts)

    # Display sample tweets
    print("\nSAMPLE TWEETS")

    for label_id, sentiment in [
        (0, "Negative"),
        (1, "Neutral"),
        (2, "Positive")
    ]:

        print(f"\n{sentiment}:")

        sample_count = 0

        for text, label in zip(texts, numeric_labels):

            if label == label_id:

                print("-", text[:250])

                sample_count += 1

            if sample_count == 3:
                break


# --------------------------------------------------
# CROSS-SPLIT DUPLICATE INSPECTION
# --------------------------------------------------

print("\n" + "=" * 65)
print("CROSS-SPLIT TEXT OVERLAP")
print("=" * 65)

split_names = list(datasets.keys())

text_sets = {
    name: set(data["texts"])
    for name, data in datasets.items()
}

for i in range(len(split_names)):

    for j in range(i + 1, len(split_names)):

        first = split_names[i]
        second = split_names[j]

        overlap = text_sets[first].intersection(
            text_sets[second]
        )

        print(
            f"{first} vs {second}: "
            f"{len(overlap)} identical texts"
        )


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("FINAL INSPECTION SUMMARY")
print("=" * 65)

print("Total records:", f"{total_records:,}")
print("Expected sentiment classes:", len(EXPECTED_LABELS))
print("All text-label alignments:", "PASS" if all_checks_passed else "FAIL")

print("\nINSPECTION COMPLETED")
print("=" * 65)
