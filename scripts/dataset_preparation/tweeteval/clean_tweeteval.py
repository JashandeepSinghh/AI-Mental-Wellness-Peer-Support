
import pandas as pd
from pathlib import Path
from collections import Counter

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_PATH = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

RAW_PATH = (
    PROJECT_PATH
    / "datasets"
    / "raw"
    / "tweeteval"
    / "sentiment"
)

OUTPUT_PATH = (
    PROJECT_PATH
    / "datasets"
    / "processed"
    / "tweeteval"
    / "sentiment"
    / "cleaned"
)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "Training": ("train_text.txt", "train_labels.txt", "train_cleaned.csv"),
    "Validation": ("val_text.txt", "val_labels.txt", "validation_cleaned.csv"),
    "Testing": ("test_text.txt", "test_labels.txt", "test_cleaned.csv")
}

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

all_datasets = {}

total_original = 0
total_cleaned = 0

print("=" * 65)
print("TWEETEVAL DATASET CLEANING")
print("=" * 65)

# --------------------------------------------------
# CLEAN EACH OFFICIAL SPLIT
# --------------------------------------------------

for split_name, (text_file, label_file, output_file) in SPLITS.items():

    # Load original text
    texts = (
        RAW_PATH / text_file
    ).read_text(encoding="utf-8").splitlines()

    # Load original labels
    labels = [
        int(label)
        for label in (
            RAW_PATH / label_file
        ).read_text(encoding="utf-8").splitlines()
    ]

    if len(texts) != len(labels):
        raise ValueError(
            f"{split_name}: text-label count mismatch."
        )

    # Create original DataFrame
    original_df = pd.DataFrame({
        "text": texts,
        "label": labels
    })

    # Remove only leading and trailing whitespace
    cleaned_texts = [text.strip() for text in texts]

    # Preserve all labels and records
    cleaned_df = pd.DataFrame({
        "text": cleaned_texts,
        "label": labels,
        "sentiment": [
            LABEL_NAMES[label] for label in labels
        ]
    })

    # Verify no empty text was created
    empty_count = cleaned_df["text"].eq("").sum()

    if empty_count > 0:
        raise ValueError(
            f"{split_name}: empty text detected after cleaning."
        )

    # Verify label preservation
    labels_preserved = (
        original_df["label"].equals(cleaned_df["label"])
    )

    if not labels_preserved:
        raise ValueError(
            f"{split_name}: labels changed during cleaning."
        )

    # Count whitespace changes
    changed_count = sum(
        old != new
        for old, new in zip(texts, cleaned_texts)
    )

    # Verify internal text content is unchanged
    internal_content_preserved = all(
        old.strip() == new
        for old, new in zip(texts, cleaned_texts)
    )

    if not internal_content_preserved:
        raise ValueError(
            f"{split_name}: unexpected text modification."
        )

    # Save cleaned dataset
    output_file_path = OUTPUT_PATH / output_file

    cleaned_df.to_csv(
        output_file_path,
        index=False,
        encoding="utf-8"
    )

    all_datasets[split_name] = cleaned_df

    total_original += len(original_df)
    total_cleaned += len(cleaned_df)

    print("\n" + "-" * 55)
    print(split_name.upper())
    print("-" * 55)

    print("Original records:", len(original_df))
    print("Cleaned records:", len(cleaned_df))
    print("Whitespace-normalized records:", changed_count)
    print("Records removed:", len(original_df) - len(cleaned_df))
    print("Empty records after cleaning:", empty_count)
    print("Labels preserved:", labels_preserved)
    print("Internal text content preserved:", internal_content_preserved)
    print("Output:", output_file_path)

# --------------------------------------------------
# CROSS-SPLIT DUPLICATE CHECK AFTER CLEANING
# --------------------------------------------------

print("\n" + "=" * 65)
print("CROSS-SPLIT DUPLICATE CHECK AFTER CLEANING")
print("=" * 65)

split_names = list(all_datasets.keys())

for i in range(len(split_names)):

    for j in range(i + 1, len(split_names)):

        first = split_names[i]
        second = split_names[j]

        first_texts = set(all_datasets[first]["text"])
        second_texts = set(all_datasets[second]["text"])

        overlap = first_texts.intersection(second_texts)

        print(
            f"{first} vs {second}: "
            f"{len(overlap)} identical texts"
        )

# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("CLEANING COMPLETED")
print("=" * 65)

print("Total original records:", total_original)
print("Total cleaned records:", total_cleaned)
print("Total records removed:", total_original - total_cleaned)

print("Output directory:", OUTPUT_PATH)

if total_original != total_cleaned:
    raise ValueError("Record count changed unexpectedly.")

print("\nALL CLEANING CHECKS PASSED")
print("=" * 65)
