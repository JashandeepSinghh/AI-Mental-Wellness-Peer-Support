
import pandas as pd
from pathlib import Path
from collections import Counter

# --------------------------------------------------
# PATH CONFIGURATION
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

LABEL_NAMES = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

datasets = {}

print("=" * 65)
print("TWEETEVAL DATASET QUALITY ASSESSMENT")
print("=" * 65)

# --------------------------------------------------
# LOAD DATASETS
# --------------------------------------------------

for split_name, (text_file, label_file) in SPLITS.items():

    texts = (
        DATA_PATH / text_file
    ).read_text(encoding="utf-8").splitlines()

    labels = [
        int(x)
        for x in (
            DATA_PATH / label_file
        ).read_text(encoding="utf-8").splitlines()
    ]

    df = pd.DataFrame({
        "text": texts,
        "label": labels
    })

    datasets[split_name] = df

# --------------------------------------------------
# 1. DUPLICATE TEXT QUALITY ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 65)
print("DUPLICATE TEXT ANALYSIS")
print("=" * 65)

for split_name, df in datasets.items():

    duplicate_mask = df["text"].duplicated(keep=False)

    duplicate_records = df[duplicate_mask]

    duplicate_groups = duplicate_records.groupby(
        "text",
        dropna=False
    )

    conflicting_groups = 0
    consistent_groups = 0

    print(f"\n{split_name}")

    for text, group in duplicate_groups:

        unique_labels = group["label"].nunique()

        if unique_labels > 1:
            conflicting_groups += 1
        else:
            consistent_groups += 1

    print("Duplicate text records:", int(df["text"].duplicated().sum()))
    print("Duplicate text groups:", len(duplicate_groups))
    print("Groups with consistent labels:", consistent_groups)
    print("Groups with conflicting labels:", conflicting_groups)

    if len(duplicate_groups) > 0:

        print("\nDuplicate examples:")

        shown = 0

        for text, group in duplicate_groups:

            print("\nText:", repr(text[:150]))
            print("Labels:", group["label"].tolist())

            shown += 1

            if shown == 5:
                break

# --------------------------------------------------
# 2. TEXT LENGTH ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 65)
print("TEXT LENGTH ANALYSIS")
print("=" * 65)

for split_name, df in datasets.items():

    lengths = df["text"].str.len()

    print(f"\n{split_name}")

    print("Minimum characters:", lengths.min())
    print("Maximum characters:", lengths.max())
    print("Average characters:", round(lengths.mean(), 2))
    print("Median characters:", lengths.median())

    print("Tweets below 5 characters:", (lengths < 5).sum())
    print("Tweets above 280 characters:", (lengths > 280).sum())

# --------------------------------------------------
# 3. WHITESPACE ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 65)
print("WHITESPACE ANALYSIS")
print("=" * 65)

for split_name, df in datasets.items():

    leading = df["text"].str.startswith(" ").sum()
    trailing = df["text"].str.endswith(" ").sum()

    whitespace_only = (
        df["text"].str.strip().eq("").sum()
    )

    print(f"\n{split_name}")
    print("Leading whitespace:", leading)
    print("Trailing whitespace:", trailing)
    print("Whitespace-only records:", whitespace_only)

# --------------------------------------------------
# 4. CLASS IMBALANCE ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 65)
print("CLASS IMBALANCE ANALYSIS")
print("=" * 65)

for split_name, df in datasets.items():

    counts = Counter(df["label"])

    maximum = max(counts.values())
    minimum = min(counts.values())

    imbalance_ratio = maximum / minimum

    print(f"\n{split_name}")

    for label_id, name in LABEL_NAMES.items():

        count = counts.get(label_id, 0)

        percentage = count / len(df) * 100

        print(
            f"{name}: {count:,} "
            f"({percentage:.2f}%)"
        )

    print(
        "Majority-to-minority ratio:",
        round(imbalance_ratio, 2)
    )

# --------------------------------------------------
# 5. FINAL QUALITY SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("QUALITY ASSESSMENT COMPLETED")
print("=" * 65)

print("All three official splits analyzed.")
print("No source files were modified.")
print("No records were removed.")
print("=" * 65)
