
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "datasets" / "processed" / "dreaddit" / "final_splits"

FILES = {
    "train": DATA_DIR / "train.csv",
    "validation": DATA_DIR / "validation.csv",
    "test": DATA_DIR / "test.csv",
}

EXPECTED_COUNTS = {
    "train": 2266,
    "validation": 568,
    "test": 715,
}

REQUIRED_COLUMNS = {"text", "label", "stress"}
LABEL_MAPPING = {0: "non_stress", 1: "stress"}


def inspect_split(name, path):
    print("\n" + "=" * 65)
    print(f"{name.upper()} DATASET INSPECTION")
    print("=" * 65)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{name}: Missing required columns: {sorted(missing_columns)}"
        )

    if len(df) != EXPECTED_COUNTS[name]:
        raise ValueError(
            f"{name}: Expected {EXPECTED_COUNTS[name]} records, "
            f"found {len(df)}"
        )

    if df["text"].isna().any():
        raise ValueError(f"{name}: Missing text values detected")

    if df["label"].isna().any():
        raise ValueError(f"{name}: Missing labels detected")

    if df["stress"].isna().any():
        raise ValueError(f"{name}: Missing stress mappings detected")

    if not df["label"].isin([0, 1]).all():
        raise ValueError(f"{name}: Invalid numeric labels detected")

    expected_stress = df["label"].map(LABEL_MAPPING)

    if not expected_stress.equals(df["stress"]):
        raise ValueError(f"{name}: Numeric/text label mapping mismatch")

    if df["text"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{name}: Empty or whitespace-only text detected")

    text_lengths = df["text"].astype(str).str.split().str.len()

    distribution = (
        df["stress"]
        .value_counts()
        .reindex(["non_stress", "stress"], fill_value=0)
    )

    print(f"File: {path}")
    print(f"Records: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"Duplicate record IDs: {int(df['id'].duplicated().sum()) if 'id' in df else 'N/A'}")
    print(f"Duplicate post IDs: {int(df['post_id'].duplicated().sum()) if 'post_id' in df else 'N/A'}")
    print(f"Duplicate texts: {int(df['text'].duplicated().sum())}")

    print("\nClass distribution:")
    for label, count in distribution.items():
        percentage = count / len(df) * 100
        print(f"{label:12s}: {count:5,d} ({percentage:6.2f}%)")

    print("\nText length statistics (words):")
    print(f"Minimum: {text_lengths.min()}")
    print(f"Maximum: {text_lengths.max()}")
    print(f"Mean: {text_lengths.mean():.2f}")
    print(f"Median: {text_lengths.median():.2f}")
    print(f"95th percentile: {np.percentile(text_lengths, 95):.2f}")
    print(f"99th percentile: {np.percentile(text_lengths, 99):.2f}")

    print("\nSample records:")
    for _, row in df.head(3).iterrows():
        print(f"\nText: {str(row['text'])[:250]}")
        print(f"Label: {row['label']} -> {row['stress']}")

    print(f"\n{name.upper()} VALIDATION: PASSED")

    return df


def main():
    print("=" * 65)
    print("DREADDIT STRESS DETECTION DATA INSPECTION")
    print("=" * 65)

    print(f"Project root: {ROOT}")
    print(f"Dataset directory: {DATA_DIR}")

    splits = {}

    for name, path in FILES.items():
        splits[name] = inspect_split(name, path)

    print("\n" + "=" * 65)
    print("CROSS-SPLIT LEAKAGE CHECK")
    print("=" * 65)

    names = list(splits)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            first_name = names[i]
            second_name = names[j]

            first = splits[first_name]
            second = splits[second_name]

            if "post_id" in first.columns and "post_id" in second.columns:
                post_overlap = set(first["post_id"]) & set(second["post_id"])
                print(
                    f"{first_name}-{second_name} post ID overlap: "
                    f"{len(post_overlap)}"
                )

            text_overlap = (
                set(first["text"].astype(str))
                & set(second["text"].astype(str))
            )

            print(
                f"{first_name}-{second_name} exact text overlap: "
                f"{len(text_overlap)}"
            )

    print("\n" + "=" * 65)
    print("FINAL INSPECTION SUMMARY")
    print("=" * 65)

    total_records = sum(len(df) for df in splits.values())

    print(f"Total records: {total_records:,}")
    print("All split sizes: PASSED")
    print("Required columns: PASSED")
    print("Missing-value checks: PASSED")
    print("Text validation: PASSED")
    print("Label validation: PASSED")
    print("Label mapping: PASSED")
    print("Dataset inspection: COMPLETED")


if __name__ == "__main__":
    main()
