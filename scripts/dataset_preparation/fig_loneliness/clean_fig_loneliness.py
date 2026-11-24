
import pandas as pd
import numpy as np
import json

from datasets import load_from_disk
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

RAW = (
    PROJECT
    / "datasets"
    / "raw"
    / "fig_loneliness"
    / "source_repo"
)

OUTPUT = (
    PROJECT
    / "datasets"
    / "processed"
    / "fig_loneliness"
    / "cleaned"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": RAW / "train_set",
    "validation": RAW / "dev_set",
    "test": RAW / "test_set"
}

EXPECTED_COUNTS = {
    "train": 3943,
    "validation": 1126,
    "test": 564
}

LABEL_MAPPING = {
    (1, 0): (0, "non_lonely"),
    (0, 1): (1, "lonely")
}

ANNOTATION_COLUMNS = [
    "context_pri",
    "interaction",
    "interpersonal_pri",
    "lonely",
    "temporal"
]


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def convert_to_list(value):

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, tuple):
        return list(value)

    return value


def serialize_annotation(value):

    value = convert_to_list(value)

    return json.dumps(
        value,
        ensure_ascii=False
    )


def validate_label(value):

    value = convert_to_list(value)

    if not isinstance(value, (list, tuple)):
        raise ValueError("Invalid loneliness label format.")

    vector = tuple(value)

    if vector not in LABEL_MAPPING:
        raise ValueError(
            f"Invalid loneliness label vector: {vector}"
        )

    return True


# ============================================================
# 3. CLEAN EACH OFFICIAL SPLIT
# ============================================================

print("=" * 70)
print("FIG-LONELINESS DATASET CLEANING")
print("=" * 70)

summary = []

for split_name, source_path in SPLITS.items():

    print("\n" + "=" * 70)
    print(f"CLEANING: {split_name.upper()}")
    print("=" * 70)

    # Load original Arrow dataset
    dataset = load_from_disk(str(source_path))

    df = dataset.to_pandas()

    original_count = len(df)

    # Verify expected record count
    if original_count != EXPECTED_COUNTS[split_name]:
        raise ValueError(
            f"Unexpected record count in {split_name}"
        )

    # Verify required columns
    required_columns = [
        "context_pri",
        "idx",
        "interaction",
        "interpersonal_pri",
        "lonely",
        "temporal",
        "text",
        "unique_id"
    ]

    if set(df.columns) != set(required_columns):
        raise ValueError(
            f"Unexpected columns in {split_name}: {df.columns.tolist()}"
        )

    # Verify no missing values before cleaning
    if df.isnull().sum().sum() != 0:
        raise ValueError(
            f"Missing values detected in {split_name}"
        )

    # --------------------------------------------------------
    # TEXT CLEANING
    # --------------------------------------------------------

    df["text"] = df["text"].astype(str)

    original_text = df["text"].copy()

    # Remove leading/trailing whitespace only
    df["text"] = df["text"].str.strip()

    changed_whitespace = int(
        (original_text != df["text"]).sum()
    )

    if df["text"].str.strip().eq("").any():
        raise ValueError(
            f"Empty text detected after cleaning: {split_name}"
        )

    # --------------------------------------------------------
    # LABEL VALIDATION
    # --------------------------------------------------------

    invalid_labels = sum(
        not validate_label(value)
        for value in df["lonely"]
    )

    if invalid_labels:
        raise ValueError(
            f"Invalid labels detected in {split_name}"
        )

    # --------------------------------------------------------
    # ID VALIDATION
    # --------------------------------------------------------

    if df["unique_id"].duplicated().any():
        raise ValueError(
            f"Duplicate IDs detected in {split_name}"
        )

    # --------------------------------------------------------
    # SERIALIZE ANNOTATION ARRAYS FOR CSV STORAGE
    # --------------------------------------------------------

    for column in ANNOTATION_COLUMNS:

        df[column] = df[column].apply(
            serialize_annotation
        )

    # --------------------------------------------------------
    # SAVE CLEANED DATASET
    # --------------------------------------------------------

    output_file = (
        OUTPUT / f"{split_name}_cleaned.csv"
    )

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # VERIFY SAVED FILE
    # --------------------------------------------------------

    saved_df = pd.read_csv(output_file)

    if len(saved_df) != original_count:
        raise ValueError(
            f"Record count changed after saving: {split_name}"
        )

    if saved_df["unique_id"].duplicated().any():
        raise ValueError(
            f"Duplicate IDs after saving: {split_name}"
        )

    # Verify labels survived serialization
    for value in saved_df["lonely"]:

        parsed = json.loads(value)

        if tuple(parsed) not in LABEL_MAPPING:
            raise ValueError(
                f"Label serialization error: {split_name}"
            )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print("Original records:", original_count)
    print("Final records:", len(saved_df))
    print("Whitespace-cleaned records:", changed_whitespace)
    print("Records removed:", original_count - len(saved_df))
    print("Missing values:", saved_df.isnull().sum().sum())
    print("Duplicate IDs:", saved_df["unique_id"].duplicated().sum())
    print("Saved:", output_file)
    print("Verification: PASS")

    summary.append({
        "split": split_name,
        "original_records": original_count,
        "final_records": len(saved_df),
        "whitespace_cleaned": changed_whitespace,
        "records_removed": original_count - len(saved_df)
    })


# ============================================================
# 4. FINAL CLEANING SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary)

print("\n" + "=" * 70)
print("FIG-LONELINESS CLEANING SUMMARY")
print("=" * 70)

print(summary_df.to_string(index=False))

print(
    "\nTotal original records:",
    summary_df["original_records"].sum()
)

print(
    "Total final records:",
    summary_df["final_records"].sum()
)

print(
    "Total whitespace-cleaned records:",
    summary_df["whitespace_cleaned"].sum()
)

print(
    "Total records removed:",
    summary_df["records_removed"].sum()
)

print("\nOutput directory:")
print(OUTPUT)

print("\nALL FIG-LONELINESS CLEANING CHECKS PASSED")
print("=" * 70)
