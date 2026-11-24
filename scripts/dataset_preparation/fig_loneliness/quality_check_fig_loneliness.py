
import pandas as pd
import numpy as np

from datasets import load_from_disk
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT = Path(
    r"D:\Projects\4AiMentalWellnessPeerSupport"
)

REPO = (
    PROJECT
    / "datasets"
    / "raw"
    / "fig_loneliness"
    / "source_repo"
)

SPLITS = {
    "TRAIN": REPO / "train_set",
    "VALIDATION": REPO / "dev_set",
    "TEST": REPO / "test_set"
}

EXPECTED_COUNTS = {
    "TRAIN": 3943,
    "VALIDATION": 1126,
    "TEST": 564
}

# Official one-hot label mapping:
# [1, 0] = Non-lonely
# [0, 1] = Lonely

LABEL_MAPPING = {
    (1, 0): (0, "non_lonely"),
    (0, 1): (1, "lonely")
}


# ============================================================
# 2. HELPER FUNCTION FOR NESTED DATA
# ============================================================

def make_hashable(value):

    """
    Convert nested arrays and collections into hashable tuples.
    Used only for complete-row duplicate detection.
    Original dataset values remain unchanged.
    """

    if isinstance(value, np.ndarray):
        return tuple(
            make_hashable(item)
            for item in value.tolist()
        )

    if isinstance(value, (list, tuple)):
        return tuple(
            make_hashable(item)
            for item in value
        )

    if isinstance(value, dict):
        return tuple(
            sorted(
                (
                    key,
                    make_hashable(item)
                )
                for key, item in value.items()
            )
        )

    if isinstance(value, np.generic):
        return value.item()

    return value


def decode_label(value):

    """
    Convert the original two-element one-hot vector
    into its standardized binary label.
    """

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)):

        vector = tuple(value)

        if vector in LABEL_MAPPING:
            return LABEL_MAPPING[vector]

    return None


# ============================================================
# 3. LOAD ORIGINAL DATASETS
# ============================================================

print("=" * 70)
print("FIG-LONELINESS DATASET QUALITY CHECK")
print("=" * 70)

data = {}

for name, path in SPLITS.items():

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {path}"
        )

    dataset = load_from_disk(str(path))

    data[name] = dataset.to_pandas()


# ============================================================
# 4. INDIVIDUAL SPLIT QUALITY ANALYSIS
# ============================================================

for name, df in data.items():

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("Records:", len(df))
    print("Columns:", len(df.columns))

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMISSING VALUES BY COLUMN")

    print(
        df.isnull()
        .sum()
        .to_string()
    )

    # --------------------------------------------------------
    # Text quality
    # --------------------------------------------------------

    text_series = df["text"]

    missing_text = text_series.isna().sum()

    text_clean = text_series.fillna("").astype(str)

    empty_text = (
        text_clean.str.len() == 0
    ).sum()

    whitespace_only = (
        text_clean.str.strip() == ""
    ).sum()

    text_lengths = text_clean.str.len()

    short_texts = (
        text_lengths < 5
    ).sum()

    print("\nTEXT QUALITY")

    print("Missing text:", missing_text)
    print("Empty text:", empty_text)
    print("Whitespace-only text:", whitespace_only)
    print("Minimum length:", text_lengths.min())
    print("Maximum length:", text_lengths.max())
    print("Mean length:", round(text_lengths.mean(), 2))
    print("Median length:", text_lengths.median())
    print("Text shorter than 5 characters:", short_texts)

    # --------------------------------------------------------
    # Unique ID quality
    # --------------------------------------------------------

    missing_ids = df["unique_id"].isna().sum()

    duplicate_ids = (
        df["unique_id"].duplicated().sum()
    )

    print("\nID QUALITY")

    print("Missing unique IDs:", missing_ids)
    print("Duplicate unique IDs:", duplicate_ids)

    # --------------------------------------------------------
    # Duplicate text analysis
    # --------------------------------------------------------

    duplicate_texts = (
        text_clean.duplicated().sum()
    )

    print("\nTEXT DUPLICATE ANALYSIS")

    print("Duplicate text records:", duplicate_texts)
    print(
        "Unique text count:",
        text_clean.nunique()
    )

    # --------------------------------------------------------
    # Complete row duplicate analysis
    # Handles NumPy arrays and nested annotations
    # --------------------------------------------------------

    hashable_df = df.copy()

    for column in hashable_df.columns:

        hashable_df[column] = (
            hashable_df[column].map(make_hashable)
        )

    duplicate_rows = (
        hashable_df.duplicated().sum()
    )

    print("\nCOMPLETE ROW DUPLICATE ANALYSIS")

    print("Duplicate complete rows:", duplicate_rows)

    # --------------------------------------------------------
    # Label validation
    # --------------------------------------------------------

    decoded_labels = df["lonely"].apply(
        decode_label
    )

    valid_labels = decoded_labels.notna()

    invalid_labels = (
        (~valid_labels).sum()
    )

    print("\nLABEL QUALITY")

    print("Valid label vectors:", valid_labels.sum())
    print("Invalid label vectors:", invalid_labels)

    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    if invalid_labels == 0:

        numeric_labels = decoded_labels.apply(
            lambda value: value[0]
        )

        readable_labels = decoded_labels.apply(
            lambda value: value[1]
        )

        print("\nCLASS DISTRIBUTION")

        print(
            readable_labels.value_counts()
            .to_string()
        )

        print("\nCLASS PERCENTAGES")

        print(
            (
                readable_labels.value_counts(
                    normalize=True
                ) * 100
            )
            .round(2)
            .to_string()
        )

    # --------------------------------------------------------
    # Expected split count
    # --------------------------------------------------------

    print("\nEXPECTED RECORD COUNT")

    print("Expected:", EXPECTED_COUNTS[name])
    print("Actual:", len(df))

    print(
        "Count matches:",
        len(df) == EXPECTED_COUNTS[name]
    )


# ============================================================
# 5. CROSS-SPLIT LEAKAGE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CROSS-SPLIT LEAKAGE ANALYSIS")
print("=" * 70)

pairs = [
    ("TRAIN", "VALIDATION"),
    ("TRAIN", "TEST"),
    ("VALIDATION", "TEST")
]

for first, second in pairs:

    df1 = data[first]
    df2 = data[second]

    # ID overlap
    ids1 = set(
        df1["unique_id"].dropna()
    )

    ids2 = set(
        df2["unique_id"].dropna()
    )

    id_overlap = ids1 & ids2

    # Exact text overlap
    texts1 = set(
        df1["text"].dropna()
    )

    texts2 = set(
        df2["text"].dropna()
    )

    text_overlap = texts1 & texts2

    print(f"\n{first} vs {second}")

    print(
        "Overlapping unique IDs:",
        len(id_overlap)
    )

    print(
        "Identical text overlap:",
        len(text_overlap)
    )


# ============================================================
# 6. COMBINED DATASET ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("COMBINED LABEL DISTRIBUTION")
print("=" * 70)

combined = pd.concat(
    data.values(),
    ignore_index=True
)

combined_labels = combined["lonely"].apply(
    decode_label
)

valid_combined = combined_labels.notna()

if not valid_combined.all():

    print(
        "Invalid combined label vectors:",
        (~valid_combined).sum()
    )

else:

    readable_labels = combined_labels.apply(
        lambda value: value[1]
    )

    numeric_labels = combined_labels.apply(
        lambda value: value[0]
    )

    print(
        readable_labels.value_counts()
        .to_string()
    )

    print("\nCombined class percentages:")

    print(
        (
            readable_labels.value_counts(
                normalize=True
            ) * 100
        )
        .round(2)
        .to_string()
    )

print("\nTOTAL DATASET STATISTICS")

print("Total records:", len(combined))

print(
    "Unique IDs:",
    combined["unique_id"].nunique()
)

print(
    "Unique texts:",
    combined["text"].nunique()
)

print(
    "Duplicate IDs across combined dataset:",
    combined["unique_id"].duplicated().sum()
)


# ============================================================
# 7. FINAL INSPECTION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FIG-LONELINESS QUALITY INSPECTION COMPLETED")
print("=" * 70)

print("Original training records:", len(data["TRAIN"]))
print("Original validation records:", len(data["VALIDATION"]))
print("Original testing records:", len(data["TEST"]))

print(
    "Total records:",
    sum(len(df) for df in data.values())
)

print("\nNo original dataset files were modified.")

print("=" * 70)
