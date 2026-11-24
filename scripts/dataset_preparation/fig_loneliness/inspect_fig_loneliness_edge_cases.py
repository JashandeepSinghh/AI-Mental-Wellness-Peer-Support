
import pandas as pd
import numpy as np

from datasets import load_from_disk
from pathlib import Path


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

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

LABEL_MAPPING = {
    (1, 0): "non_lonely",
    (0, 1): "lonely"
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

def convert_array(value):

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def decode_label(value):

    value = convert_array(value)

    if isinstance(value, (list, tuple)):
        return LABEL_MAPPING.get(tuple(value), "INVALID")

    return "INVALID"


# ============================================================
# 3. LOAD DATASETS
# ============================================================

data = {}

for name, path in SPLITS.items():

    dataset = load_from_disk(str(path))
    data[name] = dataset.to_pandas()


# ============================================================
# 4. DUPLICATE TEXT INVESTIGATION
# ============================================================

print("=" * 70)
print("FIG-LONELINESS EDGE CASE INSPECTION")
print("=" * 70)

train = data["TRAIN"]

duplicate_mask = train["text"].duplicated(keep=False)

duplicate_records = train[duplicate_mask].copy()

print("\nDUPLICATE TEXT INVESTIGATION")
print("-" * 60)

print("Records belonging to duplicate groups:", len(duplicate_records))
print("Unique duplicated texts:", duplicate_records["text"].nunique())

if not duplicate_records.empty:

    grouped = duplicate_records.groupby("text", sort=False)

    for group_number, (text, group) in enumerate(
        grouped,
        start=1
    ):

        print(f"\nDuplicate Group {group_number}")
        print("-" * 60)

        print("Text:", text[:1000])
        print("Number of records:", len(group))

        for _, row in group.iterrows():

            print({
                "unique_id": row["unique_id"],
                "idx": row["idx"],
                "label_vector": convert_array(row["lonely"]),
                "decoded_label": decode_label(row["lonely"])
            })

        print(
            "Distinct labels:",
            group["lonely"].apply(decode_label).unique().tolist()
        )


# ============================================================
# 5. WHITESPACE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("WHITESPACE ANALYSIS")
print("=" * 70)

for name, df in data.items():

    text = df["text"].fillna("").astype(str)

    leading = (
        text.str.len() != text.str.lstrip().str.len()
    ).sum()

    trailing = (
        text.str.len() != text.str.rstrip().str.len()
    ).sum()

    outer = (
        text.str.len() != text.str.strip().str.len()
    ).sum()

    print(f"\n{name}")
    print("Leading whitespace:", leading)
    print("Trailing whitespace:", trailing)
    print("Records with outer whitespace:", outer)


# ============================================================
# 6. ANNOTATION VECTOR STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("ANNOTATION VECTOR STRUCTURE")
print("=" * 70)

for name, df in data.items():

    print(f"\n{name}")

    for column in ANNOTATION_COLUMNS:

        lengths = df[column].apply(
            lambda value: len(convert_array(value))
            if isinstance(convert_array(value), (list, tuple))
            else -1
        )

        print(
            f"{column}: "
            f"minimum={lengths.min()}, "
            f"maximum={lengths.max()}, "
            f"invalid={(lengths < 0).sum()}"
        )


# ============================================================
# 7. INDEX AND ID INVESTIGATION
# ============================================================

print("\n" + "=" * 70)
print("INDEX AND ID INVESTIGATION")
print("=" * 70)

for name, df in data.items():

    print(f"\n{name}")

    print("Duplicate idx values:", df["idx"].duplicated().sum())
    print("Unique idx values:", df["idx"].nunique())

    print(
        "Duplicate unique_id values:",
        df["unique_id"].duplicated().sum()
    )

    print(
        "Unique unique_id values:",
        df["unique_id"].nunique()
    )


# ============================================================
# 8. EXTREME TEXT LENGTH INSPECTION
# ============================================================

print("\n" + "=" * 70)
print("EXTREME TEXT LENGTH INSPECTION")
print("=" * 70)

for name, df in data.items():

    lengths = df["text"].str.len()

    threshold = lengths.quantile(0.99)

    # Sort by text length instead of using nlargest(key=...)
    long_records = df.assign(
        text_length=lengths
    ).sort_values(
        by="text_length",
        ascending=False
    )

    print(f"\n{name}")
    print("99th percentile length:", round(threshold, 2))
    print("Records at or above threshold:", (lengths >= threshold).sum())

    for _, row in long_records.head(3).iterrows():

        print("\nRecord ID:", row["unique_id"])
        print("Text length:", row["text_length"])
        print("Label:", decode_label(row["lonely"]))
        print("Text preview:", row["text"][:400])


# ============================================================
# 9. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("EDGE CASE INSPECTION COMPLETED")
print("=" * 70)

print("Original datasets remain unchanged.")
