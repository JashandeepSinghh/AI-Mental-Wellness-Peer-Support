
import pandas as pd
import json

from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

BASE = (
    PROJECT
    / "datasets"
    / "processed"
    / "fig_loneliness"
)

CLEANED = BASE / "cleaned"
STANDARDIZED = BASE / "standardized"

STANDARDIZED.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": CLEANED / "train_cleaned.csv",
    "validation": CLEANED / "validation_cleaned.csv",
    "test": CLEANED / "test_cleaned.csv"
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


# ============================================================
# 2. LABEL CONVERSION FUNCTION
# ============================================================

def standardize_label(value):

    vector = json.loads(value)

    if not isinstance(vector, list) or len(vector) != 2:
        raise ValueError(
            f"Invalid loneliness vector: {value}"
        )

    key = tuple(vector)

    if key not in LABEL_MAPPING:
        raise ValueError(
            f"Unknown loneliness vector: {key}"
        )

    return LABEL_MAPPING[key]


# ============================================================
# 3. STANDARDIZE EACH SPLIT
# ============================================================

print("=" * 70)
print("FIG-LONELINESS LABEL STANDARDIZATION")
print("=" * 70)

summary = []

for split_name, source_file in SPLITS.items():

    print("\n" + "=" * 70)
    print("PROCESSING:", split_name.upper())
    print("=" * 70)

    df = pd.read_csv(source_file)

    original_count = len(df)

    if original_count != EXPECTED_COUNTS[split_name]:
        raise ValueError(
            f"Unexpected record count in {split_name}"
        )

    # Convert original one-hot vectors
    decoded = df["lonely"].apply(
        standardize_label
    )

    df["label"] = decoded.apply(
        lambda value: value[0]
    )

    df["loneliness"] = decoded.apply(
        lambda value: value[1]
    )

    # Arrange columns:
    # text and labels first, followed by metadata
    column_order = [
        "text",
        "label",
        "loneliness",
        "unique_id",
        "idx",
        "lonely",
        "context_pri",
        "interaction",
        "interpersonal_pri",
        "temporal"
    ]

    df = df[column_order]

    # Verify labels
    if not df["label"].isin([0, 1]).all():
        raise ValueError("Invalid standardized labels.")

    expected_names = df["label"].map({
        0: "non_lonely",
        1: "lonely"
    })

    if not expected_names.equals(df["loneliness"]):
        raise ValueError("Label mapping mismatch.")

    # Verify no missing values
    if df.isnull().sum().sum() != 0:
        raise ValueError("Missing values detected.")

    # Save
    output_file = (
        STANDARDIZED / f"{split_name}.csv"
    )

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    # Reload and verify saved file
    saved = pd.read_csv(output_file)

    if len(saved) != original_count:
        raise ValueError("Record count changed after saving.")

    if not saved["label"].isin([0, 1]).all():
        raise ValueError("Saved labels are invalid.")

    if saved["unique_id"].duplicated().any():
        raise ValueError("Duplicate IDs detected.")

    counts = saved["loneliness"].value_counts()

    print("Original records:", original_count)
    print("Standardized records:", len(saved))

    print("\nLabel distribution:")
    print(counts.to_string())

    print("\nLabel percentages:")
    print(
        (
            saved["loneliness"].value_counts(
                normalize=True
            ) * 100
        ).round(2).to_string()
    )

    print("\nOutput:", output_file)
    print("Verification: PASS")

    summary.append({
        "split": split_name,
        "records": len(saved),
        "lonely": int((saved["label"] == 1).sum()),
        "non_lonely": int((saved["label"] == 0).sum())
    })


# ============================================================
# 4. SAVE LABEL MAPPING
# ============================================================

mapping = {
    "original_encoding": {
        "[1, 0]": {
            "label": 0,
            "loneliness": "non_lonely"
        },
        "[0, 1]": {
            "label": 1,
            "loneliness": "lonely"
        }
    },
    "model_target": "label",
    "model_input": "text",
    "number_of_classes": 2
}

with open(
    STANDARDIZED / "label_mapping.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        mapping,
        file,
        indent=4
    )


# ============================================================
# 5. FINAL SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary)

print("\n" + "=" * 70)
print("STANDARDIZATION SUMMARY")
print("=" * 70)

print(summary_df.to_string(index=False))

print(
    "\nTotal standardized records:",
    summary_df["records"].sum()
)

print(
    "Total lonely:",
    summary_df["lonely"].sum()
)

print(
    "Total non-lonely:",
    summary_df["non_lonely"].sum()
)

print("\nLabel mapping saved:")
print(STANDARDIZED / "label_mapping.json")

print("\nALL FIG-LONELINESS STANDARDIZATION CHECKS PASSED")
print("=" * 70)
