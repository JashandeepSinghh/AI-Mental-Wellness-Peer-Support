
import pandas as pd
import json
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

INPUT = (
    PROJECT / "datasets" / "processed"
    / "dreaddit" / "cleaned"
)

OUTPUT = (
    PROJECT / "datasets" / "processed"
    / "dreaddit" / "standardized"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "train": INPUT / "train_cleaned.csv",
    "test": INPUT / "test_cleaned.csv"
}

LABEL_MAPPING = {
    0: "non_stress",
    1: "stress"
}

# --------------------------------------------------
# STANDARDIZATION
# --------------------------------------------------

print("=" * 65)
print("DREADDIT LABEL STANDARDIZATION")
print("=" * 65)

for split, file_path in FILES.items():

    print(f"\nPROCESSING {split.upper()}")
    print("-" * 55)

    df = pd.read_csv(file_path)

    original_count = len(df)

    # Validate numeric labels
    if not df["label"].isin(LABEL_MAPPING).all():
        raise ValueError("Unexpected label detected.")

    # Create standardized label
    df["stress"] = df["label"].map(LABEL_MAPPING)

    # Standardized column order
    df = df[
        [
            "text",
            "label",
            "stress",
            "id",
            "post_id",
            "confidence"
        ]
    ]

    # Verify mapping
    if not (df["stress"] == df["label"].map(LABEL_MAPPING)).all():
        raise ValueError("Label mapping verification failed.")

    if len(df) != original_count:
        raise ValueError("Record count changed.")

    if df.isnull().any().any():
        raise ValueError("Missing values detected.")

    # Save standardized dataset
    output_file = OUTPUT / f"{split}.csv"

    df.to_csv(output_file, index=False, encoding="utf-8")

    print("Records:", len(df))
    print("Columns:", list(df.columns))

    print("\nStandardized label distribution:")
    print(df["stress"].value_counts().sort_index())

    print("\nSaved:", output_file)

# --------------------------------------------------
# SAVE LABEL MAPPING
# --------------------------------------------------

mapping_file = OUTPUT / "label_mapping.json"

with open(mapping_file, "w", encoding="utf-8") as file:
    json.dump(
        {str(k): v for k, v in LABEL_MAPPING.items()},
        file,
        indent=4
    )

print("\nMapping saved:", mapping_file)

print("\n" + "=" * 65)
print("DREADDIT STANDARDIZATION COMPLETED")
print("=" * 65)
