
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

RAW = PROJECT / "datasets" / "raw" / "dreaddit"

OUTPUT = (
    PROJECT
    / "datasets"
    / "processed"
    / "dreaddit"
    / "cleaned"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "train": RAW / "dreaddit-train.csv",
    "test": RAW / "dreaddit-test.csv"
}

# --------------------------------------------------
# CLEANING
# --------------------------------------------------

print("=" * 65)
print("DREADDIT DATASET CLEANING")
print("=" * 65)

for split, path in FILES.items():

    print(f"\nPROCESSING {split.upper()}")
    print("-" * 55)

    df = pd.read_csv(path)

    original_count = len(df)

    # Retain only required text and audit metadata
    df = df[
        ["text", "label", "id", "post_id", "confidence"]
    ].copy()

    # Remove leading and trailing whitespace
    df["text"] = df["text"].astype(str).str.strip()

    # Identify malformed placeholder records
    placeholder_mask = (
        df["text"].str.strip().str.upper() == "#NAME?"
    )

    placeholder_count = int(placeholder_mask.sum())

    # Remove only the exact malformed placeholder
    df = df.loc[~placeholder_mask].copy()

    # Validate remaining records
    if df["text"].isna().any():
        raise ValueError("Missing text found.")

    if (df["text"].str.len() == 0).any():
        raise ValueError("Empty text found.")

    if not df["label"].isin([0, 1]).all():
        raise ValueError("Invalid stress labels found.")

    if df["label"].isna().any():
        raise ValueError("Missing labels found.")

    # Preserve confidence anomalies, but report them
    zero_confidence = int((df["confidence"] == 0).sum())

    # Save cleaned dataset
    output_file = OUTPUT / f"{split}_cleaned.csv"

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------
    # REPORT
    # --------------------------------------------------

    print("Original records:", original_count)
    print("Malformed placeholders removed:", placeholder_count)
    print("Final records:", len(df))
    print("Records removed:", original_count - len(df))
    print("Missing values:", df.isnull().sum().sum())
    print("Invalid labels:", (~df["label"].isin([0, 1])).sum())
    print("Zero-confidence records retained:", zero_confidence)
    print("Duplicate text records:", df["text"].duplicated().sum())

    print("\nFinal label distribution:")
    print(df["label"].value_counts().sort_index())

    print("\nSaved:", output_file)

print("\n" + "=" * 65)
print("DREADDIT CLEANING COMPLETED")
print("=" * 65)
