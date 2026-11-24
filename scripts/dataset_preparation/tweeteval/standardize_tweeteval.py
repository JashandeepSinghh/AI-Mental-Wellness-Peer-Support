
import pandas as pd
import json
from pathlib import Path

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_PATH = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

INPUT_PATH = (
    PROJECT_PATH
    / "datasets"
    / "processed"
    / "tweeteval"
    / "sentiment"
    / "cleaned"
)

OUTPUT_PATH = (
    PROJECT_PATH
    / "datasets"
    / "processed"
    / "tweeteval"
    / "sentiment"
    / "standardized"
)

RAW_PATH = (
    PROJECT_PATH
    / "datasets"
    / "raw"
    / "tweeteval"
    / "sentiment"
)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# OFFICIAL LABEL MAPPING
# --------------------------------------------------

EXPECTED_MAPPING = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

mapping_file = RAW_PATH / "mapping.txt"

official_mapping = {}

with open(mapping_file, encoding="utf-8") as file:

    for line in file:

        parts = line.strip().split()

        if len(parts) != 2:
            raise ValueError(
                f"Unexpected mapping format: {line!r}"
            )

        label_id = int(parts[0])
        sentiment = parts[1].lower()

        official_mapping[label_id] = sentiment

if official_mapping != EXPECTED_MAPPING:
    raise ValueError(
        f"Unexpected official mapping: {official_mapping}"
    )

# --------------------------------------------------
# PROCESS OFFICIAL SPLITS
# --------------------------------------------------

SPLITS = {
    "train_cleaned.csv": "train.csv",
    "validation_cleaned.csv": "validation.csv",
    "test_cleaned.csv": "test.csv"
}

total_records = 0

print("=" * 60)
print("TWEETEVAL LABEL STANDARDIZATION")
print("=" * 60)

for input_file, output_file in SPLITS.items():

    df = pd.read_csv(INPUT_PATH / input_file)

    # Validate required columns
    required_columns = {"text", "label", "sentiment"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Missing required columns in {input_file}"
        )

    # Ensure labels are integers
    df["label"] = pd.to_numeric(
        df["label"],
        errors="raise"
    ).astype("int64")

    # Validate class IDs
    invalid_labels = ~df["label"].isin(
        EXPECTED_MAPPING.keys()
    )

    if invalid_labels.any():
        raise ValueError(
            f"Invalid labels in {input_file}"
        )

    # Standardize sentiment names from official mapping
    df["sentiment"] = df["label"].map(
        EXPECTED_MAPPING
    )

    # Validate text and labels
    if df["text"].isna().any():
        raise ValueError("Missing text detected.")

    if df["text"].str.strip().eq("").any():
        raise ValueError("Empty text detected.")

    if df["label"].isna().any():
        raise ValueError("Missing labels detected.")

    # Consistent column order
    df = df[["text", "label", "sentiment"]]

    # Save standardized split
    output_file_path = OUTPUT_PATH / output_file

    df.to_csv(
        output_file_path,
        index=False,
        encoding="utf-8"
    )

    total_records += len(df)

    print(f"\nFile: {output_file}")
    print("Records:", len(df))
    print("Columns:", list(df.columns))
    print("Class distribution:")

    print(
        df["sentiment"]
        .value_counts()
        .reindex(
            ["negative", "neutral", "positive"],
            fill_value=0
        )
        .to_string()
    )

    print("Saved:", output_file_path)

# --------------------------------------------------
# SAVE LABEL MAPPING
# --------------------------------------------------

mapping_output = OUTPUT_PATH / "label_mapping.json"

with open(mapping_output, "w", encoding="utf-8") as file:

    json.dump(
        {
            str(label_id): sentiment
            for label_id, sentiment in EXPECTED_MAPPING.items()
        },
        file,
        indent=4
    )

# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("STANDARDIZATION COMPLETED")
print("=" * 60)

print("Total records:", total_records)
print("Number of classes:", len(EXPECTED_MAPPING))
print("Label mapping verified:", True)
print("Output directory:", OUTPUT_PATH)
print("=" * 60)
