
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_PATH = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

INPUT_PATH = PROJECT_PATH / "datasets" / "processed" / "goemotions"
OUTPUT_PATH = INPUT_PATH / "cleaned"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train.csv": "train_cleaned.csv",
    "validation.csv": "validation_cleaned.csv",
    "test.csv": "test_cleaned.csv"
}


def clean_dataset(input_file, output_file):

    file_path = INPUT_PATH / input_file

    df = pd.read_csv(
        file_path,
        dtype={"text": "string", "comment_id": "string"}
    )

    initial_count = len(df)

    # Identify the 28 emotion columns
    emotion_columns = list(df.columns[2:])

    if len(emotion_columns) != 28:
        raise ValueError(
            f"Expected 28 emotion columns, found {len(emotion_columns)}"
        )

    # Convert emotion columns to numeric values
    for column in emotion_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Validate that labels contain only 0 or 1
    invalid_label_mask = ~df[emotion_columns].isin([0, 1]).all(axis=1)

    invalid_label_count = invalid_label_mask.sum()

    if invalid_label_count > 0:
        raise ValueError(
            f"Found {invalid_label_count} records with invalid label values."
        )

    # Remove missing or empty text
    missing_text_mask = df["text"].isna()

    empty_text_mask = (
        df["text"].fillna("").str.strip().eq("")
    )

    invalid_text_mask = missing_text_mask | empty_text_mask

    missing_text_count = missing_text_mask.sum()
    empty_text_count = (
        empty_text_mask & ~missing_text_mask
    ).sum()

    df = df.loc[~invalid_text_mask].copy()

    # Remove missing comment IDs
    missing_id_mask = (
        df["comment_id"].isna()
        | df["comment_id"].str.strip().eq("")
    )

    missing_id_count = missing_id_mask.sum()

    df = df.loc[~missing_id_mask].copy()

    # Remove duplicate comment IDs, retaining the first occurrence
    duplicate_id_mask = df["comment_id"].duplicated(keep="first")

    duplicate_id_count = duplicate_id_mask.sum()

    df = df.loc[~duplicate_id_mask].copy()

    # Remove records without any assigned emotion
    no_label_mask = df[emotion_columns].sum(axis=1).eq(0)

    no_label_count = no_label_mask.sum()

    df = df.loc[~no_label_mask].copy()

    # Keep original text unchanged.
    # Duplicate text with different IDs is intentionally preserved.

    # Save cleaned dataset
    output_file_path = OUTPUT_PATH / output_file

    df.to_csv(
        output_file_path,
        index=False,
        encoding="utf-8"
    )

    removed_count = initial_count - len(df)

    print("\n" + "-" * 55)
    print("FILE:", input_file)
    print("-" * 55)

    print("Original records:", initial_count)
    print("Missing text removed:", missing_text_count)
    print("Empty text removed:", empty_text_count)
    print("Missing IDs removed:", missing_id_count)
    print("Duplicate IDs removed:", duplicate_id_count)
    print("Records without labels removed:", no_label_count)
    print("Total records removed:", removed_count)
    print("Final records:", len(df))
    print("Duplicate text records preserved:", df["text"].duplicated().sum())
    print("Emotion categories:", len(emotion_columns))
    print("Output:", output_file_path)

    return initial_count, len(df)


print("=" * 55)
print("GOEMOTIONS DATASET CLEANING")
print("=" * 55)

total_original = 0
total_cleaned = 0

for input_file, output_file in SPLITS.items():

    original, cleaned = clean_dataset(
        input_file,
        output_file
    )

    total_original += original
    total_cleaned += cleaned


print("\n" + "=" * 55)
print("DATASET CLEANING COMPLETED")
print("=" * 55)

print("Total original records:", total_original)
print("Total cleaned records:", total_cleaned)
print("Total removed records:", total_original - total_cleaned)
print("Output directory:", OUTPUT_PATH)
print("=" * 55)
