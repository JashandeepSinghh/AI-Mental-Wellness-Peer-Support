
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_PATH = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

DATA_PATH = (
    PROJECT_PATH
    / "datasets"
    / "processed"
    / "tweeteval"
    / "sentiment"
    / "standardized"
)

DOC_PATH = (
    PROJECT_PATH
    / "datasets"
    / "documentation"
    / "tweeteval"
    / "sentiment"
)

DOC_PATH.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "training": "train.csv",
    "validation": "validation.csv",
    "testing": "test.csv"
}

LABEL_MAPPING = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

datasets = {}

# --------------------------------------------------
# INITIALIZE METADATA
# --------------------------------------------------

metadata = {
    "dataset_name": "TweetEval Sentiment",
    "project": "AI-Powered Mental Wellness and Peer Support Chatbot",
    "task": "Three-class sentiment classification",
    "number_of_classes": 3,
    "label_mapping": LABEL_MAPPING,
    "input_column": "text",
    "target_columns": ["label", "sentiment"],
    "total_records": 0,
    "splits": {},
    "preprocessing": [
        "Official training, validation and testing splits retained.",
        "Leading and trailing whitespace removed using strip().",
        "Internal text content preserved.",
        "Numeric sentiment labels preserved.",
        "Numeric labels mapped to sentiment names.",
        "Missing and empty text checked.",
        "Invalid labels checked.",
        "Duplicate records preserved."
    ],
    "known_limitations": [
        "Training data contains duplicate tweet records.",
        "Three duplicate text groups have conflicting sentiment annotations.",
        "The official test distribution differs from the training distribution.",
        "Sentiment annotations are not clinical or psychological diagnoses."
    ],
    "cross_split_identical_texts": {},
    "license_note": (
        "Verify the official TweetEval repository and underlying "
        "Twitter content terms before redistribution or reuse."
    ),
    "documentation_generated": datetime.now().isoformat(
        timespec="seconds"
    )
}

# --------------------------------------------------
# LOAD AND ANALYZE SPLITS
# --------------------------------------------------

distribution_rows = []

for split_name, filename in SPLITS.items():

    file_path = DATA_PATH / filename

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    df = pd.read_csv(file_path)

    if list(df.columns) != ["text", "label", "sentiment"]:
        raise ValueError(f"Unexpected columns in {filename}")

    if df.isna().any().any():
        raise ValueError(f"Missing values in {filename}")

    if not df["label"].isin(LABEL_MAPPING).all():
        raise ValueError(f"Invalid labels in {filename}")

    if not df["sentiment"].equals(
        df["label"].map(LABEL_MAPPING)
    ):
        raise ValueError(f"Incorrect sentiment mapping in {filename}")

    datasets[split_name] = df

    metadata["total_records"] += len(df)

    metadata["splits"][split_name] = {
        "file": filename,
        "records": len(df),
        "columns": list(df.columns),
        "duplicate_text_records": int(
            df["text"].duplicated().sum()
        ),
        "class_distribution": {}
    }

    for label_id, sentiment in LABEL_MAPPING.items():

        count = int((df["label"] == label_id).sum())

        percentage = round(
            count / len(df) * 100, 4
        )

        metadata["splits"][split_name][
            "class_distribution"
        ][sentiment] = count

        distribution_rows.append({
            "split": split_name,
            "label_id": label_id,
            "sentiment": sentiment,
            "records": count,
            "percentage": percentage
        })

# --------------------------------------------------
# CROSS-SPLIT DUPLICATE TEXT ANALYSIS
# --------------------------------------------------

split_names = list(datasets.keys())

for i in range(len(split_names)):

    for j in range(i + 1, len(split_names)):

        first = split_names[i]
        second = split_names[j]

        first_texts = set(datasets[first]["text"])
        second_texts = set(datasets[second]["text"])

        overlap = first_texts.intersection(second_texts)

        metadata["cross_split_identical_texts"][
            f"{first}_vs_{second}"
        ] = len(overlap)

# --------------------------------------------------
# SAVE METADATA
# --------------------------------------------------

metadata_file = DOC_PATH / "metadata.json"

with open(metadata_file, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

# --------------------------------------------------
# SAVE CLASS DISTRIBUTION
# --------------------------------------------------

distribution_df = pd.DataFrame(distribution_rows)

distribution_file = DOC_PATH / "label_distribution.csv"

distribution_df.to_csv(
    distribution_file,
    index=False,
    encoding="utf-8"
)

# --------------------------------------------------
# GENERATE README
# --------------------------------------------------

readme = f"""# TweetEval Sentiment Dataset Documentation

## 1. Dataset Overview

- Dataset: TweetEval Sentiment
- Project: AI-Powered Mental Wellness and Peer Support Chatbot
- Task: Three-class sentiment classification
- Total records: {metadata['total_records']:,}
- Number of classes: 3

## 2. Official Label Mapping

| ID | Sentiment |
|---:|---|
| 0 | Negative |
| 1 | Neutral |
| 2 | Positive |

## 3. Official Dataset Splits

| Split | Records |
|---|---:|
"""

for split_name, details in metadata["splits"].items():
    readme += f"| {split_name.title()} | {details['records']:,} |\n"

readme += """

## 4. Class Distribution

| Split | Negative | Neutral | Positive |
|---|---:|---:|---:|
"""

for split_name, details in metadata["splits"].items():

    counts = details["class_distribution"]

    readme += (
        f"| {split_name.title()} | "
        f"{counts['negative']:,} | "
        f"{counts['neutral']:,} | "
        f"{counts['positive']:,} |\n"
    )

readme += """

## 5. Processed Dataset Format

Each CSV contains:

- text: Cleaned tweet text.
- label: Integer sentiment class.
- sentiment: Corresponding sentiment name.

## 6. Preparation Performed

"""

for step in metadata["preprocessing"]:
    readme += f"- {step}\n"

readme += """

## 7. Duplicate Text Analysis

"""

for split_name, details in metadata["splits"].items():
    readme += (
        f"- {split_name.title()}: "
        f"{details['duplicate_text_records']} duplicate text records\n"
    )

readme += """

Three duplicate text groups in the training split have
conflicting sentiment annotations. All duplicate records
were preserved.

## 8. Cross-Split Text Overlap

"""

for comparison, count in metadata[
    "cross_split_identical_texts"
].items():

    readme += f"- {comparison}: {count}\n"

readme += """

## 9. Known Limitations

"""

for limitation in metadata["known_limitations"]:
    readme += f"- {limitation}\n"

readme += f"""

## 10. License and Usage

{metadata['license_note']}

## 11. Project Application

TweetEval will support the sentiment analysis module.
The classifier will predict negative, neutral or positive
sentiment from student text.

Sentiment predictions are linguistic indicators and
must not independently determine mental health risk.

## 12. Documentation Timestamp

{metadata['documentation_generated']}
"""

readme_file = DOC_PATH / "README.md"

with open(readme_file, "w", encoding="utf-8") as file:
    file.write(readme)

# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

print("=" * 60)
print("TWEETEVAL FINAL DOCUMENTATION")
print("=" * 60)

print("Total records:", metadata["total_records"])
print("Sentiment classes:", metadata["number_of_classes"])

for split_name, details in metadata["splits"].items():

    print(
        f"{split_name.title()}: "
        f"{details['records']:,} records"
    )

print("\nGenerated documentation:")

print("1.", metadata_file)
print("2.", distribution_file)
print("3.", readme_file)

print("\nDOCUMENTATION GENERATED SUCCESSFULLY")
print("=" * 60)
