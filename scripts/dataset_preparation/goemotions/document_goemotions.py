

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
    / "goemotions"
)

CLEANED_PATH = DATA_PATH / "cleaned"

DOCUMENTATION_PATH = (
    PROJECT_PATH
    / "datasets"
    / "documentation"
    / "goemotions"
)

DOCUMENTATION_PATH.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "training": "train_cleaned.csv",
    "validation": "validation_cleaned.csv",
    "testing": "test_cleaned.csv"
}

# --------------------------------------------------
# LOAD EMOTION MAPPING
# --------------------------------------------------

mapping_file = DATA_PATH / "emotion_mapping.txt"

emotions = []

with open(mapping_file, "r", encoding="utf-8") as file:
    for line in file:
        emotion = line.strip().split(": ", 1)[1]
        emotions.append(emotion)

if len(emotions) != 28:
    raise ValueError("Expected exactly 28 emotion categories.")

# --------------------------------------------------
# ANALYZE DATASETS
# --------------------------------------------------

metadata = {
    "dataset_name": "GoEmotions",
    "project": "AI-Powered Mental Wellness and Peer Support Chatbot",
    "task": "Multi-label emotion classification",
    "number_of_emotion_categories": len(emotions),
    "emotion_labels": emotions,
    "input_columns": ["text", "comment_id"],
    "output_columns": emotions,
    "total_records": 0,
    "splits": {},
    "preprocessing": [
        "Official train, validation and test splits retained.",
        "Numeric emotion IDs converted into emotion names.",
        "Multi-label annotations converted into binary vectors.",
        "Missing and empty text checked.",
        "Missing comment IDs checked.",
        "Duplicate comment IDs checked.",
        "Records without emotion labels checked.",
        "Original text preserved without normalization."
    ],
    "known_limitations": [
        "Some identical normalized texts occur across official splits under different comment IDs.",
        "Duplicate text records were preserved to maintain official benchmark splits.",
        "Emotion labels represent dataset annotations and are not clinical diagnoses.",
        "Dataset annotations may contain subjectivity and class imbalance."
    ],
    "license_note": (
        "Check the upstream repository's license and terms before "
        "redistribution or use outside the permitted scope."
    ),
    "documentation_generated": datetime.now().isoformat(
        timespec="seconds"
    )
}

distribution_rows = []
datasets = {}

for split_name, filename in SPLITS.items():

    file_path = CLEANED_PATH / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {file_path}")

    df = pd.read_csv(file_path)

    if len(df.columns) != 30:
        raise ValueError(
            f"{filename} should have 30 columns, "
            f"but has {len(df.columns)}."
        )

    datasets[split_name] = df

    record_count = len(df)

    metadata["splits"][split_name] = {
        "file": filename,
        "records": record_count,
        "columns": len(df.columns),
        "emotion_categories": len(emotions),
        "duplicate_comment_ids": int(
            df["comment_id"].duplicated().sum()
        ),
        "duplicate_text_records": int(
            df["text"].duplicated().sum()
        )
    }

    metadata["total_records"] += record_count

    for emotion in emotions:

        count = int(df[emotion].sum())

        distribution_rows.append({
            "split": split_name,
            "emotion": emotion,
            "positive_records": count,
            "percentage": round(
                count / record_count * 100, 4
            )
        })

# --------------------------------------------------
# CROSS-SPLIT TEXT OVERLAP
# --------------------------------------------------

text_sets = {}

for split_name, df in datasets.items():

    normalized = (
        df["text"]
        .fillna("")
        .str.strip()
        .str.lower()
    )

    text_sets[split_name] = set(normalized)

split_names = list(text_sets.keys())

text_overlap = {}

for i in range(len(split_names)):

    for j in range(i + 1, len(split_names)):

        first = split_names[i]
        second = split_names[j]

        overlap = text_sets[first].intersection(
            text_sets[second]
        )

        text_overlap[f"{first}_vs_{second}"] = len(overlap)

metadata["cross_split_identical_text_counts"] = text_overlap

# --------------------------------------------------
# SAVE METADATA JSON
# --------------------------------------------------

metadata_file = DOCUMENTATION_PATH / "metadata.json"

with open(metadata_file, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4, ensure_ascii=False)

# --------------------------------------------------
# SAVE LABEL DISTRIBUTION CSV
# --------------------------------------------------

distribution_df = pd.DataFrame(distribution_rows)

distribution_file = (
    DOCUMENTATION_PATH / "label_distribution.csv"
)

distribution_df.to_csv(
    distribution_file,
    index=False,
    encoding="utf-8"
)

# --------------------------------------------------
# GENERATE README DOCUMENTATION
# --------------------------------------------------

readme = f"""# GoEmotions Dataset Documentation

## 1. Dataset Overview

- Dataset: GoEmotions
- Project: AI-Powered Mental Wellness and Peer Support Chatbot
- Task: Multi-label emotion classification
- Emotion categories: {len(emotions)}
- Total records: {metadata['total_records']:,}

## 2. Dataset Splits

| Split | Records |
|---|---:|
"""

for split_name, details in metadata["splits"].items():
    readme += (
        f"| {split_name.title()} | "
        f"{details['records']:,} |\n"
    )

readme += """
## 3. Emotion Categories

"""

for index, emotion in enumerate(emotions):
    readme += f"{index}. {emotion}\n"

readme += """

## 4. Dataset Format

Each processed CSV contains:

- text: Original comment text.
- comment_id: Original comment identifier.
- 28 emotion columns: Binary multi-hot labels.

A value of 1 indicates an assigned emotion.
A value of 0 indicates that the emotion is not assigned.

## 5. Preparation Performed

"""

for step in metadata["preprocessing"]:
    readme += f"- {step}\n"

readme += """

## 6. Duplicate Text Across Official Splits

Identical normalized text counts:

"""

for comparison, count in text_overlap.items():
    readme += f"- {comparison}: {count}\n"

readme += """

These are text overlaps, not comment ID overlaps.
They were preserved to retain the official benchmark splits.

## 7. Known Limitations

"""

for limitation in metadata["known_limitations"]:
    readme += f"- {limitation}\n"

readme += f"""

## 8. License and Usage

{metadata['license_note']}

Verify the upstream license before redistribution.

## 9. Project Usage

GoEmotions will be used for the project's emotion
classification module. The model will predict multiple
emotion categories from a student's text input.

These predictions are supportive language indicators,
not clinical diagnoses.

## 10. Documentation Timestamp

{metadata['documentation_generated']}
"""

readme_file = DOCUMENTATION_PATH / "README.md"

with open(readme_file, "w", encoding="utf-8") as file:
    file.write(readme)

# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

print("=" * 60)
print("GOEMOTIONS FINAL DOCUMENTATION")
print("=" * 60)

print("Total records:", metadata["total_records"])
print("Emotion categories:", len(emotions))

for split_name, details in metadata["splits"].items():
    print(
        f"{split_name.title()}: "
        f"{details['records']:,} records"
    )

print("\nGenerated documentation:")

print("1.", metadata_file)
print("2.", distribution_file)
print("3.", readme_file)

print("\nCross-split identical text counts:")

for comparison, count in text_overlap.items():
    print(f"{comparison}: {count}")

print("\nDOCUMENTATION GENERATED SUCCESSFULLY")
print("=" * 60)
