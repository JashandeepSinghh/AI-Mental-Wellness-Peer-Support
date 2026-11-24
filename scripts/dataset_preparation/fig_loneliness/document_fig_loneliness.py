
import json
from pathlib import Path

import pandas as pd


# ============================================================
# FIG-LONELINESS DATASET DOCUMENTATION GENERATOR
# ============================================================

ROOT = Path("datasets/processed/fig_loneliness")
DATA_DIR = ROOT / "standardized"

OUTPUT_DIR = Path("datasets/documentation/fig_loneliness")

SPLITS = ["train", "validation", "test"]

LABEL_MAPPING = {
    "0": "non_lonely",
    "1": "lonely"
}

SOURCE_REPOSITORY = (
    "https://huggingface.co/datasets/FIG-Loneliness/FIG-Loneliness"
)

LICENSE = "CC BY-NC 4.0"


def load_splits():
    """Load all standardized dataset splits."""

    frames = {}

    for split in SPLITS:
        path = DATA_DIR / f"{split}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing standardized dataset: {path}"
            )

        frames[split] = pd.read_csv(path)

    return frames


def generate_label_distribution(frames):
    """Generate class counts and percentages for every split."""

    records = []

    for split, df in frames.items():
        total = len(df)

        counts = df["loneliness"].value_counts()

        for label in ["lonely", "non_lonely"]:
            count = int(counts.get(label, 0))

            records.append({
                "split": split,
                "label": label,
                "count": count,
                "percentage": round((count / total) * 100, 2)
            })

    combined = pd.concat(
        list(frames.values()),
        ignore_index=True
    )

    total = len(combined)
    counts = combined["loneliness"].value_counts()

    for label in ["lonely", "non_lonely"]:
        count = int(counts.get(label, 0))

        records.append({
            "split": "all",
            "label": label,
            "count": count,
            "percentage": round((count / total) * 100, 2)
        })

    return pd.DataFrame(records)


def generate_metadata(frames):
    """Create structured dataset metadata."""

    combined = pd.concat(
        list(frames.values()),
        ignore_index=True
    )

    split_metadata = {}

    for split, df in frames.items():
        split_metadata[split] = {
            "records": int(len(df)),
            "lonely": int((df["label"] == 1).sum()),
            "non_lonely": int((df["label"] == 0).sum()),
            "unique_ids": int(df["unique_id"].nunique())
        }

    metadata = {
        "dataset_name": "FIG-Loneliness",
        "project_module": "Social Isolation / Loneliness Indicators",
        "source": SOURCE_REPOSITORY,
        "license": LICENSE,
        "usage_restriction": (
            "Non-commercial use with attribution. "
            "Commercial use is not permitted under this license."
        ),
        "total_records": int(len(combined)),
        "total_unique_ids": int(combined["unique_id"].nunique()),
        "number_of_classes": 2,
        "labels": LABEL_MAPPING,
        "label_encoding": {
            "[1, 0]": {
                "label": 0,
                "name": "non_lonely"
            },
            "[0, 1]": {
                "label": 1,
                "name": "lonely"
            }
        },
        "model_input": "text",
        "model_target": "label",
        "metadata_columns": [
            "unique_id",
            "idx",
            "lonely",
            "context_pri",
            "interaction",
            "interpersonal_pri",
            "temporal"
        ],
        "split_strategy": (
            "Original official train, validation, and test splits "
            "preserved without re-splitting."
        ),
        "splits": split_metadata,
        "quality_checks": {
            "missing_values": 0,
            "invalid_label_vectors": 0,
            "duplicate_unique_ids": 0,
            "cross_split_id_overlap": 0,
            "cross_split_exact_text_overlap": 0,
            "records_removed_during_cleaning": 0,
            "records_with_outer_whitespace_cleaned": 402,
            "duplicate_text_groups_in_training": 2,
            "duplicate_text_records_in_training": 4
        },
        "text_statistics": {
            "maximum_characters": int(
                combined["text"].astype(str).str.len().max()
            ),
            "minimum_characters": int(
                combined["text"].astype(str).str.len().min()
            ),
            "mean_characters": round(
                float(combined["text"].astype(str).str.len().mean()),
                2
            ),
            "median_characters": round(
                float(combined["text"].astype(str).str.len().median()),
                2
            )
        },
        "ethical_interpretation": (
            "The dataset captures loneliness expressed in Reddit text. "
            "It does not establish a person's objective social isolation, "
            "mental health diagnosis, or clinical condition."
        ),
        "preprocessing": [
            "Removed leading and trailing whitespace from text.",
            "Preserved all records.",
            "Standardized the original one-hot loneliness labels.",
            "Serialized annotation vectors as JSON strings in CSV files.",
            "Preserved official dataset splits.",
            "Did not truncate long posts during cleaning."
        ]
    }

    return metadata


def generate_readme(metadata, distribution):
    """Generate the human-readable dataset README."""

    total = metadata["total_records"]

    lonely = int(
        distribution[
            (distribution["split"] == "all")
            & (distribution["label"] == "lonely")
        ]["count"].iloc[0]
    )

    non_lonely = int(
        distribution[
            (distribution["split"] == "all")
            & (distribution["label"] == "non_lonely")
        ]["count"].iloc[0]
    )

    lines = [
        "# FIG-Loneliness Dataset Documentation",
        "",
        "## 1. Dataset Overview",
        "",
        "FIG-Loneliness is used for the project's "
        "Social Isolation / Loneliness Indicators module.",
        "",
        f"- Dataset: FIG-Loneliness",
        f"- Source: {SOURCE_REPOSITORY}",
        f"- License: {LICENSE}",
        f"- Total records: {total:,}",
        "- Task: Binary text classification",
        "",
        "## 2. Dataset Splits",
        "",
        "| Split | Records | Lonely | Non-lonely |",
        "|---|---:|---:|---:|"
    ]

    for split, values in metadata["splits"].items():
        lines.append(
            f"| {split} | {values['records']} | "
            f"{values['lonely']} | {values['non_lonely']} |"
        )

    lines.extend([
        "",
        "## 3. Label Mapping",
        "",
        "| Original vector | Numeric label | Meaning |",
        "|---|---:|---|",
        "| `[1, 0]` | 0 | non_lonely |",
        "| `[0, 1]` | 1 | lonely |",
        "",
        f"- Total lonely records: {lonely:,} "
        f"({lonely / total * 100:.2f}%)",
        f"- Total non-lonely records: {non_lonely:,} "
        f"({non_lonely / total * 100:.2f}%)",
        "",
        "## 4. Model Input and Target",
        "",
        "- Model input: `text`",
        "- Prediction target: `label`",
        "- Label names: `lonely`, `non_lonely`",
        "",
        "The other annotation fields are retained as metadata. "
        "They must not be included as model inputs in the baseline "
        "text-classification experiment.",
        "",
        "## 5. Annotation Metadata",
        "",
        "- `unique_id`: unique record identifier",
        "- `idx`: original dataset index",
        "- `lonely`: original two-element loneliness label vector",
        "- `context_pri`: context annotation distribution",
        "- `interaction`: interaction annotation distribution",
        "- `interpersonal_pri`: interpersonal annotation distribution",
        "- `temporal`: temporal annotation distribution",
        "",
        "## 6. Cleaning and Standardization",
        "",
        "- Removed leading and trailing whitespace only.",
        "- Cleaned outer whitespace in 402 records.",
        "- Removed zero records.",
        "- Preserved duplicate text records.",
        "- Preserved all annotation vectors.",
        "- Converted original one-hot labels into numeric labels.",
        "- Preserved official train, validation, and test splits.",
        "- Long posts were not truncated during cleaning.",
        "",
        "## 7. Quality and Leakage Checks",
        "",
        "- Missing values: 0",
        "- Invalid label vectors: 0",
        "- Duplicate unique IDs: 0",
        "- Duplicate indices within splits: 0",
        "- Cross-split ID overlap: 0",
        "- Cross-split exact-text overlap: 0",
        "- All 67 standardized verification checks passed.",
        "",
        "Two duplicated text groups were found in training. "
        "Both groups have consistent loneliness labels and were preserved.",
        "",
        "## 8. Text Length Statistics",
        "",
        f"- Minimum: {metadata['text_statistics']['minimum_characters']:,} characters",
        f"- Maximum: {metadata['text_statistics']['maximum_characters']:,} characters",
        f"- Mean: {metadata['text_statistics']['mean_characters']:.2f} characters",
        f"- Median: {metadata['text_statistics']['median_characters']:.2f} characters",
        "",
        "Tokenizer truncation and sequence-length selection will be "
        "handled during model development, not during dataset cleaning.",
        "",
        "## 9. License and Usage Restrictions",
        "",
        f"The dataset is distributed under {LICENSE}. "
        "It is intended for non-commercial use with appropriate attribution. "
        "Do not assume that public download permits commercial deployment.",
        "",
        "## 10. Ethical Interpretation and Limitations",
        "",
        "This dataset captures loneliness expressed in Reddit text. "
        "It does not measure objective social isolation and must not be "
        "used to diagnose loneliness or any mental health condition.",
        "",
        "Predictions from this module are indicators for supportive "
        "conversation only. They must be interpreted cautiously and "
        "must not independently determine a student's mental health status.",
        "",
        "## 11. Model Development Recommendation",
        "",
        "- Input: text",
        "- Output: 2-class prediction",
        "- Output labels: non_lonely (0), lonely (1)",
        "- Loss: CrossEntropyLoss",
        "- Evaluation: Accuracy, Macro-F1, Weighted-F1, per-class "
        "Precision/Recall/F1, and Confusion Matrix",
        "- Use training data for fitting, validation for model selection, "
        "and test data only for final evaluation.",
        "",
        "## 12. Dataset File Locations",
        "",
        "```text",
        "datasets/",
        "├── raw/",
        "│   └── fig_loneliness/",
        "│       └── source_repo/",
        "└── processed/",
        "    └── fig_loneliness/",
        "        ├── cleaned/",
        "        └── standardized/",
        "            ├── train.csv",
        "            ├── validation.csv",
        "            ├── test.csv",
        "            └── label_mapping.json",
        "```",
        "",
        "Documentation generated by the project preprocessing pipeline.",
        ""
    ])

    return "\n".join(lines)


def main():
    print("=" * 65)
    print("FIG-LONELINESS DOCUMENTATION GENERATION")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = load_splits()

    distribution = generate_label_distribution(frames)
    metadata = generate_metadata(frames)
    readme = generate_readme(metadata, distribution)

    # Save README
    readme_path = OUTPUT_DIR / "README.md"

    with open(readme_path, "w", encoding="utf-8") as file:
        file.write(readme)

    # Save structured metadata
    metadata_path = OUTPUT_DIR / "metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    # Save class distribution
    distribution_path = OUTPUT_DIR / "label_distribution.csv"

    distribution.to_csv(
        distribution_path,
        index=False,
        encoding="utf-8"
    )

    # Final output verification
    outputs = [
        readme_path,
        metadata_path,
        distribution_path
    ]

    print("\nGENERATED DOCUMENTATION FILES")

    for path in outputs:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(
                f"Documentation output missing or empty: {path}"
            )

        print(f"PASS: {path}")

    print("\nDATASET SUMMARY")
    print(f"Total records: {metadata['total_records']}")
    print(f"Lonely: {metadata['splits']['train']['lonely'] + metadata['splits']['validation']['lonely'] + metadata['splits']['test']['lonely']}")
    print(f"Non-lonely: {metadata['splits']['train']['non_lonely'] + metadata['splits']['validation']['non_lonely'] + metadata['splits']['test']['non_lonely']}")

    print("\nALL DOCUMENTATION FILES GENERATED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
