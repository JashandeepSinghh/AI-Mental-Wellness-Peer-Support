
import json
import pandas as pd
from pathlib import Path
from datetime import date

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

BASE = PROJECT / "datasets" / "processed" / "dreaddit"
RAW = PROJECT / "datasets" / "raw" / "dreaddit"
DOCS = PROJECT / "datasets" / "documentation" / "dreaddit"

FINAL = BASE / "final_splits"

DOCS.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD FINAL DATASETS
# --------------------------------------------------

train = pd.read_csv(FINAL / "train.csv")
validation = pd.read_csv(FINAL / "validation.csv")
test = pd.read_csv(FINAL / "test.csv")

splits = {
    "train": train,
    "validation": validation,
    "test": test
}

# --------------------------------------------------
# LABEL DISTRIBUTION
# --------------------------------------------------

distribution_rows = []

for split_name, df in splits.items():

    counts = df["stress"].value_counts()
    total = len(df)

    for label in ["stress", "non_stress"]:

        count = int(counts.get(label, 0))

        distribution_rows.append({
            "split": split_name,
            "label": label,
            "count": count,
            "percentage": round(count / total * 100, 2)
        })

distribution = pd.DataFrame(distribution_rows)

distribution.to_csv(
    DOCS / "label_distribution.csv",
    index=False
)

# --------------------------------------------------
# DATASET STATISTICS
# --------------------------------------------------

total_records = sum(len(df) for df in splits.values())

zero_confidence_test = int(
    (test["confidence"] == 0).sum()
)

post_overlap = {
    "train_validation": len(
        set(train["post_id"]) & set(validation["post_id"])
    ),
    "train_test": len(
        set(train["post_id"]) & set(test["post_id"])
    ),
    "validation_test": len(
        set(validation["post_id"]) & set(test["post_id"])
    )
}

text_overlap = {
    "train_validation": len(
        set(train["text"]) & set(validation["text"])
    ),
    "train_test": len(
        set(train["text"]) & set(test["text"])
    ),
    "validation_test": len(
        set(validation["text"]) & set(test["text"])
    )
}

# --------------------------------------------------
# METADATA JSON
# --------------------------------------------------

metadata = {
    "dataset_name": "Dreaddit",
    "project": "AI-Powered Mental Wellness and Peer Support Chatbot",
    "module": "Stress Detection",
    "source": {
        "name": "Dreaddit: A Reddit Dataset for Stress Analysis",
        "url": "https://www.cs.columbia.edu/~eturcan/data/dreaddit.zip",
        "access_type": "Publicly downloadable research dataset",
        "license_status": (
            "Verify the original publisher's usage and redistribution "
            "terms before redistribution. Public download does not "
            "automatically establish an open license."
        )
    },
    "dataset_structure": {
        "original_training_records": 2838,
        "original_testing_records": 715,
        "removed_training_placeholders": 4,
        "final_records": total_records
    },
    "final_splits": {
        "training": len(train),
        "validation": len(validation),
        "testing": len(test)
    },
    "labels": {
        "0": "non_stress",
        "1": "stress"
    },
    "label_counts": {
        "stress": int(
            sum((df["label"] == 1).sum() for df in splits.values())
        ),
        "non_stress": int(
            sum((df["label"] == 0).sum() for df in splits.values())
        )
    },
    "preprocessing": {
        "outer_whitespace_removed": True,
        "exact_name_error_placeholders_removed": 4,
        "missing_text_records": 0,
        "missing_label_records": 0,
        "duplicate_record_ids": 0,
        "other_records_preserved": True
    },
    "split_methodology": {
        "method": "StratifiedGroupKFold",
        "n_splits": 5,
        "random_state": 42,
        "selected_fold": 1,
        "approximate_train_validation_ratio": "80:20",
        "grouping_column": "post_id",
        "official_test_set_preserved": True
    },
    "leakage_checks": {
        "post_id_overlap": post_overlap,
        "exact_text_overlap": text_overlap
    },
    "data_quality_notes": {
        "zero_confidence_test_records": zero_confidence_test,
        "zero_confidence_handling": (
            "Retained. Confidence excluded from model input and "
            "not used as a sample weight."
        ),
        "repeated_post_ids": (
            "Multiple text segments from the same Reddit post are "
            "preserved as separate labelled records."
        ),
        "identical_text_overlaps": (
            "Three identical texts occur between training and testing "
            "with conflicting labels. Preserved and documented."
        )
    },
    "modeling_plan": {
        "model": "DistilBERT",
        "input_feature": "text",
        "target": "label",
        "output_classes": 2,
        "loss_function": "CrossEntropyLoss",
        "excluded_features": [
            "id",
            "post_id",
            "confidence",
            "precomputed linguistic features",
            "social and metadata features"
        ]
    },
    "intended_use": (
        "Research-level linguistic stress indicator detection. "
        "Not a clinical diagnostic system."
    ),
    "documentation_date": str(date.today())
}

with open(DOCS / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

# --------------------------------------------------
# README DOCUMENTATION
# --------------------------------------------------

readme = f"""
# Dreaddit Dataset Documentation

## 1. Dataset Overview

- Dataset: Dreaddit
- Project: AI-Powered Mental Wellness and Peer Support Chatbot
- Module: Stress Detection
- Task: Binary text classification
- Total final records: {total_records}

The dataset contains Reddit text segments labelled for stress-related
content. It is used to develop a research-level linguistic stress
indicator classifier.

## 2. Dataset Source

Original research dataset:
https://www.cs.columbia.edu/~eturcan/data/dreaddit.zip

The dataset was publicly downloadable during acquisition.

Important: Public accessibility does not automatically establish
an open license. Verify the original publisher's usage and
redistribution terms before redistributing the dataset or its contents.

## 3. Original Dataset Structure

| Split | Original records |
|---|---:|
| Training | 2,838 |
| Testing | 715 |
| Total | 3,553 |

## 4. Preprocessing

Operations performed:

1. Retained the text, label, ID, post ID and confidence fields.
2. Removed leading and trailing whitespace from text.
3. Removed four exact malformed #NAME? placeholder records from training.
4. Preserved all other valid records.
5. Standardized binary labels.
6. Kept repeated post segments as separate records.
7. Excluded metadata and precomputed features from model input.

Final records: {total_records}

## 5. Label Mapping

| Numeric label | Standardized label |
|---|---|
| 0 | non_stress |
| 1 | stress |

The target is the original binary stress label.
The separate sentiment and confidence fields are not target labels.

## 6. Final Dataset Splits

| Split | Records |
|---|---:|
| Training | {len(train)} |
| Validation | {len(validation)} |
| Testing | {len(test)} |
| Total | {total_records} |

Training and validation were generated using
StratifiedGroupKFold with five folds and random state 42.

The official test set was preserved unchanged.

## 7. Group Leakage Verification

Grouping field: post_id

| Comparison | Shared post IDs |
|---|---:|
| Training vs Validation | {post_overlap["train_validation"]} |
| Training vs Testing | {post_overlap["train_test"]} |
| Validation vs Testing | {post_overlap["validation_test"]} |

No post-level leakage was detected.

## 8. Identical Text Overlap

| Comparison | Identical text count |
|---|---:|
| Training vs Validation | {text_overlap["train_validation"]} |
| Training vs Testing | {text_overlap["train_test"]} |
| Validation vs Testing | {text_overlap["validation_test"]} |

Three identical-text overlaps occur between training and testing.
These have conflicting labels and were preserved to avoid modifying
the official test set. This limitation should be considered during
model evaluation.

## 9. Confidence Field Anomaly

The official test data contains {zero_confidence_test} records
with confidence equal to zero.

These records were retained. Confidence was excluded from model
input and was not used for sample weighting.

The confidence field should not be interpreted as a validated
measure of model certainty.

## 10. Model Development Plan

Planned baseline:

- Architecture: DistilBERT
- Input: text
- Target: binary stress label
- Output: two logits
- Loss: CrossEntropyLoss

ID, post_id, confidence and precomputed linguistic/social features
will not be used as model input.

Evaluation will use accuracy, precision, recall, F1-score,
macro-F1 and a confusion matrix.

## 11. Ethical and Research Limitations

- The dataset represents stress-related language in Reddit posts.
- It does not establish a person's clinical stress condition.
- Predictions must be treated as uncertain linguistic indicators.
- The classifier must not independently diagnose users.
- Negative sentiment must not automatically be interpreted as stress.
- Human support and appropriate safety procedures remain necessary.
- Dataset usage must follow the original source's applicable terms.

## 12. Output Files

Final splits:

- datasets/processed/dreaddit/final_splits/train.csv
- datasets/processed/dreaddit/final_splits/validation.csv
- datasets/processed/dreaddit/final_splits/test.csv

Documentation:

- datasets/documentation/dreaddit/README.md
- datasets/documentation/dreaddit/metadata.json
- datasets/documentation/dreaddit/label_distribution.csv
"""

with open(DOCS / "README.md", "w", encoding="utf-8") as f:
    f.write(readme.strip() + "\n")

# --------------------------------------------------
# COMPLETION REPORT
# --------------------------------------------------

print("=" * 65)
print("DREADDIT DOCUMENTATION GENERATED")
print("=" * 65)

print("\nDocumentation directory:")
print(DOCS)

print("\nGenerated files:")

for file in DOCS.iterdir():
    print(file.name, "-", file.stat().st_size, "bytes")

print("\nFinal record count:", total_records)
print("Zero-confidence test records:", zero_confidence_test)
print("Post-level leakage:", post_overlap)
print("Exact text overlaps:", text_overlap)

print("\nALL DOCUMENTATION FILES CREATED")
print("=" * 65)
