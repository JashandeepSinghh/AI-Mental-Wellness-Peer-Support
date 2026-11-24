# Dreaddit Dataset Documentation

## 1. Dataset Overview

- Dataset: Dreaddit
- Project: AI-Powered Mental Wellness and Peer Support Chatbot
- Module: Stress Detection
- Task: Binary text classification
- Total final records: 3549

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

Final records: 3549

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
| Training | 2266 |
| Validation | 568 |
| Testing | 715 |
| Total | 3549 |

Training and validation were generated using
StratifiedGroupKFold with five folds and random state 42.

The official test set was preserved unchanged.

## 7. Group Leakage Verification

Grouping field: post_id

| Comparison | Shared post IDs |
|---|---:|
| Training vs Validation | 0 |
| Training vs Testing | 0 |
| Validation vs Testing | 0 |

No post-level leakage was detected.

## 8. Identical Text Overlap

| Comparison | Identical text count |
|---|---:|
| Training vs Validation | 3 |
| Training vs Testing | 3 |
| Validation vs Testing | 0 |

Three identical-text overlaps occur between training and testing.
These have conflicting labels and were preserved to avoid modifying
the official test set. This limitation should be considered during
model evaluation.

## 9. Confidence Field Anomaly

The official test data contains 98 records
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


## 12. License and Usage

The Dreaddit dataset was originally published by Elsbeth Turcan and Kathy McKeown and is publicly downloadable from the original Columbia University research webpage.

**Original Dataset Source:**
https://www.cs.columbia.edu/~eturcan/data/dreaddit.zip

**Original Research Publication:**

Turcan, Elsbeth, and Kathy McKeown (2019).
Dreaddit: A Reddit Dataset for Stress Analysis in Social Media.
Proceedings of the Tenth International Workshop on Health Text Mining and Information Analysis (LOUHI 2019).

https://aclanthology.org/D19-6213/

### License Status

An explicit open-source license or dataset redistribution license has not been verified from the available original-source documentation.

Public availability of the dataset does not automatically grant permission to redistribute its contents.

### Usage and Redistribution Considerations

- Retain attribution to the original dataset creators and research publication.
- Follow applicable Reddit data policies and the original source's terms.
- Do not assume that public download grants unrestricted redistribution rights.
- Obtain clarification or permission from the original creators or relevant rights holders before redistributing the original or processed dataset files.

### Project Compliance

The dataset is used for non-commercial academic and educational development of the project's stress detection module.

The original dataset files and derived text-containing dataset files will remain outside the public Git repository until their redistribution permissions have been established.

Users should obtain the original dataset from its official source and review the applicable terms.


## 12. Output Files

Final splits:

- datasets/processed/dreaddit/final_splits/train.csv
- datasets/processed/dreaddit/final_splits/validation.csv
- datasets/processed/dreaddit/final_splits/test.csv

Documentation:

- datasets/documentation/dreaddit/README.md
- datasets/documentation/dreaddit/metadata.json
- datasets/documentation/dreaddit/label_distribution.csv
