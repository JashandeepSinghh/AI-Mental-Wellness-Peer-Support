# TweetEval Sentiment Dataset Documentation

## 1. Dataset Overview

- Dataset: TweetEval Sentiment
- Project: AI-Powered Mental Wellness and Peer Support Chatbot
- Task: Three-class sentiment classification
- Total records: 59,899
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
| Training | 45,615 |
| Validation | 2,000 |
| Testing | 12,284 |


## 4. Class Distribution

| Split | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Training | 7,093 | 20,673 | 17,849 |
| Validation | 312 | 869 | 819 |
| Testing | 3,972 | 5,937 | 2,375 |


## 5. Processed Dataset Format

Each CSV contains:

- text: Cleaned tweet text.
- label: Integer sentiment class.
- sentiment: Corresponding sentiment name.

## 6. Preparation Performed

- Official training, validation and testing splits retained.
- Leading and trailing whitespace removed using strip().
- Internal text content preserved.
- Numeric sentiment labels preserved.
- Numeric labels mapped to sentiment names.
- Missing and empty text checked.
- Invalid labels checked.
- Duplicate records preserved.


## 7. Duplicate Text Analysis

- Training: 29 duplicate text records
- Validation: 0 duplicate text records
- Testing: 0 duplicate text records


Three duplicate text groups in the training split have
conflicting sentiment annotations. All duplicate records
were preserved.

## 8. Cross-Split Text Overlap

- training_vs_validation: 0
- training_vs_testing: 0
- validation_vs_testing: 0


## 9. Known Limitations

- Training data contains duplicate tweet records.
- Three duplicate text groups have conflicting sentiment annotations.
- The official test distribution differs from the training distribution.
- Sentiment annotations are not clinical or psychological diagnoses.

\
## 10. License and Usage

The TweetEval Sentiment dataset is identified as being distributed under the Creative Commons Attribution 3.0 Unported License (CC BY 3.0).

**Dataset:** TweetEval Sentiment

**License:** Creative Commons Attribution 3.0 Unported (CC BY 3.0)

**Official TweetEval Repository:**
https://github.com/cardiffnlp/tweeteval

**License Reference:**
https://creativecommons.org/licenses/by/3.0/

### Usage Conditions

- **Attribution:** Credit the original dataset creators and relevant research publications.
- **Twitter Terms:** Follow applicable Twitter Terms of Service and Twitter API Terms of Service.
- **Redistribution:** Any redistribution must comply with the applicable dataset license and underlying Twitter content requirements.
- **Source Content:** The dataset contains user-generated Twitter content. The CC BY 3.0 designation should not be interpreted as overriding separate platform terms or third-party rights.

### Required Research Citations

1. Barbieri, Francesco, et al. (2020).
TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification.
Findings of EMNLP 2020.

https://aclanthology.org/2020.findings-emnlp.148/

2. Rosenthal, Sara, Farra, Noura, and Nakov, Preslav (2017).
SemEval-2017 Task 4: Sentiment Analysis in Twitter.
Proceedings of the 11th International Workshop on Semantic Evaluation.

### Project Compliance

This dataset is used for academic and educational development of the project's sentiment analysis module.

The project will retain appropriate attribution and follow applicable dataset and Twitter content requirements.


## 11. Project Application

TweetEval will support the sentiment analysis module.
The classifier will predict negative, neutral or positive
sentiment from student text.

Sentiment predictions are linguistic indicators and
must not independently determine mental health risk.

## 12. Documentation Timestamp

2026-11-20T05:28:57
