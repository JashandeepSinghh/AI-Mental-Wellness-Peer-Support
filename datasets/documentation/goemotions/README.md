# GoEmotions Dataset Documentation

## 1. Dataset Overview

- Dataset: GoEmotions
- Project: AI-Powered Mental Wellness and Peer Support Chatbot
- Task: Multi-label emotion classification
- Emotion categories: 28
- Total records: 54,263

## 2. Dataset Splits

| Split | Records |
|---|---:|
| Training | 43,410 |
| Validation | 5,426 |
| Testing | 5,427 |

## 3. Emotion Categories

0. admiration
1. amusement
2. anger
3. annoyance
4. approval
5. caring
6. confusion
7. curiosity
8. desire
9. disappointment
10. disapproval
11. disgust
12. embarrassment
13. excitement
14. fear
15. gratitude
16. grief
17. joy
18. love
19. nervousness
20. optimism
21. pride
22. realization
23. relief
24. remorse
25. sadness
26. surprise
27. neutral


## 4. Dataset Format

Each processed CSV contains:

- text: Original comment text.
- comment_id: Original comment identifier.
- 28 emotion columns: Binary multi-hot labels.

A value of 1 indicates an assigned emotion.
A value of 0 indicates that the emotion is not assigned.

## 5. Preparation Performed

- Official train, validation and test splits retained.
- Numeric emotion IDs converted into emotion names.
- Multi-label annotations converted into binary vectors.
- Missing and empty text checked.
- Missing comment IDs checked.
- Duplicate comment IDs checked.
- Records without emotion labels checked.
- Original text preserved without normalization.


## 6. Duplicate Text Across Official Splits

Identical normalized text counts:

- training_vs_validation: 46
- training_vs_testing: 37
- validation_vs_testing: 11


These are text overlaps, not comment ID overlaps.
They were preserved to retain the official benchmark splits.

## 7. Known Limitations

- Some identical normalized texts occur across official splits under different comment IDs.
- Duplicate text records were preserved to maintain official benchmark splits.
- Emotion labels represent dataset annotations and are not clinical diagnoses.
- Dataset annotations may contain subjectivity and class imbalance.


## 8. License and Usage

The GoEmotions dataset is distributed through the official Google Research repository, which is licensed under the Apache License 2.0.

**Official Repository:**
https://github.com/google-research/google-research/tree/master/goemotions

**Repository License:**
Apache License 2.0

**License Reference:**
https://github.com/google-research/google-research/blob/master/LICENSE

### Usage and Redistribution Considerations

- The Apache License 2.0 governs the materials covered by the official repository license.
- The dataset consists of Reddit comments, which originate from third-party user-generated content.
- The repository license should not be interpreted as transferring copyright ownership of individual Reddit comments.
- Users should review applicable Reddit terms and the original dataset documentation before redistribution or other reuse.
- Appropriate attribution and citation of the original research are required.

### Required Research Citation

Demszky, Dorottya, et al. (2020).
GoEmotions: A Dataset of Fine-Grained Emotions.
Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics.

https://arxiv.org/abs/2005.00547

**Project Compliance:** The project will retain dataset source information and attribution. Redistribution of dataset files will be evaluated separately against applicable source and content rights.


## 9. Project Usage

GoEmotions will be used for the project's emotion
classification module. The model will predict multiple
emotion categories from a student's text input.

These predictions are supportive language indicators,
not clinical diagnoses.

## 10. Documentation Timestamp

2026-11-20T05:21:07
