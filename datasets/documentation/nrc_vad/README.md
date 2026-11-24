# NRC Valence-Arousal-Dominance (VAD) Lexicon v2.1

## 1. Dataset Overview

The NRC VAD Lexicon is a lexical resource that associates English words and expressions with three continuous dimensions of emotional meaning.

- Valence: pleasantness or unpleasantness
- Arousal: emotional activation or calmness
- Dominance: perceived control or power

Official source: https://saifmohammad.com/WebPages/nrc-vad.html

## 2. Dataset Statistics

- Total valid terms: 54,800
- Unique terms: 54,800
- Unigrams: 44,727
- Multi-word expressions: 10,073
- VAD dimensions: 3
- Score range: -1.0 to +1.0

## 3. Standardized Schema

| Column | Description |
|---|---|
| term | Word or lexical expression |
| term_type | unigram or multi_word |
| valence | Pleasantness/unpleasantness score |
| arousal | Emotional activation score |
| dominance | Perceived control/power score |

## 4. VAD Dimensions

| Dimension | Interpretation |
|---|---|
| Valence | Negative to positive emotional meaning |
| Arousal | Calm to highly activated emotional meaning |
| Dominance | Weak/submissive to powerful/in-control meaning |

The scores are continuous lexical ratings. They should not be converted into arbitrary emotion labels without a separately defined and evaluated method.

## 5. Term Distribution

| Term type | Records | Percentage |
|---|---:|---:|
| Unigram | 44,727 | 81.62% |
| Multi-word expression | 10,073 | 18.38% |
| Total | 54,800 | 100.00% |

The combined lexicon includes both unigrams and multi-word expressions. Separate unigram and MWE source files were not concatenated, avoiding duplicate records.

## 6. Score Statistics

| Dimension | Mean | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| valence | -0.001362 | 0.476609 | -1.000 | 1.000 |
| arousal | 0.018420 | 0.463864 | -1.000 | 1.000 |
| dominance | 0.044617 | 0.414248 | -1.000 | 1.000 |

## 7. Cleaning and Preprocessing

- Original records: 54,801
- Removed records: 1
- Final records: 54,800
- The removed record had a missing term.
- No duplicate terms were found.
- No missing or invalid VAD scores were found.
- All scores were within the range [-1, +1].
- Original term capitalization was preserved.
- Original punctuation and internal spaces were preserved.
- Original numerical score precision was preserved.
- All valid multi-word expressions were retained.

## 8. Quality Verification

- Standardization checks: 7 passed.
- Final verification checks: 31 passed.
- Final verification failures: 0.
- All 54,800 terms matched the cleaned source.
- All three VAD dimensions matched the cleaned source.

## 9. Integration into the Mental Wellness Chatbot

The lexicon supports the project's VAD Analysis module.

Suggested processing workflow:

1. Receive a student's text message.
2. Tokenize the message while preserving phrase matching.
3. Match unigrams and multi-word expressions against the lexicon.
4. Retrieve the corresponding valence, arousal, and dominance scores.
5. Aggregate matched scores into message-level VAD features.
6. Pass these features to the chatbot's analysis and response pipeline.

Multi-word expressions should be matched before their component unigrams where appropriate, to avoid double-counting the same phrase. The aggregation and phrase-matching strategy will be designed and evaluated during implementation.

The lexicon provides linguistic indicators, not direct measurements of a student's internal emotional state.

## 10. Limitations

- Lexical scores do not fully represent sentence context.
- Negation can reverse or modify the meaning of a term.
- Sarcasm and figurative language may be difficult to interpret.
- Domain-specific and slang expressions may be absent.
- A matched word does not establish a user's actual emotional state.
- VAD outputs must not be used as clinical diagnoses.


## 11. License and Usage

The NRC VAD Lexicon v2.1 is subject to the usage conditions specified by its original publisher, the National Research Council Canada (NRC).

According to the official Terms of Use:

- **Research and Educational Use:** Permitted for non-commercial research and educational purposes.
- **Citation:** Users must cite the research papers associated with the lexicon.
- **Application Attribution:** Applications using the lexicon must acknowledge the resource, its creator, and the National Research Council Canada in their About page and relevant documentation.
- **Redistribution:** Redistribution of the original lexicon data is prohibited. Interested users must be directed to the official lexicon webpage.
- **Commercial Use:** Separate permission must be obtained from the author for commercial use.

Official Terms of Use and Source:
https://saifmohammad.com/WebPages/nrc-vad.html

Official Publisher: National Research Council Canada (NRC)

Creator: Dr. Saif M. Mohammad

### Required Research Citations

1. Mohammad, Saif M. (2025). NRC VAD Lexicon v2: Norms for Valence, Arousal, and Dominance for over 55k English Terms.
https://arxiv.org/abs/2503.23547

2. Mohammad, Saif M. (2018). Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words. Proceedings of ACL 2018.

**Project Compliance:** The original NRC VAD lexicon files must not be redistributed through this project repository. Users should obtain the lexicon directly from its official source and follow the publisher's usage conditions.


## 12. File Organization

```text
datasets/
├── raw/
│   └── nrc_vad/
│       ├── nrc_vad_v2.1.zip
│       └── extracted/
├── processed/
│   └── nrc_vad/
│       ├── cleaned/
│       │   └── nrc_vad_cleaned.csv
│       └── standardized/
│           ├── nrc_vad.csv
│           └── metadata.json
└── documentation/
    └── nrc_vad/
        ├── README.md
        ├── metadata.json
        ├── term_distribution.csv
        └── score_statistics.csv
```

Documentation generated by the project preprocessing pipeline.
