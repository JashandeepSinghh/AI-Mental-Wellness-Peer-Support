
import json
from pathlib import Path

import pandas as pd


# ============================================================
# NRC VAD LEXICON V2.1 - DOCUMENTATION GENERATOR
# ============================================================

ROOT = Path("datasets/processed/nrc_vad")

DATA_FILE = ROOT / "standardized/nrc_vad.csv"
SOURCE_METADATA_FILE = ROOT / "standardized/metadata.json"

OUTPUT_DIR = Path("datasets/documentation/nrc_vad")

SCORE_COLUMNS = ["valence", "arousal", "dominance"]


def load_dataset():
    """Load standardized lexicon and existing metadata."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    if not SOURCE_METADATA_FILE.exists():
        raise FileNotFoundError(SOURCE_METADATA_FILE)

    df = pd.read_csv(
        DATA_FILE,
        encoding="utf-8",
        dtype={"term": "string"}
    )

    with open(
        SOURCE_METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        source_metadata = json.load(file)

    return df, source_metadata


def create_term_distribution(df):
    """Create term-type count and percentage report."""

    counts = df["term_type"].value_counts()

    total = len(df)

    rows = []

    for term_type in ["unigram", "multi_word"]:
        count = int(counts.get(term_type, 0))

        rows.append({
            "term_type": term_type,
            "count": count,
            "percentage": round(count / total * 100, 2)
        })

    rows.append({
        "term_type": "total",
        "count": total,
        "percentage": 100.0
    })

    return pd.DataFrame(rows)


def create_score_statistics(df):
    """Create descriptive statistics for VAD dimensions."""

    statistics = (
        df[SCORE_COLUMNS]
        .describe()
        .transpose()
        .reset_index()
        .rename(columns={"index": "dimension"})
    )

    statistics.insert(
        1,
        "score_range_min",
        [
            float(df[column].min())
            for column in SCORE_COLUMNS
        ]
    )

    statistics.insert(
        2,
        "score_range_max",
        [
            float(df[column].max())
            for column in SCORE_COLUMNS
        ]
    )

    statistics = statistics.rename(columns={
        "count": "record_count",
        "mean": "mean",
        "std": "standard_deviation",
        "min": "minimum",
        "25%": "first_quartile",
        "50%": "median",
        "75%": "third_quartile",
        "max": "maximum"
    })

    return statistics.round(6)


def create_metadata(df, source_metadata):
    """Generate complete structured dataset metadata."""

    term_counts = (
        df["term_type"]
        .value_counts()
        .to_dict()
    )

    metadata = {
        "dataset_name": "NRC Valence-Arousal-Dominance Lexicon",
        "short_name": "NRC VAD",
        "version": "2.1",
        "source": (
            "https://saifmohammad.com/WebPages/nrc-vad.html"
        ),
        "original_source_file": (
            "NRC-VAD-Lexicon-v2.1.txt"
        ),
        "license": "CC BY-NC 4.0",
        "license_url": (
            "https://creativecommons.org/licenses/by-nc/4.0/"
        ),
        "usage_restriction": (
            "Non-commercial research and educational use with "
            "appropriate attribution. Do not redistribute the "
            "lexicon files. Commercial use requires separate permission."
        ),
        "total_records": int(len(df)),
        "unique_terms": int(df["term"].nunique()),
        "term_distribution": {
            "unigram": int(term_counts.get("unigram", 0)),
            "multi_word": int(term_counts.get("multi_word", 0))
        },
        "columns": [
            "term",
            "term_type",
            "valence",
            "arousal",
            "dominance"
        ],
        "score_dimensions": {
            "valence": {
                "description": "Pleasantness or unpleasantness",
                "minimum": float(df["valence"].min()),
                "maximum": float(df["valence"].max()),
                "mean": float(df["valence"].mean())
            },
            "arousal": {
                "description": "Emotional activation or calmness",
                "minimum": float(df["arousal"].min()),
                "maximum": float(df["arousal"].max()),
                "mean": float(df["arousal"].mean())
            },
            "dominance": {
                "description": "Perceived control or power",
                "minimum": float(df["dominance"].min()),
                "maximum": float(df["dominance"].max()),
                "mean": float(df["dominance"].mean())
            }
        },
        "score_scale": {
            "minimum": -1.0,
            "maximum": 1.0
        },
        "project_module": "Valence-Arousal-Dominance Analysis",
        "resource_type": "Lexicon-based NLP resource",
        "model_input": "term",
        "model_outputs": [
            "valence",
            "arousal",
            "dominance"
        ],
        "preprocessing": {
            "original_records": 54801,
            "removed_records": 1,
            "final_records": int(len(df)),
            "removal_reason": (
                "One record contained a missing term."
            ),
            "duplicate_terms": 0,
            "missing_scores": 0,
            "score_range_violations": 0,
            "capitalization_preserved": True,
            "punctuation_preserved": True,
            "multi_word_expressions_preserved": True,
            "score_precision_preserved": True
        },
        "quality_verification": {
            "standardization_checks_passed": 7,
            "final_verification_checks_passed": 31,
            "final_verification_checks_failed": 0
        },
        "training_requirement": (
            "This is a lexicon resource, not a supervised training dataset."
        ),
        "source_metadata": source_metadata
    }

    return metadata


def create_readme(df, metadata, distribution, statistics):
    """Generate human-readable documentation."""

    unigram_count = metadata["term_distribution"]["unigram"]
    multi_word_count = metadata["term_distribution"]["multi_word"]

    lines = [
        "# NRC Valence-Arousal-Dominance (VAD) Lexicon v2.1",
        "",
        "## 1. Dataset Overview",
        "",
        "The NRC VAD Lexicon is a lexical resource that associates "
        "English words and expressions with three continuous "
        "dimensions of emotional meaning.",
        "",
        "- Valence: pleasantness or unpleasantness",
        "- Arousal: emotional activation or calmness",
        "- Dominance: perceived control or power",
        "",
        "Official source: "
        "https://saifmohammad.com/WebPages/nrc-vad.html",
        "",
        "## 2. Dataset Statistics",
        "",
        f"- Total valid terms: {len(df):,}",
        f"- Unique terms: {df['term'].nunique():,}",
        f"- Unigrams: {unigram_count:,}",
        f"- Multi-word expressions: {multi_word_count:,}",
        "- VAD dimensions: 3",
        "- Score range: -1.0 to +1.0",
        "",
        "## 3. Standardized Schema",
        "",
        "| Column | Description |",
        "|---|---|",
        "| term | Word or lexical expression |",
        "| term_type | unigram or multi_word |",
        "| valence | Pleasantness/unpleasantness score |",
        "| arousal | Emotional activation score |",
        "| dominance | Perceived control/power score |",
        "",
        "## 4. VAD Dimensions",
        "",
        "| Dimension | Interpretation |",
        "|---|---|",
        "| Valence | Negative to positive emotional meaning |",
        "| Arousal | Calm to highly activated emotional meaning |",
        "| Dominance | Weak/submissive to powerful/in-control meaning |",
        "",
        "The scores are continuous lexical ratings. They should not "
        "be converted into arbitrary emotion labels without a separately "
        "defined and evaluated method.",
        "",
        "## 5. Term Distribution",
        "",
        "| Term type | Records | Percentage |",
        "|---|---:|---:|",
        f"| Unigram | {unigram_count:,} | "
        f"{unigram_count / len(df) * 100:.2f}% |",
        f"| Multi-word expression | {multi_word_count:,} | "
        f"{multi_word_count / len(df) * 100:.2f}% |",
        f"| Total | {len(df):,} | 100.00% |",
        "",
        "The combined lexicon includes both unigrams and multi-word "
        "expressions. Separate unigram and MWE source files were not "
        "concatenated, avoiding duplicate records.",
        "",
        "## 6. Score Statistics",
        "",
        "| Dimension | Mean | Standard deviation | Minimum | Maximum |",
        "|---|---:|---:|---:|---:|"
    ]

    for _, row in statistics.iterrows():
        lines.append(
            f"| {row['dimension']} | {row['mean']:.6f} | "
            f"{row['standard_deviation']:.6f} | "
            f"{row['minimum']:.3f} | {row['maximum']:.3f} |"
        )

    lines.extend([
        "",
        "## 7. Cleaning and Preprocessing",
        "",
        "- Original records: 54,801",
        "- Removed records: 1",
        "- Final records: 54,800",
        "- The removed record had a missing term.",
        "- No duplicate terms were found.",
        "- No missing or invalid VAD scores were found.",
        "- All scores were within the range [-1, +1].",
        "- Original term capitalization was preserved.",
        "- Original punctuation and internal spaces were preserved.",
        "- Original numerical score precision was preserved.",
        "- All valid multi-word expressions were retained.",
        "",
        "## 8. Quality Verification",
        "",
        "- Standardization checks: 7 passed.",
        "- Final verification checks: 31 passed.",
        "- Final verification failures: 0.",
        "- All 54,800 terms matched the cleaned source.",
        "- All three VAD dimensions matched the cleaned source.",
        "",
        "## 9. Integration into the Mental Wellness Chatbot",
        "",
        "The lexicon supports the project's VAD Analysis module.",
        "",
        "Suggested processing workflow:",
        "",
        "1. Receive a student's text message.",
        "2. Tokenize the message while preserving phrase matching.",
        "3. Match unigrams and multi-word expressions against the lexicon.",
        "4. Retrieve the corresponding valence, arousal, and dominance scores.",
        "5. Aggregate matched scores into message-level VAD features.",
        "6. Pass these features to the chatbot's analysis and response pipeline.",
        "",
        "Multi-word expressions should be matched before their component "
        "unigrams where appropriate, to avoid double-counting the same "
        "phrase. The aggregation and phrase-matching strategy will be "
        "designed and evaluated during implementation.",
        "",
        "The lexicon provides linguistic indicators, not direct "
        "measurements of a student's internal emotional state.",
        "",
        "## 10. Limitations",
        "",
        "- Lexical scores do not fully represent sentence context.",
        "- Negation can reverse or modify the meaning of a term.",
        "- Sarcasm and figurative language may be difficult to interpret.",
        "- Domain-specific and slang expressions may be absent.",
        "- A matched word does not establish a user's actual emotional state.",
        "- VAD outputs must not be used as clinical diagnoses.",
        "",
        "## 11. License and Usage",
        "",
        "License information recorded for this project: CC BY-NC 4.0.",
        "Non-commercial research and educational use with attribution. "
        "Do not redistribute the lexicon files. Commercial use requires "
        "separate permission.",
        "",
        "License: https://creativecommons.org/licenses/by-nc/4.0/",
        "",
        "Users of the lexicon should cite the original NRC VAD publication "
        "and follow the official source's attribution and usage conditions.",
        "",
        "## 12. File Organization",
        "",
        "```text",
        "datasets/",
        "├── raw/",
        "│   └── nrc_vad/",
        "│       ├── nrc_vad_v2.1.zip",
        "│       └── extracted/",
        "├── processed/",
        "│   └── nrc_vad/",
        "│       ├── cleaned/",
        "│       │   └── nrc_vad_cleaned.csv",
        "│       └── standardized/",
        "│           ├── nrc_vad.csv",
        "│           └── metadata.json",
        "└── documentation/",
        "    └── nrc_vad/",
        "        ├── README.md",
        "        ├── metadata.json",
        "        ├── term_distribution.csv",
        "        └── score_statistics.csv",
        "```",
        "",
        "Documentation generated by the project preprocessing pipeline.",
        ""
    ])

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("NRC VAD V2.1 - DOCUMENTATION GENERATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, source_metadata = load_dataset()

    distribution = create_term_distribution(df)
    statistics = create_score_statistics(df)
    metadata = create_metadata(df, source_metadata)
    readme = create_readme(
        df,
        metadata,
        distribution,
        statistics
    )

    # --------------------------------------------------------
    # Save documentation files
    # --------------------------------------------------------

    readme_path = OUTPUT_DIR / "README.md"
    metadata_path = OUTPUT_DIR / "metadata.json"
    distribution_path = OUTPUT_DIR / "term_distribution.csv"
    statistics_path = OUTPUT_DIR / "score_statistics.csv"

    readme_path.write_text(readme, encoding="utf-8")

    metadata_path.write_text(
        json.dumps(metadata, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    distribution.to_csv(
        distribution_path,
        index=False,
        encoding="utf-8"
    )

    statistics.to_csv(
        statistics_path,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Verify generated files
    # --------------------------------------------------------

    outputs = [
        readme_path,
        metadata_path,
        distribution_path,
        statistics_path
    ]

    print("\nGENERATED DOCUMENTATION FILES")

    for path in outputs:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(
                f"Documentation file missing or empty: {path}"
            )

        print(f"PASS: {path}")

    # Verify metadata and CSV summaries.
    saved_metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    saved_distribution = pd.read_csv(distribution_path)
    saved_statistics = pd.read_csv(statistics_path)

    if saved_metadata["total_records"] != len(df):
        raise RuntimeError("Documentation metadata count mismatch.")

    if int(saved_distribution.iloc[-1]["count"]) != len(df):
        raise RuntimeError("Term distribution total mismatch.")

    if len(saved_statistics) != 3:
        raise RuntimeError("VAD statistics must contain three dimensions.")

    print("\nDATASET DOCUMENTATION SUMMARY")
    print(f"Total terms: {len(df):,}")
    print(f"Unigrams: {metadata['term_distribution']['unigram']:,}")
    print(f"Multi-word expressions: {metadata['term_distribution']['multi_word']:,}")

    print("\nALL NRC VAD DOCUMENTATION FILES GENERATED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
