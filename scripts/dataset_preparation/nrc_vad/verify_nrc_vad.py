
import json
from pathlib import Path

import pandas as pd


# ============================================================
# NRC VAD V2.1 - FINAL STANDARDIZED DATA VERIFICATION
# ============================================================

ROOT = Path("datasets/processed/nrc_vad")

CLEANED_FILE = ROOT / "cleaned/nrc_vad_cleaned.csv"
STANDARDIZED_FILE = ROOT / "standardized/nrc_vad.csv"
METADATA_FILE = ROOT / "standardized/metadata.json"

SCORE_COLUMNS = ["valence", "arousal", "dominance"]

EXPECTED_RECORDS = 54800

EXPECTED_COLUMNS = [
    "term",
    "term_type",
    "valence",
    "arousal",
    "dominance"
]


def load_data():
    cleaned = pd.read_csv(
        CLEANED_FILE,
        encoding="utf-8",
        dtype={"term": "string"}
    )

    standardized = pd.read_csv(
        STANDARDIZED_FILE,
        encoding="utf-8",
        dtype={"term": "string"}
    )

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return cleaned, standardized, metadata


def main():
    print("=" * 70)
    print("NRC VAD V2.1 - FINAL STANDARDIZED DATA VERIFICATION")
    print("=" * 70)

    cleaned, standardized, metadata = load_data()

    checks = {}

    # --------------------------------------------------------
    # 1. Record count verification
    # --------------------------------------------------------

    checks["expected record count"] = (
        len(standardized) == EXPECTED_RECORDS
    )

    checks["cleaned-standardized count match"] = (
        len(cleaned) == len(standardized)
    )

    # --------------------------------------------------------
    # 2. Column structure verification
    # --------------------------------------------------------

    checks["standardized column structure"] = (
        standardized.columns.tolist() == EXPECTED_COLUMNS
    )

    # --------------------------------------------------------
    # 3. Missing value verification
    # --------------------------------------------------------

    checks["no missing values"] = (
        not standardized.isnull().any().any()
    )

    checks["no empty terms"] = (
        standardized["term"].str.strip().ne("").all()
    )

    # --------------------------------------------------------
    # 4. Term preservation
    # --------------------------------------------------------

    checks["all terms preserved"] = (
        cleaned["term"].tolist()
        == standardized["term"].tolist()
    )

    checks["all terms unique"] = (
        standardized["term"].is_unique
    )

    # --------------------------------------------------------
    # 5. VAD score preservation
    # --------------------------------------------------------

    for column in SCORE_COLUMNS:
        checks[f"{column} scores preserved"] = (
            cleaned[column].tolist()
            == standardized[column].tolist()
        )

        checks[f"{column} scores numeric"] = (
            pd.api.types.is_numeric_dtype(
                standardized[column]
            )
        )

        checks[f"{column} scores within [-1, 1]"] = (
            standardized[column].between(-1, 1).all()
        )

    # --------------------------------------------------------
    # 6. Term type verification
    # --------------------------------------------------------

    expected_term_types = (
        cleaned["term"]
        .str.split()
        .str.len()
        .map(
            lambda count:
            "unigram" if count == 1 else "multi_word"
        )
    )

    checks["term types correctly assigned"] = (
        standardized["term_type"].tolist()
        == expected_term_types.tolist()
    )

    checks["only valid term types"] = (
        set(standardized["term_type"].unique())
        == {"unigram", "multi_word"}
    )

    # --------------------------------------------------------
    # 7. Metadata verification
    # --------------------------------------------------------

    checks["metadata record count correct"] = (
        metadata.get("total_records") == EXPECTED_RECORDS
    )

    checks["metadata unique term count correct"] = (
        metadata.get("unique_terms") == EXPECTED_RECORDS
    )

    checks["metadata columns correct"] = (
        metadata.get("columns") == EXPECTED_COLUMNS
    )

    checks["metadata model input correct"] = (
        metadata.get("model_input") == "term"
    )

    checks["metadata VAD dimensions correct"] = (
        metadata.get("output_dimensions") == SCORE_COLUMNS
    )

    # --------------------------------------------------------
    # 8. Label-free lexicon statistics
    # --------------------------------------------------------

    unigram_count = int(
        (standardized["term_type"] == "unigram").sum()
    )

    multi_word_count = int(
        (standardized["term_type"] == "multi_word").sum()
    )

    checks["unigram count correct"] = unigram_count == 44727
    checks["multi-word count correct"] = multi_word_count == 10073

    # --------------------------------------------------------
    # 9. Cross-check score ranges against metadata
    # --------------------------------------------------------

    for column in SCORE_COLUMNS:
        expected_range = metadata["score_dimensions"][column]

        checks[f"{column} metadata minimum correct"] = (
            float(standardized[column].min())
            == expected_range["minimum"]
        )

        checks[f"{column} metadata maximum correct"] = (
            float(standardized[column].max())
            == expected_range["maximum"]
        )

    # --------------------------------------------------------
    # 10. Display individual verification results
    # --------------------------------------------------------

    print("\nINDIVIDUAL VERIFICATION RESULTS")

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    # --------------------------------------------------------
    # 11. Display dataset summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STANDARDIZED DATASET SUMMARY")
    print("=" * 70)

    print(f"Total records: {len(standardized):,}")
    print(f"Unique terms: {standardized['term'].nunique():,}")
    print(f"Unigrams: {unigram_count:,}")
    print(f"Multi-word expressions: {multi_word_count:,}")

    print("\nVAD SCORE STATISTICS")

    print(
        standardized[SCORE_COLUMNS]
        .describe()
        .round(4)
        .to_string()
    )

    # --------------------------------------------------------
    # 12. Final result
    # --------------------------------------------------------

    failed = [
        name for name, passed in checks.items()
        if not passed
    ]

    print("\n" + "=" * 70)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 70)

    print(f"Total checks: {len(checks)}")
    print(f"Passed: {sum(checks.values())}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFAILED CHECKS:")

        for item in failed:
            print(f"- {item}")

        raise SystemExit(
            "\nNRC VAD STANDARDIZED VERIFICATION FAILED."
        )

    print("\nALL NRC VAD VERIFICATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
