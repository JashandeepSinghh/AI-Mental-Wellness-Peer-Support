
import json
from pathlib import Path

import pandas as pd


# ============================================================
# NRC VAD LEXICON V2.1 - STANDARDIZATION
# ============================================================

ROOT = Path("datasets/processed/nrc_vad")

INPUT_FILE = ROOT / "cleaned/nrc_vad_cleaned.csv"

OUTPUT_DIR = ROOT / "standardized"

OUTPUT_FILE = OUTPUT_DIR / "nrc_vad.csv"
MAPPING_FILE = OUTPUT_DIR / "metadata.json"

EXPECTED_RECORDS = 54800

SCORE_COLUMNS = [
    "valence",
    "arousal",
    "dominance"
]


def main():
    print("=" * 65)
    print("NRC VAD LEXICON V2.1 - STANDARDIZATION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load cleaned dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8",
        dtype={"term": "string"}
    )

    print(f"\nOriginal cleaned records: {len(df):,}")

    # --------------------------------------------------------
    # 2. Verify source structure
    # --------------------------------------------------------

    expected_columns = [
        "term",
        "valence",
        "arousal",
        "dominance"
    ]

    if list(df.columns) != expected_columns:
        raise ValueError(
            f"Unexpected columns: {df.columns.tolist()}"
        )

    if len(df) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS} records, found {len(df)}"
        )

    # --------------------------------------------------------
    # 3. Classify terms
    # --------------------------------------------------------

    df["term_type"] = df["term"].str.split().str.len().map(
        lambda count: "unigram" if count == 1 else "multi_word"
    )

    # --------------------------------------------------------
    # 4. Set standardized column order
    # --------------------------------------------------------

    df = df[
        [
            "term",
            "term_type",
            "valence",
            "arousal",
            "dominance"
        ]
    ]

    # --------------------------------------------------------
    # 5. Verify standardized data
    # --------------------------------------------------------

    checks = {
        "expected record count": len(df) == EXPECTED_RECORDS,
        "no missing values": not df.isnull().any().any(),
        "all terms unique": df["term"].is_unique,
        "all terms non-empty": df["term"].str.strip().ne("").all(),
        "valid term types": set(df["term_type"].unique()).issubset(
            {"unigram", "multi_word"}
        ),
        "all VAD scores numeric": all(
            pd.api.types.is_numeric_dtype(df[column])
            for column in SCORE_COLUMNS
        ),
        "VAD scores within [-1, 1]": (
            (
                (df[SCORE_COLUMNS] >= -1)
                & (df[SCORE_COLUMNS] <= 1)
            ).all().all()
        ),
    }

    print("\nSTANDARDIZATION CHECKS")

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    if not all(checks.values()):
        raise RuntimeError(
            "NRC VAD standardization verification failed."
        )

    # --------------------------------------------------------
    # 6. Generate term statistics
    # --------------------------------------------------------

    term_distribution = (
        df["term_type"]
        .value_counts()
        .to_dict()
    )

    # --------------------------------------------------------
    # 7. Save standardized dataset
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 8. Save metadata
    # --------------------------------------------------------

    metadata = {
        "dataset_name": "NRC VAD Lexicon",
        "version": "2.1",
        "total_records": len(df),
        "unique_terms": int(df["term"].nunique()),
        "columns": [
            "term",
            "term_type",
            "valence",
            "arousal",
            "dominance"
        ],
        "term_types": {
            "unigram": int(term_distribution.get("unigram", 0)),
            "multi_word": int(term_distribution.get("multi_word", 0))
        },
        "score_dimensions": {
            "valence": {
                "minimum": float(df["valence"].min()),
                "maximum": float(df["valence"].max())
            },
            "arousal": {
                "minimum": float(df["arousal"].min()),
                "maximum": float(df["arousal"].max())
            },
            "dominance": {
                "minimum": float(df["dominance"].min()),
                "maximum": float(df["dominance"].max())
            }
        },
        "model_input": "term",
        "output_dimensions": [
            "valence",
            "arousal",
            "dominance"
        ],
        "resource_type": "Lexicon-based VAD analysis",
        "license": "CC BY-NC 4.0",
        "usage": "Non-commercial research and educational use with attribution"
    }

    with open(
        MAPPING_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # 9. Verify saved dataset
    # --------------------------------------------------------

    saved = pd.read_csv(
        OUTPUT_FILE,
        encoding="utf-8",
        dtype={"term": "string"}
    )

    saved_metadata = json.loads(
        MAPPING_FILE.read_text(encoding="utf-8")
    )

    if len(saved) != EXPECTED_RECORDS:
        raise RuntimeError("Saved record count mismatch.")

    if saved["term"].tolist() != df["term"].tolist():
        raise RuntimeError("Saved terms differ from standardized terms.")

    for column in SCORE_COLUMNS:
        if saved[column].tolist() != df[column].tolist():
            raise RuntimeError(
                f"Saved {column} values do not match."
            )

    if saved_metadata["total_records"] != EXPECTED_RECORDS:
        raise RuntimeError("Saved metadata record count mismatch.")

    # --------------------------------------------------------
    # 10. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("STANDARDIZATION SUMMARY")
    print("=" * 65)

    print(f"Total records: {len(df):,}")
    print(f"Unique terms: {df['term'].nunique():,}")

    print("\nTerm type distribution:")
    print(df["term_type"].value_counts().to_string())

    print("\nVAD score ranges:")

    for column in SCORE_COLUMNS:
        print(
            f"{column}: "
            f"{df[column].min()} to {df[column].max()}"
        )

    print(f"\nDataset output: {OUTPUT_FILE.resolve()}")
    print(f"Metadata output: {MAPPING_FILE.resolve()}")

    print("\nSaved dataset verification: PASS")
    print("\nALL NRC VAD STANDARDIZATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
