
from pathlib import Path

import pandas as pd


# ============================================================
# NRC VAD LEXICON V2.1 - CLEANING
# ============================================================

ROOT = Path(
    "datasets/raw/nrc_vad/extracted/NRC-VAD-Lexicon-v2.1"
)

SOURCE_FILE = ROOT / "NRC-VAD-Lexicon-v2.1.txt"

OUTPUT_DIR = Path("datasets/processed/nrc_vad/cleaned")
OUTPUT_FILE = OUTPUT_DIR / "nrc_vad_cleaned.csv"

SCORE_COLUMNS = ["valence", "arousal", "dominance"]
EXPECTED_RECORDS = 54800


def load_source():
    """Load the original combined NRC VAD lexicon."""

    df = pd.read_csv(
        SOURCE_FILE,
        sep="\t",
        encoding="utf-8-sig",
        dtype={"term": "string"},
        keep_default_na=True
    )

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    expected_columns = ["term"] + SCORE_COLUMNS

    if list(df.columns) != expected_columns:
        raise ValueError(
            f"Unexpected source columns: {df.columns.tolist()}"
        )

    for column in SCORE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


def clean_data(df):
    """Remove only records without a valid term."""

    original_count = len(df)

    # Identify invalid term records.
    missing_term_mask = (
        df["term"].isna()
        | df["term"].astype("string").str.strip().eq("")
    )

    removed_records = df[missing_term_mask].copy()

    # Retain every record with a valid term.
    cleaned = df.loc[~missing_term_mask].copy()

    # Preserve term text exactly; do not normalize it.
    cleaned["term"] = cleaned["term"].astype("string")

    # Reset row numbering after cleaning.
    cleaned.reset_index(drop=True, inplace=True)

    print("\nCLEANING SUMMARY")
    print(f"Original records: {original_count:,}")
    print(f"Removed invalid-term records: {len(removed_records)}")
    print(f"Final records: {len(cleaned):,}")

    if not removed_records.empty:
        print("\nREMOVED RECORDS")
        print(removed_records.to_string(index=False))

    return cleaned, removed_records


def verify_cleaned_data(source, cleaned, removed):
    """Verify that cleaning changed only the invalid-term record."""

    print("\n" + "=" * 65)
    print("CLEANED DATA VERIFICATION")
    print("=" * 65)

    checks = {}

    checks["expected final record count"] = (
        len(cleaned) == EXPECTED_RECORDS
    )

    checks["exactly one record removed"] = (
        len(source) - len(cleaned) == 1
    )

    checks["removed record has missing term"] = (
        len(removed) == 1
        and removed["term"].isna().all()
    )

    checks["no missing terms"] = (
        cleaned["term"].notna().all()
    )

    checks["no empty terms"] = (
        cleaned["term"].astype(str).str.strip().ne("").all()
    )

    checks["all terms unique"] = (
        cleaned["term"].is_unique
    )

    checks["no missing scores"] = (
        not cleaned[SCORE_COLUMNS].isnull().any().any()
    )

    checks["all scores within [-1, 1]"] = (
        (
            (cleaned[SCORE_COLUMNS] >= -1)
            & (cleaned[SCORE_COLUMNS] <= 1)
        ).all().all()
    )

    checks["no leading whitespace introduced"] = (
        cleaned["term"].eq(cleaned["term"].str.lstrip()).all()
    )

    checks["no trailing whitespace introduced"] = (
        cleaned["term"].eq(cleaned["term"].str.rstrip()).all()
    )

    # Verify all valid source records are preserved in order.
    valid_source = source.loc[source["term"].notna()].reset_index(drop=True)

    checks["all valid source terms preserved"] = (
        valid_source["term"].tolist()
        == cleaned["term"].tolist()
    )

    for column in SCORE_COLUMNS:
        checks[f"{column} scores preserved"] = (
            valid_source[column].tolist()
            == cleaned[column].tolist()
        )

    # Display checks.
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    passed = sum(checks.values())

    print(f"\nChecks passed: {passed}/{len(checks)}")

    if not all(checks.values()):
        raise RuntimeError(
            "NRC VAD cleaning verification failed."
        )

    print("\nALL CLEANING CHECKS PASSED.")


def main():
    print("=" * 65)
    print("NRC VAD LEXICON V2.1 - CLEANING PROCESS")
    print("=" * 65)

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Original lexicon not found: {SOURCE_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load original data.
    source = load_source()

    # Clean data.
    cleaned, removed = clean_data(source)

    # Verify cleaned data before saving.
    verify_cleaned_data(source, cleaned, removed)

    # Save cleaned dataset.
    cleaned.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # Verify saved file.
    saved = pd.read_csv(
        OUTPUT_FILE,
        encoding="utf-8",
        dtype={"term": "string"}
    )

    if len(saved) != len(cleaned):
        raise RuntimeError(
            "Saved dataset record count does not match."
        )

    if saved["term"].tolist() != cleaned["term"].tolist():
        raise RuntimeError(
            "Saved dataset terms do not match cleaned data."
        )

    print("\n" + "=" * 65)
    print("FINAL CLEANING SUMMARY")
    print("=" * 65)

    print(f"Original records: {len(source):,}")
    print(f"Removed records: {len(removed):,}")
    print(f"Cleaned records: {len(cleaned):,}")
    print(f"Unique terms: {cleaned['term'].nunique():,}")

    print("\nFinal VAD score ranges:")

    for column in SCORE_COLUMNS:
        print(
            f"{column}: "
            f"{cleaned[column].min()} to "
            f"{cleaned[column].max()}"
        )

    print(f"\nOutput: {OUTPUT_FILE.resolve()}")
    print("Saved file verification: PASS")
    print("\nNRC VAD CLEANING COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
