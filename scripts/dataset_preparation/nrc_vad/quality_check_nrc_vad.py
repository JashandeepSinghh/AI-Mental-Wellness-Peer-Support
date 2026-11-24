
from pathlib import Path

import pandas as pd


# ============================================================
# NRC VAD LEXICON V2.1 - DETAILED QUALITY INSPECTION
# ============================================================

ROOT = Path("datasets/raw/nrc_vad/extracted/NRC-VAD-Lexicon-v2.1")

MAIN_FILE = ROOT / "NRC-VAD-Lexicon-v2.1.txt"
UNIGRAM_FILE = ROOT / "Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt"
MWE_FILE = ROOT / "MWE/mwe-NRC-VAD-Lexicon-v2.1.txt"

SCORE_COLUMNS = ["valence", "arousal", "dominance"]


def load_lexicon(path):
    """Load a tab-separated NRC VAD lexicon file."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8-sig",
        dtype={"term": "string"},
        keep_default_na=True
    )

    df.columns = [str(column).strip().lower() for column in df.columns]

    required = ["term"] + SCORE_COLUMNS

    if not all(column in df.columns for column in required):
        raise ValueError(
            f"Unexpected columns in {path.name}: {df.columns.tolist()}"
        )

    df = df[required].copy()

    for column in SCORE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


def inspect_file(name, path):
    """Inspect record integrity, term structure, and score values."""

    print("\n" + "=" * 70)
    print(f"QUALITY INSPECTION: {name}")
    print("=" * 70)

    df = load_lexicon(path)

    # --------------------------------------------------------
    # 1. Basic information
    # --------------------------------------------------------

    print("\nBASIC INFORMATION")
    print(f"File: {path}")
    print(f"Records: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")

    # --------------------------------------------------------
    # 2. Missing values
    # --------------------------------------------------------

    print("\nMISSING VALUE ANALYSIS")
    print(df.isnull().sum())

    # --------------------------------------------------------
    # 3. Term analysis
    # --------------------------------------------------------

    print("\nTERM ANALYSIS")

    missing_terms = int(df["term"].isnull().sum())

    stripped_terms = df["term"].dropna().astype(str).str.strip()

    empty_terms = int((stripped_terms == "").sum())

    leading_whitespace = int(
        df["term"].dropna().astype(str).ne(
            df["term"].dropna().astype(str).str.lstrip()
        ).sum()
    )

    trailing_whitespace = int(
        df["term"].dropna().astype(str).ne(
            df["term"].dropna().astype(str).str.rstrip()
        ).sum()
    )

    duplicate_terms = int(
        df["term"].dropna().duplicated().sum()
    )

    unique_terms = int(df["term"].nunique(dropna=True))

    term_lengths = df["term"].dropna().astype(str).str.len()

    word_counts = (
        df["term"]
        .dropna()
        .astype(str)
        .str.split()
        .str.len()
    )

    print(f"Missing terms: {missing_terms}")
    print(f"Empty terms: {empty_terms}")
    print(f"Leading whitespace: {leading_whitespace}")
    print(f"Trailing whitespace: {trailing_whitespace}")
    print(f"Duplicate term records: {duplicate_terms}")
    print(f"Unique terms: {unique_terms:,}")

    if len(term_lengths):
        print(f"Minimum term length: {term_lengths.min()}")
        print(f"Maximum term length: {term_lengths.max()}")
        print(f"Mean term length: {term_lengths.mean():.2f}")

    print("\nTERM TYPE DISTRIBUTION")
    print("Single-word terms:", int((word_counts == 1).sum()))
    print("Multi-word terms:", int((word_counts > 1).sum()))

    # --------------------------------------------------------
    # 4. Numeric score validation
    # --------------------------------------------------------

    print("\nNUMERIC SCORE ANALYSIS")

    for column in SCORE_COLUMNS:
        values = df[column]

        print(f"\n{column.upper()}")
        print(f"Missing values: {values.isnull().sum()}")
        print(f"Minimum: {values.min()}")
        print(f"Maximum: {values.max()}")
        print(f"Mean: {values.mean():.6f}")
        print(f"Median: {values.median():.6f}")

    all_scores = df[SCORE_COLUMNS]

    invalid_numeric_rows = int(
        all_scores.isnull().any(axis=1).sum()
    )

    out_of_range = int(
        (
            (all_scores < -1)
            | (all_scores > 1)
        ).any(axis=1).sum()
    )

    print("\nSCORE VALIDATION")
    print(f"Rows with missing/non-numeric scores: {invalid_numeric_rows}")
    print(f"Rows with scores outside [-1, 1]: {out_of_range}")

    # --------------------------------------------------------
    # 5. Duplicate record analysis
    # --------------------------------------------------------

    print("\nDUPLICATE ANALYSIS")

    duplicate_rows = int(df.duplicated().sum())

    duplicate_full_terms = int(
        df.duplicated(subset=["term"], keep=False).sum()
    )

    print(f"Duplicate complete rows: {duplicate_rows}")
    print(f"Rows belonging to duplicate-term groups: {duplicate_full_terms}")

    if duplicate_full_terms:
        print("\nDUPLICATE TERM SAMPLES")

        print(
            df[df.duplicated(subset=["term"], keep=False)]
            .sort_values("term")
            .head(20)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # 6. Sample records
    # --------------------------------------------------------

    print("\nFIRST 10 RECORDS")
    print(df.head(10).to_string(index=False))

    print("\nLAST 5 RECORDS")
    print(df.tail(5).to_string(index=False))

    # --------------------------------------------------------
    # 7. Basic quality flags
    # --------------------------------------------------------

    checks = {
        "file is not empty": len(df) > 0,
        "required columns present": all(
            column in df.columns for column in required_columns()
        ),
        "no missing terms": missing_terms == 0,
        "no empty terms": empty_terms == 0,
        "no missing/non-numeric scores": invalid_numeric_rows == 0,
        "all scores within [-1, 1]": out_of_range == 0,
    }

    print("\nBASIC QUALITY CHECKS")

    for check, passed in checks.items():
        print(f"{check}: {'PASS' if passed else 'FAIL'}")

    return df


def required_columns():
    return ["term"] + SCORE_COLUMNS


def main():
    print("=" * 70)
    print("NRC VAD LEXICON V2.1 - DATASET QUALITY ASSESSMENT")
    print("=" * 70)

    main_df = inspect_file("COMBINED MAIN LEXICON", MAIN_FILE)

    unigram_df = inspect_file("UNIGRAM LEXICON", UNIGRAM_FILE)

    mwe_df = inspect_file("MULTI-WORD EXPRESSION LEXICON", MWE_FILE)

    # --------------------------------------------------------
    # 8. Cross-file comparison
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CROSS-FILE COMPARISON")
    print("=" * 70)

    main_terms = set(main_df["term"].dropna())
    unigram_terms = set(unigram_df["term"].dropna())
    mwe_terms = set(mwe_df["term"].dropna())

    print(f"Main lexicon records: {len(main_df):,}")
    print(f"Unigram records: {len(unigram_df):,}")
    print(f"MWE records: {len(mwe_df):,}")

    print(
        "Main lexicon unique terms:",
        len(main_terms)
    )

    print(
        "Unigram terms missing from main lexicon:",
        len(unigram_terms - main_terms)
    )

    print(
        "MWE terms missing from main lexicon:",
        len(mwe_terms - main_terms)
    )

    print(
        "Unigram/MWE intersection:",
        len(unigram_terms & mwe_terms)
    )

    print(
        "Main lexicon terms not in unigram or MWE:",
        len(main_terms - (unigram_terms | mwe_terms))
    )

    # --------------------------------------------------------
    # 9. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("NRC VAD QUALITY INSPECTION COMPLETED")
    print("=" * 70)

    print("\nPrimary model resource:")
    print("NRC-VAD-Lexicon-v2.1.txt")

    print("\nSupporting resources:")
    print("Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt")
    print("MWE/mwe-NRC-VAD-Lexicon-v2.1.txt")

    print("\nNo records have been modified or removed.")


if __name__ == "__main__":
    main()
