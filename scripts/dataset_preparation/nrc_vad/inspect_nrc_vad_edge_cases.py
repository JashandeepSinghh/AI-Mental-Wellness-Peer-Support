
from pathlib import Path

import pandas as pd


# ============================================================
# NRC VAD V2.1 - EDGE CASE INSPECTION
# ============================================================

ROOT = Path(
    "datasets/raw/nrc_vad/extracted/NRC-VAD-Lexicon-v2.1"
)

MAIN_FILE = ROOT / "NRC-VAD-Lexicon-v2.1.txt"

SCORE_COLUMNS = ["valence", "arousal", "dominance"]


def load_data():
    return pd.read_csv(
        MAIN_FILE,
        sep="\t",
        encoding="utf-8-sig",
        dtype={"term": "string"}
    )


def main():
    print("=" * 70)
    print("NRC VAD V2.1 - EDGE CASE INSPECTION")
    print("=" * 70)

    df = load_data()

    # --------------------------------------------------------
    # 1. Inspect missing-term rows
    # --------------------------------------------------------

    print("\n1. MISSING TERM RECORDS")

    missing = df[df["term"].isna()]

    print(f"Missing-term records: {len(missing)}")

    if not missing.empty:
        print("\nMissing-term row details:")
        print(missing.to_string(index=True))

        print("\nMissing-term row scores:")
        print(missing[SCORE_COLUMNS].to_string(index=True))

    # --------------------------------------------------------
    # 2. Inspect empty and whitespace-only terms
    # --------------------------------------------------------

    print("\n2. EMPTY AND WHITESPACE TERMS")

    valid_terms = df["term"].dropna().astype(str)

    empty_terms = valid_terms[
        valid_terms.str.strip() == ""
    ]

    leading = valid_terms[
        valid_terms != valid_terms.str.lstrip()
    ]

    trailing = valid_terms[
        valid_terms != valid_terms.str.rstrip()
    ]

    print(f"Empty/whitespace-only terms: {len(empty_terms)}")
    print(f"Leading whitespace terms: {len(leading)}")
    print(f"Trailing whitespace terms: {len(trailing)}")

    # --------------------------------------------------------
    # 3. Case-insensitive duplicate inspection
    # --------------------------------------------------------

    print("\n3. CASE-INSENSITIVE DUPLICATES")

    normalized = (
        df["term"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    case_duplicates = normalized[
        normalized.duplicated(keep=False)
    ]

    print(
        "Records belonging to case-insensitive duplicate groups:",
        len(case_duplicates)
    )

    if not case_duplicates.empty:
        duplicate_indices = case_duplicates.index

        print(
            df.loc[
                duplicate_indices,
                ["term"] + SCORE_COLUMNS
            ]
            .sort_values("term")
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # 4. Normalized whitespace duplicate inspection
    # --------------------------------------------------------

    print("\n4. NORMALIZED WHITESPACE DUPLICATES")

    normalized_whitespace = (
        df["term"]
        .dropna()
        .astype(str)
        .str.split()
        .str.join(" ")
        .str.casefold()
    )

    whitespace_duplicates = normalized_whitespace[
        normalized_whitespace.duplicated(keep=False)
    ]

    print(
        "Records belonging to normalized duplicate groups:",
        len(whitespace_duplicates)
    )

    if not whitespace_duplicates.empty:
        print(
            df.loc[
                whitespace_duplicates.index,
                ["term"] + SCORE_COLUMNS
            ]
            .sort_values("term")
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # 5. Character and punctuation analysis
    # --------------------------------------------------------

    print("\n5. SPECIAL CHARACTER ANALYSIS")

    terms = df["term"].dropna().astype(str)

    non_ascii = terms[
        terms.apply(lambda value: not value.isascii())
    ]

    punctuation_terms = terms[
        terms.str.contains(r"[^\w\s'-]", regex=True)
    ]

    hyphen_terms = terms[
        terms.str.contains("-", regex=False)
    ]

    apostrophe_terms = terms[
        terms.str.contains("'", regex=False)
    ]

    print(f"Terms containing non-ASCII characters: {len(non_ascii)}")
    print(f"Terms containing other punctuation: {len(punctuation_terms)}")
    print(f"Terms containing hyphens: {len(hyphen_terms)}")
    print(f"Terms containing apostrophes: {len(apostrophe_terms)}")

    if not non_ascii.empty:
        print("\nNon-ASCII examples:")
        print(non_ascii.head(20).to_string(index=False))

    # --------------------------------------------------------
    # 6. Term length and word count distribution
    # --------------------------------------------------------

    print("\n6. TERM LENGTH DISTRIBUTION")

    word_counts = terms.str.split().str.len()

    print("\nWord count distribution:")
    print(word_counts.value_counts().sort_index().to_string())

    print("\nLongest terms:")
    longest = (
        df.dropna(subset=["term"])
        .assign(term_length=lambda x: x["term"].astype(str).str.len())
        .nlargest(15, "term_length")
    )

    print(
        longest[
            ["term", "term_length"] + SCORE_COLUMNS
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # 7. Score precision inspection
    # --------------------------------------------------------

    print("\n7. SCORE PRECISION")

    for column in SCORE_COLUMNS:
        decimal_places = (
            df[column]
            .dropna()
            .astype(str)
            .str.split(".")
            .str[-1]
            .str.len()
        )

        print(f"\n{column}:")
        print(decimal_places.value_counts().sort_index().to_string())

    # --------------------------------------------------------
    # 8. Exact raw line inspection for missing term
    # --------------------------------------------------------

    print("\n8. RAW FILE LINE INSPECTION")

    with open(
        MAIN_FILE,
        "r",
        encoding="utf-8-sig"
    ) as file:
        lines = file.readlines()

    for index in missing.index:
        # DataFrame index 0 corresponds to file line 2
        line_number = int(index) + 2

        if line_number <= len(lines):
            raw_line = lines[line_number - 1]

            print(f"\nMissing-term source line: {line_number}")
            print(f"Raw representation: {raw_line!r}")
            print(f"Tab-separated fields: {raw_line.rstrip().split(chr(9))}")

    # --------------------------------------------------------
    # 9. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EDGE CASE INSPECTION SUMMARY")
    print("=" * 70)

    print(f"Original records: {len(df):,}")
    print(f"Missing terms: {len(missing)}")
    print(f"Valid terms: {df['term'].notna().sum():,}")
    print(f"Unique valid terms: {df['term'].nunique():,}")
    print(f"Case-insensitive duplicate records: {len(case_duplicates)}")
    print(f"Whitespace-normalized duplicate records: {len(whitespace_duplicates)}")

    print("\nNo source files were modified.")

    print("\nEDGE CASE INSPECTION COMPLETED.")


if __name__ == "__main__":
    main()
