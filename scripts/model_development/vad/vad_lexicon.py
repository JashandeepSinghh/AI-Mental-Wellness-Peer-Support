
"""
NRC VAD Lexicon Loader
======================

Loads and validates the prepared NRC VAD Lexicon.

Responsibilities:
1. Load the standardized CSV dataset.
2. Validate its schema and records.
3. Validate VAD score ranges.
4. Detect missing and duplicate terms.
5. Preserve multiword expressions.
6. Build a normalized term lookup dictionary.
7. Save a lexicon validation summary.

This module does not perform text-level VAD analysis.
"""

import json
import re
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from scripts.model_development.vad import vad_config as config


# ============================================================
# 1. TERM NORMALIZATION
# ============================================================

def normalize_term(term):
    """
    Normalize a lexicon term for case-insensitive matching.

    Multiple internal whitespace characters are converted
    into a single space.

    Multiword expressions remain intact.
    """

    if not isinstance(term, str):

        raise TypeError("Lexicon term must be a string.")

    normalized = term.strip().casefold()

    normalized = re.sub(
        r"\s+",
        " ",
        normalized
    )

    return normalized


# ============================================================
# 2. LOAD RAW LEXICON
# ============================================================

def load_raw_lexicon():

    if not config.DATASET_PATH.exists():

        raise FileNotFoundError(
            f"VAD dataset not found: {config.DATASET_PATH}"
        )

    dataframe = pd.read_csv(
        config.DATASET_PATH,
        encoding="utf-8"
    )

    if dataframe.empty:

        raise ValueError(
            "The VAD lexicon is empty."
        )

    return dataframe


# ============================================================
# 3. VALIDATE DATASET SCHEMA
# ============================================================

def validate_schema(dataframe):

    missing_columns = [

        column

        for column in config.REQUIRED_COLUMNS

        if column not in dataframe.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return True


# ============================================================
# 4. VALIDATE LEXICON TERMS
# ============================================================

def validate_terms(dataframe):

    terms = dataframe[config.TERM_COLUMN]

    missing_terms = int(terms.isna().sum())

    non_string_terms = int(

        sum(
            not isinstance(term, str)
            for term in terms.dropna()
        )
    )

    normalized_terms = [

        normalize_term(term)

        for term in terms

        if isinstance(term, str)
    ]

    empty_terms = sum(
        not term
        for term in normalized_terms
    )

    duplicate_terms = int(

        pd.Series(normalized_terms).duplicated().sum()
    )

    if missing_terms > 0:

        raise ValueError(
            f"Missing lexicon terms detected: {missing_terms}"
        )

    if non_string_terms > 0:

        raise ValueError(
            f"Non-string lexicon terms detected: {non_string_terms}"
        )

    if empty_terms > 0:

        raise ValueError(
            f"Empty lexicon terms detected: {empty_terms}"
        )

    if config.REQUIRE_UNIQUE_TERMS and duplicate_terms > 0:

        raise ValueError(
            "Duplicate normalized lexicon terms detected: "
            f"{duplicate_terms}"
        )

    return {

        "missing_terms": missing_terms,

        "non_string_terms": non_string_terms,

        "empty_terms": empty_terms,

        "duplicate_normalized_terms": duplicate_terms,

        "unique_terms": len(set(normalized_terms))
    }


# ============================================================
# 5. VALIDATE VAD SCORES
# ============================================================

def validate_scores(dataframe):

    score_columns = [

        config.VALENCE_COLUMN,

        config.AROUSAL_COLUMN,

        config.DOMINANCE_COLUMN
    ]

    invalid_scores = {}

    score_statistics = {}

    for column in score_columns:

        numeric_scores = pd.to_numeric(
            dataframe[column],
            errors="coerce"
        )

        missing_or_invalid = int(
            numeric_scores.isna().sum()
        )

        out_of_range = int(

            (
                (numeric_scores < config.MIN_VAD_SCORE)
                |
                (numeric_scores > config.MAX_VAD_SCORE)
            ).sum()
        )

        if missing_or_invalid > 0:

            invalid_scores[column] = {

                "missing_or_non_numeric": missing_or_invalid,

                "out_of_range": out_of_range
            }

            continue

        if out_of_range > 0:

            invalid_scores[column] = {

                "missing_or_non_numeric": 0,

                "out_of_range": out_of_range
            }

            continue

        score_statistics[column] = {

            "minimum": float(numeric_scores.min()),

            "maximum": float(numeric_scores.max()),

            "mean": round(
                float(numeric_scores.mean()),
                6
            ),

            "standard_deviation": round(
                float(numeric_scores.std()),
                6
            )
        }

    if invalid_scores:

        raise ValueError(
            f"Invalid VAD score values detected: {invalid_scores}"
        )

    return score_statistics


# ============================================================
# 6. VALIDATE TERM TYPES
# ============================================================

def validate_term_types(dataframe):

    term_types = dataframe[config.TERM_TYPE_COLUMN]

    missing_types = int(term_types.isna().sum())

    normalized_types = [

        str(value).strip().lower()

        for value in term_types.dropna()
    ]

    type_distribution = dict(
        Counter(normalized_types)
    )

    if missing_types > 0:

        raise ValueError(
            f"Missing term_type values: {missing_types}"
        )

    if any(not value for value in normalized_types):

        raise ValueError(
            "Empty term_type values detected."
        )

    return {

        "missing_term_types": missing_types,

        "term_type_distribution": type_distribution
    }


# ============================================================
# 7. PREPARE VALIDATED DATAFRAME
# ============================================================

def prepare_lexicon(dataframe):

    dataframe = dataframe.copy()

    # --------------------------------------------------------
    # A. Normalize terms
    # --------------------------------------------------------

    dataframe["_normalized_term"] = dataframe[
        config.TERM_COLUMN
    ].apply(normalize_term)

    # --------------------------------------------------------
    # B. Normalize term types
    # --------------------------------------------------------

    dataframe["_normalized_term_type"] = (

        dataframe[config.TERM_TYPE_COLUMN]

        .astype(str)

        .str.strip()

        .str.lower()
    )

    # --------------------------------------------------------
    # C. Convert score columns to numeric
    # --------------------------------------------------------

    score_columns = [

        config.VALENCE_COLUMN,

        config.AROUSAL_COLUMN,

        config.DOMINANCE_COLUMN
    ]

    for column in score_columns:

        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise"
        )

    # --------------------------------------------------------
    # D. Sort by normalized term
    # --------------------------------------------------------

    dataframe = dataframe.sort_values(
        by="_normalized_term"
    ).reset_index(drop=True)

    return dataframe


# ============================================================
# 8. BUILD LOOKUP DICTIONARY
# ============================================================

def build_lexicon_lookup(dataframe):
    """
    Build a dictionary for constant-time term lookup.

    Example:

    {
        "happy": {
            "term": "happy",
            "term_type": "unigram",
            "valence": 0.9,
            "arousal": 0.7,
            "dominance": 0.8
        }
    }

    The actual term_type strings are preserved from the
    standardized dataset after lowercase normalization.
    """

    lookup = {}

    for record in dataframe.to_dict(orient="records"):

        term = record["_normalized_term"]

        lookup[term] = {

            "term": record[config.TERM_COLUMN],

            "term_type": record["_normalized_term_type"],

            "valence": float(
                record[config.VALENCE_COLUMN]
            ),

            "arousal": float(
                record[config.AROUSAL_COLUMN]
            ),

            "dominance": float(
                record[config.DOMINANCE_COLUMN]
            )
        }

    return lookup


# ============================================================
# 9. GENERATE VALIDATION SUMMARY
# ============================================================

def generate_validation_summary(
    dataframe,
    term_statistics,
    score_statistics,
    type_statistics
):

    multiword_count = int(

        dataframe["_normalized_term"].str.contains(
            r"\s",
            regex=True
        ).sum()
    )

    unigram_count = len(dataframe) - multiword_count

    return {

        "module_name": config.MODULE_NAME,

        "module_version": config.MODULE_VERSION,

        "lexicon_name": config.LEXICON_NAME,

        "lexicon_version": config.LEXICON_VERSION,

        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "dataset_path": str(config.DATASET_PATH),

        "total_rows": int(len(dataframe)),

        "expected_rows": config.EXPECTED_LEXICON_ROWS,

        "row_count_matches_expected": (
            len(dataframe) == config.EXPECTED_LEXICON_ROWS
        ),

        "unique_terms": term_statistics["unique_terms"],

        "term_statistics": term_statistics,

        "score_statistics": score_statistics,

        "term_type_statistics": type_statistics,

        "term_structure": {

            "single_token_terms": unigram_count,

            "multiword_terms": multiword_count
        },

        "validation_status": "PASSED",

        "clinical_diagnosis": False,

        "actual_psychological_state_confirmed": False
    }


# ============================================================
# 10. SAVE VALIDATION SUMMARY
# ============================================================

def save_validation_summary(summary):

    config.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        config.LEXICON_SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# 11. COMPLETE LEXICON LOADING PIPELINE
# ============================================================

def load_vad_lexicon():

    # --------------------------------------------------------
    # A. Load dataset
    # --------------------------------------------------------

    dataframe = load_raw_lexicon()

    # --------------------------------------------------------
    # B. Validate schema
    # --------------------------------------------------------

    validate_schema(dataframe)

    # --------------------------------------------------------
    # C. Validate row count
    # --------------------------------------------------------

    if len(dataframe) != config.EXPECTED_LEXICON_ROWS:

        raise ValueError(

            f"Expected {config.EXPECTED_LEXICON_ROWS} rows, "

            f"but found {len(dataframe)}."
        )

    # --------------------------------------------------------
    # D. Validate terms
    # --------------------------------------------------------

    term_statistics = validate_terms(dataframe)

    # --------------------------------------------------------
    # E. Validate VAD scores
    # --------------------------------------------------------

    score_statistics = validate_scores(dataframe)

    # --------------------------------------------------------
    # F. Validate term types
    # --------------------------------------------------------

    type_statistics = validate_term_types(dataframe)

    # --------------------------------------------------------
    # G. Prepare dataset
    # --------------------------------------------------------

    dataframe = prepare_lexicon(dataframe)

    # --------------------------------------------------------
    # H. Build lookup dictionary
    # --------------------------------------------------------

    lookup = build_lexicon_lookup(dataframe)

    # --------------------------------------------------------
    # I. Generate validation report
    # --------------------------------------------------------

    summary = generate_validation_summary(

        dataframe,

        term_statistics,

        score_statistics,

        type_statistics
    )

    # --------------------------------------------------------
    # J. Save report
    # --------------------------------------------------------

    save_validation_summary(summary)

    return lookup, summary


# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print("NRC VAD LEXICON VALIDATION")

    print("=" * 70)

    lookup, summary = load_vad_lexicon()

    print("\nDataset validation: PASSED")

    print(
        f"Total lexicon entries: {summary['total_rows']:,}"
    )

    print(
        f"Unique terms: {summary['unique_terms']:,}"
    )

    print(
        "Single-token terms:",
        summary["term_structure"]["single_token_terms"]
    )

    print(
        "Multiword terms:",
        summary["term_structure"]["multiword_terms"]
    )

    print("\nVAD score statistics:")

    for dimension, statistics in summary[
        "score_statistics"
    ].items():

        print(f"\n{dimension.upper()}")

        for key, value in statistics.items():

            print(f"  {key}: {value}")

    print("\nTerm-type distribution:")

    for term_type, count in summary[
        "term_type_statistics"
    ]["term_type_distribution"].items():

        print(f"  {term_type}: {count}")

    print("\nLookup dictionary:")

    print(
        f"  Total searchable terms: {len(lookup):,}"
    )

    print("\nValidation report:")

    print(config.LEXICON_SUMMARY_PATH)

    print("\nNRC VAD lexicon loading: PASSED")

    print("=" * 70)
