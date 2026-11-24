
"""
FIG-Loneliness Dataset Token-Length Analysis

Purpose:
1. Inspect the standardized dataset splits.
2. Verify the expected text and label columns.
3. Calculate token-length distributions.
4. Estimate truncation at candidate sequence lengths.
5. Generate evidence for selecting max sequence length.

No model training or dataset modification occurs.
"""

import json
import logging

from pathlib import Path

import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "fig_loneliness"
    / "standardized"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training"
    / "loneliness"
    / "fig_loneliness_distilbert"
    / "dataset_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

SPLITS = {
    "train": DATASET_DIR / "train.csv",
    "validation": DATASET_DIR / "validation.csv",
    "test": DATASET_DIR / "test.csv",
}

CANDIDATE_LENGTHS = [
    128,
    160,
    192,
    224,
    256,
    320,
    384,
]

TOKENIZATION_BATCH_SIZE = 64

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD TOKENIZER
# ============================================================

def load_tokenizer():

    logger.info(
        "Loading tokenizer: %s",
        MODEL_NAME,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    logger.info(
        "Tokenizer loaded successfully."
    )

    return tokenizer


# ============================================================
# LOAD AND INSPECT DATASET
# ============================================================

def load_dataset_splits():

    datasets = {}

    for split_name, file_path in SPLITS.items():

        if not file_path.exists():

            raise FileNotFoundError(
                f"Dataset file not found: {file_path}"
            )

        dataframe = pd.read_csv(
            file_path
        )

        logger.info(
            "%s split loaded: %s records",
            split_name,
            len(dataframe),
        )

        logger.info(
            "%s columns: %s",
            split_name,
            list(dataframe.columns),
        )

        required_columns = {
            TEXT_COLUMN,
            LABEL_COLUMN,
        }

        missing_columns = (
            required_columns
            - set(dataframe.columns)
        )

        if missing_columns:

            raise ValueError(
                f"{split_name} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if dataframe.empty:

            raise ValueError(
                f"{split_name} dataset is empty."
            )

        if dataframe[TEXT_COLUMN].isna().any():

            raise ValueError(
                f"{split_name} contains missing text values."
            )

        if dataframe[LABEL_COLUMN].isna().any():

            raise ValueError(
                f"{split_name} contains missing labels."
            )

        if not dataframe[TEXT_COLUMN].map(
            lambda value: isinstance(value, str)
        ).all():

            raise TypeError(
                f"{split_name} contains non-string text values."
            )

        datasets[split_name] = dataframe

    return datasets


# ============================================================
# TOKENIZE WITHOUT TRUNCATION
# ============================================================

def calculate_token_lengths(
    texts,
    tokenizer,
):

    lengths = []

    for start in tqdm(
        range(
            0,
            len(texts),
            TOKENIZATION_BATCH_SIZE,
        ),
        desc="Calculating token lengths",
    ):

        batch = texts[
            start:
            start + TOKENIZATION_BATCH_SIZE
        ]

        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )

        batch_lengths = [
            len(input_ids)
            for input_ids in encoded["input_ids"]
        ]

        lengths.extend(
            batch_lengths
        )

    return np.asarray(
        lengths,
        dtype=np.int64,
    )


# ============================================================
# ANALYZE ONE DATASET SPLIT
# ============================================================

def analyze_split(
    split_name,
    dataframe,
    tokenizer,
):

    logger.info(
        "Analyzing %s split...",
        split_name,
    )

    texts = dataframe[
        TEXT_COLUMN
    ].tolist()

    lengths = calculate_token_lengths(
        texts,
        tokenizer,
    )

    statistics = {

        "split": split_name,

        "records": int(len(lengths)),

        "minimum_tokens": int(lengths.min()),

        "maximum_tokens": int(lengths.max()),

        "mean_tokens": round(
            float(lengths.mean()),
            4,
        ),

        "median_tokens": round(
            float(np.median(lengths)),
            4,
        ),

        "p75_tokens": round(
            float(np.percentile(lengths, 75)),
            4,
        ),

        "p90_tokens": round(
            float(np.percentile(lengths, 90)),
            4,
        ),

        "p95_tokens": round(
            float(np.percentile(lengths, 95)),
            4,
        ),

        "p99_tokens": round(
            float(np.percentile(lengths, 99)),
            4,
        ),

    }

    truncation_statistics = []

    for max_length in CANDIDATE_LENGTHS:

        truncated_records = int(
            np.sum(lengths > max_length)
        )

        truncation_percentage = (
            truncated_records
            / len(lengths)
        ) * 100

        truncation_statistics.append({

            "split": split_name,

            "max_length": max_length,

            "truncated_records": truncated_records,

            "total_records": len(lengths),

            "truncation_percentage": round(
                truncation_percentage,
                4,
            ),

        })

    return (
        statistics,
        truncation_statistics,
        lengths,
    )


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

def analyze_labels(
    datasets,
):

    label_results = []

    for split_name, dataframe in datasets.items():

        distribution = (
            dataframe[LABEL_COLUMN]
            .value_counts()
            .sort_index()
        )

        for label, count in distribution.items():

            label_results.append({

                "split": split_name,

                "label": int(label),

                "count": int(count),

                "percentage": round(
                    (count / len(dataframe)) * 100,
                    4,
                ),

            })

    return label_results


# ============================================================
# SAVE ANALYSIS
# ============================================================

def save_results(
    dataset_statistics,
    truncation_statistics,
    label_statistics,
    all_lengths,
):

    pd.DataFrame(
        dataset_statistics
    ).to_csv(
        OUTPUT_DIR / "token_length_summary.csv",
        index=False,
    )

    pd.DataFrame(
        truncation_statistics
    ).to_csv(
        OUTPUT_DIR / "truncation_analysis.csv",
        index=False,
    )

    pd.DataFrame(
        label_statistics
    ).to_csv(
        OUTPUT_DIR / "label_distribution.csv",
        index=False,
    )

    for split_name, lengths in all_lengths.items():

        pd.DataFrame({
            "token_length": lengths
        }).to_csv(
            OUTPUT_DIR / f"{split_name}_token_lengths.csv",
            index=False,
        )

    summary = {

        "model_tokenizer": MODEL_NAME,

        "dataset_directory": str(DATASET_DIR),

        "candidate_max_lengths": CANDIDATE_LENGTHS,

        "dataset_statistics": dataset_statistics,

        "truncation_statistics": truncation_statistics,

        "label_statistics": label_statistics,

    }

    with open(
        OUTPUT_DIR / "token_length_analysis.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    logger.info(
        "Analysis files saved to: %s",
        OUTPUT_DIR,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 80)
    print("FIG-LONELINESS TOKEN-LENGTH ANALYSIS")
    print("=" * 80)

    tokenizer = load_tokenizer()

    datasets = load_dataset_splits()

    dataset_statistics = []

    truncation_statistics = []

    all_lengths = {}

    for split_name, dataframe in datasets.items():

        (
            statistics,
            truncation,
            lengths,
        ) = analyze_split(
            split_name,
            dataframe,
            tokenizer,
        )

        dataset_statistics.append(
            statistics
        )

        truncation_statistics.extend(
            truncation
        )

        all_lengths[split_name] = lengths

    label_statistics = analyze_labels(
        datasets
    )

    save_results(
        dataset_statistics,
        truncation_statistics,
        label_statistics,
        all_lengths,
    )

    print("\nDATASET TOKEN STATISTICS")
    print("=" * 80)

    print(
        pd.DataFrame(
            dataset_statistics
        ).to_string(index=False)
    )

    print("\nTRUNCATION ANALYSIS")
    print("=" * 80)

    print(
        pd.DataFrame(
            truncation_statistics
        ).pivot(
            index="max_length",
            columns="split",
            values="truncation_percentage",
        ).to_string()
    )

    print("\nLABEL DISTRIBUTION")
    print("=" * 80)

    print(
        pd.DataFrame(
            label_statistics
        ).to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("TOKEN-LENGTH ANALYSIS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":

    main()
