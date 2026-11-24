
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "datasets" / "processed" / "dreaddit" / "final_splits"

MODEL_NAME = "distilbert-base-uncased"

SPLITS = {
    "train": DATA_DIR / "train.csv",
    "validation": DATA_DIR / "validation.csv",
    "test": DATA_DIR / "test.csv",
}

MAX_LENGTH_CANDIDATES = [128, 160, 192, 224, 256, 320, 384, 512]


def analyze_split(name, path, tokenizer):
    print("\n" + "=" * 65)
    print(f"{name.upper()} TOKEN LENGTH ANALYSIS")
    print("=" * 65)

    df = pd.read_csv(path)

    texts = df["text"].astype(str).tolist()

    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_length=True,
    )

    lengths = np.asarray(encoded["length"], dtype=np.int32)

    print(f"Records: {len(lengths):,}")
    print(f"Minimum tokens: {lengths.min()}")
    print(f"Maximum tokens: {lengths.max()}")
    print(f"Mean tokens: {lengths.mean():.2f}")
    print(f"Median tokens: {np.median(lengths):.2f}")

    for percentile in [90, 95, 97, 98, 99, 99.5]:
        print(
            f"{percentile}th percentile: "
            f"{np.percentile(lengths, percentile):.2f}"
        )

    print("\nTruncation analysis:")

    for max_length in MAX_LENGTH_CANDIDATES:
        truncated = int(np.sum(lengths > max_length))
        percentage = truncated / len(lengths) * 100

        print(
            f"MAX_LENGTH {max_length:3d}: "
            f"{truncated:4d} records truncated "
            f"({percentage:.2f}%)"
        )

    return lengths


def main():
    print("=" * 65)
    print("DREADDIT DISTILBERT TOKEN LENGTH ANALYSIS")
    print("=" * 65)

    print(f"Project root: {ROOT}")
    print(f"Tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    all_lengths = {}

    for name, path in SPLITS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}")

        all_lengths[name] = analyze_split(name, path, tokenizer)

    print("\n" + "=" * 65)
    print("COMBINED TOKEN LENGTH ANALYSIS")
    print("=" * 65)

    combined = np.concatenate(list(all_lengths.values()))

    print(f"Total records: {len(combined):,}")
    print(f"Maximum tokens: {combined.max()}")
    print(f"Mean tokens: {combined.mean():.2f}")

    print("\nCOMBINED TRUNCATION ANALYSIS:")

    for max_length in MAX_LENGTH_CANDIDATES:
        truncated = int(np.sum(combined > max_length))
        percentage = truncated / len(combined) * 100

        print(
            f"MAX_LENGTH {max_length:3d}: "
            f"{truncated:4d} records truncated "
            f"({percentage:.2f}%)"
        )

    print("\nTOKEN LENGTH ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
