
import pandas as pd
import os

BASE_PATH = r"D:\Projects\4AiMentalWellnessPeerSupport\datasets\raw\goemotions\official_splits"

FILES = {
    "train": "train.tsv",
    "validation": "dev.tsv",
    "test": "test.tsv"
}

def load_split(filename):
    return pd.read_csv(
        os.path.join(BASE_PATH, filename),
        sep="\t",
        header=None,
        names=["text", "label_ids", "comment_id"],
        dtype=str
    )

datasets = {
    name: load_split(filename)
    for name, filename in FILES.items()
}

print("\n========== GOEMOTIONS QUALITY REPORT ==========")

# 1. Basic statistics
for name, df in datasets.items():
    print(f"\n{name.upper()}")
    print("Records:", len(df))
    print("Duplicate IDs:", df["comment_id"].duplicated().sum())
    print("Duplicate texts:", df["text"].duplicated().sum())
    print("Missing text:", df["text"].isna().sum())
    print("Empty text:", df["text"].str.strip().eq("").sum())
    print("Text shorter than 5 characters:",
          df["text"].str.strip().str.len().lt(5).sum())

# 2. Check ID leakage between splits
print("\n========== CROSS-SPLIT ID LEAKAGE ==========")

split_names = list(datasets.keys())

for i in range(len(split_names)):
    for j in range(i + 1, len(split_names)):
        first = split_names[i]
        second = split_names[j]

        ids_first = set(datasets[first]["comment_id"])
        ids_second = set(datasets[second]["comment_id"])

        overlap = ids_first.intersection(ids_second)

        print(f"{first} vs {second}: {len(overlap)} overlapping IDs")

# 3. Check text leakage between splits
print("\n========== CROSS-SPLIT TEXT LEAKAGE ==========")

for i in range(len(split_names)):
    for j in range(i + 1, len(split_names)):
        first = split_names[i]
        second = split_names[j]

        texts_first = set(datasets[first]["text"].str.strip().str.lower())
        texts_second = set(datasets[second]["text"].str.strip().str.lower())

        overlap = texts_first.intersection(texts_second)

        print(f"{first} vs {second}: {len(overlap)} overlapping texts")

print("\n========== QUALITY ASSESSMENT COMPLETED ==========")
