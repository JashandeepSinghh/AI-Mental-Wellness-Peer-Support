
import pandas as pd
import os
from collections import Counter

BASE_PATH = r"D:\Projects\4AiMentalWellnessPeerSupport\datasets\raw\goemotions\official_splits"

labels = [
    line.strip()
    for line in open(
        os.path.join(BASE_PATH, "emotions.txt"),
        encoding="utf-8"
    )
]

splits = ["train.tsv", "dev.tsv", "test.tsv"]

for filename in splits:
    filepath = os.path.join(BASE_PATH, filename)

    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=["text", "label_ids", "comment_id"],
        dtype=str
    )

    counts = Counter()

    for label_group in df["label_ids"]:
        for label_id in label_group.split(","):
            counts[labels[int(label_id)]] += 1

    print(f"\n========== {filename} ==========")
    print("Total records:", len(df))
    print("Total emotion assignments:", sum(counts.values()))

    for emotion, count in counts.most_common():
        print(f"{emotion:20} {count}")

print("\nLABEL DISTRIBUTION ANALYSIS COMPLETED")
