
import pandas as pd
import os

BASE_PATH = r"D:\Projects\4AiMentalWellnessPeerSupport\datasets\raw\goemotions\official_splits"

labels = [
    line.strip()
    for line in open(
        os.path.join(BASE_PATH, "emotions.txt"),
        encoding="utf-8"
    )
]

files = ["train.tsv", "dev.tsv", "test.tsv"]

print("\n========== GOEMOTIONS DATASET VERIFICATION ==========")

for filename in files:

    filepath = os.path.join(BASE_PATH, filename)

    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=["text", "label_ids", "comment_id"],
        dtype=str
    )

    label_lists = df["label_ids"].str.split(",")

    invalid_count = sum(
        int(label) < 0 or int(label) >= len(labels)
        for row in label_lists
        for label in row
    )

    multilabel_count = label_lists.apply(
        lambda row: len(row) > 1
    ).sum()
    print(f"\nFile: {filename}")
    print(f"Total records: {len(df)}")
    print(f"Invalid label IDs: {invalid_count}")
    print(f"Multi-label records: {multilabel_count}")
    print(f"Missing text: {df['text'].isna().sum()}")

print("\nTotal emotion categories:", len(labels))
print("\n========== VERIFICATION COMPLETED ==========")
