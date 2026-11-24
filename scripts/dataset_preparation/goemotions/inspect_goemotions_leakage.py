
import pandas as pd
import os

BASE = r"D:\Projects\4AiMentalWellnessPeerSupport\datasets\raw\goemotions\official_splits"

def load_file(filename):
    return pd.read_csv(
        os.path.join(BASE, filename),
        sep="\t",
        header=None,
        names=["text", "label_ids", "comment_id"],
        dtype=str
    )

train = load_file("train.tsv")
validation = load_file("dev.tsv")
test = load_file("test.tsv")

pairs = [
    ("TRAIN", train, "VALIDATION", validation),
    ("TRAIN", train, "TEST", test),
    ("VALIDATION", validation, "TEST", test)
]

for name_a, df_a, name_b, df_b in pairs:

    print(f"\n========== {name_a} vs {name_b} ==========")

    duplicates = df_a.merge(
        df_b,
        on="text",
        suffixes=("_a", "_b")
    )

    print("Overlapping records:", len(duplicates))

    if not duplicates.empty:
        print(
            duplicates[
                ["text", "label_ids_a", "label_ids_b",
                 "comment_id_a", "comment_id_b"]
            ].head(10).to_string(index=False)
        )

print("\nLEAKAGE INSPECTION COMPLETED")
