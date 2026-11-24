
import pandas as pd
from pathlib import Path
from collections import Counter

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_PATH = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

DATA_PATH = (
    PROJECT_PATH
    / "datasets"
    / "raw"
    / "tweeteval"
    / "sentiment"
)

LABEL_NAMES = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# --------------------------------------------------
# LOAD TRAINING DATA
# --------------------------------------------------

texts = (
    DATA_PATH / "train_text.txt"
).read_text(encoding="utf-8").splitlines()

labels = [
    int(x) for x in (
        DATA_PATH / "train_labels.txt"
    ).read_text(encoding="utf-8").splitlines()
]

df = pd.DataFrame({
    "text": texts,
    "label": labels
})

print("=" * 65)
print("TWEETEVAL EDGE CASE INVESTIGATION")
print("=" * 65)

# --------------------------------------------------
# 1. IDENTIFY CONFLICTING DUPLICATE GROUPS
# --------------------------------------------------

print("\n1. CONFLICTING DUPLICATE TEXT GROUPS")
print("-" * 55)

groups = df.groupby("text", sort=False)

conflict_count = 0

for text, group in groups:

    if group["label"].nunique() > 1:

        conflict_count += 1

        print(f"\nConflict Group {conflict_count}")
        print("Text:", repr(text))

        for index, row in group.iterrows():

            label_id = int(row["label"])

            print(
                f"Record index: {index} | "
                f"Label: {label_id} | "
                f"Sentiment: {LABEL_NAMES[label_id]}"
            )

print("\nTotal conflicting groups:", conflict_count)

# --------------------------------------------------
# 2. IDENTIFY TRAILING WHITESPACE TYPES
# --------------------------------------------------

print("\n2. TRAILING WHITESPACE INVESTIGATION")
print("-" * 55)

trailing_count = 0
suffix_counter = Counter()

examples = []

for index, text in enumerate(texts):

    if text and text[-1].isspace():

        trailing_count += 1

        suffix_counter[text[-1]] += 1

        if len(examples) < 10:
            examples.append((index, text))

print("Records with trailing whitespace:", trailing_count)

print("\nTrailing whitespace character counts:")

for character, count in suffix_counter.items():

    print(
        repr(character),
        "Unicode:", ord(character),
        "Count:", count
    )

print("\nSample records with trailing whitespace:")

for index, text in examples:

    print("\nIndex:", index)
    print("Original:", repr(text))
    print("After strip:", repr(text.strip()))

# --------------------------------------------------
# 3. INVESTIGATE LEADING WHITESPACE
# --------------------------------------------------

print("\n3. LEADING WHITESPACE INVESTIGATION")
print("-" * 55)

leading_count = 0

for index, text in enumerate(texts):

    if text and text[0].isspace():

        leading_count += 1

        if leading_count <= 10:

            print("Index:", index)
            print("Text:", repr(text))

print("\nTotal leading whitespace records:", leading_count)

# --------------------------------------------------
# 4. CHECK WHETHER TRIMMING CHANGES TEXT
# --------------------------------------------------

print("\n4. WHITESPACE NORMALIZATION IMPACT")
print("-" * 55)

changed_records = sum(
    original != original.strip()
    for original in texts
)

print("Original records:", len(texts))
print("Records changed by strip():", changed_records)
print(
    "Records unchanged:",
    len(texts) - changed_records
)

print("\n" + "=" * 65)
print("EDGE CASE INVESTIGATION COMPLETED")
print("=" * 65)
