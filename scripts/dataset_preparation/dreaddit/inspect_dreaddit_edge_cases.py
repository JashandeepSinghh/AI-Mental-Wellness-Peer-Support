
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

RAW = Path(
    r"D:\Projects\4AiMentalWellnessPeerSupport\datasets\raw\dreaddit"
)

train = pd.read_csv(RAW / "dreaddit-train.csv")
test = pd.read_csv(RAW / "dreaddit-test.csv")

# --------------------------------------------------
# 1. CROSS-SPLIT IDENTICAL TEXTS
# --------------------------------------------------

print("=" * 65)
print("CROSS-SPLIT IDENTICAL TEXT INSPECTION")
print("=" * 65)

train_text = train["text"].fillna("").astype(str).str.strip()
test_text = test["text"].fillna("").astype(str).str.strip()

overlap = set(train_text) & set(test_text)

print("Identical texts:", len(overlap))

for text in sorted(overlap):

    print("\nTEXT:")
    print(text[:500])

    print("\nTRAINING RECORDS:")

    rows = train[train_text == text]

    print(
        rows[["id", "post_id", "label", "confidence"]]
        .to_string(index=False)
    )

    print("\nTESTING RECORDS:")

    rows = test[test_text == text]

    print(
        rows[["id", "post_id", "label", "confidence"]]
        .to_string(index=False)
    )

# --------------------------------------------------
# 2. REPEATED POST IDS
# --------------------------------------------------

print("\n" + "=" * 65)
print("REPEATED POST ID ANALYSIS")
print("=" * 65)

for name, df in [("TRAINING", train), ("TESTING", test)]:

    print(f"\n{name}")

    counts = df["post_id"].value_counts()

    repeated_ids = counts[counts > 1]

    print("Unique post IDs:", df["post_id"].nunique())
    print("Repeated post ID groups:", len(repeated_ids))
    print("Records belonging to repeated post IDs:",
          int(repeated_ids.sum()))

    # Check whether segments from one post have different labels
    mixed_label_groups = 0

    for post_id in repeated_ids.index:

        group = df[df["post_id"] == post_id]

        if group["label"].nunique() > 1:
            mixed_label_groups += 1

    print("Repeated post IDs with mixed labels:", mixed_label_groups)

    print("\nFive largest repeated post groups:")

    for post_id in repeated_ids.head(5).index:

        group = df[df["post_id"] == post_id]

        print("\nPost ID:", post_id)
        print("Segments:", len(group))
        print("Labels:", group["label"].value_counts().to_dict())

# --------------------------------------------------
# 3. ZERO-CONFIDENCE TEST RECORDS
# --------------------------------------------------

print("\n" + "=" * 65)
print("ZERO-CONFIDENCE TEST RECORD INSPECTION")
print("=" * 65)

zero_confidence = test[test["confidence"] == 0]

print("Zero-confidence records:", len(zero_confidence))

print("\nLabel distribution:")
print(zero_confidence["label"].value_counts().sort_index())

print("\nSample records:")

for _, row in zero_confidence.head(10).iterrows():

    print("\nID:", row["id"])
    print("Post ID:", row["post_id"])
    print("Label:", row["label"])
    print("Confidence:", row["confidence"])
    print("Text:", str(row["text"])[:350])

# --------------------------------------------------
# 4. TRAINING DUPLICATE TEXTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("TRAINING DUPLICATE TEXT INSPECTION")
print("=" * 65)

duplicate_mask = train_text.duplicated(keep=False)

duplicates = train.loc[duplicate_mask].copy()

print("Records belonging to duplicate text groups:",
      len(duplicates))

print("Unique duplicated text groups:",
      duplicates["text"].nunique())

for text, group in duplicates.groupby("text", sort=False):

    print("\nTEXT:")
    print(str(text)[:300])

    print(
        group[["id", "post_id", "label", "confidence"]]
        .to_string(index=False)
    )

print("\nEDGE CASE INSPECTION COMPLETED")
print("=" * 65)
