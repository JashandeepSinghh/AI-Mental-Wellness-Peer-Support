
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")
RAW = PROJECT / "datasets" / "raw" / "dreaddit"

FILES = {
    "TRAINING": RAW / "dreaddit-train.csv",
    "TESTING": RAW / "dreaddit-test.csv"
}

datasets = {}

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

for name, path in FILES.items():
    datasets[name] = pd.read_csv(path)

# --------------------------------------------------
# QUALITY ASSESSMENT
# --------------------------------------------------

print("=" * 65)
print("DREADDIT DATASET QUALITY ASSESSMENT")
print("=" * 65)

for name, df in datasets.items():

    print(f"\n{name}")
    print("-" * 55)

    print("Records:", len(df))
    print("Columns:", len(df.columns))

    # Missing values
    print("\nMissing values:")
    print("Total missing:", df.isnull().sum().sum())
    print("Missing text:", df["text"].isnull().sum())
    print("Missing labels:", df["label"].isnull().sum())

    # Empty text
    text = df["text"].fillna("").astype(str)

    print("\nText quality:")
    print("Empty text:", (text.str.strip() == "").sum())
    print("Whitespace-only text:", (
        text.str.strip().str.len() == 0
    ).sum())

    # Duplicate records
    print("\nDuplicate analysis:")
    print("Duplicate complete rows:", df.duplicated().sum())
    print("Duplicate IDs:", df["id"].duplicated().sum())
    print("Duplicate post IDs:", df["post_id"].duplicated().sum())
    print("Duplicate texts:", text.duplicated().sum())

    # Label validation
    print("\nLabel distribution:")
    print(df["label"].value_counts(dropna=False).sort_index())

    print("Invalid labels:", (~df["label"].isin([0, 1])).sum())

    # Text length
    lengths = text.str.len()

    print("\nText length:")
    print("Minimum:", lengths.min())
    print("Maximum:", lengths.max())
    print("Mean:", round(lengths.mean(), 2))
    print("Median:", lengths.median())

    print("Texts under 5 characters:", (lengths < 5).sum())

    # Annotation confidence
    print("\nAnnotation confidence:")
    print("Missing:", df["confidence"].isnull().sum())
    print("Minimum:", df["confidence"].min())
    print("Maximum:", df["confidence"].max())
    print("Mean:", round(df["confidence"].mean(), 4))

    print("\nConfidence distribution:")
    print(df["confidence"].value_counts().sort_index().to_string())

# --------------------------------------------------
# TRAIN / TEST COMPARISON
# --------------------------------------------------

train = datasets["TRAINING"]
test = datasets["TESTING"]

print("\n" + "=" * 65)
print("TRAINING VS TESTING INTEGRITY")
print("=" * 65)

# Column order
print("Same columns ignoring order:",
      set(train.columns) == set(test.columns))

print("Same column order:",
      list(train.columns) == list(test.columns))

# ID overlap
print("\nID overlap:",
      len(set(train["id"]) & set(test["id"])))

# Post overlap
print("Post ID overlap:",
      len(set(train["post_id"]) & set(test["post_id"])))

# Exact text overlap
train_texts = set(train["text"].dropna().astype(str).str.strip())
test_texts = set(test["text"].dropna().astype(str).str.strip())

print("Exact text overlap:",
      len(train_texts & test_texts))

# Label distributions
print("\nTraining label percentages:")
print((train["label"].value_counts(normalize=True) * 100).round(2))

print("\nTesting label percentages:")
print((test["label"].value_counts(normalize=True) * 100).round(2))

print("\nQUALITY ASSESSMENT COMPLETED")
print("=" * 65)
