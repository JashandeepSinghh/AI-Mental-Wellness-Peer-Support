
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_PATH = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

CLEANED_PATH = (
    PROJECT_PATH
    / "datasets"
    / "processed"
    / "tweeteval"
    / "sentiment"
    / "cleaned"
)

STANDARD_PATH = (
    PROJECT_PATH
    / "datasets"
    / "processed"
    / "tweeteval"
    / "sentiment"
    / "standardized"
)

SPLITS = {
    "Training": ("train_cleaned.csv", "train.csv"),
    "Validation": ("validation_cleaned.csv", "validation.csv"),
    "Testing": ("test_cleaned.csv", "test.csv")
}

LABEL_MAPPING = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

datasets = {}
all_checks_passed = True

print("=" * 65)
print("TWEETEVAL OFFICIAL SPLIT VERIFICATION")
print("=" * 65)

# --------------------------------------------------
# 1. VERIFY EACH SPLIT
# --------------------------------------------------

for split_name, (cleaned_file, standardized_file) in SPLITS.items():

    original = pd.read_csv(
        CLEANED_PATH / cleaned_file
    )

    standardized = pd.read_csv(
        STANDARD_PATH / standardized_file
    )

    # Record count
    count_match = len(original) == len(standardized)

    # Text preservation
    text_match = original["text"].equals(
        standardized["text"]
    )

    # Numeric label preservation
    label_match = original["label"].equals(
        standardized["label"]
    )

    # Sentiment mapping
    expected_sentiments = (
        standardized["label"].map(LABEL_MAPPING)
    )

    mapping_match = (
        expected_sentiments.equals(
            standardized["sentiment"]
        )
    )

    # Missing values
    missing_values = standardized.isna().sum().sum()

    # Invalid labels
    invalid_labels = (
        ~standardized["label"].isin(
            LABEL_MAPPING.keys()
        )
    ).sum()

    # Duplicate texts within split
    duplicate_texts = standardized["text"].duplicated().sum()

    checks = [
        count_match,
        text_match,
        label_match,
        mapping_match,
        missing_values == 0,
        invalid_labels == 0
    ]

    passed = all(checks)

    all_checks_passed = all_checks_passed and passed

    datasets[split_name] = standardized

    print(f"\n{split_name.upper()}")
    print("-" * 50)

    print("Records:", len(standardized))
    print("Record count preserved:", count_match)
    print("Text preserved:", text_match)
    print("Numeric labels preserved:", label_match)
    print("Sentiment mapping correct:", mapping_match)
    print("Missing values:", missing_values)
    print("Invalid labels:", invalid_labels)
    print("Duplicate texts within split:", duplicate_texts)
    print("Verification:", "PASS" if passed else "FAIL")


# --------------------------------------------------
# 2. CROSS-SPLIT TEXT LEAKAGE
# --------------------------------------------------

print("\n" + "=" * 65)
print("CROSS-SPLIT TEXT LEAKAGE CHECK")
print("=" * 65)

split_names = list(datasets.keys())

for i in range(len(split_names)):

    for j in range(i + 1, len(split_names)):

        first = split_names[i]
        second = split_names[j]

        first_texts = set(datasets[first]["text"])
        second_texts = set(datasets[second]["text"])

        overlap = first_texts.intersection(second_texts)

        print(
            f"{first} vs {second}: "
            f"{len(overlap)} identical texts"
        )

        if overlap:
            all_checks_passed = False


# --------------------------------------------------
# 3. FINAL DATASET DISTRIBUTION
# --------------------------------------------------

print("\n" + "=" * 65)
print("FINAL DATASET DISTRIBUTION")
print("=" * 65)

total_records = 0

for split_name, df in datasets.items():

    total_records += len(df)

    print(f"\n{split_name}:")

    print(
        df["sentiment"]
        .value_counts()
        .reindex(
            ["negative", "neutral", "positive"],
            fill_value=0
        )
        .to_string()
    )

print("\nTotal records:", total_records)

# --------------------------------------------------
# 4. FINAL RESULT
# --------------------------------------------------

print("\n" + "=" * 65)

if all_checks_passed and total_records == 59899:

    print("FINAL RESULT: ALL VERIFICATION CHECKS PASSED")

else:

    print("FINAL RESULT: VERIFICATION FAILED")

    raise ValueError(
        "One or more TweetEval verification checks failed."
    )

print("=" * 65)
