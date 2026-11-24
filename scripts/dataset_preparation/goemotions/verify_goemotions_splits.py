
import pandas as pd
from pathlib import Path

PROJECT_PATH = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

STANDARD_PATH = (
    PROJECT_PATH / "datasets" / "processed" / "goemotions"
)

CLEANED_PATH = STANDARD_PATH / "cleaned"

SPLITS = {
    "Training": ("train.csv", "train_cleaned.csv"),
    "Validation": ("validation.csv", "validation_cleaned.csv"),
    "Testing": ("test.csv", "test_cleaned.csv")
}

print("=" * 65)
print("GOEMOTIONS OFFICIAL SPLIT VERIFICATION")
print("=" * 65)

datasets = {}
all_checks_passed = True

# --------------------------------------------------
# 1. Verify each cleaned split against its source
# --------------------------------------------------

for split_name, (source_file, cleaned_file) in SPLITS.items():

    source = pd.read_csv(
        STANDARD_PATH / source_file,
        dtype=str
    )

    cleaned = pd.read_csv(
        CLEANED_PATH / cleaned_file,
        dtype=str
    )

    label_columns = list(source.columns[2:])

    count_match = len(source) == len(cleaned)

    columns_match = list(source.columns) == list(cleaned.columns)

    ids_match = source["comment_id"].equals(
        cleaned["comment_id"]
    )

    text_match = source["text"].equals(
        cleaned["text"]
    )

    labels_match = source[label_columns].equals(
        cleaned[label_columns]
    )

    checks = [
        count_match,
        columns_match,
        ids_match,
        text_match,
        labels_match
    ]

    passed = all(checks)

    all_checks_passed = all_checks_passed and passed

    datasets[split_name] = cleaned

    print(f"\n{split_name.upper()} SPLIT")
    print("-" * 45)
    print("Records:", len(cleaned))
    print("Columns preserved:", columns_match)
    print("Record count preserved:", count_match)
    print("Comment IDs preserved:", ids_match)
    print("Original text preserved:", text_match)
    print("Emotion labels preserved:", labels_match)
    print("Split verification:", "PASS" if passed else "FAIL")


# --------------------------------------------------
# 2. Verify duplicate comment IDs within each split
# --------------------------------------------------

print("\n" + "=" * 65)
print("COMMENT ID VALIDATION")
print("=" * 65)

id_sets = {}

for split_name, df in datasets.items():

    duplicate_ids = df["comment_id"].duplicated().sum()

    id_sets[split_name] = set(df["comment_id"])

    print(f"{split_name}:")
    print("Duplicate IDs:", duplicate_ids)

    if duplicate_ids > 0:
        all_checks_passed = False


# --------------------------------------------------
# 3. Verify cross-split ID leakage
# --------------------------------------------------

print("\n" + "=" * 65)
print("CROSS-SPLIT ID LEAKAGE CHECK")
print("=" * 65)

split_names = list(id_sets.keys())

for i in range(len(split_names)):

    for j in range(i + 1, len(split_names)):

        first = split_names[i]
        second = split_names[j]

        overlap = id_sets[first].intersection(
            id_sets[second]
        )

        print(f"{first} vs {second}: {len(overlap)} overlapping IDs")

        if overlap:
            all_checks_passed = False


# --------------------------------------------------
# 4. Report identical text across different splits
# --------------------------------------------------

print("\n" + "=" * 65)
print("CROSS-SPLIT TEXT OVERLAP REPORT")
print("=" * 65)

text_sets = {}

for split_name, df in datasets.items():

    normalized_text = (
        df["text"]
        .fillna("")
        .str.strip()
        .str.lower()
    )

    text_sets[split_name] = set(normalized_text)

for i in range(len(split_names)):

    for j in range(i + 1, len(split_names)):

        first = split_names[i]
        second = split_names[j]

        overlap = text_sets[first].intersection(
            text_sets[second]
        )

        print(
            f"{first} vs {second}: "
            f"{len(overlap)} identical normalized texts"
        )

print("\nNote: Identical text with different IDs is reported,")
print("not treated as comment ID leakage.")


# --------------------------------------------------
# 5. Final split distribution
# --------------------------------------------------

total_records = sum(
    len(df) for df in datasets.values()
)

print("\n" + "=" * 65)
print("FINAL SPLIT DISTRIBUTION")
print("=" * 65)

for split_name, df in datasets.items():

    percentage = len(df) / total_records * 100

    print(
        f"{split_name}: {len(df):,} records "
        f"({percentage:.2f}%)"
    )

print("Total records:", f"{total_records:,}")


# --------------------------------------------------
# 6. Final result
# --------------------------------------------------

print("\n" + "=" * 65)

if all_checks_passed and total_records == 54263:
    print("FINAL RESULT: ALL VERIFICATION CHECKS PASSED")
else:
    print("FINAL RESULT: VERIFICATION FAILED")
    raise ValueError(
        "One or more dataset split verification checks failed."
    )

print("=" * 65)
