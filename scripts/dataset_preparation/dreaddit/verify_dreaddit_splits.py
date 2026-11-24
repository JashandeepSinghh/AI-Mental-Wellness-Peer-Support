
import pandas as pd
import json
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

CLEANED = PROJECT / "datasets" / "processed" / "dreaddit" / "cleaned"
STANDARDIZED = PROJECT / "datasets" / "processed" / "dreaddit" / "standardized"

FILES = {
    "TRAINING": ("train_cleaned.csv", "train.csv"),
    "TESTING": ("test_cleaned.csv", "test.csv")
}

MAPPING = {
    0: "non_stress",
    1: "stress"
}

# --------------------------------------------------
# LOAD MAPPING
# --------------------------------------------------

with open(STANDARDIZED / "label_mapping.json", encoding="utf-8") as f:
    saved_mapping = json.load(f)

assert saved_mapping == {
    str(k): v for k, v in MAPPING.items()
}, "Label mapping file mismatch."

results = {}

# --------------------------------------------------
# VERIFY EACH SPLIT
# --------------------------------------------------

print("=" * 65)
print("DREADDIT STANDARDIZED DATA VERIFICATION")
print("=" * 65)

for split, filenames in FILES.items():

    cleaned_file, standardized_file = filenames

    original = pd.read_csv(CLEANED / cleaned_file)
    standard = pd.read_csv(STANDARDIZED / standardized_file)

    print(f"\n{split}")
    print("-" * 55)

    checks = {
        "Record count preserved": len(original) == len(standard),
        "Text preserved": original["text"].equals(standard["text"]),
        "Numeric labels preserved": original["label"].equals(standard["label"]),
        "IDs preserved": original["id"].equals(standard["id"]),
        "Post IDs preserved": original["post_id"].equals(standard["post_id"]),
        "Confidence preserved": original["confidence"].equals(standard["confidence"]),
        "Stress mapping correct": (
            standard["stress"] == standard["label"].map(MAPPING)
        ).all(),
        "No missing values": standard.isnull().sum().sum() == 0,
        "No invalid labels": standard["label"].isin([0, 1]).all()
    }

    for name, passed in checks.items():
        print(f"{name}: {passed}")

    if not all(checks.values()):
        raise ValueError(f"{split} verification failed.")

    results[split] = standard

    print("\nRecords:", len(standard))
    print("Label distribution:")
    print(standard["stress"].value_counts().to_string())

    print("\nVerification: PASS")

# --------------------------------------------------
# CROSS-SPLIT CHECKS
# --------------------------------------------------

train = results["TRAINING"]
test = results["TESTING"]

print("\n" + "=" * 65)
print("OFFICIAL TRAIN/TEST LEAKAGE CHECK")
print("=" * 65)

id_overlap = set(train["id"]) & set(test["id"])
post_overlap = set(train["post_id"]) & set(test["post_id"])
text_overlap = set(train["text"]) & set(test["text"])

print("Overlapping record IDs:", len(id_overlap))
print("Overlapping post IDs:", len(post_overlap))
print("Identical texts:", len(text_overlap))

if id_overlap or post_overlap:
    raise ValueError("ID or post leakage detected.")

print("\nTotal standardized records:", len(train) + len(test))

print("\nALL STANDARDIZATION CHECKS PASSED")
print("=" * 65)
