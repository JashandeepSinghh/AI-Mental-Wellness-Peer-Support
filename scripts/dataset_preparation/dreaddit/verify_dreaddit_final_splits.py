
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

BASE = PROJECT / "datasets" / "processed" / "dreaddit"

FINAL = BASE / "final_splits"
STANDARDIZED = BASE / "standardized"

FILES = {
    "TRAINING": FINAL / "train.csv",
    "VALIDATION": FINAL / "validation.csv",
    "TESTING": FINAL / "test.csv"
}

# --------------------------------------------------
# LOAD SAVED FILES
# --------------------------------------------------

data = {
    name: pd.read_csv(path)
    for name, path in FILES.items()
}

original_test = pd.read_csv(STANDARDIZED / "test.csv")

expected_counts = {
    "TRAINING": 2266,
    "VALIDATION": 568,
    "TESTING": 715
}

# --------------------------------------------------
# VERIFY EACH SPLIT
# --------------------------------------------------

print("=" * 65)
print("DREADDIT FINAL SPLIT VERIFICATION")
print("=" * 65)

for name, df in data.items():

    print(f"\n{name}")
    print("-" * 50)

    checks = {
        "Expected record count": len(df) == expected_counts[name],
        "No missing values": df.isnull().sum().sum() == 0,
        "Valid binary labels": df["label"].isin([0, 1]).all(),
        "Correct stress mapping": (
            df["stress"] == df["label"].map({
                0: "non_stress",
                1: "stress"
            })
        ).all(),
        "No duplicate record IDs": df["id"].duplicated().sum() == 0,
        "No missing post IDs": df["post_id"].notna().all()
    }

    for check, result in checks.items():
        print(f"{check}: {result}")

    if not all(checks.values()):
        raise ValueError(f"{name} verification failed.")

    print("Records:", len(df))
    print("Unique post IDs:", df["post_id"].nunique())
    print("Class distribution:")
    print(df["stress"].value_counts().to_string())

    print("Verification: PASS")

# --------------------------------------------------
# POST ID LEAKAGE
# --------------------------------------------------

train = data["TRAINING"]
validation = data["VALIDATION"]
test = data["TESTING"]

groups = {
    "TRAINING": set(train["post_id"]),
    "VALIDATION": set(validation["post_id"]),
    "TESTING": set(test["post_id"])
}

print("\n" + "=" * 65)
print("POST ID LEAKAGE VERIFICATION")
print("=" * 65)

pairs = [
    ("TRAINING", "VALIDATION"),
    ("TRAINING", "TESTING"),
    ("VALIDATION", "TESTING")
]

for first, second in pairs:

    overlap = groups[first] & groups[second]

    print(f"{first} vs {second}: {len(overlap)}")

    if overlap:
        raise ValueError("Post ID leakage detected.")

# --------------------------------------------------
# OFFICIAL TEST PRESERVATION
# --------------------------------------------------

print("\n" + "=" * 65)
print("OFFICIAL TEST PRESERVATION")
print("=" * 65)

test_preserved = test.equals(original_test)

print("Official test file unchanged:", test_preserved)

if not test_preserved:
    raise ValueError("Official test data was modified.")

# --------------------------------------------------
# TRAIN / VALIDATION ID PRESERVATION
# --------------------------------------------------

original_train = pd.read_csv(STANDARDIZED / "train.csv")

combined_ids = set(train["id"]) | set(validation["id"])

print("\nOriginal training records:", len(original_train))
print("Final training + validation:", len(train) + len(validation))

print(
    "All original training IDs preserved:",
    set(original_train["id"]) == combined_ids
)

if set(original_train["id"]) != combined_ids:
    raise ValueError("Training record preservation failed.")

# --------------------------------------------------
# TEXT OVERLAP REPORT
# --------------------------------------------------

print("\n" + "=" * 65)
print("IDENTICAL TEXT OVERLAP REPORT")
print("=" * 65)

for first, second in pairs:

    overlap = (
        set(data[first]["text"])
        & set(data[second]["text"])
    )

    print(f"{first} vs {second}: {len(overlap)}")

print("\n" + "=" * 65)
print("ALL FINAL SPLIT CHECKS PASSED")
print("=" * 65)
