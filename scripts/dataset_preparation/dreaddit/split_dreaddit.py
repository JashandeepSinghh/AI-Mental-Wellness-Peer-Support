
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

INPUT = (
    PROJECT / "datasets" / "processed"
    / "dreaddit" / "standardized"
)

OUTPUT = (
    PROJECT / "datasets" / "processed"
    / "dreaddit" / "final_splits"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

train_df = pd.read_csv(INPUT / "train.csv")
test_df = pd.read_csv(INPUT / "test.csv")

print("=" * 65)
print("DREADDIT GROUP-AWARE TRAIN/VALIDATION SPLITTING")
print("=" * 65)

if train_df["post_id"].isna().any():
    raise ValueError("Missing post IDs in training data.")

if train_df["label"].isna().any():
    raise ValueError("Missing training labels.")

if not train_df["label"].isin([0, 1]).all():
    raise ValueError("Invalid training labels.")

# --------------------------------------------------
# GROUP-AWARE STRATIFIED SPLITTING
# --------------------------------------------------

X = train_df["text"]
y = train_df["label"]
groups = train_df["post_id"]

splitter = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# Evaluate all candidate folds and select the one
# closest to 20% validation with balanced labels.

candidates = []

for fold, (train_idx, val_idx) in enumerate(
    splitter.split(X, y, groups)
):

    candidate_train = train_df.iloc[train_idx]
    candidate_val = train_df.iloc[val_idx]

    validation_size_ratio = len(candidate_val) / len(train_df)

    full_distribution = (
        train_df["label"].value_counts(normalize=True)
    )

    validation_distribution = (
        candidate_val["label"].value_counts(normalize=True)
    )

    distribution_difference = sum(
        abs(
            full_distribution.get(label, 0)
            - validation_distribution.get(label, 0)
        )
        for label in [0, 1]
    )

    size_difference = abs(validation_size_ratio - 0.20)

    score = size_difference + distribution_difference

    candidates.append(
        (score, fold, train_idx, val_idx)
    )

# Select the best candidate
candidates.sort(key=lambda item: item[0])

best_score, best_fold, train_idx, val_idx = candidates[0]

final_train = train_df.iloc[train_idx].copy()
final_validation = train_df.iloc[val_idx].copy()

# Preserve the official test split unchanged
final_test = test_df.copy()

# --------------------------------------------------
# VERIFY GROUP SEPARATION
# --------------------------------------------------

train_groups = set(final_train["post_id"])
validation_groups = set(final_validation["post_id"])
test_groups = set(final_test["post_id"])

train_val_overlap = train_groups & validation_groups
train_test_overlap = train_groups & test_groups
val_test_overlap = validation_groups & test_groups

if train_val_overlap or train_test_overlap or val_test_overlap:
    raise ValueError("Post ID leakage detected.")

# --------------------------------------------------
# VERIFY RECORD PRESERVATION
# --------------------------------------------------

if len(final_train) + len(final_validation) != len(train_df):
    raise ValueError("Training/validation record loss detected.")

original_ids = set(train_df["id"])

split_ids = (
    set(final_train["id"])
    | set(final_validation["id"])
)

if original_ids != split_ids:
    raise ValueError("Training/validation ID preservation failed.")

if len(final_test) != len(test_df):
    raise ValueError("Official test record count changed.")

# --------------------------------------------------
# SAVE FINAL SPLITS
# --------------------------------------------------

outputs = {
    "train.csv": final_train,
    "validation.csv": final_validation,
    "test.csv": final_test
}

for filename, dataframe in outputs.items():

    dataframe.to_csv(
        OUTPUT / filename,
        index=False,
        encoding="utf-8"
    )

# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

print("\nSELECTED FOLD:", best_fold + 1)
print("Random state:", RANDOM_STATE)

for name, dataframe in [
    ("TRAINING", final_train),
    ("VALIDATION", final_validation),
    ("TESTING", final_test)
]:

    print("\n" + name)
    print("-" * 50)

    print("Records:", len(dataframe))
    print("Percentage of original training data:"
          if name != "TESTING" else "Official test records:",
          round(len(dataframe) / len(train_df) * 100, 2)
          if name != "TESTING" else len(dataframe))

    print("Unique post IDs:", dataframe["post_id"].nunique())

    print("Label distribution:")
    print(dataframe["stress"].value_counts().to_string())

    print("Label percentages:")
    print(
        (dataframe["stress"].value_counts(normalize=True) * 100)
        .round(2)
        .to_string()
    )

# --------------------------------------------------
# LEAKAGE REPORT
# --------------------------------------------------

print("\n" + "=" * 65)
print("POST-LEVEL LEAKAGE VERIFICATION")
print("=" * 65)

print("Training vs Validation:", len(train_val_overlap))
print("Training vs Testing:", len(train_test_overlap))
print("Validation vs Testing:", len(val_test_overlap))

print("\nExact text overlap:")

print(
    "Training vs Validation:",
    len(set(final_train["text"]) & set(final_validation["text"]))
)

print(
    "Training vs Testing:",
    len(set(final_train["text"]) & set(final_test["text"]))
)

print(
    "Validation vs Testing:",
    len(set(final_validation["text"]) & set(final_test["text"]))
)

print("\nTotal final records:",
      len(final_train) + len(final_validation) + len(final_test))

print("\nSaved files:")
for filename in outputs:
    print(OUTPUT / filename)

print("\nALL GROUP SPLITTING CHECKS PASSED")
print("=" * 65)
