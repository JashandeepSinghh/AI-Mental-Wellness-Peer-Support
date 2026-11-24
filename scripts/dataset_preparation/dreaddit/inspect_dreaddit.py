
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_PATH = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

RAW_PATH = PROJECT_PATH / "datasets" / "raw" / "dreaddit"

FILES = {
    "TRAINING": RAW_PATH / "dreaddit-train.csv",
    "TESTING": RAW_PATH / "dreaddit-test.csv"
}

# --------------------------------------------------
# INSPECTION
# --------------------------------------------------

print("=" * 65)
print("DREADDIT RAW DATASET INSPECTION")
print("=" * 65)

datasets = {}

for split_name, file_path in FILES.items():

    print(f"\n{split_name}")
    print("-" * 55)

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    df = pd.read_csv(file_path)

    datasets[split_name] = df

    print("File:", file_path.name)
    print("Shape:", df.shape)
    print("Records:", len(df))
    print("Columns:", len(df.columns))

    print("\nColumn names:")
    for column in df.columns:
        print("-", column)

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    print("\nDuplicate complete rows:", df.duplicated().sum())

    print("\nFirst 3 records:")
    print(df.head(3).to_string(index=False))

    # --------------------------------------------------
    # LABEL COLUMN CANDIDATES
    # --------------------------------------------------

    print("\nPotential label columns:")

    for column in df.columns:

        name = column.lower()

        if any(
            keyword in name
            for keyword in [
                "label", "stress", "class", "target"
            ]
        ):

            print(f"\nColumn: {column}")
            print("Unique values:", df[column].nunique())
            print("Value distribution:")
            print(df[column].value_counts(dropna=False).to_string())

# --------------------------------------------------
# COMPARISON
# --------------------------------------------------

print("\n" + "=" * 65)
print("TRAINING VS TESTING COMPARISON")
print("=" * 65)

train = datasets["TRAINING"]
test = datasets["TESTING"]

print("Training records:", len(train))
print("Testing records:", len(test))

print("\nSame columns:", list(train.columns) == list(test.columns))

print("\nTraining-only columns:",
      sorted(set(train.columns) - set(test.columns)))

print("\nTesting-only columns:",
      sorted(set(test.columns) - set(train.columns)))

print("\nTOTAL RECORDS:", len(train) + len(test))

print("\nINSPECTION COMPLETED")
print("=" * 65)
