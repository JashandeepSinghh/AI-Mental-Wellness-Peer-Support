
from pathlib import Path
import pandas as pd

# Project root
ROOT = Path(__file__).resolve().parents[3]

# Final cleaned GoEmotions dataset directory
DATA_DIR = ROOT / "datasets" / "processed" / "goemotions" / "cleaned"

FILES = {
    "Training": DATA_DIR / "train_cleaned.csv",
    "Validation": DATA_DIR / "validation_cleaned.csv",
    "Testing": DATA_DIR / "test_cleaned.csv"
}

print("=" * 75)
print("GOEMOTIONS - MODEL TRAINING DATA INSPECTION")
print("=" * 75)

for split_name, file_path in FILES.items():

    print(f"\n{'=' * 25} {split_name.upper()} {'=' * 25}")

    if not file_path.exists():
        print("ERROR: File not found:", file_path)
        continue

    df = pd.read_csv(file_path)

    print("\nFile:", file_path)
    print("Records:", len(df))
    print("Columns:", list(df.columns))

    print("\nColumn data types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nFirst 3 records:")
    print(df.head(3).to_string(index=False, max_colwidth=100))

    print("\n" + "-" * 65)

print("\nINSPECTION COMPLETED")
print("=" * 75)
