
import pandas as pd
from pathlib import Path

RAW = Path(
    r"D:\Projects\4AiMentalWellnessPeerSupport\datasets\raw\dreaddit"
)

files = {
    "TRAINING": RAW / "dreaddit-train.csv",
    "TESTING": RAW / "dreaddit-test.csv"
}

for split, path in files.items():

    df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print(split)
    print("=" * 60)

    print("\nCOLUMN NAMES AND DATA TYPES")

    for column in df.columns:
        print(f"{column}: {df[column].dtype}")

    print("\nPOTENTIAL LABEL COLUMNS")

    for column in df.columns:
        if any(word in column.lower() for word in
               ["label", "stress", "class", "target"]):

            print(f"\n{column}")
            print(df[column].value_counts(dropna=False).to_string())

    print("\nFIRST TWO COMPLETE RECORDS")

    print(df.head(2).to_string(index=False))

print("\nINSPECTION FINISHED")
