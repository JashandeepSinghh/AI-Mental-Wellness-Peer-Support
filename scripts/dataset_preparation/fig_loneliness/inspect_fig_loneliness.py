
import json
from pathlib import Path

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

REPO = (
    PROJECT
    / "datasets"
    / "raw"
    / "fig_loneliness"
    / "source_repo"
)

SPLITS = {
    "TRAIN": REPO / "train_set",
    "VALIDATION": REPO / "dev_set",
    "TEST": REPO / "test_set"
}

# --------------------------------------------------
# CHECK REQUIRED LIBRARIES
# --------------------------------------------------

try:
    from datasets import load_from_disk
except ImportError:
    raise ImportError(
        "Hugging Face datasets library is missing. "
        "Install it using: pip install datasets"
    )

# --------------------------------------------------
# INSPECT REPOSITORY README
# --------------------------------------------------

print("=" * 70)
print("FIG-LONELINESS DATASET INSPECTION")
print("=" * 70)

readme_path = REPO / "README.md"

if readme_path.exists():

    print("\nREADME available:", True)
    print("README size:", readme_path.stat().st_size, "bytes")

else:
    print("\nREADME available:", False)

# --------------------------------------------------
# INSPECT EACH SPLIT
# --------------------------------------------------

loaded = {}

for split_name, split_path in SPLITS.items():

    print("\n" + "=" * 70)
    print(split_name)
    print("=" * 70)

    arrow_path = split_path / "dataset.arrow"
    info_path = split_path / "dataset_info.json"

    # File checks
    print("\nArrow file exists:", arrow_path.exists())

    if arrow_path.exists():
        print(
            "Arrow file size:",
            arrow_path.stat().st_size,
            "bytes"
        )

    if not arrow_path.exists():
        raise FileNotFoundError(arrow_path)

    # Dataset metadata
    if info_path.exists():

        with open(info_path, "r", encoding="utf-8") as file:
            info = json.load(file)

        print("\nDataset metadata:")
        print(json.dumps(info, indent=2))

    # Load Arrow dataset
    print("\nLoading Arrow dataset...")

    dataset = load_from_disk(str(split_path))

    loaded[split_name] = dataset

    print("Arrow loading: SUCCESS")
    print("Records:", len(dataset))
    print("Columns:", dataset.column_names)
    print("Features:", dataset.features)

    # Preview records
    print("\nFirst 3 records:")

    for index in range(min(3, len(dataset))):

        record = dataset[index]

        print(f"\nRecord {index + 1}:")

        for column, value in record.items():

            value_text = str(value)

            if len(value_text) > 350:
                value_text = value_text[:350] + "..."

            print(f"{column}: {value_text}")

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("SPLIT SUMMARY")
print("=" * 70)

total_records = 0

for split_name, dataset in loaded.items():

    count = len(dataset)
    total_records += count

    print(f"{split_name}: {count}")

print("\nTotal records:", total_records)

print("\n" + "=" * 70)
print("INITIAL DATASET INSPECTION COMPLETED")
print("=" * 70)
