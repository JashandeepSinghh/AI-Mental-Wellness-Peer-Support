
import json
from pathlib import Path

import pandas as pd


# ============================================================
# FIG-LONELINESS STANDARDIZED DATA VERIFICATION
# ============================================================

ROOT = Path("datasets/processed/fig_loneliness")

CLEANED_DIR = ROOT / "cleaned"
STANDARDIZED_DIR = ROOT / "standardized"

SPLITS = {
    "train": 3943,
    "validation": 1126,
    "test": 564,
}

EXPECTED_MAPPING = {
    "[1, 0]": {"label": 0, "loneliness": "non_lonely"},
    "[0, 1]": {"label": 1, "loneliness": "lonely"},
}

ANNOTATION_COLUMNS = [
    "lonely",
    "context_pri",
    "interaction",
    "interpersonal_pri",
    "temporal",
]

EXPECTED_COLUMNS = [
    "text",
    "label",
    "loneliness",
    "unique_id",
    "idx",
    "lonely",
    "context_pri",
    "interaction",
    "interpersonal_pri",
    "temporal",
]


def parse_vector(value):
    """Convert a serialized JSON annotation vector into a list."""
    if isinstance(value, list):
        return value
    return json.loads(value)


def verify_split(split):
    print("\n" + "=" * 65)
    print(f"VERIFYING: {split.upper()}")
    print("=" * 65)

    cleaned_path = CLEANED_DIR / f"{split}_cleaned.csv"
    standardized_path = STANDARDIZED_DIR / f"{split}.csv"

    cleaned = pd.read_csv(cleaned_path)
    standardized = pd.read_csv(standardized_path)

    checks = {}

    # --------------------------------------------------------
    # 1. Record count
    # --------------------------------------------------------

    checks["expected record count"] = (
        len(standardized) == SPLITS[split]
    )

    checks["cleaned-standardized count match"] = (
        len(cleaned) == len(standardized)
    )

    # --------------------------------------------------------
    # 2. Expected columns
    # --------------------------------------------------------

    checks["standardized column structure"] = (
        list(standardized.columns) == EXPECTED_COLUMNS
    )

    # --------------------------------------------------------
    # 3. Missing values
    # --------------------------------------------------------

    checks["no missing values"] = (
        not standardized.isnull().any().any()
    )

    # --------------------------------------------------------
    # 4. Text preservation
    # --------------------------------------------------------

    checks["text preserved"] = (
        cleaned["text"].astype(str).tolist()
        == standardized["text"].astype(str).tolist()
    )

    # --------------------------------------------------------
    # 5. Unique ID and index preservation
    # --------------------------------------------------------

    checks["unique IDs preserved"] = (
        cleaned["unique_id"].astype(str).tolist()
        == standardized["unique_id"].astype(str).tolist()
    )

    checks["indices preserved"] = (
        cleaned["idx"].tolist()
        == standardized["idx"].tolist()
    )

    checks["no duplicate unique IDs"] = (
        standardized["unique_id"].is_unique
    )

    checks["no duplicate indices"] = (
        standardized["idx"].is_unique
    )

    # --------------------------------------------------------
    # 6. Label mapping verification
    # --------------------------------------------------------

    expected_labels = []
    expected_names = []

    for value in cleaned["lonely"]:
        vector = parse_vector(value)
        key = json.dumps(vector)

        if key not in EXPECTED_MAPPING:
            raise ValueError(
                f"Invalid loneliness vector in {split}: {vector}"
            )

        expected_labels.append(
            EXPECTED_MAPPING[key]["label"]
        )

        expected_names.append(
            EXPECTED_MAPPING[key]["loneliness"]
        )

    checks["numeric labels correctly mapped"] = (
        standardized["label"].tolist() == expected_labels
    )

    checks["text labels correctly mapped"] = (
        standardized["loneliness"].tolist() == expected_names
    )

    checks["only valid numeric labels"] = (
        set(standardized["label"].unique()).issubset({0, 1})
    )

    checks["numeric and text labels agree"] = (
        (
            (standardized["label"] == 0)
            & (standardized["loneliness"] == "non_lonely")
        )
        |
        (
            (standardized["label"] == 1)
            & (standardized["loneliness"] == "lonely")
        )
    ).all()

    # --------------------------------------------------------
    # 7. Annotation metadata preservation
    # --------------------------------------------------------

    for column in ANNOTATION_COLUMNS:
        original_vectors = [
            parse_vector(value)
            for value in cleaned[column]
        ]

        standardized_vectors = [
            parse_vector(value)
            for value in standardized[column]
        ]

        checks[f"{column} annotations preserved"] = (
            original_vectors == standardized_vectors
        )

    # --------------------------------------------------------
    # 8. Label distribution
    # --------------------------------------------------------

    distribution = standardized["loneliness"].value_counts()

    lonely = int(distribution.get("lonely", 0))
    non_lonely = int(distribution.get("non_lonely", 0))

    print(f"Records: {len(standardized)}")
    print(f"Lonely: {lonely}")
    print(f"Non-lonely: {non_lonely}")

    # --------------------------------------------------------
    # 9. Display verification results
    # --------------------------------------------------------

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status}")

    passed_count = sum(checks.values())
    total_count = len(checks)

    print(f"\nChecks passed: {passed_count}/{total_count}")

    return standardized, checks


def main():
    print("=" * 65)
    print("FIG-LONELINESS FINAL SPLIT VERIFICATION")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Verify label mapping file
    # --------------------------------------------------------

    mapping_path = STANDARDIZED_DIR / "label_mapping.json"

    with open(mapping_path, "r", encoding="utf-8") as file:
        mapping = json.load(file)

    mapping_checks = {
        "original encoding": (
            mapping.get("original_encoding") == EXPECTED_MAPPING
        ),
        "model target": mapping.get("model_target") == "label",
        "model input": mapping.get("model_input") == "text",
        "number of classes": mapping.get("number_of_classes") == 2,
    }

    print("\nLABEL MAPPING VERIFICATION")

    for name, passed in mapping_checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    # --------------------------------------------------------
    # 2. Verify each split
    # --------------------------------------------------------

    frames = {}
    all_checks = dict(mapping_checks)

    for split in SPLITS:
        frame, checks = verify_split(split)
        frames[split] = frame

        for name, passed in checks.items():
            all_checks[f"{split}: {name}"] = passed

    # --------------------------------------------------------
    # 3. Cross-split leakage checks
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("CROSS-SPLIT LEAKAGE VERIFICATION")
    print("=" * 65)

    for first, second in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        first_df = frames[first]
        second_df = frames[second]

        id_overlap = (
            set(first_df["unique_id"])
            & set(second_df["unique_id"])
        )

        text_overlap = (
            set(first_df["text"])
            & set(second_df["text"])
        )

        id_check = len(id_overlap) == 0
        text_check = len(text_overlap) == 0

        all_checks[f"{first}-{second}: ID leakage"] = id_check
        all_checks[f"{first}-{second}: exact text leakage"] = text_check

        print(
            f"{first} vs {second} ID overlap: {len(id_overlap)} "
            f"({'PASS' if id_check else 'FAIL'})"
        )

        print(
            f"{first} vs {second} exact text overlap: {len(text_overlap)} "
            f"({'PASS' if text_check else 'FAIL'})"
        )

    # --------------------------------------------------------
    # 4. Combined dataset verification
    # --------------------------------------------------------

    combined = pd.concat(
        [frames[split] for split in SPLITS],
        ignore_index=True
    )

    total_records = sum(SPLITS.values())

    combined_checks = {
        "total records": len(combined) == total_records,
        "total unique IDs": combined["unique_id"].nunique() == total_records,
        "total valid labels": set(combined["label"].unique()).issubset({0, 1}),
    }

    all_checks.update(combined_checks)

    print("\nCOMBINED DATASET SUMMARY")
    print(f"Total records: {len(combined)}")
    print(f"Unique IDs: {combined['unique_id'].nunique()}")

    print("\nCombined label distribution:")
    print(combined["loneliness"].value_counts())

    print("\nCombined label percentages:")
    print(
        (combined["loneliness"].value_counts(normalize=True) * 100)
        .round(2)
    )

    # --------------------------------------------------------
    # 5. Final result
    # --------------------------------------------------------

    failed = [
        name for name, passed in all_checks.items()
        if not passed
    ]

    print("\n" + "=" * 65)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 65)

    print(f"Total checks: {len(all_checks)}")
    print(f"Passed: {sum(all_checks.values())}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFAILED CHECKS:")
        for item in failed:
            print(f"- {item}")

        raise SystemExit("\nFIG-LONELINESS VERIFICATION FAILED.")

    print("\nALL FIG-LONELINESS VERIFICATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
