
import subprocess
from pathlib import Path

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

RAW_DIR = PROJECT / "datasets" / "raw" / "fig_loneliness"

REPO_DIR = RAW_DIR / "source_repo"

REPOSITORY = (
    "https://huggingface.co/datasets/"
    "FIG-Loneliness/FIG-Loneliness"
)

RAW_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# DOWNLOAD OFFICIAL REPOSITORY
# --------------------------------------------------

print("=" * 65)
print("FIG-LONELINESS DATASET DOWNLOAD")
print("=" * 65)

if REPO_DIR.exists():

    print("\nRepository directory already exists.")
    print("Skipping download:", REPO_DIR)

else:

    print("\nCloning official dataset repository...")

    result = subprocess.run(
        [
            "git",
            "clone",
            REPOSITORY,
            str(REPO_DIR)
        ],
        check=False
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Git clone failed. Check Git installation and connection."
        )

# --------------------------------------------------
# VERIFY OFFICIAL SPLIT DIRECTORIES
# --------------------------------------------------

expected_splits = [
    "train_set",
    "dev_set",
    "test_set"
]

print("\nChecking dataset structure...")

for split in expected_splits:

    split_path = REPO_DIR / split

    exists = split_path.exists()

    print(f"{split}: {exists}")

    if not exists:
        raise FileNotFoundError(
            f"Missing expected dataset directory: {split_path}"
        )

# --------------------------------------------------
# REPORT FILES
# --------------------------------------------------

print("\nRepository files:")

for path in REPO_DIR.rglob("*"):

    if path.is_file():

        print(
            path.relative_to(REPO_DIR),
            "-",
            path.stat().st_size,
            "bytes"
        )

print("\n" + "=" * 65)
print("FIG-LONELINESS REPOSITORY DOWNLOADED")
print("=" * 65)

print("Location:", REPO_DIR)
