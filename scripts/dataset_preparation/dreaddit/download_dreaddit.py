
import subprocess
import zipfile
from pathlib import Path

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_PATH = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

RAW_PATH = PROJECT_PATH / "datasets" / "raw" / "dreaddit"

RAW_PATH.mkdir(parents=True, exist_ok=True)

ZIP_PATH = RAW_PATH / "dreaddit.zip"

# Original Columbia University dataset
DATASET_URL = (
    "https://www.cs.columbia.edu/~eturcan/data/dreaddit.zip"
)
# --------------------------------------------------
# DOWNLOAD DATASET
# --------------------------------------------------

print("=" * 60)
print("DREADDIT DATASET DOWNLOAD")
print("=" * 60)

print("\nDownloading original dataset...")

command = [
    "curl.exe",
    "-L",
    "--fail",
    "--retry", "3",
    "--output", str(ZIP_PATH),
    DATASET_URL
]

try:
    result = subprocess.run(command, check=True)

except FileNotFoundError:
    raise RuntimeError(
        "curl.exe was not found. Check Windows curl availability."
    )

except subprocess.CalledProcessError:
    raise RuntimeError(
        "Dataset download failed. Check internet connection and URL."
    )

if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size == 0:
    raise RuntimeError("Downloaded ZIP file is missing or empty.")

print("\nDownload successful.")
print("ZIP size:", ZIP_PATH.stat().st_size, "bytes")

# --------------------------------------------------
# INSPECT ZIP CONTENTS
# --------------------------------------------------

print("\nInspecting ZIP contents...")

if not zipfile.is_zipfile(ZIP_PATH):
    raise RuntimeError("Downloaded file is not a valid ZIP archive.")

with zipfile.ZipFile(ZIP_PATH, "r") as archive:

    for item in archive.infolist():
        print(
            item.filename,
            "|",
            item.file_size,
            "bytes"
        )

    # Extract original files into raw directory
    archive.extractall(RAW_PATH)

print("\nExtraction completed.")

# --------------------------------------------------
# FINAL FILE INVENTORY
# --------------------------------------------------

print("\nExtracted files:")

for file_path in sorted(RAW_PATH.rglob("*")):

    if file_path.is_file() and file_path != ZIP_PATH:

        print(
            file_path.relative_to(RAW_PATH),
            "|",
            file_path.stat().st_size,
            "bytes"
        )

print("\nDREADDIT DOWNLOAD COMPLETED")
print("=" * 60)
