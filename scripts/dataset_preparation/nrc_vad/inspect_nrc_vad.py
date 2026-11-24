
from pathlib import Path
import zipfile

# ============================================================
# NRC VAD LEXICON V2.1 - INITIAL FILE INSPECTION
# ============================================================

ROOT = Path("datasets/raw/nrc_vad")
ZIP_FILE = ROOT / "nrc_vad_v2.1.zip"
EXTRACTED_DIR = ROOT / "extracted"

print("=" * 65)
print("NRC VAD LEXICON V2.1 - FILE INSPECTION")
print("=" * 65)

# 1. Verify downloaded archive
if not ZIP_FILE.exists():
    raise FileNotFoundError(f"Archive not found: {ZIP_FILE}")

print("\nARCHIVE INFORMATION")
print(f"Archive: {ZIP_FILE}")
print(f"Size: {ZIP_FILE.stat().st_size:,} bytes")

if not zipfile.is_zipfile(ZIP_FILE):
    raise ValueError("Downloaded file is not a valid ZIP archive.")

print("ZIP integrity: PASS")

# 2. Inspect archive contents
print("\nARCHIVE CONTENTS")

with zipfile.ZipFile(ZIP_FILE, "r") as archive:
    bad_file = archive.testzip()

    if bad_file:
        raise ValueError(f"Corrupt archive member: {bad_file}")

    members = archive.infolist()

    for member in members:
        print(
            f"{member.filename} "
            f"({member.file_size:,} bytes)"
        )

    print(f"\nTotal archive entries: {len(members)}")

# 3. Inspect extracted directory
if not EXTRACTED_DIR.exists():
    raise FileNotFoundError(
        f"Extraction directory not found: {EXTRACTED_DIR}"
    )

print("\nEXTRACTED FILES")

files = sorted(
    path for path in EXTRACTED_DIR.rglob("*")
    if path.is_file()
)

if not files:
    raise RuntimeError("No extracted files found.")

for file in files:
    print(
        f"{file.relative_to(EXTRACTED_DIR)} "
        f"({file.stat().st_size:,} bytes)"
    )

# 4. Locate main lexicon and README
print("\nIMPORTANT FILE DISCOVERY")

lexicon_candidates = [
    file for file in files
    if file.name.lower() == "nrc-vad-lexicon-v2.1.txt"
]

readme_candidates = [
    file for file in files
    if "readme" in file.name.lower()
]

if not lexicon_candidates:
    print("Main lexicon filename not found exactly.")
    print("Review the extracted file list above.")
else:
    main_lexicon = lexicon_candidates[0]
    print(f"Main lexicon: {main_lexicon}")

if readme_candidates:
    for readme in readme_candidates:
        print(f"README: {readme}")

# 5. Inspect main lexicon sample
if lexicon_candidates:
    main_lexicon = lexicon_candidates[0]

    print("\nMAIN LEXICON SAMPLE")

    with open(
        main_lexicon,
        "r",
        encoding="utf-8-sig"
    ) as file:

        for line_number, line in enumerate(file, start=1):
            if line_number > 12:
                break

            print(f"{line_number}: {line.rstrip()}")

    print("\nInspection completed.")
else:
    print("\nMain lexicon inspection requires filename confirmation.")

print("\n" + "=" * 65)
print("INITIAL NRC VAD INSPECTION COMPLETED")
print("=" * 65)
