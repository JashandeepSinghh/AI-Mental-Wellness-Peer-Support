
import urllib.request
from pathlib import Path

# Official Cardiff NLP repository
BASE_URL = (
    "https://raw.githubusercontent.com/"
    "cardiffnlp/tweeteval/main/datasets/sentiment/"
)

# Project directories
PROJECT_PATH = Path(r"D:\Projects\4AiMentalWellnessPeerSupport")

OUTPUT_PATH = (
    PROJECT_PATH
    / "datasets"
    / "raw"
    / "tweeteval"
    / "sentiment"
)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Official sentiment dataset files
FILES = [
    "train_text.txt",
    "train_labels.txt",
    "val_text.txt",
    "val_labels.txt",
    "test_text.txt",
    "test_labels.txt",
    "mapping.txt"
]

print("=" * 60)
print("TWEETEVAL SENTIMENT DATASET DOWNLOAD")
print("=" * 60)

for filename in FILES:

    url = BASE_URL + filename
    output_file = OUTPUT_PATH / filename

    print(f"\nDownloading: {filename}")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:

            data = response.read()

        if not data:
            raise ValueError("Downloaded file is empty.")

        with open(output_file, "wb") as file:
            file.write(data)

        print("Status: SUCCESS")
        print("File size:", len(data), "bytes")
        print("Saved:", output_file)

        if filename.endswith(".txt") and filename != "mapping.txt":
            line_count = len(data.decode("utf-8").splitlines())
            print("Line count:", line_count)

    except Exception as error:

        print("Status: FAILED")
        print("Error:", error)

        raise

print("\n" + "=" * 60)
print("TWEETEVAL DOWNLOAD COMPLETED")
print("Files downloaded:", len(FILES))
print("Dataset directory:", OUTPUT_PATH)
print("=" * 60)
