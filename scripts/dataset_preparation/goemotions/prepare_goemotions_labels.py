
import pandas as pd
import os

# Project dataset paths
BASE_PATH = r"D:\Projects\4AiMentalWellnessPeerSupport\datasets\raw\goemotions\official_splits"

OUTPUT_PATH = r"D:\Projects\4AiMentalWellnessPeerSupport\datasets\processed\goemotions"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load official emotion mapping
with open(os.path.join(BASE_PATH, "emotions.txt"), encoding="utf-8") as file:
    emotions = [line.strip() for line in file if line.strip()]

NUM_LABELS = len(emotions)

print("=" * 55)
print("GOEMOTIONS LABEL STANDARDIZATION")
print("=" * 55)

def prepare_split(filename, output_filename):

    input_file = os.path.join(BASE_PATH, filename)

    df = pd.read_csv(
        input_file,
        sep="\t",
        header=None,
        names=["text", "label_ids", "comment_id"],
        dtype=str
    )

    # Create one binary column per emotion
    label_vectors = []

    for label_group in df["label_ids"]:

        vector = [0] * NUM_LABELS

        for label_id in label_group.split(","):

            label_id = int(label_id)

            if label_id < 0 or label_id >= NUM_LABELS:
                raise ValueError(f"Invalid emotion ID: {label_id}")

            vector[label_id] = 1

        label_vectors.append(vector)

    # Convert vectors into a DataFrame
    label_df = pd.DataFrame(
        label_vectors,
        columns=emotions
    )

    # Combine text, comment ID and emotion columns
    processed_df = pd.concat(
        [
            df[["text", "comment_id"]].reset_index(drop=True),
            label_df
        ],
        axis=1
    )

    # Save processed dataset
    output_file = os.path.join(OUTPUT_PATH, output_filename)

    processed_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    print(f"\nFile: {output_filename}")
    print("Records:", len(processed_df))
    print("Columns:", len(processed_df.columns))
    print("Emotion categories:", NUM_LABELS)
    print("Saved:", output_file)


# Process each official split independently
prepare_split("train.tsv", "train.csv")
prepare_split("dev.tsv", "validation.csv")
prepare_split("test.tsv", "test.csv")

# Save the official emotion mapping
mapping_file = os.path.join(OUTPUT_PATH, "emotion_mapping.txt")

with open(mapping_file, "w", encoding="utf-8") as file:
    for index, emotion in enumerate(emotions):
        file.write(f"{index}: {emotion}\n")

print("\n" + "=" * 55)
print("LABEL STANDARDIZATION COMPLETED")
print("Total emotion categories:", NUM_LABELS)
print("Output directory:", OUTPUT_PATH)
print("=" * 55)
