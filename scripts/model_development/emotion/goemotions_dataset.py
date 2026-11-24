
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from config.goemotions_config import (
    MODEL_NAME,
    MAX_LENGTH,
    BATCH_SIZE,
    LABEL_COLUMNS,
    TRAIN_FILE,
    VALIDATION_FILE,
    TEST_FILE,
    DEVICE
)


# ============================================================
# 1. TOKENIZER INITIALIZATION
# ============================================================

print("=" * 65)
print("GOEMOTIONS DATASET AND DATALOADER")
print("=" * 65)

print("\nLoading DistilBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded successfully.")
print("Tokenizer vocabulary size:", tokenizer.vocab_size)


# ============================================================
# 2. CUSTOM PYTORCH DATASET
# ============================================================

class GoEmotionsDataset(Dataset):

    def __init__(self, csv_path, tokenizer, label_columns, max_length):

        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.csv_path}"
            )

        print(f"\nLoading dataset: {self.csv_path.name}")

        # Load the CSV file
        dataframe = pd.read_csv(self.csv_path)

        # Validate required columns
        required_columns = ["text"] + label_columns

        missing_columns = [
            column for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        # Validate text
        if dataframe["text"].isnull().any():
            raise ValueError("Missing text values detected.")

        # Extract text
        texts = dataframe["text"].astype(str).tolist()

        # Extract 28 binary emotion labels
        labels = dataframe[label_columns].to_numpy(
            dtype=np.float32
        )

        if not np.isin(labels, [0.0, 1.0]).all():
            raise ValueError("Labels must contain only 0 and 1.")

        self.labels = torch.tensor(
            labels,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # 3. BATCH TOKENIZATION
        # ----------------------------------------------------

        self.input_ids = []
        self.attention_masks = []

        tokenization_batch_size = 1000

        print("Tokenizing text...")

        for start in tqdm(
            range(0, len(texts), tokenization_batch_size),
            desc="Tokenization"
        ):

            batch_texts = texts[
                start:start + tokenization_batch_size
            ]

            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_attention_mask=True
            )

            self.input_ids.extend(encoded["input_ids"])

            self.attention_masks.extend(
                encoded["attention_mask"]
            )

        print("Records loaded:", len(self.labels))
        print("Emotion labels:", self.labels.shape[1])

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_masks[index],
            "labels": self.labels[index]
        }


# ============================================================
# 4. DYNAMIC PADDING COLLATOR
# ============================================================

class GoEmotionsCollator:

    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

    def __call__(self, batch):

        # Separate tokenized inputs and labels
        input_features = [
            {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"]
            }
            for item in batch
        ]

        labels = torch.stack([
            item["labels"] for item in batch
        ])

        # Pad sequences to the longest sequence in this batch
        padded_inputs = self.tokenizer.pad(
            input_features,
            padding=True,
            return_tensors="pt"
        )

        padded_inputs["labels"] = labels

        return padded_inputs


# ============================================================
# 5. DATASET INITIALIZATION
# ============================================================

print("\nInitializing training dataset...")

train_dataset = GoEmotionsDataset(
    csv_path=TRAIN_FILE,
    tokenizer=tokenizer,
    label_columns=LABEL_COLUMNS,
    max_length=MAX_LENGTH
)

print("\nInitializing validation dataset...")

validation_dataset = GoEmotionsDataset(
    csv_path=VALIDATION_FILE,
    tokenizer=tokenizer,
    label_columns=LABEL_COLUMNS,
    max_length=MAX_LENGTH
)

print("\nInitializing testing dataset...")

test_dataset = GoEmotionsDataset(
    csv_path=TEST_FILE,
    tokenizer=tokenizer,
    label_columns=LABEL_COLUMNS,
    max_length=MAX_LENGTH
)


# ============================================================
# 6. DATALOADER INITIALIZATION
# ============================================================

collator = GoEmotionsCollator(tokenizer)

pin_memory = DEVICE.type == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collator,
    num_workers=0,
    pin_memory=pin_memory
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collator,
    num_workers=0,
    pin_memory=pin_memory
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collator,
    num_workers=0,
    pin_memory=pin_memory
)


# ============================================================
# 7. DATALOADER VERIFICATION
# ============================================================

print("\n" + "=" * 65)
print("DATALOADER VERIFICATION")
print("=" * 65)

print("Training batches:", len(train_loader))
print("Validation batches:", len(validation_loader))
print("Testing batches:", len(test_loader))

# Retrieve one training batch
sample_batch = next(iter(train_loader))

print("\nSAMPLE TRAINING BATCH")

print(
    "Input IDs shape:",
    tuple(sample_batch["input_ids"].shape)
)

print(
    "Attention Mask shape:",
    tuple(sample_batch["attention_mask"].shape)
)

print(
    "Labels shape:",
    tuple(sample_batch["labels"].shape)
)

print(
    "Labels data type:",
    sample_batch["labels"].dtype
)

print(
    "Input IDs data type:",
    sample_batch["input_ids"].dtype
)

print(
    "Maximum token length:",
    sample_batch["input_ids"].shape[1]
)

print(
    "Number of emotion outputs:",
    sample_batch["labels"].shape[1]
)

print("\nDATASET AND DATALOADERS READY")
print("=" * 65)
