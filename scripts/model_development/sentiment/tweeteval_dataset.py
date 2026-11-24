
import random

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from config.tweeteval_config import (
    MODEL_NAME,
    MAX_LENGTH,
    BATCH_SIZE,
    RANDOM_SEED,
    LABEL2ID,
    TRAIN_FILE,
    VALIDATION_FILE,
    DEVICE,
)


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

def set_seed(seed=RANDOM_SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. TWEETEVAL DATASET
# ============================================================

class TweetEvalDataset(Dataset):

    def __init__(self, csv_path, tokenizer):

        self.csv_path = csv_path

        print(f"\nLoading dataset: {csv_path.name}")

        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {csv_path}"
            )

        data = pd.read_csv(csv_path)

        required_columns = [
            "text",
            "label",
            "sentiment",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        if data[required_columns].isnull().any().any():
            raise ValueError(
                f"Missing values found in {csv_path.name}"
            )

        # ----------------------------------------------------
        # Validate sentiment labels
        # ----------------------------------------------------

        data["label"] = pd.to_numeric(
            data["label"],
            errors="coerce",
        )

        if data["label"].isnull().any():
            raise ValueError("Non-numeric sentiment labels found.")

        valid_labels = set(LABEL2ID.values())

        actual_labels = set(
            data["label"].astype(int).unique()
        )

        if not actual_labels.issubset(valid_labels):
            raise ValueError(
                f"Invalid labels detected: {actual_labels}"
            )

        # Verify numeric labels and sentiment names agree.
        expected_sentiments = data["label"].map({
            label_id: sentiment
            for sentiment, label_id in LABEL2ID.items()
        })

        if not expected_sentiments.equals(
            data["sentiment"].reset_index(drop=True)
        ):
            raise ValueError(
                "Numeric labels and sentiment names do not match."
            )

        self.texts = data["text"].astype(str).tolist()

        self.labels = data["label"].astype(
            np.int64
        ).to_numpy()

        # ----------------------------------------------------
        # Tokenization
        # ----------------------------------------------------

        self.encodings = []

        print("Tokenizing text...")

        chunk_size = 512

        for start in tqdm(
            range(0, len(self.texts), chunk_size),
            desc="Tokenization",
        ):

            text_batch = self.texts[
                start:start + chunk_size
            ]

            encoded = tokenizer(
                text_batch,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=False,
            )

            for index in range(len(text_batch)):

                self.encodings.append({
                    "input_ids": encoded["input_ids"][index],
                    "attention_mask": encoded["attention_mask"][index],
                })

        if len(self.encodings) != len(self.labels):
            raise RuntimeError(
                "Tokenized data and labels have different lengths."
            )

        print("Records loaded:", len(self.labels))
        print("Number of classes:", len(LABEL2ID))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        return {
            "input_ids": self.encodings[index]["input_ids"],
            "attention_mask": self.encodings[index]["attention_mask"],
            "labels": int(self.labels[index]),
        }


# ============================================================
# 3. DYNAMIC PADDING COLLATOR
# ============================================================

class TweetEvalCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):

        features = [
            {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
            }
            for item in batch
        ]

        # Pad each batch to its longest sequence.
        padded = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        # CrossEntropyLoss requires integer class IDs.
        padded["labels"] = torch.tensor(
            [
                item["labels"]
                for item in batch
            ],
            dtype=torch.long,
        )

        return padded


# ============================================================
# 4. CREATE DATA LOADERS
# ============================================================

def create_dataloaders():

    set_seed()

    print("=" * 65)
    print("TWEETEVAL DATASET AND DATALOADER")
    print("=" * 65)

    print("\nLoading DistilBERT tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded successfully.")
    print("Tokenizer vocabulary size:", len(tokenizer))

    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    print("\nInitializing training dataset...")

    train_dataset = TweetEvalDataset(
        TRAIN_FILE,
        tokenizer,
    )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    print("\nInitializing validation dataset...")

    validation_dataset = TweetEvalDataset(
        VALIDATION_FILE,
        tokenizer,
    )

    # --------------------------------------------------------
    # Collator
    # --------------------------------------------------------

    collator = TweetEvalCollator(tokenizer)

    # --------------------------------------------------------
    # Reproducible training shuffle
    # --------------------------------------------------------

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    # --------------------------------------------------------
    # Training DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=collator,
        generator=generator,
    )

    # --------------------------------------------------------
    # Validation DataLoader
    # --------------------------------------------------------

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=collator,
    )

    return (
        tokenizer,
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader,
    )


# ============================================================
# 5. DATALOADER VERIFICATION
# ============================================================

def verify_dataloaders(
    tokenizer,
    train_dataset,
    validation_dataset,
    train_loader,
    validation_loader,
):

    print("\n" + "=" * 65)
    print("DATALOADER VERIFICATION")
    print("=" * 65)

    print("Training records:", len(train_dataset))
    print("Validation records:", len(validation_dataset))

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))

    # Inspect one training batch.
    batch = next(iter(train_loader))

    print("\nSAMPLE TRAINING BATCH")

    print(
        "Input IDs shape:",
        tuple(batch["input_ids"].shape),
    )

    print(
        "Attention Mask shape:",
        tuple(batch["attention_mask"].shape),
    )

    print(
        "Labels shape:",
        tuple(batch["labels"].shape),
    )

    print(
        "Labels data type:",
        batch["labels"].dtype,
    )

    print(
        "Input IDs data type:",
        batch["input_ids"].dtype,
    )

    print(
        "Maximum token length:",
        batch["input_ids"].shape[1],
    )

    print(
        "Unique labels in sample:",
        torch.unique(batch["labels"]).tolist(),
    )

    # Structural validation.
    assert batch["input_ids"].shape[0] == BATCH_SIZE

    assert batch["labels"].shape == (
        BATCH_SIZE,
        1,
    ) or batch["labels"].shape == (BATCH_SIZE,)

    assert batch["labels"].dtype == torch.long

    assert batch["input_ids"].dtype == torch.long

    assert batch["attention_mask"].shape == (
        BATCH_SIZE,
        batch["input_ids"].shape[1],
    )

    assert batch["input_ids"].shape[1] <= MAX_LENGTH

    assert torch.all(
        (batch["labels"] >= 0)
        & (batch["labels"] < len(LABEL2ID))
    )

    print("\nDATALOADER VERIFICATION PASSED")
    print("=" * 65)


# ============================================================
# 6. EXECUTION
# ============================================================

if __name__ == "__main__":

    components = create_dataloaders()

    verify_dataloaders(*components)

    print("\nTWEETEVAL DATASET AND DATALOADERS READY")
