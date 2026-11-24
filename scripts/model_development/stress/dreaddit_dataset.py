

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from config.dreaddit_config import (
    MODEL_NAME,
    MAX_LENGTH,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    TRAIN_FILE,
    VALIDATION_FILE,
    TEST_FILE,
    LABEL2ID,
)


class DreadditDataset(Dataset):

    def __init__(self, csv_path, tokenizer):

        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.csv_path}"
            )

        df = pd.read_csv(self.csv_path)

        required_columns = {"text", "label", "stress"}
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing columns in {self.csv_path.name}: "
                f"{sorted(missing)}"
            )

        if df[["text", "label", "stress"]].isna().any().any():
            raise ValueError(
                f"Missing required values in {self.csv_path.name}"
            )

        if not df["label"].isin([0, 1]).all():
            raise ValueError(
                f"Invalid labels in {self.csv_path.name}"
            )

        expected_stress = df["label"].map({
            0: "non_stress",
            1: "stress",
        })

        if not expected_stress.equals(df["stress"]):
            raise ValueError(
                f"Label mapping mismatch in {self.csv_path.name}"
            )

        self.texts = df["text"].astype(str).tolist()

        self.labels = torch.tensor(
            df["label"].astype(int).tolist(),
            dtype=torch.long,
        )

        self.tokenizer = tokenizer

        # Tokenize in chunks to keep preprocessing memory controlled.
        self.encodings = {
            "input_ids": [],
            "attention_mask": [],
        }

        chunk_size = 256

        for start in range(0, len(self.texts), chunk_size):

            chunk = self.texts[start:start + chunk_size]

            encoded = tokenizer(
                chunk,
                max_length=MAX_LENGTH,
                truncation=True,
                padding=False,
                add_special_tokens=True,
            )

            self.encodings["input_ids"].extend(
                encoded["input_ids"]
            )

            self.encodings["attention_mask"].extend(
                encoded["attention_mask"]
            )

        if len(self.encodings["input_ids"]) != len(self.labels):
            raise RuntimeError(
                "Tokenized records and labels are misaligned."
            )

        print(
            f"Loaded {self.csv_path.name}: "
            f"{len(self.labels):,} records"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        return {
            "input_ids": torch.tensor(
                self.encodings["input_ids"][index],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                self.encodings["attention_mask"][index],
                dtype=torch.long,
            ),
            "labels": self.labels[index],
        }


class DreadditCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):

        input_features = [
            {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
            }
            for item in batch
        ]

        padded = self.tokenizer.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )

        labels = torch.stack(
            [item["labels"] for item in batch]
        )

        padded["labels"] = labels

        return padded


def create_dataloaders():

    print("=" * 65)
    print("CREATING DREADDIT DATASETS AND DATALOADERS")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = DreadditDataset(
        TRAIN_FILE,
        tokenizer,
    )

    validation_dataset = DreadditDataset(
        VALIDATION_FILE,
        tokenizer,
    )

    test_dataset = DreadditDataset(
        TEST_FILE,
        tokenizer,
    )

    collator = DreadditCollator(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collator,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        tokenizer,
    )


def test_dataloaders():

    (
        train_loader,
        validation_loader,
        test_loader,
        tokenizer,
    ) = create_dataloaders()

    print("\n" + "=" * 65)
    print("DATALOADER VERIFICATION")
    print("=" * 65)

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(validation_loader)}")
    print(f"Test batches: {len(test_loader)}")

    batch = next(iter(train_loader))

    print("\nSample batch:")

    for key, value in batch.items():
        print(
            f"{key}: shape={tuple(value.shape)}, "
            f"dtype={value.dtype}"
        )

    if batch["input_ids"].shape[0] != BATCH_SIZE:
        raise RuntimeError("Unexpected training batch size.")

    if batch["labels"].dtype != torch.long:
        raise RuntimeError("Labels must be torch.long.")

    if not torch.all(
        (batch["labels"] == 0) | (batch["labels"] == 1)
    ):
        raise RuntimeError("Invalid labels in batch.")

    if batch["input_ids"].shape[1] > MAX_LENGTH:
        raise RuntimeError("Sequence exceeds MAX_LENGTH.")

    if not torch.all(
        batch["attention_mask"].sum(dim=1) > 0
    ):
        raise RuntimeError("Empty token sequence detected.")

    print("\nTokenizer vocabulary size:", len(tokenizer))
    print("Dynamic padding: VERIFIED")
    print("Label tensor format: VERIFIED")
    print("Sequence length: VERIFIED")
    print("DATALOADER TEST PASSED")


if __name__ == "__main__":
    test_dataloaders()
