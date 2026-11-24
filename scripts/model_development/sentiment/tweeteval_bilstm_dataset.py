
import json
import re
from collections import Counter

import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

from config.tweeteval_bilstm_config import (
    TRAIN_FILE,
    VALIDATION_FILE,
    MODEL_DIR,
    MAX_VOCAB_SIZE,
    MIN_WORD_FREQUENCY,
    MAX_SEQUENCE_LENGTH,
    BATCH_SIZE,
    PAD_TOKEN,
    UNK_TOKEN,
    PAD_INDEX,
    UNK_INDEX,
    VOCABULARY_NAME,
    LABEL2ID,
    DEVICE,
    RANDOM_SEED,
)


# ============================================================
# 1. TOKENIZATION
# ============================================================

TOKEN_PATTERN = re.compile(
    r"https?://\S+|"
    r"www\.\S+|"
    r"@[A-Za-z0-9_]+|"
    r"#[A-Za-z0-9_]+|"
    r"[A-Za-z]+(?:'[A-Za-z]+)?|"
    r"\d+(?:[.,]\d+)*|"
    r"[^\w\s]",
    flags=re.UNICODE,
)


def tokenize_text(text):

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    return TOKEN_PATTERN.findall(
        text.strip().lower()
    )


# ============================================================
# 2. LOAD AND VALIDATE CSV DATA
# ============================================================

def load_csv(csv_path):

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

    data["label"] = pd.to_numeric(
        data["label"],
        errors="coerce",
    )

    if data["label"].isnull().any():
        raise ValueError(
            "Non-numeric labels detected."
        )

    if not (
        data["label"]
        == data["label"].astype(int)
    ).all():
        raise ValueError(
            "Non-integer sentiment labels detected."
        )

    data["label"] = data["label"].astype(int)

    valid_label_ids = set(LABEL2ID.values())

    actual_label_ids = set(
        data["label"].unique()
    )

    if not actual_label_ids.issubset(valid_label_ids):
        raise ValueError(
            f"Invalid label IDs: {actual_label_ids}"
        )

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

    data["text"] = data["text"].astype(str)

    if data["text"].str.strip().eq("").any():
        raise ValueError(
            "Empty text records detected."
        )

    return data


# ============================================================
# 3. BUILD VOCABULARY FROM TRAINING DATA ONLY
# ============================================================

def build_vocabulary():

    print("=" * 65)
    print("BUILDING TWEETEVAL VOCABULARY")
    print("=" * 65)

    train_data = load_csv(TRAIN_FILE)

    word_counter = Counter()
    total_tokens = 0

    print("\nCounting training tokens...")

    for text in train_data["text"]:

        tokens = tokenize_text(text)

        word_counter.update(tokens)
        total_tokens += len(tokens)

    eligible_tokens = [
        (token, frequency)
        for token, frequency in word_counter.items()
        if frequency >= MIN_WORD_FREQUENCY
        and token not in {PAD_TOKEN, UNK_TOKEN}
    ]

    # Deterministic ordering.
    eligible_tokens.sort(
        key=lambda item: (
            -item[1],
            item[0],
        )
    )

    available_slots = MAX_VOCAB_SIZE - 2

    selected_tokens = eligible_tokens[
        :available_slots
    ]

    vocabulary = {
        PAD_TOKEN: PAD_INDEX,
        UNK_TOKEN: UNK_INDEX,
    }

    for token, _ in selected_tokens:
        vocabulary[token] = len(vocabulary)

    if len(vocabulary) > MAX_VOCAB_SIZE:
        raise RuntimeError(
            "Vocabulary exceeds configured size."
        )

    # --------------------------------------------------------
    # Correct configured model directory.
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    vocabulary_path = (
        MODEL_DIR / VOCABULARY_NAME
    )

    vocabulary_metadata = {
        "vocabulary": vocabulary,
        "vocabulary_size": len(vocabulary),
        "max_vocabulary_size": MAX_VOCAB_SIZE,
        "minimum_word_frequency": MIN_WORD_FREQUENCY,
        "pad_token": PAD_TOKEN,
        "pad_index": PAD_INDEX,
        "unk_token": UNK_TOKEN,
        "unk_index": UNK_INDEX,
        "training_records": len(train_data),
        "unique_training_tokens": len(word_counter),
        "total_training_tokens": total_tokens,
        "eligible_tokens": len(eligible_tokens),
        "retained_tokens": len(selected_tokens),
    }

    with open(
        vocabulary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            vocabulary_metadata,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("\nVOCABULARY SUMMARY")
    print("-" * 65)

    print("Training records:", len(train_data))
    print("Total training tokens:", total_tokens)
    print("Unique training tokens:", len(word_counter))
    print("Eligible tokens:", len(eligible_tokens))
    print("Retained tokens:", len(selected_tokens))
    print("Final vocabulary size:", len(vocabulary))

    print("PAD index:", vocabulary[PAD_TOKEN])
    print("UNK index:", vocabulary[UNK_TOKEN])

    print("\nVocabulary saved:", vocabulary_path)

    return vocabulary


# ============================================================
# 4. ENCODE TEXT INTO INTEGER SEQUENCES
# ============================================================

def encode_text(text, vocabulary):

    tokens = tokenize_text(text)

    token_ids = [
        vocabulary.get(token, UNK_INDEX)
        for token in tokens
    ]

    if not token_ids:
        token_ids = [UNK_INDEX]

    return token_ids[:MAX_SEQUENCE_LENGTH]


# ============================================================
# 5. TWEETEVAL BILSTM DATASET
# ============================================================

class TweetEvalBiLSTMDataset(Dataset):

    def __init__(self, csv_path, vocabulary):

        print(f"\nLoading dataset: {csv_path.name}")

        self.data = load_csv(csv_path)

        self.texts = self.data["text"].tolist()

        self.labels = self.data[
            "label"
        ].to_numpy(dtype="int64")

        self.vocabulary = vocabulary

        print("Encoding text sequences...")

        self.sequences = [
            encode_text(text, vocabulary)
            for text in self.texts
        ]

        self.lengths = [
            len(sequence)
            for sequence in self.sequences
        ]

        if len(self.sequences) != len(self.labels):
            raise RuntimeError(
                "Sequence and label counts differ."
            )

        if any(
            length <= 0 or length > MAX_SEQUENCE_LENGTH
            for length in self.lengths
        ):
            raise RuntimeError(
                "Invalid encoded sequence length."
            )

        print("Records:", len(self.labels))

        print(
            "Mean sequence length:",
            round(
                sum(self.lengths) / len(self.lengths),
                2,
            ),
        )

        print(
            "Maximum sequence length:",
            max(self.lengths),
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        return {
            "input_ids": torch.tensor(
                self.sequences[index],
                dtype=torch.long,
            ),

            "length": self.lengths[index],

            "labels": torch.tensor(
                self.labels[index],
                dtype=torch.long,
            ),
        }


# ============================================================
# 6. DYNAMIC PADDING COLLATOR
# ============================================================

class TweetEvalBiLSTMCollator:

    def __call__(self, batch):

        lengths = torch.tensor(
            [item["length"] for item in batch],
            dtype=torch.long,
        )

        max_length = int(lengths.max().item())

        batch_size = len(batch)

        input_ids = torch.full(
            (batch_size, max_length),
            fill_value=PAD_INDEX,
            dtype=torch.long,
        )

        labels = torch.empty(
            batch_size,
            dtype=torch.long,
        )

        for index, item in enumerate(batch):

            sequence = item["input_ids"]

            input_ids[
                index,
                :len(sequence)
            ] = sequence

            labels[index] = item["labels"]

        return {
            "input_ids": input_ids,
            "lengths": lengths,
            "labels": labels,
        }


# ============================================================
# 7. CREATE DATA LOADERS
# ============================================================

def create_bilstm_dataloaders():

    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    print("\nInitializing vocabulary...")

    vocabulary = build_vocabulary()

    print("\nInitializing training dataset...")

    train_dataset = TweetEvalBiLSTMDataset(
        TRAIN_FILE,
        vocabulary,
    )

    print("\nInitializing validation dataset...")

    validation_dataset = TweetEvalBiLSTMDataset(
        VALIDATION_FILE,
        vocabulary,
    )

    collator = TweetEvalBiLSTMCollator()

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=collator,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=collator,
    )

    return (
        vocabulary,
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader,
    )


# ============================================================
# 8. VERIFY DATA LOADERS
# ============================================================

def verify_bilstm_dataloaders(
    vocabulary,
    train_dataset,
    validation_dataset,
    train_loader,
    validation_loader,
):

    print("\n" + "=" * 65)
    print("BILSTM DATALOADER VERIFICATION")
    print("=" * 65)

    print("Vocabulary size:", len(vocabulary))
    print("Training records:", len(train_dataset))
    print("Validation records:", len(validation_dataset))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))

    batch = next(iter(train_loader))

    print("\nSAMPLE TRAINING BATCH")

    print("Input IDs shape:", tuple(batch["input_ids"].shape))
    print("Lengths shape:", tuple(batch["lengths"].shape))
    print("Labels shape:", tuple(batch["labels"].shape))

    print("Input IDs dtype:", batch["input_ids"].dtype)
    print("Labels dtype:", batch["labels"].dtype)

    print("Minimum sequence length:", batch["lengths"].min().item())
    print("Maximum sequence length:", batch["lengths"].max().item())

    print(
        "Unique labels:",
        torch.unique(batch["labels"]).tolist(),
    )

    assert batch["input_ids"].shape[0] == BATCH_SIZE
    assert batch["lengths"].shape == (BATCH_SIZE,)
    assert batch["labels"].shape == (BATCH_SIZE,)

    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long

    assert torch.all(batch["lengths"] > 0)
    assert torch.all(batch["lengths"] <= MAX_SEQUENCE_LENGTH)

    assert torch.all(
        batch["lengths"] <= batch["input_ids"].shape[1]
    )

    for index, length in enumerate(batch["lengths"].tolist()):

        padding = batch["input_ids"][index, length:]

        assert torch.all(padding == PAD_INDEX)

    assert torch.all(
        (batch["labels"] >= 0)
        & (batch["labels"] < len(LABEL2ID))
    )

    print("\nALL DATALOADER CHECKS PASSED")
    print("=" * 65)


# ============================================================
# 9. EXECUTION
# ============================================================

if __name__ == "__main__":

    components = create_bilstm_dataloaders()

    verify_bilstm_dataloaders(*components)

    print("\nTWEETEVAL BILSTM DATA PIPELINE READY")
