
"""
FIG-Loneliness Dataset and DataLoader
=====================================

Purpose:
    Load, validate, tokenize and batch the FIG-Loneliness dataset
    for DistilBERT binary sequence classification.

Labels:
    0 = Non-lonely
    1 = Lonely

Features:
    - Dataset integrity validation
    - DistilBERT tokenization
    - Maximum sequence length enforcement
    - Dynamic batch padding
    - Reproducible training shuffle
    - Separate train, validation and test loaders
"""

import logging
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from scripts.model_development.loneliness.config import (
    fig_loneliness_config as config
)


# ============================================================
# 1. LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducible data loading."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. DATASET VALIDATION
# ============================================================

def validate_dataframe(
    dataframe: pd.DataFrame,
    split_name: str
) -> None:
    """
    Validate the standardized dataset before tokenization.
    """

    required_columns = [
        config.TEXT_COLUMN,
        config.LABEL_COLUMN
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{split_name}: Missing columns: {missing_columns}"
        )

    if dataframe.empty:
        raise ValueError(
            f"{split_name}: Dataset is empty."
        )

    if dataframe[required_columns].isnull().any().any():
        raise ValueError(
            f"{split_name}: Missing text or label values detected."
        )

    if not dataframe[config.TEXT_COLUMN].map(
        lambda value: isinstance(value, str)
    ).all():
        raise TypeError(
            f"{split_name}: Every text value must be a string."
        )

    if dataframe[config.TEXT_COLUMN].str.strip().eq("").any():
        raise ValueError(
            f"{split_name}: Empty text records detected."
        )

    labels = set(
        dataframe[config.LABEL_COLUMN].unique().tolist()
    )

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"{split_name}: Invalid labels detected: {labels}"
        )

    if len(labels) != 2:
        raise ValueError(
            f"{split_name}: Both binary classes must be present."
        )

    if not pd.api.types.is_integer_dtype(
        dataframe[config.LABEL_COLUMN]
    ):
        raise TypeError(
            f"{split_name}: Labels must use an integer data type."
        )

    expected_size = config.EXPECTED_SPLIT_SIZES[split_name]

    if len(dataframe) != expected_size:
        raise ValueError(
            f"{split_name}: Expected {expected_size} records, "
            f"found {len(dataframe)}."
        )

    logger.info(
        "%s dataset validation passed: %d records",
        split_name,
        len(dataframe)
    )


# ============================================================
# 4. LOAD DATASET SPLITS
# ============================================================

def load_dataset_splits() -> Dict[str, pd.DataFrame]:
    """
    Load and validate the official standardized splits.
    """

    file_paths = {
        "train": config.TRAIN_FILE,
        "validation": config.VALIDATION_FILE,
        "test": config.TEST_FILE
    }

    datasets = {}

    for split_name, file_path in file_paths.items():

        if not file_path.exists():
            raise FileNotFoundError(
                f"{split_name} dataset not found: {file_path}"
            )

        dataframe = pd.read_csv(file_path)

        validate_dataframe(
            dataframe=dataframe,
            split_name=split_name
        )

        # Compare current label distribution with the
        # distribution recorded during dataset preparation.
        expected_distribution = (
            config.EXPECTED_LABEL_DISTRIBUTION[split_name]
        )

        actual_distribution = (
            dataframe[config.LABEL_COLUMN]
            .value_counts()
            .to_dict()
        )

        if actual_distribution != expected_distribution:
            raise ValueError(
                f"{split_name}: Label distribution mismatch. "
                f"Expected {expected_distribution}, "
                f"found {actual_distribution}."
            )

        datasets[split_name] = dataframe

        logger.info(
            "%s split loaded successfully.",
            split_name
        )

    return datasets


# ============================================================
# 5. TOKENIZER LOADING
# ============================================================

def load_tokenizer():
    """Load the pretrained DistilBERT tokenizer."""

    logger.info(
        "Loading tokenizer: %s",
        config.MODEL_NAME
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        use_fast=True
    )

    if tokenizer.pad_token_id is None:
        raise ValueError(
            "Tokenizer does not have a padding token."
        )

    logger.info(
        "Tokenizer loaded successfully. Vocabulary size: %d",
        len(tokenizer)
    )

    return tokenizer


# ============================================================
# 6. FIG-LONELINESS PYTORCH DATASET
# ============================================================

class FigLonelinessDataset(Dataset):
    """
    PyTorch Dataset for FIG-Loneliness.

    Input:
        Text

    Target:
        Binary loneliness label
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        split_name: str
    ):

        self.split_name = split_name

        texts = dataframe[
            config.TEXT_COLUMN
        ].tolist()

        self.labels = dataframe[
            config.LABEL_COLUMN
        ].astype(int).tolist()

        logger.info(
            "Tokenizing %s split: %d records",
            split_name,
            len(texts)
        )

        # Tokenize without padding here.
        # Padding will happen dynamically inside each batch.
        self.encodings = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=config.TOKENIZER_TRUNCATION,
            max_length=config.MAX_LENGTH,
            padding=False,
            return_attention_mask=True,
            verbose=False
        )

        if len(self.encodings["input_ids"]) != len(self.labels):
            raise RuntimeError(
                f"{split_name}: Tokenization record count mismatch."
            )

        logger.info(
            "%s tokenization completed.",
            split_name
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict:

        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": self.labels[index]
        }


# ============================================================
# 7. DYNAMIC PADDING COLLATOR
# ============================================================

class FigLonelinessCollator:
    """
    Dynamically pad sequences to the longest sequence
    in the current batch.
    """

    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:

        labels = torch.tensor(
            [item["labels"] for item in batch],
            dtype=torch.long
        )

        features = [
            {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"]
            }
            for item in batch
        ]

        padded_batch = self.tokenizer.pad(
            features,
            padding=config.TOKENIZER_PADDING,
            return_tensors="pt"
        )

        padded_batch["labels"] = labels

        return padded_batch


# ============================================================
# 8. CREATE PYTORCH DATASETS
# ============================================================

def create_pytorch_datasets(
    dataframes: Dict[str, pd.DataFrame],
    tokenizer
) -> Dict[str, FigLonelinessDataset]:

    datasets = {}

    for split_name, dataframe in dataframes.items():

        datasets[split_name] = FigLonelinessDataset(
            dataframe=dataframe,
            tokenizer=tokenizer,
            split_name=split_name
        )

    return datasets


# ============================================================
# 9. CREATE DATALOADERS
# ============================================================

def create_dataloaders(
    datasets: Dict[str, FigLonelinessDataset],
    tokenizer
) -> Dict[str, DataLoader]:

    collator = FigLonelinessCollator(tokenizer)

    generator = torch.Generator()
    generator.manual_seed(config.RANDOM_SEED)

    dataloaders = {}

    for split_name, dataset in datasets.items():

        is_training = split_name == "train"

        batch_size = (
            config.TRAIN_BATCH_SIZE
            if is_training
            else config.EVAL_BATCH_SIZE
        )

        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_training,
            collate_fn=collator,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            drop_last=False,
            generator=generator if is_training else None
        )

        logger.info(
            "%s DataLoader created: %d batches",
            split_name,
            len(dataloaders[split_name])
        )

    return dataloaders


# ============================================================
# 10. DATA PIPELINE INITIALIZATION
# ============================================================

def create_data_pipeline():

    set_seed(config.RANDOM_SEED)

    config.validate_config()

    logger.info("=" * 65)
    logger.info("FIG-LONELINESS DATA PIPELINE")
    logger.info("=" * 65)

    dataframes = load_dataset_splits()

    tokenizer = load_tokenizer()

    datasets = create_pytorch_datasets(
        dataframes=dataframes,
        tokenizer=tokenizer
    )

    dataloaders = create_dataloaders(
        datasets=datasets,
        tokenizer=tokenizer
    )

    return tokenizer, datasets, dataloaders


# ============================================================
# 11. DATA PIPELINE SMOKE TEST
# ============================================================

def run_smoke_test(dataloaders):

    logger.info("=" * 65)
    logger.info("DATA PIPELINE SMOKE TEST")
    logger.info("=" * 65)

    for split_name, dataloader in dataloaders.items():

        batch = next(iter(dataloader))

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Basic tensor validation
        assert input_ids.ndim == 2
        assert attention_mask.shape == input_ids.shape
        assert labels.ndim == 1

        assert input_ids.shape[0] <= (
            config.TRAIN_BATCH_SIZE
        )

        assert input_ids.shape[1] <= config.MAX_LENGTH

        assert labels.dtype == torch.long

        assert set(labels.tolist()).issubset({0, 1})

        assert torch.all(
            attention_mask.sum(dim=1) > 0
        )

        logger.info("%s batch validation passed.", split_name)

        print(f"\n{split_name.upper()} BATCH")
        print("-" * 45)

        print(f"Input IDs shape     : {tuple(input_ids.shape)}")
        print(f"Attention mask shape: {tuple(attention_mask.shape)}")
        print(f"Labels shape        : {tuple(labels.shape)}")

        print(f"Labels              : {labels.tolist()}")

        print(
            "Sequence lengths    : "
            f"{attention_mask.sum(dim=1).tolist()}"
        )

        print(
            "Number of padded tokens: "
            f"{int((attention_mask == 0).sum().item())}"
        )

    logger.info("=" * 65)
    logger.info("ALL DATA PIPELINE SMOKE TESTS PASSED")
    logger.info("=" * 65)


# ============================================================
# 12. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    tokenizer, datasets, dataloaders = create_data_pipeline()

    run_smoke_test(dataloaders)

    print("\nDATASET SUMMARY")
    print("=" * 50)

    for split_name, dataset in datasets.items():

        print(
            f"{split_name:<12}: "
            f"{len(dataset):>6} records | "
            f"{len(dataloaders[split_name]):>4} batches"
        )

    print("\nFIG-LONELINESS DATA PIPELINE COMPLETED.")
