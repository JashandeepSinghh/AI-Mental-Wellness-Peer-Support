
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from config.tweeteval_config import (
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    GRADIENT_ACCUMULATION_STEPS,
    MAX_GRAD_NORM,
    RANDOM_SEED,
    DEVICE,
    USE_FP16,
    OUTPUT_DIR,
    MODEL_DIR,
    ID2LABEL,
    NUM_LABELS,
)

from scripts.model_development.sentiment.tweeteval_dataset import (
    create_dataloaders,
)

from scripts.model_development.sentiment.tweeteval_model import (
    TweetEvalModel,
)


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

def set_seed():

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# 2. METRIC CALCULATION
# ============================================================

def calculate_metrics(true_labels, predicted_labels):

    return {
        "accuracy": accuracy_score(
            true_labels,
            predicted_labels,
        ),

        "macro_f1": f1_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        ),

        "weighted_f1": f1_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        ),

        "macro_precision": precision_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        ),

        "macro_recall": recall_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        ),
    }


# ============================================================
# 3. VALIDATION
# ============================================================

def evaluate_model(
    model,
    validation_loader,
    criterion,
    epoch,
):

    model.eval()

    total_loss = 0.0
    total_examples = 0

    all_predictions = []
    all_labels = []

    progress = tqdm(
        validation_loader,
        desc=f"Epoch {epoch} Validation",
    )

    with torch.no_grad():

        for batch in progress:

            input_ids = batch["input_ids"].to(
                DEVICE,
                non_blocking=True,
            )

            attention_mask = batch["attention_mask"].to(
                DEVICE,
                non_blocking=True,
            )

            labels = batch["labels"].to(
                DEVICE,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=USE_FP16,
            ):

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                logits = outputs.logits

                loss = criterion(
                    logits.float(),
                    labels,
                )

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_examples += batch_size

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    validation_loss = total_loss / total_examples

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
    )

    metrics["validation_loss"] = validation_loss

    return metrics, all_labels, all_predictions


# ============================================================
# 4. CHECKPOINT SAVING
# ============================================================

def save_best_checkpoint(
    model,
    tokenizer,
    epoch,
    metrics,
):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = MODEL_DIR / "best_checkpoint"

    # Save model weights and configuration.
    model.model.save_pretrained(
        checkpoint_path,
    )

    # Save tokenizer alongside the model.
    tokenizer.save_pretrained(
        checkpoint_path,
    )

    checkpoint_metadata = {
        "epoch": epoch,
        "best_metric": "validation_macro_f1",
        "metrics": metrics,
        "id2label": {
            str(key): value
            for key, value in ID2LABEL.items()
        },
        "num_labels": NUM_LABELS,
        "model_name": "distilbert-base-uncased",
    }

    metadata_path = (
        checkpoint_path / "training_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            checkpoint_metadata,
            file,
            indent=4,
        )

    print(
        f"\nBest checkpoint saved: {checkpoint_path}"
    )


# ============================================================
# 5. TRAINING ENGINE
# ============================================================

def train_model():

    set_seed()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("TWEETEVAL DISTILBERT TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        tokenizer,
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader,
    ) = create_dataloaders()

    # --------------------------------------------------------
    # Initialize model
    # --------------------------------------------------------

    print("\nInitializing model...")

    model = TweetEvalModel().to(DEVICE)

    # CrossEntropyLoss expects integer class labels.
    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    updates_per_epoch = math.ceil(
        len(train_loader) / GRADIENT_ACCUMULATION_STEPS
    )

    total_training_steps = (
        updates_per_epoch * EPOCHS
    )

    warmup_steps = int(
        total_training_steps * WARMUP_RATIO
    )

    def lr_lambda(current_step):

        if current_step < warmup_steps:

            return float(current_step) / max(
                1,
                warmup_steps,
            )

        remaining_steps = (
            total_training_steps - current_step
        )

        decay_steps = max(
            1,
            total_training_steps - warmup_steps,
        )

        return max(
            0.0,
            float(remaining_steps) / decay_steps,
        )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda,
    )

    # --------------------------------------------------------
    # Mixed precision scaler
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_FP16,
    )

    # --------------------------------------------------------
    # TensorBoard
    # --------------------------------------------------------

    log_dir = OUTPUT_DIR / "tensorboard"

    writer = SummaryWriter(
        log_dir=str(log_dir),
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    best_macro_f1 = -1.0
    patience_counter = 0
    early_stopping_patience = 2

    history = []

    optimizer.zero_grad(
        set_to_none=True,
    )

    global_step = 0

    print("\nTRAINING CONFIGURATION")
    print("-" * 70)

    print("Device:", DEVICE)
    print("FP16:", USE_FP16)
    print("Epochs:", EPOCHS)
    print("Training batches:", len(train_loader))
    print("Updates per epoch:", updates_per_epoch)
    print("Total optimizer steps:", total_training_steps)
    print("Warmup steps:", warmup_steps)
    print("Learning rate:", LEARNING_RATE)
    print("Gradient accumulation:", GRADIENT_ACCUMULATION_STEPS)

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.time()

        model.train()

        total_training_loss = 0.0
        total_training_examples = 0

        progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{EPOCHS}",
        )

        for batch_index, batch in progress:

            input_ids = batch["input_ids"].to(
                DEVICE,
                non_blocking=True,
            )

            attention_mask = batch["attention_mask"].to(
                DEVICE,
                non_blocking=True,
            )

            labels = batch["labels"].to(
                DEVICE,
                non_blocking=True,
            )

            batch_size = labels.size(0)

            # ------------------------------------------------
            # Determine accumulation group size.
            # Handles the final incomplete accumulation group.
            # ------------------------------------------------

            group_start = (
                batch_index // GRADIENT_ACCUMULATION_STEPS
            ) * GRADIENT_ACCUMULATION_STEPS

            group_size = min(
                GRADIENT_ACCUMULATION_STEPS,
                len(train_loader) - group_start,
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=USE_FP16,
            ):

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                logits = outputs.logits

                loss = criterion(
                    logits.float(),
                    labels,
                )

                scaled_loss = loss / group_size

            # ------------------------------------------------
            # Backward pass
            # ------------------------------------------------

            scaler.scale(
                scaled_loss
            ).backward()

            total_training_loss += (
                loss.item() * batch_size
            )

            total_training_examples += batch_size

            # ------------------------------------------------
            # Optimizer update
            # ------------------------------------------------

            is_accumulation_boundary = (
                (batch_index + 1)
                % GRADIENT_ACCUMULATION_STEPS == 0
            )

            is_last_batch = (
                batch_index + 1 == len(train_loader)
            )

            if (
                is_accumulation_boundary
                or is_last_batch
            ):

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True,
                )

                scheduler.step()

                global_step += 1

            current_lr = scheduler.get_last_lr()[0]

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{current_lr:.2e}",
            )

        # ----------------------------------------------------
        # Epoch training statistics
        # ----------------------------------------------------

        training_loss = (
            total_training_loss
            / total_training_examples
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_metrics, true_labels, predictions = (
            evaluate_model(
                model,
                validation_loader,
                criterion,
                epoch,
            )
        )

        epoch_duration = (
            time.time() - epoch_start
        )

        current_macro_f1 = (
            validation_metrics["macro_f1"]
        )

        # ----------------------------------------------------
        # Save TensorBoard metrics
        # ----------------------------------------------------

        writer.add_scalar(
            "Loss/Training",
            training_loss,
            epoch,
        )

        writer.add_scalar(
            "Loss/Validation",
            validation_metrics["validation_loss"],
            epoch,
        )

        writer.add_scalar(
            "Metrics/Validation_Macro_F1",
            current_macro_f1,
            epoch,
        )

        writer.add_scalar(
            "Metrics/Validation_Weighted_F1",
            validation_metrics["weighted_f1"],
            epoch,
        )

        writer.add_scalar(
            "Metrics/Validation_Accuracy",
            validation_metrics["accuracy"],
            epoch,
        )

        writer.add_scalar(
            "Learning_Rate",
            scheduler.get_last_lr()[0],
            epoch,
        )

        # ----------------------------------------------------
        # Record history
        # ----------------------------------------------------

        epoch_results = {
            "epoch": epoch,
            "training_loss": training_loss,
            **validation_metrics,
            "duration_seconds": epoch_duration,
            "optimizer_steps": global_step,
        }

        history.append(
            epoch_results
        )

        # ----------------------------------------------------
        # Display epoch results
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print(f"EPOCH {epoch} RESULTS")
        print("=" * 70)

        print(f"Training Loss: {training_loss:.6f}")
        print(
            f"Validation Loss: "
            f"{validation_metrics['validation_loss']:.6f}"
        )

        print(
            f"Accuracy: "
            f"{validation_metrics['accuracy']:.6f}"
        )

        print(
            f"Macro-F1: "
            f"{validation_metrics['macro_f1']:.6f}"
        )

        print(
            f"Weighted-F1: "
            f"{validation_metrics['weighted_f1']:.6f}"
        )

        print(
            f"Macro Precision: "
            f"{validation_metrics['macro_precision']:.6f}"
        )

        print(
            f"Macro Recall: "
            f"{validation_metrics['macro_recall']:.6f}"
        )

        print(
            f"Duration: {epoch_duration:.2f} seconds"
        )

        print(
            f"Optimizer steps: {global_step}"
        )

        # ----------------------------------------------------
        # Best checkpoint selection
        # ----------------------------------------------------

        if current_macro_f1 > best_macro_f1:

            best_macro_f1 = current_macro_f1

            patience_counter = 0

            save_best_checkpoint(
                model,
                tokenizer,
                epoch,
                validation_metrics,
            )

            print("New best validation Macro-F1.")

        else:

            patience_counter += 1

            print(
                "No improvement. Patience:",
                patience_counter,
                "/",
                early_stopping_patience,
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if patience_counter >= early_stopping_patience:

            print(
                "\nEarly stopping triggered."
            )

            break

    # ========================================================
    # 6. SAVE TRAINING HISTORY
    # ========================================================

    history_path = (
        OUTPUT_DIR / "training_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
        )

    # ========================================================
    # 7. FINAL TRAINING SUMMARY
    # ========================================================

    writer.close()

    print("\n" + "=" * 70)
    print("TWEETEVAL TRAINING COMPLETED")
    print("=" * 70)

    print("Best Validation Macro-F1:", best_macro_f1)
    print("Total optimizer steps:", global_step)

    print("Best checkpoint:", MODEL_DIR / "best_checkpoint")
    print("Training history:", history_path)
    print("TensorBoard logs:", log_dir)

    if torch.cuda.is_available():

        print(
            "Peak GPU memory:",
            round(
                torch.cuda.max_memory_allocated() / (1024 ** 3),
                3,
            ),
            "GB",
        )

    print("\nTRAINING ENGINE FINISHED")


# ============================================================
# 8. EXECUTION
# ============================================================

if __name__ == "__main__":

    train_model()
