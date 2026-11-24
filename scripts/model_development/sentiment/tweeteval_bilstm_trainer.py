
import argparse
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from config.tweeteval_bilstm_config import (
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    EARLY_STOPPING_PATIENCE,
    RANDOM_SEED,
    DEVICE,
    OUTPUT_DIR,
    MODEL_DIR,
    ID2LABEL,
    NUM_CLASSES,
    VOCABULARY_NAME,
    CHECKPOINT_NAME,
    HISTORY_NAME,
)

from scripts.model_development.sentiment.tweeteval_bilstm_dataset import (
    create_bilstm_dataloaders,
)

from scripts.model_development.sentiment.tweeteval_bilstm_model import (
    TweetEvalBiLSTM,
)


# ============================================================
# 1. TRAINING CONSTANTS
# ============================================================

WARMUP_RATIO = 0.10

CLASS_IDS = list(range(NUM_CLASSES))

CLASS_NAMES = [
    ID2LABEL[index]
    for index in CLASS_IDS
]


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed():

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            RANDOM_SEED
        )


# ============================================================
# 3. METRIC CALCULATION
# ============================================================

def calculate_metrics(
    true_labels,
    predicted_labels,
):

    return {
        "accuracy": float(
            accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),

        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),

        "weighted_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="weighted",
                zero_division=0,
            )
        ),

        "macro_precision": float(
            precision_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),

        "macro_recall": float(
            recall_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


# ============================================================
# 4. DEVICE TRANSFER
# ============================================================

def move_batch_to_device(batch):

    input_ids = batch["input_ids"].to(
        DEVICE,
        non_blocking=True,
    )

    lengths = batch["lengths"]

    labels = batch["labels"].to(
        DEVICE,
        non_blocking=True,
    )

    return input_ids, lengths, labels


# ============================================================
# 5. VALIDATION ENGINE
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

    all_true_labels = []
    all_predictions = []

    progress = tqdm(
        validation_loader,
        desc=f"Epoch {epoch} Validation",
    )

    with torch.inference_mode():

        for batch in progress:

            input_ids, lengths, labels = (
                move_batch_to_device(batch)
            )

            logits = model(
                input_ids,
                lengths,
            )

            loss = criterion(
                logits,
                labels,
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_examples += batch_size

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_true_labels.extend(
                labels.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

    if total_examples == 0:
        raise RuntimeError(
            "Validation dataset is empty."
        )

    validation_loss = (
        total_loss / total_examples
    )

    metrics = calculate_metrics(
        all_true_labels,
        all_predictions,
    )

    metrics["validation_loss"] = float(
        validation_loss
    )

    return (
        metrics,
        all_true_labels,
        all_predictions,
    )


# ============================================================
# 6. CHECKPOINT SAVING
# ============================================================

def save_best_checkpoint(
    model,
    vocabulary,
    epoch,
    metrics,
):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        MODEL_DIR / CHECKPOINT_NAME
    )

    temporary_path = (
        MODEL_DIR / f"{CHECKPOINT_NAME}.tmp"
    )

    checkpoint = {
        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "vocabulary_size": len(vocabulary),

        "vocabulary": vocabulary,

        "metrics": metrics,

        "best_metric": "validation_macro_f1",

        "model_config": {
            "embedding_dim": model.embedding.embedding_dim,
            "hidden_dim": model.lstm.hidden_size,
            "num_layers": model.lstm.num_layers,
            "bidirectional": model.lstm.bidirectional,
            "num_classes": NUM_CLASSES,
            "vocabulary_size": len(vocabulary),
        },

        "label_mapping": ID2LABEL,
    }

    # Save to a temporary file first.
    torch.save(
        checkpoint,
        temporary_path,
    )

    # Replace the previous checkpoint only after
    # the new checkpoint has been written.
    temporary_path.replace(
        checkpoint_path
    )

    print(
        "\nBest checkpoint saved:",
        checkpoint_path,
    )


# ============================================================
# 7. ONE-STEP SMOKE TEST
# ============================================================

def run_smoke_test():

    print("=" * 70)
    print("TWEETEVAL BILSTM TRAINING SMOKE TEST")
    print("=" * 70)

    set_seed()

    (
        vocabulary,
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader,
    ) = create_bilstm_dataloaders()

    model = TweetEvalBiLSTM(
        len(vocabulary)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # One training batch.
    # --------------------------------------------------------

    model.train()

    batch = next(
        iter(train_loader)
    )

    input_ids, lengths, labels = (
        move_batch_to_device(batch)
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = model(
        input_ids,
        lengths,
    )

    loss = criterion(
        logits,
        labels,
    )

    if not torch.isfinite(loss).item():
        raise RuntimeError(
            "Smoke test training loss is not finite."
        )

    loss.backward()

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        MAX_GRAD_NORM,
    )

    if not torch.isfinite(gradient_norm).item():
        raise RuntimeError(
            "Non-finite gradients detected."
        )

    optimizer.step()

    print("\nTRAINING SMOKE TEST")

    print("Input shape:", tuple(input_ids.shape))
    print("Logits shape:", tuple(logits.shape))
    print("Training loss:", loss.item())
    print("Gradient norm:", gradient_norm.item())

    # --------------------------------------------------------
    # One validation batch.
    # --------------------------------------------------------

    model.eval()

    validation_batch = next(
        iter(validation_loader)
    )

    val_input_ids, val_lengths, val_labels = (
        move_batch_to_device(validation_batch)
    )

    with torch.inference_mode():

        val_logits = model(
            val_input_ids,
            val_lengths,
        )

        val_loss = criterion(
            val_logits,
            val_labels,
        )

    if not torch.isfinite(val_loss).item():
        raise RuntimeError(
            "Smoke test validation loss is not finite."
        )

    print("\nVALIDATION SMOKE TEST")

    print(
        "Validation logits shape:",
        tuple(val_logits.shape),
    )

    print(
        "Validation loss:",
        val_loss.item(),
    )

    # --------------------------------------------------------
    # Verify gradients.
    # --------------------------------------------------------

    if model.embedding.weight.grad is None:
        raise RuntimeError(
            "Embedding gradients are missing."
        )

    if model.lstm.weight_ih_l0.grad is None:
        raise RuntimeError(
            "LSTM gradients are missing."
        )

    if model.classifier.weight.grad is None:
        raise RuntimeError(
            "Classifier gradients are missing."
        )

    print("\nGradient verification: PASSED")
    print("Optimizer update: PASSED")
    print("Validation forward pass: PASSED")

    print("\nSMOKE TEST PASSED")
    print("No checkpoint or training history was modified.")
    print("=" * 70)


# ============================================================
# 8. FULL TRAINING ENGINE
# ============================================================

def train_model():

    set_seed()

    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("TWEETEVAL BILSTM FULL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets and vocabulary.
    # --------------------------------------------------------

    (
        vocabulary,
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader,
    ) = create_bilstm_dataloaders()

    # --------------------------------------------------------
    # Initialize model.
    # --------------------------------------------------------

    model = TweetEvalBiLSTM(
        len(vocabulary)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Initialize optimizer.
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Learning-rate scheduler.
    # --------------------------------------------------------

    total_training_steps = (
        len(train_loader) * EPOCHS
    )

    warmup_steps = int(
        total_training_steps * WARMUP_RATIO
    )

    def lr_lambda(current_step):

        if current_step < warmup_steps:

            return (
                float(current_step)
                / max(1, warmup_steps)
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
    # TensorBoard.
    # --------------------------------------------------------

    log_dir = (
        OUTPUT_DIR / "tensorboard"
    )

    writer = SummaryWriter(
        log_dir=str(log_dir)
    )

    # --------------------------------------------------------
    # Training state.
    # --------------------------------------------------------

    best_macro_f1 = -1.0

    patience_counter = 0

    history = []

    global_step = 0

    # --------------------------------------------------------
    # Display configuration.
    # --------------------------------------------------------

    print("\nTRAINING CONFIGURATION")
    print("-" * 70)

    print("Device:", DEVICE)
    print("Epochs:", EPOCHS)
    print("Batch size:", train_loader.batch_size)
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Warmup steps:", warmup_steps)
    print("Total optimizer steps:", total_training_steps)
    print("Loss:", "CrossEntropyLoss")
    print("Precision:", "FP32")

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    try:

        for epoch in range(1, EPOCHS + 1):

            epoch_start = time.time()

            model.train()

            total_training_loss = 0.0
            total_training_examples = 0

            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{EPOCHS}",
            )

            # ------------------------------------------------
            # Training batches.
            # ------------------------------------------------

            for batch in progress:

                input_ids, lengths, labels = (
                    move_batch_to_device(batch)
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                logits = model(
                    input_ids,
                    lengths,
                )

                loss = criterion(
                    logits,
                    labels,
                )

                if not torch.isfinite(loss).item():
                    raise RuntimeError(
                        f"Non-finite training loss at epoch {epoch}."
                    )

                loss.backward()

                gradient_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        MAX_GRAD_NORM,
                    )
                )

                if not torch.isfinite(gradient_norm).item():
                    raise RuntimeError(
                        "Non-finite gradient norm detected."
                    )

                optimizer.step()

                scheduler.step()

                global_step += 1

                batch_size = labels.size(0)

                total_training_loss += (
                    loss.item() * batch_size
                )

                total_training_examples += (
                    batch_size
                )

                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

            # ------------------------------------------------
            # Training loss.
            # ------------------------------------------------

            training_loss = (
                total_training_loss
                / total_training_examples
            )

            # ------------------------------------------------
            # Validation.
            # ------------------------------------------------

            (
                validation_metrics,
                true_labels,
                predictions,
            ) = evaluate_model(
                model,
                validation_loader,
                criterion,
                epoch,
            )

            epoch_duration = (
                time.time() - epoch_start
            )

            current_macro_f1 = (
                validation_metrics["macro_f1"]
            )

            # ------------------------------------------------
            # TensorBoard logging.
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Save epoch history.
            # ------------------------------------------------

            epoch_results = {
                "epoch": epoch,
                "training_loss": float(training_loss),
                **validation_metrics,
                "duration_seconds": float(epoch_duration),
                "optimizer_steps": global_step,
            }

            history.append(
                epoch_results
            )

            # ------------------------------------------------
            # Display results.
            # ------------------------------------------------

            print("\n" + "=" * 70)
            print(f"EPOCH {epoch} RESULTS")
            print("=" * 70)

            print(
                f"Training Loss: {training_loss:.6f}"
            )

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

            # ------------------------------------------------
            # Best checkpoint selection.
            # ------------------------------------------------

            if current_macro_f1 > best_macro_f1:

                best_macro_f1 = current_macro_f1

                patience_counter = 0

                save_best_checkpoint(
                    model,
                    vocabulary,
                    epoch,
                    validation_metrics,
                )

                print(
                    "New best validation Macro-F1."
                )

            else:

                patience_counter += 1

                print(
                    "No improvement. Patience:",
                    patience_counter,
                    "/",
                    EARLY_STOPPING_PATIENCE,
                )

            # ------------------------------------------------
            # Save history after every epoch.
            # ------------------------------------------------

            history_path = (
                OUTPUT_DIR / HISTORY_NAME
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

            # ------------------------------------------------
            # Early stopping.
            # ------------------------------------------------

            if (
                patience_counter
                >= EARLY_STOPPING_PATIENCE
            ):

                print(
                    "\nEarly stopping triggered."
                )

                break

    finally:

        writer.close()

    # ========================================================
    # FINAL TRAINING SUMMARY
    # ========================================================

    history_path = (
        OUTPUT_DIR / HISTORY_NAME
    )

    print("\n" + "=" * 70)
    print("TWEETEVAL BILSTM TRAINING COMPLETED")
    print("=" * 70)

    print("Best Validation Macro-F1:", best_macro_f1)
    print("Total optimizer steps:", global_step)

    print(
        "Best checkpoint:",
        MODEL_DIR / CHECKPOINT_NAME,
    )

    print("Training history:", history_path)
    print("TensorBoard logs:", log_dir)

    if torch.cuda.is_available():

        print(
            "Peak GPU memory:",
            round(
                torch.cuda.max_memory_allocated()
                / (1024 ** 3),
                3,
            ),
            "GB",
        )

    print("\nTRAINING ENGINE FINISHED")


# ============================================================
# 9. EXECUTION
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one training and validation batch only.",
    )

    args = parser.parse_args()

    if args.smoke_test:

        run_smoke_test()

    else:

        train_model()
