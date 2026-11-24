
import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from config.goemotions_config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    GRADIENT_ACCUMULATION_STEPS,
    USE_FP16,
    RANDOM_SEED,
    LABEL_COLUMNS,
    MODEL_DIR,
    OUTPUT_DIR,
)

from scripts.model_development.emotion.goemotions_dataset import (
    train_loader,
    validation_loader,
    tokenizer,
)

from scripts.model_development.emotion.goemotions_model import (
    GoEmotionsModel,
)


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility without forcing deterministic CUDA kernels.
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. METRIC CALCULATION
# ============================================================

def calculate_metrics(all_targets, all_probabilities):
    """
    Calculates multi-label metrics using a 0.5 threshold.

    Each emotion is evaluated independently.
    """

    targets = np.concatenate(all_targets, axis=0)
    probabilities = np.concatenate(all_probabilities, axis=0)

    predictions = (probabilities >= 0.5).astype(np.int32)

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        targets,
        predictions,
        average="micro",
        zero_division=0,
    )

    macro_precision = precision_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    no_prediction_rate = float(
        np.mean(np.sum(predictions, axis=1) == 0)
    )

    per_label = {}

    for index, label in enumerate(LABEL_COLUMNS):
        per_label[label] = {
            "precision": float(
                precision_score(
                    targets[:, index],
                    predictions[:, index],
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    targets[:, index],
                    predictions[:, index],
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    targets[:, index],
                    predictions[:, index],
                    zero_division=0,
                )
            ),
            "support": int(targets[:, index].sum()),
        }

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "no_prediction_rate": no_prediction_rate,
        "per_label": per_label,
    }


# ============================================================
# 3. BATCH TRANSFER
# ============================================================

def move_batch_to_device(batch):
    return {
        key: value.to(
            DEVICE,
            non_blocking=(DEVICE.type == "cuda"),
        )
        for key, value in batch.items()
    }


# ============================================================
# 4. VALIDATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    data_loader,
    scaler,
    max_batches=None,
):
    model.eval()

    total_loss = 0.0
    total_examples = 0

    all_targets = []
    all_probabilities = []

    use_amp = USE_FP16 and DEVICE.type == "cuda"

    with torch.inference_mode():

        progress = tqdm(
            data_loader,
            desc="Validation",
            leave=False,
            total=min(len(data_loader), max_batches)
            if max_batches is not None
            else len(data_loader),
        )

        for batch_index, batch in enumerate(progress):

            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break

            batch = move_batch_to_device(batch)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                loss = outputs.loss

            batch_size = batch["labels"].size(0)

            total_loss += loss.item() * batch_size
            total_examples += batch_size

            probabilities = torch.sigmoid(
                outputs.logits.float()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_targets.append(
                batch["labels"].cpu().numpy()
            )

    if total_examples == 0:
        raise RuntimeError("Validation processed zero examples.")

    metrics = calculate_metrics(
        all_targets,
        all_probabilities,
    )

    metrics["loss"] = total_loss / total_examples
    metrics["examples"] = total_examples

    return metrics


# ============================================================
# 5. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    data_loader,
    optimizer,
    scheduler,
    scaler,
    epoch,
    writer,
    global_step,
    max_batches=None,
):
    model.train()

    total_loss = 0.0
    total_examples = 0

    optimizer.zero_grad(set_to_none=True)

    use_amp = USE_FP16 and DEVICE.type == "cuda"

    number_of_batches = (
        min(len(data_loader), max_batches)
        if max_batches is not None
        else len(data_loader)
    )

    progress = tqdm(
        enumerate(data_loader),
        total=number_of_batches,
        desc=f"Epoch {epoch} - Training",
    )

    for batch_index, batch in progress:

        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        # Determine the actual accumulation group size.
        group_start = (
            batch_index // GRADIENT_ACCUMULATION_STEPS
        ) * GRADIENT_ACCUMULATION_STEPS

        group_size = min(
            GRADIENT_ACCUMULATION_STEPS,
            number_of_batches - group_start,
        )

        batch = move_batch_to_device(batch)

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            loss = outputs.loss

        if not torch.isfinite(loss).item():
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(
                f"Non-finite training loss at epoch {epoch}, "
                f"batch {batch_index}."
            )

        batch_size = batch["labels"].size(0)

        total_loss += loss.detach().item() * batch_size
        total_examples += batch_size

        # Scale loss for gradient accumulation.
        scaled_loss = loss / group_size

        scaler.scale(scaled_loss).backward()

        is_accumulation_boundary = (
            (batch_index + 1) % GRADIENT_ACCUMULATION_STEPS == 0
        )

        is_last_batch = (
            batch_index + 1 == number_of_batches
        )

        if is_accumulation_boundary or is_last_batch:

            # Unscale gradients before clipping.
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

            scheduler.step()

            global_step += 1

            writer.add_scalar(
                "Training/Batch_Loss",
                loss.detach().item(),
                global_step,
            )

            writer.add_scalar(
                "Training/Learning_Rate",
                scheduler.get_last_lr()[0],
                global_step,
            )

        progress.set_postfix(
            loss=f"{loss.detach().item():.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    if total_examples == 0:
        raise RuntimeError("Training processed zero examples.")

    return total_loss / total_examples, global_step


# ============================================================
# 6. CHECKPOINT SAVING
# ============================================================

def save_best_checkpoint(model, metrics, epoch):
    """
    Saves the best Hugging Face model and tokenizer.
    """

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_dir = MODEL_DIR / "best_checkpoint"
    temporary_dir = MODEL_DIR / "best_checkpoint_tmp"

    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)

    temporary_dir.mkdir(parents=True)

    model.model.save_pretrained(temporary_dir)
    tokenizer.save_pretrained(temporary_dir)

    metadata = {
        "epoch": epoch,
        "selection_metric": "validation_macro_f1",
        "validation_metrics": metrics,
        "labels": LABEL_COLUMNS,
        "model_name": "distilbert-base-uncased",
        "num_labels": len(LABEL_COLUMNS),
        "threshold": 0.5,
        "random_seed": RANDOM_SEED,
    }

    with open(
        temporary_dir / "training_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    # Replace the previous best checkpoint only after
    # the new checkpoint has been saved successfully.
    backup_dir = MODEL_DIR / "best_checkpoint_backup"

    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    if best_dir.exists():
        best_dir.rename(backup_dir)

    try:
        temporary_dir.rename(best_dir)

    except Exception:
        if backup_dir.exists() and not best_dir.exists():
            backup_dir.rename(best_dir)

        raise

    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    print("Best checkpoint saved:", best_dir)


# ============================================================
# 7. PER-EMOTION REPORT
# ============================================================

def save_per_label_report(metrics, epoch):
    report_dir = OUTPUT_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for label, values in metrics["per_label"].items():

        rows.append({
            "emotion": label,
            "precision": values["precision"],
            "recall": values["recall"],
            "f1": values["f1"],
            "support": values["support"],
        })

    report = pd.DataFrame(rows)

    report.to_csv(
        report_dir / f"validation_emotion_metrics_epoch_{epoch}.csv",
        index=False,
    )


# ============================================================
# 8. TRAINING ENGINE
# ============================================================

def run_training(smoke_test=False):

    set_seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_dir = OUTPUT_DIR / "tensorboard"

    writer = SummaryWriter(log_dir=str(log_dir))

    print("=" * 70)
    print("GOEMOTIONS DISTILBERT TRAINING ENGINE")
    print("=" * 70)

    print("Device:", DEVICE)
    print("Mixed precision:", USE_FP16)
    print("Training records:", len(train_loader.dataset))
    print("Validation records:", len(validation_loader.dataset))

    model = GoEmotionsModel().to(DEVICE)

    # AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Determine the number of training batches.
    batches_per_epoch = (
        min(len(train_loader), 2)
        if smoke_test
        else len(train_loader)
    )

    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / GRADIENT_ACCUMULATION_STEPS
    )

    total_training_steps = (
        optimizer_steps_per_epoch
        if smoke_test
        else optimizer_steps_per_epoch * EPOCHS
    )

    warmup_steps = int(
        total_training_steps * WARMUP_RATIO
    )

    # Linear learning-rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # FP16 gradient scaler
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_FP16 and DEVICE.type == "cuda",
    )

    if smoke_test:

        print("\nRUNNING TRAINING SMOKE TEST")
        print("Training batches: 2")
        print("Validation batches: 2")

        train_loss, global_step = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=1,
            writer=writer,
            global_step=0,
            max_batches=2,
        )

        validation_metrics = evaluate_model(
            model=model,
            data_loader=validation_loader,
            scaler=scaler,
            max_batches=2,
        )

        print("\nSMOKE TEST RESULTS")
        print("Training loss:", round(train_loss, 6))
        print(
            "Validation loss:",
            round(validation_metrics["loss"], 6),
        )
        print(
            "Validation Macro-F1:",
            round(validation_metrics["macro_f1"], 6),
        )
        print(
            "Validation Micro-F1:",
            round(validation_metrics["micro_f1"], 6),
        )
        print("Optimizer steps:", global_step)

        if DEVICE.type == "cuda":
            print(
                "Peak GPU memory:",
                round(
                    torch.cuda.max_memory_allocated() / (1024 ** 3),
                    3,
                ),
                "GB",
            )

        print("\nTRAINING SMOKE TEST PASSED")
        return

    # --------------------------------------------------------
    # FULL TRAINING
    # --------------------------------------------------------

    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    patience = 2
    global_step = 0

    history = []

    for epoch in range(1, EPOCHS + 1):

        print(f"\n{'=' * 70}")
        print(f"EPOCH {epoch}/{EPOCHS}")
        print("=" * 70)

        epoch_start = time.time()

        train_loss, global_step = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            writer=writer,
            global_step=global_step,
        )

        validation_metrics = evaluate_model(
            model=model,
            data_loader=validation_loader,
            scaler=scaler,
        )

        epoch_duration = time.time() - epoch_start

        print("\nEPOCH RESULTS")
        print("Training loss:", round(train_loss, 6))
        print(
            "Validation loss:",
            round(validation_metrics["loss"], 6),
        )
        print(
            "Validation Macro-F1:",
            round(validation_metrics["macro_f1"], 6),
        )
        print(
            "Validation Micro-F1:",
            round(validation_metrics["micro_f1"], 6),
        )
        print(
            "Validation no-prediction rate:",
            round(validation_metrics["no_prediction_rate"], 6),
        )
        print("Duration:", round(epoch_duration, 2), "seconds")

        writer.add_scalar(
            "Epoch/Training_Loss",
            train_loss,
            epoch,
        )

        writer.add_scalar(
            "Epoch/Validation_Loss",
            validation_metrics["loss"],
            epoch,
        )

        writer.add_scalar(
            "Epoch/Validation_Macro_F1",
            validation_metrics["macro_f1"],
            epoch,
        )

        writer.add_scalar(
            "Epoch/Validation_Micro_F1",
            validation_metrics["micro_f1"],
            epoch,
        )

        save_per_label_report(validation_metrics, epoch)

        history.append({
            "epoch": epoch,
            "training_loss": train_loss,
            "validation_loss": validation_metrics["loss"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_micro_f1": validation_metrics["micro_f1"],
            "duration_seconds": epoch_duration,
        })

        # Select checkpoint using validation Macro-F1.
        if validation_metrics["macro_f1"] > best_macro_f1:

            best_macro_f1 = validation_metrics["macro_f1"]

            epochs_without_improvement = 0

            save_best_checkpoint(
                model=model,
                metrics=validation_metrics,
                epoch=epoch,
            )

            print("New best validation Macro-F1.")

        else:

            epochs_without_improvement += 1

            print(
                "No Macro-F1 improvement:",
                epochs_without_improvement,
                "/",
                patience,
            )

        # Save training history after every epoch.
        with open(
            OUTPUT_DIR / "training_history.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(history, file, indent=4)

        if epochs_without_improvement >= patience:

            print("\nEarly stopping activated.")
            break

    writer.close()

    print("\n" + "=" * 70)
    print("GOEMOTIONS TRAINING COMPLETED")
    print("=" * 70)

    print("Best validation Macro-F1:", round(best_macro_f1, 6))
    print("Global optimizer steps:", global_step)
    print("Best model directory:", MODEL_DIR / "best_checkpoint")
    print("TensorBoard logs:", log_dir)


# ============================================================
# 9. COMMAND-LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="GoEmotions DistilBERT training engine"
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run two training and two validation batches only.",
    )

    group.add_argument(
        "--train",
        action="store_true",
        help="Run the complete training workflow.",
    )

    args = parser.parse_args()

    try:
        run_training(smoke_test=args.smoke_test)

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
