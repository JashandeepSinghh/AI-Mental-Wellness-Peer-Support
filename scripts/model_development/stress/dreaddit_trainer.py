
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from config.dreaddit_config import (
    ROOT,
    EPOCHS,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    MAX_GRAD_NORM,
    EARLY_STOPPING_PATIENCE,
    RANDOM_SEED,
    DEVICE,
    USE_FP16,
    MODEL_DIR,
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    TENSORBOARD_DIR,
    HISTORY_FILE,
)

from scripts.model_development.stress.dreaddit_dataset import (
    create_dataloaders,
)

from scripts.model_development.stress.dreaddit_model import (
    DreadditStressModel,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(actual, predicted):

    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=[0, 1],
        zero_division=0,
    )

    return {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),

        "macro_precision": float(
            np.mean(precision)
        ),

        "macro_recall": float(
            np.mean(recall)
        ),

        "macro_f1": float(
            np.mean(f1)
        ),

        "weighted_f1": float(
            precision_recall_fscore_support(
                actual,
                predicted,
                average="weighted",
                zero_division=0,
            )[2]
        ),

        "non_stress_f1": float(f1[0]),
        "stress_f1": float(f1[1]),

        "non_stress_support": int(support[0]),
        "stress_support": int(support[1]),
    }


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

def create_scheduler(optimizer, total_steps):

    warmup_steps = int(total_steps * WARMUP_RATIO)

    def lr_lambda(current_step):

        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)

        remaining_steps = total_steps - current_step

        decay_steps = max(1, total_steps - warmup_steps)

        return max(
            0.0,
            float(remaining_steps) / decay_steps,
        )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda,
    )

    return scheduler, warmup_steps


# ============================================================
# ATOMIC JSON SAVING
# ============================================================

def save_json_atomic(data, path):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )

    temporary_path.replace(path)


# ============================================================
# CHECKPOINT SAVING
# ============================================================

def save_checkpoint(model, optimizer, epoch, metrics):

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        CHECKPOINT_DIR / "best_checkpoint.pt"
    )

    temporary_path = (
        CHECKPOINT_DIR / "best_checkpoint.tmp"
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": metrics,
        "model_name": "distilbert-base-uncased",
        "random_seed": RANDOM_SEED,
    }

    torch.save(
        checkpoint,
        temporary_path,
    )

    temporary_path.replace(checkpoint_path)

    print(f"Best checkpoint saved: {checkpoint_path}")


# ============================================================
# TRAINING EPOCH
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    epoch,
    writer,
    global_step,
):

    model.train()

    total_loss = 0.0
    actual_labels = []
    predicted_labels = []

    optimizer.zero_grad(set_to_none=True)

    total_batches = len(train_loader)

    for batch_index, batch in enumerate(train_loader):

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
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=USE_FP16,
        ):

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            scaled_loss = (
                loss / GRADIENT_ACCUMULATION_STEPS
            )

        scaler.scale(scaled_loss).backward()

        should_update = (
            (batch_index + 1)
            % GRADIENT_ACCUMULATION_STEPS == 0
            or batch_index + 1 == total_batches
        )

        if should_update:

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

            scheduler.step()

            global_step += 1

            current_lr = scheduler.get_last_lr()[0]

            writer.add_scalar(
                "Training/Learning_Rate",
                current_lr,
                global_step,
            )

        total_loss += loss.item()

        predictions = torch.argmax(
            outputs.logits.detach(),
            dim=1,
        )

        actual_labels.extend(
            labels.detach().cpu().tolist()
        )

        predicted_labels.extend(
            predictions.cpu().tolist()
        )

        if (
            (batch_index + 1) % 25 == 0
            or batch_index + 1 == total_batches
        ):

            print(
                f"Epoch {epoch} | "
                f"Batch {batch_index + 1}/{total_batches} | "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / total_batches

    metrics = calculate_metrics(
        actual_labels,
        predicted_labels,
    )

    metrics["loss"] = float(average_loss)

    writer.add_scalar(
        "Training/Loss",
        average_loss,
        epoch,
    )

    writer.add_scalar(
        "Training/Macro_F1",
        metrics["macro_f1"],
        epoch,
    )

    writer.add_scalar(
        "Training/Accuracy",
        metrics["accuracy"],
        epoch,
    )

    return metrics, global_step


# ============================================================
# VALIDATION EPOCH
# ============================================================

def validate_one_epoch(
    model,
    validation_loader,
    epoch,
    writer,
):

    model.eval()

    total_loss = 0.0

    actual_labels = []
    predicted_labels = []

    with torch.no_grad():

        for batch in validation_loader:

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
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_FP16,
            ):

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            total_loss += outputs.loss.item()

            predictions = torch.argmax(
                outputs.logits,
                dim=1,
            )

            actual_labels.extend(
                labels.cpu().tolist()
            )

            predicted_labels.extend(
                predictions.cpu().tolist()
            )

    average_loss = (
        total_loss / len(validation_loader)
    )

    metrics = calculate_metrics(
        actual_labels,
        predicted_labels,
    )

    metrics["loss"] = float(average_loss)

    writer.add_scalar(
        "Validation/Loss",
        average_loss,
        epoch,
    )

    writer.add_scalar(
        "Validation/Macro_F1",
        metrics["macro_f1"],
        epoch,
    )

    writer.add_scalar(
        "Validation/Accuracy",
        metrics["accuracy"],
        epoch,
    )

    return metrics


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def train():

    print("=" * 70)
    print("DREADDIT DISTILBERT STRESS DETECTION TRAINING")
    print("=" * 70)

    set_seed(RANDOM_SEED)

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    (
        train_loader,
        validation_loader,
        _test_loader,
        tokenizer,
    ) = create_dataloaders()

    print("\nDataset information:")
    print(f"Training records: {len(train_loader.dataset):,}")
    print(f"Validation records: {len(validation_loader.dataset):,}")

    print("\nTest data is not used for training or model selection.")

    model = DreadditStressModel().to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    updates_per_epoch = math.ceil(
        len(train_loader) / GRADIENT_ACCUMULATION_STEPS
    )

    total_steps = updates_per_epoch * EPOCHS

    scheduler, warmup_steps = create_scheduler(
        optimizer,
        total_steps,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_FP16,
    )

    writer = SummaryWriter(
        log_dir=str(TENSORBOARD_DIR)
    )

    print("\nTraining configuration:")
    print(f"Epochs: {EPOCHS}")
    print(f"Optimizer updates per epoch: {updates_per_epoch}")
    print(f"Total optimizer updates: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"FP16: {USE_FP16}")
    print(f"Device: {DEVICE}")

    history = []

    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0

    start_time = time.time()

    try:

        for epoch in range(1, EPOCHS + 1):

            epoch_start = time.time()

            print("\n" + "=" * 70)
            print(f"STARTING EPOCH {epoch}/{EPOCHS}")
            print("=" * 70)

            train_metrics, global_step = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                writer=writer,
                global_step=global_step,
            )

            validation_metrics = validate_one_epoch(
                model=model,
                validation_loader=validation_loader,
                epoch=epoch,
                writer=writer,
            )

            epoch_duration = time.time() - epoch_start

            epoch_result = {
                "epoch": epoch,
                "duration_seconds": round(
                    epoch_duration,
                    2,
                ),
                "global_step": global_step,
                "train": train_metrics,
                "validation": validation_metrics,
            }

            history.append(epoch_result)

            print("\nTRAINING RESULTS")
            for key, value in train_metrics.items():
                print(f"{key}: {value:.6f}")

            print("\nVALIDATION RESULTS")
            for key, value in validation_metrics.items():
                print(f"{key}: {value:.6f}")

            print(
                f"\nEpoch duration: "
                f"{epoch_duration:.2f} seconds"
            )

            current_macro_f1 = (
                validation_metrics["macro_f1"]
            )

            if current_macro_f1 > best_macro_f1:

                best_macro_f1 = current_macro_f1
                best_epoch = epoch
                epochs_without_improvement = 0

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=validation_metrics,
                )

                # Save the tokenizer with the selected checkpoint.
                tokenizer.save_pretrained(
                    CHECKPOINT_DIR
                )

            else:

                epochs_without_improvement += 1

                print(
                    "\nNo validation Macro-F1 improvement. "
                    f"Early stopping counter: "
                    f"{epochs_without_improvement}/"
                    f"{EARLY_STOPPING_PATIENCE}"
                )

            save_json_atomic(
                history,
                HISTORY_FILE,
            )

            if (
                epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):

                print("\nEARLY STOPPING ACTIVATED")
                break

    finally:

        writer.close()

    total_duration = time.time() - start_time

    peak_gpu_memory_gb = None

    if DEVICE.type == "cuda":

        peak_gpu_memory_gb = round(
            torch.cuda.max_memory_allocated()
            / (1024 ** 3),
            3,
        )

    summary = {
        "model": "DistilBERT",
        "dataset": "Dreaddit",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "epochs_completed": len(history),
        "total_optimizer_steps": global_step,
        "total_training_duration_seconds": round(
            total_duration,
            2,
        ),
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
        "device": str(DEVICE),
        "fp16_enabled": USE_FP16,
        "checkpoint": str(
            CHECKPOINT_DIR / "best_checkpoint.pt"
        ),
    }

    save_json_atomic(
        summary,
        OUTPUT_DIR / "training_summary.json",
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation Macro-F1: {best_macro_f1:.6f}")
    print(f"Epochs completed: {len(history)}")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Peak GPU memory: {peak_gpu_memory_gb} GB")
    print(f"Checkpoint: {CHECKPOINT_DIR}")
    print(f"History: {HISTORY_FILE}")

    print("\nTRAINING PIPELINE PASSED")


if __name__ == "__main__":
    train()
