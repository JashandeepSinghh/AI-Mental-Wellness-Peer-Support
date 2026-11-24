
"""
FIG-Loneliness DistilBERT Training Engine
=========================================

Responsibilities:
    - AdamW optimizer
    - Linear learning-rate scheduler
    - FP16 mixed precision
    - Gradient accumulation
    - Gradient clipping
    - Training and validation metrics
    - Best checkpoint management
    - Early stopping
    - TensorBoard logging
    - JSON training history

The official test split is never evaluated during training.
"""

import argparse
import json
import logging
import math
import time

from pathlib import Path

import numpy as np
import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from transformers import AutoTokenizer

from scripts.model_development.loneliness.config import (
    fig_loneliness_config as config
)

from scripts.model_development.loneliness.fig_loneliness_model import (
    initialize_model
)

from scripts.model_development.loneliness.fig_loneliness_dataset import (
    create_data_pipeline
)


# ============================================================
# 1. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# 2. TRAINING ENGINE
# ============================================================

class FigLonelinessTrainer:

    def __init__(
        self,
        model,
        train_loader,
        validation_loader,
        tokenizer
    ):

        self.model = model

        self.train_loader = train_loader

        self.validation_loader = validation_loader

        self.tokenizer = tokenizer

        self.device = config.DEVICE

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # ----------------------------------------------------
        # Calculate optimizer update steps
        # ----------------------------------------------------

        self.steps_per_epoch = math.ceil(
            len(self.train_loader)
            / config.GRADIENT_ACCUMULATION_STEPS
        )

        self.total_training_steps = (
            self.steps_per_epoch * config.NUM_EPOCHS
        )

        self.warmup_steps = int(
            self.total_training_steps * config.WARMUP_RATIO
        )

        # ----------------------------------------------------
        # Learning-rate scheduler
        # ----------------------------------------------------

        def lr_lambda(current_step):

            if current_step < self.warmup_steps:

                return float(current_step) / max(
                    1,
                    self.warmup_steps
                )

            remaining_steps = (
                self.total_training_steps - current_step
            )

            decay_steps = max(
                1,
                self.total_training_steps - self.warmup_steps
            )

            return max(
                0.0,
                float(remaining_steps) / decay_steps
            )

        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda
        )

        # ----------------------------------------------------
        # FP16 gradient scaler
        # ----------------------------------------------------

        self.use_fp16 = config.USE_MIXED_PRECISION

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_fp16
        )

        # ----------------------------------------------------
        # Training state
        # ----------------------------------------------------

        self.best_validation_macro_f1 = -1.0

        self.best_epoch = 0

        self.epochs_without_improvement = 0

        self.training_history = []

        # ----------------------------------------------------
        # Output directories
        # ----------------------------------------------------

        config.MODEL_OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        config.TRAINING_OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        config.CHECKPOINT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # TensorBoard
        # ----------------------------------------------------

        self.writer = SummaryWriter(
            log_dir=str(config.TENSORBOARD_DIR)
        )

        logger.info("Training engine initialized.")

        logger.info(
            "Optimizer update steps per epoch: %d",
            self.steps_per_epoch
        )

        logger.info(
            "Total optimizer update steps: %d",
            self.total_training_steps
        )

        logger.info(
            "Warmup steps: %d",
            self.warmup_steps
        )

        logger.info(
            "Mixed precision enabled: %s",
            self.use_fp16
        )

    # ========================================================
    # 3. BATCH TRANSFER
    # ========================================================

    def move_batch_to_device(self, batch):

        return {
            key: value.to(
                self.device,
                non_blocking=True
            )
            for key, value in batch.items()
        }

    # ========================================================
    # 4. METRIC CALCULATION
    # ========================================================

    @staticmethod
    def calculate_metrics(
        actual_labels,
        predicted_labels
    ):

        accuracy = accuracy_score(
            actual_labels,
            predicted_labels
        )

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                actual_labels,
                predicted_labels,
                labels=[0, 1],
                average=None,
                zero_division=0
            )
        )

        macro_precision = float(np.mean(precision))

        macro_recall = float(np.mean(recall))

        macro_f1 = float(np.mean(f1))

        weighted_f1 = float(
            precision_recall_fscore_support(
                actual_labels,
                predicted_labels,
                labels=[0, 1],
                average="weighted",
                zero_division=0
            )[2]
        )

        matrix = confusion_matrix(
            actual_labels,
            predicted_labels,
            labels=[0, 1]
        )

        return {
            "accuracy": float(accuracy),
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "non_lonely_precision": float(precision[0]),
            "non_lonely_recall": float(recall[0]),
            "non_lonely_f1": float(f1[0]),
            "lonely_precision": float(precision[1]),
            "lonely_recall": float(recall[1]),
            "lonely_f1": float(f1[1]),
            "non_lonely_support": int(support[0]),
            "lonely_support": int(support[1]),
            "confusion_matrix": matrix.tolist()
        }

    # ========================================================
    # 5. TRAINING EPOCH
    # ========================================================

    def train_one_epoch(self, epoch):

        self.model.train()

        total_loss = 0.0

        all_predictions = []

        all_labels = []

        self.optimizer.zero_grad(set_to_none=True)

        epoch_start = time.time()

        total_batches = len(self.train_loader)

        for batch_index, batch in enumerate(
            self.train_loader,
            start=1
        ):

            batch = self.move_batch_to_device(batch)

            labels = batch["labels"]

            # Determine the actual accumulation group size.
            # This handles the final incomplete group correctly.
            group_start = (
                (batch_index - 1)
                // config.GRADIENT_ACCUMULATION_STEPS
            ) * config.GRADIENT_ACCUMULATION_STEPS

            current_group_size = min(
                config.GRADIENT_ACCUMULATION_STEPS,
                total_batches - group_start
            )

            # ------------------------------------------------
            # Forward pass with mixed precision
            # ------------------------------------------------

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.use_fp16
            ):

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=labels
                )

                loss = outputs.loss

            # ------------------------------------------------
            # Gradient accumulation
            # ------------------------------------------------

            scaled_loss = loss / current_group_size

            self.scaler.scale(
                scaled_loss
            ).backward()

            # ------------------------------------------------
            # Store training metrics
            # ------------------------------------------------

            total_loss += loss.detach().item()

            predictions = torch.argmax(
                outputs.logits.detach(),
                dim=-1
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

            # ------------------------------------------------
            # Optimizer update
            # ------------------------------------------------

            should_update = (
                batch_index % config.GRADIENT_ACCUMULATION_STEPS == 0
                or batch_index == total_batches
            )

            if should_update:

                self.scaler.unscale_(
                    self.optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    config.MAX_GRAD_NORM
                )

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

                self.scheduler.step()

                self.optimizer.zero_grad(
                    set_to_none=True
                )

            if (
                batch_index == 1
                or batch_index % 50 == 0
                or batch_index == total_batches
            ):

                logger.info(
                    "Epoch %d | Batch %d/%d | Loss %.5f",
                    epoch,
                    batch_index,
                    total_batches,
                    loss.detach().item()
                )

        # ----------------------------------------------------
        # Epoch-level metrics
        # ----------------------------------------------------

        metrics = self.calculate_metrics(
            all_labels,
            all_predictions
        )

        metrics["loss"] = (
            total_loss / total_batches
        )

        metrics["duration_seconds"] = (
            time.time() - epoch_start
        )

        return metrics

    # ========================================================
    # 6. VALIDATION EPOCH
    # ========================================================

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        total_loss = 0.0

        all_predictions = []

        all_labels = []

        for batch in self.validation_loader:

            batch = self.move_batch_to_device(batch)

            labels = batch["labels"]

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.use_fp16
            ):

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=labels
                )

            total_loss += outputs.loss.item()

            predictions = torch.argmax(
                outputs.logits,
                dim=-1
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

        metrics = self.calculate_metrics(
            all_labels,
            all_predictions
        )

        metrics["loss"] = (
            total_loss / len(self.validation_loader)
        )

        return metrics

    # ========================================================
    # 7. SAVE BEST CHECKPOINT
    # ========================================================

    def save_best_checkpoint(
        self,
        epoch,
        validation_metrics
    ):

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_validation_macro_f1": (
                validation_metrics["macro_f1"]
            ),
            "validation_metrics": validation_metrics,
            "model_name": config.MODEL_NAME,
            "max_length": config.MAX_LENGTH,
            "label2id": config.LABEL2ID,
            "id2label": config.ID2LABEL,
            "random_seed": config.RANDOM_SEED
        }

        checkpoint_path = (
            config.CHECKPOINT_DIR
            / "best_checkpoint.pt"
        )

        torch.save(
            checkpoint,
            checkpoint_path
        )

        self.tokenizer.save_pretrained(
            config.CHECKPOINT_DIR
        )

        logger.info(
            "Best checkpoint saved: %s",
            checkpoint_path
        )

    # ========================================================
    # 8. TENSORBOARD LOGGING
    # ========================================================

    def log_tensorboard(
        self,
        epoch,
        training_metrics,
        validation_metrics
    ):

        metric_names = [
            "loss",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1"
        ]

        for metric in metric_names:

            self.writer.add_scalars(
                metric,
                {
                    "training": training_metrics[metric],
                    "validation": validation_metrics[metric]
                },
                epoch
            )

        self.writer.add_scalar(
            "learning_rate",
            self.optimizer.param_groups[0]["lr"],
            epoch
        )

        self.writer.flush()

    # ========================================================
    # 9. SAVE TRAINING HISTORY
    # ========================================================

    def save_training_history(self):

        history_path = (
            config.TRAINING_OUTPUT_DIR
            / "training_history.json"
        )

        with open(
            history_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.training_history,
                file,
                indent=4
            )

        logger.info(
            "Training history saved: %s",
            history_path
        )

    # ========================================================
    # 10. TRAINING SUMMARY
    # ========================================================

    def save_training_summary(
        self,
        status,
        total_duration
    ):

        summary = {
            "status": status,
            "model": config.MODEL_NAME,
            "best_epoch": self.best_epoch,
            "best_validation_macro_f1": (
                self.best_validation_macro_f1
            ),
            "epochs_completed": len(self.training_history),
            "total_duration_seconds": total_duration,
            "max_sequence_length": config.MAX_LENGTH,
            "training_batch_size": config.TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": (
                config.GRADIENT_ACCUMULATION_STEPS
            ),
            "effective_batch_size": (
                config.TRAIN_BATCH_SIZE
                * config.GRADIENT_ACCUMULATION_STEPS
            ),
            "learning_rate": config.LEARNING_RATE,
            "random_seed": config.RANDOM_SEED
        }

        summary_path = (
            config.TRAINING_OUTPUT_DIR
            / "training_summary.json"
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                summary,
                file,
                indent=4
            )

        logger.info(
            "Training summary saved: %s",
            summary_path
        )

    # ========================================================
    # 11. COMPLETE TRAINING LOOP
    # ========================================================

    def train(self):

        logger.info("=" * 70)
        logger.info("FIG-LONELINESS DISTILBERT TRAINING")
        logger.info("=" * 70)

        start_time = time.time()

        status = "completed"

        try:

            for epoch in range(
                1,
                config.NUM_EPOCHS + 1
            ):

                logger.info("")
                logger.info("=" * 70)
                logger.info(
                    "STARTING EPOCH %d/%d",
                    epoch,
                    config.NUM_EPOCHS
                )
                logger.info("=" * 70)

                # --------------------------------------------
                # Training
                # --------------------------------------------

                training_metrics = self.train_one_epoch(
                    epoch
                )

                # --------------------------------------------
                # Validation
                # --------------------------------------------

                validation_metrics = self.validate()

                # --------------------------------------------
                # Print epoch results
                # --------------------------------------------

                logger.info(
                    "Epoch %d Training | Loss %.5f | "
                    "Accuracy %.5f | Macro-F1 %.5f",
                    epoch,
                    training_metrics["loss"],
                    training_metrics["accuracy"],
                    training_metrics["macro_f1"]
                )

                logger.info(
                    "Epoch %d Validation | Loss %.5f | "
                    "Accuracy %.5f | Macro-F1 %.5f",
                    epoch,
                    validation_metrics["loss"],
                    validation_metrics["accuracy"],
                    validation_metrics["macro_f1"]
                )

                logger.info(
                    "Validation classes | "
                    "Non-lonely F1 %.5f | Lonely F1 %.5f",
                    validation_metrics["non_lonely_f1"],
                    validation_metrics["lonely_f1"]
                )

                # --------------------------------------------
                # Save epoch history
                # --------------------------------------------

                epoch_record = {
                    "epoch": epoch,
                    "training": training_metrics,
                    "validation": validation_metrics
                }

                self.training_history.append(
                    epoch_record
                )

                self.log_tensorboard(
                    epoch,
                    training_metrics,
                    validation_metrics
                )

                self.save_training_history()

                # --------------------------------------------
                # Best checkpoint and early stopping
                # --------------------------------------------

                current_macro_f1 = (
                    validation_metrics["macro_f1"]
                )

                if (
                    current_macro_f1
                    > self.best_validation_macro_f1
                    + config.MINIMUM_IMPROVEMENT
                ):

                    self.best_validation_macro_f1 = (
                        current_macro_f1
                    )

                    self.best_epoch = epoch

                    self.epochs_without_improvement = 0

                    self.save_best_checkpoint(
                        epoch,
                        validation_metrics
                    )

                    logger.info(
                        "New best validation Macro-F1: %.5f",
                        current_macro_f1
                    )

                else:

                    self.epochs_without_improvement += 1

                    logger.info(
                        "No validation improvement. "
                        "Early stopping counter: %d/%d",
                        self.epochs_without_improvement,
                        config.EARLY_STOPPING_PATIENCE
                    )

                if (
                    self.epochs_without_improvement
                    >= config.EARLY_STOPPING_PATIENCE
                ):

                    logger.info(
                        "Early stopping triggered at epoch %d.",
                        epoch
                    )

                    status = "early_stopped"

                    break

            total_duration = time.time() - start_time

            self.save_training_summary(
                status=status,
                total_duration=total_duration
            )

            logger.info("=" * 70)
            logger.info("TRAINING FINISHED")
            logger.info(
                "Best epoch: %d",
                self.best_epoch
            )

            logger.info(
                "Best validation Macro-F1: %.5f",
                self.best_validation_macro_f1
            )

            logger.info(
                "Total duration: %.2f seconds",
                total_duration
            )

            logger.info("=" * 70)

        except Exception:

            logger.exception(
                "Training failed. Saving available history."
            )

            status = "failed"

            raise

        finally:

            self.writer.close()

            if status == "failed":

                self.save_training_history()


# ============================================================
# 12. TRAINING SMOKE TEST
# ============================================================

def run_training_smoke_test(
    model,
    train_loader,
    validation_loader,
    tokenizer
):

    """
    Run a short training-engine test.

    This is not full training.
    It verifies optimizer updates, FP16, scheduler,
    gradients and validation execution.
    """

    logger.info("=" * 70)
    logger.info("TRAINING ENGINE SMOKE TEST")
    logger.info("=" * 70)

    device = config.DEVICE

    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.USE_MIXED_PRECISION
    )

    model.train()

    optimizer.zero_grad(set_to_none=True)

    batches_to_test = min(
        2,
        len(train_loader)
    )

    initial_parameter = (
        next(model.parameters())
        .detach()
        .clone()
    )

    for batch_index, batch in enumerate(
        train_loader,
        start=1
    ):

        if batch_index > batches_to_test:
            break

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=config.USE_MIXED_PRECISION
        ):

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )

            loss = outputs.loss

        assert torch.isfinite(loss).item()

        scaler.scale(
            loss / batches_to_test
        ).backward()

        logger.info(
            "Smoke batch %d | Loss %.6f",
            batch_index,
            loss.item()
        )

    scaler.unscale_(optimizer)

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        config.MAX_GRAD_NORM
    )

    assert torch.isfinite(gradient_norm).item()

    scaler.step(optimizer)

    scaler.update()

    optimizer.zero_grad(set_to_none=True)

    updated_parameter = next(
        model.parameters()
    ).detach()

    parameter_changed = not torch.equal(
        initial_parameter,
        updated_parameter
    )

    assert parameter_changed, (
        "Optimizer did not update model parameters."
    )

    logger.info(
        "Optimizer update successful. Gradient norm: %.6f",
        gradient_norm.item()
    )

    # --------------------------------------------------------
    # Validation smoke test
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        validation_batch = next(
            iter(validation_loader)
        )

        validation_batch = {
            key: value.to(device)
            for key, value in validation_batch.items()
        }

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=config.USE_MIXED_PRECISION
        ):

            validation_output = model(
                input_ids=validation_batch["input_ids"],
                attention_mask=validation_batch["attention_mask"],
                labels=validation_batch["labels"]
            )

    assert torch.isfinite(
        validation_output.loss
    ).item()

    logger.info(
        "Validation smoke test passed. Loss: %.6f",
        validation_output.loss.item()
    )

    if device.type == "cuda":

        torch.cuda.synchronize()

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        logger.info(
            "Peak GPU memory allocated: %.3f GB",
            peak_memory
        )

    print("\nTRAINING ENGINE SMOKE TEST")
    print("=" * 55)
    print("FP16 forward pass       : PASSED")
    print("Gradient computation    : PASSED")
    print("Gradient clipping       : PASSED")
    print("Optimizer update        : PASSED")
    print("Validation execution    : PASSED")
    print(f"Gradient norm           : {gradient_norm.item():.6f}")
    print(f"Parameter updated       : {parameter_changed}")
    print("=" * 55)


# ============================================================
# 13. MAIN EXECUTION
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a short training-engine test instead of full training."
    )

    args = parser.parse_args()

    config.validate_config()

    # Load data pipeline
    tokenizer, datasets, dataloaders = create_data_pipeline()

    # Initialize model
    model = initialize_model()

    if args.smoke_test:

        run_training_smoke_test(
            model=model,
            train_loader=dataloaders["train"],
            validation_loader=dataloaders["validation"],
            tokenizer=tokenizer
        )

        return

    # Initialize complete training engine
    trainer = FigLonelinessTrainer(
        model=model,
        train_loader=dataloaders["train"],
        validation_loader=dataloaders["validation"],
        tokenizer=tokenizer
    )

    trainer.train()


if __name__ == "__main__":

    main()

