
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config.dreaddit_config import (
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRADIENT_ACCUMULATION_STEPS,
    MAX_GRAD_NORM,
    USE_FP16,
)

from scripts.model_development.stress.dreaddit_dataset import (
    create_dataloaders,
)

from scripts.model_development.stress.dreaddit_model import (
    DreadditStressModel,
)

from scripts.model_development.stress.dreaddit_trainer import (
    train_one_epoch,
    validate_one_epoch,
)


def main():

    print("=" * 65)
    print("DREADDIT TRAINING ENGINE SMOKE TEST")
    print("=" * 65)

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    (
        train_loader,
        validation_loader,
        _test_loader,
        _tokenizer,
    ) = create_dataloaders()

    model = DreadditStressModel().to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Use only two training batches for the smoke test.
    # This checks gradient accumulation with the configured value.
    optimizer.zero_grad(set_to_none=True)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_FP16,
    )

    model.train()

    initial_parameter = (
        model.model.classifier.weight.detach().clone()
    )

    losses = []

    for batch_index, batch in enumerate(train_loader):

        if batch_index >= 2:
            break

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

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss.")

        scaler.scale(scaled_loss).backward()

        losses.append(loss.item())

        print(
            f"Smoke batch {batch_index + 1}/2 "
            f"| Loss: {loss.item():.6f}"
        )

    if len(losses) != 2:
        raise RuntimeError("Smoke test did not process two batches.")

    scaler.unscale_(optimizer)

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        MAX_GRAD_NORM,
    )

    if not torch.isfinite(gradient_norm):
        raise RuntimeError("Non-finite gradient norm.")

    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad(set_to_none=True)

    updated_parameter = model.model.classifier.weight.detach()

    if torch.equal(initial_parameter, updated_parameter):
        raise RuntimeError("Optimizer did not update model parameters.")

    print("\nOptimizer update: PASSED")
    print(f"Gradient norm: {gradient_norm.item():.6f}")
    print(f"Mixed precision enabled: {USE_FP16}")

    # Verify the validation loop independently.
    optimizer.zero_grad(set_to_none=True)

    class NullWriter:
        def add_scalar(self, *args, **kwargs):
            pass

    validation_metrics = validate_one_epoch(
        model=model,
        validation_loader=validation_loader,
        epoch=1,
        writer=NullWriter(),
    )

    print("\nValidation smoke test:")
    for key, value in validation_metrics.items():
        print(f"{key}: {value:.6f}")

    if not torch.isfinite(
        torch.tensor(validation_metrics["loss"])
    ):
        raise RuntimeError("Invalid validation loss.")

    if DEVICE.type == "cuda":
        peak_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        print(f"\nPeak GPU memory: {peak_memory:.3f} GB")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
