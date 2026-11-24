
import torch
import torch.nn as nn

from transformers import AutoModelForSequenceClassification

from config.dreaddit_config import (
    MODEL_NAME,
    NUM_LABELS,
    ID2LABEL,
    LABEL2ID,
    DEVICE,
)

from scripts.model_development.stress.dreaddit_dataset import (
    create_dataloaders,
)


class DreadditStressModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            problem_type="single_label_classification",
        )

        self.model.config.problem_type = (
            "single_label_classification"
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
    ):

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs


def test_model():

    print("=" * 65)
    print("DREADDIT DISTILBERT MODEL VERIFICATION")
    print("=" * 65)

    model = DreadditStressModel().to(DEVICE)

    model.train()

    train_loader, _, _, _ = create_dataloaders()

    batch = next(iter(train_loader))

    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    logits = outputs.logits
    loss = outputs.loss

    print(f"\nInput shape: {tuple(input_ids.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Loss: {loss.item():.6f}")
    print(f"Device: {DEVICE}")

    if logits.shape != (input_ids.shape[0], NUM_LABELS):
        raise RuntimeError("Unexpected model output shape.")

    if not torch.isfinite(loss):
        raise RuntimeError("Model produced non-finite loss.")

    if not torch.isfinite(logits).all():
        raise RuntimeError("Model produced non-finite logits.")

    loss.backward()

    gradient_found = any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum().item() > 0
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if not gradient_found:
        raise RuntimeError("Model gradients were not calculated.")

    print("\nBackward propagation: PASSED")
    print("Gradient verification: PASSED")

    model.eval()

    with torch.no_grad():

        inference_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        probabilities = torch.softmax(
            inference_outputs.logits,
            dim=1,
        )

        probability_sums = probabilities.sum(dim=1)

    if not torch.allclose(
        probability_sums,
        torch.ones_like(probability_sums),
        atol=1e-5,
    ):
        raise RuntimeError("Softmax probabilities do not sum to 1.")

    print("Inference verification: PASSED")
    print("Probability normalization: PASSED")

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"\nTotal parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    print("\nMODEL VERIFICATION PASSED")


if __name__ == "__main__":
    test_model()
