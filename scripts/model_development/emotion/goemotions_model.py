
import torch
import torch.nn as nn

from transformers import AutoModelForSequenceClassification

from config.goemotions_config import (
    MODEL_NAME,
    NUM_LABELS,
    LABEL_COLUMNS,
    DEVICE
)


# ============================================================
# 1. GOEMOTIONS DISTILBERT MODEL
# ============================================================

class GoEmotionsModel(nn.Module):

    def __init__(self):

        super().__init__()

        print("=" * 65)
        print("INITIALIZING GOEMOTIONS DISTILBERT MODEL")
        print("=" * 65)

        # Load pretrained DistilBERT with a 28-output classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification",
            id2label={
                index: label
                for index, label in enumerate(LABEL_COLUMNS)
            },
            label2id={
                label: index
                for index, label in enumerate(LABEL_COLUMNS)
            }
        )

        # Explicitly configure multi-label classification
        self.model.config.problem_type = "multi_label_classification"

        print("Pretrained encoder:", MODEL_NAME)
        print("Number of output neurons:", NUM_LABELS)
        print("Classification type: Multi-label")
        print("Loss: BCEWithLogitsLoss")

        print("\nModel initialized successfully.")

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None
    ):

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs


# ============================================================
# 2. MODEL SMOKE TEST
# ============================================================

def test_model():

    print("\n" + "=" * 65)
    print("GOEMOTIONS MODEL SMOKE TEST")
    print("=" * 65)

    model = GoEmotionsModel()
    model = model.to(DEVICE)

    model.eval()

    # Create a small illustrative input batch
    batch_size = 2
    sequence_length = 16

    input_ids = torch.randint(
        low=0,
        high=model.model.config.vocab_size,
        size=(batch_size, sequence_length),
        dtype=torch.long,
        device=DEVICE
    )

    attention_mask = torch.ones(
        (batch_size, sequence_length),
        dtype=torch.long,
        device=DEVICE
    )

    # Illustrative multi-label targets
    labels = torch.zeros(
        (batch_size, NUM_LABELS),
        dtype=torch.float32,
        device=DEVICE
    )

    labels[0, 0] = 1.0
    labels[0, 5] = 1.0
    labels[1, 2] = 1.0

    # Run a forward pass
    with torch.inference_mode():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    logits = outputs.logits
    loss = outputs.loss

    # Validate output dimensions
    expected_shape = (batch_size, NUM_LABELS)

    assert tuple(logits.shape) == expected_shape, (
        f"Expected {expected_shape}, got {tuple(logits.shape)}"
    )

    # Validate loss
    assert loss is not None
    assert torch.isfinite(loss).item(), "Non-finite loss detected."

    print("\nMODEL TEST RESULTS")

    print("Device:", DEVICE)
    print("Input shape:", tuple(input_ids.shape))
    print("Attention mask shape:", tuple(attention_mask.shape))
    print("Label shape:", tuple(labels.shape))
    print("Logits shape:", tuple(logits.shape))
    print("Loss:", round(loss.item(), 6))
    print("Loss function:", "BCEWithLogitsLoss")

    # Verify gradients can be calculated
    model.train()
    model.zero_grad(set_to_none=True)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    outputs.loss.backward()

    gradient_exists = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert gradient_exists, "Gradient calculation failed."

    print("Gradient calculation: SUCCESS")

    model.zero_grad(set_to_none=True)

    print("\nMODEL ARCHITECTURE VERIFIED")
    print("=" * 65)

    return model


# ============================================================
# 3. EXECUTION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    test_model()
