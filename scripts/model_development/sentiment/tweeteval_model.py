
import torch
import torch.nn as nn

from transformers import AutoModelForSequenceClassification

from config.tweeteval_config import (
    MODEL_NAME,
    NUM_LABELS,
    ID2LABEL,
    LABEL2ID,
    DEVICE,
)


# ============================================================
# 1. TWEETEVAL DISTILBERT MODEL
# ============================================================

class TweetEvalModel(nn.Module):

    def __init__(self):

        super().__init__()

        print("\nLoading pretrained DistilBERT model...")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            problem_type="single_label_classification",
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        # Explicitly configure single-label classification.
        self.model.config.problem_type = (
            "single_label_classification"
        )

        print("DistilBERT model loaded successfully.")

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


# ============================================================
# 2. MODEL SMOKE TEST
# ============================================================

def test_model():

    print("=" * 65)
    print("TWEETEVAL MODEL VERIFICATION")
    print("=" * 65)

    model = TweetEvalModel()

    model = model.to(DEVICE)

    model.train()

    # --------------------------------------------------------
    # Create a synthetic input batch.
    # --------------------------------------------------------

    batch_size = 4
    sequence_length = 32

    input_ids = torch.randint(
        low=0,
        high=model.model.config.vocab_size,
        size=(batch_size, sequence_length),
        dtype=torch.long,
        device=DEVICE,
    )

    attention_mask = torch.ones(
        (batch_size, sequence_length),
        dtype=torch.long,
        device=DEVICE,
    )

    # One integer class label per example.
    labels = torch.tensor(
        [0, 1, 2, 1],
        dtype=torch.long,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Forward pass.
    # --------------------------------------------------------

    print("\nRunning forward pass...")

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    logits = outputs.logits
    loss = outputs.loss

    print("Logits shape:", tuple(logits.shape))
    print("Loss:", loss.item())

    # --------------------------------------------------------
    # Validate output.
    # --------------------------------------------------------

    assert logits.shape == (
        batch_size,
        NUM_LABELS,
    )

    assert torch.isfinite(loss).item()

    assert model.model.config.problem_type == (
        "single_label_classification"
    )

    # --------------------------------------------------------
    # Verify probability conversion.
    # --------------------------------------------------------

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    row_sums = probabilities.sum(dim=1)

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-5,
    )

    print(
        "Probability shape:",
        tuple(probabilities.shape),
    )

    print(
        "Probability row sums:",
        row_sums.detach().cpu().tolist(),
    )

    # --------------------------------------------------------
    # Backpropagation test.
    # --------------------------------------------------------

    print("\nRunning backward pass...")

    loss.backward()

    classifier_gradient = (
        model.model.classifier.weight.grad
    )

    assert classifier_gradient is not None

    assert torch.isfinite(
        classifier_gradient
    ).all().item()

    print("Gradient verification: PASSED")

    print("\nDevice:", DEVICE)
    print("Classification type: Single-label")
    print("Loss function: CrossEntropyLoss")

    print("\nMODEL VERIFICATION PASSED")
    print("=" * 65)

    del model
    del outputs
    del logits
    del probabilities

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 3. EXECUTION
# ============================================================

if __name__ == "__main__":

    test_model()
