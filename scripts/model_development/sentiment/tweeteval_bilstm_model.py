
import torch
import torch.nn as nn

from torch.nn.utils.rnn import (
    pack_padded_sequence,
)

from config.tweeteval_bilstm_config import (
    EMBEDDING_DIM,
    HIDDEN_DIM,
    NUM_LAYERS,
    NUM_CLASSES,
    DROPOUT,
    BIDIRECTIONAL,
    PAD_INDEX,
    DEVICE,
)

from scripts.model_development.sentiment.tweeteval_bilstm_dataset import (
    create_bilstm_dataloaders,
)


# ============================================================
# 1. BILSTM CLASSIFICATION MODEL
# ============================================================

class TweetEvalBiLSTM(nn.Module):

    def __init__(self, vocabulary_size):

        super().__init__()

        self.vocabulary_size = vocabulary_size

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=EMBEDDING_DIM,
            padding_idx=PAD_INDEX,
        )

        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=BIDIRECTIONAL,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )

        directions = 2 if BIDIRECTIONAL else 1

        representation_size = (
            HIDDEN_DIM * directions
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )

        self.classifier = nn.Linear(
            representation_size,
            NUM_CLASSES,
        )

        self._initialize_weights()

    # --------------------------------------------------------
    # Initialize model weights
    # --------------------------------------------------------

    def _initialize_weights(self):

        nn.init.uniform_(
            self.embedding.weight,
            -0.1,
            0.1,
        )

        # Keep padding embedding at zero.
        with torch.no_grad():

            self.embedding.weight[
                PAD_INDEX
            ].zero_()

        nn.init.xavier_uniform_(
            self.classifier.weight
        )

        nn.init.zeros_(
            self.classifier.bias
        )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    def forward(
        self,
        input_ids,
        lengths,
    ):

        if input_ids.ndim != 2:
            raise ValueError(
                "Input IDs must have shape [batch, sequence]."
            )

        if lengths.ndim != 1:
            raise ValueError(
                "Lengths must have shape [batch]."
            )

        if input_ids.size(0) != lengths.size(0):
            raise ValueError(
                "Batch sizes of input IDs and lengths differ."
            )

        if torch.any(lengths <= 0):
            raise ValueError(
                "Sequence lengths must be positive."
            )

        if torch.any(lengths > input_ids.size(1)):
            raise ValueError(
                "A sequence length exceeds padded input width."
            )

        # ----------------------------------------------------
        # Word embedding representation.
        # ----------------------------------------------------

        embedded = self.embedding(
            input_ids
        )

        # ----------------------------------------------------
        # Pack variable-length sequences.
        # Padding positions are excluded from LSTM processing.
        # ----------------------------------------------------

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # ----------------------------------------------------
        # Bidirectional LSTM.
        # ----------------------------------------------------

        _, (hidden, _) = self.lstm(
            packed
        )

        # ----------------------------------------------------
        # Extract final forward and backward hidden states.
        # ----------------------------------------------------

        if BIDIRECTIONAL:

            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]

            representation = torch.cat(
                [
                    forward_hidden,
                    backward_hidden,
                ],
                dim=1,
            )

        else:

            representation = hidden[-1]

        # ----------------------------------------------------
        # Dropout regularization.
        # ----------------------------------------------------

        representation = self.dropout(
            representation
        )

        # ----------------------------------------------------
        # Three-class sentiment logits.
        # ----------------------------------------------------

        logits = self.classifier(
            representation
        )

        return logits


# ============================================================
# 2. MODEL SMOKE TEST
# ============================================================

def test_model():

    print("=" * 65)
    print("TWEETEVAL BILSTM MODEL VERIFICATION")
    print("=" * 65)

    # --------------------------------------------------------
    # Load vocabulary and one real batch.
    # --------------------------------------------------------

    (
        vocabulary,
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader,
    ) = create_bilstm_dataloaders()

    vocabulary_size = len(
        vocabulary
    )

    model = TweetEvalBiLSTM(
        vocabulary_size
    ).to(DEVICE)

    model.train()

    batch = next(
        iter(train_loader)
    )

    input_ids = batch["input_ids"].to(
        DEVICE
    )

    lengths = batch["lengths"]

    labels = batch["labels"].to(
        DEVICE
    )

    # --------------------------------------------------------
    # Forward pass.
    # --------------------------------------------------------

    print("\nRunning forward pass...")

    logits = model(
        input_ids,
        lengths,
    )

    print(
        "Logits shape:",
        tuple(logits.shape),
    )

    assert logits.shape == (
        input_ids.size(0),
        NUM_CLASSES,
    )

    # --------------------------------------------------------
    # Loss calculation.
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    loss = criterion(
        logits,
        labels,
    )

    print(
        "CrossEntropyLoss:",
        loss.item(),
    )

    assert torch.isfinite(loss).item()

    # --------------------------------------------------------
    # Probability verification.
    # --------------------------------------------------------

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    probability_sums = probabilities.sum(
        dim=1
    )

    assert torch.allclose(
        probability_sums,
        torch.ones_like(probability_sums),
        atol=1e-5,
    )

    print(
        "Probability shape:",
        tuple(probabilities.shape),
    )

    print(
        "Probability row sums:",
        probability_sums.detach().cpu().tolist(),
    )

    # --------------------------------------------------------
    # Backpropagation.
    # --------------------------------------------------------

    print("\nRunning backward pass...")

    loss.backward()

    assert model.embedding.weight.grad is not None

    assert model.classifier.weight.grad is not None

    assert model.lstm.weight_ih_l0.grad is not None

    assert torch.isfinite(
        model.embedding.weight.grad
    ).all().item()

    assert torch.isfinite(
        model.lstm.weight_ih_l0.grad
    ).all().item()

    print("Embedding gradients: PASSED")
    print("LSTM gradients: PASSED")
    print("Classifier gradients: PASSED")

    # --------------------------------------------------------
    # Verify padding embedding.
    # --------------------------------------------------------

    assert torch.all(
        model.embedding.weight[PAD_INDEX] == 0
    )

    print("Padding embedding verification: PASSED")

    # --------------------------------------------------------
    # Model information.
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nMODEL INFORMATION")

    print("Vocabulary size:", vocabulary_size)
    print("Embedding dimension:", EMBEDDING_DIM)
    print("Hidden dimension:", HIDDEN_DIM)
    print("LSTM layers:", NUM_LAYERS)
    print("Bidirectional:", BIDIRECTIONAL)

    print(
        "Total parameters:",
        f"{total_parameters:,}",
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )

    print("Device:", DEVICE)

    print("\nBILSTM MODEL VERIFICATION PASSED")
    print("=" * 65)


# ============================================================
# 3. EXECUTION
# ============================================================

if __name__ == "__main__":

    test_model()
