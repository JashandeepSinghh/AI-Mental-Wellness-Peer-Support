
"""
FIG-Loneliness DistilBERT Model
================================

Purpose:
    Binary sequence classification for expressed loneliness.

Architecture:
    DistilBERT pretrained encoder
        -> Dropout
        -> Linear classification head
        -> 2 output logits

Labels:
    0 = Non-lonely
    1 = Lonely

Important:
    The model identifies linguistic signals of expressed
    loneliness. It does not diagnose a person or establish
    their actual social circumstances.
"""

import logging

import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification
)

from scripts.model_development.loneliness.config import (
    fig_loneliness_config as config
)


# ============================================================
# 1. LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# 2. MODEL CLASS
# ============================================================

class FigLonelinessModel(nn.Module):
    """
    DistilBERT-based binary sequence classification model.
    """

    def __init__(self, pretrained_model_name=None):

        super().__init__()

        if pretrained_model_name is None:
            pretrained_model_name = config.MODEL_NAME

        logger.info(
            "Initializing FIG-Loneliness model: %s",
            pretrained_model_name
        )

        # ----------------------------------------------------
        # Load pretrained model configuration
        # ----------------------------------------------------

        self.model_config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=config.NUM_LABELS,
            id2label=config.ID2LABEL,
            label2id=config.LABEL2ID,
            problem_type="single_label_classification"
        )

        # ----------------------------------------------------
        # Load pretrained DistilBERT classification model
        # ----------------------------------------------------

        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=self.model_config
        )

        logger.info(
            "Pretrained encoder and classification head initialized."
        )

    # ========================================================
    # 3. FORWARD PASS
    # ========================================================

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None
    ):
        """
        Forward propagation.

        Parameters:
            input_ids:
                Token IDs with shape (batch_size, sequence_length).

            attention_mask:
                Attention mask with shape (batch_size, sequence_length).

            labels:
                Optional class labels with shape (batch_size,).

        Returns:
            Hugging Face SequenceClassifierOutput containing:
                loss (when labels supplied)
                logits
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs

    # ========================================================
    # 4. PREDICTION
    # ========================================================

    @torch.no_grad()
    def predict(self, input_ids, attention_mask):
        """
        Generate class predictions and softmax scores.

        Note:
            Softmax scores are not automatically calibrated
            probabilities of actual psychological loneliness.
        """

        self.eval()

        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        predicted_labels = torch.argmax(
            probabilities,
            dim=-1
        )

        return {
            "logits": outputs.logits,
            "probabilities": probabilities,
            "predictions": predicted_labels
        }


# ============================================================
# 5. PARAMETER COUNT
# ============================================================

def count_parameters(model):

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters
    }


# ============================================================
# 6. MODEL INITIALIZATION
# ============================================================

def initialize_model():

    model = FigLonelinessModel()

    model = model.to(config.DEVICE)

    parameter_stats = count_parameters(model)

    logger.info(
        "Total parameters: %s",
        f"{parameter_stats['total_parameters']:,}"
    )

    logger.info(
        "Trainable parameters: %s",
        f"{parameter_stats['trainable_parameters']:,}"
    )

    logger.info(
        "Model moved to device: %s",
        config.DEVICE
    )

    return model


# ============================================================
# 7. MODEL SMOKE TEST
# ============================================================

def run_model_smoke_test(model, dataloader):

    logger.info("=" * 65)
    logger.info("FIG-LONELINESS MODEL SMOKE TEST")
    logger.info("=" * 65)

    model.train()

    # --------------------------------------------------------
    # Load one real training batch
    # --------------------------------------------------------

    batch = next(iter(dataloader))

    input_ids = batch["input_ids"].to(config.DEVICE)

    attention_mask = batch["attention_mask"].to(config.DEVICE)

    labels = batch["labels"].to(config.DEVICE)

    logger.info(
        "Input batch shape: %s",
        tuple(input_ids.shape)
    )

    # --------------------------------------------------------
    # Clear old gradients
    # --------------------------------------------------------

    model.zero_grad(set_to_none=True)

    # --------------------------------------------------------
    # Forward propagation
    # --------------------------------------------------------

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    loss = outputs.loss
    logits = outputs.logits

    assert loss is not None
    assert torch.isfinite(loss).item()

    assert logits.shape == (
        input_ids.shape[0],
        config.NUM_LABELS
    )

    logger.info(
        "Forward pass successful. Loss: %.6f",
        loss.item()
    )

    logger.info(
        "Logits shape: %s",
        tuple(logits.shape)
    )

    # --------------------------------------------------------
    # Backward propagation
    # --------------------------------------------------------

    loss.backward()

    gradients_found = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            assert torch.isfinite(
                parameter.grad
            ).all().item()

            gradients_found += 1

    assert gradients_found > 0

    logger.info(
        "Backward pass successful. Parameters with gradients: %d",
        gradients_found
    )

    # --------------------------------------------------------
    # Inference test
    # --------------------------------------------------------

    model.eval()

    predictions = model.predict(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    predicted_labels = predictions["predictions"]

    probabilities = predictions["probabilities"]

    assert probabilities.shape == (
        input_ids.shape[0],
        config.NUM_LABELS
    )

    assert predicted_labels.shape == labels.shape

    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(
            input_ids.shape[0],
            device=config.DEVICE
        ),
        atol=1e-5
    )

    logger.info("Inference test successful.")

    # --------------------------------------------------------
    # GPU memory inspection
    # --------------------------------------------------------

    if config.DEVICE.type == "cuda":

        torch.cuda.synchronize()

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        logger.info(
            "Peak GPU memory allocated: %.3f GB",
            peak_memory
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nMODEL SMOKE TEST RESULTS")
    print("=" * 55)

    print(f"Device              : {config.DEVICE}")
    print(f"Input shape         : {tuple(input_ids.shape)}")
    print(f"Logits shape        : {tuple(logits.shape)}")
    print(f"Loss                : {loss.item():.6f}")

    print(f"Gradient parameters : {gradients_found}")

    print(
        f"Predictions         : "
        f"{predicted_labels.detach().cpu().tolist()}"
    )

    print(
        f"Actual labels       : "
        f"{labels.detach().cpu().tolist()}"
    )

    print(
        f"Probabilities       : "
        f"{probabilities.detach().cpu().tolist()}"
    )

    print("\nFORWARD PASS       : PASSED")
    print("BACKWARD PASS      : PASSED")
    print("INFERENCE          : PASSED")

    print("=" * 55)

    logger.info("MODEL SMOKE TEST COMPLETED SUCCESSFULLY.")


# ============================================================
# 8. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    from scripts.model_development.loneliness.fig_loneliness_dataset import (
        create_data_pipeline
    )

    config.validate_config()

    # Initialize model
    model = initialize_model()

    # Load actual training data
    tokenizer, datasets, dataloaders = create_data_pipeline()

    # Test model using a real batch
    run_model_smoke_test(
        model=model,
        dataloader=dataloaders["train"]
    )
