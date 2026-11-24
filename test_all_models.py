import torch

from backend.app.inference.model_manager import ModelManager


def main():
    manager = ModelManager()

    print("=" * 55)
    print("CENTRALIZED MODEL MANAGER - INTEGRATION TEST")
    print("=" * 55)

    modules = [
        "emotion",
        "sentiment",
        "stress",
        "loneliness"
    ]

    for module in modules:
        print(f"\nLoading {module.upper()} model...")

        model = manager.get_model(module)
        tokenizer = manager.get_tokenizer(module)

        test_text = "I have been feeling overwhelmed lately."

        inputs = tokenizer(
            test_text,
            return_tensors="pt",
            truncation=True,
            max_length=384
        )

        inputs = {
            key: value.to(manager.device)
            for key, value in inputs.items()
        }

        model.eval()

        with torch.inference_mode():
            outputs = model(**inputs)

        logits = outputs.logits

        print("Output shape:", tuple(logits.shape))
        print("Finite output:", torch.isfinite(logits).all().item())

    print("\n" + "=" * 55)
    print("FINAL MODEL STATUS")
    print("=" * 55)

    status = manager.get_status()
    print(status)

    all_loaded = all(
        status["modules"][module]["status"] == "loaded"
        for module in modules
    )

    print("\nAll models loaded:", all_loaded)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print(
            "GPU memory allocated:",
            round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
            "MB"
        )

    assert all_loaded, "One or more models failed to load."


if __name__ == "__main__":
    main()