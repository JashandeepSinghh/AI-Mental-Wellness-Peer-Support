
import torch

from backend.app.inference.model_manager import ModelManager


def main():
    manager = ModelManager()

    modules = {
        "emotion": (manager.load_emotion, 28, 256),
        "sentiment": (manager.load_sentiment, 3, 160),
        "stress": (manager.load_stress, 2, 256),
        "loneliness": (manager.load_loneliness, 2, 384),
    }

    results = {}

    test_text = (
        "I have been feeling overwhelmed "
        "and disconnected lately."
    )

    for name, (loader, expected_classes, max_length) in modules.items():

        print(f"\nTesting {name.upper()}...")

        model = loader()
        tokenizer = manager.get_tokenizer(name)

        inputs = tokenizer(
            test_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        inputs = {
            key: value.to(manager.device)
            for key, value in inputs.items()
        }

        model.eval()

        with torch.no_grad():
            output = model(**inputs)

        actual_shape = tuple(output.logits.shape)

        assert output.logits.shape[-1] == expected_classes

        results[name] = actual_shape

        print("Output shape:", actual_shape)
        print("Status: PASSED")

    print("\n========== FINAL RESULTS ==========")
    print(results)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)

        print(f"GPU allocated memory: {allocated:.2f} MB")
        print(f"GPU reserved memory: {reserved:.2f} MB")

    print("\nAll four transformer models passed.")


if __name__ == "__main__":
    main()
