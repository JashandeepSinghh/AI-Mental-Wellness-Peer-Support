import torch

from backend.app.inference.model_manager import ModelManager


def main():
    manager = ModelManager()

    print("Device:", manager.device)
    print("\nLoading Emotion model...")

    model = manager.load_emotion()
    tokenizer = manager.get_tokenizer("emotion")

    text = "I feel happy and grateful, but also a little nervous."

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {
        key: value.to(manager.device)
        for key, value in inputs.items()
    }

    model.eval()

    with torch.inference_mode():
        outputs = model(**inputs)

    logits = outputs.logits

    print("\n--- EMOTION MODEL TEST ---")
    print("Logits shape:", tuple(logits.shape))
    print("Finite output:", torch.isfinite(logits).all().item())
    print("Sigmoid scores:", torch.sigmoid(logits.float()).cpu().tolist())
    print("\nModel status:", manager.get_status())


if __name__ == "__main__":
    main()