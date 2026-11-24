import torch

from backend.app.inference.model_manager import ModelManager


def main():
    manager = ModelManager()

    print("Device:", manager.device)
    print("\nLoading Stress model...")

    model = manager.load_stress()
    tokenizer = manager.get_tokenizer("stress")

    text = "I have been feeling overwhelmed by my workload."

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
    probabilities = torch.softmax(logits.float(), dim=-1)

    print("\n--- STRESS MODEL TEST ---")
    print("Logits shape:", tuple(logits.shape))
    print("Finite output:", torch.isfinite(logits).all().item())
    print("Probabilities:", probabilities.cpu().tolist())
    print("\nModel status:", manager.get_status())


if __name__ == "__main__":
    main()