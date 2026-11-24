
import torch
from pathlib import Path

CHECKPOINTS = {
    "STRESS": Path(
        "models/stress/dreaddit_distilbert/best_checkpoint/best_checkpoint.pt"
    ),
    "LONELINESS": Path(
        "models/loneliness/fig_loneliness_distilbert/best_checkpoint/best_checkpoint.pt"
    )
}


def inspect_checkpoint(name, path):

    print("\n" + "=" * 70)
    print(f"{name} CHECKPOINT")
    print("=" * 70)

    if not path.exists():
        print(f"FILE NOT FOUND: {path}")
        return

    # Load checkpoint safely onto CPU.
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    if not isinstance(checkpoint, dict):
        print("Unexpected checkpoint format.")
        return

    print("\nCHECKPOINT KEYS:")
    for key in checkpoint:
        print("-", key)

    print("\nMODEL METADATA:")

    metadata_keys = [
        "epoch",
        "model_name",
        "max_length",
        "label2id",
        "id2label",
        "best_validation_macro_f1",
        "validation_metrics",
        "random_seed"
    ]

    for key in metadata_keys:
        if key in checkpoint:
            print(f"\n{key}:")
            print(checkpoint[key])

    state_dict = checkpoint.get("model_state_dict")

    if not isinstance(state_dict, dict):
        print("\nMODEL STATE DICTIONARY NOT FOUND.")
        return

    print("\nMODEL STATE DICTIONARY:")
    print("Total parameter tensors:", len(state_dict))

    total_parameters = 0

    for key, tensor in state_dict.items():

        if isinstance(tensor, torch.Tensor):
            total_parameters += tensor.numel()

            print(
                f"{key}: "
                f"shape={tuple(tensor.shape)}, "
                f"dtype={tensor.dtype}"
            )

    print("\nTOTAL PARAMETERS:", total_parameters)

    del checkpoint


if __name__ == "__main__":

    for name, path in CHECKPOINTS.items():
        inspect_checkpoint(name, path)
