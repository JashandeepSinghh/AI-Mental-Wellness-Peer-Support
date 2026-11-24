
import importlib
import importlib.metadata
import torch

print("=" * 65)
print("ML DEPENDENCY VERIFICATION")
print("=" * 65)

packages = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "scikit-learn": "sklearn",
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "tqdm": "tqdm",
    "accelerate": "accelerate",
    "evaluate": "evaluate",
    "tensorboard": "tensorboard"
}

print("\n[1] PACKAGE VERSIONS")

for distribution, module in packages.items():
    try:
        importlib.import_module(module)
        version = importlib.metadata.version(distribution)
        print(f"PASS: {distribution:<15} {version}")
    except Exception as error:
        print(f"FAIL: {distribution:<15} {error}")

print("\n[2] TRANSFORMERS VERIFICATION")

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification
    )

    print("PASS: Transformer classes imported successfully")

except Exception as error:
    print("FAIL:", error)

print("\n[3] GPU VERIFICATION")

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA runtime:", torch.version.cuda)
else:
    print("WARNING: GPU unavailable")

print("\n" + "=" * 65)
print("VERIFICATION COMPLETED")
print("=" * 65)
