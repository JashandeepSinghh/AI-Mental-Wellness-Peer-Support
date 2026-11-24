
import sys
import platform
import importlib.util

print("=" * 65)
print("AI MENTAL WELLNESS CHATBOT")
print("TRAINING ENVIRONMENT VERIFICATION")
print("=" * 65)

# 1. Python environment
print("\n[1] PYTHON ENVIRONMENT")

print("Python Version :", sys.version.split()[0])
print("Python Path    :", sys.executable)
print("Platform       :", platform.platform())

# 2. Required package availability
packages = [
    "torch",
    "transformers",
    "datasets",
    "sklearn",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "tqdm",
    "accelerate"
]

print("\n[2] PACKAGE AVAILABILITY")

for package in packages:
    installed = importlib.util.find_spec(package) is not None
    status = "INSTALLED" if installed else "NOT INSTALLED"
    print(f"{package:<15}: {status}")

# 3. PyTorch and CUDA
print("\n[3] PYTORCH AND CUDA")

if importlib.util.find_spec("torch") is None:
    print("PyTorch is not installed.")
else:
    import torch

    print("PyTorch Version :", torch.__version__)
    print("CUDA Available  :", torch.cuda.is_available())
    print("CUDA Version    :", torch.version.cuda)

    if torch.cuda.is_available():

        print("GPU Count       :", torch.cuda.device_count())

        for index in range(torch.cuda.device_count()):

            gpu = torch.cuda.get_device_properties(index)

            print(f"\nGPU {index}")
            print("GPU Name        :", gpu.name)
            print(
                "Total VRAM      :",
                round(gpu.total_memory / (1024 ** 3), 2),
                "GB"
            )
            print(
                "Compute Version :",
                f"{gpu.major}.{gpu.minor}"
            )

        # 4. Actual GPU computation test
        print("\n[4] GPU COMPUTATION TEST")

        try:
            device = torch.device("cuda:0")

            tensor_a = torch.randn(1000, 1000, device=device)
            tensor_b = torch.randn(1000, 1000, device=device)

            result = torch.matmul(tensor_a, tensor_b)

            torch.cuda.synchronize()

            print("GPU Matrix Multiplication: SUCCESS")
            print("Result Device:", result.device)

        except Exception as error:
            print("GPU Computation: FAILED")
            print("Error:", error)

    else:
        print("CUDA GPU access is unavailable.")

print("\n" + "=" * 65)
print("ENVIRONMENT VERIFICATION COMPLETED")
print("=" * 65)

