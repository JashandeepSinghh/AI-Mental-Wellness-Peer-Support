
import torch

print("=" * 55)
print("PYTORCH GPU VERIFICATION")
print("=" * 55)

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Runtime:", torch.version.cuda)

if torch.cuda.is_available():

    print("GPU Name:", torch.cuda.get_device_name(0))

    print(
        "GPU Memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )

    # Allocate tensors directly on the GPU
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.randn(1000, 1000, device="cuda")

    # Perform GPU matrix multiplication
    result = torch.matmul(x, y)

    torch.cuda.synchronize()

    print("Matrix Multiplication: SUCCESS")
    print("Tensor Device:", result.device)

    print("\nGPU IS READY FOR MODEL TRAINING.")

else:
    print("\nCUDA NOT AVAILABLE.")
    print("GPU training setup requires troubleshooting.")

print("=" * 55)
