import torch
import triton

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Triton:", triton.__version__)

x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
y = torch.tensor([10.0, 20.0, 30.0], device="cuda")

z = x + y

print("x:", x)
print("y:", y)
print("x + y:", z)

del x
del y
del z