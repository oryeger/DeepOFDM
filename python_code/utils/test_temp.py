# import os
# import sys
# import torch
#
# print("Python executable:", sys.executable)
# print("Torch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
# print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
# print("DRJIT_LIBLLVM_PATH:", os.environ.get("DRJIT_LIBLLVM_PATH"))
#
# for key, value in os.environ.items():
#     if "CUDA" in key or "LD_LIBRARY_PATH" in key:
#         print(f"{key}={value}")


import os
import torch
import ctypes
import subprocess

print("CUDA available:", torch.cuda.is_available())
print("CUDA version (from torch):", torch.version.cuda)

# Try to print the actual CUDA library loaded
try:
    lib = ctypes.CDLL("libcudart.so")
    print("Loaded libcudart.so from:", lib._name)
except Exception as e:
    print("Could not load libcudart.so:", e)

# Print nvcc path if available
try:
    nvcc_path = subprocess.check_output("which nvcc", shell=True).decode().strip()
    print("nvcc found at:", nvcc_path)
except Exception as e:
    print("nvcc not found:", e)
