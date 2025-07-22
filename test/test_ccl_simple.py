import os
import sys

# Set up Intel MPI environment
os.environ['I_MPI_ROOT'] = '/opt/aurora/24.347.0/oneapi/mpi/latest'
os.environ['LD_LIBRARY_PATH'] = f"{os.environ['I_MPI_ROOT']}/lib:{os.environ['I_MPI_ROOT']}/lib/release:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Import after setting environment
import torch
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch

print("Successfully imported all modules!")
print(f"PyTorch version: {torch.__version__}")
print(f"Intel Extension for PyTorch: {intel_extension_for_pytorch.__version__}")
print("OneCCL bindings imported successfully")