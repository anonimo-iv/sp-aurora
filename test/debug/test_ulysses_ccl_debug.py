#!/usr/bin/env python3
"""
Modified Ulysses test with extensive debugging for CCL all-to-all issue
"""

import torch
import torch.distributed as dist
import os
import sys
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import oneccl_bindings to register CCL backend
import oneccl_bindings_for_pytorch

# Get MPI info
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

print(f"[Rank {mpi_rank}] Starting CCL debug test", flush=True)
print(f"[Rank {mpi_rank}] oneCCL bindings imported successfully", flush=True)

# Setup distributed
os.environ['RANK'] = str(mpi_rank)
os.environ['WORLD_SIZE'] = str(mpi_size)
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

if mpi_size > 1 and mpi_rank == 0:
    import socket
    os.environ['MASTER_ADDR'] = socket.gethostname()

# Broadcast master address
if mpi_size > 1:
    master_addr = mpi_comm.bcast(os.environ['MASTER_ADDR'], root=0)
    os.environ['MASTER_ADDR'] = master_addr
    mpi_comm.Barrier()

# Initialize process group with CCL
print(f"[Rank {mpi_rank}] Initializing CCL backend", flush=True)
dist.init_process_group(
    backend='ccl',
    init_method='env://',
    world_size=mpi_size,
    rank=mpi_rank
)

# Set device
device = torch.device(f'xpu:{mpi_rank}')
torch.xpu.set_device(device)

print(f"[Rank {mpi_rank}] Testing specific Ulysses tensor shapes", flush=True)

# Test the exact tensor shapes used in Ulysses
batch_size = 2
seq_len_per_gpu = 512
num_heads = 16
head_dim = 64
world_size = dist.get_world_size()

# Create input tensor matching Ulysses forward pass
input_tensor = torch.randn(
    batch_size, seq_len_per_gpu, num_heads, head_dim,
    device=device, dtype=torch.float16
)

print(f"[Rank {mpi_rank}] Input shape: {input_tensor.shape}", flush=True)

# Reshape exactly as in intel_all_to_all_4d
seq_len = seq_len_per_gpu * world_size
shard_heads = num_heads // world_size

print(f"[Rank {mpi_rank}] Reshaping for all-to-all...", flush=True)
input_reshaped = input_tensor.reshape(batch_size, seq_len_per_gpu, world_size, shard_heads, head_dim)
print(f"[Rank {mpi_rank}] Reshaped: {input_reshaped.shape}", flush=True)

# Transpose
input_t = input_reshaped.transpose(0, 2).contiguous()
print(f"[Rank {mpi_rank}] Transposed: {input_t.shape}", flush=True)

# Try all-to-all with smaller tensor first
print(f"[Rank {mpi_rank}] Testing small tensor all-to-all...", flush=True)
small_tensor = torch.ones(4, device=device, dtype=torch.float16)
try:
    dist.all_to_all_single(small_tensor, small_tensor)
    print(f"[Rank {mpi_rank}] ✓ Small tensor all-to-all succeeded", flush=True)
except Exception as e:
    print(f"[Rank {mpi_rank}] ✗ Small tensor all-to-all failed: {e}", flush=True)

# Try with float32 instead of float16
print(f"[Rank {mpi_rank}] Testing with float32...", flush=True)
input_f32 = input_t.float()
output_f32 = torch.empty_like(input_f32)
try:
    dist.all_to_all_single(output_f32, input_f32)
    print(f"[Rank {mpi_rank}] ✓ Float32 all-to-all succeeded", flush=True)
except Exception as e:
    print(f"[Rank {mpi_rank}] ✗ Float32 all-to-all failed: {e}", flush=True)

# Try the actual float16 all-to-all
print(f"[Rank {mpi_rank}] Testing actual float16 all-to-all...", flush=True)
output = torch.empty_like(input_t)
try:
    print(f"[Rank {mpi_rank}] Calling all_to_all_single...", flush=True)
    dist.all_to_all_single(output, input_t)
    print(f"[Rank {mpi_rank}] ✓ Float16 all-to-all succeeded", flush=True)
except Exception as e:
    print(f"[Rank {mpi_rank}] ✗ Float16 all-to-all failed: {e}", flush=True)
    import traceback
    traceback.print_exc()

print(f"[Rank {mpi_rank}] Test completed", flush=True)

# Cleanup
if dist.is_initialized():
    dist.destroy_process_group()

print(f"[Rank {mpi_rank}] Script finished", flush=True)