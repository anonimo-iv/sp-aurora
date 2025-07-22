#!/usr/bin/env python3
"""
Fixed test to debug ring communication hang issue with CCL requirements
"""

import datetime
from time import perf_counter_ns
import sys
import os
import socket
from mpi4py import MPI
import intel_extension_for_pytorch  # Added Extra
import torch
import torch.nn.parallel
import torch.distributed as dist
import oneccl_bindings_for_pytorch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ring flash attention modules
from ring_flash_attn.intel_ring_flash_attn import intel_ring_flash_attn_func

def main():

    MPI.COMM_WORLD.Barrier()

    os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
    os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
    mpi_world_size = MPI.COMM_WORLD.Get_size()
    mpi_my_rank = MPI.COMM_WORLD.Get_rank()

    if mpi_my_rank == 0:
        master_addr = socket.gethostname()
        sock = socket.socket()
        sock.bind(('', 0))
        # master_port = sock.getsockname()[1] 
        master_port = 2345
    else:
        master_addr = None
        master_port = None

    master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    master_port = MPI.COMM_WORLD.bcast(master_port, root=0)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    MPI.COMM_WORLD.Barrier()
    dist.init_process_group(backend="ccl", init_method='env://', world_size=mpi_world_size, rank=mpi_my_rank, timeout=datetime.timedelta(seconds=3600))
    MPI.COMM_WORLD.Barrier()

    dist_my_rank = dist.get_rank()
    dist_world_size = dist.get_world_size()

    def get_default_device():
        if torch.xpu.is_available():
            return torch.device(f"xpu:{dist_my_rank % 12}")
        else:
            return torch.device('cpu')

    device = get_default_device()
    print(f"\n[Rank {dist_my_rank}] Starting fixed debug test")
    
    print(f"[Rank {dist_my_rank}] Setup complete - world_size={dist_world_size}")
    
    # Skip the barrier to see if we can get past initialization
    print(f"[Rank {dist_my_rank}] Skipping initial barrier to test ring attention directly")
    
    # Simple test parameters
    batch_size = 1
    seqlen = 256  # Small size for debugging
    nheads = 8
    d = 64
    dtype = torch.float16
    
    # Create local tensors
    local_seqlen = seqlen // dist_world_size
    print(f"[Rank {dist_my_rank}] Creating tensors with local_seqlen={local_seqlen}")
    
    q = torch.randn(batch_size, local_seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, local_seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, local_seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    
    print(f"[Rank {dist_my_rank}] Tensors created - q.shape={q.shape}")
    
    # Skip P2P test and barrier - go directly to ring attention
    print(f"\n[Rank {dist_my_rank}] Skipping P2P test and barriers to test ring attention directly")
    
    # Test ring attention
    try:
        print(f"\n[Rank {dist_my_rank}] *** CALLING intel_ring_flash_attn_func ***")
        start_time = perf_counter_ns()
        
        out = intel_ring_flash_attn_func(
            q, k, v,
            dropout_p=0.0,
            causal=True,
            return_attn_probs=False,
        )
        
        elapsed = perf_counter_ns() - start_time
        print(f"[Rank {dist_my_rank}] *** intel_ring_flash_attn_func COMPLETED in {elapsed / 1e9:.2f}s ***")
        print(f"[Rank {dist_my_rank}] Output shape: {out.shape}")
        
    except Exception as e:
        print(f"[Rank {dist_my_rank}] ERROR in ring attention: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Skip final barrier
    print(f"[Rank {dist_my_rank}] Skipping final barrier")
    
    # Cleanup
    print(f"[Rank {dist_my_rank}] Test completed successfully!")
    dist.destroy_process_group()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())