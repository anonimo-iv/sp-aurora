#!/usr/bin/env python3
"""
Fixed test to debug ring communication hang issue with CCL requirements
"""

import os
import sys
import torch
import torch.distributed as dist
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ring flash attention modules
from ring_flash_attn.intel_ring_flash_attn import intel_ring_flash_attn_func

def main():
    print(f"\n[PID {os.getpid()}] Starting fixed debug test")
    
    # Initialize distributed
    if not dist.is_initialized():
        print(f"[PID {os.getpid()}] Initializing process group with backend='ccl'")
        dist.init_process_group(backend='ccl')
        print(f"[PID {os.getpid()}] Process group initialized")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"[Rank {rank}] Setup complete - world_size={world_size}")
    
    # Skip the barrier to see if we can get past initialization
    print(f"[Rank {rank}] Skipping initial barrier to test ring attention directly")
    
    # Simple test parameters
    batch_size = 1
    seqlen = 256  # Small size for debugging
    nheads = 8
    d = 64
    device = 'xpu'
    dtype = torch.float16
    
    # Create local tensors
    local_seqlen = seqlen // world_size
    print(f"[Rank {rank}] Creating tensors with local_seqlen={local_seqlen}")
    
    q = torch.randn(batch_size, local_seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, local_seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, local_seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    
    print(f"[Rank {rank}] Tensors created - q.shape={q.shape}")
    
    # Skip P2P test and barrier - go directly to ring attention
    print(f"\n[Rank {rank}] Skipping P2P test and barriers to test ring attention directly")
    
    # Test ring attention
    try:
        print(f"\n[Rank {rank}] *** CALLING intel_ring_flash_attn_func ***")
        start_time = time.time()
        
        out = intel_ring_flash_attn_func(
            q, k, v,
            dropout_p=0.0,
            causal=True,
            return_attn_probs=False,
        )
        
        elapsed = time.time() - start_time
        print(f"[Rank {rank}] *** intel_ring_flash_attn_func COMPLETED in {elapsed:.2f}s ***")
        print(f"[Rank {rank}] Output shape: {out.shape}")
        
    except Exception as e:
        print(f"[Rank {rank}] ERROR in ring attention: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Skip final barrier
    print(f"[Rank {rank}] Skipping final barrier")
    
    # Cleanup
    print(f"[Rank {rank}] Test completed successfully!")
    dist.destroy_process_group()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())