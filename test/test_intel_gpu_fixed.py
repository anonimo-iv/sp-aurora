#!/usr/bin/env python3
"""
Test script for fixed Intel GPU Ring Flash Attention with additional debugging
"""

import os
import sys
import torch
import torch.distributed as dist

# Set environment variables for Intel GPU and oneCCL
os.environ['CCL_BACKEND'] = 'native'
os.environ['CCL_ATL_TRANSPORT'] = 'mpi'
os.environ['FI_PROVIDER'] = 'cxi'
os.environ['CCL_ZE_IPC_EXCHANGE'] = 'drmfd'
os.environ['CCL_ZE_ENABLE'] = '1'
os.environ['CCL_LOG_LEVEL'] = 'info'
os.environ['IPEX_XPU_ONEDNN_LAYOUT'] = '1'
os.environ['IPEX_OFFLINE_COMPILER'] = '1'
os.environ['SYCL_CACHE_PERSISTENT'] = '1'
os.environ['SYCL_DEVICE_FILTER'] = 'level_zero:*'
os.environ['SYCL_PI_LEVEL_ZERO_PROGRAM_BUILD_TRACK'] = '2'
os.environ['CCL_ATL_SYNC_COLL'] = '1'
os.environ['CCL_OP_SYNC'] = '1'

# Additional debugging environment variables
os.environ['CCL_P2P_ACCESS_POLICY'] = 'on'
os.environ['CCL_ALLREDUCE'] = 'ring'
os.environ['FI_CXI_DISABLE_HOST_REGISTER'] = '1'

# Check for Intel GPU support
try:
    import intel_extension_for_pytorch as ipex
    if not torch.xpu.is_available():
        print("Intel GPU not available, exiting")
        sys.exit(0)
except ImportError:
    print("Intel Extension for PyTorch not installed, exiting")
    sys.exit(0)

# Import ring flash attention modules
from ring_flash_attn.intel_ring_flash_attn_fixed import intel_ring_flash_attn_func

def test_fixed_ring_attention():
    """Test the fixed ring attention implementation"""
    print("\n" + "="*60)
    print("TEST: Fixed Intel Ring Attention with Synchronous P2P")
    print("="*60)
    
    # Check if we're in distributed mode
    is_distributed_run = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    
    if not dist.is_initialized():
        if not is_distributed_run:
            print("⚠️  Skipping distributed test - not running with torchrun")
            print("   Run with: torchrun --nproc_per_node=2 test_intel_gpu_fixed.py")
            return True
        
        print("Attempting distributed initialization...")
        try:
            import datetime
            dist.init_process_group(backend='ccl', timeout=datetime.timedelta(seconds=30))
            print("✅ CCL backend initialized successfully")
        except Exception as e:
            print(f"❌ CCL initialization failed: {e}")
            return False
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'xpu:{rank}' if torch.xpu.device_count() > 1 else 'xpu'
    dtype = torch.float16
    
    print(f"Process {rank}/{world_size} using device {device}")
    
    # Test configuration
    batch_size = 1
    seqlen = 512  # Must be divisible by world_size
    nheads = 8
    d = 64
    
    if seqlen % world_size != 0:
        seqlen = (seqlen // world_size) * world_size
    
    # Create and broadcast test tensors
    if rank == 0:
        qkv = torch.randn(batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True)
    else:
        qkv = torch.empty(batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True)
    
    dist.broadcast(qkv, src=0)
    
    # Get local chunk
    local_qkv = qkv.chunk(world_size, dim=1)[rank].detach().clone()
    local_qkv.requires_grad = True
    
    try:
        import time
        
        print(f"[Rank {rank}] Starting ring attention forward pass...")
        start_time = time.time()
        
        # Test ring attention forward with fixed implementation
        ring_out, ring_lse, _ = intel_ring_flash_attn_func(
            local_qkv[:, :, 0],  # q
            local_qkv[:, :, 1],  # k
            local_qkv[:, :, 2],  # v
            dropout_p=0.0,
            causal=True,
            return_attn_probs=False,
        )
        
        elapsed = time.time() - start_time
        
        print(f"[Rank {rank}] ✅ Ring attention forward pass successful in {elapsed:.2f}s")
        print(f"[Rank {rank}] Output shape: {ring_out.shape}, LSE shape: {ring_lse.shape}")
        
        # Test backward
        local_dout = torch.randn_like(ring_out)
        ring_out.backward(local_dout)
        
        print(f"[Rank {rank}] ✅ Ring attention backward pass successful")
        
    except Exception as e:
        print(f"[Rank {rank}] ❌ Ring attention test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Barrier to synchronize all ranks
    try:
        print(f"[Rank {rank}] Waiting at barrier...")
        dist.barrier()
        print(f"[Rank {rank}] Passed barrier")
    except Exception as e:
        print(f"[Rank {rank}] ⚠️  Barrier failed: {e}")
    
    if rank == 0:
        print("\n✅ Fixed distributed ring attention test passed!")
    
    return True


def main():
    """Run the fixed Intel GPU test"""
    print("🚀 Intel GPU Ring Flash Attention - Fixed Version Test")
    print("="*80)
    
    # Check Intel GPU availability
    if not torch.xpu.is_available():
        print("❌ Intel GPU not available, exiting")
        return 1
    
    print(f"✅ Intel GPU detected: {torch.xpu.device_count()} device(s)")
    print(f"✅ Intel Extension for PyTorch version: {ipex.__version__}")
    
    success = test_fixed_ring_attention()
    
    if success:
        print("\n🎉 Fixed Intel GPU Ring Flash Attention test passed!")
        return 0
    else:
        print("\n❌ Test failed.")
        return 1


if __name__ == "__main__":
    # Clean up any existing process groups
    if dist.is_initialized():
        dist.destroy_process_group()
    
    sys.exit(main())