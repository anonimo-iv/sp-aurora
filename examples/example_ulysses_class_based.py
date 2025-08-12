"""
Example usage of the new class-based Ulysses Attention implementation.

This example demonstrates how to use the reorganized Ulysses attention
module that follows the yunchang structure.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import sys
import time

# Add parent directory to path to import sp_aurora
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sp_aurora import UlyssesAttention
from sp_aurora.ulysses.attn_layer import AttnType


def setup(rank, world_size):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    
    # Use CCL backend for Intel GPUs if available
    backend = 'gloo'  # Default backend
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        try:
            import oneccl_bindings_for_pytorch
            backend = 'ccl'
            print(f"Rank {rank}: Using CCL backend for Intel GPU")
        except ImportError:
            print(f"Rank {rank}: CCL not available, using Gloo backend")
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    """Clean up the distributed environment"""
    dist.destroy_process_group()


def demo_class_based_ulysses():
    """Demonstrate the new class-based Ulysses attention interface"""
    print("\n=== Class-Based Ulysses Attention Demo ===")
    
    device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Model parameters
    batch_size = 2
    seq_len = 2048
    num_heads = 16
    head_dim = 64
    
    print(f"\nModel dimensions:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dimension: {head_dim}")
    
    # Create Ulysses attention module
    ulysses_attn = UlyssesAttention(
        sequence_process_group=None,  # None for single GPU
        scatter_idx=2,
        gather_idx=1,
        use_sync=False,
        attn_type=AttnType.TORCH  # Will auto-detect Intel SYCL if available
    )
    
    # Create input tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    
    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        _ = ulysses_attn(q, k, v, causal=True)
    
    # Benchmark
    if device == 'xpu':
        torch.xpu.synchronize()
    
    start_time = time.time()
    output = ulysses_attn(q, k, v, causal=True, dropout_p=0.1, deterministic=True)
    
    if device == 'xpu':
        torch.xpu.synchronize()
    
    end_time = time.time()
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Execution time: {(end_time - start_time) * 1000:.2f} ms")
    print(f"Attention type used: {ulysses_attn.attn_type.value}")
    
    # Memory usage
    if device == 'xpu':
        allocated = torch.xpu.memory_allocated() / 1024**3
        print(f"GPU memory allocated: {allocated:.2f} GB")


def distributed_class_based_worker(rank, world_size):
    """Worker function for distributed class-based Ulysses attention"""
    setup(rank, world_size)
    
    device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
    
    # Each GPU handles a portion of the sequence
    batch_size = 2
    seq_len_per_gpu = 1024
    total_seq_len = seq_len_per_gpu * world_size
    num_heads = 16
    head_dim = 64
    
    if rank == 0:
        print(f"\n=== Distributed Class-Based Ulysses Attention Demo ===")
        print(f"World size: {world_size}")
        print(f"Total sequence length: {total_seq_len}")
        print(f"Sequence length per GPU: {seq_len_per_gpu}")
    
    # Create Ulysses attention module with process group
    ulysses_attn = UlyssesAttention(
        sequence_process_group=dist.group.WORLD,
        scatter_idx=2,
        gather_idx=1,
        use_sync=False,
        attn_type=AttnType.TORCH
    )
    
    # Create local input tensors
    q_local = torch.randn(batch_size, seq_len_per_gpu, num_heads, head_dim, 
                         device=device, dtype=torch.float16)
    k_local = torch.randn(batch_size, seq_len_per_gpu, num_heads, head_dim,
                         device=device, dtype=torch.float16)
    v_local = torch.randn(batch_size, seq_len_per_gpu, num_heads, head_dim,
                         device=device, dtype=torch.float16)
    
    # Synchronize all processes
    dist.barrier()
    
    # Run Ulysses attention
    start_time = time.time()
    output = ulysses_attn(q_local, k_local, v_local, causal=True)
    dist.barrier()
    end_time = time.time()
    
    if rank == 0:
        print(f"\nOutput shape per GPU: {output.shape}")
        print(f"Execution time: {(end_time - start_time) * 1000:.2f} ms")
        print(f"Attention type used: {ulysses_attn.attn_type.value}")
    
    cleanup()


def demo_different_attention_types():
    """Demonstrate different attention backend types"""
    print("\n=== Attention Backend Types Demo ===")
    
    device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
    
    batch_size = 1
    seq_len = 512
    num_heads = 8
    head_dim = 64
    
    # Create input tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
    
    # Test different attention types
    attention_types = [AttnType.TORCH]
    if device == 'xpu':
        attention_types.append(AttnType.INTEL_SYCL)
    
    for attn_type in attention_types:
        print(f"\n--- Testing {attn_type.value} ---")
        try:
            ulysses_attn = UlyssesAttention(attn_type=attn_type)
            
            start_time = time.time()
            output = ulysses_attn(q, k, v, causal=True)
            if device == 'xpu':
                torch.xpu.synchronize()
            end_time = time.time()
            
            print(f"  Output shape: {output.shape}")
            print(f"  Execution time: {(end_time - start_time) * 1000:.2f} ms")
            print(f"  Success!")
        except Exception as e:
            print(f"  Failed: {str(e)}")


def main():
    """Main function to run all demos"""
    print("Class-Based Ulysses Flash Attention Examples")
    print("=" * 50)
    
    # Check available backend
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print(f"Intel GPU detected: {torch.xpu.get_device_name(0)}")
        print(f"Number of Intel GPUs: {torch.xpu.device_count()}")
    else:
        print("No Intel GPU detected, running on CPU")
    
    # Demo 1: Single GPU class-based
    demo_class_based_ulysses()
    
    # Demo 2: Different attention types
    demo_different_attention_types()
    
    # Demo 3: Distributed class-based
    available_gpus = torch.xpu.device_count() if hasattr(torch, 'xpu') and torch.xpu.is_available() else 0
    world_size = min(2, available_gpus) if available_gpus > 0 else 2
    
    if world_size > 1 and available_gpus > 1:
        print(f"\n\nRunning distributed demo with {world_size} processes...")
        mp.spawn(distributed_class_based_worker, args=(world_size,), nprocs=world_size, join=True)
    else:
        print("\n\nSkipping distributed demo (requires multiple GPUs)")
    
    print("\n\nAll demos completed successfully!")


if __name__ == "__main__":
    main()