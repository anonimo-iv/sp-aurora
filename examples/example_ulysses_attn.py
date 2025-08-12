"""
Example usage of Intel Ulysses (Sequence Parallel) Flash Attention

This example demonstrates how to use Ulysses attention for distributing
long sequences across multiple GPUs using all-to-all communication.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import sys
import time

# Add parent directory to path to import sp_aurora
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sp_aurora import ulysses_flash_attn_func, ulysses_flash_attn_qkvpacked_func


def setup(rank, world_size):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    
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


def demo_single_gpu():
    """Demonstrate Ulysses attention on a single GPU"""
    print("\n=== Single GPU Ulysses Attention Demo ===")
    
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
    
    # Create input tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    
    # Warmup
    for _ in range(3):
        _ = ulysses_flash_attn_func(q, k, v, causal=True)
    
    # Benchmark
    if device == 'xpu':
        torch.xpu.synchronize()
    
    start_time = time.time()
    output = ulysses_flash_attn_func(q, k, v, causal=True)
    
    if device == 'xpu':
        torch.xpu.synchronize()
    
    end_time = time.time()
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Execution time: {(end_time - start_time) * 1000:.2f} ms")
    
    # Memory usage
    if device == 'xpu':
        allocated = torch.xpu.memory_allocated() / 1024**3
        print(f"GPU memory allocated: {allocated:.2f} GB")


def distributed_worker(rank, world_size):
    """Worker function for distributed Ulysses attention"""
    setup(rank, world_size)
    
    device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
    
    # Each GPU handles a portion of the sequence
    batch_size = 2
    seq_len_per_gpu = 1024  # Each GPU processes 1024 tokens
    total_seq_len = seq_len_per_gpu * world_size
    num_heads = 16
    head_dim = 64
    
    if rank == 0:
        print(f"\n=== Distributed Ulysses Attention Demo ===")
        print(f"World size: {world_size}")
        print(f"Total sequence length: {total_seq_len}")
        print(f"Sequence length per GPU: {seq_len_per_gpu}")
        print(f"Number of heads: {num_heads}")
        print(f"Head dimension: {head_dim}")
    
    # Create local input tensors (each GPU has its portion)
    q_local = torch.randn(batch_size, seq_len_per_gpu, num_heads, head_dim, 
                         device=device, dtype=torch.float16)
    k_local = torch.randn(batch_size, seq_len_per_gpu, num_heads, head_dim,
                         device=device, dtype=torch.float16)
    v_local = torch.randn(batch_size, seq_len_per_gpu, num_heads, head_dim,
                         device=device, dtype=torch.float16)
    
    # Synchronize all processes
    dist.barrier()
    
    # Warmup
    for _ in range(3):
        _ = ulysses_flash_attn_func(q_local, k_local, v_local, causal=True)
    
    # Synchronize before timing
    dist.barrier()
    
    start_time = time.time()
    
    # Run Ulysses attention
    # This will:
    # 1. All-to-all to redistribute sequences across GPUs
    # 2. Compute local attention on full sequence with reduced heads
    # 3. All-to-all to redistribute back to original partitioning
    output = ulysses_flash_attn_func(q_local, k_local, v_local, causal=True)
    
    # Synchronize after computation
    dist.barrier()
    end_time = time.time()
    
    if rank == 0:
        print(f"\nOutput shape per GPU: {output.shape}")
        print(f"Execution time: {(end_time - start_time) * 1000:.2f} ms")
        
        if device == 'xpu':
            allocated = torch.xpu.memory_allocated() / 1024**3
            print(f"GPU memory allocated (rank 0): {allocated:.2f} GB")
    
    # Test QKV packed format
    if rank == 0:
        print("\n--- Testing QKV Packed Format ---")
    
    qkv_packed = torch.stack([q_local, k_local, v_local], dim=2)
    output_packed = ulysses_flash_attn_qkvpacked_func(qkv_packed, causal=True)
    
    # Verify outputs match
    torch.testing.assert_close(output, output_packed)
    
    if rank == 0:
        print("✓ QKV packed format produces identical results")
    
    cleanup()


def demo_attention_patterns():
    """Demonstrate different attention patterns and options"""
    print("\n=== Attention Patterns Demo ===")
    
    device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
    
    batch_size = 1
    seq_len = 512
    num_heads = 8
    head_dim = 64
    
    # Create input tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
    
    # 1. Causal attention
    print("\n1. Causal Attention:")
    output_causal = ulysses_flash_attn_func(q, k, v, causal=True)
    print(f"   Output shape: {output_causal.shape}")
    
    # 2. Non-causal attention
    print("\n2. Non-causal Attention:")
    output_non_causal = ulysses_flash_attn_func(q, k, v, causal=False)
    print(f"   Output shape: {output_non_causal.shape}")
    
    # 3. Attention with dropout
    print("\n3. Attention with Dropout (p=0.1):")
    output_dropout = ulysses_flash_attn_func(q, k, v, causal=True, dropout_p=0.1, deterministic=True)
    print(f"   Output shape: {output_dropout.shape}")
    
    # 4. Custom softmax scale
    print("\n4. Custom Softmax Scale:")
    custom_scale = 1.0 / (head_dim ** 0.25)  # Different from default sqrt(d)
    output_scaled = ulysses_flash_attn_func(q, k, v, causal=True, softmax_scale=custom_scale)
    print(f"   Output shape: {output_scaled.shape}")
    print(f"   Softmax scale: {custom_scale:.4f}")


def main():
    """Main function to run all demos"""
    print("Intel Ulysses Flash Attention Examples")
    print("=" * 50)
    
    # Check available backend
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print(f"Intel GPU detected: {torch.xpu.get_device_name(0)}")
        print(f"Number of Intel GPUs: {torch.xpu.device_count()}")
    else:
        print("No Intel GPU detected, running on CPU")
    
    # Demo 1: Single GPU
    demo_single_gpu()
    
    # Demo 2: Attention patterns
    demo_attention_patterns()
    
    # Demo 3: Distributed (multi-GPU)
    # Limit to 2 GPUs for the demo
    available_gpus = torch.xpu.device_count() if hasattr(torch, 'xpu') and torch.xpu.is_available() else 0
    world_size = min(2, available_gpus) if available_gpus > 0 else 2
    
    if world_size > 1 and available_gpus > 1:
        print(f"\n\nRunning distributed demo with {world_size} processes (out of {available_gpus} available)...")
        mp.spawn(distributed_worker, args=(world_size,), nprocs=world_size, join=True)
    else:
        print("\n\nSkipping distributed demo (requires multiple GPUs)")
    
    print("\n\nAll demos completed successfully!")


if __name__ == "__main__":
    main()