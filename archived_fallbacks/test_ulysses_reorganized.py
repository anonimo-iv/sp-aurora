#!/usr/bin/env python3
"""
Test script for the reorganized Ulysses attention implementation
MPI-compatible test that works with both torchrun and mpiexec

Usage:
    # With torchrun
    torchrun --nproc_per_node=2 test_ulysses_reorganized.py
    
    # With mpiexec
    mpiexec -n 2 python test_ulysses_reorganized.py
    
    # With Intel MPI (Intel GPU)
    mpiexec -n 2 -genv CCL_BACKEND=native -genv CCL_ATL_TRANSPORT=ofi \
        python test_ulysses_reorganized.py
"""

import torch
import torch.distributed as dist
import sys
import os
import traceback
import socket
import datetime
from mpi4py import MPI
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ring_flash_attn import UlyssesAttention, ulysses_flash_attn_func
from ring_flash_attn.ulysses.attn_layer import AttnType
from ring_flash_attn.comm.all_to_all import SeqAllToAll4D, all_to_all_4D


def test_ulysses_class_single_gpu():
    """Test UlyssesAttention class on single GPU"""
    print("Testing UlyssesAttention class (single GPU)...")
    
    device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
    
    # Create attention module
    ulysses_attn = UlyssesAttention(attn_type=AttnType.TORCH)
    
    # Test parameters
    batch_size = 2
    seq_len = 512
    num_heads = 8
    head_dim = 64
    
    # Create tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
    
    # Forward pass
    output = ulysses_attn(q, k, v, causal=True)
    
    # Check output shape
    assert output.shape == q.shape, f"Output shape mismatch: {output.shape} vs {q.shape}"
    
    # Compare with PyTorch SDPA
    with torch.no_grad():
        ref_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=1.0/torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))
        )
    
    # Check if outputs are close
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-3)
    
    print("✓ UlyssesAttention class test passed")


def test_all_to_all_4d():
    """Test all_to_all_4D function"""
    print("\nTesting all_to_all_4D function...")
    
    device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
    
    # Test with single process (should return input unchanged)
    batch_size = 2
    seq_len = 512
    num_heads = 8
    head_dim = 64
    
    input_tensor = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    
    # Forward direction
    output = all_to_all_4D(input_tensor, scatter_idx=2, gather_idx=1)
    assert output.shape == input_tensor.shape
    torch.testing.assert_close(output, input_tensor)
    
    # Backward direction
    output = all_to_all_4D(input_tensor, scatter_idx=1, gather_idx=2)
    assert output.shape == input_tensor.shape
    torch.testing.assert_close(output, input_tensor)
    
    print("✓ all_to_all_4D test passed")


def test_seq_all_to_all_autograd():
    """Test SeqAllToAll4D autograd function"""
    print("\nTesting SeqAllToAll4D autograd...")
    
    device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
    
    batch_size = 1
    seq_len = 256
    num_heads = 8
    head_dim = 64
    
    # Create tensor with gradients
    input_tensor = torch.randn(
        batch_size, seq_len, num_heads, head_dim, 
        device=device, dtype=torch.float32, requires_grad=True
    )
    
    # Forward pass
    output = SeqAllToAll4D.apply(None, input_tensor, 2, 1, False)
    
    # Create gradient
    grad_output = torch.randn_like(output)
    
    # Backward pass
    output.backward(grad_output)
    
    # Check gradient exists and has correct shape
    assert input_tensor.grad is not None
    assert input_tensor.grad.shape == input_tensor.shape
    
    print("✓ SeqAllToAll4D autograd test passed")


def test_backward_compatibility():
    """Test that old function interface still works"""
    print("\nTesting backward compatibility...")
    
    device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
    
    batch_size = 1
    seq_len = 256
    num_heads = 8
    head_dim = 64
    
    # Create tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    
    # Test old function interface
    output = ulysses_flash_attn_func(q, k, v, causal=True)
    
    # Check output shape
    assert output.shape == q.shape
    
    print("✓ Backward compatibility test passed")


def test_attention_options():
    """Test various attention options"""
    print("\nTesting attention options...")
    
    device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
    ulysses_attn = UlyssesAttention()
    
    batch_size = 1
    seq_len = 128
    num_heads = 4
    head_dim = 32
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
    
    # Test different options
    options = [
        {"causal": True},
        {"causal": False},
        {"dropout_p": 0.1, "deterministic": True},
        {"softmax_scale": 0.5},
        {"window_size": (64, 0)},
    ]
    
    for i, opts in enumerate(options):
        output = ulysses_attn(q, k, v, **opts)
        assert output.shape == q.shape
        print(f"  ✓ Option {i+1} passed: {opts}")
    
    print("✓ All attention options tests passed")


def test_distributed_ulysses_class(rank, world_size, device):
    """Test UlyssesAttention class in distributed setting"""
    print(f"\n{'='*60}")
    print("TEST: Distributed UlyssesAttention Class")
    print(f"{'='*60}")
    
    try:
        
        if world_size == 1:
            print(f"[Rank {rank}] Single process - skipping distributed test")
            return True
        
        print(f"[Rank {rank}] Testing UlyssesAttention with {world_size} GPUs")
        
        # Create attention module
        ulysses_attn = UlyssesAttention(attn_type=AttnType.TORCH)
        
        # Test parameters (per GPU)
        batch_size = 2
        seq_len_per_gpu = 256
        num_heads = 8
        head_dim = 64
        dtype = torch.float32 if device.type == 'cpu' else torch.float16
        
        # Create tensors
        q = torch.randn(batch_size, seq_len_per_gpu, num_heads, head_dim, 
                       device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len_per_gpu, num_heads, head_dim, 
                       device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len_per_gpu, num_heads, head_dim, 
                       device=device, dtype=dtype)
        
        print(f"[Rank {rank}] Running distributed attention...")
        
        # Forward pass
        output = ulysses_attn(q, k, v, causal=True)
        
        # Check output shape
        assert output.shape == q.shape
        print(f"[Rank {rank}] ✓ Output shape correct: {output.shape}")
        
        # Test with different options
        output_non_causal = ulysses_attn(q, k, v, causal=False)
        assert output_non_causal.shape == q.shape
        print(f"[Rank {rank}] ✓ Non-causal attention passed")
        
        return True
        
    except Exception as e:
        print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] ❌ Distributed test failed: {e}")
        traceback.print_exc()
        return False


def test_distributed_all_to_all():
    """Test all-to-all operations in distributed setting"""
    if not dist.is_initialized():
        print("Distributed not initialized, skipping test")
        return True
    
    print(f"\n{'='*60}")
    print("TEST: Distributed All-to-All Operations")
    print(f"{'='*60}")
    
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device('xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu')
        
        # Test parameters
        batch_size = 2
        seq_len_per_gpu = 256
        num_heads = 8 * world_size  # Total heads divided across GPUs
        head_dim = 64
        
        # Create input tensor
        input_tensor = torch.randn(batch_size, seq_len_per_gpu, num_heads, head_dim, device=device)
        
        # Forward all-to-all (scatter heads, gather sequence)
        print(f"[Rank {rank}] Testing forward all-to-all...")
        output_forward = all_to_all_4D(input_tensor, scatter_idx=2, gather_idx=1)
        expected_shape = (batch_size, seq_len_per_gpu * world_size, num_heads // world_size, head_dim)
        assert output_forward.shape == expected_shape
        print(f"[Rank {rank}] ✓ Forward all-to-all shape: {output_forward.shape}")
        
        # Backward all-to-all (scatter sequence, gather heads)
        print(f"[Rank {rank}] Testing backward all-to-all...")
        output_backward = all_to_all_4D(output_forward, scatter_idx=1, gather_idx=2)
        assert output_backward.shape == input_tensor.shape
        print(f"[Rank {rank}] ✓ Backward all-to-all restored shape: {output_backward.shape}")
        
        return True
        
    except Exception as e:
        print(f"[Rank {rank}] ❌ All-to-all test failed: {e}")
        traceback.print_exc()
        return False


def run_single_gpu_tests():
    """Run tests that don't require distributed setup"""
    print("\nRunning single GPU tests...")
    print("=" * 50)
    
    test_ulysses_class_single_gpu()
    test_all_to_all_4d()
    test_seq_all_to_all_autograd()
    test_backward_compatibility()
    test_attention_options()


def setup_distributed():
    """Initialize distributed environment with MPI coordination"""
    # Get MPI info
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    
    # Setup environment variables
    os.environ['RANK'] = str(mpi_rank)
    os.environ['WORLD_SIZE'] = str(mpi_size)
    
    # Broadcast master address and port from rank 0
    if mpi_size > 1:
        if mpi_rank == 0:
            master_addr = socket.gethostname()
            master_port = 12356  # Different port from other test
        else:
            master_addr = None
            master_port = None
        
        master_addr = mpi_comm.bcast(master_addr, root=0)
        master_port = mpi_comm.bcast(master_port, root=0)
        
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
    
    # Synchronize before initializing process group
    if mpi_size > 1:
        mpi_comm.Barrier()
    
    # Set device
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device(f'xpu:{mpi_rank % torch.xpu.device_count()}')
        torch.xpu.set_device(device)
        backend = 'ccl'
    else:
        device = torch.device('cpu')
        backend = 'gloo'
    
    # Initialize process group
    if mpi_size > 1:
        dist.init_process_group(
            backend=backend,
            init_method='env://',
            world_size=mpi_size,
            rank=mpi_rank,
            timeout=datetime.timedelta(seconds=300)
        )
    
    return {
        'rank': mpi_rank,
        'world_size': mpi_size,
        'device': device,
        'backend': backend
    }


def cleanup_distributed():
    """Clean up distributed environment"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    """Main test function with MPI support"""
    print(f"\n{'='*60}")
    print("Testing Reorganized Ulysses Attention Implementation")
    print(f"{'='*60}")
    
    try:
        # Setup distributed environment
        setup_info = setup_distributed()
        
        rank = setup_info['rank']
        world_size = setup_info['world_size']
        device = setup_info['device']
        backend = setup_info['backend']
        
        print(f"[Rank {rank}] World size: {world_size}")
        print(f"[Rank {rank}] Device: {device}")
        print(f"[Rank {rank}] Backend: {backend}")
        
        # Run single GPU tests on rank 0 only
        if rank == 0:
            run_single_gpu_tests()
        
        # Synchronize before distributed tests
        if dist.is_initialized():
            dist.barrier()
        
        # Run distributed tests if multiple processes
        if world_size > 1:
            success = test_distributed_ulysses_class(rank, world_size, device)
            if not success:
                return 1
            
            success = test_distributed_all_to_all()
            if not success:
                return 1
        
        # Final synchronization
        if dist.is_initialized():
            dist.barrier()
        
        if rank == 0:
            print(f"\n{'='*60}")
            print("✅ All tests passed successfully!")
            print(f"{'='*60}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        traceback.print_exc()
        return 1
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    exit(main())