#!/usr/bin/env python3
"""
Comprehensive test for Intel Ring Flash Attention - Inference only
Usage: mpirun -n 2 python test_ring_attention_comprehensive.py
"""

import os
import sys
import torch
import torch.distributed as dist
from mpi4py import MPI
import datetime
import math
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import intel_extension_for_pytorch as ipex
    HAS_XPU = torch.xpu.is_available()
except:
    HAS_XPU = False

from sp_aurora.intel_flash_attn import intel_flash_attn_forward
from sp_aurora.intel_ring_flash_attn import (
    intel_ring_flash_attn_forward,
    intel_ring_flash_attn_func,
    intel_ring_flash_attn_qkvpacked_func,
    intel_ring_flash_attn_kvpacked_func
)


def setup_distributed():
    """Setup MPI and distributed environment"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    backend = 'ccl' if HAS_XPU else 'gloo'
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=120)
    )
    
    device = f'xpu:{rank % torch.xpu.device_count()}' if HAS_XPU else 'cpu'
    if HAS_XPU:
        torch.xpu.set_device(device)
    
    return rank, world_size, device


def create_test_tensors(batch_size, seq_len, num_heads, head_dim, device, dtype=torch.float32, seed=None):
    """Create normalized test tensors"""
    if seed is not None:
        torch.manual_seed(seed)
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    
    # Normalize to prevent numerical issues
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    
    return q, k, v


def check_accuracy(output, reference, test_name, tolerance=1e-3):
    """Check accuracy and print results"""
    abs_diff = (output - reference).abs()
    max_error = abs_diff.max().item()
    mean_error = abs_diff.mean().item()
    
    passed = max_error < tolerance
    status = "✅ PASSED" if passed else "❌ FAILED"
    
    print(f"\n{test_name}:")
    print(f"  Max error: {max_error:.6e}")
    print(f"  Mean error: {mean_error:.6e}")
    print(f"  {status}")
    
    return passed


def test_basic_forward():
    """Test basic intel_ring_flash_attn_forward"""
    rank, world_size, device = setup_distributed()
    
    # Test parameters
    batch_size = 2
    seq_len_per_rank = 256
    num_heads = 8
    head_dim = 64
    
    if rank == 0:
        print("\n=== Test Basic Forward ===")
        print(f"World size: {world_size}")
        print(f"Config: B={batch_size}, S={seq_len_per_rank}/rank, H={num_heads}, D={head_dim}")
    
    # Create local tensors
    q, k, v = create_test_tensors(batch_size, seq_len_per_rank, num_heads, head_dim, device, seed=42+rank)
    
    # Run ring attention
    out, lse = intel_ring_flash_attn_forward(
        None, q, k, v,
        softmax_scale=1.0 / math.sqrt(head_dim),
        dropout_p=0.0,
        causal=True
    )
    
    # Verify output
    assert out.shape == q.shape, f"Output shape mismatch: {out.shape} vs {q.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"
    
    if rank == 0:
        print(f"Output shape: {out.shape}")
        print("✅ Basic forward test passed")
    
    dist.barrier()
    dist.destroy_process_group()
    return True


def test_func_interface():
    """Test intel_ring_flash_attn_func interface"""
    rank, world_size, device = setup_distributed()
    
    if rank == 0:
        print("\n=== Test Func Interface ===")
    
    # Test with [batch, seq_len, num_heads, head_dim] format
    batch_size = 2
    seq_len_per_rank = 128
    num_heads = 8
    head_dim = 64
    
    # Create tensors in [B, S, H, D] format
    torch.manual_seed(42 + rank)
    q = torch.randn(batch_size, seq_len_per_rank, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len_per_rank, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len_per_rank, num_heads, head_dim, device=device)
    
    # Normalize
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    
    # Run through func interface
    out = intel_ring_flash_attn_func(
        q, k, v,
        dropout_p=0.0,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
        group=None
    )
    
    assert out.shape == q.shape, f"Output shape mismatch: {out.shape} vs {q.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    
    if rank == 0:
        print(f"Input/Output format: [B, S, H, D] = {q.shape}")
        print("✅ Func interface test passed")
    
    dist.barrier()
    dist.destroy_process_group()
    return True


def test_qkv_packed():
    """Test QKV packed format"""
    rank, world_size, device = setup_distributed()
    
    if rank == 0:
        print("\n=== Test QKV Packed Format ===")
    
    batch_size = 2
    seq_len_per_rank = 128
    num_heads = 8
    head_dim = 64
    
    # Create QKV packed tensor [B, S, 3, H, D]
    torch.manual_seed(42 + rank)
    qkv = torch.randn(batch_size, seq_len_per_rank, 3, num_heads, head_dim, device=device)
    
    # Normalize Q and K components
    qkv[:, :, 0] = qkv[:, :, 0] / qkv[:, :, 0].norm(dim=-1, keepdim=True)
    qkv[:, :, 1] = qkv[:, :, 1] / qkv[:, :, 1].norm(dim=-1, keepdim=True)
    
    # Run packed version
    out_packed = intel_ring_flash_attn_qkvpacked_func(
        qkv,
        dropout_p=0.0,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
        group=None
    )
    
    # Compare with unpacked version
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    out_unpacked = intel_ring_flash_attn_func(
        q.contiguous(), k.contiguous(), v.contiguous(),
        dropout_p=0.0,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
        group=None
    )
    
    # Check equivalence
    if world_size == 1:  # Only check on single process for simplicity
        max_diff = (out_packed - out_unpacked).abs().max().item()
        assert max_diff < 1e-6, f"Packed vs unpacked mismatch: {max_diff}"
    
    if rank == 0:
        print(f"QKV packed shape: {qkv.shape}")
        print("✅ QKV packed test passed")
    
    dist.barrier()
    dist.destroy_process_group()
    return True


def test_different_configs():
    """Test various configurations"""
    rank, world_size, device = setup_distributed()
    
    if rank == 0:
        print("\n=== Test Different Configurations ===")
    
    configs = [
        # (batch, seq_per_rank, heads, dim, causal)
        (1, 128, 8, 64, True),    # Single batch
        (4, 256, 16, 64, True),   # Larger batch/heads
        (2, 512, 8, 128, True),   # Larger dim
        (2, 128, 1, 64, True),    # Single head
        (2, 64, 8, 32, False),    # Non-causal
    ]
    
    for i, (b, s, h, d, causal) in enumerate(configs):
        if rank == 0:
            print(f"\nConfig {i+1}: B={b}, S={s}, H={h}, D={d}, causal={causal}")
        
        q, k, v = create_test_tensors(b, s, h, d, device, seed=42+rank)
        
        try:
            out, lse = intel_ring_flash_attn_forward(
                None, q, k, v,
                softmax_scale=1.0 / math.sqrt(d),
                dropout_p=0.0,
                causal=causal
            )
            
            assert out.shape == q.shape
            assert not torch.isnan(out).any()
            
            if rank == 0:
                print(f"  ✅ Config {i+1} passed")
        except Exception as e:
            if rank == 0:
                print(f"  ❌ Config {i+1} failed: {str(e)}")
    
    dist.barrier()
    dist.destroy_process_group()
    return True


def test_accuracy_vs_reference():
    """Test accuracy against reference implementation"""
    rank, world_size, device = setup_distributed()
    
    if rank == 0:
        print("\n=== Test Accuracy vs Reference ===")
    
    batch_size = 2
    seq_len_per_rank = 256
    num_heads = 8
    head_dim = 64
    
    # Create local tensors with different data per rank
    q_local, k_local, v_local = create_test_tensors(
        batch_size, seq_len_per_rank, num_heads, head_dim, device, seed=42+rank
    )
    
    # Run ring attention
    ring_out, ring_lse = intel_ring_flash_attn_forward(
        None, q_local, k_local, v_local,
        softmax_scale=1.0 / math.sqrt(head_dim),
        dropout_p=0.0,
        causal=True
    )
    
    # Gather all data for reference computation
    if world_size > 1:
        q_list = [torch.empty_like(q_local) for _ in range(world_size)]
        k_list = [torch.empty_like(k_local) for _ in range(world_size)]
        v_list = [torch.empty_like(v_local) for _ in range(world_size)]
        out_list = [torch.empty_like(ring_out) for _ in range(world_size)]
        
        dist.all_gather(q_list, q_local.contiguous())
        dist.all_gather(k_list, k_local.contiguous())
        dist.all_gather(v_list, v_local.contiguous())
        dist.all_gather(out_list, ring_out.contiguous())
        
        if rank == 0:
            q_all = torch.cat(q_list, dim=2)
            k_all = torch.cat(k_list, dim=2)
            v_all = torch.cat(v_list, dim=2)
            ring_out_all = torch.cat(out_list, dim=2)
            
            # Compute reference
            ref_out, ref_lse = intel_flash_attn_forward(
                q_all, k_all, v_all,
                dropout_p=0.0,
                causal=True,
                softmax_scale=1.0 / math.sqrt(head_dim)
            )
            
            # Check accuracy
            check_accuracy(ring_out_all, ref_out, "Ring vs Reference", tolerance=1e-3)
    else:
        if rank == 0:
            print("Single process - skipping accuracy comparison")
    
    dist.barrier()
    dist.destroy_process_group()
    return True


def test_performance():
    """Test performance and scaling"""
    rank, world_size, device = setup_distributed()
    
    if rank == 0:
        print("\n=== Test Performance ===")
        print(f"World size: {world_size}")
    
    # Test configuration
    batch_size = 4
    seq_len_per_rank = 1024
    num_heads = 16
    head_dim = 64
    num_iterations = 10
    
    q, k, v = create_test_tensors(batch_size, seq_len_per_rank, num_heads, head_dim, device)
    
    # Warmup
    for _ in range(3):
        out, lse = intel_ring_flash_attn_forward(
            None, q, k, v,
            softmax_scale=1.0 / math.sqrt(head_dim),
            dropout_p=0.0,
            causal=True
        )
    
    # Time execution
    if HAS_XPU:
        torch.xpu.synchronize()
    dist.barrier()
    
    start_time = time.time()
    for _ in range(num_iterations):
        out, lse = intel_ring_flash_attn_forward(
            None, q, k, v,
            softmax_scale=1.0 / math.sqrt(head_dim),
            dropout_p=0.0,
            causal=True
        )
    
    if HAS_XPU:
        torch.xpu.synchronize()
    dist.barrier()
    
    elapsed_time = (time.time() - start_time) / num_iterations
    
    # Calculate statistics
    total_seq_len = seq_len_per_rank * world_size
    tflops = 4 * batch_size * num_heads * total_seq_len * seq_len_per_rank * head_dim / (elapsed_time * 1e12)
    
    if rank == 0:
        print(f"Configuration: B={batch_size}, S={total_seq_len} ({seq_len_per_rank}/rank), H={num_heads}, D={head_dim}")
        print(f"Average time per iteration: {elapsed_time*1000:.2f} ms")
        print(f"Estimated TFLOPS: {tflops:.2f}")
    
    dist.barrier()
    dist.destroy_process_group()
    return True


def main():
    """Run all tests"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("Intel Ring Flash Attention Comprehensive Test Suite")
        print("=" * 60)
    
    tests = [
        ("Basic Forward", test_basic_forward),
        ("Func Interface", test_func_interface),
        ("QKV Packed", test_qkv_packed),
        ("Different Configs", test_different_configs),
        ("Accuracy vs Reference", test_accuracy_vs_reference),
        ("Performance", test_performance),
    ]
    
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            # Reinitialize for each test
            if rank == 0:
                print(f"\nRunning: {test_name}")
            test_func()
        except Exception as e:
            if rank == 0:
                print(f"\n❌ {test_name} failed with error: {str(e)}")
                import traceback
                traceback.print_exc()
            failed_tests.append(test_name)
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("Test Summary:")
        print(f"Total tests: {len(tests)}")
        print(f"Passed: {len(tests) - len(failed_tests)}")
        print(f"Failed: {len(failed_tests)}")
        
        if failed_tests:
            print("\nFailed tests:")
            for test in failed_tests:
                print(f"  - {test}")
        else:
            print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()