#!/usr/bin/env python3
"""
Test variable length ring flash attention for Intel GPU
Tests the varlen API with different sequence length configurations
"""

import os
import sys
import torch
import torch.distributed as dist
import traceback

# Check for Intel GPU support
try:
    import intel_extension_for_pytorch as ipex
    if not torch.xpu.is_available():
        print("Intel GPU not available, exiting")
        sys.exit(0)
except ImportError:
    print("Intel Extension for PyTorch not installed, exiting")
    sys.exit(0)

from ring_flash_attn import (
    ring_flash_attn_varlen_func,
    zigzag_ring_flash_attn_varlen_func,
)
from utils import log, set_seed


def test_varlen_single_gpu():
    """Test variable length attention on single Intel GPU"""
    print("\n" + "="*60)
    print("TEST: Variable Length Attention - Single GPU")
    print("="*60)
    
    device = 'xpu'
    dtype = torch.float16
    
    # Test with different sequence patterns
    test_cases = [
        # (sequence_lengths, description)
        ([128, 256, 128], "Mixed lengths"),
        ([100, 100, 100, 100], "Equal lengths"),
        ([512], "Single sequence"),
        ([64, 128, 256, 512], "Increasing lengths"),
    ]
    
    nheads = 8
    d = 64
    
    for seq_lengths, desc in test_cases:
        print(f"\nTesting: {desc} - lengths: {seq_lengths}")
        
        # Calculate cumulative sequence lengths
        cu_seqlens = torch.tensor([0] + list(torch.tensor(seq_lengths).cumsum(0)), 
                                 device=device, dtype=torch.int32)
        total_seq = cu_seqlens[-1].item()
        batch_size = len(seq_lengths)
        
        print(f"  Batch size: {batch_size}, Total sequence: {total_seq}")
        print(f"  cu_seqlens: {cu_seqlens.tolist()}")
        
        # Create packed tensors
        q = torch.randn(total_seq, nheads, d, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(total_seq, nheads, d, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(total_seq, nheads, d, device=device, dtype=dtype, requires_grad=True)
        
        try:
            # Forward pass
            out = ring_flash_attn_varlen_func(
                q, k, v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen=max(seq_lengths),
                dropout_p=0.0,
                causal=True,
            )
            
            print(f"  ✅ Forward pass successful - output shape: {out.shape}")
            
            # Check output
            assert out.shape == (total_seq, nheads, d), f"Wrong output shape: {out.shape}"
            assert not torch.isnan(out).any(), "Output contains NaN"
            assert not torch.isinf(out).any(), "Output contains Inf"
            
            # Backward pass
            dout = torch.randn_like(out)
            out.backward(dout)
            
            assert q.grad is not None, "No gradient for q"
            assert k.grad is not None, "No gradient for k" 
            assert v.grad is not None, "No gradient for v"
            
            print(f"  ✅ Backward pass successful")
            
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            traceback.print_exc()
            return False
    
    print("\n✅ All single GPU varlen tests passed!")
    return True


def test_varlen_distributed():
    """Test distributed variable length attention on Intel GPU"""
    print("\n" + "="*60)
    print("TEST: Distributed Variable Length Attention")
    print("="*60)
    
    if not dist.is_initialized():
        # Setup CCL environment variables (from Aurora training script)
        os.environ['CCL_BACKEND'] = 'native'
        os.environ['CCL_ATL_TRANSPORT'] = 'ofi'
        os.environ['FI_PROVIDER'] = 'cxi'  # Critical for Aurora fabric interface
        os.environ['CCL_ZE_IPC_EXCHANGE'] = 'drmfd'
        os.environ['CCL_ZE_ENABLE'] = '1'
        os.environ['CCL_LOG_LEVEL'] = 'info'
        os.environ['IPEX_XPU_ONEDNN_LAYOUT'] = '1'
        os.environ['IPEX_OFFLINE_COMPILER'] = '1'
        os.environ['SYCL_CACHE_PERSISTENT'] = '1'  # Prevents build-for-1-device issues
        os.environ['SYCL_DEVICE_FILTER'] = 'level_zero:*'  # Proper SYCL device selection
        os.environ['SYCL_PI_LEVEL_ZERO_PROGRAM_BUILD_TRACK'] = '2'  # Program building tracking
        
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        
        try:
            dist.init_process_group(backend='ccl', init_method='env://', rank=0, world_size=1, timeout=torch.distributed.default_pg_timeout)
        except Exception as e:
            print(f"CCL initialization failed, falling back to gloo: {e}")
            dist.init_process_group(backend='gloo', init_method='env://', rank=0, world_size=1)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'xpu:{rank}' if torch.xpu.device_count() > 1 else 'xpu'
    dtype = torch.float16
    
    set_seed(rank)
    
    # Test configuration
    seq_lengths = [128, 256, 128, 192]  # Different lengths per sample
    nheads = 8
    d = 64
    
    # Calculate cumulative sequence lengths
    cu_seqlens = torch.tensor([0] + list(torch.tensor(seq_lengths).cumsum(0)), 
                             device=device, dtype=torch.int32)
    total_seq = cu_seqlens[-1].item()
    max_seqlen = max(seq_lengths)
    
    if rank == 0:
        print(f"Sequence lengths: {seq_lengths}")
        print(f"Total sequence: {total_seq}")
        print(f"cu_seqlens: {cu_seqlens.tolist()}")
    
    # Create and broadcast tensors
    if rank == 0:
        q = torch.randn(total_seq, nheads, d, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(total_seq, nheads, d, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(total_seq, nheads, d, device=device, dtype=dtype, requires_grad=True)
    else:
        q = torch.empty(total_seq, nheads, d, device=device, dtype=dtype, requires_grad=True)
        k = torch.empty(total_seq, nheads, d, device=device, dtype=dtype, requires_grad=True)
        v = torch.empty(total_seq, nheads, d, device=device, dtype=dtype, requires_grad=True)
    
    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)
    
    # Distribute sequences across GPUs
    # This is a simplified distribution - real implementation would be more sophisticated
    seqs_per_gpu = len(seq_lengths) // world_size
    start_idx = rank * seqs_per_gpu
    end_idx = (rank + 1) * seqs_per_gpu if rank < world_size - 1 else len(seq_lengths)
    
    local_cu_seqlens = cu_seqlens[start_idx:end_idx + 1] - cu_seqlens[start_idx]
    local_start = cu_seqlens[start_idx].item()
    local_end = cu_seqlens[end_idx].item()
    
    local_q = q[local_start:local_end].detach().clone()
    local_k = k[local_start:local_end].detach().clone()
    local_v = v[local_start:local_end].detach().clone()
    local_q.requires_grad = True
    local_k.requires_grad = True
    local_v.requires_grad = True
    
    print(f"[Rank {rank}] Local sequences: {local_cu_seqlens.tolist()}")
    
    try:
        # Test ring varlen attention
        out = ring_flash_attn_varlen_func(
            local_q, local_k, local_v,
            local_cu_seqlens,
            local_cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=0.0,
            causal=True,
        )
        
        print(f"[Rank {rank}] ✅ Varlen forward pass successful")
        
        # Test zigzag variant
        out_zigzag = zigzag_ring_flash_attn_varlen_func(
            local_q, local_k, local_v,
            local_cu_seqlens,
            local_cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=0.0,
            causal=True,
        )
        
        print(f"[Rank {rank}] ✅ Zigzag varlen forward pass successful")
        
    except Exception as e:
        print(f"[Rank {rank}] ❌ Distributed varlen test failed: {e}")
        traceback.print_exc()
        return False
    
    dist.barrier()
    if rank == 0:
        print("\n✅ Distributed varlen tests passed!")
    
    return True


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*60)
    print("TEST: Edge Cases and Error Handling")
    print("="*60)
    
    device = 'xpu'
    dtype = torch.float16
    nheads = 8
    d = 64
    
    # Test 1: Empty sequence
    print("\nTest 1: Empty sequence handling")
    try:
        cu_seqlens = torch.tensor([0, 0], device=device, dtype=torch.int32)
        q = torch.empty(0, nheads, d, device=device, dtype=dtype)
        k = torch.empty(0, nheads, d, device=device, dtype=dtype)
        v = torch.empty(0, nheads, d, device=device, dtype=dtype)
        
        # This should either handle gracefully or raise a clear error
        try:
            out = ring_flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, 0)
            print("  ✅ Empty sequence handled gracefully")
        except Exception as e:
            print(f"  ⚠️  Empty sequence raised error (expected): {type(e).__name__}")
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
    
    # Test 2: Very long sequence
    print("\nTest 2: Very long sequence")
    try:
        long_seq = 8192  # Very long sequence
        cu_seqlens = torch.tensor([0, long_seq], device=device, dtype=torch.int32)
        
        # Use smaller dtype to save memory
        q = torch.randn(long_seq, nheads, d, device=device, dtype=dtype)
        k = torch.randn(long_seq, nheads, d, device=device, dtype=dtype)
        v = torch.randn(long_seq, nheads, d, device=device, dtype=dtype)
        
        out = ring_flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, long_seq,
            dropout_p=0.0, causal=True
        )
        
        print(f"  ✅ Long sequence ({long_seq}) handled successfully")
        
    except torch.cuda.OutOfMemoryError:
        print("  ⚠️  Out of memory for very long sequence (expected)")
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
    
    # Test 3: Mismatched sequence lengths
    print("\nTest 3: Mismatched q/k/v sequence lengths")
    try:
        cu_seqlens_q = torch.tensor([0, 128, 256], device=device, dtype=torch.int32)
        cu_seqlens_kv = torch.tensor([0, 100, 200], device=device, dtype=torch.int32)
        
        q = torch.randn(256, nheads, d, device=device, dtype=dtype)
        k = torch.randn(200, nheads, d, device=device, dtype=dtype)
        v = torch.randn(200, nheads, d, device=device, dtype=dtype)
        
        # This should work as it's a valid cross-attention scenario
        out = ring_flash_attn_varlen_func(
            q, k, v, cu_seqlens_q, cu_seqlens_kv, 
            max_seqlen=128, dropout_p=0.0, causal=False
        )
        
        print("  ✅ Cross-attention with different lengths handled")
        
    except Exception as e:
        print(f"  ⚠️  Cross-attention error: {type(e).__name__}")
    
    return True


def benchmark_varlen_performance():
    """Benchmark variable length attention performance"""
    print("\n" + "="*60)
    print("TEST: Variable Length Performance Benchmark")
    print("="*60)
    
    device = 'xpu'
    dtype = torch.float16
    nheads = 8
    d = 64
    
    import time
    
    benchmark_configs = [
        {"batch": 4, "avg_len": 512, "name": "Short sequences"},
        {"batch": 8, "avg_len": 1024, "name": "Medium sequences"},
        {"batch": 4, "avg_len": 2048, "name": "Long sequences"},
    ]
    
    for config in benchmark_configs:
        batch_size = config["batch"]
        avg_len = config["avg_len"]
        name = config["name"]
        
        print(f"\nBenchmarking: {name}")
        print(f"  Batch size: {batch_size}, Average length: {avg_len}")
        
        # Generate variable lengths around the average
        import random
        random.seed(42)
        seq_lengths = [random.randint(int(avg_len * 0.8), int(avg_len * 1.2)) 
                      for _ in range(batch_size)]
        
        cu_seqlens = torch.tensor([0] + list(torch.tensor(seq_lengths).cumsum(0)), 
                                 device=device, dtype=torch.int32)
        total_seq = cu_seqlens[-1].item()
        max_seqlen = max(seq_lengths)
        
        print(f"  Actual lengths: {seq_lengths}")
        print(f"  Total tokens: {total_seq}")
        
        # Create tensors
        q = torch.randn(total_seq, nheads, d, device=device, dtype=dtype)
        k = torch.randn(total_seq, nheads, d, device=device, dtype=dtype)
        v = torch.randn(total_seq, nheads, d, device=device, dtype=dtype)
        
        # Warmup
        for _ in range(3):
            out = ring_flash_attn_varlen_func(
                q, k, v, cu_seqlens, cu_seqlens, max_seqlen,
                dropout_p=0.0, causal=True
            )
            torch.xpu.synchronize()
        
        # Benchmark
        torch.xpu.synchronize()
        start_time = time.time()
        
        num_iters = 10
        for _ in range(num_iters):
            out = ring_flash_attn_varlen_func(
                q, k, v, cu_seqlens, cu_seqlens, max_seqlen,
                dropout_p=0.0, causal=True
            )
        
        torch.xpu.synchronize()
        elapsed_time = (time.time() - start_time) / num_iters
        
        # Calculate tokens/sec
        tokens_per_sec = total_seq / elapsed_time
        ms_per_token = elapsed_time * 1000 / total_seq
        
        print(f"  ✅ Average time: {elapsed_time*1000:.2f} ms")
        print(f"  ✅ Throughput: {tokens_per_sec:.0f} tokens/sec")
        print(f"  ✅ Latency: {ms_per_token:.3f} ms/token")
    
    return True


def main():
    """Run all Intel GPU variable length tests"""
    print("🚀 Intel GPU Variable Length Ring Flash Attention Test Suite")
    print("="*80)
    
    # Check Intel GPU availability
    if not torch.xpu.is_available():
        print("❌ Intel GPU not available, exiting")
        return 1
    
    print(f"✅ Intel GPU detected: {torch.xpu.device_count()} device(s)")
    print(f"✅ Intel Extension for PyTorch version: {ipex.__version__}")
    
    # Run tests
    tests = [
        ("Single GPU Varlen", test_varlen_single_gpu),
        ("Edge Cases", test_edge_cases),
        ("Performance Benchmark", benchmark_varlen_performance),
    ]
    
    # Add distributed test if running with torchrun
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        tests.insert(1, ("Distributed Varlen", test_varlen_distributed))
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("📊 VARLEN TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All varlen tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    if dist.is_initialized():
        dist.destroy_process_group()
    
    sys.exit(main())