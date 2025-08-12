#!/usr/bin/env python3
"""
MPI-compatible test for Ring Flash Attention
This test works with both torchrun and mpiexec

Usage:
    # With torchrun (existing)
    torchrun --nproc_per_node=2 test_mpi_sp_aurora.py
    
    # With mpiexec (new)
    mpiexec -n 2 python test_mpi_sp_aurora.py
    
    # With Intel MPI (Intel GPU)
    mpiexec -n 2 -genv CCL_BACKEND=native -genv CCL_ATL_TRANSPORT=ofi \
        python test_mpi_sp_aurora.py
"""

import os
import sys
import torch
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sp_aurora.mpi_utils import setup_mpi_distributed, cleanup_distributed


def test_basic_mpi_setup():
    """Test basic MPI setup and distributed initialization"""
    print(f"\n{'='*60}")
    print("TEST: Basic MPI Setup and Distributed Initialization")
    print(f"{'='*60}")
    
    try:
        # Setup distributed environment
        setup_info = setup_mpi_distributed()
        
        rank = setup_info['rank']
        world_size = setup_info['world_size']
        device = setup_info['device']
        launcher = setup_info['launcher']
        backend = setup_info['backend']
        
        print(f"[Rank {rank}] Setup successful!")
        print(f"[Rank {rank}] Launcher: {launcher}")
        print(f"[Rank {rank}] World size: {world_size}")
        print(f"[Rank {rank}] Device: {device}")
        print(f"[Rank {rank}] Backend: {backend}")
        
        # Test basic tensor operations on assigned device
        x = torch.randn(4, 4, device=device)
        y = torch.randn(4, 4, device=device)
        z = torch.matmul(x, y)
        
        print(f"[Rank {rank}] ✅ Basic tensor operations successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic MPI setup failed: {e}")
        traceback.print_exc()
        return False


def test_ring_attention_with_mpi():
    """Test Ring Flash Attention with MPI compatibility"""
    print(f"\n{'='*60}")
    print("TEST: Ring Flash Attention with MPI")
    print(f"{'='*60}")
    
    try:
        # Setup distributed
        setup_info = setup_mpi_distributed()
        rank = setup_info['rank']
        world_size = setup_info['world_size']
        device = setup_info['device']
        
        if world_size == 1:
            print(f"[Rank {rank}] Single process - skipping ring attention test")
            return True
        
        # Import ring attention after distributed setup
        from sp_aurora import ring_flash_attn_func
        
        # Test parameters
        batch_size = 1
        seqlen_per_rank = 256
        nheads = 8
        d = 64
        dtype = torch.float16 if device.type in ['cuda', 'xpu'] else torch.float32
        
        print(f"[Rank {rank}] Creating test tensors...")
        
        # Create test tensors
        q = torch.randn(batch_size, seqlen_per_rank, nheads, d, 
                       device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(batch_size, seqlen_per_rank, nheads, d, 
                       device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(batch_size, seqlen_per_rank, nheads, d, 
                       device=device, dtype=dtype, requires_grad=True)
        
        print(f"[Rank {rank}] Running ring attention forward pass...")
        
        # Test ring attention
        import torch.distributed as dist
        import signal
        import time
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Ring attention timed out at rank {rank}")
        
        # Set timeout to detect deadlocks
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        start_time = time.time()
        
        # Run ring attention
        output = ring_flash_attn_func(q, k, v, causal=True)
        
        elapsed = time.time() - start_time
        signal.alarm(0)  # Cancel timeout
        
        print(f"[Rank {rank}] ✅ Ring attention forward pass successful ({elapsed:.2f}s)")
        print(f"[Rank {rank}] Output shape: {output.shape}")
        
        # Test backward pass
        dout = torch.randn_like(output)
        output.backward(dout)
        
        print(f"[Rank {rank}] ✅ Ring attention backward pass successful")
        
        # Synchronize all processes
        if dist.is_initialized():
            dist.barrier()
        
        if rank == 0:
            print("\n✅ All ranks completed ring attention test successfully!")
        
        return True
        
    except TimeoutError as e:
        signal.alarm(0)
        print(f"[Rank {rank}] ❌ {e}")
        return False
    except Exception as e:
        if 'signal' in locals():
            signal.alarm(0)
        print(f"[Rank {rank}] ❌ Ring attention test failed: {e}")
        traceback.print_exc()
        return False


def test_intel_specific_mpi():
    """Test Intel GPU specific functionality with MPI"""
    print(f"\n{'='*60}")
    print("TEST: Intel GPU Ring Flash Attention with MPI")
    print(f"{'='*60}")
    
    # Check if Intel GPU is available
    if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
        print("Intel GPU not available, skipping Intel-specific test")
        return True
    
    try:
        import intel_extension_for_pytorch as ipex
        from sp_aurora.intel_ring_flash_attn import intel_ring_flash_attn_func
        
        # Setup distributed
        setup_info = setup_mpi_distributed(backend='ccl')
        rank = setup_info['rank']
        world_size = setup_info['world_size']
        device = setup_info['device']
        
        if world_size == 1:
            print(f"[Rank {rank}] Single process - testing single GPU Intel implementation")
            
            # Test single GPU Intel flash attention
            from sp_aurora.intel_flash_attn import intel_flash_attn_forward
            
            batch_size = 1
            seqlen = 256
            nheads = 8
            d = 64
            
            q = torch.randn(batch_size, nheads, seqlen, d, device=device, dtype=torch.float16)
            k = torch.randn(batch_size, nheads, seqlen, d, device=device, dtype=torch.float16)
            v = torch.randn(batch_size, nheads, seqlen, d, device=device, dtype=torch.float16)
            
            out, lse = intel_flash_attn_forward(q, k, v, causal=True)
            print(f"[Rank {rank}] ✅ Intel single GPU flash attention successful")
            return True
        
        print(f"[Rank {rank}] Testing Intel ring attention with {world_size} processes...")
        
        # Test parameters for multi-GPU
        batch_size = 1
        seqlen_per_rank = 256
        nheads = 8
        d = 64
        
        # Create test tensors
        q = torch.randn(batch_size, seqlen_per_rank, nheads, d, 
                       device=device, dtype=torch.float16, requires_grad=True)
        k = torch.randn(batch_size, seqlen_per_rank, nheads, d, 
                       device=device, dtype=torch.float16, requires_grad=True)
        v = torch.randn(batch_size, seqlen_per_rank, nheads, d, 
                       device=device, dtype=torch.float16, requires_grad=True)
        
        print(f"[Rank {rank}] Running Intel ring attention...")
        
        # Test with timeout
        import signal
        import time
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Intel ring attention timed out at rank {rank}")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        start_time = time.time()
        
        output, lse, _ = intel_ring_flash_attn_func(
            q, k, v, dropout_p=0.0, causal=True, return_attn_probs=False
        )
        
        elapsed = time.time() - start_time
        signal.alarm(0)
        
        print(f"[Rank {rank}] ✅ Intel ring attention successful ({elapsed:.2f}s)")
        
        # Test backward
        dout = torch.randn_like(output)
        output.backward(dout)
        
        print(f"[Rank {rank}] ✅ Intel ring attention backward successful")
        
        # Synchronize
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
        
        if rank == 0:
            print("\n✅ Intel GPU ring attention with MPI successful!")
        
        return True
        
    except ImportError as e:
        print(f"Intel GPU support not available: {e}")
        return True  # Not a failure, just not available
    except TimeoutError as e:
        signal.alarm(0)
        print(f"[Rank {rank}] ❌ {e}")
        return False
    except Exception as e:
        if 'signal' in locals():
            signal.alarm(0)
        print(f"[Rank {rank}] ❌ Intel MPI test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run MPI compatibility tests"""
    print("🚀 Ring Flash Attention MPI Compatibility Test Suite")
    print("="*80)
    
    # Detect environment
    from sp_aurora.mpi_utils import detect_mpi_environment, setup_distributed_environment
    
    env_info = setup_distributed_environment()
    print(f"Detected launcher: {env_info['launcher']}")
    print(f"Process info: rank={env_info['rank']}, world_size={env_info['world_size']}")
    
    # Run tests
    tests = [
        ("Basic MPI Setup", test_basic_mpi_setup),
        ("Ring Attention with MPI", test_ring_attention_with_mpi),
        ("Intel GPU with MPI", test_intel_specific_mpi),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}")
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary (only from rank 0 to avoid spam)
    if env_info['rank'] == 0:
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 All MPI compatibility tests passed!")
            return_code = 0
        else:
            print(f"\n⚠️ {total - passed} test(s) failed.")
            return_code = 1
    else:
        return_code = 0
    
    # Cleanup
    cleanup_distributed()
    
    return return_code


if __name__ == "__main__":
    sys.exit(main())