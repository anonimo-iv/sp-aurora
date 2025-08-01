#!/usr/bin/env python3
"""
Test suite for Intel Ulysses (Sequence Parallel) Flash Attention
MPI-compatible test that works with both torchrun and mpiexec

Usage:
    # With torchrun
    torchrun --nproc_per_node=2 test_intel_ulysses_attn.py
    
    # With mpiexec
    mpiexec -n 2 python test_intel_ulysses_attn.py
    
    # With Intel MPI (Intel GPU)
    mpiexec -n 2 -genv CCL_BACKEND=native -genv CCL_ATL_TRANSPORT=ofi \
        python test_intel_ulysses_attn.py
"""

import torch
import torch.distributed as dist
import os
import sys
import traceback
import socket
import datetime
import atexit
from mpi4py import MPI

# Debug: Print early MPI info
mpi_rank_early = MPI.COMM_WORLD.Get_rank()
print(f"[Early Rank {mpi_rank_early}] Starting test_intel_ulysses_attn.py", flush=True)

# Register cleanup handler
def cleanup_mpi():
    """Cleanup MPI on exit"""
    if MPI.Is_initialized() and not MPI.Is_finalized():
        rank = MPI.COMM_WORLD.Get_rank()
        print(f"[Rank {rank}] Finalizing MPI in atexit handler", flush=True)
        MPI.Finalize()

atexit.register(cleanup_mpi)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for Intel GPU support and OneCCL
try:
    print(f"[Early Rank {mpi_rank_early}] Importing IPEX...", flush=True)
    import intel_extension_for_pytorch as ipex
    print(f"[Early Rank {mpi_rank_early}] Importing oneCCL bindings...", flush=True)
    import oneccl_bindings_for_pytorch
    INTEL_GPU_AVAILABLE = torch.xpu.is_available() if hasattr(torch, 'xpu') else False
    print(f"[Early Rank {mpi_rank_early}] Intel GPU available: {INTEL_GPU_AVAILABLE}", flush=True)
except ImportError as e:
    print(f"[Early Rank {mpi_rank_early}] Warning: Intel Extension or OneCCL not available: {e}")
    INTEL_GPU_AVAILABLE = False

# Defer ring_flash_attn imports until after distributed initialization
print(f"[Early Rank {mpi_rank_early}] Deferring ring_flash_attn imports", flush=True)


def setup_distributed():
    """Initialize distributed environment with MPI coordination"""
    # Get MPI info
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    
    print(f"[Rank {mpi_rank}] MPI initialized - size: {mpi_size}", flush=True)
    
    # Setup environment variables
    os.environ['RANK'] = str(mpi_rank)
    os.environ['WORLD_SIZE'] = str(mpi_size)
    print(f"[Rank {mpi_rank}] Environment variables set", flush=True)
    
    # Broadcast master address and port from rank 0
    if mpi_size > 1:
        if mpi_rank == 0:
            master_addr = socket.gethostname()
            master_port = 12355
        else:
            master_addr = None
            master_port = None
        
        master_addr = mpi_comm.bcast(master_addr, root=0)
        master_port = mpi_comm.bcast(master_port, root=0)
        
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
    
    # Synchronize before initializing process group
    if mpi_size > 1:
        print(f"[Rank {mpi_rank}] Entering MPI barrier before dist init", flush=True)
        mpi_comm.Barrier()
        print(f"[Rank {mpi_rank}] Passed MPI barrier", flush=True)
    
    # Set device
    if INTEL_GPU_AVAILABLE:
        device = torch.device(f'xpu:{mpi_rank % torch.xpu.device_count()}')
        torch.xpu.set_device(device)
        # Use CCL backend if oneccl_bindings was imported successfully
        try:
            import oneccl_bindings_for_pytorch
            backend = 'ccl'
            print(f"[Rank {mpi_rank}] Using CCL backend for Intel GPU", flush=True)
        except ImportError:
            backend = 'gloo'
            print(f"[Rank {mpi_rank}] Warning: oneCCL bindings not available, using gloo", flush=True)
    else:
        device = torch.device('cpu')
        backend = 'gloo'
    
    # Initialize process group
    if mpi_size > 1:
        print(f"[Rank {mpi_rank}] Initializing process group with backend: {backend}", flush=True)
        try:
            dist.init_process_group(
                backend=backend,
                init_method='env://',
                world_size=mpi_size,
                rank=mpi_rank,
                timeout=datetime.timedelta(seconds=300)
            )
            print(f"[Rank {mpi_rank}] Process group initialized successfully", flush=True)
        except Exception as e:
            print(f"[Rank {mpi_rank}] Failed to initialize process group: {e}", flush=True)
            raise
    
    return {
        'rank': mpi_rank,
        'world_size': mpi_size,
        'device': device,
        'backend': backend
    }


def cleanup_distributed():
    """Clean up distributed environment"""
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    print(f"[Rank {mpi_rank}] Cleaning up distributed environment...", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()
        print(f"[Rank {mpi_rank}] Process group destroyed", flush=True)


class TestIntelUlyssesAttention:
    """Test cases for Intel Ulysses attention implementation"""
    
    def test_single_gpu_attention(self, batch_size, seq_len, num_heads, head_dim):
        """Test Ulysses attention on single GPU (should behave like regular attention)"""
        device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
        
        # Create random tensors
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        
        # Compute Ulysses attention (single GPU case)
        out_ulysses = ulysses_flash_attn_func(q, k, v, causal=True)
        
        # Compute reference with PyTorch SDPA
        out_ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=1.0/torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))
        )
        
        # Check shapes match
        assert out_ulysses.shape == out_ref.shape
        
        # Check values are close (allow some tolerance for numerical differences)
        torch.testing.assert_close(out_ulysses, out_ref, rtol=1e-2, atol=1e-3)
    
    def test_all_to_all_4d_forward(self):
        """Test the all-to-all 4D operation"""
        if not dist.is_initialized():
            print("Distributed not initialized, skipping test")
            return
            
        device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Test scatter_idx=2, gather_idx=1 (forward direction)
        batch_size = 2
        seq_len_per_gpu = 512
        num_heads = 16
        head_dim = 64
        
        # Create input tensor
        input_tensor = torch.randn(
            batch_size, seq_len_per_gpu, num_heads, head_dim, 
            device=device, dtype=torch.float16
        )
        
        # Apply all-to-all
        output = intel_all_to_all_4d(input_tensor, scatter_idx=2, gather_idx=1)
        
        # Check output shape
        expected_shape = (batch_size, seq_len_per_gpu * world_size, num_heads // world_size, head_dim)
        assert output.shape == expected_shape
    
    def test_all_to_all_4d_backward(self):
        """Test the all-to-all 4D operation in backward direction"""
        if not dist.is_initialized():
            print("Distributed not initialized, skipping test")
            return
            
        device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Test scatter_idx=1, gather_idx=2 (backward direction)
        batch_size = 2
        seq_len = 512 * world_size
        num_heads_per_gpu = 16 // world_size
        head_dim = 64
        
        # Create input tensor
        input_tensor = torch.randn(
            batch_size, seq_len, num_heads_per_gpu, head_dim,
            device=device, dtype=torch.float16
        )
        
        # Apply all-to-all
        output = intel_all_to_all_4d(input_tensor, scatter_idx=1, gather_idx=2)
        
        # Check output shape
        expected_shape = (batch_size, seq_len // world_size, num_heads_per_gpu * world_size, head_dim)
        assert output.shape == expected_shape
    
    def test_autograd_function(self):
        """Test the autograd function for gradient computation"""
        device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
        
        batch_size = 2
        seq_len = 512
        num_heads = 8
        head_dim = 64
        
        # Create tensors with requires_grad
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, 
                       dtype=torch.float32, requires_grad=True)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device,
                       dtype=torch.float32, requires_grad=True)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device,
                       dtype=torch.float32, requires_grad=True)
        
        # Forward pass
        output = ulysses_flash_attn_func(q, k, v, causal=True)
        
        # Create random gradient
        grad_output = torch.randn_like(output)
        
        # Backward pass
        output.backward(grad_output)
        
        # Check gradients exist
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        
        # Check gradient shapes
        assert q.grad.shape == q.shape
        assert k.grad.shape == k.shape
        assert v.grad.shape == v.shape
    
    def test_qkvpacked_format(self):
        """Test QKV packed format"""
        device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
        
        batch_size = 2
        seq_len = 512
        num_heads = 8
        head_dim = 64
        
        # Create packed QKV tensor
        qkv = torch.randn(batch_size, seq_len, 3, num_heads, head_dim, 
                         device=device, dtype=torch.float16)
        
        # Test packed function
        out_packed = ulysses_flash_attn_qkvpacked_func(qkv, causal=True)
        
        # Test with unpacked tensors
        q, k, v = qkv.unbind(dim=2)
        out_unpacked = ulysses_flash_attn_func(q, k, v, causal=True)
        
        # Results should be identical
        torch.testing.assert_close(out_packed, out_unpacked)
    
    def test_kvpacked_format(self):
        """Test KV packed format"""
        device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
        
        batch_size = 2
        seq_len = 512
        num_heads = 8
        head_dim = 64
        
        # Create Q and packed KV tensors
        q = torch.randn(batch_size, seq_len, num_heads, head_dim,
                       device=device, dtype=torch.float16)
        kv = torch.randn(batch_size, seq_len, 2, num_heads, head_dim,
                        device=device, dtype=torch.float16)
        
        # Test packed function
        out_packed = ulysses_flash_attn_kvpacked_func(q, kv, causal=True)
        
        # Test with unpacked tensors
        k, v = kv.unbind(dim=2)
        out_unpacked = ulysses_flash_attn_func(q, k, v, causal=True)
        
        # Results should be identical
        torch.testing.assert_close(out_packed, out_unpacked)
    
    def test_attention_options(self, causal, dropout_p):
        """Test various attention options"""
        device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
        
        batch_size = 1
        seq_len = 256
        num_heads = 8
        head_dim = 64
        
        # Create tensors
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
        
        # Test with different options
        output = ulysses_flash_attn_func(
            q, k, v,
            causal=causal,
            dropout_p=dropout_p,
            deterministic=True
        )
        
        # Check output shape
        assert output.shape == q.shape


def test_distributed_ulysses(rank, world_size, device):
    """Test Ulysses attention in distributed setting"""
    print(f"\n{'='*60}")
    print("TEST: Distributed Ulysses Attention")
    print(f"{'='*60}")
    
    try:
        if world_size == 1:
            print(f"[Rank {rank}] Single process - skipping distributed test")
            return True
        
        print(f"[Rank {rank}] Testing Ulysses attention with {world_size} GPUs")
        
        # Set per-GPU dimensions
        batch_size = 2
        seq_len_per_gpu = 512
        num_heads = 16
        head_dim = 64
        dtype = torch.float16 if device.type in ['cuda', 'xpu'] else torch.float32
        
        # Create tensors on each GPU
        q = torch.randn(batch_size, seq_len_per_gpu, num_heads, head_dim, 
                       device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len_per_gpu, num_heads, head_dim,
                       device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len_per_gpu, num_heads, head_dim,
                       device=device, dtype=dtype)
        
        print(f"[Rank {rank}] Running Ulysses attention...")
        
        # Run Ulysses attention
        output = ulysses_flash_attn_func(q, k, v, causal=True)
        
        # Check output shape (should match input shape)
        assert output.shape == q.shape
        print(f"[Rank {rank}] ✓ Output shape correct: {output.shape}")
        
        # Test gradient flow with float32
        if dtype == torch.float32:
            output.sum().backward()
            print(f"[Rank {rank}] ✓ Gradient flow test passed")
        
        # Test all-to-all operations
        test_all_to_all = TestIntelUlyssesAttention()
        test_all_to_all.test_all_to_all_4d_forward()
        test_all_to_all.test_all_to_all_4d_backward()
        print(f"[Rank {rank}] ✓ All-to-all operations test passed")
        
        return True
        
    except Exception as e:
        print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] ❌ Distributed test failed: {e}")
        traceback.print_exc()
        return False


def run_single_gpu_tests():
    """Run tests that don't require distributed setup"""
    test = TestIntelUlyssesAttention()
    
    print("\nTesting single GPU attention...")
    test.test_single_gpu_attention(batch_size=2, seq_len=512, num_heads=8, head_dim=64)
    print("✓ Single GPU test passed")
    
    print("\nTesting QKV packed format...")
    test.test_qkvpacked_format()
    print("✓ QKV packed test passed")
    
    print("\nTesting KV packed format...")
    test.test_kvpacked_format()
    print("✓ KV packed test passed")
    
    print("\nTesting autograd...")
    test.test_autograd_function()
    print("✓ Autograd test passed")
    
    print("\nTesting attention options...")
    test.test_attention_options(causal=True, dropout_p=0.0)
    test.test_attention_options(causal=False, dropout_p=0.1)
    print("✓ Attention options test passed")


def main():
    """Main test function with MPI support"""
    # Get initial MPI rank for debugging
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    print(f"[Rank {mpi_rank}] Starting Intel Ulysses Attention Test Suite", flush=True)
    
    print(f"\n{'='*60}")
    print("Intel Ulysses Attention Test Suite")
    print(f"{'='*60}")
    
    exit_code = 0
    try:
        # Setup distributed environment
        setup_info = setup_distributed()
        
        # Import ring_flash_attn modules after distributed initialization
        print(f"[Rank {mpi_rank}] Importing ring_flash_attn modules...", flush=True)
        global ulysses_flash_attn_func, ulysses_flash_attn_qkvpacked_func, ulysses_flash_attn_kvpacked_func
        global IntelSeqAllToAll4D, intel_all_to_all_4d
        from ring_flash_attn import ulysses_flash_attn_func, ulysses_flash_attn_qkvpacked_func, ulysses_flash_attn_kvpacked_func
        from ring_flash_attn.intel_ulysses_attn import IntelSeqAllToAll4D, intel_all_to_all_4d
        print(f"[Rank {mpi_rank}] Imports successful", flush=True)
        
        rank = setup_info['rank']
        world_size = setup_info['world_size']
        device = setup_info['device']
        backend = setup_info['backend']
        
        print(f"[Rank {rank}] World size: {world_size}")
        print(f"[Rank {rank}] Device: {device}")
        print(f"[Rank {rank}] Backend: {backend}")
        
        # Run single GPU tests on rank 0 only
        test_success = True
        if rank == 0:
            try:
                run_single_gpu_tests()
            except Exception as e:
                print(f"[Rank 0] Single GPU tests failed: {e}")
                test_success = False
        
        # Broadcast test result to all ranks
        if world_size > 1:
            test_success = mpi_comm.bcast(test_success, root=0)
            
        # Exit early if single GPU tests failed
        if not test_success:
            exit_code = 1
        else:
            # Run distributed tests if multiple processes
            if world_size > 1:
                success = test_distributed_ulysses(rank, world_size, device)
                if not success:
                    exit_code = 1
            
            if rank == 0 and exit_code == 0:
                print(f"\n{'='*60}")
                print("✅ All tests passed successfully!")
                print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        traceback.print_exc()
        exit_code = 1
    finally:
        cleanup_distributed()
        print(f"[Rank {mpi_rank}] Cleanup completed, returning from main()", flush=True)
        
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    # Ensure MPI is properly finalized
    if MPI.Is_initialized() and not MPI.Is_finalized():
        print(f"[Rank {MPI.COMM_WORLD.Get_rank()}] Finalizing MPI before exit", flush=True)
        MPI.Finalize()
    sys.exit(exit_code)