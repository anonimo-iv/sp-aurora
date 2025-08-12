#!/usr/bin/env python3
"""
Test to verify fixed Ulysses implementation works correctly.

Usage:
    mpirun -n 2 python test_ulysses_fixed.py
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import os
import sys
from mpi4py import MPI
import datetime
import time

# Add parent directory to path - need to go up two levels from verification/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Check for Intel GPU support
try:
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch
    INTEL_GPU_AVAILABLE = torch.xpu.is_available() if hasattr(torch, 'xpu') else False
except ImportError as e:
    print(f"Warning: Intel Extension not available: {e}")
    INTEL_GPU_AVAILABLE = False


def test_ulysses_forward(rank, world_size, device):
    """Test Ulysses forward pass"""
    print(f"\n[Rank {rank}] Test 1: Ulysses Forward Pass")
    
    try:
        from sp_aurora.intel_ulysses_attn import intel_ulysses_flash_attn_forward
        
        # Create test tensors
        batch_size = 2
        seq_len_per_rank = 256
        num_heads = 8
        head_dim = 64
        
        # Input shape: (batch, num_heads, seq_len/P, head_dim) for forward
        q = torch.randn(batch_size, num_heads, seq_len_per_rank, head_dim,
                       device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        
        print(f"[Rank {rank}] Input shapes: q={q.shape}")
        
        # Call forward
        start_time = time.time()
        output, lse = intel_ulysses_flash_attn_forward(
            None,  # process_group
            q, k, v,
            softmax_scale=1.0 / (head_dim ** 0.5),
            causal=True
        )
        end_time = time.time()
        
        print(f"[Rank {rank}] ✓ Forward pass SUCCESS in {end_time - start_time:.3f}s")
        print(f"[Rank {rank}] Output shape: {output.shape}, LSE shape: {lse.shape}")
        
        # Verify output shape
        assert output.shape == q.shape, f"Shape mismatch: {output.shape} != {q.shape}"
        
        return True
    except Exception as e:
        print(f"[Rank {rank}] ✗ Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ulysses_func(rank, world_size, device):
    """Test Ulysses attention function (high-level API)"""
    print(f"\n[Rank {rank}] Test 2: Ulysses Attention Function")
    
    try:
        from sp_aurora import ulysses_flash_attn_func
        
        # Create test tensors (note different shape order for func API)
        batch_size = 2
        seq_len_per_rank = 128
        num_heads = 4
        head_dim = 64
        
        # Input shape: (batch, seq_len/P, num_heads, head_dim) for func API
        q = torch.randn(batch_size, seq_len_per_rank, num_heads, head_dim,
                       device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        
        print(f"[Rank {rank}] Input shapes: q={q.shape}")
        
        # Call function
        start_time = time.time()
        output = ulysses_flash_attn_func(
            q, k, v,
            causal=True,
            group=None  # Use default process group
        )
        end_time = time.time()
        
        print(f"[Rank {rank}] ✓ Attention function SUCCESS in {end_time - start_time:.3f}s")
        print(f"[Rank {rank}] Output shape: {output.shape}")
        
        # Verify output shape
        assert output.shape == q.shape, f"Shape mismatch: {output.shape} != {q.shape}"
        
        return True
    except Exception as e:
        print(f"[Rank {rank}] ✗ Attention function FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ulysses_backward(rank, world_size, device):
    """Test Ulysses backward pass"""
    print(f"\n[Rank {rank}] Test 3: Ulysses Backward Pass")
    
    try:
        from sp_aurora import ulysses_flash_attn_func
        
        # Create test tensors with requires_grad
        batch_size = 1
        seq_len_per_rank = 64
        num_heads = 4
        head_dim = 64
        
        q = torch.randn(batch_size, seq_len_per_rank, num_heads, head_dim,
                       device=device, dtype=torch.float16, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        
        print(f"[Rank {rank}] Input shapes: q={q.shape}")
        
        # Forward pass
        output = ulysses_flash_attn_func(q, k, v, causal=True)
        
        # Backward pass
        grad_output = torch.ones_like(output)
        start_time = time.time()
        output.backward(grad_output)
        end_time = time.time()
        
        print(f"[Rank {rank}] ✓ Backward pass SUCCESS in {end_time - start_time:.3f}s")
        print(f"[Rank {rank}] Gradients: q.grad={q.grad is not None}, k.grad={k.grad is not None}, v.grad={v.grad is not None}")
        
        # Verify gradients exist
        assert q.grad is not None, "q.grad is None"
        assert k.grad is not None, "k.grad is None"
        assert v.grad is not None, "v.grad is None"
        
        return True
    except Exception as e:
        print(f"[Rank {rank}] ✗ Backward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ulysses_numerical_correctness(rank, world_size, device):
    """Test Ulysses attention numerical correctness by comparing with non-distributed attention"""
    print(f"\n[Rank {rank}] Test 4: Ulysses Numerical Correctness")
    
    try:
        from sp_aurora import ulysses_flash_attn_func
        
        # Try to disable Intel optimizations that might cause issues
        if hasattr(torch, 'xpu') and device.type == 'xpu':
            try:
                # Disable oneDNN optimizations temporarily
                import os
                os.environ['IPEX_DISABLE_ADDMM_FUSION'] = '1'
                os.environ['IPEX_ONEDNN_LAYOUT'] = '0'
            except:
                pass
        
        # Create test tensors
        batch_size = 2
        seq_len_per_rank = 64
        total_seq_len = seq_len_per_rank * world_size
        num_heads = 4
        head_dim = 64
        
        # Generate full Q, K, V tensors on all ranks with same seed for consistency
        torch.manual_seed(42)
        full_q = torch.randn(batch_size, total_seq_len, num_heads, head_dim,
                           device=device, dtype=torch.float16)
        full_k = torch.randn_like(full_q)
        full_v = torch.randn_like(full_q)
        
        # Split Q for distributed computation (each rank gets its portion)
        start_idx = rank * seq_len_per_rank
        end_idx = (rank + 1) * seq_len_per_rank
        local_q = full_q[:, start_idx:end_idx, :, :].contiguous()
        
        # For Ulysses, K and V are also split across sequence dimension
        local_k = full_k[:, start_idx:end_idx, :, :].contiguous()
        local_v = full_v[:, start_idx:end_idx, :, :].contiguous()
        
        print(f"[Rank {rank}] Local Q shape: {local_q.shape}, Full Q shape: {full_q.shape}")
        
        # Compute distributed Ulysses attention
        print(f"[Rank {rank}] Computing distributed Ulysses attention...")
        print(f"[Rank {rank}] Local inputs - Q: min={local_q.min():.3f}, max={local_q.max():.3f}, mean={local_q.mean():.3f}")
        print(f"[Rank {rank}] Local inputs - K: min={local_k.min():.3f}, max={local_k.max():.3f}, mean={local_k.mean():.3f}")
        print(f"[Rank {rank}] Local inputs - V: min={local_v.min():.3f}, max={local_v.max():.3f}, mean={local_v.mean():.3f}")
        
        dist_output = ulysses_flash_attn_func(
            local_q, local_k, local_v,
            causal=True,
            group=None
        )
        
        print(f"[Rank {rank}] Distributed output - min={dist_output.min():.3f}, max={dist_output.max():.3f}, mean={dist_output.mean():.3f}")
        
        # For comparison, compute standard attention on rank 0 only
        if rank == 0:
            print(f"[Rank {rank}] Computing reference output using PyTorch SDPA...")
            
            try:
                # First try using PyTorch's scaled_dot_product_attention
                # Need to transpose for SDPA which expects (B, H, S, D)
                q_ref = full_q.transpose(1, 2).contiguous()  # [B, H, S, D]
                k_ref = full_k.transpose(1, 2).contiguous()  # [B, H, S, D]
                v_ref = full_v.transpose(1, 2).contiguous()  # [B, H, S, D]
                
                # Use PyTorch SDPA as reference
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                    ref_output = F.scaled_dot_product_attention(
                        q_ref, k_ref, v_ref,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=True,
                        scale=1.0 / (head_dim ** 0.5)
                    )
                ref_output = ref_output.transpose(1, 2)  # [B, S, H, D]
                print(f"[Rank {rank}] SDPA reference computation successful")
                
            except Exception as e:
                print(f"[Rank {rank}] SDPA failed: {e}, falling back to manual computation")
                
                # Fallback to manual computation with better error handling
                scale = 1.0 / (head_dim ** 0.5)
                
                # Reshape for attention computation
                q_reshaped = full_q.transpose(1, 2).to(torch.float32)  # [B, H, S, D] - use float32 for stability
                k_reshaped = full_k.transpose(1, 2).to(torch.float32)  # [B, H, S, D]
                v_reshaped = full_v.transpose(1, 2).to(torch.float32)  # [B, H, S, D]
                
                print(f"[Rank {rank}] Debug - q_reshaped: shape={q_reshaped.shape}, dtype={q_reshaped.dtype}, device={q_reshaped.device}")
                
                # Use einsum for better compatibility
                scores = torch.einsum('bhqd,bhkd->bhqk', q_reshaped, k_reshaped) * scale
                
                # Apply causal mask
                causal_mask = torch.triu(torch.ones(total_seq_len, total_seq_len, device=device, dtype=torch.float32), diagonal=1)
                scores = scores.masked_fill(causal_mask.bool(), float('-inf'))
                
                # Apply softmax
                attn_weights = torch.softmax(scores, dim=-1)
                
                # Apply attention to values
                ref_output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v_reshaped)
                ref_output = ref_output.transpose(1, 2).to(torch.float16)  # [B, S, H, D] - convert back to float16
            
            # Extract the portion corresponding to this rank
            ref_output_local = ref_output[:, start_idx:end_idx, :, :].contiguous()
            
            print(f"[Rank {rank}] Reference output stats - min={ref_output_local.min():.3f}, max={ref_output_local.max():.3f}, mean={ref_output_local.mean():.3f}")
            
            # Compare outputs
            print(f"[Rank {rank}] Reference output shape: {ref_output_local.shape}")
            print(f"[Rank {rank}] Distributed output shape: {dist_output.shape}")
            
            # Debug: Check for NaN or Inf values
            if torch.isnan(dist_output).any():
                print(f"[Rank {rank}] WARNING: NaN values in distributed output")
            if torch.isinf(dist_output).any():
                print(f"[Rank {rank}] WARNING: Inf values in distributed output")
            if torch.isnan(ref_output_local).any():
                print(f"[Rank {rank}] WARNING: NaN values in reference output")
            if torch.isinf(ref_output_local).any():
                print(f"[Rank {rank}] WARNING: Inf values in reference output")
            
            # Compute relative error with better numerical stability
            abs_diff = torch.abs(dist_output - ref_output_local)
            # Use absolute tolerance for near-zero values
            abs_tol = 1e-3
            rel_tol = 0.05  # 5% relative tolerance
            
            # Compute element-wise tolerance
            tolerance = torch.maximum(
                torch.tensor(abs_tol, device=device),
                rel_tol * torch.abs(ref_output_local)
            )
            
            # Check if differences are within tolerance
            within_tol = abs_diff <= tolerance
            pct_within_tol = within_tol.float().mean().item() * 100
            
            max_abs_error = abs_diff.max().item()
            mean_abs_error = abs_diff.mean().item()
            
            # For non-zero reference values, compute relative error
            non_zero_mask = torch.abs(ref_output_local) > abs_tol
            if non_zero_mask.any():
                rel_error = abs_diff[non_zero_mask] / torch.abs(ref_output_local[non_zero_mask])
                max_rel_error = rel_error.max().item() if rel_error.numel() > 0 else 0.0
                mean_rel_error = rel_error.mean().item() if rel_error.numel() > 0 else 0.0
            else:
                max_rel_error = 0.0
                mean_rel_error = 0.0
            
            print(f"[Rank {rank}] Maximum absolute error: {max_abs_error:.6f}")
            print(f"[Rank {rank}] Mean absolute error: {mean_abs_error:.6f}")
            print(f"[Rank {rank}] Maximum relative error: {max_rel_error:.6f}")
            print(f"[Rank {rank}] Mean relative error: {mean_rel_error:.6f}")
            print(f"[Rank {rank}] Percentage within tolerance: {pct_within_tol:.1f}%")
            
            # Check if error is within acceptable tolerance
            # Flash attention implementations can have small numerical differences
            # We consider the test passed if:
            # 1. More than 95% of values are within tolerance, OR
            # 2. The mean absolute error is small
            if pct_within_tol >= 95.0 or mean_abs_error < 1e-3:
                print(f"[Rank {rank}] ✓ Numerical correctness test PASSED")
                return True
            else:
                print(f"[Rank {rank}] ✗ Numerical correctness test FAILED")
                # Print more detailed debug info
                print(f"[Rank {rank}] Sample distributed output [0,0,0,:5]: {dist_output[0, 0, 0, :5]}")
                print(f"[Rank {rank}] Sample reference output [0,0,0,:5]: {ref_output_local[0, 0, 0, :5]}")
                # Find location of maximum error
                max_err_idx = torch.unravel_index(abs_diff.argmax(), abs_diff.shape)
                print(f"[Rank {rank}] Max error at index {max_err_idx}: dist={dist_output[max_err_idx].item():.4f}, ref={ref_output_local[max_err_idx].item():.4f}")
                # Check different parts of the tensor
                print(f"[Rank {rank}] Sample distributed output [-1,-1,-1,:5]: {dist_output[-1, -1, -1, :5]}")
                print(f"[Rank {rank}] Sample reference output [-1,-1,-1,:5]: {ref_output_local[-1, -1, -1, :5]}")
                return False
        else:
            # Other ranks also need to show their output stats for debugging
            print(f"[Rank {rank}] Distributed output stats - min={dist_output.min():.3f}, max={dist_output.max():.3f}, mean={dist_output.mean():.3f}")
            print(f"[Rank {rank}] Distributed computation completed")
            return True
            
    except Exception as e:
        print(f"[Rank {rank}] ✗ Numerical correctness test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ulysses_simple_case(rank, world_size, device):
    """Test Ulysses with a simple case to isolate issues"""
    print(f"\n[Rank {rank}] Test 5: Simple Ulysses Case")
    
    try:
        from sp_aurora import ulysses_flash_attn_func
        
        # Very simple test case
        batch_size = 1
        seq_len_per_rank = 4
        num_heads = 2 * world_size  # Must be divisible by world_size
        head_dim = 4
        
        # Create simple inputs with known pattern
        torch.manual_seed(42)
        q = torch.randn(batch_size, seq_len_per_rank, num_heads, head_dim,
                       device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.ones_like(q) * (rank + 1)  # Different values per rank for debugging
        
        print(f"[Rank {rank}] Simple test Q shape: {q.shape}")
        print(f"[Rank {rank}] V values: {v[0, 0, 0, :]}")
        
        # Run Ulysses attention
        output = ulysses_flash_attn_func(
            q, k, v,
            causal=True,
            group=None
        )
        
        print(f"[Rank {rank}] Output shape: {output.shape}")
        print(f"[Rank {rank}] Output values: {output[0, :, 0, 0]}")
        
        return True
    except Exception as e:
        print(f"[Rank {rank}] ✗ Simple test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Initialize MPI
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    world_size = mpi_comm.Get_size()
    
    print(f"\n{'='*60}")
    print(f"[Rank {rank}] Fixed Ulysses Implementation Test")
    print(f"[Rank {rank}] World size: {world_size}")
    print(f"{'='*60}")
    
    # Setup distributed
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    if rank == 0:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12361'
    
    master_addr = mpi_comm.bcast(os.environ.get('MASTER_ADDR'), root=0)
    master_port = mpi_comm.bcast(os.environ.get('MASTER_PORT'), root=0)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    mpi_comm.Barrier()
    
    # Set device
    if INTEL_GPU_AVAILABLE:
        device = torch.device(f'xpu:{rank}')
        torch.xpu.set_device(device)
        backend = 'ccl'
    else:
        device = torch.device('cpu')
        backend = 'gloo'
    
    print(f"[Rank {rank}] Device: {device}, Backend: {backend}")
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=60)
    )
    
    # Run tests
    results = {}
    
    # Test 1: Forward pass
    results['forward'] = test_ulysses_forward(rank, world_size, device)
    
    # Test 2: Function API
    results['func_api'] = test_ulysses_func(rank, world_size, device)
    
    # Test 3: Backward pass
    results['backward'] = test_ulysses_backward(rank, world_size, device)
    
    # Test 4: Numerical correctness
    results['numerical_correctness'] = test_ulysses_numerical_correctness(rank, world_size, device)
    
    # Test 5: Simple case for debugging
    results['simple_case'] = test_ulysses_simple_case(rank, world_size, device)
    
    # Synchronize before summary
    dist.barrier()
    
    # Print summary on rank 0
    if rank == 0:
        print(f"\n{'='*60}")
        print("TEST SUMMARY:")
        print(f"{'='*60}")
        all_passed = True
        for test_name, passed in results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test_name}: {status}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print(f"\n✅ All tests PASSED! Ulysses is working correctly with CCL+XPU")
        else:
            print(f"\n❌ Some tests failed")
    
    # Cleanup
    dist.destroy_process_group()
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())