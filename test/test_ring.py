#!/usr/bin/env python3
"""
Modular test suite to isolate ring flash attention components and identify hang issues.
Each test can be run independently by commenting out others or using TEST_ONLY env var.

Usage:
    # Run all tests
    mpiexec -n 2 python test_ring_components_isolation.py
    
    # Run specific test only
    TEST_ONLY=test_communication_primitives mpiexec -n 2 python test_ring_components_isolation.py
    
    # Enable debug mode
    DEBUG=1 mpiexec -n 2 python test_ring_components_isolation.py
    
    # Skip certain tests
    SKIP_TESTS=test_full_ring_attention,test_ring_backward_only mpiexec -n 2 python test_ring_components_isolation.py
"""

import os
import sys
import socket
import datetime
import signal
import time
from time import perf_counter_ns
import torch
import torch.distributed as dist
from mpi4py import MPI
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for Intel GPU support
try:
    import intel_extension_for_pytorch as ipex
    import oneccl_bindings_for_pytorch
    INTEL_GPU_AVAILABLE = torch.xpu.is_available() if hasattr(torch, 'xpu') else False
except ImportError as e:
    print(f"Warning: Intel Extension not available: {e}")
    INTEL_GPU_AVAILABLE = False

# Import ring flash attention modules
from ring_flash_attn.intel_ring_flash_attn import intel_ring_flash_attn_func, intel_ring_flash_attn_forward, intel_ring_flash_attn_backward
from ring_flash_attn.intel_utils import IntelRingComm, update_out_and_lse
from ring_flash_attn.intel_flash_attn import _flash_attn_forward, _flash_attn_backward

# Global variables for setup
RANK = None
WORLD_SIZE = None
DEVICE = None
DEBUG = os.environ.get('DEBUG', '0') == '1'
TEST_ONLY = os.environ.get('TEST_ONLY', None)
SKIP_TESTS = os.environ.get('SKIP_TESTS', '').split(',') if os.environ.get('SKIP_TESTS') else []

def debug_print(msg):
    """Print debug messages if DEBUG is enabled"""
    if DEBUG:
        print(f"[DEBUG][Rank {RANK}] {msg}")

def test_marker(test_name, status="START"):
    """Print clear test markers"""
    print(f"\n{'='*60}")
    print(f"[Rank {RANK}] TEST {status}: {test_name}")
    print(f"{'='*60}")

def setup_timeout(seconds=30):
    """Setup timeout handler"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Test timed out after {seconds} seconds at rank {RANK}")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

def cancel_timeout():
    """Cancel timeout"""
    signal.alarm(0)

def should_run_test(test_name):
    """Check if test should be run based on environment variables"""
    if TEST_ONLY and TEST_ONLY != test_name:
        return False
    if test_name in SKIP_TESTS:
        return False
    return True

# ========== TEST 1: Basic Setup Test ==========
def test_basic_setup():
    """Test basic MPI and distributed setup"""
    if not should_run_test('test_basic_setup'):
        return True
        
    test_marker("test_basic_setup", "START")
    
    try:
        # Test MPI initialization
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        mpi_size = mpi_comm.Get_size()
        
        print(f"[Rank {mpi_rank}] MPI initialized: rank={mpi_rank}, size={mpi_size}")
        
        # Test device availability
        if INTEL_GPU_AVAILABLE:
            device_count = torch.xpu.device_count()
            device_id = mpi_rank % device_count
            device = torch.device(f'xpu:{device_id}')
            print(f"[Rank {mpi_rank}] Intel GPU available: {device_count} devices, using {device}")
        else:
            device = torch.device('cpu')
            print(f"[Rank {mpi_rank}] Using CPU (Intel GPU not available)")
        
        # Test tensor creation
        test_tensor = torch.randn(10, 10, device=device)
        print(f"[Rank {mpi_rank}] Tensor creation successful: shape={test_tensor.shape}, device={test_tensor.device}")
        
        test_marker("test_basic_setup", "PASS")
        return True
        
    except Exception as e:
        print(f"[Rank {RANK}] ❌ Basic setup failed: {e}")
        traceback.print_exc()
        test_marker("test_basic_setup", "FAIL")
        return False

# ========== TEST 2: Communication Primitives Test ==========
def test_communication_primitives():
    """Test basic PyTorch distributed communication primitives"""
    if not should_run_test('test_communication_primitives'):
        return True
        
    test_marker("test_communication_primitives", "START")
    
    try:
        if WORLD_SIZE == 1:
            print(f"[Rank {RANK}] Skipping communication test (single process)")
            return True
        
        # Test 1: Basic all_reduce
        debug_print("Testing all_reduce...")
        test_tensor = torch.tensor([float(RANK)], device=DEVICE)
        dist.all_reduce(test_tensor)
        expected = sum(range(WORLD_SIZE))
        if abs(test_tensor.item() - expected) > 1e-6:
            raise ValueError(f"all_reduce failed: expected {expected}, got {test_tensor.item()}")
        print(f"[Rank {RANK}] ✓ all_reduce successful")
        
        # Test 2: Broadcast
        debug_print("Testing broadcast...")
        if RANK == 0:
            bcast_tensor = torch.tensor([42.0], device=DEVICE)
        else:
            bcast_tensor = torch.tensor([0.0], device=DEVICE)
        dist.broadcast(bcast_tensor, src=0)
        if abs(bcast_tensor.item() - 42.0) > 1e-6:
            raise ValueError(f"broadcast failed: expected 42.0, got {bcast_tensor.item()}")
        print(f"[Rank {RANK}] ✓ broadcast successful")
        
        # Test 3: Basic send/recv with rank parity ordering
        debug_print("Testing send/recv with rank parity ordering...")
        send_tensor = torch.tensor([float(RANK) * 10], device=DEVICE)
        recv_tensor = torch.empty_like(send_tensor)
        
        next_rank = (RANK + 1) % WORLD_SIZE
        prev_rank = (RANK - 1) % WORLD_SIZE
        
        setup_timeout(10)
        
        if RANK % 2 == 0:
            # Even ranks: send first, then receive
            debug_print(f"Even rank {RANK}: sending to {next_rank}")
            dist.send(send_tensor, next_rank)
            debug_print(f"Even rank {RANK}: receiving from {prev_rank}")
            dist.recv(recv_tensor, prev_rank)
        else:
            # Odd ranks: receive first, then send
            debug_print(f"Odd rank {RANK}: receiving from {prev_rank}")
            dist.recv(recv_tensor, prev_rank)
            debug_print(f"Odd rank {RANK}: sending to {next_rank}")
            dist.send(send_tensor, next_rank)
        
        cancel_timeout()
        
        expected_value = float(prev_rank) * 10
        if abs(recv_tensor.item() - expected_value) > 1e-6:
            raise ValueError(f"send/recv failed: expected {expected_value}, got {recv_tensor.item()}")
        print(f"[Rank {RANK}] ✓ send/recv with rank parity successful")
        
        # Test 4: isend/irecv
        debug_print("Testing isend/irecv...")
        send_tensor2 = torch.tensor([float(RANK) * 100], device=DEVICE)
        recv_tensor2 = torch.empty_like(send_tensor2)
        
        setup_timeout(10)
        
        if RANK % 2 == 0:
            send_req = dist.isend(send_tensor2, next_rank)
            recv_req = dist.irecv(recv_tensor2, prev_rank)
        else:
            recv_req = dist.irecv(recv_tensor2, prev_rank)
            send_req = dist.isend(send_tensor2, next_rank)
        
        send_req.wait()
        recv_req.wait()
        
        cancel_timeout()
        
        expected_value2 = float(prev_rank) * 100
        if abs(recv_tensor2.item() - expected_value2) > 1e-6:
            raise ValueError(f"isend/irecv failed: expected {expected_value2}, got {recv_tensor2.item()}")
        print(f"[Rank {RANK}] ✓ isend/irecv successful")
        
        test_marker("test_communication_primitives", "PASS")
        return True
        
    except TimeoutError as e:
        cancel_timeout()
        print(f"[Rank {RANK}] ❌ Communication timeout: {e}")
        print(f"[Rank {RANK}] This indicates a deadlock in basic communication")
        test_marker("test_communication_primitives", "FAIL")
        return False
    except Exception as e:
        cancel_timeout()
        print(f"[Rank {RANK}] ❌ Communication primitives failed: {e}")
        traceback.print_exc()
        test_marker("test_communication_primitives", "FAIL")
        return False

# ========== TEST 3: Ring Communication Test ==========
def test_ring_comm():
    """Test IntelRingComm class"""
    if not should_run_test('test_ring_comm'):
        return True
        
    test_marker("test_ring_comm", "START")
    
    try:
        if WORLD_SIZE == 1:
            print(f"[Rank {RANK}] Skipping ring comm test (single process)")
            return True
        
        # Check if distributed is properly initialized
        if not dist.is_initialized():
            print(f"[Rank {RANK}] ERROR: Distributed not initialized for ring comm test")
            print(f"[Rank {RANK}] This test requires proper distributed setup")
            return False
        
        # Determine the appropriate device for communication
        comm_device = DEVICE
        
        # For XPU devices, ensure proper synchronization
        if INTEL_GPU_AVAILABLE and DEVICE.type == 'xpu':
            # Synchronize XPU device before communication
            torch.xpu.synchronize(DEVICE)
            print(f"[Rank {RANK}] Using XPU device: {DEVICE}")
        else:
            print(f"[Rank {RANK}] Using device: {DEVICE}")
        
        # Initialize communicators with a collective operation (required by PyTorch)
        # Create dummy tensor on the same device as we'll use for communication
        dummy = torch.tensor([1.0], device=comm_device, dtype=torch.float32)
        dist.broadcast(dummy, src=0)
        print(f"[Rank {RANK}] Communicators initialized with broadcast, dummy={dummy.item()}")

        # Initialize ring communicator
        debug_print("Initializing IntelRingComm...")
        comm = IntelRingComm(None)  # Use default process group
        print(f"[Rank {RANK}] RingComm initialized: send_rank={comm.send_rank}, recv_rank={comm.recv_rank}")
        
        # Override the _ensure_xpu_tensor method to handle both CPU and XPU properly
        original_ensure_xpu = comm._ensure_xpu_tensor
        
        def smart_ensure_tensor(tensor):
            """Smart tensor device handler that respects the current device context"""
            if INTEL_GPU_AVAILABLE and DEVICE.type == 'xpu':
                # If we're using XPU, ensure tensor is on the correct XPU device
                if tensor.device.type != 'xpu':
                    return tensor.to(DEVICE)
                return tensor
            else:
                # For CPU or when XPU is not available, keep tensor as is
                return tensor
        
        # Replace the method
        comm._ensure_xpu_tensor = smart_ensure_tensor

        # Test 1: Simple send_recv
        debug_print("Testing simple send_recv...")
        test_data = torch.tensor([float(RANK) + 0.5], device=DEVICE)
        
        setup_timeout(10)
        next_data = comm.send_recv(test_data)
        comm.wait()
        cancel_timeout()
        
        expected = float((RANK - 1) % WORLD_SIZE) + 0.5
        if abs(next_data.item() - expected) > 1e-6:
            raise ValueError(f"send_recv failed: expected {expected}, got {next_data.item()}")
        print(f"[Rank {RANK}] ✓ send_recv successful")
        
        # Test 2: send_recv_kv
        debug_print("Testing send_recv_kv...")
        k_data = torch.randn(2, 4, device=DEVICE) * (RANK + 1)
        v_data = torch.randn(2, 4, device=DEVICE) * (RANK + 10)
        
        setup_timeout(10)
        next_k, next_v = comm.send_recv_kv(k_data, v_data)
        comm.wait()
        cancel_timeout()
        
        print(f"[Rank {RANK}] ✓ send_recv_kv successful")
        print(f"[Rank {RANK}]   Sent k sum: {k_data.sum().item():.2f}, Received k sum: {next_k.sum().item():.2f}")
        print(f"[Rank {RANK}]   Sent v sum: {v_data.sum().item():.2f}, Received v sum: {next_v.sum().item():.2f}")
        
        test_marker("test_ring_comm", "PASS")
        return True
        
    except TimeoutError as e:
        cancel_timeout()
        print(f"[Rank {RANK}] ❌ Ring comm timeout: {e}")
        test_marker("test_ring_comm", "FAIL")
        return False
    except Exception as e:
        cancel_timeout()
        print(f"[Rank {RANK}] ❌ Ring comm failed: {e}")
        traceback.print_exc()
        test_marker("test_ring_comm", "FAIL")
        return False

# ========== TEST 4: Flash Attention Components Test ==========
def test_flash_attn_components():
    """Test individual flash attention components"""
    if not should_run_test('test_flash_attn_components'):
        return True
        
    test_marker("test_flash_attn_components", "START")
    
    try:
        # Test parameters
        batch_size = 1
        seq_len = 64
        num_heads = 4
        head_dim = 32
        
        # Create test tensors
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
        
        # Test 1: Flash attention forward
        debug_print("Testing flash attention forward...")
        out, lse = _flash_attn_forward(q, k, v, causal=True, softmax_scale=1.0)
        print(f"[Rank {RANK}] ✓ Flash attention forward successful")
        print(f"[Rank {RANK}]   Output shape: {out.shape}, LSE shape: {lse.shape}")
        
        # Test 2: update_out_and_lse
        debug_print("Testing update_out_and_lse...")
        block_out = torch.randn_like(out)
        block_lse = torch.randn(batch_size, num_heads, seq_len, device=DEVICE, dtype=out.dtype)
        
        updated_out, updated_lse = update_out_and_lse(out.float(), lse.unsqueeze(-1), block_out, block_lse)
        print(f"[Rank {RANK}] ✓ update_out_and_lse successful")
        
        # Test 3: Dimension handling
        debug_print("Testing dimension conversions...")
        # Test transpose between [batch, seq, heads, dim] and [batch, heads, seq, dim]
        q_transposed = q.transpose(1, 2)  # [batch, seq, heads, dim]
        print(f"[Rank {RANK}] ✓ Dimension handling successful")
        print(f"[Rank {RANK}]   Original shape: {q.shape}, Transposed shape: {q_transposed.shape}")
        
        test_marker("test_flash_attn_components", "PASS")
        return True
        
    except Exception as e:
        print(f"[Rank {RANK}] ❌ Flash attention components failed: {e}")
        traceback.print_exc()
        test_marker("test_flash_attn_components", "FAIL")
        return False

# ========== TEST 5: Ring Flash Attention Forward Only ==========
def test_ring_forward_only():
    """Test only the forward pass of ring flash attention"""
    if not should_run_test('test_ring_forward_only'):
        return True
        
    test_marker("test_ring_forward_only", "START")
    
    try:
        if WORLD_SIZE == 1:
            print(f"[Rank {RANK}] Skipping ring forward test (single process)")
            return True
        
        # Small test parameters for debugging
        batch_size = 1
        seq_len = 128  # Total sequence length
        num_heads = 4
        head_dim = 32
        
        local_seq_len = seq_len // WORLD_SIZE
        
        # Create local tensors
        q = torch.randn(batch_size, local_seq_len, num_heads, head_dim, device=DEVICE, dtype=torch.float16)
        k = torch.randn(batch_size, local_seq_len, num_heads, head_dim, device=DEVICE, dtype=torch.float16)
        v = torch.randn(batch_size, local_seq_len, num_heads, head_dim, device=DEVICE, dtype=torch.float16)
        
        print(f"[Rank {RANK}] Created tensors: q.shape={q.shape}")
        
        debug_print("Calling intel_ring_flash_attn_forward...")
        setup_timeout(30)
        
        out, lse = intel_ring_flash_attn_forward(
            process_group=None,
            q=q,
            k=k,
            v=v,
            softmax_scale=1.0 / (head_dim ** 0.5),
            dropout_p=0.0,
            causal=True
        )
        
        cancel_timeout()
        
        print(f"[Rank {RANK}] ✓ Ring forward pass successful")
        print(f"[Rank {RANK}]   Output shape: {out.shape}, LSE shape: {lse.shape}")
        print(f"[Rank {RANK}]   Output sum: {out.sum().item():.4f}")
        
        test_marker("test_ring_forward_only", "PASS")
        return True
        
    except TimeoutError as e:
        cancel_timeout()
        print(f"[Rank {RANK}] ❌ Ring forward timeout: {e}")
        print(f"[Rank {RANK}] Likely deadlock in ring communication pattern")
        test_marker("test_ring_forward_only", "FAIL")
        return False
    except Exception as e:
        cancel_timeout()
        print(f"[Rank {RANK}] ❌ Ring forward failed: {e}")
        traceback.print_exc()
        test_marker("test_ring_forward_only", "FAIL")
        return False

# ========== TEST 6: Ring Flash Attention Backward Only ==========
def test_ring_backward_only():
    """Test backward pass isolation"""
    if not should_run_test('test_ring_backward_only'):
        return True
        
    test_marker("test_ring_backward_only", "START")
    
    try:
        if WORLD_SIZE == 1:
            print(f"[Rank {RANK}] Skipping ring backward test (single process)")
            return True
        
        # Small test parameters
        batch_size = 1
        seq_len = 64
        num_heads = 2
        head_dim = 32
        
        local_seq_len = seq_len // WORLD_SIZE
        
        # Create tensors with gradients
        q = torch.randn(batch_size, local_seq_len, num_heads, head_dim, device=DEVICE, dtype=torch.float16, requires_grad=True)
        k = torch.randn(batch_size, local_seq_len, num_heads, head_dim, device=DEVICE, dtype=torch.float16, requires_grad=True)
        v = torch.randn(batch_size, local_seq_len, num_heads, head_dim, device=DEVICE, dtype=torch.float16, requires_grad=True)
        
        # First run forward to get outputs
        debug_print("Running forward pass for backward test...")
        out, lse = intel_ring_flash_attn_forward(None, q, k, v, 1.0/(head_dim**0.5), 0.0, True)
        
        # Create gradient tensor
        dout = torch.randn_like(out)
        
        debug_print("Testing backward pass...")
        setup_timeout(30)
        
        dq, dk, dv = intel_ring_flash_attn_backward(
            process_group=None,
            dout=dout,
            q=q,
            k=k,
            v=v,
            out=out,
            softmax_lse=lse,
            softmax_scale=1.0/(head_dim**0.5),
            dropout_p=0.0,
            causal=True
        )
        
        cancel_timeout()
        
        print(f"[Rank {RANK}] ✓ Ring backward pass successful")
        print(f"[Rank {RANK}]   dq shape: {dq.shape}, dk shape: {dk.shape}, dv shape: {dv.shape}")
        print(f"[Rank {RANK}]   Gradient norms: dq={dq.norm().item():.4f}, dk={dk.norm().item():.4f}, dv={dv.norm().item():.4f}")
        
        test_marker("test_ring_backward_only", "PASS")
        return True
        
    except TimeoutError as e:
        cancel_timeout()
        print(f"[Rank {RANK}] ❌ Ring backward timeout: {e}")
        test_marker("test_ring_backward_only", "FAIL")
        return False
    except Exception as e:
        cancel_timeout()
        print(f"[Rank {RANK}] ❌ Ring backward failed: {e}")
        traceback.print_exc()
        test_marker("test_ring_backward_only", "FAIL")
        return False

# ========== TEST 7: Full Ring Flash Attention Test ==========
def test_full_ring_attention():
    """Test complete ring flash attention with forward and backward"""
    if not should_run_test('test_full_ring_attention'):
        return True
        
    test_marker("test_full_ring_attention", "START")
    
    try:
        if WORLD_SIZE == 1:
            print(f"[Rank {RANK}] Skipping full ring test (single process)")
            return True
        
        # Test parameters
        batch_size = 2
        seq_len = 256
        num_heads = 8
        head_dim = 64
        
        local_seq_len = seq_len // WORLD_SIZE
        
        # Create tensors
        q = torch.randn(batch_size, local_seq_len, num_heads, head_dim, device=DEVICE, dtype=torch.float16, requires_grad=True)
        k = torch.randn(batch_size, local_seq_len, num_heads, head_dim, device=DEVICE, dtype=torch.float16, requires_grad=True)
        v = torch.randn(batch_size, local_seq_len, num_heads, head_dim, device=DEVICE, dtype=torch.float16, requires_grad=True)
        
        print(f"[Rank {RANK}] Testing full ring attention with shape: {q.shape}")
        
        debug_print("Running full ring flash attention...")
        setup_timeout(60)
        
        # Forward pass
        out = intel_ring_flash_attn_func(
            q, k, v,
            dropout_p=0.0,
            causal=True,
            return_attn_probs=False
        )
        
        print(f"[Rank {RANK}] ✓ Forward pass complete")
        
        # Backward pass
        dout = torch.randn_like(out)
        out.backward(dout)
        
        cancel_timeout()
        
        print(f"[Rank {RANK}] ✓ Full ring attention successful")
        print(f"[Rank {RANK}]   Output shape: {out.shape}")
        print(f"[Rank {RANK}]   Gradients: q.grad={q.grad is not None}, k.grad={k.grad is not None}, v.grad={v.grad is not None}")
        
        test_marker("test_full_ring_attention", "PASS")
        return True
        
    except TimeoutError as e:
        cancel_timeout()
        print(f"[Rank {RANK}] ❌ Full ring timeout: {e}")
        test_marker("test_full_ring_attention", "FAIL")
        return False
    except Exception as e:
        cancel_timeout()
        print(f"[Rank {RANK}] ❌ Full ring failed: {e}")
        traceback.print_exc()
        test_marker("test_full_ring_attention", "FAIL")
        return False

# ========== Main Test Runner ==========
def main():
    global RANK, WORLD_SIZE, DEVICE
    
    # Get actual MPI rank and size from MPI.COMM_WORLD
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    
    # Debug: Print what MPI sees vs PMI env vars
    print(f"[Process] MPI reports: rank={mpi_rank}, size={mpi_size}")
    print(f"[Process] PMI env: PMI_RANK={os.environ.get('PMI_RANK')}, PMI_SIZE={os.environ.get('PMI_SIZE')}")
    
    # If MPI is not seeing the correct world size, try to use PMI values
    if mpi_size == 1 and os.environ.get('PMI_SIZE', '1') != '1':
        # MPI might not be properly initialized for multi-process
        print(f"[Process] WARNING: MPI size mismatch. Using PMI values instead.")
        RANK = int(os.environ.get('PMI_RANK', 0))
        WORLD_SIZE = int(os.environ.get('PMI_SIZE', 1))
    else:
        RANK = mpi_rank
        WORLD_SIZE = mpi_size
    
    # Setup environment variables for PyTorch distributed
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(WORLD_SIZE)
    # Broadcast master address and port (only if actually multi-process)
    if WORLD_SIZE > 1:
        if RANK == 0:
            master_addr = socket.gethostname()
            master_port = 2345
        else:
            master_addr = None
            master_port = None
        
        # Only use MPI collective if MPI sees multiple processes
        if mpi_size > 1:
            master_addr = mpi_comm.bcast(master_addr, root=0)
            master_port = mpi_comm.bcast(master_port, root=0)
        else:
            # Fallback: rank 0 should set these, others read from env
            if RANK == 0:
                os.environ["MASTER_ADDR"] = master_addr
                os.environ["MASTER_PORT"] = str(master_port)
            # For non-zero ranks, these should already be in environment
        
        os.environ["MASTER_ADDR"] = master_addr if master_addr else os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = str(master_port) if master_port else os.environ.get("MASTER_PORT", "2345")
    
    # Only barrier if MPI actually sees multiple processes
    if mpi_size > 1:
        mpi_comm.Barrier()
    
    # Set device early (before any tests)
    if INTEL_GPU_AVAILABLE:
        DEVICE = torch.device(f"xpu:{RANK % torch.xpu.device_count()}")
        # Initialize XPU device
        torch.xpu.set_device(DEVICE)
        print(f"[Rank {RANK}] XPU device set: {DEVICE}")
    else:
        DEVICE = torch.device('cpu')
        print(f"[Rank {RANK}] Using CPU device")
    
    # Initialize process group early if running specific tests that need it
    if WORLD_SIZE > 1 and TEST_ONLY in ['test_ring_comm', 'test_ring_forward_only', 'test_ring_backward_only', 'test_full_ring_attention']:
        print(f"[Rank {RANK}] Early initialization for {TEST_ONLY}")
        print(f"[Rank {RANK}] MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}")
        
        # Initialize with appropriate backend
        backend = "ccl" if INTEL_GPU_AVAILABLE else "gloo"
        print(f"[Rank {RANK}] Using backend: {backend}")
        
        dist.init_process_group(
            backend=backend, 
            init_method='env://', 
            world_size=WORLD_SIZE, 
            rank=RANK,
            timeout=datetime.timedelta(seconds=360)
        )
        if mpi_size > 1:
            mpi_comm.Barrier()
    
    # Run basic setup test first
    if not test_basic_setup():
        return 1
    
    # Initialize process group for remaining tests (if not already initialized)
    if WORLD_SIZE > 1 and not dist.is_initialized():
        print(f"[Rank {RANK}] Initializing process group...")
        print(f"[Rank {RANK}] MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}")
        
        # Choose backend based on device availability
        backend = "ccl" if INTEL_GPU_AVAILABLE else "gloo"
        print(f"[Rank {RANK}] Using backend: {backend}")
        
        dist.init_process_group(
            backend=backend, 
            init_method='env://', 
            world_size=WORLD_SIZE, 
            rank=RANK,
            timeout=datetime.timedelta(seconds=360)
        )
        if mpi_size > 1:
            mpi_comm.Barrier()
    
    # Verify distributed is initialized correctly
    if dist.is_initialized():
        dist_rank = dist.get_rank()
        dist_world_size = dist.get_world_size()
        if dist_rank != RANK or dist_world_size != WORLD_SIZE:
            print(f"[Rank {RANK}] WARNING: Distributed rank mismatch: expected {RANK}/{WORLD_SIZE}, got {dist_rank}/{dist_world_size}")
            RANK = dist_rank
            WORLD_SIZE = dist_world_size
    
    print(f"\n[Rank {RANK}] Starting component isolation tests")
    print(f"[Rank {RANK}] Configuration: WORLD_SIZE={WORLD_SIZE}, DEVICE={DEVICE}")
    if TEST_ONLY:
        print(f"[Rank {RANK}] Running only: {TEST_ONLY}")
    if SKIP_TESTS:
        print(f"[Rank {RANK}] Skipping: {SKIP_TESTS}")
    
    # Run tests in order
    tests = [
        ("Communication Primitives", test_communication_primitives),
        ("Ring Communication", test_ring_comm),
        ("Flash Attention Components", test_flash_attn_components),
        ("Ring Forward Only", test_ring_forward_only),
        ("Ring Backward Only", test_ring_backward_only),
        ("Full Ring Attention", test_full_ring_attention),
    ]
    
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            if not test_func():
                failed_tests.append(test_name)
                # Option to stop on first failure
                if os.environ.get('STOP_ON_FAILURE', '0') == '1':
                    break
        except Exception as e:
            print(f"[Rank {RANK}] Unexpected error in {test_name}: {e}")
            traceback.print_exc()
            failed_tests.append(test_name)
            if os.environ.get('STOP_ON_FAILURE', '0') == '1':
                break
    
    # Summary
    print(f"\n{'='*60}")
    print(f"[Rank {RANK}] TEST SUMMARY")
    print(f"{'='*60}")
    if failed_tests:
        print(f"[Rank {RANK}] ❌ Failed tests: {', '.join(failed_tests)}")
    else:
        print(f"[Rank {RANK}] ✅ All tests passed!")
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    
    return len(failed_tests)

if __name__ == "__main__":
    sys.exit(main())