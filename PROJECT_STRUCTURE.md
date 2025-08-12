# Project Structure - Ring Flash Attention

## Overview

This project implements Ring Flash Attention and Ulysses (Sequence Parallel) Attention for Intel GPUs using XPU and oneCCL.

## Key Findings

**CCL fully supports XPU operations** - All collective operations (all_reduce, all_to_all, send/recv, etc.) work correctly on Intel XPU devices with the CCL backend. CPU fallback mechanisms are not needed.

## Directory Structure

```
Ring-FT/
├── sp_aurora/              # Main source code
│   ├── intel_sp_aurora.py  # Ring attention implementation
│   ├── intel_ulysses_attn.py     # Ulysses attention implementation
│   ├── intel_utils.py            # Shared utilities (RingComm, etc.)
│   ├── intel_flash_attn.py       # Base flash attention operations
│   ├── intel_flash_attn_sycl.py  # SYCL kernel interface
│   ├── ulysses/                  # Ulysses-specific modules
│   │   ├── __init__.py
│   │   └── attn_layer.py         # Ulysses attention layer
│   ├── comm/                     # Communication utilities
│   │   └── __init__.py
│   └── sycl_bindings/            # SYCL C++ bindings
│       └── bindings.cpp
│
├── test/                         # Test suite
│   ├── verification/             # Verification tests
│   │   ├── test_xpu_alltoall_minimal.py
│   │   ├── test_xpu_collectives_debug.py
│   │   ├── test_ring_init_pattern.py
│   │   ├── test_intel_alltoall_4d.py
│   │   ├── test_ulysses_fixed.py
│   │   └── test_ring_ulysses_final.py
│   ├── debug/                    # Debug and diagnostic tests
│   │   ├── test_ulysses_minimal_hang.py
│   │   ├── diagnose_ccl_hang.py
│   │   └── test_ccl_minimal.py
│   ├── test_ring.py              # Main Ring attention tests
│   ├── test_intel_ulysses_attn.py # Main Ulysses tests
│   └── *.sh                      # Test runner scripts
│
├── cclogs/                       # Documentation and logs
│   ├── ccl_alltoall_hang_mpiexec_issue_20250731.md  # Historical issue doc
│   └── ccl_xpu_support_verified_20250801.md         # Current status
│
├── archived_fallbacks/           # Archived fallback implementations
│   ├── intel_ulysses_attn_cpu_fallback.py
│   ├── intel_ulysses_attn_with_fallback.py
│   ├── all_to_all_fallback.py
│   └── test_*_fallback.py
│
└── examples/                     # Example usage
    ├── example_ulysses_attn.py
    └── example_ulysses_class_based.py
```

## Key Components

### Ring Flash Attention (`intel_sp_aurora.py`)
- Uses point-to-point communication (isend/irecv)
- Processes attention in a ring pattern across GPUs
- Each GPU computes partial attention with its local K,V

### Ulysses Attention (`intel_ulysses_attn.py`)
- Uses all-to-all collective for sequence redistribution
- Partitions sequence dimension across GPUs
- Each GPU computes full attention on its sequence partition

### Utilities (`intel_utils.py`)
- `IntelRingComm`: Handles ring communication pattern
- `update_out_and_lse`: Combines attention outputs
- XPU device management

## Important Notes

1. **Use mpirun, not mpiexec** in PBS environments
2. **CCL backend works with XPU** - no fallback needed
3. **Dummy all_reduce pattern** - Ring uses a dummy all_reduce before P2P ops; this was added to Ulysses for consistency

## Testing

Run verification tests:
```bash
mpirun -n 2 python test/verification/test_ring_ulysses_final.py
```

Run specific component tests:
```bash
# Test Ring attention
mpirun -n 2 python test/test_ring.py

# Test Ulysses attention  
mpirun -n 2 python test/test_intel_ulysses_attn.py
```

## Environment

- PyTorch 2.5.1+
- Intel Extension for PyTorch (IPEX) 2.5.10+xpu
- oneCCL bindings 2.5.0+xpu
- Intel GPU (XPU) support required