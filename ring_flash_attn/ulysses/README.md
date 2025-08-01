# Ulysses (Sequence Parallel) Attention

This module implements Ulysses attention following the yunchang structure, adapted for Intel GPU support.

## Structure

The Ulysses attention implementation is organized as follows:

```
ring_flash_attn/
├── ulysses/
│   ├── __init__.py
│   └── attn_layer.py    # Main UlyssesAttention class
└── comm/
    ├── __init__.py
    └── all_to_all.py     # All-to-all communication operations
```

## Usage

### Class-based Interface (Recommended)

```python
from ring_flash_attn import UlyssesAttention
from ring_flash_attn.ulysses.attn_layer import AttnType

# Create Ulysses attention module
ulysses_attn = UlyssesAttention(
    sequence_process_group=None,  # Process group for distributed
    scatter_idx=2,                # Dimension to scatter
    gather_idx=1,                 # Dimension to gather
    use_sync=False,               # Whether to sync after all-to-all
    attn_type=AttnType.TORCH      # Attention backend type
)

# Forward pass
output = ulysses_attn(query, key, value, causal=True)
```

### Function-based Interface (Backward Compatible)

```python
from ring_flash_attn import ulysses_flash_attn_func

# Direct function call
output = ulysses_flash_attn_func(q, k, v, causal=True)
```

## Features

- **Modular Design**: Clean separation of attention layer, communication, and kernel selection
- **Intel GPU Support**: Optimized for Intel Data Center GPUs with XPU backend
- **Flexible Backends**: Support for multiple attention implementations (PyTorch SDPA, Intel SYCL)
- **Distributed Support**: Built-in support for sequence parallelism across multiple GPUs
- **Backward Compatible**: Maintains compatibility with existing function-based API

## Attention Types

- `AttnType.TORCH`: PyTorch scaled dot-product attention (default)
- `AttnType.INTEL_SYCL`: Intel SYCL optimized kernels (when available)
- `AttnType.INTEL_ONEDNN`: Intel oneDNN backend (future)

## Communication Pattern

Ulysses attention redistributes sequences across GPUs:

1. **Forward All-to-All**: (bs, seq/P, heads, dim) → (bs, seq, heads/P, dim)
2. **Local Attention**: Compute attention on full sequence with reduced heads
3. **Backward All-to-All**: (bs, seq, heads/P, dim) → (bs, seq/P, heads, dim)

This allows processing longer sequences by distributing them across multiple GPUs.

## Examples

See `examples/example_ulysses_class_based.py` for comprehensive usage examples.