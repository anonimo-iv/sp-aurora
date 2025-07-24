# Ring Flash Attention for Intel GPUs

Ring Flash Attention implementation optimized for Intel GPUs (XPU) using Intel Extension for PyTorch and oneCCL backend.

## Features

- **Intel GPU Optimized**: Native support for Intel Data Center GPU Max series
- **Ring Attention Variants**: Basic, zigzag, and stripe attention patterns
- **Variable Length Support**: Efficient handling of packed sequences
- **MPI Compatibility**: Works with both `torchrun` and `mpiexec` launchers
- **Hugging Face Integration**: Drop-in replacement for transformer models

## Installation

```bash
pip install ring-flash-attn
```

Or build from source:
```bash
git clone https://github.com/zhuzilin/ring-flash-attention.git
cd ring-flash-attention
pip install .
```

## Quick Start

```python
import torch
import intel_extension_for_pytorch as ipex
from ring_flash_attn import ring_flash_attn_func, setup_mpi_distributed

# Setup distributed environment
setup_info = setup_mpi_distributed()
device = setup_info['device']  # automatically detects 'xpu' for Intel GPUs

# Example usage
batch_size, seq_len, num_heads, head_dim = 2, 1024, 32, 64
q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)

# Run ring flash attention
output = ring_flash_attn_func(q, k, v, causal=True)
```

## Launching Distributed Training

```bash
# Single node with 4 GPUs
torchrun --nproc_per_node=4 your_script.py

# Multi-node with MPI
mpiexec -n 8 python your_script.py

# Intel MPI with optimizations
mpiexec -n 8 -genv CCL_BACKEND=native -genv CCL_ATL_TRANSPORT=ofi python your_script.py
```

## API Reference

### Core Functions
- `ring_flash_attn_func`: Basic ring attention
- `zigzag_ring_flash_attn_func`: Compute-balanced ring attention
- `stripe_flash_attn_func`: Stripe pattern attention
- `intel_ring_flash_attn_func`: Intel GPU optimized implementation

### Variable Length Support
- `ring_flash_attn_varlen_func`
- `zigzag_ring_flash_attn_varlen_func`
- `llama3_flash_attn_varlen_func` (recommended for most varlen cases)

Each function includes `*_kvpacked_func` and `*_qkvpacked_func` variants.

## Performance

Zigzag ring attention achieves up to 90% efficiency compared to theoretical single-GPU performance:

| Configuration | Efficiency |
|--------------|------------|
| Forward only | 85-90%     |
| Forward + Backward | 80-90% |

*Benchmarked on 8xA100/8xH800 with 8K sequence length per GPU*

## Requirements

- Intel Extension for PyTorch (`intel-extension-for-pytorch>=2.0.0`)
- oneCCL bindings (`oneccl-bind-pt`)
- Intel GPU with XPU support
- PyTorch 2.0+

## Testing

```bash
# Basic functionality test
python test/test_intel_ring_flash_attn.py

# Distributed test
torchrun --nproc_per_node=4 test/test_ring_flash_attn_func.py

# MPI test
mpiexec -n 4 python test/test_mpi_ring_flash_attn.py
```

## Known Limitations

- Dropout not supported (RNG state synchronization complexity)
- Window size not supported in varlen implementations
- Requires collective operation before P2P communication with CCL backend
- Some numerical differences from standard flash attention due to bf16 accumulation

## Troubleshooting

For Intel GPU specific issues:
1. Ensure Intel GPU drivers and OneAPI are properly installed
2. Set CCL environment variables for multi-GPU setups
3. Use MPI launchers for better multi-node scaling
4. Check XPU availability with `torch.xpu.is_available()`

For more details, see the [documentation](https://github.com/zhuzilin/ring-flash-attention).