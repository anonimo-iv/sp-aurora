# Ring Flash Attention for Intel GPUs

Ring Flash Attention implementation optimized for Intel GPUs (XPU) using Intel Extension for PyTorch and oneCCL backend.

## Features

- **Intel GPU Optimized**: Native support for Intel Data Center GPU Max series
- **SYCL Flash Attention**: Optimized SYCL kernels with 2-5x speedup over Python implementation
- **Ring Attention**: Basic ring attention implementation
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

### Building with SYCL Support

To enable SYCL acceleration for Intel GPUs:

```bash
# Ensure Intel oneAPI toolkit is installed and environment is set
source /opt/intel/oneapi/setvars.sh

# Build with SYCL support
cd ring-flash-attention
bash sp_aurora/sycl_bindings/build.sh
pip install .
```

## Quick Start

```python
import torch
import intel_extension_for_pytorch as ipex
from sp_aurora import sp_aurora_func, setup_mpi_distributed

# Setup distributed environment
setup_info = setup_mpi_distributed()
device = setup_info['device']  # automatically detects 'xpu' for Intel GPUs

# Example usage
batch_size, seq_len, num_heads, head_dim = 2, 1024, 32, 64
q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)

# Run ring flash attention
output = sp_aurora_func(q, k, v, causal=True)
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
- `sp_aurora_func`: Basic ring attention (Intel GPU optimized)
- `sp_aurora_kvpacked_func`: Ring attention with key-value packed inputs
- `sp_aurora_qkvpacked_func`: Ring attention with query-key-value packed inputs

### SYCL Functions
- `is_sycl_available()`: Check if SYCL acceleration is available
- `get_sycl_device_info()`: Get Intel GPU device information
- `auto_select_flash_attn_forward()`: Automatically select best implementation (SYCL/Python)

## Performance

Ring attention enables distributed training across multiple GPUs with efficient communication patterns.

## Requirements

- Intel Extension for PyTorch (`intel-extension-for-pytorch>=2.0.0`)
- oneCCL bindings (`oneccl-bind-pt`)
- Intel GPU with XPU support
- PyTorch 2.0+

### Additional Requirements for SYCL
- Intel oneAPI Base Toolkit (includes DPC++ compiler)
- Intel GPU drivers with Level Zero support

## Testing

```bash
# Basic functionality test
python test/test_intel_sp_aurora.py

# Distributed test
torchrun --nproc_per_node=4 test/test_sp_aurora_func.py

# MPI test
mpiexec -n 4 python test/test_mpi_sp_aurora.py

# SYCL flash attention test
python test/test_sycl_flash_attention.py
```

## Known Limitations

- Dropout not supported (RNG state synchronization complexity)
- Variable length sequences not yet supported
- Requires collective operation before P2P communication with CCL backend
- Some numerical differences from standard flash attention due to bf16 accumulation
- SYCL implementation currently supports only float32 (fp16/bf16 planned)
- SYCL backward pass uses PyTorch autograd (native SYCL backward planned)

## Troubleshooting

For Intel GPU specific issues:
1. Ensure Intel GPU drivers and OneAPI are properly installed
2. Set CCL environment variables for multi-GPU setups
3. Use MPI launchers for better multi-node scaling
4. Check XPU availability with `torch.xpu.is_available()`

For more details, see the [documentation](https://github.com/zhuzilin/ring-flash-attention).