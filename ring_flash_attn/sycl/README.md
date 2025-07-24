# SYCL Flash Attention for Intel GPUs

This directory contains optimized SYCL kernels for Flash Attention on Intel GPUs.

## Features

- **Fused Attention + LSE**: Computes attention and log-sum-exp in a single kernel
- **Tiled Computation**: Memory-efficient processing with configurable block sizes
- **Intel GPU Optimizations**: 
  - Utilizes shared local memory (SLM)
  - Optimized for Intel GPU EU architecture
  - Coalesced memory access patterns
- **Python Integration**: Seamless integration with existing Ring Flash Attention

## Building

### Prerequisites

1. Intel oneAPI Base Toolkit (for DPC++ compiler)
2. Intel GPU with Level Zero driver
3. CMake 3.20+
4. Python 3.7+ with PyTorch and Intel Extension for PyTorch
5. pybind11

### Build Instructions

1. Source Intel oneAPI environment:
   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```

2. Build using the provided script:
   ```bash
   cd ring_flash_attn/sycl
   ./build.sh
   ```

3. Or build manually:
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_CXX_COMPILER=icpx
   make -j
   ```

4. Install the Python module:
   ```bash
   cd ../..  # Back to repo root
   BUILD_SYCL=1 pip install -e .
   ```

## Usage

The SYCL implementation is automatically used when available:

```python
from ring_flash_attn.intel_flash_attn_sycl import (
    is_sycl_available,
    intel_flash_attn_forward_sycl
)

# Check if SYCL is available
if is_sycl_available():
    print("Using SYCL Flash Attention")

# Use like regular flash attention
output, lse = intel_flash_attn_forward_sycl(
    q, k, v,
    causal=True,
    softmax_scale=1.0 / math.sqrt(head_dim)
)
```

## Performance

Expected performance improvements over Python implementation:
- 2-5x speedup for attention computation
- Reduced memory bandwidth usage
- Better scaling with sequence length

## Architecture

### Kernel Design

1. **Tiled Processing**: Processes Q, K, V in tiles to fit in shared memory
2. **Online Softmax**: Computes softmax on-the-fly with numerical stability
3. **Warp-Level Operations**: Uses Intel GPU subgroup operations for reductions
4. **Memory Coalescing**: Optimized memory access patterns

### File Structure

- `flash_attn_kernel.h/cpp`: Core flash attention kernel
- `flash_attn_kernel_optimized.cpp`: Optimized version with auto-tuning
- `ring_flash_attn_kernel.cpp`: Ring-aware kernel (optional)
- `utils.h`: SYCL utilities and helper functions
- `CMakeLists.txt`: Build configuration
- `bindings.cpp`: Python bindings

## Limitations

Current limitations:
- Only supports float32 (fp16 support planned)
- No ALiBi or local window attention yet
- Backward pass uses autograd (native backward planned)

## Troubleshooting

1. **Build fails with "icpx not found"**:
   - Ensure Intel oneAPI is installed and sourced
   - Check that `which icpx` returns a valid path

2. **Runtime error "No Intel GPU found"**:
   - Check Intel GPU is present: `sycl-ls`
   - Ensure Level Zero driver is installed

3. **Performance not improved**:
   - Check you're using XPU device in PyTorch
   - Verify SYCL kernel is being called (not fallback)
   - Profile with Intel VTune for bottlenecks