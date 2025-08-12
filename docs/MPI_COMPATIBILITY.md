# MPI Compatibility for Ring Flash Attention

This document describes the MPI compatibility features added to Ring Flash Attention, enabling the use of `mpiexec` alongside the existing `torchrun` launcher.

## Overview

The Ring Flash Attention library now supports both `torchrun` and `mpiexec` for distributed training. This enhancement provides:

- **HPC Integration**: Works seamlessly with HPC job schedulers (Slurm, PBS, LSF)
- **Multi-node Scaling**: Native support for multi-node distributed training
- **Intel GPU Optimization**: Enhanced support for Intel GPU clusters with oneCCL
- **Backward Compatibility**: Existing `torchrun` workflows remain unchanged

## Quick Start

### Basic Usage

```python
from sp_aurora import setup_mpi_distributed, sp_aurora_func

# Setup distributed environment (works with both torchrun and mpiexec)
setup_info = setup_mpi_distributed()

rank = setup_info['rank']
world_size = setup_info['world_size']
device = setup_info['device']

# Your existing Ring Flash Attention code works unchanged
output = sp_aurora_func(q, k, v, causal=True)
```

### Launching with Different Methods

```bash
# Method 1: torchrun (existing)
torchrun --nproc_per_node=4 your_script.py

# Method 2: mpiexec (new)
mpiexec -n 4 python your_script.py

# Method 3: Multi-node with mpiexec
mpiexec -n 8 -hostfile hosts python your_script.py

# Method 4: Slurm with MPI
srun --mpi=pmix -n 8 python your_script.py
```

## Installation Requirements

### Basic MPI Support

For basic MPI functionality, you need an MPI implementation:

```bash
# Ubuntu/Debian
sudo apt-get install openmpi-bin libopenmpi-dev

# RHEL/CentOS/Rocky
sudo yum install openmpi openmpi-devel

# Conda
conda install -c conda-forge openmpi
```

### Intel GPU Support

For Intel GPU clusters, install Intel MPI and oneCCL:

```bash
# Intel oneAPI (includes Intel MPI and oneCCL)
# Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html

# Source the environment
source /opt/intel/oneapi/setvars.sh

# Install Intel Extension for PyTorch
pip install intel_extension_for_pytorch
pip install oneccl_bindings_for_pytorch
```

## Environment Variables

### MPI Detection

The library automatically detects MPI environments by checking these variables:

- `OMPI_COMM_WORLD_RANK` (Open MPI)
- `PMI_RANK` (Intel MPI, Slurm)
- `PMIX_RANK` (PMIx-based launchers)
- `MV2_COMM_WORLD_RANK` (MVAPICH2)
- `MPI_RANK` (Generic MPI)

### Manual Configuration

You can manually set distributed parameters:

```bash
export MASTER_ADDR=node001
export MASTER_PORT=29500
mpiexec -n 4 python your_script.py
```

### Intel GPU Optimization

For Intel GPU clusters, set these environment variables:

```bash
export CCL_BACKEND=native
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=cxi
export CCL_ZE_IPC_EXCHANGE=drmfd
export CCL_ZE_ENABLE=1
```

## API Reference

### Core Functions

#### `setup_mpi_distributed(backend=None)`

Sets up distributed training with automatic MPI/torchrun detection.

**Parameters:**
- `backend` (str, optional): Distributed backend ('nccl', 'ccl', 'gloo', or None for auto-detection)

**Returns:**
- Dictionary with setup information:
  ```python
  {
      'rank': int,           # Process rank
      'world_size': int,     # Total number of processes
      'local_rank': int,     # Local rank on current node
      'device': torch.device,# Assigned device
      'launcher': str,       # Launcher type ('torchrun', 'mpiexec', 'single')
      'backend': str         # Distributed backend being used
  }
  ```

#### `init_distributed_backend(backend=None, timeout_seconds=1800)`

Initializes distributed backend with fallback support.

**Parameters:**
- `backend` (str, optional): Specific backend to use
- `timeout_seconds` (int): Timeout for initialization

**Returns:**
- `bool`: True if successful, False otherwise

#### `cleanup_distributed()`

Cleans up distributed process group.

#### `get_device_for_rank(rank=None)`

Gets the appropriate device for a given rank.

**Parameters:**
- `rank` (int, optional): Process rank (uses current rank if None)

**Returns:**
- `torch.device`: Device to use

#### `detect_mpi_environment()`

Detects if running under MPI environment.

**Returns:**
- `bool`: True if MPI detected, False otherwise

## Examples

### Example 1: Basic Integration

```python
#!/usr/bin/env python3
import torch
from sp_aurora import setup_mpi_distributed, sp_aurora_func, cleanup_distributed

def main():
    # Setup distributed (works with both torchrun and mpiexec)
    setup_info = setup_mpi_distributed()
    
    rank = setup_info['rank']
    world_size = setup_info['world_size']
    device = setup_info['device']
    
    print(f"Rank {rank}/{world_size} on device {device}")
    
    # Create sample data
    batch_size = 1
    seq_len_per_rank = 512
    nheads = 8
    d = 64
    
    q = torch.randn(batch_size, seq_len_per_rank, nheads, d, device=device)
    k = torch.randn(batch_size, seq_len_per_rank, nheads, d, device=device)
    v = torch.randn(batch_size, seq_len_per_rank, nheads, d, device=device)
    
    # Ring attention (handles distributed communication automatically)
    output = sp_aurora_func(q, k, v, causal=True)
    
    print(f"Rank {rank}: Output shape {output.shape}")
    
    # Cleanup
    cleanup_distributed()

if __name__ == "__main__":
    main()
```

### Example 2: HPC Job Script

```bash
#!/bin/bash
#SBATCH --job-name=ring-flash-attn
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00

# Load modules
module load openmpi/4.1.0
module load cuda/11.8

# Set up environment
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Run with srun (uses PMI)
srun --mpi=pmix python your_training_script.py

# Alternative: Run with mpiexec
# mpiexec -n $SLURM_NTASKS python your_training_script.py
```

### Example 3: Intel GPU Cluster

```bash
#!/bin/bash
# Intel GPU cluster script

# Source Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Set Intel GPU optimizations
export CCL_BACKEND=native
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=cxi
export CCL_ZE_IPC_EXCHANGE=drmfd
export CCL_ZE_ENABLE=1

# Run with Intel MPI
mpiexec -n 8 -ppn 2 python your_script.py
```

## Backend Selection

The library automatically selects the best backend:

1. **Intel GPU**: CCL backend (with oneCCL fallback to Gloo)
2. **NVIDIA GPU**: NCCL backend (with fallback to Gloo)
3. **CPU**: Gloo backend

You can override backend selection:

```python
# Force specific backend
setup_info = setup_mpi_distributed(backend='nccl')

# Or use the lower-level function
from sp_aurora import init_distributed_backend
success = init_distributed_backend(backend='ccl')
```

## Troubleshooting

### Common Issues

#### 1. MPI Not Detected

**Problem**: Script runs as single process even with `mpiexec`

**Solution**: Check MPI environment variables:
```bash
mpiexec -n 2 env | grep -E "(OMPI|PMI|MPI)_"
```

#### 2. Backend Initialization Fails

**Problem**: CCL or NCCL backend fails to initialize

**Solution**: The library automatically falls back to Gloo. Check logs for backend selection.

#### 3. Multi-node Communication Issues

**Problem**: Hangs during distributed initialization

**Solution**: Ensure `MASTER_ADDR` is accessible from all nodes:
```bash
export MASTER_ADDR=node001.cluster.local
mpiexec -n 8 -hostfile hosts python script.py
```

#### 4. Intel GPU Issues

**Problem**: Intel GPU not detected or performance issues

**Solution**: Verify Intel GPU setup:
```bash
# Check Intel GPU availability
python -c "import torch; import intel_extension_for_pytorch; print(torch.xpu.is_available())"

# Check oneCCL
python -c "import oneccl_bindings_for_pytorch; print('oneCCL available')"
```

### Debugging

Enable verbose logging:

```python
import os
os.environ['CCL_LOG_LEVEL'] = 'info'  # For Intel GPU
os.environ['NCCL_DEBUG'] = 'INFO'     # For NVIDIA GPU

# Your distributed code here
```

## Performance Considerations

### Memory Usage

MPI-launched processes may have different memory behavior compared to `torchrun`. Monitor memory usage:

```python
import torch

if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    print(f"XPU memory: {torch.xpu.max_memory_allocated() / 1e9:.2f} GB")
```

### Network Optimization

For multi-node setups, consider:

1. **InfiniBand**: Ensure proper IB configuration for NCCL
2. **Intel Omni-Path**: Use appropriate CCL transport settings
3. **Ethernet**: May require tuning for large-scale deployments

### Scaling Guidelines

- **Single Node**: Both `torchrun` and `mpiexec` perform similarly
- **Multi-Node**: `mpiexec` often provides better job scheduler integration
- **Large Scale**: Consider specialized MPI implementations (Intel MPI, Spectrum MPI)

## Migration Guide

### From torchrun to mpiexec

Minimal changes required:

```python
# Before (torchrun only)
import torch.distributed as dist
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
device = torch.device(f"cuda:{rank}")

# After (both torchrun and mpiexec)
from sp_aurora import setup_mpi_distributed
setup_info = setup_mpi_distributed()
rank = setup_info['rank']
device = setup_info['device']
```

### Launcher Commands

```bash
# Old
torchrun --nproc_per_node=4 script.py

# New (both work)
torchrun --nproc_per_node=4 script.py  # Still works
mpiexec -n 4 python script.py          # New option
```

## Contributing

When contributing to MPI compatibility features:

1. Test with both `torchrun` and `mpiexec`
2. Verify multi-node functionality
3. Test with different MPI implementations
4. Ensure Intel GPU compatibility
5. Update documentation and examples

## License

MPI compatibility features follow the same license as the main Ring Flash Attention project.