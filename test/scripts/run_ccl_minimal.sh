#!/bin/bash
# Minimal CCL setup to avoid hangs

# Set library path for Intel Extension for PyTorch
export LD_LIBRARY_PATH=/lus/flare/projects/hp-ptycho/binkma/venv/infer/lib/python3.10/site-packages/intel_extension_for_pytorch/lib:$LD_LIBRARY_PATH

# Minimal CCL settings - avoid problematic ones
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export FI_PROVIDER=cxi

# Disable CCL optimizations that might cause hangs
export CCL_ALLREDUCE=ring  # Use simple ring algorithm
export CCL_ALLTOALL=naive   # Use naive implementation
export CCL_ATL_SYNC_COLL=0  # Disable sync collectives
export CCL_OP_SYNC=0        # Disable operation sync

# Use default process launcher (not pmix)
unset CCL_PROCESS_LAUNCHER

# Run with mpirun
echo "Running with minimal CCL configuration..."
mpirun -n 2 python test_intel_ulysses_attn.py