#!/bin/bash
# Run ring communication test with mpirun instead of torchrun

echo "🚀 Testing Ring Communication with mpirun"
echo "======================================="

# Set Intel GPU environment variables
export CCL_PROCESS_LAUNCHER=pmix  
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export CCL_LOG_LEVEL=info
export CCL_ZE_ENABLE=1

# Add Intel Extension library path
export LD_LIBRARY_PATH=/lus/flare/projects/hp-ptycho/binkma/venv/infer/lib/python3.10/site-packages/intel_extension_for_pytorch/lib:$LD_LIBRARY_PATH

echo "Running with mpirun..."
mpirun -np 2 python test_ring_comm_simple.py

echo "Test completed."