#!/bin/bash
# Run ring communication test with MPI-style initialization

echo "🚀 Testing Ring Communication with MPI-style initialization"
echo "=========================================================="

# Set Intel GPU environment variables
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export CCL_LOG_LEVEL=info
export CCL_ZE_ENABLE=1
export CCL_KVS_USE_MPI_RANKS=1
export CCL_ATL_SYNC_COLL=1
export CCL_OP_SYNC=1

# Additional environment variables
export FI_PROVIDER=cxi
export CCL_ZE_IPC_EXCHANGE=drmfd
export IPEX_XPU_ONEDNN_LAYOUT=1
export IPEX_OFFLINE_COMPILER=1
export SYCL_CACHE_PERSISTENT=1
export SYCL_DEVICE_FILTER=level_zero:*
export MPIR_CVAR_ENABLE_GPU=1

# Add Intel Extension library path
export LD_LIBRARY_PATH=/lus/flare/projects/hp-ptycho/binkma/venv/infer/lib/python3.10/site-packages/intel_extension_for_pytorch/lib:$LD_LIBRARY_PATH

# Preload MPI libraries to resolve symbol issues
# export LD_PRELOAD=$MPICH_DIR/lib/libmpi.so:$MPICH_DIR/lib/libmpifort.so:$LD_PRELOAD

echo "Running with mpirun..."
mpirun -np 4 python run_ddp_ccl.py 4

echo "Test completed."