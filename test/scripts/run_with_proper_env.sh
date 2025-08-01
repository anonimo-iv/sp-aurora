#!/bin/bash
# Wrapper script to run MPI tests with proper Intel GPU environment

# Set library path for Intel Extension for PyTorch
export LD_LIBRARY_PATH=/lus/flare/projects/hp-ptycho/binkma/venv/infer/lib/python3.10/site-packages/intel_extension_for_pytorch/lib:$LD_LIBRARY_PATH

# Set OneCCL environment variables
# Remove pmix launcher since it's failing - use default (hydra)
# export CCL_PROCESS_LAUNCHER=pmix  
export CCL_ATL_TRANSPORT=mpi
export CCL_ALLREDUCE_SCALEOUT="direct:0-1048576;rabenseifner:1048577-max"
export CCL_BCAST=double_tree
export CCL_KVS_MODE=mpi
export CCL_CONFIGURATION_PATH=""
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_KVS_CONNECTION_TIMEOUT=600 
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=1024
export CCL_KVS_USE_MPI_RANKS=1
export FI_PROVIDER=cxi
export CCL_ZE_IPC_EXCHANGE=drmfd
export CCL_ZE_ENABLE=1
export CCL_LOG_LEVEL=warn  # Reduce verbosity
export IPEX_XPU_ONEDNN_LAYOUT=1
export IPEX_OFFLINE_COMPILER=1
export SYCL_CACHE_PERSISTENT=1
export SYCL_DEVICE_FILTER=level_zero:*
export SYCL_PI_LEVEL_ZERO_PROGRAM_BUILD_TRACK=2
export CCL_ATL_SYNC_COLL=1
export CCL_OP_SYNC=1
export CCL_ALLREDUCE=topo
export FI_CXI_DISABLE_HOST_REGISTER=1

# Run the command
echo "Running: mpirun -n 2 python test_intel_ulysses_attn.py"
mpirun -n 2 python test_intel_ulysses_attn.py