#!/bin/bash
# Run CCL test with MPI initialization

echo "🚀 Testing CCL with MPI-style initialization"
echo "==========================================="

# Set Intel GPU environment variables (matching Aurora working script)
export CCL_BACKEND=native
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=cxi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ZE_ENABLE=1
export CCL_LOG_LEVEL=info
export IPEX_XPU_ONEDNN_LAYOUT=1
export IPEX_OFFLINE_COMPILER=1

# Additional environment from test_intel_fixed.sh
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_ALLREDUCE_SCALEOUT="direct:0-1048576;rabenseifner:1048577-max"
export CCL_BCAST=double_tree
export CCL_KVS_MODE=mpi
export CCL_CONFIGURATION_PATH=""
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_KVS_CONNECTION_TIMEOUT=600
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=1024
export CCL_KVS_USE_MPI_RANKS=1
export CCL_ATL_SYNC_COLL=1
export CCL_OP_SYNC=1
export CCL_ALLREDUCE=topo
export FI_CXI_DISABLE_HOST_REGISTER=1

# Add library path for Intel Extension
export LD_LIBRARY_PATH=/lus/flare/projects/hp-ptycho/binkma/venv/infer/lib/python3.10/site-packages/intel_extension_for_pytorch/lib:$LD_LIBRARY_PATH

echo "Environment configured for Intel GPU"

# Run with mpiexec
echo -e "\nRunning with mpiexec..."
mpiexec -n 2 python test/test_ccl_mpi_style.py

echo -e "\nTest completed."