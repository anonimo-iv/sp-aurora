#!/bin/bash
# Test script for XPU all-to-all minimal test

echo "🚀 Testing XPU All-to-All Behavior"
echo "=============================================="

# Aurora-specific Intel GPU environment variables (from test_ring.sh)
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

echo "Environment variables set for Aurora Intel GPU"
echo ""

# Make the test script executable
chmod +x test_xpu_alltoall_minimal.py

# Test with different configurations
echo "=== Test 1: CCL backend (default) ==="
mpirun -n 2 python test_xpu_alltoall_minimal.py

echo -e "\n=== Test 2: With PyTorch XPU fallback ==="
PYTORCH_ENABLE_XPU_FALLBACK=1 mpirun -n 2 python test_xpu_alltoall_minimal.py

echo -e "\n=== Test 3: Force gloo backend ==="
FORCE_BACKEND=gloo mpirun -n 2 python test_xpu_alltoall_minimal.py

# Check exit code
if [ $? -eq 0 ]; then
    echo -e "\n✅ Tests completed!"
else
    echo -e "\n❌ Tests failed!"
    exit 1
fi