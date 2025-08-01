#!/bin/bash
# Test script for Ulysses attention with MPI on Intel GPU

echo "🚀 Testing Ulysses Attention with MPI"
echo "=============================================="

# Aurora-specific Intel GPU environment variables
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

# Set Intel oneAPI environment if available
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    echo "Setting up Intel oneAPI environment..."
    source /opt/intel/oneapi/setvars.sh
fi

echo "Environment variables set for Intel GPU"
echo ""

# Function to run test and capture result
run_test() {
    local test_file=$1
    local test_name=$2
    local num_procs=$3
    
    echo "=============================================="
    echo "Testing: $test_name"
    echo "Command: mpirun -n $num_procs python $test_file"
    echo "=============================================="
    
    mpirun -n $num_procs python $test_file
    
    if [ $? -eq 0 ]; then
        echo "✅ $test_name completed successfully!"
        return 0
    else
        echo "❌ $test_name failed!"
        return 1
    fi
}

# Change to the script directory
cd "$(dirname "$0")"

# Test 1: Intel Ulysses Attention
echo "Test 1: Intel Ulysses Attention (2 processes)"
run_test "/home/binkma/bm_dif/Ring-FT/test/verification/test_ulysses_fixed.py" "Intel Ulysses Attention" 2
TEST1_RESULT=$?

echo ""

# # Test 2: Simple Ulysses Test
# echo "Test 2: Simple Ulysses Test (2 processes)"
# run_test "./test_ulysses_simple.py" "Simple Ulysses Test" 2
# TEST2_RESULT=$?

# echo ""

# # Test 3: Intel AllToAll 4D Test
# echo "Test 3: Intel AllToAll 4D Test (2 processes)"
# run_test "./test_intel_alltoall_4d.py" "Intel AllToAll 4D" 2
# TEST3_RESULT=$?

# echo ""

# # Test with 4 processes if available
# if [ -z "$SKIP_4PROC" ]; then
#     echo "Test 4: Intel Ulysses Attention (4 processes)"
#     run_test "./test_intel_ulysses_attn.py" "Intel Ulysses Attention (4 proc)" 4
#     TEST4_RESULT=$?
# else
#     TEST4_RESULT=0
# fi

# echo ""
# echo "=============================================="
# echo "Test Summary"
# echo "=============================================="

# if [ $TEST1_RESULT -eq 0 ] && [ $TEST2_RESULT -eq 0 ] && [ $TEST3_RESULT -eq 0 ] && [ $TEST4_RESULT -eq 0 ]; then
#     echo "✅ All tests passed!"
#     exit 0
# else
#     echo "❌ Some tests failed:"
#     [ $TEST1_RESULT -ne 0 ] && echo "  - Intel Ulysses Attention (2 processes)"
#     [ $TEST2_RESULT -ne 0 ] && echo "  - Simple Ulysses Test (2 processes)"
#     [ $TEST3_RESULT -ne 0 ] && echo "  - Intel AllToAll 4D (2 processes)"
#     [ $TEST4_RESULT -ne 0 ] && echo "  - Intel Ulysses Attention (4 processes)"
#     exit 1
# fi

# Since only Test 1 is active, check its result
if [ $TEST1_RESULT -eq 0 ]; then
    echo "✅ Test passed!"
    exit 0
else
    echo "❌ Test failed!"
    exit 1
fi