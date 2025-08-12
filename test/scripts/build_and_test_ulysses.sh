#!/bin/bash
# Build and test script for Ulysses attention with MPI support

echo "🚀 Building and Testing Ulysses Attention with MPI Support"
echo "=========================================================="

Function to print colored output
print_status() {
    if [ "$2" = "success" ]; then
        echo -e "\033[32m✓ $1\033[0m"
    elif [ "$2" = "error" ]; then
        echo -e "\033[31m✗ $1\033[0m"
    else
        echo -e "\033[33m→ $1\033[0m"
    fi
}

# Check if we're on Aurora or a system with Intel GPU
if command -v sycl-ls &> /dev/null; then
    print_status "Intel GPU environment detected" "success"
    USE_SYCL=1
else
    print_status "No Intel GPU detected, building without SYCL support" "info"
    USE_SYCL=0
fi

# Step 1: Set up Intel oneAPI environment if available
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    print_status "Setting up Intel oneAPI environment..." "info"
    source /opt/intel/oneapi/setvars.sh
elif [ -f ~/intel/oneapi/setvars.sh ]; then
    print_status "Setting up Intel oneAPI environment from home directory..." "info"
    source ~/intel/oneapi/setvars.sh
fi

# Aurora-specific Intel GPU environment variables
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export CCL_LOG_LEVEL=info
export CCL_ZE_ENABLE=1
export CCL_KVS_USE_MPI_RANKS=1
export CCL_ATL_SYNC_COLL=1
export CCL_OP_SYNC=1

# Additional environment variables for Intel GPU
export FI_PROVIDER=cxi
export CCL_ZE_IPC_EXCHANGE=drmfd
export IPEX_XPU_ONEDNN_LAYOUT=1
export IPEX_OFFLINE_COMPILER=1
export SYCL_CACHE_PERSISTENT=1
export SYCL_DEVICE_FILTER=level_zero:*
export MPIR_CVAR_ENABLE_GPU=1

print_status "Environment variables configured" "success"

# # Step 2: Build SYCL components if available
# if [ "$USE_SYCL" = "1" ] && [ -f sp_aurora/sycl/CMakeLists.txt ]; then
#     print_status "Building SYCL components..." "info"
    
#     cd sp_aurora/sycl
    
#     # Clean previous build
#     rm -rf build
#     mkdir -p build
#     cd build
    
#     # Configure with CMake
#     if command -v icpx &> /dev/null; then
#         cmake .. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
#         if [ $? -eq 0 ]; then
#             print_status "CMake configuration successful" "success"
            
#             # Build
#             make -j$(nproc)
#             if [ $? -eq 0 ]; then
#                 print_status "SYCL build successful" "success"
#             else
#                 print_status "SYCL build failed" "error"
#                 USE_SYCL=0
#             fi
#         else
#             print_status "CMake configuration failed" "error"
#             USE_SYCL=0
#         fi
#     else
#         print_status "Intel DPC++ compiler (icpx) not found" "error"
#         USE_SYCL=0
#     fi
    
#     cd ../../..  # Back to project root
# fi

# # Step 3: Install Python package
# print_status "Installing Ring-FT Python package..." "info"

# if [ "$USE_SYCL" = "1" ]; then
#     BUILD_SYCL=1 pip install -e . --no-deps
# else
#     pip install -e . --no-deps
# fi

# if [ $? -eq 0 ]; then
#     print_status "Python package installation successful" "success"
# else
#     print_status "Python package installation failed" "error"
#     exit 1
# fi

# # Step 4: Verify installation
# print_status "Verifying installation..." "info"

# python -c "import sp_aurora; print('Ring Flash Attention imported successfully')"
# if [ $? -ne 0 ]; then
#     print_status "Failed to import sp_aurora" "error"
#     exit 1
# fi

# if [ "$USE_SYCL" = "1" ]; then
#     python -c "from sp_aurora import is_sycl_available; print(f'SYCL available: {is_sycl_available()}')"
# fi

# print_status "Installation verified" "success"

echo ""
echo "=========================================================="
echo "Running Ulysses Attention Tests"
echo "=========================================================="

# Function to run tests
run_test() {
    local test_file=$1
    local test_name=$2
    local num_procs=$3
    
    echo ""
    print_status "Running $test_name..." "info"
    
    if [ $num_procs -eq 1 ]; then
        # Single process test
        python $test_file
    else
        # Multi-process test with MPI
        if command -v mpiexec &> /dev/null; then
            mpiexec -n $num_procs python $test_file
        elif command -v mpirun &> /dev/null; then
            mpirun -n $num_procs python $test_file
        else
            print_status "MPI not found, running with torchrun instead" "info"
            torchrun --nproc_per_node=$num_procs $test_file
        fi
    fi
    
    if [ $? -eq 0 ]; then
        print_status "$test_name passed" "success"
        return 0
    else
        print_status "$test_name failed" "error"
        return 1
    fi
}

# Step 5: Run tests
cd test  # Change to test directory

# Test 1: Single process tests
print_status "Testing single process execution..." "info"
run_test "test_intel_ulysses_attn.py" "Intel Ulysses Attention (single process)" 1
SINGLE_TEST1=$?

run_test "test_ulysses_reorganized.py" "Reorganized Ulysses Attention (single process)" 1
SINGLE_TEST2=$?

# Test 2: Multi-process tests (if MPI is available)
if command -v mpiexec &> /dev/null || command -v mpirun &> /dev/null || command -v torchrun &> /dev/null; then
    print_status "Testing multi-process execution..." "info"
    
    # Determine number of processes based on available GPUs
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    elif command -v xpu-smi &> /dev/null; then
        NUM_GPUS=$(xpu-smi discovery | grep "Device ID" | wc -l)
    else
        NUM_GPUS=2  # Default to 2 processes
    fi
    
    # Cap at 4 for testing
    NUM_PROCS=$(( NUM_GPUS < 4 ? NUM_GPUS : 4 ))
    
    if [ $NUM_PROCS -gt 1 ]; then
        print_status "Running distributed tests with $NUM_PROCS processes..." "info"
        
        run_test "test_intel_ulysses_attn.py" "Intel Ulysses Attention (distributed)" $NUM_PROCS
        DIST_TEST1=$?
        
        run_test "test_ulysses_reorganized.py" "Reorganized Ulysses Attention (distributed)" $NUM_PROCS
        DIST_TEST2=$?
    else
        print_status "Only 1 GPU available, skipping distributed tests" "info"
        DIST_TEST1=0
        DIST_TEST2=0
    fi
else
    print_status "No MPI or torchrun found, skipping distributed tests" "info"
    DIST_TEST1=0
    DIST_TEST2=0
fi

cd ..  # Back to project root

# Step 6: Summary
echo ""
echo "=========================================================="
echo "Test Summary"
echo "=========================================================="

if [ $SINGLE_TEST1 -eq 0 ] && [ $SINGLE_TEST2 -eq 0 ] && [ $DIST_TEST1 -eq 0 ] && [ $DIST_TEST2 -eq 0 ]; then
    print_status "All tests passed successfully!" "success"
    echo ""
    echo "You can now run the tests manually with:"
    echo "  Single process: python test/test_intel_ulysses_attn.py"
    echo "  With MPI:       mpiexec -n 2 python test/test_intel_ulysses_attn.py"
    echo "  With torchrun:  torchrun --nproc_per_node=2 test/test_intel_ulysses_attn.py"
    exit 0
else
    print_status "Some tests failed" "error"
    echo ""
    echo "Failed tests:"
    [ $SINGLE_TEST1 -ne 0 ] && echo "  - Intel Ulysses Attention (single process)"
    [ $SINGLE_TEST2 -ne 0 ] && echo "  - Reorganized Ulysses Attention (single process)"
    [ $DIST_TEST1 -ne 0 ] && echo "  - Intel Ulysses Attention (distributed)"
    [ $DIST_TEST2 -ne 0 ] && echo "  - Reorganized Ulysses Attention (distributed)"
    exit 1
fi