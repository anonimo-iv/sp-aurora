#!/bin/bash

echo "========================================"
echo "Testing MPI support for Ulysses tests"
echo "========================================"
echo ""

# Test 1: test_intel_ulysses_attn.py with 2 processes
echo "1. Testing test_intel_ulysses_attn.py with mpiexec -n 2"
echo "Command: mpiexec -n 2 python test/test_intel_ulysses_attn.py"
echo ""

# Test 2: test_ulysses_reorganized.py with 2 processes 
echo "2. Testing test_ulysses_reorganized.py with mpiexec -n 2"
echo "Command: mpiexec -n 2 python test/test_ulysses_reorganized.py"
echo ""

echo "========================================"
echo "Example usage with Intel MPI on Intel GPU:"
echo ""
echo "mpiexec -n 2 -genv CCL_BACKEND=native -genv CCL_ATL_TRANSPORT=ofi \\"
echo "    python test/test_intel_ulysses_attn.py"
echo ""
echo "========================================"
echo "Example usage with torchrun:"
echo ""
echo "torchrun --nproc_per_node=2 test/test_intel_ulysses_attn.py"
echo "========================================"