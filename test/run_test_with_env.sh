#!/bin/bash
# Run the test with proper environment setup

export PYTHONPATH=/home/binkma/bm_dif/Ring-FT:$PYTHONPATH

echo "Running Intel Ring Flash Attention test..."
echo "PYTHONPATH=$PYTHONPATH"

# Run with timeout to prevent hanging
timeout 200s torchrun --nproc_per_node=2 test_intel_ring_flash_attn.py

exit_code=$?
if [ $exit_code -eq 124 ]; then
    echo "Test timed out after 60 seconds"
elif [ $exit_code -eq 0 ]; then
    echo "Test completed successfully"
else
    echo "Test failed with exit code: $exit_code"
fi

exit $exit_code