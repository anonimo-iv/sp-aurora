#!/bin/bash
# Build script for SYCL Flash Attention

# Source Intel oneAPI environment if available
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh
elif [ -n "$ONEAPI_ROOT" ]; then
    source $ONEAPI_ROOT/setvars.sh
else
    echo "Warning: Intel oneAPI environment not found. Make sure icpx is in PATH."
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=$(which python)

# Build
make -j$(nproc)

echo "Build complete. To install, run:"
echo "cd ../.. && BUILD_SYCL=1 pip install -e ."