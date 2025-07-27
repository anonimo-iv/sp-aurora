cmake_minimum_required(VERSION 3.20)
project(SyclFlashAttention LANGUAGES CXX)

# Find Intel oneAPI DPC++ compiler
find_package(IntelSYCL REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SYCL specific flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=spir64_gen")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xs \"-device pvc\"")  # For Intel Data Center GPU Max

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-finite-math-only")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ftree-vectorize")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -finline-functions")
# Intel GPU specific optimizations
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fsycl-force-inline-kernel-lambda")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fsycl-id-queries-fit-in-int")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
# find_package(pybind11 REQUIRED)
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})

# Find PyTorch
execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH};${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)

# Add PyTorch include directories
include_directories(${TORCH_INCLUDE_DIRS})

# Find torch_python library
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS ${TORCH_INSTALL_PREFIX}/lib)

# SYCL Flash Attention library
add_library(sycl_flash_attn SHARED
    utils.cpp
    flash_attn_kernel_optimized_v3.cpp
    flash_attn_kernel_optimized_v3_fixed.cpp
    flash_attn_kernel_optimized_v4.cpp
    flash_attn_kernel_optimized_v5.cpp
    flash_attn_kernel_optimized_v7.cpp
    flash_attn_kernel_optimized_v8.cpp
    flash_attn_kernel_xmx_simple.cpp
    ring_flash_attn_kernel.cpp
)

target_compile_options(sycl_flash_attn PRIVATE -fPIC)
target_link_libraries(sycl_flash_attn PRIVATE sycl)

# Python bindings
pybind11_add_module(sycl_flash_attn_py ../sycl_bindings/bindings.cpp)
target_link_libraries(sycl_flash_attn_py PRIVATE sycl_flash_attn ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_compile_features(sycl_flash_attn_py PRIVATE cxx_std_17)
# Add torch extension compile definitions
target_compile_definitions(sycl_flash_attn_py PRIVATE TORCH_EXTENSION_NAME=sycl_flash_attn)
target_compile_options(sycl_flash_attn_py PRIVATE ${TORCH_CXX_FLAGS}) 

# Set properties for Python module
set_target_properties(sycl_flash_attn_py PROPERTIES
    OUTPUT_NAME "sycl_flash_attn"
)
# Only set output directory if not already provided by setup.py
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(sycl_flash_attn_py PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../"
    )
endif()

# Installation
install(TARGETS sycl_flash_attn_py
    LIBRARY DESTINATION ${Python_SITELIB}/ring_flash_attn
)