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

# Find oneDNN
find_path(DNNL_ROOT_DIR
    NAMES include/dnnl.h include/oneapi/dnnl/dnnl.h
    PATHS
        $ENV{DNNLROOT}
        $ENV{MKLROOT}
        /opt/intel/oneapi/dnnl/latest
        /opt/aurora/24.347.0/oneapi/dnnl/latest
)

if(DNNL_ROOT_DIR)
    set(DNNL_INCLUDE_DIR ${DNNL_ROOT_DIR}/include)
    find_library(DNNL_LIBRARY
        NAMES dnnl mkldnn
        PATHS ${DNNL_ROOT_DIR}/lib ${DNNL_ROOT_DIR}/lib/intel64
    )
    
    if(DNNL_LIBRARY)
        message(STATUS "Found oneDNN: ${DNNL_LIBRARY}")
        include_directories(${DNNL_INCLUDE_DIR})
    else()
        message(WARNING "oneDNN library not found, oneDNN kernel will not be built")
    endif()
else()
    message(WARNING "oneDNN not found, oneDNN kernel will not be built")
endif()

# Add PyTorch include directories
include_directories(${TORCH_INCLUDE_DIRS})

# Find torch_python library
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS ${TORCH_INSTALL_PREFIX}/lib)

# SYCL Flash Attention library
set(SYCL_FLASH_ATTN_SOURCES
    utils.cpp
    flash_attn_kernel_optimized_v3.cpp
    flash_attn_kernel_optimized_v3_fixed.cpp
    flash_attn_kernel_optimized_v5.cpp
    flash_attn_kernel_xmx_simple.cpp
    ring_flash_attn_kernel.cpp
)

# Add oneDNN kernel if oneDNN is found
if(DNNL_LIBRARY)
    list(APPEND SYCL_FLASH_ATTN_SOURCES flash_attn_kernel_onednn.cpp)
    add_compile_definitions(HAS_ONEDNN)
endif()

add_library(sycl_flash_attn SHARED ${SYCL_FLASH_ATTN_SOURCES})

target_compile_options(sycl_flash_attn PRIVATE -fPIC)
target_link_libraries(sycl_flash_attn PRIVATE sycl)

# Link oneDNN if found
if(DNNL_LIBRARY)
    target_link_libraries(sycl_flash_attn PRIVATE ${DNNL_LIBRARY})
endif()

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