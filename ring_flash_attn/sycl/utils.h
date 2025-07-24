#ifndef SYCL_FLASH_ATTN_UTILS_H
#define SYCL_FLASH_ATTN_UTILS_H

#include <sycl/sycl.hpp>
#include <cmath>
#include <limits>

namespace flash_attn {

// Type definitions
using namespace sycl;
using float16 = sycl::half;

// Constants
constexpr int WARP_SIZE = 16;  // Intel GPU subgroup size
constexpr int DEFAULT_BLOCK_SIZE = 128;
constexpr float EPSILON = 1e-10f;

// Helper functions
template<typename T>
inline T divUp(T a, T b) {
    return (a + b - 1) / b;
}

// Load data from global memory to local memory with coalescing
template<typename T, int TILE_SIZE>
inline void loadTile(const T* __restrict__ global_ptr,
                     T* __restrict__ local_ptr,
                     int row_offset,
                     int col_offset,
                     int rows,
                     int cols,
                     int stride,
                     const nd_item<2>& item) {
    int local_id = item.get_local_id(1);
    int group_size = item.get_local_range(1);
    
    #pragma unroll
    for (int i = local_id; i < TILE_SIZE * TILE_SIZE; i += group_size) {
        int tile_row = i / TILE_SIZE;
        int tile_col = i % TILE_SIZE;
        int global_row = row_offset + tile_row;
        int global_col = col_offset + tile_col;
        
        if (global_row < rows && global_col < cols) {
            local_ptr[tile_row * TILE_SIZE + tile_col] = 
                global_ptr[global_row * stride + global_col];
        } else {
            local_ptr[tile_row * TILE_SIZE + tile_col] = T(0);
        }
    }
}

// Compute exponential with numerical stability
template<typename T>
inline T stableExp(T x, T max_val) {
    return sycl::exp(x - max_val);
}

// Warp-level reduction for max
template<typename T>
inline T warpReduceMax(T val, const nd_item<2>& item) {
    auto sg = item.get_sub_group();
    return reduce_over_group(sg, val, maximum<T>());
}

// Warp-level reduction for sum
template<typename T>
inline T warpReduceSum(T val, const nd_item<2>& item) {
    auto sg = item.get_sub_group();
    return reduce_over_group(sg, val, plus<T>());
}

// Block-level reduction for max
template<typename T, int BLOCK_SIZE>
inline T blockReduceMax(T val, T* shared, const nd_item<2>& item) {
    int tid = item.get_local_id(1);
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    // Reduce within warp
    val = warpReduceMax(val, item);
    
    // Write warp result to shared memory
    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    item.barrier();
    
    // Final reduction
    if (tid < num_warps) {
        val = shared[tid];
    } else {
        val = -std::numeric_limits<T>::infinity();
    }
    
    if (warp_id == 0) {
        val = warpReduceMax(val, item);
    }
    
    return val;
}

// Block-level reduction for sum
template<typename T, int BLOCK_SIZE>
inline T blockReduceSum(T val, T* shared, const nd_item<2>& item) {
    int tid = item.get_local_id(1);
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    // Reduce within warp
    val = warpReduceSum(val, item);
    
    // Write warp result to shared memory
    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    item.barrier();
    
    // Final reduction
    if (tid < num_warps) {
        val = shared[tid];
    } else {
        val = T(0);
    }
    
    if (warp_id == 0) {
        val = warpReduceSum(val, item);
    }
    
    return val;
}

}  // namespace flash_attn

#endif  // SYCL_FLASH_ATTN_UTILS_H