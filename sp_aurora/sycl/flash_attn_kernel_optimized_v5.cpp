#include "flash_attn_kernel.h"
#include "utils.h"
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <sycl/sycl.hpp>

namespace flash_attn {

// Optimized V5 kernel - Non-causal optimized for DiT inference
// Key optimizations:
// 1. Remove all causal masking branches
// 2. Optimized tiling for sequences 128-512  
// 3. Intel XMX-friendly 16x16 tile operations
// 4. Reduced shared memory pressure
// 5. Better thread utilization for Intel GPU

template<int BLOCK_M, int BLOCK_N, int HEAD_DIM, int THREADS, typename scalar_t = float>
class FlashAttnKernelOptimizedV5 {
private:
    static constexpr int WARP_SIZE = 16;  // Intel GPU subgroup size
    static constexpr int WARPS = THREADS / WARP_SIZE;
    static constexpr int TILE_K = 16;  // XMX-friendly tile size
    static constexpr int VEC_SIZE = 16;  // Optimized for XMX
    
    const scalar_t* __restrict__ q_global;
    const scalar_t* __restrict__ k_global;
    const scalar_t* __restrict__ v_global;
    scalar_t* __restrict__ out_global;
    float* __restrict__ lse_global;
    FlashAttnConfig config;
    sycl::local_accessor<char, 1> local_mem;
    
public:
    FlashAttnKernelOptimizedV5(const scalar_t* q, const scalar_t* k, const scalar_t* v,
                               scalar_t* out, float* lse,
                               const FlashAttnConfig& cfg,
                               const sycl::local_accessor<char, 1>& lm)
        : q_global(q), k_global(k), v_global(v),
          out_global(out), lse_global(lse), config(cfg), local_mem(lm) {}
    
    void operator()(sycl::nd_item<3> item) const {
        const int tid = item.get_local_id(2);
        const int batch_idx = item.get_group(0);
        const int head_idx = item.get_group(1);
        const int q_block_idx = item.get_group(2);
        
        auto sg = item.get_sub_group();
        const int lane_id = sg.get_local_id();
        const int warp_id = tid / WARP_SIZE;
        
        // Calculate global offsets - optimized layout
        const size_t batch_head_stride = config.num_heads * config.seq_len_q * config.head_dim;
        const size_t head_stride = config.seq_len_q * config.head_dim;
        const size_t q_offset = batch_idx * batch_head_stride + head_idx * head_stride;
        const size_t k_offset = batch_idx * config.num_heads * config.seq_len_k * config.head_dim + 
                               head_idx * config.seq_len_k * config.head_dim;
        
        // Optimized shared memory layout for Intel GPU
        char* smem_base = local_mem.get_multi_ptr<sycl::access::decorated::no>().get();
        scalar_t* q_smem = reinterpret_cast<scalar_t*>(smem_base);
        scalar_t* k_smem = q_smem + BLOCK_M * HEAD_DIM;
        scalar_t* v_smem = k_smem + BLOCK_N * HEAD_DIM;
        float* acc_smem = reinterpret_cast<float*>(v_smem + BLOCK_N * HEAD_DIM);
        
        // Per-thread state - optimized for register pressure
        float row_max = -INFINITY;
        float row_sum = 0.0f;
        
        // Calculate Q block boundaries
        const int q_start = q_block_idx * BLOCK_M;
        const int q_end = sycl::min(q_start + BLOCK_M, config.seq_len_q);
        const int q_rows = q_end - q_start;
        
        // Each thread handles one or more Q rows
        const int rows_per_thread = (q_rows + THREADS - 1) / THREADS;
        const int my_q_row = tid;
        
        // Clear accumulator - vectorized for performance
        if (HEAD_DIM == 128) {
            // Optimized path for head_dim=128
            for (int idx = tid; idx < BLOCK_M * HEAD_DIM / 16; idx += THREADS) {
                const int offset = idx * 16;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    acc_smem[offset + i] = 0.0f;
                }
            }
        } else {
            for (int idx = tid; idx < BLOCK_M * HEAD_DIM; idx += THREADS) {
                acc_smem[idx] = 0.0f;
            }
        }
        
        // Load Q tile - optimized for coalesced access
        for (int idx = tid; idx < q_rows * HEAD_DIM; idx += THREADS) {
            const int row = idx / HEAD_DIM;
            const int col = idx % HEAD_DIM;
            q_smem[row * HEAD_DIM + col] = q_global[q_offset + (q_start + row) * config.head_dim + col];
        }
        
        item.barrier(sycl::access::fence_space::local_space);
        
        // Main loop over K/V blocks - optimized for non-causal
        const int num_k_blocks = divUp(config.seq_len_k, BLOCK_N);
        
        for (int k_block = 0; k_block < num_k_blocks; k_block++) {
            const int k_start = k_block * BLOCK_N;
            const int k_end = sycl::min(k_start + BLOCK_N, config.seq_len_k);
            const int k_rows = k_end - k_start;
            
            // Load K and V tiles - vectorized loads
            for (int idx = tid; idx < k_rows * HEAD_DIM; idx += THREADS) {
                const int row = idx / HEAD_DIM;
                const int col = idx % HEAD_DIM;
                const size_t offset = k_offset + (k_start + row) * config.head_dim + col;
                k_smem[row * HEAD_DIM + col] = k_global[offset];
                v_smem[row * HEAD_DIM + col] = v_global[offset];
            }
            
            item.barrier(sycl::access::fence_space::local_space);
            
            // Compute QK^T for my row - optimized for XMX
            if (my_q_row < q_rows) {
                float local_max = row_max;
                float local_sum = 0.0f;
                
                // Temporary buffer for scores
                float scores[BLOCK_N];
                
                // Compute all K scores for this Q row
                #pragma unroll 4
                for (int k_idx = 0; k_idx < k_rows; k_idx++) {
                    float score = 0.0f;
                    
                    // Optimized dot product for Intel XMX
                    if (HEAD_DIM == 128) {
                        // Process in 16-element chunks for XMX
                        #pragma unroll
                        for (int d = 0; d < HEAD_DIM; d += 16) {
                            // Manual vectorization for better control
                            float sum = 0.0f;
                            #pragma unroll
                            for (int i = 0; i < 16; i++) {
                                sum += q_smem[my_q_row * HEAD_DIM + d + i] * 
                                       k_smem[k_idx * HEAD_DIM + d + i];
                            }
                            score += sum;
                        }
                    } else if (HEAD_DIM == 64) {
                        // Optimized for head_dim=64
                        #pragma unroll
                        for (int d = 0; d < HEAD_DIM; d += 16) {
                            float sum = 0.0f;
                            #pragma unroll
                            for (int i = 0; i < 16; i++) {
                                sum += q_smem[my_q_row * HEAD_DIM + d + i] * 
                                       k_smem[k_idx * HEAD_DIM + d + i];
                            }
                            score += sum;
                        }
                    } else {
                        // Generic path
                        #pragma unroll 8
                        for (int d = 0; d < HEAD_DIM; d++) {
                            score += q_smem[my_q_row * HEAD_DIM + d] * k_smem[k_idx * HEAD_DIM + d];
                        }
                    }
                    
                    score *= config.softmax_scale;
                    scores[k_idx] = score;
                    local_max = sycl::max(local_max, score);
                }
                
                // Online softmax - numerically stable
                float max_diff = local_max - row_max;
                float scale = sycl::exp(-max_diff);
                
                // Update accumulator if needed
                if (k_block > 0 && scale != 1.0f) {
                    row_sum *= scale;
                    // Scale accumulator
                    #pragma unroll 8
                    for (int d = 0; d < HEAD_DIM; d++) {
                        acc_smem[my_q_row * HEAD_DIM + d] *= scale;
                    }
                }
                
                // Compute exp and accumulate
                float block_sum = 0.0f;
                #pragma unroll 4
                for (int k_idx = 0; k_idx < k_rows; k_idx++) {
                    float exp_score = sycl::exp(scores[k_idx] - local_max);
                    scores[k_idx] = exp_score;  // Reuse for V accumulation
                    block_sum += exp_score;
                }
                
                // Update running statistics
                row_max = local_max;
                row_sum += block_sum;
                
                // Accumulate V values - optimized for XMX
                #pragma unroll 4
                for (int k_idx = 0; k_idx < k_rows; k_idx++) {
                    const float weight = scores[k_idx];
                    
                    if (HEAD_DIM == 128) {
                        // XMX-optimized path
                        #pragma unroll
                        for (int d = 0; d < HEAD_DIM; d += 16) {
                            #pragma unroll
                            for (int i = 0; i < 16; i++) {
                                acc_smem[my_q_row * HEAD_DIM + d + i] += 
                                    weight * v_smem[k_idx * HEAD_DIM + d + i];
                            }
                        }
                    } else {
                        // Generic path
                        #pragma unroll 8
                        for (int d = 0; d < HEAD_DIM; d++) {
                            acc_smem[my_q_row * HEAD_DIM + d] += 
                                weight * v_smem[k_idx * HEAD_DIM + d];
                        }
                    }
                }
            }
            
            item.barrier(sycl::access::fence_space::local_space);
        }
        
        // Final normalization and output
        if (my_q_row < q_rows) {
            const float inv_sum = 1.0f / row_sum;
            const size_t out_row_offset = q_offset + (q_start + my_q_row) * config.head_dim;
            
            // Write normalized output
            #pragma unroll 8
            for (int d = 0; d < HEAD_DIM; d++) {
                out_global[out_row_offset + d] = scalar_t(acc_smem[my_q_row * HEAD_DIM + d] * inv_sum);
            }
            
            // Write LSE
            const size_t lse_offset = batch_idx * config.num_heads * config.seq_len_q + 
                                     head_idx * config.seq_len_q + (q_start + my_q_row);
            lse_global[lse_offset] = sycl::log(row_sum) + row_max;
        }
    }
};

// Optimized V5 kernel launcher with better configuration selection
FlashAttnOutput flash_attn_forward_optimized_v5_sycl(
    sycl::queue& q,
    const float* query,
    const float* key,
    const float* value,
    const FlashAttnConfig& config
) {
    // Optimized block size selection for sequences 128-512
    int block_m, block_n, threads;
    
    if (config.seq_len_q <= 64) {
        block_m = 32;
        block_n = 32;
        threads = 64;  // Reduced for lower overhead
    } else if (config.seq_len_q <= 128) {
        block_m = 32;
        block_n = 64;
        threads = 128;  // Better balance
    } else if (config.seq_len_q <= 256) {
        block_m = 64;
        block_n = 64;
        threads = 256;  // Optimal for this range
    } else if (config.seq_len_q <= 512) {
        block_m = 64;
        block_n = 128;
        threads = 256;  // Avoid too many threads
    } else if (config.seq_len_q <= 1024) {
        block_m = 128;
        block_n = 128;
        threads = 512;
    } else {
        block_m = 128;
        block_n = 256;
        threads = 512;
    }
    
    // Ensure threads is multiple of 16 (Intel EU width)
    threads = ((threads + 15) / 16) * 16;
    
    // Allocate output tensors
    const size_t output_size = config.batch_size * config.num_heads * config.seq_len_q * config.head_dim;
    const size_t lse_size = config.batch_size * config.num_heads * config.seq_len_q;
    
    float* d_output = sycl::malloc_device<float>(output_size, q);
    float* d_lse = sycl::malloc_device<float>(lse_size, q);
    
    // Calculate shared memory size - optimized layout
    const size_t shmem_size = sizeof(float) * (
        block_m * config.head_dim +      // Q tile
        block_n * config.head_dim +      // K tile  
        block_n * config.head_dim +      // V tile
        block_m * config.head_dim        // Accumulator
    );
    
    // Launch kernel
    const int num_q_blocks = divUp(config.seq_len_q, block_m);
    sycl::range<3> global_range(config.batch_size, config.num_heads, num_q_blocks);
    sycl::range<3> local_range(1, 1, threads);
    
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<char, 1> local_mem(sycl::range<1>(shmem_size), h);
        
        h.parallel_for(
            sycl::nd_range<3>(global_range * local_range, local_range),
            [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(16)]] {
                // Template instantiation based on configuration
                if (config.head_dim == 64) {
                    if (block_m == 32 && block_n == 32) {
                        FlashAttnKernelOptimizedV5<32, 32, 64, 64> kernel(
                            query, key, value, d_output, d_lse, config, local_mem);
                        kernel(item);
                    } else if (block_m == 32 && block_n == 64) {
                        FlashAttnKernelOptimizedV5<32, 64, 64, 128> kernel(
                            query, key, value, d_output, d_lse, config, local_mem);
                        kernel(item);
                    } else if (block_m == 64 && block_n == 64) {
                        FlashAttnKernelOptimizedV5<64, 64, 64, 256> kernel(
                            query, key, value, d_output, d_lse, config, local_mem);
                        kernel(item);
                    } else if (block_m == 64 && block_n == 128) {
                        FlashAttnKernelOptimizedV5<64, 128, 64, 256> kernel(
                            query, key, value, d_output, d_lse, config, local_mem);
                        kernel(item);
                    } else if (block_m == 128 && block_n == 128) {
                        FlashAttnKernelOptimizedV5<128, 128, 64, 512> kernel(
                            query, key, value, d_output, d_lse, config, local_mem);
                        kernel(item);
                    } else if (block_m == 128 && block_n == 256) {
                        FlashAttnKernelOptimizedV5<128, 256, 64, 512> kernel(
                            query, key, value, d_output, d_lse, config, local_mem);
                        kernel(item);
                    }
                } else if (config.head_dim == 128) {
                    if (block_m == 32 && block_n == 32) {
                        FlashAttnKernelOptimizedV5<32, 32, 128, 64> kernel(
                            query, key, value, d_output, d_lse, config, local_mem);
                        kernel(item);
                    } else if (block_m == 32 && block_n == 64) {
                        FlashAttnKernelOptimizedV5<32, 64, 128, 128> kernel(
                            query, key, value, d_output, d_lse, config, local_mem);
                        kernel(item);
                    } else if (block_m == 64 && block_n == 64) {
                        FlashAttnKernelOptimizedV5<64, 64, 128, 256> kernel(
                            query, key, value, d_output, d_lse, config, local_mem);
                        kernel(item);
                    } else if (block_m == 64 && block_n == 128) {
                        FlashAttnKernelOptimizedV5<64, 128, 128, 256> kernel(
                            query, key, value, d_output, d_lse, config, local_mem);
                        kernel(item);
                    } else if (block_m == 128 && block_n == 128) {
                        FlashAttnKernelOptimizedV5<128, 128, 128, 512> kernel(
                            query, key, value, d_output, d_lse, config, local_mem);
                        kernel(item);
                    } else if (block_m == 128 && block_n == 256) {
                        FlashAttnKernelOptimizedV5<128, 256, 128, 512> kernel(
                            query, key, value, d_output, d_lse, config, local_mem);
                        kernel(item);
                    }
                }
            }
        );
    });
    
    q.wait();
    
    return {d_output, d_lse, nullptr};
}

}  // namespace flash_attn