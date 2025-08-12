#include "flash_attn_kernel.h"
#include "utils.h"
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>

namespace flash_attn {

// Fixed optimized kernel without atomic operations
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM, int THREADS>
class FlashAttnKernelOptimizedV3 {
private:
    static constexpr int WARPS = THREADS / WARP_SIZE;
    static constexpr int ROWS_PER_THREAD = (BLOCK_M + THREADS - 1) / THREADS;
    
    const float* __restrict__ q_global;
    const float* __restrict__ k_global;
    const float* __restrict__ v_global;
    float* __restrict__ out_global;
    float* __restrict__ lse_global;
    FlashAttnConfig config;
    sycl::local_accessor<float, 1> local_mem;
    
public:
    FlashAttnKernelOptimizedV3(const float* q, const float* k, const float* v,
                               float* out, float* lse,
                               const FlashAttnConfig& cfg,
                               const sycl::local_accessor<float, 1>& lm)
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
        
        // Calculate global offsets
        const int q_batch_head_offset = (batch_idx * config.num_heads + head_idx) * 
                                         config.seq_len_q * config.head_dim;
        const int k_batch_head_offset = (batch_idx * config.num_heads + head_idx) * 
                                         config.seq_len_k * config.head_dim;
        
        // Shared memory layout
        float* smem_base = local_mem.get_multi_ptr<sycl::access::decorated::no>().get();
        float* q_smem = smem_base;
        float* k_smem = q_smem + BLOCK_M * HEAD_DIM;
        float* v_smem = k_smem + BLOCK_N * HEAD_DIM;
        float* s_smem = v_smem + BLOCK_N * HEAD_DIM;
        
        // Per-thread accumulators in registers (max 2 rows per thread)
        float acc[2][HEAD_DIM];
        float row_max[2];
        float row_sum[2];
        
        // Initialize per-thread accumulators
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            row_max[i] = -INFINITY;
            row_sum[i] = 0.0f;
            #pragma unroll
            for (int j = 0; j < HEAD_DIM; j++) {
                acc[i][j] = 0.0f;
            }
        }
        
        // Calculate Q block boundaries
        const int q_start = q_block_idx * BLOCK_M;
        const int q_end = sycl::min(q_start + BLOCK_M, config.seq_len_q);
        const int q_rows = q_end - q_start;
        
        // Calculate which rows this thread will process
        const int my_q_start = tid * ROWS_PER_THREAD;
        const int my_q_end = sycl::min(my_q_start + ROWS_PER_THREAD, q_rows);
        const int my_num_rows = my_q_end - my_q_start;
        
        // Load Q tile (coalesced access)
        for (int idx = tid; idx < q_rows * HEAD_DIM; idx += THREADS) {
            const int row = idx / HEAD_DIM;
            const int col = idx % HEAD_DIM;
            q_smem[row * HEAD_DIM + col] = 
                q_global[q_batch_head_offset + (q_start + row) * config.head_dim + col];
        }
        
        item.barrier();
        
        // Process K/V blocks
        const int num_k_blocks = divUp(config.seq_len_k, BLOCK_N);
        
        for (int k_block = 0; k_block < num_k_blocks; k_block++) {
            const int k_start = k_block * BLOCK_N;
            const int k_end = sycl::min(k_start + BLOCK_N, config.seq_len_k);
            const int k_rows = k_end - k_start;
            
            // Load K and V tiles with vectorized loads
            if (HEAD_DIM >= 4) {
                for (int idx = tid; idx < k_rows * HEAD_DIM / 4; idx += THREADS) {
                    const int row = idx / (HEAD_DIM / 4);
                    const int col4 = idx % (HEAD_DIM / 4);
                    
                    if (row < k_rows) {
                        const int global_offset = k_batch_head_offset + 
                                                  (k_start + row) * config.head_dim + col4 * 4;
                        
                        sycl::float4 k_val = *reinterpret_cast<const sycl::float4*>(&k_global[global_offset]);
                        sycl::float4 v_val = *reinterpret_cast<const sycl::float4*>(&v_global[global_offset]);
                        
                        *reinterpret_cast<sycl::float4*>(&k_smem[row * HEAD_DIM + col4 * 4]) = k_val;
                        *reinterpret_cast<sycl::float4*>(&v_smem[row * HEAD_DIM + col4 * 4]) = v_val;
                    }
                }
            }
            
            item.barrier();
            
            // Process my assigned Q rows
            for (int local_q = 0; local_q < my_num_rows && local_q < 2; local_q++) {
                const int q_idx = my_q_start + local_q;
                float local_max = row_max[local_q];
                float local_sum = 0.0f;
                
                // Compute QK^T for this Q row
                for (int k_idx = 0; k_idx < k_rows; k_idx++) {
                    float score = 0.0f;
                    
                    // Vectorized dot product
                    if (HEAD_DIM == 64) {
                        #pragma unroll
                        for (int d = 0; d < 64; d += 8) {
                            sycl::float4 q_vec1 = *reinterpret_cast<const sycl::float4*>(&q_smem[q_idx * HEAD_DIM + d]);
                            sycl::float4 k_vec1 = *reinterpret_cast<const sycl::float4*>(&k_smem[k_idx * HEAD_DIM + d]);
                            sycl::float4 q_vec2 = *reinterpret_cast<const sycl::float4*>(&q_smem[q_idx * HEAD_DIM + d + 4]);
                            sycl::float4 k_vec2 = *reinterpret_cast<const sycl::float4*>(&k_smem[k_idx * HEAD_DIM + d + 4]);
                            
                            score += q_vec1.x() * k_vec1.x() + q_vec1.y() * k_vec1.y() + 
                                     q_vec1.z() * k_vec1.z() + q_vec1.w() * k_vec1.w() +
                                     q_vec2.x() * k_vec2.x() + q_vec2.y() * k_vec2.y() + 
                                     q_vec2.z() * k_vec2.z() + q_vec2.w() * k_vec2.w();
                        }
                    } else if (HEAD_DIM == 128) {
                        #pragma unroll
                        for (int d = 0; d < 128; d += 8) {
                            sycl::float4 q_vec1 = *reinterpret_cast<const sycl::float4*>(&q_smem[q_idx * HEAD_DIM + d]);
                            sycl::float4 k_vec1 = *reinterpret_cast<const sycl::float4*>(&k_smem[k_idx * HEAD_DIM + d]);
                            sycl::float4 q_vec2 = *reinterpret_cast<const sycl::float4*>(&q_smem[q_idx * HEAD_DIM + d + 4]);
                            sycl::float4 k_vec2 = *reinterpret_cast<const sycl::float4*>(&k_smem[k_idx * HEAD_DIM + d + 4]);
                            
                            score += q_vec1.x() * k_vec1.x() + q_vec1.y() * k_vec1.y() + 
                                     q_vec1.z() * k_vec1.z() + q_vec1.w() * k_vec1.w() +
                                     q_vec2.x() * k_vec2.x() + q_vec2.y() * k_vec2.y() + 
                                     q_vec2.z() * k_vec2.z() + q_vec2.w() * k_vec2.w();
                        }
                    }
                    
                    score *= config.softmax_scale;
                    
                    // Apply causal mask
                    if (config.is_causal) {
                        const int global_q_idx = q_start + q_idx;
                        const int global_k_idx = k_start + k_idx;
                        if (global_k_idx > global_q_idx) {
                            score = -INFINITY;
                        }
                    }
                    
                    s_smem[q_idx * BLOCK_N + k_idx] = score;
                    local_max = sycl::max(local_max, score);
                }
                
                // Online softmax update
                float max_diff = local_max - row_max[local_q];
                float scale = sycl::exp(-max_diff);
                
                // Rescale previous sum and accumulator
                row_sum[local_q] *= scale;
                if (scale != 1.0f) {
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d++) {
                        acc[local_q][d] *= scale;
                    }
                }
                
                // Compute exp and accumulate V
                for (int k_idx = 0; k_idx < k_rows; k_idx++) {
                    float exp_score = sycl::exp(s_smem[q_idx * BLOCK_N + k_idx] - local_max);
                    local_sum += exp_score;
                    
                    // Accumulate weighted V values
                    if (HEAD_DIM == 64) {
                        #pragma unroll
                        for (int d = 0; d < 64; d += 4) {
                            sycl::float4 v_vec = *reinterpret_cast<const sycl::float4*>(&v_smem[k_idx * HEAD_DIM + d]);
                            acc[local_q][d + 0] += exp_score * v_vec.x();
                            acc[local_q][d + 1] += exp_score * v_vec.y();
                            acc[local_q][d + 2] += exp_score * v_vec.z();
                            acc[local_q][d + 3] += exp_score * v_vec.w();
                        }
                    } else if (HEAD_DIM == 128) {
                        #pragma unroll
                        for (int d = 0; d < 128; d += 4) {
                            sycl::float4 v_vec = *reinterpret_cast<const sycl::float4*>(&v_smem[k_idx * HEAD_DIM + d]);
                            acc[local_q][d + 0] += exp_score * v_vec.x();
                            acc[local_q][d + 1] += exp_score * v_vec.y();
                            acc[local_q][d + 2] += exp_score * v_vec.z();
                            acc[local_q][d + 3] += exp_score * v_vec.w();
                        }
                    }
                }
                
                row_max[local_q] = local_max;
                row_sum[local_q] += local_sum;
            }
            
            item.barrier();
        }
        
        // Write output for my rows
        for (int local_q = 0; local_q < my_num_rows && local_q < 2; local_q++) {
            const int q_idx = my_q_start + local_q;
            const int global_q_idx = q_start + q_idx;
            
            const float inv_sum = 1.0f / row_sum[local_q];
            
            // Write normalized output
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                const int out_offset = q_batch_head_offset + global_q_idx * config.head_dim + d;
                out_global[out_offset] = acc[local_q][d] * inv_sum;
            }
            
            // Write LSE
            const int lse_offset = batch_idx * config.num_heads * config.seq_len_q + 
                                   head_idx * config.seq_len_q + global_q_idx;
            lse_global[lse_offset] = sycl::log(row_sum[local_q]) + row_max[local_q];
        }
    }
};

// Optimized kernel launcher with better configuration selection
FlashAttnOutput flash_attn_forward_optimized_v3_sycl(
    sycl::queue& q,
    const float* query,
    const float* key,
    const float* value,
    const FlashAttnConfig& config
) {
    // More aggressive block size selection for Intel Max GPU
    int block_m, block_n, threads;
    
    if (config.seq_len_q <= 128) {
        block_m = 32;
        block_n = 64;
        threads = 128;
    } else if (config.seq_len_q <= 512) {
        block_m = 64;
        block_n = 128;
        threads = 256;
    } else if (config.seq_len_q <= 2048) {
        block_m = 128;
        block_n = 128;
        threads = 256;
    } else if (config.seq_len_q <= 8192) {
        // Use larger blocks to leverage L2 cache
        block_m = 256;
        block_n = 256;
        threads = 512;
    } else {
        // For very long sequences, use streaming approach
        block_m = 128;
        block_n = 512;
        threads = 256;
    }
    
    // Ensure threads is a multiple of WARP_SIZE
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    
    // Allocate output tensors
    const size_t output_size = config.batch_size * config.num_heads * config.seq_len_q * config.head_dim;
    const size_t lse_size = config.batch_size * config.num_heads * config.seq_len_q;
    
    float* d_output = sycl::malloc_device<float>(output_size, q);
    float* d_lse = sycl::malloc_device<float>(lse_size, q);
    
    // Calculate shared memory size
    const size_t shmem_size = sizeof(float) * (
        block_m * config.head_dim +  // Q tile
        block_n * config.head_dim +  // K tile
        block_n * config.head_dim +  // V tile
        block_m * block_n           // Attention scores
    );
    
    // Launch kernel
    const int num_q_blocks = divUp(config.seq_len_q, block_m);
    sycl::range<3> global_range(config.batch_size, config.num_heads, num_q_blocks);
    sycl::range<3> local_range(1, 1, threads);
    
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> local_mem(sycl::range<1>(shmem_size / sizeof(float)), h);
        
        h.parallel_for(
            sycl::nd_range<3>(global_range * local_range, local_range),
            [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(16)]] {
                if (config.head_dim == 64) {
                    if (block_m == 32 && block_n == 64) {
                        FlashAttnKernelOptimizedV3<32, 64, 64, 128> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 64 && block_n == 128) {
                        FlashAttnKernelOptimizedV3<64, 128, 64, 256> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 128 && block_n == 128) {
                        FlashAttnKernelOptimizedV3<128, 128, 64, 256> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 256 && block_n == 256) {
                        FlashAttnKernelOptimizedV3<256, 256, 64, 512> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 128 && block_n == 512) {
                        FlashAttnKernelOptimizedV3<128, 512, 64, 256> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    }
                } else if (config.head_dim == 128) {
                    if (block_m == 32 && block_n == 64) {
                        FlashAttnKernelOptimizedV3<32, 64, 128, 128> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 64 && block_n == 128) {
                        FlashAttnKernelOptimizedV3<64, 128, 128, 256> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 128 && block_n == 128) {
                        FlashAttnKernelOptimizedV3<128, 128, 128, 256> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 256 && block_n == 256) {
                        FlashAttnKernelOptimizedV3<256, 256, 128, 512> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 128 && block_n == 512) {
                        FlashAttnKernelOptimizedV3<128, 512, 128, 256> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
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