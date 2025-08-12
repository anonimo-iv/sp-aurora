#include "flash_attn_kernel.h"
#include "utils.h"
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>

namespace flash_attn {

// Optimized kernel with dynamic memory and Intel GPU optimizations
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM, int THREADS>
class FlashAttnKernelOptimizedV2 {
private:
    static constexpr int WARPS = THREADS / WARP_SIZE;
    static constexpr int THREADS_PER_M = THREADS / (BLOCK_M / 4); // 4 rows per thread group
    
    const float* __restrict__ q_global;
    const float* __restrict__ k_global;
    const float* __restrict__ v_global;
    float* __restrict__ out_global;
    float* __restrict__ lse_global;
    FlashAttnConfig config;
    sycl::local_accessor<float, 1> local_mem;
    
public:
    FlashAttnKernelOptimizedV2(const float* q, const float* k, const float* v,
                               float* out, float* lse,
                               const FlashAttnConfig& cfg,
                               const sycl::local_accessor<float, 1>& lm)
        : q_global(q), k_global(k), v_global(v),
          out_global(out), lse_global(lse), config(cfg), local_mem(lm) {}
    
    void operator()(sycl::nd_item<3> item) const {
        // Get thread indices
        const int tid = item.get_local_id(2);
        const int batch_idx = item.get_group(0);
        const int head_idx = item.get_group(1);
        const int q_block_idx = item.get_group(2);
        
        // Intel GPU specific: get subgroup info
        auto sg = item.get_sub_group();
        const int lane_id = sg.get_local_id();
        const int warp_id = tid / WARP_SIZE;
        
        // Calculate global offsets
        const int q_batch_head_offset = (batch_idx * config.num_heads + head_idx) * 
                                         config.seq_len_q * config.head_dim;
        const int k_batch_head_offset = (batch_idx * config.num_heads + head_idx) * 
                                         config.seq_len_k * config.head_dim;
        
        // Dynamic shared memory layout
        float* smem_base = local_mem.get_multi_ptr<sycl::access::decorated::no>().get();
        float* q_smem = smem_base;
        float* k_smem = q_smem + BLOCK_M * HEAD_DIM;
        float* v_smem = k_smem + BLOCK_N * HEAD_DIM;
        float* s_smem = v_smem + BLOCK_N * HEAD_DIM;
        float* acc_smem = s_smem + BLOCK_M * BLOCK_N;  // Accumulator in shared memory
        
        // Per-thread registers
        float row_max[4];  // Max 4 rows per thread
        float row_sum[4];
        
        // Initialize with proper constant
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            row_max[i] = -std::numeric_limits<float>::infinity();
            row_sum[i] = 0.0f;
        }
        
        // Clear accumulator in shared memory
        const int acc_elements = BLOCK_M * HEAD_DIM;
        for (int i = tid; i < acc_elements; i += THREADS) {
            acc_smem[i] = 0.0f;
        }
        
        // Calculate Q block boundaries
        const int q_start = q_block_idx * BLOCK_M;
        const int q_end = sycl::min(q_start + BLOCK_M, config.seq_len_q);
        const int q_rows = q_end - q_start;
        
        // Load Q tile once (coalesced access)
        const int q_elements = q_rows * HEAD_DIM;
        for (int idx = tid; idx < q_elements; idx += THREADS) {
            const int row = idx / HEAD_DIM;
            const int col = idx % HEAD_DIM;
            if (row < q_rows) {
                q_smem[row * HEAD_DIM + col] = 
                    q_global[q_batch_head_offset + (q_start + row) * config.head_dim + col];
            }
        }
        
        item.barrier();
        
        // Main loop over K/V blocks
        const int num_k_blocks = divUp(config.seq_len_k, BLOCK_N);
        
        for (int k_block = 0; k_block < num_k_blocks; k_block++) {
            const int k_start = k_block * BLOCK_N;
            const int k_end = sycl::min(k_start + BLOCK_N, config.seq_len_k);
            const int k_rows = k_end - k_start;
            
            // Load K and V tiles (vectorized when possible)
            const int kv_elements = k_rows * HEAD_DIM;
            
            // Try to use float2 loads for better bandwidth
            if (HEAD_DIM % 2 == 0) {
                for (int idx = tid; idx < kv_elements / 2; idx += THREADS) {
                    const int row = idx / (HEAD_DIM / 2);
                    const int col2 = idx % (HEAD_DIM / 2);
                    const int col = col2 * 2;
                    
                    if (row < k_rows) {
                        const int global_offset = k_batch_head_offset + 
                                                  (k_start + row) * config.head_dim + col;
                        
                        // Load as float2 for better memory bandwidth
                        sycl::float2 k_val = *reinterpret_cast<const sycl::float2*>(&k_global[global_offset]);
                        sycl::float2 v_val = *reinterpret_cast<const sycl::float2*>(&v_global[global_offset]);
                        
                        reinterpret_cast<sycl::float2*>(&k_smem[row * HEAD_DIM + col])[0] = k_val;
                        reinterpret_cast<sycl::float2*>(&v_smem[row * HEAD_DIM + col])[0] = v_val;
                    }
                }
            } else {
                // Fallback to scalar loads
                for (int idx = tid; idx < kv_elements; idx += THREADS) {
                    const int row = idx / HEAD_DIM;
                    const int col = idx % HEAD_DIM;
                    if (row < k_rows) {
                        const int global_offset = k_batch_head_offset + 
                                                  (k_start + row) * config.head_dim + col;
                        k_smem[row * HEAD_DIM + col] = k_global[global_offset];
                        v_smem[row * HEAD_DIM + col] = v_global[global_offset];
                    }
                }
            }
            
            item.barrier();
            
            // Compute QK^T with improved parallelism
            // Each thread handles multiple Q rows for better efficiency
            const int rows_per_thread = (q_rows + THREADS - 1) / THREADS;
            const int my_q_start = tid * rows_per_thread;
            const int my_q_end = sycl::min(my_q_start + rows_per_thread, q_rows);
            
            for (int q_idx = my_q_start; q_idx < my_q_end; q_idx++) {
                const int local_row_idx = q_idx - my_q_start;
                if (local_row_idx >= 4) break;  // Max 4 rows per thread
                
                float local_max = row_max[local_row_idx];
                float local_sum = 0.0f;
                
                // Compute dot products for this Q row
                for (int k_idx = 0; k_idx < k_rows; k_idx++) {
                    float score = 0.0f;
                    
                    // Vectorized dot product
                    if (HEAD_DIM % 4 == 0) {
                        #pragma unroll
                        for (int d = 0; d < HEAD_DIM; d += 4) {
                            sycl::float4 q_vec = *reinterpret_cast<const sycl::float4*>(&q_smem[q_idx * HEAD_DIM + d]);
                            sycl::float4 k_vec = *reinterpret_cast<const sycl::float4*>(&k_smem[k_idx * HEAD_DIM + d]);
                            score += q_vec.x() * k_vec.x() + q_vec.y() * k_vec.y() + 
                                     q_vec.z() * k_vec.z() + q_vec.w() * k_vec.w();
                        }
                    } else {
                        #pragma unroll
                        for (int d = 0; d < HEAD_DIM; d++) {
                            score += q_smem[q_idx * HEAD_DIM + d] * k_smem[k_idx * HEAD_DIM + d];
                        }
                    }
                    
                    score *= config.softmax_scale;
                    
                    // Apply causal mask
                    if (config.is_causal) {
                        const int global_q_idx = q_start + q_idx;
                        const int global_k_idx = k_start + k_idx;
                        if (global_k_idx > global_q_idx) {
                            score = -std::numeric_limits<float>::infinity();
                        }
                    }
                    
                    s_smem[q_idx * BLOCK_N + k_idx] = score;
                    local_max = sycl::max(local_max, score);
                }
                
                // Update max and compute exponentials
                float max_diff = local_max - row_max[local_row_idx];
                float scale = sycl::exp(-max_diff);
                
                // Rescale previous sum
                row_sum[local_row_idx] *= scale;
                
                // Compute exp and sum for current block
                for (int k_idx = 0; k_idx < k_rows; k_idx++) {
                    float exp_score = sycl::exp(s_smem[q_idx * BLOCK_N + k_idx] - local_max);
                    s_smem[q_idx * BLOCK_N + k_idx] = exp_score;
                    local_sum += exp_score;
                }
                
                // Update global max and sum
                row_max[local_row_idx] = local_max;
                row_sum[local_row_idx] += local_sum;
                
                // Rescale accumulator if needed
                if (max_diff != 0.0f && k_block > 0) {  // Only rescale if not first block
                    for (int d = 0; d < HEAD_DIM; d++) {
                        acc_smem[q_idx * HEAD_DIM + d] *= scale;
                    }
                }
                
                // Accumulate V weighted by attention scores
                for (int k_idx = 0; k_idx < k_rows; k_idx++) {
                    float attn_weight = s_smem[q_idx * BLOCK_N + k_idx];
                    
                    // Vectorized accumulation
                    if (HEAD_DIM % 4 == 0) {
                        #pragma unroll
                        for (int d = 0; d < HEAD_DIM; d += 4) {
                            sycl::float4 v_vec = *reinterpret_cast<const sycl::float4*>(&v_smem[k_idx * HEAD_DIM + d]);
                            sycl::float4 acc_vec = *reinterpret_cast<const sycl::float4*>(&acc_smem[q_idx * HEAD_DIM + d]);
                            acc_vec += attn_weight * v_vec;
                            *reinterpret_cast<sycl::float4*>(&acc_smem[q_idx * HEAD_DIM + d]) = acc_vec;
                        }
                    } else {
                        #pragma unroll
                        for (int d = 0; d < HEAD_DIM; d++) {
                            acc_smem[q_idx * HEAD_DIM + d] += attn_weight * v_smem[k_idx * HEAD_DIM + d];
                        }
                    }
                }
            }
            
            item.barrier();
        }
        
        // Calculate rows per thread again for output writing
        const int output_rows_per_thread = (q_rows + THREADS - 1) / THREADS;
        const int output_my_q_start = tid * output_rows_per_thread;
        const int output_my_q_end = sycl::min(output_my_q_start + output_rows_per_thread, q_rows);
        
        // First, synchronize to ensure all threads have finished computing
        item.barrier();
        
        // Write output - simplified approach to avoid broadcast issues
        for (int q_idx = 0; q_idx < q_rows; q_idx++) {
            // Find which thread computed this row
            const int owner_tid = q_idx / output_rows_per_thread;
            const int owner_local_row = q_idx % output_rows_per_thread;
            
            if (owner_tid == tid && owner_local_row < 4) {
                // This thread owns this row, write it out
                const float inv_sum = 1.0f / row_sum[owner_local_row];
                
                for (int d = 0; d < HEAD_DIM; d++) {
                    const int out_offset = q_batch_head_offset + (q_start + q_idx) * config.head_dim + d;
                    out_global[out_offset] = acc_smem[q_idx * HEAD_DIM + d] * inv_sum;
                }
                
                // Also write LSE for this row
                const int lse_offset = batch_idx * config.num_heads * config.seq_len_q + 
                                       head_idx * config.seq_len_q + (q_start + q_idx);
                lse_global[lse_offset] = sycl::log(row_sum[owner_local_row]) + row_max[owner_local_row];
            }
        }
        
        // LSE writing is now done in the output loop above
    }
};

// Kernel launcher with adaptive configuration
FlashAttnOutput flash_attn_forward_optimized_v2_sycl(
    sycl::queue& q,
    const float* query,
    const float* key,
    const float* value,
    const FlashAttnConfig& config
) {
    // Adaptive block size selection based on sequence length
    // Optimized for small to medium sequences only
    int block_m, block_n, threads;
    
    if (config.seq_len_q <= 64) {
        block_m = 32;
        block_n = 32;
        threads = 128;
    } else if (config.seq_len_q <= 128) {
        block_m = 32;
        block_n = 32;
        threads = 128;
    } else if (config.seq_len_q <= 256) {
        block_m = 64;
        block_n = 32;
        threads = 128;
    } else {
        // For larger sequences, this kernel won't be used
        // Fallback to basic kernel in bindings
        block_m = 32;
        block_n = 32;
        threads = 128;
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
        block_m * block_n +          // Attention scores
        block_m * config.head_dim    // Accumulator
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
                    if (block_m == 32 && block_n == 32 && threads == 128) {
                        FlashAttnKernelOptimizedV2<32, 32, 64, 128> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 32 && block_n == 32 && threads == 64) {
                        FlashAttnKernelOptimizedV2<32, 32, 64, 64> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 64 && block_n == 32) {
                        FlashAttnKernelOptimizedV2<64, 32, 64, 128> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 32 && block_n == 16) {
                        FlashAttnKernelOptimizedV2<32, 16, 64, 64> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 16 && block_n == 16 && threads == 64) {
                        FlashAttnKernelOptimizedV2<16, 16, 64, 64> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 16 && block_n == 16 && threads == 32) {
                        FlashAttnKernelOptimizedV2<16, 16, 64, 32> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    }
                } else if (config.head_dim == 128) {
                    if (block_m == 32 && block_n == 32 && threads == 128) {
                        FlashAttnKernelOptimizedV2<32, 32, 128, 128> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 32 && block_n == 32 && threads == 64) {
                        FlashAttnKernelOptimizedV2<32, 32, 128, 64> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 64 && block_n == 32) {
                        FlashAttnKernelOptimizedV2<64, 32, 128, 128> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 32 && block_n == 16) {
                        FlashAttnKernelOptimizedV2<32, 16, 128, 64> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 16 && block_n == 16 && threads == 64) {
                        FlashAttnKernelOptimizedV2<16, 16, 128, 64> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 16 && block_n == 16 && threads == 32) {
                        FlashAttnKernelOptimizedV2<16, 16, 128, 32> kernel(
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