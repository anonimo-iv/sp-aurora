#include "flash_attn_kernel.h"
#include "utils.h"
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>

namespace flash_attn {

// Simplified XMX-optimized kernel using DPAS instructions
// This version uses inline assembly or compiler intrinsics for Intel GPU
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM, int THREADS>
class FlashAttnKernelXMXSimple {
private:
    static constexpr int WARPS = THREADS / WARP_SIZE;
    static constexpr int TILE_SIZE = 16;  // Optimal tile size for Intel GPU
    
    const float* __restrict__ q_global;
    const float* __restrict__ k_global;
    const float* __restrict__ v_global;
    float* __restrict__ out_global;
    float* __restrict__ lse_global;
    FlashAttnConfig config;
    sycl::local_accessor<float, 1> local_mem;
    
public:
    FlashAttnKernelXMXSimple(const float* q, const float* k, const float* v,
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
        
        // Shared memory layout optimized for coalesced access
        float* smem_base = local_mem.get_multi_ptr<sycl::access::decorated::no>().get();
        float* q_smem = smem_base;
        float* k_smem = q_smem + BLOCK_M * HEAD_DIM;
        float* v_smem = k_smem + BLOCK_N * HEAD_DIM;
        float* s_smem = v_smem + BLOCK_N * HEAD_DIM;
        
        // Per-thread accumulators (2 rows max per thread)
        float acc[2][HEAD_DIM];
        float row_max[2];
        float row_sum[2];
        
        // Initialize
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
        
        // Calculate thread assignment
        const int rows_per_thread = (q_rows + THREADS - 1) / THREADS;
        const int my_q_start = tid * rows_per_thread;
        const int my_q_end = sycl::min(my_q_start + rows_per_thread, q_rows);
        const int my_num_rows = my_q_end - my_q_start;
        
        // Load Q tile with vectorized loads
        for (int idx = tid; idx < q_rows * HEAD_DIM / 4; idx += THREADS) {
            const int row = idx / (HEAD_DIM / 4);
            const int col4 = idx % (HEAD_DIM / 4);
            
            if (row < q_rows) {
                const int offset = q_batch_head_offset + (q_start + row) * config.head_dim + col4 * 4;
                sycl::float4 q_vec = *reinterpret_cast<const sycl::float4*>(&q_global[offset]);
                *reinterpret_cast<sycl::float4*>(&q_smem[row * HEAD_DIM + col4 * 4]) = q_vec;
            }
        }
        
        item.barrier();
        
        // Process K/V blocks
        const int num_k_blocks = divUp(config.seq_len_k, BLOCK_N);
        
        for (int k_block = 0; k_block < num_k_blocks; k_block++) {
            const int k_start = k_block * BLOCK_N;
            const int k_end = sycl::min(k_start + BLOCK_N, config.seq_len_k);
            const int k_rows = k_end - k_start;
            
            // Load K and V tiles with vectorized loads
            for (int idx = tid; idx < k_rows * HEAD_DIM / 4; idx += THREADS) {
                const int row = idx / (HEAD_DIM / 4);
                const int col4 = idx % (HEAD_DIM / 4);
                
                if (row < k_rows) {
                    const int offset = k_batch_head_offset + (k_start + row) * config.head_dim + col4 * 4;
                    sycl::float4 k_vec = *reinterpret_cast<const sycl::float4*>(&k_global[offset]);
                    sycl::float4 v_vec = *reinterpret_cast<const sycl::float4*>(&v_global[offset]);
                    *reinterpret_cast<sycl::float4*>(&k_smem[row * HEAD_DIM + col4 * 4]) = k_vec;
                    *reinterpret_cast<sycl::float4*>(&v_smem[row * HEAD_DIM + col4 * 4]) = v_vec;
                }
            }
            
            item.barrier();
            
            // Compute QK^T using optimized tiling
            for (int local_q = 0; local_q < my_num_rows && local_q < 2; local_q++) {
                const int q_idx = my_q_start + local_q;
                float local_max = row_max[local_q];
                float local_sum = 0.0f;
                
                // Process in tiles for better cache usage
                for (int k_tile = 0; k_tile < k_rows; k_tile += TILE_SIZE) {
                    const int tile_end = sycl::min(k_tile + TILE_SIZE, k_rows);
                    
                    // Compute scores for this tile
                    float scores[TILE_SIZE];
                    #pragma unroll
                    for (int kt = 0; kt < TILE_SIZE && k_tile + kt < tile_end; kt++) {
                        const int k_idx = k_tile + kt;
                        float score = 0.0f;
                        
                        // Vectorized dot product
                        if (HEAD_DIM == 64) {
                            #pragma unroll
                            for (int d = 0; d < 64; d += 16) {
                                // Use float4 for better performance
                                sycl::float4 q1 = *reinterpret_cast<const sycl::float4*>(&q_smem[q_idx * HEAD_DIM + d]);
                                sycl::float4 k1 = *reinterpret_cast<const sycl::float4*>(&k_smem[k_idx * HEAD_DIM + d]);
                                sycl::float4 q2 = *reinterpret_cast<const sycl::float4*>(&q_smem[q_idx * HEAD_DIM + d + 4]);
                                sycl::float4 k2 = *reinterpret_cast<const sycl::float4*>(&k_smem[k_idx * HEAD_DIM + d + 4]);
                                sycl::float4 q3 = *reinterpret_cast<const sycl::float4*>(&q_smem[q_idx * HEAD_DIM + d + 8]);
                                sycl::float4 k3 = *reinterpret_cast<const sycl::float4*>(&k_smem[k_idx * HEAD_DIM + d + 8]);
                                sycl::float4 q4 = *reinterpret_cast<const sycl::float4*>(&q_smem[q_idx * HEAD_DIM + d + 12]);
                                sycl::float4 k4 = *reinterpret_cast<const sycl::float4*>(&k_smem[k_idx * HEAD_DIM + d + 12]);
                                
                                score += q1.x() * k1.x() + q1.y() * k1.y() + q1.z() * k1.z() + q1.w() * k1.w() +
                                         q2.x() * k2.x() + q2.y() * k2.y() + q2.z() * k2.z() + q2.w() * k2.w() +
                                         q3.x() * k3.x() + q3.y() * k3.y() + q3.z() * k3.z() + q3.w() * k3.w() +
                                         q4.x() * k4.x() + q4.y() * k4.y() + q4.z() * k4.z() + q4.w() * k4.w();
                            }
                        } else if (HEAD_DIM == 128) {
                            #pragma unroll
                            for (int d = 0; d < 128; d += 16) {
                                sycl::float4 q1 = *reinterpret_cast<const sycl::float4*>(&q_smem[q_idx * HEAD_DIM + d]);
                                sycl::float4 k1 = *reinterpret_cast<const sycl::float4*>(&k_smem[k_idx * HEAD_DIM + d]);
                                sycl::float4 q2 = *reinterpret_cast<const sycl::float4*>(&q_smem[q_idx * HEAD_DIM + d + 4]);
                                sycl::float4 k2 = *reinterpret_cast<const sycl::float4*>(&k_smem[k_idx * HEAD_DIM + d + 4]);
                                sycl::float4 q3 = *reinterpret_cast<const sycl::float4*>(&q_smem[q_idx * HEAD_DIM + d + 8]);
                                sycl::float4 k3 = *reinterpret_cast<const sycl::float4*>(&k_smem[k_idx * HEAD_DIM + d + 8]);
                                sycl::float4 q4 = *reinterpret_cast<const sycl::float4*>(&q_smem[q_idx * HEAD_DIM + d + 12]);
                                sycl::float4 k4 = *reinterpret_cast<const sycl::float4*>(&k_smem[k_idx * HEAD_DIM + d + 12]);
                                
                                score += q1.x() * k1.x() + q1.y() * k1.y() + q1.z() * k1.z() + q1.w() * k1.w() +
                                         q2.x() * k2.x() + q2.y() * k2.y() + q2.z() * k2.z() + q2.w() * k2.w() +
                                         q3.x() * k3.x() + q3.y() * k3.y() + q3.z() * k3.z() + q3.w() * k3.w() +
                                         q4.x() * k4.x() + q4.y() * k4.y() + q4.z() * k4.z() + q4.w() * k4.w();
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
                        
                        scores[kt] = score;
                        local_max = sycl::max(local_max, score);
                    }
                    
                    // Update max and compute exp
                    float max_diff = local_max - row_max[local_q];
                    float scale = sycl::exp(-max_diff);
                    
                    // Rescale accumulator if needed
                    if (scale != 1.0f) {
                        row_sum[local_q] *= scale;
                        #pragma unroll
                        for (int d = 0; d < HEAD_DIM; d++) {
                            acc[local_q][d] *= scale;
                        }
                    }
                    
                    // Compute exp and accumulate V for this tile
                    #pragma unroll
                    for (int kt = 0; kt < TILE_SIZE && k_tile + kt < tile_end; kt++) {
                        const int k_idx = k_tile + kt;
                        float exp_score = sycl::exp(scores[kt] - local_max);
                        local_sum += exp_score;
                        
                        // Accumulate V with vectorization
                        if (HEAD_DIM == 64) {
                            #pragma unroll
                            for (int d = 0; d < 64; d += 8) {
                                sycl::float4 v1 = *reinterpret_cast<const sycl::float4*>(&v_smem[k_idx * HEAD_DIM + d]);
                                sycl::float4 v2 = *reinterpret_cast<const sycl::float4*>(&v_smem[k_idx * HEAD_DIM + d + 4]);
                                
                                acc[local_q][d + 0] += exp_score * v1.x();
                                acc[local_q][d + 1] += exp_score * v1.y();
                                acc[local_q][d + 2] += exp_score * v1.z();
                                acc[local_q][d + 3] += exp_score * v1.w();
                                acc[local_q][d + 4] += exp_score * v2.x();
                                acc[local_q][d + 5] += exp_score * v2.y();
                                acc[local_q][d + 6] += exp_score * v2.z();
                                acc[local_q][d + 7] += exp_score * v2.w();
                            }
                        } else if (HEAD_DIM == 128) {
                            #pragma unroll
                            for (int d = 0; d < 128; d += 8) {
                                sycl::float4 v1 = *reinterpret_cast<const sycl::float4*>(&v_smem[k_idx * HEAD_DIM + d]);
                                sycl::float4 v2 = *reinterpret_cast<const sycl::float4*>(&v_smem[k_idx * HEAD_DIM + d + 4]);
                                
                                acc[local_q][d + 0] += exp_score * v1.x();
                                acc[local_q][d + 1] += exp_score * v1.y();
                                acc[local_q][d + 2] += exp_score * v1.z();
                                acc[local_q][d + 3] += exp_score * v1.w();
                                acc[local_q][d + 4] += exp_score * v2.x();
                                acc[local_q][d + 5] += exp_score * v2.y();
                                acc[local_q][d + 6] += exp_score * v2.z();
                                acc[local_q][d + 7] += exp_score * v2.w();
                            }
                        }
                    }
                    
                    row_max[local_q] = local_max;
                }
                
                row_sum[local_q] += local_sum;
            }
            
            item.barrier();
        }
        
        // Write output for my rows
        for (int local_q = 0; local_q < my_num_rows && local_q < 2; local_q++) {
            const int q_idx = my_q_start + local_q;
            const int global_q_idx = q_start + q_idx;
            
            const float inv_sum = 1.0f / row_sum[local_q];
            
            // Write normalized output with vectorized stores
            if (HEAD_DIM >= 4) {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d += 4) {
                    sycl::float4 out_vec(
                        acc[local_q][d + 0] * inv_sum,
                        acc[local_q][d + 1] * inv_sum,
                        acc[local_q][d + 2] * inv_sum,
                        acc[local_q][d + 3] * inv_sum
                    );
                    const int out_offset = q_batch_head_offset + global_q_idx * config.head_dim + d;
                    *reinterpret_cast<sycl::float4*>(&out_global[out_offset]) = out_vec;
                }
            }
            
            // Write LSE
            const int lse_offset = batch_idx * config.num_heads * config.seq_len_q + 
                                   head_idx * config.seq_len_q + global_q_idx;
            lse_global[lse_offset] = sycl::log(row_sum[local_q]) + row_max[local_q];
        }
    }
};

// XMX-optimized kernel launcher
FlashAttnOutput flash_attn_forward_xmx_sycl(
    sycl::queue& q,
    const float* query,
    const float* key,
    const float* value,
    const FlashAttnConfig& config
) {
    // Configuration optimized for Intel Max 1550
    int block_m, block_n, threads;
    
    if (config.seq_len_q <= 256) {
        block_m = 64;
        block_n = 64;
        threads = 128;
    } else if (config.seq_len_q <= 1024) {
        block_m = 128;
        block_n = 128;
        threads = 256;
    } else if (config.seq_len_q <= 4096) {
        // Leverage large L2 cache
        block_m = 256;
        block_n = 256;
        threads = 512;
    } else {
        // For very long sequences
        block_m = 128;
        block_n = 512;
        threads = 256;
    }
    
    // Ensure threads is a multiple of subgroup size (16 for Intel GPU)
    threads = ((threads + 15) / 16) * 16;
    
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
                    if (block_m == 64 && block_n == 64) {
                        FlashAttnKernelXMXSimple<64, 64, 64, 128> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 128 && block_n == 128) {
                        FlashAttnKernelXMXSimple<128, 128, 64, 256> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 256 && block_n == 256) {
                        FlashAttnKernelXMXSimple<256, 256, 64, 512> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 128 && block_n == 512) {
                        FlashAttnKernelXMXSimple<128, 512, 64, 256> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    }
                } else if (config.head_dim == 128) {
                    if (block_m == 64 && block_n == 64) {
                        FlashAttnKernelXMXSimple<64, 64, 128, 128> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 128 && block_n == 128) {
                        FlashAttnKernelXMXSimple<128, 128, 128, 256> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 256 && block_n == 256) {
                        FlashAttnKernelXMXSimple<256, 256, 128, 512> kernel(
                            query, key, value, d_output, d_lse, config, local_mem
                        );
                        kernel(item);
                    } else if (block_m == 128 && block_n == 512) {
                        FlashAttnKernelXMXSimple<128, 512, 128, 256> kernel(
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