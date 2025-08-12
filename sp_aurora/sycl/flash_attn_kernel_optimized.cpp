#include "flash_attn_kernel.h"
#include "utils.h"
#include <iostream>
#include <algorithm>
#include <limits>

namespace flash_attn {

// Optimized kernel with Intel GPU specific features
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM, int WARPS_M, int WARPS_N>
class FlashAttnKernelOptimized {
    static constexpr int THREADS = WARPS_M * WARPS_N * WARP_SIZE;
    static constexpr int THREADS_PER_ROW = WARPS_N * WARP_SIZE;
    
public:
    FlashAttnKernelOptimized(const float* q, const float* k, const float* v,
                             float* out, float* lse,
                             const FlashAttnConfig& cfg)
        : q_global(q), k_global(k), v_global(v),
          out_global(out), lse_global(lse), config(cfg) {}

    void operator()(nd_item<2> item) const {
        // Thread and block indices
        const int tid = item.get_local_id(1);
        const int warp_id = tid / WARP_SIZE;
        const int lane_id = tid % WARP_SIZE;
        const int warp_m = warp_id / WARPS_N;
        const int warp_n = warp_id % WARPS_N;
        
        const int batch_idx = item.get_group(0);
        const int head_idx = item.get_group(1);
        
        // Shared memory layout optimized for bank conflicts
        // Note: In SYCL, we don't use extern __shared__, memory is passed via accessor
        float* q_smem = nullptr;  // Will be initialized from local_accessor
        float* k_smem = q_smem + BLOCK_M * HEAD_DIM;
        float* v_smem = k_smem + BLOCK_N * HEAD_DIM;
        float* s_smem = v_smem + BLOCK_N * HEAD_DIM;
        
        // Registers for accumulation
        float acc[BLOCK_M / WARPS_M] = {0.0f};
        float m_new[BLOCK_M / WARPS_M];
        float l_new[BLOCK_M / WARPS_M];
        
        // Initialize max and sum
        #pragma unroll
        for (int i = 0; i < BLOCK_M / WARPS_M; i++) {
            m_new[i] = -std::numeric_limits<float>::infinity();
            l_new[i] = 0.0f;
        }
        
        // Calculate base pointers
        const int seq_offset = batch_idx * config.num_heads + head_idx;
        const float* q_base = q_global + seq_offset * config.seq_len_q * config.head_dim;
        const float* k_base = k_global + seq_offset * config.seq_len_k * config.head_dim;
        const float* v_base = v_global + seq_offset * config.seq_len_k * config.head_dim;
        
        // Main loop over K/V blocks
        const int num_blocks_k = divUp(config.seq_len_k, BLOCK_N);
        
        for (int block_k = 0; block_k < num_blocks_k; block_k++) {
            const int k_start = block_k * BLOCK_N;
            const int k_end = min(k_start + BLOCK_N, config.seq_len_k);
            const int actual_block_n = k_end - k_start;
            
            // Collaborative loading of K and V tiles with coalescing
            // Each warp loads a portion of the tile
            const int elems_per_thread = (BLOCK_N * HEAD_DIM) / THREADS;
            #pragma unroll
            for (int i = 0; i < elems_per_thread; i++) {
                int idx = tid + i * THREADS;
                int row = idx / HEAD_DIM;
                int col = idx % HEAD_DIM;
                
                if (row < actual_block_n) {
                    k_smem[row * HEAD_DIM + col] = k_base[(k_start + row) * HEAD_DIM + col];
                    v_smem[row * HEAD_DIM + col] = v_base[(k_start + row) * HEAD_DIM + col];
                }
            }
            
            // Load Q tile (only on first iteration)
            if (block_k == 0) {
                const int q_elems_per_thread = (BLOCK_M * HEAD_DIM) / THREADS;
                #pragma unroll
                for (int i = 0; i < q_elems_per_thread; i++) {
                    int idx = tid + i * THREADS;
                    int row = idx / HEAD_DIM;
                    int col = idx % HEAD_DIM;
                    
                    if (row < BLOCK_M && row < config.seq_len_q) {
                        q_smem[row * HEAD_DIM + col] = q_base[row * HEAD_DIM + col];
                    }
                }
            }
            
            item.barrier();
            
            // Compute QK^T for assigned rows
            #pragma unroll
            for (int m = 0; m < BLOCK_M / WARPS_M; m++) {
                const int row = warp_m * (BLOCK_M / WARPS_M) + m;
                if (row >= config.seq_len_q) continue;
                
                float m_prev = m_new[m];
                float l_prev = l_new[m];
                float m_curr = -INFINITY;
                
                // Each thread in warp computes part of the row
                for (int n = lane_id; n < actual_block_n; n += WARP_SIZE) {
                    float score = 0.0f;
                    
                    // Dot product with vectorization hint
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d += 4) {
                        float4 q_vec = *((float4*)&q_smem[row * HEAD_DIM + d]);
                        float4 k_vec = *((float4*)&k_smem[n * HEAD_DIM + d]);
                        score += q_vec.x * k_vec.x + q_vec.y * k_vec.y + 
                                 q_vec.z * k_vec.z + q_vec.w * k_vec.w;
                    }
                    
                    score *= config.softmax_scale;
                    
                    // Causal masking
                    if (config.is_causal && (k_start + n) > row) {
                        score = -INFINITY;
                    }
                    
                    s_smem[row * BLOCK_N + n] = score;
                    m_curr = max(m_curr, score);
                }
                
                // Warp-level reduction for max
                m_curr = warpReduceMax(m_curr, item);
                
                // Compute exponentials and sum
                float l_curr = 0.0f;
                for (int n = lane_id; n < actual_block_n; n += WARP_SIZE) {
                    float exp_score = exp(s_smem[row * BLOCK_N + n] - m_curr);
                    s_smem[row * BLOCK_N + n] = exp_score;
                    l_curr += exp_score;
                }
                
                // Warp-level reduction for sum
                l_curr = warpReduceSum(l_curr, item);
                
                // Update running max and sum
                float m_new_val = max(m_prev, m_curr);
                float l_new_val = exp(m_prev - m_new_val) * l_prev + 
                                  exp(m_curr - m_new_val) * l_curr;
                
                m_new[m] = m_new_val;
                l_new[m] = l_new_val;
                
                // Rescale accumulator
                acc[m] *= exp(m_prev - m_new_val);
                
                // Accumulate weighted values
                for (int n = lane_id; n < actual_block_n; n += WARP_SIZE) {
                    float weight = s_smem[row * BLOCK_N + n] * exp(m_curr - m_new_val);
                    
                    // Vectorized accumulation
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d += 4) {
                        float4 v_vec = *((float4*)&v_smem[n * HEAD_DIM + d]);
                        float4* acc_vec = (float4*)&shared_mem[row * HEAD_DIM + d];
                        acc_vec->x += weight * v_vec.x;
                        acc_vec->y += weight * v_vec.y;
                        acc_vec->z += weight * v_vec.z;
                        acc_vec->w += weight * v_vec.w;
                    }
                }
            }
            
            item.barrier();
        }
        
        // Write output with normalization
        #pragma unroll
        for (int m = 0; m < BLOCK_M / WARPS_M; m++) {
            const int row = warp_m * (BLOCK_M / WARPS_M) + m;
            if (row >= config.seq_len_q) continue;
            
            const float inv_sum = 1.0f / l_new[m];
            const int out_offset = seq_offset * config.seq_len_q * config.head_dim + 
                                   row * config.head_dim;
            
            // Coalesced writes
            for (int d = lane_id; d < config.head_dim; d += WARP_SIZE) {
                out_global[out_offset + d] = shared_mem[row * HEAD_DIM + d] * inv_sum;
            }
            
            // Write LSE (only thread 0 of each warp)
            if (lane_id == 0) {
                const int lse_offset = seq_offset * config.seq_len_q + row;
                lse_global[lse_offset] = m_new[m] + log(l_new[m]);
            }
        }
    }

private:
    const float* q_global;
    const float* k_global;
    const float* v_global;
    float* out_global;
    float* lse_global;
    FlashAttnConfig config;
    
    // Helper for float4 operations
    struct float4 {
        float x, y, z, w;
    };
};

// Optimized forward function with auto-tuning
FlashAttnOutput flash_attn_forward_optimized_sycl(
    sycl::queue& q,
    const float* query,
    const float* key,
    const float* value,
    const FlashAttnConfig& config
) {
    // Auto-tune block sizes based on sequence length
    int block_m, block_n;
    if (config.seq_len_q <= 512) {
        block_m = 64;
        block_n = 64;
    } else if (config.seq_len_q <= 2048) {
        block_m = 128;
        block_n = 64;
    } else {
        block_m = 128;
        block_n = 128;
    }
    
    // Configure warps
    const int warps_m = 4;
    const int warps_n = 2;
    const int threads = warps_m * warps_n * WARP_SIZE;
    
    // Allocate output
    const size_t output_size = config.batch_size * config.num_heads * 
                               config.seq_len_q * config.head_dim;
    const size_t lse_size = config.batch_size * config.num_heads * config.seq_len_q;
    
    float* d_output = sycl::malloc_device<float>(output_size, q);
    float* d_lse = sycl::malloc_device<float>(lse_size, q);
    
    // Calculate shared memory size
    const size_t shmem_size = sizeof(float) * (
        block_m * config.head_dim +  // Q tile + output accumulator
        block_n * config.head_dim +  // K tile  
        block_n * config.head_dim +  // V tile
        block_m * block_n           // Attention scores
    );
    
    // Launch optimized kernel
    sycl::range<2> global_range(config.batch_size, config.num_heads);
    sycl::range<2> local_range(1, threads);
    
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> local_mem(sycl::range<1>(shmem_size / sizeof(float)), h);
        
        h.parallel_for(
            sycl::nd_range<2>(global_range * local_range, local_range),
            [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(WARP_SIZE)]] {
                if (config.head_dim == 64 && block_m == 128 && block_n == 64) {
                    FlashAttnKernelOptimized<128, 64, 64, warps_m, warps_n> kernel(
                        query, key, value, d_output, d_lse, config
                    );
                    kernel(item);
                } else if (config.head_dim == 128 && block_m == 128 && block_n == 128) {
                    FlashAttnKernelOptimized<128, 128, 128, warps_m, warps_n> kernel(
                        query, key, value, d_output, d_lse, config
                    );
                    kernel(item);
                }
                // Add more configurations as needed
            }
        );
    });
    
    q.wait();
    
    return {d_output, d_lse, nullptr};
}

}  // namespace flash_attn