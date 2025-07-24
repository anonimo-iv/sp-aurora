#include "flash_attn_kernel.h"
#include "utils.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <limits>

namespace flash_attn {

// Forward declaration for SYCL_EXTERNAL
template<int BLOCK_Q, int BLOCK_K, int HEAD_DIM>
class FlashAttnKernel;

// Kernel implementation for flash attention forward pass
template<int BLOCK_Q, int BLOCK_K, int HEAD_DIM>
class FlashAttnKernel {
private:
    const float* q_global;
    const float* k_global;
    const float* v_global;
    float* out_global;
    float* lse_global;
    FlashAttnConfig config;
    sycl::local_accessor<float, 1> local_mem;

public:
    FlashAttnKernel(const float* q, const float* k, const float* v,
                     float* out, float* lse,
                     const FlashAttnConfig& cfg,
                     const sycl::local_accessor<float, 1>& lm)
        : q_global(q), k_global(k), v_global(v),
          out_global(out), lse_global(lse), config(cfg), local_mem(lm) {}

    void operator()(sycl::nd_item<2> item) const {
        // Get thread indices
        const int tid = item.get_local_id(1);
        const int batch_idx = item.get_group(0);
        const int head_idx = item.get_group(1);
        
        // Calculate global offsets
        const int q_batch_offset = batch_idx * config.num_heads * config.seq_len_q * config.head_dim;
        const int k_batch_offset = batch_idx * config.num_heads * config.seq_len_k * config.head_dim;
        const int q_head_offset = head_idx * config.seq_len_q * config.head_dim;
        const int k_head_offset = head_idx * config.seq_len_k * config.head_dim;
        
        // Shared memory layout
        float* shared_mem = local_mem.get_multi_ptr<sycl::access::decorated::no>().get();
        float* q_shared = shared_mem;
        float* k_shared = q_shared + BLOCK_Q * HEAD_DIM;
        float* v_shared = k_shared + BLOCK_K * HEAD_DIM;
        float* s_shared = v_shared + BLOCK_K * HEAD_DIM;  // For attention scores
        
        // Thread-local accumulators
        float acc[BLOCK_Q][HEAD_DIM];
        float row_max[BLOCK_Q];
        float row_sum[BLOCK_Q];
        
        // Initialize accumulators
        for (int i = 0; i < BLOCK_Q; i++) {
            row_max[i] = -std::numeric_limits<float>::infinity();
            row_sum[i] = 0.0f;
            for (int j = 0; j < HEAD_DIM; j++) {
                acc[i][j] = 0.0f;
            }
        }
        
        // Loop over K/V blocks
        const int num_blocks_k = divUp(config.seq_len_k, BLOCK_K);
        
        for (int block_k = 0; block_k < num_blocks_k; block_k++) {
            const int k_start = block_k * BLOCK_K;
            
            // Load Q block (reuse for all K blocks)
            if (block_k == 0) {
                loadTile<float, BLOCK_Q>(
                    q_global + q_batch_offset + q_head_offset,
                    q_shared,
                    0, 0,
                    config.seq_len_q, config.head_dim,
                    config.head_dim,
                    item
                );
            }
            
            // Load K block
            loadTile<float, BLOCK_K>(
                k_global + k_batch_offset + k_head_offset + k_start * config.head_dim,
                k_shared,
                0, 0,
                std::min(BLOCK_K, config.seq_len_k - k_start), config.head_dim,
                config.head_dim,
                item
            );
            
            // Load V block
            loadTile<float, BLOCK_K>(
                v_global + k_batch_offset + k_head_offset + k_start * config.head_dim,
                v_shared,
                0, 0,
                std::min(BLOCK_K, config.seq_len_k - k_start), config.head_dim,
                config.head_dim,
                item
            );
            
            item.barrier();
            
            // Compute QK^T for this block
            for (int q_idx = tid; q_idx < BLOCK_Q; q_idx += item.get_local_range(1)) {
                float local_max = row_max[q_idx];
                float local_sum = 0.0f;
                
                // Compute attention scores for this Q row
                for (int k_idx = 0; k_idx < BLOCK_K; k_idx++) {
                    float score = 0.0f;
                    
                    // Dot product Q[q_idx] · K[k_idx]
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d++) {
                        score += q_shared[q_idx * HEAD_DIM + d] * k_shared[k_idx * HEAD_DIM + d];
                    }
                    
                    score *= config.softmax_scale;
                    
                    // Apply causal mask if needed
                    if (config.is_causal) {
                        int global_q_idx = q_idx;  // Assuming we process Q sequentially
                        int global_k_idx = k_start + k_idx;
                        if (global_k_idx > global_q_idx) {
                            score = -std::numeric_limits<float>::infinity();
                        }
                    }
                    
                    s_shared[q_idx * BLOCK_K + k_idx] = score;
                    local_max = sycl::max(local_max, score);
                }
                
                // Update max for numerical stability
                float max_diff = local_max - row_max[q_idx];
                float scale = sycl::exp(max_diff);
                
                // Rescale previous sum
                row_sum[q_idx] *= sycl::exp(-max_diff);
                
                // Compute exp and sum for current block
                for (int k_idx = 0; k_idx < BLOCK_K; k_idx++) {
                    float exp_score = sycl::exp(s_shared[q_idx * BLOCK_K + k_idx] - local_max);
                    s_shared[q_idx * BLOCK_K + k_idx] = exp_score;
                    local_sum += exp_score;
                }
                
                // Update global max and sum
                row_max[q_idx] = local_max;
                row_sum[q_idx] += local_sum;
                
                // Rescale accumulated values
                for (int d = 0; d < HEAD_DIM; d++) {
                    acc[q_idx][d] *= sycl::exp(-max_diff);
                }
                
                // Accumulate V weighted by attention scores
                for (int k_idx = 0; k_idx < BLOCK_K; k_idx++) {
                    float attn_weight = s_shared[q_idx * BLOCK_K + k_idx];
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d++) {
                        acc[q_idx][d] += attn_weight * v_shared[k_idx * HEAD_DIM + d];
                    }
                }
            }
            
            item.barrier();
        }
        
        // Write output and LSE
        for (int q_idx = tid; q_idx < BLOCK_Q; q_idx += item.get_local_range(1)) {
            const int out_offset = q_batch_offset + q_head_offset + q_idx * config.head_dim;
            const int lse_offset = batch_idx * config.num_heads * config.seq_len_q + 
                                   head_idx * config.seq_len_q + q_idx;
            
            // Normalize by sum
            float inv_sum = 1.0f / row_sum[q_idx];
            
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                out_global[out_offset + d] = acc[q_idx][d] * inv_sum;
            }
            
            // Write log-sum-exp
            lse_global[lse_offset] = sycl::log(row_sum[q_idx]) + row_max[q_idx];
        }
    }
};

// Main kernel launch function
FlashAttnOutput flash_attn_forward_sycl(
    sycl::queue& q,
    const float* query,
    const float* key,
    const float* value,
    const FlashAttnConfig& config
) {
    // Allocate output tensors
    size_t output_size = config.batch_size * config.num_heads * config.seq_len_q * config.head_dim;
    size_t lse_size = config.batch_size * config.num_heads * config.seq_len_q;
    
    float* d_output = sycl::malloc_device<float>(output_size, q);
    float* d_lse = sycl::malloc_device<float>(lse_size, q);
    
    // Initialize outputs
    q.memset(d_output, 0, output_size * sizeof(float));
    
    // Set block sizes
    const int BLOCK_Q = 64;
    const int BLOCK_K = 64;
    const int THREADS = 256;
    
    // Calculate shared memory size
    const size_t shmem_size = sizeof(float) * (
        BLOCK_Q * config.head_dim +  // Q tile
        BLOCK_K * config.head_dim +  // K tile  
        BLOCK_K * config.head_dim +  // V tile
        BLOCK_Q * BLOCK_K            // Attention scores
    );
    
    // Launch kernel
    sycl::range<2> global_range(config.batch_size, config.num_heads);
    sycl::range<2> local_range(1, THREADS);
    
    q.submit([&](sycl::handler& h) {
        // Request local memory
        sycl::local_accessor<float, 1> local_mem(sycl::range<1>(shmem_size / sizeof(float)), h);
        
        h.parallel_for(
            sycl::nd_range<2>(global_range * local_range, local_range),
            [=](sycl::nd_item<2> item) {
                if (config.head_dim == 64) {
                    FlashAttnKernel<BLOCK_Q, BLOCK_K, 64> kernel(
                        query, key, value, d_output, d_lse, config, local_mem
                    );
                    kernel(item);
                } else if (config.head_dim == 128) {
                    FlashAttnKernel<BLOCK_Q, BLOCK_K, 128> kernel(
                        query, key, value, d_output, d_lse, config, local_mem
                    );
                    kernel(item);
                }
                // Add more head dimensions as needed
            }
        );
    });
    
    q.wait();
    
    return {d_output, d_lse, nullptr};
}

// Utility functions
sycl::device get_intel_gpu_device() {
    auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    
    for (const auto& device : gpu_devices) {
        if (device.get_info<sycl::info::device::vendor>().find("Intel") != std::string::npos) {
            return device;
        }
    }
    
    throw std::runtime_error("No Intel GPU found");
}

void print_device_info(const sycl::device& device) {
    std::cout << "Device: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Vendor: " << device.get_info<sycl::info::device::vendor>() << std::endl;
    std::cout << "Max compute units: " << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "Max work group size: " << device.get_info<sycl::info::device::max_work_group_size>() << std::endl;
    std::cout << "Local memory size: " << device.get_info<sycl::info::device::local_mem_size>() << " bytes" << std::endl;
}

size_t estimate_memory_usage(const FlashAttnConfig& config) {
    size_t input_mem = config.batch_size * config.num_heads * 
                       (config.seq_len_q + 2 * config.seq_len_k) * config.head_dim * sizeof(float);
    size_t output_mem = config.batch_size * config.num_heads * config.seq_len_q * 
                        (config.head_dim + 1) * sizeof(float);
    return input_mem + output_mem;
}

// Backward pass placeholder
void flash_attn_backward_sycl(
    sycl::queue& q,
    const float* dout,
    const float* query,
    const float* key,
    const float* value,
    const float* output,
    const float* lse,
    float* dq,
    float* dk,
    float* dv,
    const FlashAttnConfig& config
) {
    // TODO: Implement backward pass
    throw std::runtime_error("Backward pass not yet implemented");
}

}  // namespace flash_attn