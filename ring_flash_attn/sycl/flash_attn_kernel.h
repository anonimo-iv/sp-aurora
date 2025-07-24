#ifndef SYCL_FLASH_ATTN_KERNEL_H
#define SYCL_FLASH_ATTN_KERNEL_H

#include <sycl/sycl.hpp>
#include <vector>

namespace flash_attn {

// Forward pass result structure
struct FlashAttnOutput {
    float* output;      // [batch, num_heads, seq_len, head_dim]
    float* lse;         // [batch, num_heads, seq_len] - log sum exp values
    float* debug_info;  // Optional debug information
};

// Configuration structure
struct FlashAttnConfig {
    int batch_size;
    int num_heads;
    int seq_len_q;
    int seq_len_k;
    int head_dim;
    float softmax_scale;
    bool is_causal;
    float dropout_p;
    int block_size_q;
    int block_size_k;
};

// Main kernel function declarations
FlashAttnOutput flash_attn_forward_sycl(
    sycl::queue& q,
    const float* query,     // [batch, num_heads, seq_len_q, head_dim]
    const float* key,       // [batch, num_heads, seq_len_k, head_dim]  
    const float* value,     // [batch, num_heads, seq_len_k, head_dim]
    const FlashAttnConfig& config
);

// Backward pass (placeholder for now)
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
);

// Utility functions
sycl::device get_intel_gpu_device();
void print_device_info(const sycl::device& device);
size_t estimate_memory_usage(const FlashAttnConfig& config);

}  // namespace flash_attn

#endif  // SYCL_FLASH_ATTN_KERNEL_H