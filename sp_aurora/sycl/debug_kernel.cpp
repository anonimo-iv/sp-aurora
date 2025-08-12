#include "flash_attn_kernel.h"
#include <iostream>

namespace flash_attn {

// Simple debug kernel to test basic functionality
void debug_kernel(sycl::queue& q, float* data, int size) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
            data[idx] = static_cast<float>(idx[0]) + 1.0f;
        });
    });
    q.wait();
}

// Test if we can write to output at all
FlashAttnOutput flash_attn_debug_test(
    sycl::queue& q,
    const float* query,
    const float* key, 
    const float* value,
    const FlashAttnConfig& config
) {
    size_t output_size = config.batch_size * config.num_heads * config.seq_len_q * config.head_dim;
    size_t lse_size = config.batch_size * config.num_heads * config.seq_len_q;
    
    float* d_output = sycl::malloc_device<float>(output_size, q);
    float* d_lse = sycl::malloc_device<float>(lse_size, q);
    
    // Simple test: just copy query to output
    q.memcpy(d_output, query, output_size * sizeof(float));
    
    // Set LSE to a constant
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(lse_size), [=](sycl::id<1> idx) {
            d_lse[idx[0]] = 1.0f;
        });
    });
    
    q.wait();
    
    return {d_output, d_lse, nullptr};
}

}  // namespace flash_attn