#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include "flash_attn_kernel.h"
#include "utils.h"

// namespace dnnl = oneapi::dnnl;  // Already defined in dnnl_sycl.hpp
using namespace dnnl;

// Optimized oneDNN implementation with proper transpose handling
void flash_attention_forward_onednn_v2(
    sycl::queue& q,
    const float* Q,    // [batch_size, num_heads, seq_len, head_dim]
    const float* K,    // [batch_size, num_heads, seq_len, head_dim]
    const float* V,    // [batch_size, num_heads, seq_len, head_dim]
    float* O,          // [batch_size, num_heads, seq_len, head_dim]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool is_causal) {
    
    try {
        // Create oneDNN engine and stream
        dnnl::engine eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, q);
        
        // Use blocked memory format for better cache efficiency
        const int batch_heads = batch_size * num_heads;
        
        // Memory descriptors with optimized formats
        // Using blocked format for better performance on Intel GPU
        dnnl::memory::desc q_md({batch_heads, seq_len, head_dim}, 
                                dnnl::memory::data_type::f32,
                                dnnl::memory::format_tag::abc);
        
        dnnl::memory::desc k_md({batch_heads, seq_len, head_dim}, 
                                dnnl::memory::data_type::f32,
                                dnnl::memory::format_tag::abc);
        
        dnnl::memory::desc v_md({batch_heads, seq_len, head_dim}, 
                                dnnl::memory::data_type::f32,
                                dnnl::memory::format_tag::abc);
        
        dnnl::memory::desc o_md({batch_heads, seq_len, head_dim}, 
                                dnnl::memory::data_type::f32,
                                dnnl::memory::format_tag::abc);
        
        // Create memory objects
        auto q_mem = dnnl::sycl_interop::make_memory(q_md, eng, 
                                                     dnnl::sycl_interop::memory_kind::usm, 
                                                     const_cast<float*>(Q));
        auto k_mem = dnnl::sycl_interop::make_memory(k_md, eng, 
                                                     dnnl::sycl_interop::memory_kind::usm, 
                                                     const_cast<float*>(K));
        auto v_mem = dnnl::sycl_interop::make_memory(v_md, eng, 
                                                     dnnl::sycl_interop::memory_kind::usm, 
                                                     const_cast<float*>(V));
        auto o_mem = dnnl::sycl_interop::make_memory(o_md, eng, 
                                                     dnnl::sycl_interop::memory_kind::usm, 
                                                     O);
        
        // Allocate temporary buffers
        float* scores_buffer = sycl::malloc_device<float>(batch_heads * seq_len * seq_len, q);
        
        // Memory descriptor for scores
        dnnl::memory::desc scores_md({batch_heads, seq_len, seq_len}, 
                                     dnnl::memory::data_type::f32,
                                     dnnl::memory::format_tag::abc);
        
        auto scores_mem = dnnl::sycl_interop::make_memory(scores_md, eng, 
                                                          dnnl::sycl_interop::memory_kind::usm, 
                                                          scores_buffer);
        
        // Step 1: Compute Q @ K^T with scaling
        // We need to properly transpose K using a simple SYCL kernel
        float* k_transposed_buffer = sycl::malloc_device<float>(batch_heads * head_dim * seq_len, q);
        
        // Transpose K from [batch_heads, seq_len, head_dim] to [batch_heads, head_dim, seq_len]
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<3>(batch_heads, seq_len, head_dim), 
                          [=](sycl::id<3> idx) {
                int b = idx[0];
                int s = idx[1];
                int d = idx[2];
                
                // Input index: [b, s, d]
                int in_idx = b * seq_len * head_dim + s * head_dim + d;
                // Output index: [b, d, s]
                int out_idx = b * head_dim * seq_len + d * seq_len + s;
                
                k_transposed_buffer[out_idx] = K[in_idx];
            });
        });
        q.wait();
        
        // Create memory descriptor for transposed K
        dnnl::memory::desc k_transposed_md({batch_heads, head_dim, seq_len}, 
                                           dnnl::memory::data_type::f32,
                                           dnnl::memory::format_tag::abc);
        
        auto k_transposed_mem = dnnl::sycl_interop::make_memory(k_transposed_md, eng,
                                                                dnnl::sycl_interop::memory_kind::usm,
                                                                k_transposed_buffer);
        
        // Create matmul primitive with scaling
        dnnl::primitive_attr matmul_attr;
        dnnl::post_ops po;
        po.append_eltwise(dnnl::algorithm::eltwise_linear, scale, 0.0f);
        matmul_attr.set_post_ops(po);
        
        // Create matmul primitive descriptor
        auto matmul_qk_pd = dnnl::matmul::primitive_desc(eng, q_md, k_transposed_md, scores_md, matmul_attr);
        auto matmul_qk = dnnl::matmul(matmul_qk_pd);
        
        // Execute Q @ K^T
        matmul_qk.execute(strm, {
            {DNNL_ARG_SRC, q_mem},
            {DNNL_ARG_WEIGHTS, k_transposed_mem},
            {DNNL_ARG_DST, scores_mem}
        });
        
        // Step 2: Apply causal mask if needed
        if (is_causal) {
            q.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<3>(batch_heads, seq_len, seq_len), 
                              [=](sycl::id<3> idx) {
                    int b = idx[0];
                    int i = idx[1];
                    int j = idx[2];
                    
                    if (j > i) {
                        scores_buffer[b * seq_len * seq_len + i * seq_len + j] = -INFINITY;
                    }
                });
            });
            q.wait();  // Ensure mask is applied before softmax
        }
        
        // Step 3: Apply softmax
        // Use softmax primitive descriptor constructor directly
        auto softmax_pd = dnnl::softmax_forward::primitive_desc(eng,
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::softmax_accurate,
            scores_md, scores_md, 2);  // axis=2
        auto softmax = dnnl::softmax_forward(softmax_pd);
        
        softmax.execute(strm, {
            {DNNL_ARG_SRC, scores_mem},
            {DNNL_ARG_DST, scores_mem}
        });
        
        // Step 4: Compute attention @ V
        auto matmul_sv_pd = dnnl::matmul::primitive_desc(eng, scores_md, v_md, o_md);
        auto matmul_sv = dnnl::matmul(matmul_sv_pd);
        
        matmul_sv.execute(strm, {
            {DNNL_ARG_SRC, scores_mem},
            {DNNL_ARG_WEIGHTS, v_mem},
            {DNNL_ARG_DST, o_mem}
        });
        
        // Wait for completion
        strm.wait();
        
        // Clean up
        sycl::free(scores_buffer, q);
        sycl::free(k_transposed_buffer, q);
        
    } catch (const dnnl::error& e) {
        std::cerr << "oneDNN error: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Error in flash_attention_forward_onednn_v2: " << e.what() << std::endl;
        throw;
    }
}

namespace flash_attn {

// Main entry point - use V2 implementation
void launch_flash_attention_forward_onednn(
    sycl::queue& q,
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool is_causal) {
    
    flash_attention_forward_onednn_v2(q, Q, K, V, O, batch_size, num_heads, 
                                     seq_len, head_dim, scale, is_causal);
}

}  // namespace flash_attn