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

namespace dnnl = oneapi::dnnl;

// Helper function to check oneDNN status
#define CHECK_DNNL(call) do { \
    dnnl::status_t status = call; \
    if (status != dnnl::status::success) { \
        std::cerr << "oneDNN error at " << __FILE__ << ":" << __LINE__ << " - " << dnnl_status2str(status) << std::endl; \
        throw std::runtime_error("oneDNN error"); \
    } \
} while(0)

void flash_attention_forward_onednn(
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
        // Create oneDNN engine and stream from SYCL queue
        dnnl::engine eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, q);
        
        // Dimensions for the attention computation
        const int total_seq_len = seq_len;
        
        // Process each batch and head independently for now
        // TODO: Optimize to process multiple heads/batches together
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < num_heads; ++h) {
                // Pointers to current batch/head
                const float* q_ptr = Q + (b * num_heads + h) * seq_len * head_dim;
                const float* k_ptr = K + (b * num_heads + h) * seq_len * head_dim;
                const float* v_ptr = V + (b * num_heads + h) * seq_len * head_dim;
                float* o_ptr = O + (b * num_heads + h) * seq_len * head_dim;
                
                // Step 1: Compute Q @ K^T using oneDNN matmul
                // Q: [seq_len, head_dim], K^T: [head_dim, seq_len] -> S: [seq_len, seq_len]
                
                // Create memory descriptors
                dnnl::memory::desc q_md({seq_len, head_dim}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
                dnnl::memory::desc k_md({head_dim, seq_len}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ba); // Transposed
                dnnl::memory::desc s_md({seq_len, seq_len}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
                
                // Create memory objects
                auto q_mem = dnnl::sycl_interop::make_memory(q_md, eng, dnnl::sycl_interop::memory_kind::usm, const_cast<float*>(q_ptr));
                auto k_mem = dnnl::sycl_interop::make_memory(k_md, eng, dnnl::sycl_interop::memory_kind::usm, const_cast<float*>(k_ptr));
                
                // Allocate temporary buffer for S matrix
                float* s_buffer = sycl::malloc_device<float>(seq_len * seq_len, q);
                auto s_mem = dnnl::sycl_interop::make_memory(s_md, eng, dnnl::sycl_interop::memory_kind::usm, s_buffer);
                
                // Create matmul primitive descriptor for Q @ K^T
                dnnl::matmul::primitive_desc matmul_qk_pd(eng, q_md, k_md, s_md);
                dnnl::matmul matmul_qk(matmul_qk_pd);
                
                // Execute Q @ K^T
                matmul_qk.execute(strm, {
                    {DNNL_ARG_SRC, q_mem},
                    {DNNL_ARG_WEIGHTS, k_mem},
                    {DNNL_ARG_DST, s_mem}
                });
                
                // Step 2: Apply scaling and causal mask
                q.submit([&](sycl::handler& h) {
                    h.parallel_for(sycl::range<2>(seq_len, seq_len), [=](sycl::id<2> idx) {
                        int i = idx[0];
                        int j = idx[1];
                        float val = s_buffer[i * seq_len + j] * scale;
                        
                        // Apply causal mask
                        if (is_causal && j > i) {
                            val = -INFINITY;
                        }
                        
                        s_buffer[i * seq_len + j] = val;
                    });
                });
                
                // Step 3: Apply softmax row-wise
                // oneDNN softmax primitive
                dnnl::memory::desc softmax_md({seq_len, seq_len}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
                dnnl::softmax_forward::primitive_desc softmax_pd(eng, 
                    dnnl::prop_kind::forward_inference,
                    dnnl::algorithm::softmax_accurate,
                    softmax_md, softmax_md, 1); // axis=1 for row-wise softmax
                
                dnnl::softmax_forward softmax(softmax_pd);
                softmax.execute(strm, {
                    {DNNL_ARG_SRC, s_mem},
                    {DNNL_ARG_DST, s_mem}
                });
                
                // Step 4: Compute attention @ V
                // S: [seq_len, seq_len], V: [seq_len, head_dim] -> O: [seq_len, head_dim]
                dnnl::memory::desc v_md({seq_len, head_dim}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
                dnnl::memory::desc o_md({seq_len, head_dim}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
                
                auto v_mem = dnnl::sycl_interop::make_memory(v_md, eng, dnnl::sycl_interop::memory_kind::usm, const_cast<float*>(v_ptr));
                auto o_mem = dnnl::sycl_interop::make_memory(o_md, eng, dnnl::sycl_interop::memory_kind::usm, o_ptr);
                
                // Create matmul primitive descriptor for attention @ V
                dnnl::matmul::primitive_desc matmul_sv_pd(eng, s_md, v_md, o_md);
                dnnl::matmul matmul_sv(matmul_sv_pd);
                
                // Execute attention @ V
                matmul_sv.execute(strm, {
                    {DNNL_ARG_SRC, s_mem},
                    {DNNL_ARG_WEIGHTS, v_mem},
                    {DNNL_ARG_DST, o_mem}
                });
                
                // Clean up temporary buffer
                sycl::free(s_buffer, q);
            }
        }
        
        // Wait for all operations to complete
        strm.wait();
        
    } catch (const dnnl::error& e) {
        std::cerr << "oneDNN error: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Error in flash_attention_forward_onednn: " << e.what() << std::endl;
        throw;
    }
}

// Optimized version with fused operations and better memory layout
void flash_attention_forward_onednn_optimized(
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
        
        // Process in larger batches for better efficiency
        const int batch_heads = batch_size * num_heads;
        
        // Reshape tensors to [batch_size * num_heads, seq_len, head_dim]
        // This allows us to process multiple heads at once
        dnnl::memory::desc qkv_md({batch_heads, seq_len, head_dim}, 
                                  dnnl::memory::data_type::f32, 
                                  dnnl::memory::format_tag::abc);
        
        // Memory descriptors for intermediate results
        dnnl::memory::desc scores_md({batch_heads, seq_len, seq_len}, 
                                     dnnl::memory::data_type::f32, 
                                     dnnl::memory::format_tag::abc);
        
        // Create memory objects
        auto q_mem = dnnl::sycl_interop::make_memory(qkv_md, eng, 
                                                     dnnl::sycl_interop::memory_kind::usm, 
                                                     const_cast<float*>(Q));
        auto k_mem = dnnl::sycl_interop::make_memory(qkv_md, eng, 
                                                     dnnl::sycl_interop::memory_kind::usm, 
                                                     const_cast<float*>(K));
        auto v_mem = dnnl::sycl_interop::make_memory(qkv_md, eng, 
                                                     dnnl::sycl_interop::memory_kind::usm, 
                                                     const_cast<float*>(V));
        auto o_mem = dnnl::sycl_interop::make_memory(qkv_md, eng, 
                                                     dnnl::sycl_interop::memory_kind::usm, 
                                                     O);
        
        // Allocate temporary buffer for scores
        float* scores_buffer = sycl::malloc_device<float>(batch_heads * seq_len * seq_len, q);
        auto scores_mem = dnnl::sycl_interop::make_memory(scores_md, eng, 
                                                          dnnl::sycl_interop::memory_kind::usm, 
                                                          scores_buffer);
        
        // Step 1: Batch matmul Q @ K^T with scaling
        // Use post-ops to apply scaling in the same operation
        dnnl::post_ops po;
        po.append_eltwise(dnnl::algorithm::eltwise_linear, scale, 0.0f);
        
        dnnl::primitive_attr attr;
        attr.set_post_ops(po);
        
        // Create batch matmul primitive for Q @ K^T
        dnnl::matmul::primitive_desc bmm_qk_pd(eng, qkv_md, qkv_md, scores_md, attr);
        dnnl::matmul bmm_qk(bmm_qk_pd);
        
        // Need to transpose K for the multiplication
        // Create a transposed view of K
        dnnl::memory::desc k_transposed_md({batch_heads, head_dim, seq_len}, 
                                           dnnl::memory::data_type::f32, 
                                           {head_dim * seq_len, 1, head_dim}); // Custom strides for transpose
        
        // Execute Q @ K^T with scaling
        bmm_qk.execute(strm, {
            {DNNL_ARG_SRC, q_mem},
            {DNNL_ARG_WEIGHTS, k_mem},
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
        }
        
        // Step 3: Softmax along last dimension
        dnnl::softmax_forward::primitive_desc softmax_pd(eng,
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::softmax_accurate,
            scores_md, scores_md, 2); // axis=2 for last dimension
        
        dnnl::softmax_forward softmax(softmax_pd);
        softmax.execute(strm, {
            {DNNL_ARG_SRC, scores_mem},
            {DNNL_ARG_DST, scores_mem}
        });
        
        // Step 4: Batch matmul attention @ V
        dnnl::matmul::primitive_desc bmm_sv_pd(eng, scores_md, qkv_md, qkv_md);
        dnnl::matmul bmm_sv(bmm_sv_pd);
        
        bmm_sv.execute(strm, {
            {DNNL_ARG_SRC, scores_mem},
            {DNNL_ARG_WEIGHTS, v_mem},
            {DNNL_ARG_DST, o_mem}
        });
        
        // Clean up
        sycl::free(scores_buffer, q);
        strm.wait();
        
    } catch (const dnnl::error& e) {
        std::cerr << "oneDNN error: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Error in flash_attention_forward_onednn_optimized: " << e.what() << std::endl;
        throw;
    }
}

// Entry point that selects the best implementation
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
    
    // Use optimized version for larger sequence lengths
    if (seq_len >= 256) {
        flash_attention_forward_onednn_optimized(q, Q, K, V, O, batch_size, num_heads, 
                                                seq_len, head_dim, scale, is_causal);
    } else {
        flash_attention_forward_onednn(q, Q, K, V, O, batch_size, num_heads, 
                                      seq_len, head_dim, scale, is_causal);
    }
}