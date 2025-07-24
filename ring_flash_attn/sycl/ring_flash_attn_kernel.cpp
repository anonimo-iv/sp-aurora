#include "flash_attn_kernel.h"
#include "utils.h"
#include <cstring>
#include <limits>

namespace flash_attn {

// Ring flash attention data structure
struct RingFlashAttnData {
    float* q_local;       // Local query chunk
    float* k_buffer;      // Ring buffer for K
    float* v_buffer;      // Ring buffer for V
    float* out_accum;     // Accumulated output
    float* lse_accum;     // Accumulated LSE
    float* m_accum;       // Running max values
    float* l_accum;       // Running sum values
    int local_seq_len;    // Length of local sequence chunk
    int rank;             // Process rank
    int world_size;       // Total number of processes
};

// Ring-aware kernel that processes local Q against ring-communicated K/V
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
class RingFlashAttnKernel {
private:
    RingFlashAttnData ring_data;
    FlashAttnConfig config;
    int current_step;
    sycl::local_accessor<float, 1> local_mem;

public:
    RingFlashAttnKernel(const RingFlashAttnData& data,
                        const FlashAttnConfig& cfg,
                        int ring_step,
                        const sycl::local_accessor<float, 1>& lm)
        : ring_data(data), config(cfg), current_step(ring_step), local_mem(lm) {}

    void operator()(sycl::nd_item<2> item) const {
        const int tid = item.get_local_id(1);
        const int batch_idx = item.get_group(0);
        const int head_idx = item.get_group(1);
        
        // Shared memory layout
        float* shared_mem = local_mem.get_multi_ptr<sycl::access::decorated::no>().get();
        float* q_smem = shared_mem;
        float* k_smem = q_smem + BLOCK_M * HEAD_DIM;
        float* v_smem = k_smem + BLOCK_N * HEAD_DIM;
        float* s_smem = v_smem + BLOCK_N * HEAD_DIM;
        
        // Calculate which K/V chunk we're processing in this ring step
        const int source_rank = (ring_data.rank - current_step + ring_data.world_size) % ring_data.world_size;
        const int k_seq_offset = source_rank * ring_data.local_seq_len;
        
        // Base pointers
        const int seq_offset = batch_idx * config.num_heads + head_idx;
        const float* q_base = ring_data.q_local + seq_offset * ring_data.local_seq_len * config.head_dim;
        const float* k_base = ring_data.k_buffer + seq_offset * ring_data.local_seq_len * config.head_dim;
        const float* v_base = ring_data.v_buffer + seq_offset * ring_data.local_seq_len * config.head_dim;
        
        float* out_base = ring_data.out_accum + seq_offset * ring_data.local_seq_len * config.head_dim;
        float* m_base = ring_data.m_accum + seq_offset * ring_data.local_seq_len;
        float* l_base = ring_data.l_accum + seq_offset * ring_data.local_seq_len;
        
        // Process tiles
        const int num_tiles_m = divUp(ring_data.local_seq_len, BLOCK_M);
        const int num_tiles_n = divUp(ring_data.local_seq_len, BLOCK_N);
        
        for (int tile_m = 0; tile_m < num_tiles_m; tile_m++) {
            const int m_start = tile_m * BLOCK_M;
            const int m_size = sycl::min(BLOCK_M, ring_data.local_seq_len - m_start);
            
            // Load Q tile
            loadTile<float, BLOCK_M>(
                q_base + m_start * config.head_dim,
                q_smem,
                0, 0,
                m_size, config.head_dim,
                config.head_dim,
                item
            );
            
            // Thread-local accumulators for this Q tile
            float acc[BLOCK_M][HEAD_DIM];
            float row_max[BLOCK_M];
            float row_sum[BLOCK_M];
            
            // Initialize from previous ring steps
            if (current_step == 0) {
                for (int i = 0; i < BLOCK_M; i++) {
                    if (m_start + i < ring_data.local_seq_len) {
                        row_max[i] = -std::numeric_limits<float>::infinity();
                        row_sum[i] = 0.0f;
                        for (int d = 0; d < HEAD_DIM; d++) {
                            acc[i][d] = 0.0f;
                        }
                    }
                }
            } else {
                // Load previous accumulated values
                for (int i = tid; i < BLOCK_M && m_start + i < ring_data.local_seq_len; i += item.get_local_range(1)) {
                    row_max[i] = m_base[m_start + i];
                    row_sum[i] = l_base[m_start + i];
                    for (int d = 0; d < HEAD_DIM; d++) {
                        acc[i][d] = out_base[(m_start + i) * config.head_dim + d] * row_sum[i];
                    }
                }
            }
            
            item.barrier();
            
            // Process K/V tiles
            for (int tile_n = 0; tile_n < num_tiles_n; tile_n++) {
                const int n_start = tile_n * BLOCK_N;
                const int n_size = sycl::min(BLOCK_N, ring_data.local_seq_len - n_start);
                
                // Load K tile
                loadTile<float, BLOCK_N>(
                    k_base + n_start * config.head_dim,
                    k_smem,
                    0, 0,
                    n_size, config.head_dim,
                    config.head_dim,
                    item
                );
                
                // Load V tile
                loadTile<float, BLOCK_N>(
                    v_base + n_start * config.head_dim,
                    v_smem,
                    0, 0,
                    n_size, config.head_dim,
                    config.head_dim,
                    item
                );
                
                item.barrier();
                
                // Compute attention scores
                for (int m = tid; m < m_size; m += item.get_local_range(1)) {
                    float m_curr = -std::numeric_limits<float>::infinity();
                    float l_curr = 0.0f;
                    
                    // Compute scores
                    for (int n = 0; n < n_size; n++) {
                        float score = 0.0f;
                        
                        #pragma unroll
                        for (int d = 0; d < HEAD_DIM; d++) {
                            score += q_smem[m * HEAD_DIM + d] * k_smem[n * HEAD_DIM + d];
                        }
                        
                        score *= config.softmax_scale;
                        
                        // Apply causal mask if needed
                        if (config.is_causal) {
                            int global_m = ring_data.rank * ring_data.local_seq_len + m_start + m;
                            int global_n = k_seq_offset + n_start + n;
                            if (global_n > global_m) {
                                score = -std::numeric_limits<float>::infinity();
                            }
                        }
                        
                        s_smem[m * BLOCK_N + n] = score;
                        m_curr = sycl::max(m_curr, score);
                    }
                    
                    // Update running max
                    float m_prev = row_max[m];
                    float m_new = sycl::max(m_prev, m_curr);
                    row_max[m] = m_new;
                    
                    // Compute exponentials and sum
                    float scale_prev = sycl::exp(m_prev - m_new);
                    float scale_curr = sycl::exp(m_curr - m_new);
                    
                    for (int n = 0; n < n_size; n++) {
                        float exp_score = sycl::exp(s_smem[m * BLOCK_N + n] - m_new);
                        s_smem[m * BLOCK_N + n] = exp_score;
                        l_curr += exp_score;
                    }
                    
                    // Update running sum
                    float l_prev = row_sum[m];
                    float l_new = l_prev * scale_prev + l_curr * scale_curr;
                    row_sum[m] = l_new;
                    
                    // Rescale accumulator
                    for (int d = 0; d < HEAD_DIM; d++) {
                        acc[m][d] *= scale_prev;
                    }
                    
                    // Accumulate V
                    for (int n = 0; n < n_size; n++) {
                        float attn_weight = s_smem[m * BLOCK_N + n] * scale_curr;
                        #pragma unroll
                        for (int d = 0; d < HEAD_DIM; d++) {
                            acc[m][d] += attn_weight * v_smem[n * HEAD_DIM + d];
                        }
                    }
                }
                
                item.barrier();
            }
            
            // Write back accumulated values
            for (int m = tid; m < m_size && m_start + m < ring_data.local_seq_len; m += item.get_local_range(1)) {
                m_base[m_start + m] = row_max[m];
                l_base[m_start + m] = row_sum[m];
                
                float inv_sum = 1.0f / row_sum[m];
                for (int d = 0; d < HEAD_DIM; d++) {
                    out_base[(m_start + m) * config.head_dim + d] = acc[m][d] * inv_sum;
                }
            }
            
            item.barrier();
        }
    }
};

// Launch ring flash attention kernel for one ring step
void launch_ring_flash_attn_step(
    sycl::queue& q,
    const RingFlashAttnData& ring_data,
    const FlashAttnConfig& config,
    int ring_step
) {
    const int BLOCK_M = 64;
    const int BLOCK_N = 64;
    const int THREADS = 256;
    
    // Calculate shared memory size
    const size_t shmem_size = sizeof(float) * (
        BLOCK_M * config.head_dim +  // Q tile
        BLOCK_N * config.head_dim +  // K tile  
        BLOCK_N * config.head_dim +  // V tile
        BLOCK_M * BLOCK_N           // Attention scores
    );
    
    sycl::range<2> global_range(config.batch_size, config.num_heads);
    sycl::range<2> local_range(1, THREADS);
    
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> local_mem(sycl::range<1>(shmem_size / sizeof(float)), h);
        
        h.parallel_for(
            sycl::nd_range<2>(global_range * local_range, local_range),
            [=](sycl::nd_item<2> item) {
                if (config.head_dim == 64) {
                    RingFlashAttnKernel<BLOCK_M, BLOCK_N, 64> kernel(
                        ring_data, config, ring_step, local_mem
                    );
                    kernel(item);
                } else if (config.head_dim == 128) {
                    RingFlashAttnKernel<BLOCK_M, BLOCK_N, 128> kernel(
                        ring_data, config, ring_step, local_mem
                    );
                    kernel(item);
                }
            }
        );
    });
}

// Complete ring flash attention forward pass
FlashAttnOutput ring_flash_attn_forward_sycl(
    sycl::queue& q,
    const float* query,     // Full Q tensor
    const float* key,       // Full K tensor
    const float* value,     // Full V tensor
    const FlashAttnConfig& config,
    int rank,
    int world_size
) {
    // Calculate local sequence length
    int local_seq_len = config.seq_len_q / world_size;
    
    // Allocate device memory
    size_t local_size = config.batch_size * config.num_heads * local_seq_len * config.head_dim;
    size_t lse_size = config.batch_size * config.num_heads * local_seq_len;
    
    // Allocate output and intermediate buffers
    float* d_output = sycl::malloc_device<float>(local_size, q);
    float* d_lse = sycl::malloc_device<float>(lse_size, q);
    float* d_m_accum = sycl::malloc_device<float>(lse_size, q);
    float* d_l_accum = sycl::malloc_device<float>(lse_size, q);
    
    // Allocate ring communication buffers
    float* d_k_buffer = sycl::malloc_device<float>(local_size, q);
    float* d_v_buffer = sycl::malloc_device<float>(local_size, q);
    
    // Get local Q chunk (assuming already distributed)
    const float* d_q_local = query + rank * local_size;
    
    // Initialize accumulators
    q.fill(d_m_accum, -std::numeric_limits<float>::infinity(), lse_size);
    q.memset(d_l_accum, 0, lse_size * sizeof(float));
    q.memset(d_output, 0, local_size * sizeof(float));
    
    // Create ring data structure
    RingFlashAttnData ring_data = {
        const_cast<float*>(d_q_local),
        d_k_buffer,
        d_v_buffer,
        d_output,
        d_lse,
        d_m_accum,
        d_l_accum,
        local_seq_len,
        rank,
        world_size
    };
    
    // Ring communication loop
    for (int step = 0; step < world_size; step++) {
        // Copy K/V chunks to ring buffers (in real implementation, use MPI/NCCL)
        int source_rank = (rank - step + world_size) % world_size;
        const float* k_src = key + source_rank * local_size;
        const float* v_src = value + source_rank * local_size;
        
        q.memcpy(d_k_buffer, k_src, local_size * sizeof(float));
        q.memcpy(d_v_buffer, v_src, local_size * sizeof(float));
        q.wait();
        
        // Launch kernel for this ring step
        launch_ring_flash_attn_step(q, ring_data, config, step);
        q.wait();
        
        // Ring communication would happen here
    }
    
    // Compute final LSE values
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(lse_size), [=](sycl::id<1> idx) {
            d_lse[idx] = sycl::log(d_l_accum[idx]) + d_m_accum[idx];
        });
    });
    q.wait();
    
    // Clean up intermediate buffers
    sycl::free(d_m_accum, q);
    sycl::free(d_l_accum, q);
    sycl::free(d_k_buffer, q);
    sycl::free(d_v_buffer, q);
    
    return {d_output, d_lse, nullptr};
}

}  // namespace flash_attn