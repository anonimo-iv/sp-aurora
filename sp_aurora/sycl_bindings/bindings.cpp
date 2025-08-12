#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "../sycl/flash_attn_kernel.h"
#include <sycl/sycl.hpp>
#include <iostream>

namespace py = pybind11;
using namespace flash_attn;

// Global SYCL queue - initialized once
static sycl::queue* global_queue = nullptr;

// Helper to convert PyTorch tensor to raw pointer
template<typename T>
T* tensor_ptr(torch::Tensor& t) {
    return reinterpret_cast<T*>(t.data_ptr());
}

// Initialize SYCL queue for Intel GPU
void init_sycl_queue() {
    if (global_queue == nullptr) {
        try {
            auto device = get_intel_gpu_device();
            global_queue = new sycl::queue(device);
            std::cout << "SYCL Queue initialized on device: " 
                      << device.get_info<sycl::info::device::name>() << std::endl;
        } catch (std::exception& e) {
            throw std::runtime_error("Failed to initialize SYCL queue: " + std::string(e.what()));
        }
    }
}

// Enum for kernel selection
enum class KernelType {
    AUTO = 0,
    OPTIMIZED_V3 = 2,
    XMX = 4,
    OPTIMIZED_V4 = 5,
    OPTIMIZED_V5 = 6,
    OPTIMIZED_V7 = 7,
    OPTIMIZED_V8 = 8,
#ifdef HAS_ONEDNN
    ONEDNN = 9
#endif
};

// Python-exposed forward function with kernel selection
std::tuple<torch::Tensor, torch::Tensor> flash_attn_forward_py(
    torch::Tensor query,    // [batch, num_heads, seq_len_q, head_dim]
    torch::Tensor key,      // [batch, num_heads, seq_len_k, head_dim]
    torch::Tensor value,    // [batch, num_heads, seq_len_k, head_dim]
    float dropout_p,
    float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    int kernel_type = 0  // Default to AUTO
) {
    // Validate inputs
    TORCH_CHECK(query.dim() == 4, "Query must be 4D tensor");
    TORCH_CHECK(key.dim() == 4, "Key must be 4D tensor");
    TORCH_CHECK(value.dim() == 4, "Value must be 4D tensor");
    TORCH_CHECK(query.scalar_type() == torch::kFloat32, "Only float32 is supported");
    
    // Get dimensions
    const int batch_size = query.size(0);
    const int num_heads = query.size(1);
    const int seq_len_q = query.size(2);
    const int seq_len_k = key.size(2);
    const int head_dim = query.size(3);
    
    TORCH_CHECK(key.size(0) == batch_size && key.size(1) == num_heads);
    TORCH_CHECK(value.size(0) == batch_size && value.size(1) == num_heads);
    TORCH_CHECK(key.size(3) == head_dim && value.size(3) == head_dim);
    
    // Ensure tensors are contiguous
    query = query.contiguous();
    key = key.contiguous();
    value = value.contiguous();
    
    // Move tensors to XPU if not already there
    if (query.device().type() != torch::kXPU) {
        query = query.to(torch::kXPU);
        key = key.to(torch::kXPU);
        value = value.to(torch::kXPU);
    }
    
    // Initialize SYCL queue if needed
    init_sycl_queue();
    
    // Configure kernel
    FlashAttnConfig config = {
        batch_size,
        num_heads,
        seq_len_q,
        seq_len_k,
        head_dim,
        softmax_scale > 0 ? softmax_scale : static_cast<float>(1.0f / std::sqrt(head_dim)),
        is_causal,
        dropout_p,
        64,  // block_size_q
        64   // block_size_k
    };
    
    // Dynamic algorithm selection based on sequence length and Intel GPU capabilities
    FlashAttnOutput output;
    
    // Check if Intel Max GPU (has XMX support)
    bool has_xmx = global_queue->get_device().has(sycl::aspect::ext_intel_matrix);
    
    KernelType selected_kernel = static_cast<KernelType>(kernel_type);
    
    // Manual kernel selection or auto selection
    if (selected_kernel == KernelType::AUTO) {
        // Auto selection - use V8 for best performance
        selected_kernel = KernelType::OPTIMIZED_V8;
    }
    
    // Execute selected kernel
    switch (selected_kernel) {
        case KernelType::OPTIMIZED_V3:
            output = flash_attn_forward_optimized_v3_sycl(
                *global_queue,
                tensor_ptr<float>(query),
                tensor_ptr<float>(key),
                tensor_ptr<float>(value),
                config
            );
            break;
            
        case KernelType::XMX:
            if (!has_xmx) {
                std::cerr << "Warning: XMX kernel requested but not supported on this device. "
                          << "Falling back to optimized V3." << std::endl;
                output = flash_attn_forward_optimized_v3_sycl(
                    *global_queue,
                    tensor_ptr<float>(query),
                    tensor_ptr<float>(key),
                    tensor_ptr<float>(value),
                    config
                );
            } else {
                output = flash_attn_forward_xmx_sycl(
                    *global_queue,
                    tensor_ptr<float>(query),
                    tensor_ptr<float>(key),
                    tensor_ptr<float>(value),
                    config
                );
            }
            break;
            
        case KernelType::OPTIMIZED_V4:
            output = flash_attn_forward_optimized_v4_sycl(
                *global_queue,
                tensor_ptr<float>(query),
                tensor_ptr<float>(key),
                tensor_ptr<float>(value),
                config
            );
            break;
            
        case KernelType::OPTIMIZED_V5:
            output = flash_attn_forward_optimized_v5_sycl(
                *global_queue,
                tensor_ptr<float>(query),
                tensor_ptr<float>(key),
                tensor_ptr<float>(value),
                config
            );
            break;
            
        case KernelType::OPTIMIZED_V7:
            output = flash_attn_forward_optimized_v7_sycl(
                *global_queue,
                tensor_ptr<float>(query),
                tensor_ptr<float>(key),
                tensor_ptr<float>(value),
                config
            );
            break;
            
        case KernelType::OPTIMIZED_V8:
            output = flash_attn_forward_optimized_v8_sycl(
                *global_queue,
                tensor_ptr<float>(query),
                tensor_ptr<float>(key),
                tensor_ptr<float>(value),
                config
            );
            break;
            
#ifdef HAS_ONEDNN
        case KernelType::ONEDNN:
            {
                // Allocate output tensors
                auto options = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(torch::kXPU);
                
                torch::Tensor out_tensor = torch::empty(
                    {batch_size, num_heads, seq_len_q, head_dim},
                    options
                );
                
                // Call oneDNN kernel
                launch_flash_attention_forward_onednn(
                    *global_queue,
                    tensor_ptr<float>(query),
                    tensor_ptr<float>(key),
                    tensor_ptr<float>(value),
                    tensor_ptr<float>(out_tensor),
                    batch_size,
                    num_heads,
                    seq_len_q,
                    head_dim,
                    config.softmax_scale,
                    config.is_causal
                );
                
                // Create dummy LSE tensor (oneDNN doesn't compute it)
                torch::Tensor lse_tensor = torch::zeros(
                    {batch_size, num_heads, seq_len_q},
                    options
                );
                
                // Return without using FlashAttnOutput
                return std::make_tuple(out_tensor, lse_tensor);
            }
#endif
            
        default:
            // Default to V8 for best overall performance
            output = flash_attn_forward_optimized_v8_sycl(
                *global_queue,
                tensor_ptr<float>(query),
                tensor_ptr<float>(key),
                tensor_ptr<float>(value),
                config
            );
            break;
    }
    
    // Create output tensors from device pointers
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kXPU);
    
    torch::Tensor out_tensor = torch::from_blob(
        output.output,
        {batch_size, num_heads, seq_len_q, head_dim},
        options
    );
    
    torch::Tensor lse_tensor = torch::from_blob(
        output.lse,
        {batch_size, num_heads, seq_len_q},
        options
    );
    
    // Clone to ensure memory ownership
    out_tensor = out_tensor.clone();
    lse_tensor = lse_tensor.clone();
    
    // Free SYCL allocated memory
    sycl::free(output.output, *global_queue);
    sycl::free(output.lse, *global_queue);
    
    return std::make_tuple(out_tensor, lse_tensor);
}

// Backward function placeholder
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> flash_attn_backward_py(
    torch::Tensor dout,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor out,
    torch::Tensor lse,
    float dropout_p,
    float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right
) {
    // For now, return zero gradients
    // TODO: Implement backward pass
    auto dq = torch::zeros_like(query);
    auto dk = torch::zeros_like(key);
    auto dv = torch::zeros_like(value);
    
    return std::make_tuple(dq, dk, dv);
}

// Get device info
py::dict get_device_info_py() {
    init_sycl_queue();
    
    auto device = global_queue->get_device();
    py::dict info;
    
    info["name"] = device.get_info<sycl::info::device::name>();
    info["vendor"] = device.get_info<sycl::info::device::vendor>();
    info["driver_version"] = device.get_info<sycl::info::device::driver_version>();
    info["max_compute_units"] = device.get_info<sycl::info::device::max_compute_units>();
    info["max_work_group_size"] = device.get_info<sycl::info::device::max_work_group_size>();
    info["local_mem_size"] = device.get_info<sycl::info::device::local_mem_size>();
    info["global_mem_size"] = device.get_info<sycl::info::device::global_mem_size>();
    
    return info;
}

// Python module definition
PYBIND11_MODULE(sycl_flash_attn, m) {
    m.doc() = "SYCL Flash Attention for Intel GPUs";
    
    // Kernel type enum
    py::enum_<KernelType>(m, "KernelType")
        .value("AUTO", KernelType::AUTO)
        .value("OPTIMIZED_V3", KernelType::OPTIMIZED_V3)
        .value("XMX", KernelType::XMX)
        .value("OPTIMIZED_V4", KernelType::OPTIMIZED_V4)
        .value("OPTIMIZED_V5", KernelType::OPTIMIZED_V5)
        .value("OPTIMIZED_V7", KernelType::OPTIMIZED_V7)
        .value("OPTIMIZED_V8", KernelType::OPTIMIZED_V8)
#ifdef HAS_ONEDNN
        .value("ONEDNN", KernelType::ONEDNN)
#endif
        ;
    
    m.def("forward", &flash_attn_forward_py,
          "Flash attention forward pass",
          py::arg("query"),
          py::arg("key"), 
          py::arg("value"),
          py::arg("dropout_p") = 0.0f,
          py::arg("softmax_scale") = -1.0f,
          py::arg("is_causal") = false,
          py::arg("window_size_left") = -1,
          py::arg("window_size_right") = -1,
          py::arg("kernel_type") = 0);
    
    m.def("backward", &flash_attn_backward_py,
          "Flash attention backward pass",
          py::arg("dout"),
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("out"),
          py::arg("lse"),
          py::arg("dropout_p") = 0.0f,
          py::arg("softmax_scale") = -1.0f,
          py::arg("is_causal") = false,
          py::arg("window_size_left") = -1,
          py::arg("window_size_right") = -1);
    
    m.def("get_device_info", &get_device_info_py,
          "Get Intel GPU device information");
    
    m.def("is_available", []() {
        try {
            auto device = get_intel_gpu_device();
            return true;
        } catch (...) {
            return false;
        }
    }, "Check if Intel GPU is available");
}