#include "utils.h"
#include "flash_attn_kernel.h"
#include <iostream>
#include <stdexcept>

namespace flash_attn {

sycl::device get_intel_gpu_device() {
    // Try to find Intel GPU device
    auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    
    for (const auto& device : gpu_devices) {
        if (device.get_info<sycl::info::device::vendor>().find("Intel") != std::string::npos) {
            return device;
        }
    }
    
    // If no Intel GPU found, try any GPU
    if (!gpu_devices.empty()) {
        return gpu_devices[0];
    }
    
    // Fallback to default device
    return sycl::device(sycl::default_selector_v);
}

void print_device_info(const sycl::device& device) {
    std::cout << "Device: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Vendor: " << device.get_info<sycl::info::device::vendor>() << std::endl;
    std::cout << "Driver version: " << device.get_info<sycl::info::device::driver_version>() << std::endl;
    std::cout << "Max compute units: " << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "Max work group size: " << device.get_info<sycl::info::device::max_work_group_size>() << std::endl;
    std::cout << "Local memory size: " << device.get_info<sycl::info::device::local_mem_size>() << " bytes" << std::endl;
    std::cout << "Global memory size: " << device.get_info<sycl::info::device::global_mem_size>() << " bytes" << std::endl;
}

size_t estimate_memory_usage(const FlashAttnConfig& config) {
    // Calculate memory usage for flash attention
    size_t q_size = config.batch_size * config.num_heads * config.seq_len_q * config.head_dim * sizeof(float);
    size_t k_size = config.batch_size * config.num_heads * config.seq_len_k * config.head_dim * sizeof(float);
    size_t v_size = k_size;
    size_t o_size = q_size;
    size_t lse_size = config.batch_size * config.num_heads * config.seq_len_q * sizeof(float);
    
    return q_size + k_size + v_size + o_size + lse_size;
}

}  // namespace flash_attn