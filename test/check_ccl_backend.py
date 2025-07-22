#!/usr/bin/env python3
"""
Check if CCL backend is available
"""

import torch
import torch.distributed as dist

print("Checking distributed backends...")

# Check available backends
backends = ['nccl', 'gloo', 'mpi', 'ccl']

for backend in backends:
    try:
        available = dist.is_available()
        if available and hasattr(dist, f'is_{backend}_available'):
            is_available = getattr(dist, f'is_{backend}_available')()
            print(f"{backend}: {'✅ Available' if is_available else '❌ Not available'}")
        else:
            print(f"{backend}: ❌ Check not available")
    except Exception as e:
        print(f"{backend}: ❌ Error checking: {e}")

# Try to import CCL bindings directly
print("\nChecking CCL bindings:")
try:
    import oneccl_bindings_for_pytorch
    print("✅ oneccl_bindings_for_pytorch imported successfully")
    
    # Check if ccl backend is registered
    if hasattr(dist, 'Backend'):
        print(f"Available backends in dist.Backend: {[attr for attr in dir(dist.Backend) if not attr.startswith('_')]}")
except Exception as e:
    print(f"❌ Failed to import oneccl_bindings_for_pytorch: {e}")

# Check Intel GPU
print("\nChecking Intel GPU:")
if torch.xpu.is_available():
    print(f"✅ Intel GPU available: {torch.xpu.device_count()} devices")
    
    # Try to get device properties
    for i in range(torch.xpu.device_count()):
        props = torch.xpu.get_device_properties(i)
        print(f"  Device {i}: {props.name}")
else:
    print("❌ Intel GPU not available")