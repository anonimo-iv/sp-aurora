#!/usr/bin/env python3
"""
Test script for CCL initialization based on Aurora training script setup
This script isolates and tests the CCL initialization process
"""

import os
import sys
import torch
import torch.distributed as dist

# Check for Intel GPU support
try:
    import intel_extension_for_pytorch as ipex
    if not torch.xpu.is_available():
        print("Intel GPU not available, exiting")
        sys.exit(0)
except ImportError:
    print("Intel Extension for PyTorch not installed, exiting")
    sys.exit(0)

def setup_ccl_environment():
    """Setup CCL environment variables based on Aurora training script"""
    print("Setting up CCL environment variables...")
    
    # Critical CCL environment variables from Aurora setup
    ccl_env_vars = {
        'CCL_BACKEND': 'native',
        'CCL_ATL_TRANSPORT': 'ofi',
        'FI_PROVIDER': 'cxi',  # Critical for Aurora fabric interface
        'CCL_ZE_IPC_EXCHANGE': 'drmfd',
        'CCL_ZE_ENABLE': '1',
        'CCL_LOG_LEVEL': 'info',
        'IPEX_XPU_ONEDNN_LAYOUT': '1',
        'IPEX_OFFLINE_COMPILER': '1',
        'SYCL_CACHE_PERSISTENT': '1',  # Prevents build-for-1-device issues
        'SYCL_DEVICE_FILTER': 'level_zero:*',  # Proper SYCL device selection
        'SYCL_PI_LEVEL_ZERO_PROGRAM_BUILD_TRACK': '2',  # Program building tracking
    }
    
    for key, value in ccl_env_vars.items():
        os.environ[key] = value
        print(f"  {key}={value}")
    
    print("✅ CCL environment variables set")

def test_ccl_initialization():
    """Test CCL initialization with proper environment setup"""
    print("\n" + "="*60)
    print("TEST: CCL Initialization with Aurora Setup")
    print("="*60)
    
    # Setup environment
    setup_ccl_environment()
    
    # Set distributed environment variables
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    print(f"\nInitializing distributed process group...")
    print(f"RANK: {os.environ['RANK']}")
    print(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}")
    print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
    print(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
    
    try:
        # Use the same initialization pattern as Aurora training script
        dist.init_process_group(
            backend='ccl',
            init_method='env://',
            rank=0,
            world_size=1,
            timeout=torch.distributed.default_pg_timeout
        )
        
        print("✅ CCL process group initialized successfully!")
        
        # Test basic distributed operations
        device = 'xpu'
        x = torch.randn(4, 4, device=device)
        print(f"✅ Created tensor on {device}: {x.shape}")
        
        # Test all_reduce (should work even with world_size=1)
        dist.all_reduce(x)
        print("✅ all_reduce operation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ CCL initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
            print("✅ Process group destroyed")

def test_fallback_to_gloo():
    """Test fallback to gloo if CCL fails"""
    print("\n" + "="*60)
    print("TEST: Fallback to Gloo Backend")
    print("="*60)
    
    if dist.is_initialized():
        dist.destroy_process_group()
    
    try:
        dist.init_process_group(
            backend='gloo',
            init_method='env://',
            rank=0,
            world_size=1
        )
        
        print("✅ Gloo backend initialized successfully!")
        
        # Test with CPU tensors (gloo doesn't support XPU)
        x = torch.randn(4, 4)
        dist.all_reduce(x)
        print("✅ Gloo all_reduce operation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Gloo initialization failed: {e}")
        return False
    
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def main():
    """Run CCL setup tests"""
    print("🚀 Intel GPU CCL Setup Test Suite")
    print("="*60)
    
    if not torch.xpu.is_available():
        print("❌ Intel GPU not available, exiting")
        return 1
    
    print(f"✅ Intel GPU detected: {torch.xpu.device_count()} device(s)")
    print(f"✅ Intel Extension for PyTorch version: {ipex.__version__}")
    
    # Test CCL initialization
    tests = [
        ("CCL Initialization", test_ccl_initialization),
        ("Gloo Fallback", test_fallback_to_gloo),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 CCL SETUP TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed > 0:
        print("\n🎉 CCL setup working! Use these environment variables in your tests.")
        return 0
    else:
        print("\n⚠️  All CCL tests failed. Check Intel GPU and oneCCL installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())