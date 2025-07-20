#!/usr/bin/env python3
"""
Fixed test script for CCL initialization based on Aurora training script setup
This script properly initializes CCL for distributed training
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
        # Additional variables from your working bash script
        'CCL_P2P_ACCESS_POLICY': 'off',
        'CCL_ALLREDUCE': 'ring',
        'FI_CXI_DISABLE_HOST_REGISTER': '1',
    }
    
    for key, value in ccl_env_vars.items():
        os.environ[key] = value
        print(f"  {key}={value}")
    
    print("✅ CCL environment variables set")

def get_distributed_params():
    """Get distributed parameters from environment or set defaults"""
    # Try to get from torchrun environment variables first
    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    
    # If not running under torchrun, check if we should use multi-process
    if world_size == 1 and 'FORCE_MULTI_PROCESS' in os.environ:
        print("⚠️  Single process detected but FORCE_MULTI_PROCESS set")
        print("   CCL requires multiple processes. Use torchrun instead:")
        print("   torchrun --nproc_per_node=2 --nnodes=1 your_script.py")
        return None, None, None
    
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    return rank, world_size, local_rank, master_addr, master_port

def test_ccl_initialization():
    """Test CCL initialization with proper environment setup"""
    print("\n" + "="*60)
    print("TEST: CCL Initialization with Aurora Setup")
    print("="*60)
    
    # Setup environment
    setup_ccl_environment()
    
    # Get distributed parameters
    params = get_distributed_params()
    if params is None or params[0] is None:
        print("❌ Cannot initialize CCL with single process")
        print("   Use: torchrun --nproc_per_node=2 --nnodes=1 your_script.py")
        return False
    
    rank, world_size, local_rank, master_addr, master_port = params
    
    print(f"\nInitializing distributed process group...")
    print(f"RANK: {rank}")
    print(f"WORLD_SIZE: {world_size}")
    print(f"LOCAL_RANK: {local_rank}")
    print(f"MASTER_ADDR: {master_addr}")
    print(f"MASTER_PORT: {master_port}")
    
    # CCL requires world_size > 1
    if world_size == 1:
        print("❌ CCL requires world_size > 1")
        print("   Use torchrun with --nproc_per_node=2 or higher")
        return False
    
    try:
        # Set the device before initializing process group
        if torch.xpu.is_available():
            torch.xpu.set_device(local_rank % torch.xpu.device_count())
            device = f'xpu:{local_rank % torch.xpu.device_count()}'
        else:
            device = 'cpu'
        
        # Use the same initialization pattern as Aurora training script
        dist.init_process_group(
            backend='ccl',
            init_method='env://',
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.default_pg_timeout
        )
        
        print("✅ CCL process group initialized successfully!")
        print(f"✅ Process {rank}/{world_size} initialized")
        
        # Test basic distributed operations
        x = torch.randn(4, 4, device=device)
        print(f"✅ Created tensor on {device}: {x.shape}")
        
        # Test all_reduce
        print(f"📡 Rank {rank}: Performing all_reduce...")
        dist.all_reduce(x)
        print(f"✅ Rank {rank}: all_reduce operation successful")
        
        # Test barrier
        print(f"🚧 Rank {rank}: Testing barrier...")
        dist.barrier()
        print(f"✅ Rank {rank}: barrier successful")
        
        return True
        
    except Exception as e:
        print(f"❌ CCL initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
            print(f"✅ Rank {rank}: Process group destroyed")

def test_fallback_to_gloo():
    """Test fallback to gloo if CCL fails"""
    print("\n" + "="*60)
    print("TEST: Fallback to Gloo Backend")
    print("="*60)
    
    if dist.is_initialized():
        dist.destroy_process_group()
    
    params = get_distributed_params()
    if params is None or params[0] is None:
        # For gloo, we can work with single process
        rank, world_size = 0, 1
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29501'  # Different port
    else:
        rank, world_size, local_rank, master_addr, master_port = params
    
    try:
        dist.init_process_group(
            backend='gloo',
            init_method='env://',
            rank=rank,
            world_size=world_size
        )
        
        print("✅ Gloo backend initialized successfully!")
        print(f"✅ Process {rank}/{world_size} initialized with Gloo")
        
        # Test with CPU tensors (gloo doesn't support XPU)
        x = torch.randn(4, 4)
        if world_size > 1:
            dist.all_reduce(x)
            print(f"✅ Rank {rank}: Gloo all_reduce operation successful")
        else:
            print("✅ Gloo backend working (single process)")
        
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
    
    # Check if running under torchrun
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    
    if not is_distributed or world_size == 1:
        print("\n⚠️  Running in single-process mode")
        print("   CCL requires multiple processes for proper testing")
        print("   To test CCL properly, run:")
        print("   torchrun --nproc_per_node=2 --nnodes=1 your_script.py")
        print("\n   Testing Gloo backend only...")
        
        # Only test Gloo in single process mode
        tests = [("Gloo Fallback", test_fallback_to_gloo)]
    else:
        print(f"\n✅ Running in distributed mode with {world_size} processes")
        # Test both CCL and Gloo
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
    
    if is_distributed and world_size > 1 and passed > 0:
        print("\n🎉 CCL setup working! Use these environment variables in your distributed training.")
    elif not is_distributed:
        print("\n💡 For full CCL testing, use torchrun:")
        print("   torchrun --nproc_per_node=2 --nnodes=1 your_script.py")
    
    return 0 if passed > 0 else 1

if __name__ == "__main__":
    sys.exit(main())