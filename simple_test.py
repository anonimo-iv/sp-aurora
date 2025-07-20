#!/usr/bin/env python3
"""Simple test to verify Intel GPU functionality"""

import sys
import torch

try:
    import intel_extension_for_pytorch as ipex
    print(f"Intel Extension for PyTorch version: {ipex.__version__}")
    
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print("✅ Intel GPU (XPU) is available")
        print(f"Device count: {torch.xpu.device_count()}")
        
        # Test tensor creation
        x = torch.randn(4, 4).to('xpu')
        print(f"✅ Created tensor on XPU: {x.device}")
        
        # Test import
        try:
            import ring_flash_attn
            print("✅ Ring Flash Attention imported")
            print(f"Backend detected: {getattr(ring_flash_attn, 'BACKEND', 'unknown')}")
        except Exception as e:
            print(f"❌ Import failed: {e}")
    else:
        print("❌ Intel GPU (XPU) not available")
        
except ImportError as e:
    print(f"❌ Intel Extension for PyTorch not available: {e}")
    sys.exit(1)