#!/usr/bin/env python3
"""
More detailed check of SDPA backends with performance comparison
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import time

def benchmark_backend(backend_name, backend, q, k, v, iterations=50):
    """Benchmark a specific backend"""
    try:
        with sdpa_kernel(backends=[backend]):
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            
            # Synchronize
            if q.device.type == 'xpu':
                torch.xpu.synchronize()
            elif q.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Time
            start = time.time()
            for _ in range(iterations):
                with torch.no_grad():
                    output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            
            if q.device.type == 'xpu':
                torch.xpu.synchronize()
            elif q.device.type == 'cuda':
                torch.cuda.synchronize()
                
            elapsed = (time.time() - start) / iterations * 1000
            
            # Check output shape
            assert output.shape == q.shape
            
            return elapsed, True, None
    except Exception as e:
        return None, False, str(e)

def main():
    print("=== Detailed SDPA Backend Analysis ===\n")
    
    device = 'xpu' if torch.xpu.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'xpu':
        print(f"XPU Device: {torch.xpu.get_device_name(0)}")
    
    # Test different sequence lengths
    configs = [
        {"batch": 1, "heads": 32, "seq_len": 128, "head_dim": 128},
        {"batch": 1, "heads": 32, "seq_len": 256, "head_dim": 128},
        {"batch": 1, "heads": 32, "seq_len": 512, "head_dim": 128},
        {"batch": 1, "heads": 32, "seq_len": 1024, "head_dim": 128},
    ]
    
    backends = {
        "MATH": SDPBackend.MATH,
        "FLASH_ATTENTION": SDPBackend.FLASH_ATTENTION,
        "EFFICIENT_ATTENTION": SDPBackend.EFFICIENT_ATTENTION,
    }
    
    print("\n{:<10} {:<20} {:<15} {:<15} {:<10}".format(
        "SeqLen", "Backend", "Time (ms)", "Works", "Notes"
    ))
    print("-" * 75)
    
    for config in configs:
        seq_len = config['seq_len']
        
        # Create tensors
        q = torch.randn(config['batch'], config['heads'], seq_len, config['head_dim'], 
                       device=device, dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        
        # Test default backend
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if device == 'xpu':
            torch.xpu.synchronize()
        default_time = (time.time() - start) / 10 * 1000
        
        print(f"{seq_len:<10} {'Default':<20} {default_time:<15.2f} {'Yes':<15} {'Baseline':<10}")
        
        # Test each backend
        for backend_name, backend in backends.items():
            time_ms, works, error = benchmark_backend(backend_name, backend, q, k, v, iterations=10)
            
            if works:
                speedup = default_time / time_ms
                notes = f"{speedup:.2f}x vs default"
                print(f"{seq_len:<10} {backend_name:<20} {time_ms:<15.2f} {'Yes':<15} {notes:<10}")
            else:
                print(f"{seq_len:<10} {backend_name:<20} {'N/A':<15} {'No':<15} {'Error':<10}")
    
    # Check if backends are actually different
    print("\n=== Backend Differentiation Test ===")
    print("Testing if backends produce different implementations...")
    
    # Use a specific size that might show differences
    q = torch.randn(1, 16, 512, 64, device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    
    results = {}
    for backend_name, backend in backends.items():
        time_ms, works, _ = benchmark_backend(backend_name, backend, q, k, v, iterations=20)
        if works:
            results[backend_name] = time_ms
    
    print("\nBackend timing comparison (fp16, seq=512):")
    for name, time_ms in results.items():
        print(f"  {name}: {time_ms:.3f} ms")
    
    # Check if timings are significantly different
    if len(results) > 1:
        times = list(results.values())
        max_diff = max(times) - min(times)
        avg_time = sum(times) / len(times)
        diff_percent = (max_diff / avg_time) * 100
        
        print(f"\nMax timing difference: {max_diff:.3f} ms ({diff_percent:.1f}%)")
        
        if diff_percent < 5:
            print("→ All backends have very similar performance (<5% difference)")
            print("→ This suggests they might be using the same underlying implementation")
        else:
            print("→ Significant performance differences detected")
            print("→ Different backends are likely using different implementations")
    
    # Check IPEX
    print("\n=== Intel Extension Check ===")
    try:
        import intel_extension_for_pytorch as ipex
        print(f"IPEX version: {ipex.__version__}")
        
        # Try IPEX's SDPA if available
        if hasattr(ipex.nn.functional, 'scaled_dot_product_attention'):
            print("IPEX has its own scaled_dot_product_attention!")
            
            # Time IPEX version
            start = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = ipex.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            if device == 'xpu':
                torch.xpu.synchronize()
            ipex_time = (time.time() - start) / 10 * 1000
            
            print(f"IPEX SDPA time: {ipex_time:.2f} ms")
        else:
            print("IPEX doesn't expose its own SDPA function")
            
    except ImportError:
        print("IPEX not available")

if __name__ == "__main__":
    main()