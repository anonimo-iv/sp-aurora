import torch

# Auto-detect device and import appropriate backend
def _detect_backend():
    """Auto-detect available backend (Intel GPU vs NVIDIA GPU)"""
    try:
        import intel_extension_for_pytorch as ipex
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return 'intel'
    except ImportError:
        pass
    
    if torch.cuda.is_available():
        return 'cuda'
    
    return 'cpu'

BACKEND = _detect_backend()

# Import backend-specific implementations
if BACKEND == 'intel':
    # Intel GPU backend
    try:
        from .intel_ring_flash_attn import (
            intel_ring_flash_attn_func as ring_flash_attn_func,
            intel_ring_flash_attn_kvpacked_func as ring_flash_attn_kvpacked_func,
            intel_ring_flash_attn_qkvpacked_func as ring_flash_attn_qkvpacked_func,
        )
        print("Using Intel GPU backend for Ring Flash Attention")
    except ImportError as e:
        print(f"Failed to import Intel backend: {e}")
        # Fallback to original implementation
        from .ring_flash_attn import (
            ring_flash_attn_func,
            ring_flash_attn_kvpacked_func,
            ring_flash_attn_qkvpacked_func,
        )
else:
    # Original CUDA backend
    from .ring_flash_attn import (
        ring_flash_attn_func,
        ring_flash_attn_kvpacked_func,
        ring_flash_attn_qkvpacked_func,
    )

# Import other modules (with fallbacks for Intel GPU compatibility)
try:
    from .llama3_flash_attn_varlen import (
        llama3_flash_attn_prepare_cu_seqlens,
        llama3_flash_attn_varlen_func,
        llama3_flash_attn_varlen_kvpacked_func,
        llama3_flash_attn_varlen_qkvpacked_func,
    )
except ImportError:
    print("Warning: llama3_flash_attn_varlen not available (likely due to flash-attn dependency)")
    # Provide dummy implementations for compatibility
    def llama3_flash_attn_prepare_cu_seqlens(*args, **kwargs):
        raise NotImplementedError("llama3_flash_attn_varlen not available on this backend")
    def llama3_flash_attn_varlen_func(*args, **kwargs):
        raise NotImplementedError("llama3_flash_attn_varlen not available on this backend")
    def llama3_flash_attn_varlen_kvpacked_func(*args, **kwargs):
        raise NotImplementedError("llama3_flash_attn_varlen not available on this backend")
    def llama3_flash_attn_varlen_qkvpacked_func(*args, **kwargs):
        raise NotImplementedError("llama3_flash_attn_varlen not available on this backend")

try:
    from .ring_flash_attn_varlen import (
        ring_flash_attn_varlen_func,
        ring_flash_attn_varlen_kvpacked_func,
        ring_flash_attn_varlen_qkvpacked_func,
    )
except ImportError:
    print("Warning: ring_flash_attn_varlen not available")
    def ring_flash_attn_varlen_func(*args, **kwargs):
        raise NotImplementedError("ring_flash_attn_varlen not available on this backend")
    def ring_flash_attn_varlen_kvpacked_func(*args, **kwargs):
        raise NotImplementedError("ring_flash_attn_varlen not available on this backend")
    def ring_flash_attn_varlen_qkvpacked_func(*args, **kwargs):
        raise NotImplementedError("ring_flash_attn_varlen not available on this backend")

try:
    from .zigzag_ring_flash_attn import (
        zigzag_ring_flash_attn_func,
        zigzag_ring_flash_attn_kvpacked_func,
        zigzag_ring_flash_attn_qkvpacked_func,
    )
except ImportError:
    print("Warning: zigzag_ring_flash_attn not available")
    def zigzag_ring_flash_attn_func(*args, **kwargs):
        raise NotImplementedError("zigzag_ring_flash_attn not available on this backend")
    def zigzag_ring_flash_attn_kvpacked_func(*args, **kwargs):
        raise NotImplementedError("zigzag_ring_flash_attn not available on this backend")
    def zigzag_ring_flash_attn_qkvpacked_func(*args, **kwargs):
        raise NotImplementedError("zigzag_ring_flash_attn not available on this backend")

try:
    from .zigzag_ring_flash_attn_varlen import (
        zigzag_ring_flash_attn_varlen_func,
        zigzag_ring_flash_attn_varlen_kvpacked_func,
        zigzag_ring_flash_attn_varlen_qkvpacked_func,
    )
except ImportError:
    print("Warning: zigzag_ring_flash_attn_varlen not available")
    def zigzag_ring_flash_attn_varlen_func(*args, **kwargs):
        raise NotImplementedError("zigzag_ring_flash_attn_varlen not available on this backend")
    def zigzag_ring_flash_attn_varlen_kvpacked_func(*args, **kwargs):
        raise NotImplementedError("zigzag_ring_flash_attn_varlen not available on this backend")
    def zigzag_ring_flash_attn_varlen_qkvpacked_func(*args, **kwargs):
        raise NotImplementedError("zigzag_ring_flash_attn_varlen not available on this backend")

try:
    from .stripe_flash_attn import (
        stripe_flash_attn_func,
        stripe_flash_attn_kvpacked_func,
        stripe_flash_attn_qkvpacked_func,
    )
except ImportError:
    print("Warning: stripe_flash_attn not available")
    def stripe_flash_attn_func(*args, **kwargs):
        raise NotImplementedError("stripe_flash_attn not available on this backend")
    def stripe_flash_attn_kvpacked_func(*args, **kwargs):
        raise NotImplementedError("stripe_flash_attn_kvpacked not available on this backend")
    def stripe_flash_attn_qkvpacked_func(*args, **kwargs):
        raise NotImplementedError("stripe_flash_attn_qkvpacked not available on this backend")

try:
    from .adapters import (
        substitute_hf_flash_attn,
        update_ring_flash_attn_params,
    )
except ImportError:
    print("Warning: adapters not available")
    def substitute_hf_flash_attn(*args, **kwargs):
        raise NotImplementedError("substitute_hf_flash_attn not available on this backend")
    def update_ring_flash_attn_params(*args, **kwargs):
        raise NotImplementedError("update_ring_flash_attn_params not available on this backend")

# MPI utilities for enhanced distributed compatibility
try:
    from .mpi_utils import (
        setup_mpi_distributed,
        init_distributed_backend,
        cleanup_distributed,
        get_device_for_rank,
        detect_mpi_environment,
    )
except ImportError:
    print("Warning: MPI utilities not available")
    def setup_mpi_distributed(*args, **kwargs):
        raise NotImplementedError("MPI utilities not available")
    def init_distributed_backend(*args, **kwargs):
        raise NotImplementedError("MPI utilities not available")
    def cleanup_distributed(*args, **kwargs):
        pass
    def get_device_for_rank(*args, **kwargs):
        return torch.device('cpu')
    def detect_mpi_environment(*args, **kwargs):
        return False
