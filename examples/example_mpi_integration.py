#!/usr/bin/env python3
"""
Example: MPI-compatible Ring Flash Attention integration

This example demonstrates how to use Ring Flash Attention with both
torchrun and mpiexec launchers, with automatic backend detection.

Usage:
    # With torchrun (existing approach)
    torchrun --nproc_per_node=4 example_mpi_integration.py
    
    # With mpiexec (new approach)
    mpiexec -n 4 python example_mpi_integration.py
    
    # With Intel MPI (for Intel GPU)
    mpiexec -n 4 -genv CCL_BACKEND=native python example_mpi_integration.py
    
    # Multi-node with mpiexec
    mpiexec -n 8 -hostfile hosts python example_mpi_integration.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import time

# Import Ring Flash Attention with MPI support
from ring_flash_attn import (
    ring_flash_attn_func,
    setup_mpi_distributed, 
    cleanup_distributed
)


class RingAttentionModel(nn.Module):
    """
    Example model using Ring Flash Attention with MPI compatibility
    """
    
    def __init__(self, d_model: int = 512, nheads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nheads = nheads
        self.head_dim = d_model // nheads
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = dropout
        
    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        Forward pass using Ring Flash Attention
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            causal: Whether to use causal attention
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.nheads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.nheads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.nheads, self.head_dim)
        
        # Apply ring attention
        # Note: ring_flash_attn_func automatically handles distributed communication
        attn_output = ring_flash_attn_func(
            q, k, v, 
            dropout_p=self.dropout if self.training else 0.0,
            causal=causal
        )
        
        # Reshape and apply output projection
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)
        
        return output


def create_sample_data(batch_size: int, seq_len_per_rank: int, d_model: int, 
                      device: torch.device, dtype: torch.dtype = torch.float16):
    """Create sample training data"""
    return torch.randn(batch_size, seq_len_per_rank, d_model, 
                      device=device, dtype=dtype, requires_grad=False)


def train_step(model: nn.Module, data: torch.Tensor, rank: int) -> float:
    """Single training step"""
    model.train()
    
    # Forward pass
    output = model(data, causal=True)
    
    # Simple loss (for demonstration)
    target = torch.randn_like(output)
    loss = F.mse_loss(output, target)
    
    # Backward pass
    loss.backward()
    
    # Simple gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    return loss.item()


def benchmark_model(model: nn.Module, data: torch.Tensor, rank: int, num_steps: int = 10):
    """Benchmark the model performance"""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(data, causal=True)
    
    # Synchronization point
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.synchronize()
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_steps):
            output = model(data, causal=True)
    
    # Final synchronization
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.synchronize()
    
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_steps
    
    return avg_time, output.shape


def main():
    """Main training function with MPI compatibility"""
    print("🚀 Ring Flash Attention MPI Integration Example")
    print("=" * 60)
    
    # Setup distributed environment (works with both torchrun and mpiexec)
    try:
        setup_info = setup_mpi_distributed()
    except Exception as e:
        print(f"❌ Failed to setup distributed environment: {e}")
        return 1
    
    rank = setup_info['rank']
    world_size = setup_info['world_size']
    device = setup_info['device']
    launcher = setup_info['launcher']
    backend = setup_info['backend']
    
    print(f"[Rank {rank}] Distributed setup complete:")
    print(f"[Rank {rank}]   Launcher: {launcher}")
    print(f"[Rank {rank}]   Backend: {backend}")
    print(f"[Rank {rank}]   World size: {world_size}")
    print(f"[Rank {rank}]   Device: {device}")
    
    # Model configuration
    d_model = 512
    nheads = 8
    batch_size = 2
    seq_len_per_rank = 1024  # Each rank handles this much sequence length
    dtype = torch.float16 if device.type in ['cuda', 'xpu'] else torch.float32
    
    print(f"[Rank {rank}] Model config: d_model={d_model}, nheads={nheads}")
    print(f"[Rank {rank}] Data config: batch_size={batch_size}, seq_len_per_rank={seq_len_per_rank}")
    print(f"[Rank {rank}] Using dtype: {dtype}")
    
    # Create model and move to device
    model = RingAttentionModel(d_model=d_model, nheads=nheads, dropout=0.1)
    model = model.to(device=device, dtype=dtype)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create sample data
    train_data = create_sample_data(batch_size, seq_len_per_rank, d_model, device, dtype)
    
    print(f"[Rank {rank}] ✅ Model and data prepared")
    
    # Training loop
    num_train_steps = 5
    print(f"[Rank {rank}] Starting training for {num_train_steps} steps...")
    
    for step in range(num_train_steps):
        optimizer.zero_grad()
        
        start_time = time.time()
        loss = train_step(model, train_data, rank)
        step_time = time.time() - start_time
        
        optimizer.step()
        
        print(f"[Rank {rank}] Step {step+1}/{num_train_steps}: "
              f"loss={loss:.4f}, time={step_time:.3f}s")
    
    # Benchmark
    print(f"[Rank {rank}] Running benchmark...")
    avg_time, output_shape = benchmark_model(model, train_data, rank, num_steps=10)
    
    print(f"[Rank {rank}] ✅ Benchmark results:")
    print(f"[Rank {rank}]   Average inference time: {avg_time*1000:.2f} ms")
    print(f"[Rank {rank}]   Output shape: {output_shape}")
    print(f"[Rank {rank}]   Throughput: {batch_size * seq_len_per_rank / avg_time:.0f} tokens/sec")
    
    # Synchronization barrier
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()
        if rank == 0:
            print("\n🎉 All ranks completed successfully!")
            print("="*60)
            print("💡 Key benefits of MPI compatibility:")
            print("   • Works with existing HPC job schedulers")
            print("   • Compatible with Slurm, PBS, LSF")
            print("   • Supports multi-node scaling")
            print("   • Intel GPU optimization support")
            print("   • Fallback to standard PyTorch distributed")
    
    # Cleanup
    cleanup_distributed()
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        cleanup_distributed()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        cleanup_distributed()
        sys.exit(1)