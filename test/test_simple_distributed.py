#!/usr/bin/env python3
"""
Simple test to debug distributed initialization issues
"""

import os
import sys
import torch
import torch.distributed as dist

def main():
    print(f"Process starting - PID: {os.getpid()}")
    
    # Check environment variables
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    print(f"RANK: {rank}, WORLD_SIZE: {world_size}, LOCAL_RANK: {local_rank}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'not set')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'not set')}")
    
    # Try gloo backend first (should be most reliable)
    try:
        print(f"[Rank {rank}] Attempting gloo initialization...")
        dist.init_process_group(backend='gloo', timeout=torch.distributed.default_pg_timeout)
        print(f"[Rank {rank}] ✅ Gloo backend initialized successfully")
        
        # Test basic operation
        x = torch.randn(4, 4)
        dist.all_reduce(x)
        print(f"[Rank {rank}] ✅ All-reduce successful")
        
        dist.destroy_process_group()
        print(f"[Rank {rank}] ✅ Process group destroyed")
        
    except Exception as e:
        print(f"[Rank {rank}] ❌ Gloo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()