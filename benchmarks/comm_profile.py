"""Communication profiling script for FSDP.

This script measures time spent in communication operations.
"""

import argparse
import time
import torch
import torch.distributed as dist
from fsdp.utils import setup_distributed, cleanup_distributed


def profile_all_gather(tensor_size, world_size, num_iterations=100):
    """Profile all-gather performance."""
    rank = dist.get_rank()
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Create local tensor
    local_tensor = torch.randn(tensor_size, device=device)
    output_tensor = torch.empty(tensor_size * world_size, device=device)
    
    # Warm up
    for _ in range(10):
        dist.all_gather_into_tensor(output_tensor, local_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Profile
    start_time = time.time()
    for _ in range(num_iterations):
        dist.all_gather_into_tensor(output_tensor, local_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    
    return avg_time


def profile_reduce_scatter(tensor_size, world_size, num_iterations=100):
    """Profile reduce-scatter performance."""
    rank = dist.get_rank()
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Create input tensor
    input_tensor = torch.randn(tensor_size * world_size, device=device)
    output_tensor = torch.empty(tensor_size, device=device)
    
    # Warm up
    for _ in range(10):
        dist.reduce_scatter_tensor(output_tensor, input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Profile
    start_time = time.time()
    for _ in range(num_iterations):
        dist.reduce_scatter_tensor(output_tensor, input_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    
    return avg_time


def main(rank, world_size):
    """Main profiling function."""
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    setup_distributed(rank, world_size, backend=backend)
    
    try:
        if rank == 0:
            print("=" * 60)
            print("Communication Profiling Results")
            print("=" * 60)
            print(f"World size: {world_size}")
            print(f"Backend: {backend}")
            print()
        
        # Test different tensor sizes
        tensor_sizes = [1024, 1024 * 10, 1024 * 100, 1024 * 1000]  # 1KB, 10KB, 100KB, 1MB
        
        for size in tensor_sizes:
            all_gather_time = profile_all_gather(size, world_size)
            reduce_scatter_time = profile_reduce_scatter(size, world_size)
            
            if rank == 0:
                print(f"Tensor size: {size * 4 / 1024:.1f} KB ({size} elements)")
                print(f"  All-gather: {all_gather_time * 1000:.3f} ms")
                print(f"  Reduce-scatter: {reduce_scatter_time * 1000:.3f} ms")
                print()
    
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=2)
    args = parser.parse_args()
    
    if args.world_size > 1:
        mp.spawn(main, args=(args.world_size,), nprocs=args.world_size)
    else:
        print("Warning: Profiling requires world_size >= 2")

