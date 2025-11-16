"""
Test that reduce-scatter + averaging produces correct results.

This verifies the mathematical correctness of gradient aggregation.
"""

import os
import torch
import torch.distributed as dist
from fsdp.utils import setup_distributed, cleanup_distributed, reduce_scatter_tensor


def test_reduce_scatter_avg():
    """
    Test that reduce-scatter followed by division gives correct averaged result.
    
    Setup:
    - Each rank has a tensor [rank, rank, rank, ...]
    - After reduce-scatter + avg, each rank should have the average
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    setup_distributed(rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    
    # Create test tensor: each rank has different values
    # Full tensor on each rank: [rank0_val, rank0_val, rank1_val, rank1_val, ...]
    shard_size = 4
    full_tensor = torch.zeros(world_size * shard_size, device=device)
    
    for r in range(world_size):
        start = r * shard_size
        end = start + shard_size
        full_tensor[start:end] = float(r)  # Rank r contributes value r
    
    if rank == 0:
        print(f"\n[Rank {rank}] Full tensor before reduce-scatter: {full_tensor.tolist()}")
    
    # Reduce-scatter
    output_shard = torch.zeros(shard_size, device=device)
    reduce_scatter_tensor(output_shard, full_tensor)
    
    if rank == 0:
        print(f"[Rank {rank}] After reduce-scatter (before avg): {output_shard.tolist()}")
    
    # Average
    output_shard.div_(world_size)
    
    if rank == 0:
        print(f"[Rank {rank}] After averaging: {output_shard.tolist()}")
    
    # Expected: sum of all ranks' contributions divided by world_size
    # For shard belonging to rank r: sum([r for each rank]) / world_size = r * world_size / world_size = r
    expected_value = float(rank)
    
    # Check
    success = torch.allclose(output_shard, torch.full_like(output_shard, expected_value))
    
    if rank == 0:
        print(f"\n[Rank {rank}] Expected: {expected_value}, Got: {output_shard[0].item()}")
        print(f"[Rank {rank}] Test: {'✓ PASSED' if success else '✗ FAILED'}")
    
    cleanup_distributed()
    return success


def test_same_data_scenario():
    """
    Test the scenario where all ranks have the SAME data (like in equivalence test).
    
    This simulates what happens when all GPUs process identical inputs.
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    setup_distributed(rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    
    # All ranks have the SAME gradient tensor
    # This simulates: all GPUs see same data → same gradients
    shard_size = 4
    torch.manual_seed(42)  # Same seed = same "gradient"
    local_gradient = torch.randn(shard_size, device=device)
    
    # Build full gradient (concatenate all shards, but all shards are the same!)
    full_gradient = local_gradient.repeat(world_size)
    
    if rank == 0:
        print(f"\n[Rank {rank}] Local gradient: {local_gradient.tolist()}")
        print(f"[Rank {rank}] Full gradient (repeated): {full_gradient[:8].tolist()}... (len={len(full_gradient)})")
    
    # Reduce-scatter
    output_shard = torch.zeros(shard_size, device=device)
    reduce_scatter_tensor(output_shard, full_gradient)
    
    if rank == 0:
        print(f"[Rank {rank}] After reduce-scatter: {output_shard.tolist()}")
        print(f"[Rank {rank}] This is {world_size}x the original gradient!")
    
    # Average
    output_shard.div_(world_size)
    
    if rank == 0:
        print(f"[Rank {rank}] After div by {world_size}: {output_shard.tolist()}")
    
    # Expected: should equal the original local_gradient
    success = torch.allclose(output_shard, local_gradient, atol=1e-6)
    
    if rank == 0:
        diff = (output_shard - local_gradient).abs().max().item()
        print(f"\n[Rank {rank}] Max difference from original: {diff:.2e}")
        print(f"[Rank {rank}] Test: {'✓ PASSED' if success else '✗ FAILED'}")
        
        if not success:
            print(f"[Rank {rank}] Expected: {local_gradient.tolist()}")
            print(f"[Rank {rank}] Got:      {output_shard.tolist()}")
    
    cleanup_distributed()
    return success


if __name__ == "__main__":
    if "RANK" not in os.environ:
        print("This test must be run with torchrun!")
        print("Usage: torchrun --nproc_per_node=N test_reduce_scatter_correctness.py")
        exit(1)
    
    print("="*80)
    print("Test 1: Reduce-Scatter with Different Data per Rank")
    print("="*80)
    test1_passed = test_reduce_scatter_avg()
    
    print("\n" + "="*80)
    print("Test 2: Reduce-Scatter with SAME Data on All Ranks")
    print("="*80)
    test2_passed = test_same_data_scenario()
    
    if get_rank() == 0:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Test 1 (different data): {'✓ PASSED' if test1_passed else '✗ FAILED'}")
        print(f"Test 2 (same data):      {'✓ PASSED' if test2_passed else '✗ FAILED'}")
        
        if test1_passed and test2_passed:
            print("\n✅ All reduce-scatter tests passed!")
            print("   Gradient averaging is mathematically correct.")
        else:
            print("\n✗ Some tests failed!")
            print("   There may be a bug in reduce-scatter or averaging logic.")

