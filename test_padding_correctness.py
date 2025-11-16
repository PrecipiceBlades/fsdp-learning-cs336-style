"""
Test padding implementation correctness.
"""

import torch
import sys
sys.path.insert(0, '/root/cs336-assignment2.5-fsdp')

from fsdp.flat_param import FlatParameter


def test_padding_logic():
    """Test that padding calculations are correct."""
    print("="*80)
    print("Testing Padding Logic")
    print("="*80)
    
    # Test case: total_numel=10, world_size=3
    # Expected: shard_size = ceil(10/3) = 4
    #           padded_total = 4 * 3 = 12
    #           padding = 2
    
    total_numel = 10
    world_size = 3
    
    shard_size = (total_numel + world_size - 1) // world_size
    padded_total = shard_size * world_size
    
    print(f"\nSetup:")
    print(f"  total_numel: {total_numel}")
    print(f"  world_size: {world_size}")
    print(f"  shard_size: {shard_size}")
    print(f"  padded_total: {padded_total}")
    print(f"  padding: {padded_total - total_numel}")
    
    print(f"\nShard ranges:")
    for rank in range(world_size):
        start = rank * shard_size
        end = start + shard_size
        
        # Check if this shard contains padding
        if end > total_numel:
            valid_size = total_numel - start
            padding_size = shard_size - valid_size
            print(f"  Rank {rank}: [{start}:{end}] - HAS PADDING (valid={valid_size}, padding={padding_size})")
        else:
            print(f"  Rank {rank}: [{start}:{end}] - no padding")
    
    # Now test with actual FlatParameter
    print(f"\n" + "="*80)
    print("Testing with FlatParameter")
    print("="*80)
    
    # Create some dummy parameters
    p1 = torch.nn.Parameter(torch.randn(6))  # 6 elements
    p2 = torch.nn.Parameter(torch.randn(4))  # 4 elements
    # Total: 10 elements
    
    for rank in range(world_size):
        print(f"\nRank {rank}:")
        fp = FlatParameter([p1, p2], rank=rank, world_size=world_size)
        
        print(f"  _total_numel: {fp._total_numel}")
        print(f"  _padded_total_numel: {fp._padded_total_numel}")
        print(f"  data.numel(): {fp.data.numel()}")
        print(f"  _shard_offset: {fp._shard_offset}")
        print(f"  _shard_numel: {fp._shard_numel}")
        
        # Check if this rank's shard has padding
        shard_start = rank * fp.data.numel()
        shard_end = shard_start + fp.data.numel()
        
        print(f"  Shard range: [{shard_start}:{shard_end}]")
        
        if shard_end > fp._total_numel:
            valid_size = fp._total_numel - shard_start
            print(f"  ⚠️  This shard has PADDING!")
            print(f"      Valid elements: [0:{valid_size}]")
            print(f"      Padding elements: [{valid_size}:{fp.data.numel()}]")
            
            # Check if padding is zero
            padding_sum = fp.data[valid_size:].sum().item()
            print(f"      Padding sum (should be 0): {padding_sum:.10f}")
            
            if abs(padding_sum) < 1e-10:
                print(f"      ✓ Padding is zero")
            else:
                print(f"      ✗ WARNING: Padding is NOT zero!")


def test_padding_in_all_gather():
    """Test that all_gather handles padding correctly."""
    print("\n" + "="*80)
    print("Testing All-Gather with Padding")
    print("="*80)
    
    world_size = 3
    p1 = torch.nn.Parameter(torch.ones(6))
    p2 = torch.nn.Parameter(torch.ones(4) * 2)
    
    print(f"\nOriginal parameters:")
    print(f"  p1: {p1.data.tolist()} (sum={p1.sum().item()})")
    print(f"  p2: {p2.data.tolist()} (sum={p2.sum().item()})")
    print(f"  Total sum: {p1.sum().item() + p2.sum().item()}")
    
    # Create FlatParameter for rank 0
    fp = FlatParameter([p1, p2], rank=0, world_size=world_size)
    
    print(f"\nRank 0 shard: {fp.data.tolist()}")
    print(f"  Sum: {fp.data.sum().item()}")
    
    # Simulate all-gather for world_size=1 (should just use data directly)
    fp_single = FlatParameter([p1, p2], rank=0, world_size=1)
    full = fp_single.all_gather()
    
    print(f"\nAfter all-gather (world_size=1):")
    print(f"  Full param numel: {full.numel()}")
    print(f"  Full param sum: {full.sum().item()}")
    print(f"  Expected sum: {p1.sum().item() + p2.sum().item()}")
    
    if abs(full.sum().item() - (p1.sum().item() + p2.sum().item())) < 1e-6:
        print(f"  ✓ All-gather preserves sum")
    else:
        print(f"  ✗ WARNING: Sum mismatch!")


if __name__ == "__main__":
    test_padding_logic()
    test_padding_in_all_gather()

