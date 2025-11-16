"""
Debug script to find the exact source of multi-GPU non-equivalence.
"""

import os
import torch
import torch.nn as nn
import sys

sys.path.insert(0, '/root/cs336-assignment2.5-fsdp')

from fsdp.api import fully_shard, get_flat_parameters, clear_fsdp_registry
from fsdp.optimizer import FSDPOptimizer
from fsdp.utils import setup_distributed, get_rank, get_world_size


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 6, bias=False)  # Simplify: no bias
        self.fc2 = nn.Linear(6, 2, bias=False)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_single_gpu():
    """Test on single GPU with detailed logging."""
    print("="*80)
    print("SINGLE GPU TEST")
    print("="*80)
    
    torch.manual_seed(42)
    clear_fsdp_registry()
    
    model = TinyModel()
    model.fc1 = fully_shard(model.fc1)
    model.fc2 = fully_shard(model.fc2)
    
    flat_params = get_flat_parameters(model)
    print(f"\nFlatParameters:")
    for i, fp in enumerate(flat_params):
        print(f"  FP{i}: total={fp._total_numel}, padded={fp._padded_total_numel}, shard={fp.data.numel()}")
        print(f"       Initial param sum: {fp.data.sum().item():.10f}")
    
    optimizer = FSDPOptimizer(
        list(model.parameters()),
        optimizer_cls=torch.optim.SGD,
        lr=0.1
    )
    
    # Fixed input
    torch.manual_seed(100)
    x = torch.randn(2, 4)
    y = torch.randn(2, 2)
    
    print(f"\nInput x sum: {x.sum().item():.10f}")
    print(f"Target y sum: {y.sum().item():.10f}")
    
    # Forward
    optimizer.zero_grad()
    out = model(x)
    loss = nn.MSELoss()(out, y)
    
    print(f"\nForward:")
    print(f"  Output sum: {out.sum().item():.10f}")
    print(f"  Loss: {loss.item():.10f}")
    
    # Backward
    loss.backward()
    
    print(f"\nBackward (gradients):")
    for i, fp in enumerate(flat_params):
        if fp.grad is not None:
            print(f"  FP{i} grad sum: {fp.grad.sum().item():.10f}")
            print(f"  FP{i} grad mean: {fp.grad.mean().item():.10f}")
            print(f"  FP{i} grad numel: {fp.grad.numel()}")
    
    # Step
    optimizer.step()
    
    print(f"\nAfter optimizer step:")
    for i, fp in enumerate(flat_params):
        print(f"  FP{i} param sum: {fp.data.sum().item():.10f}")
        # Check if there's padding
        if fp.data.numel() * 1 > fp._total_numel:
            print(f"       WARNING: This FlatParam has padding!")
    
    return loss.item(), [fp.data.sum().item() for fp in flat_params]


def test_multi_gpu():
    """Test on multiple GPUs with detailed logging."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    setup_distributed(rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print("="*80)
        print(f"MULTI-GPU TEST ({world_size} GPUs)")
        print("="*80)
    
    torch.manual_seed(42)
    clear_fsdp_registry()
    
    model = TinyModel().to(device)
    model.fc1 = fully_shard(model.fc1)
    model.fc2 = fully_shard(model.fc2)
    
    flat_params = get_flat_parameters(model)
    
    if rank == 0:
        print(f"\nFlatParameters:")
        for i, fp in enumerate(flat_params):
            print(f"  FP{i}: total={fp._total_numel}, padded={fp._padded_total_numel}, shard={fp.data.numel()}")
            print(f"       Shard size × world_size = {fp.data.numel()} × {world_size} = {fp.data.numel() * world_size}")
            print(f"       Has padding: {fp.data.numel() * world_size > fp._total_numel}")
    
    # Print each rank's shard
    for r in range(world_size):
        if rank == r:
            print(f"\n[Rank {rank}] Initial shard sums:")
            for i, fp in enumerate(flat_params):
                print(f"  FP{i}: {fp.data.sum().item():.10f}")
                # Check for padding in this rank's shard
                shard_start = fp.rank * fp.data.numel()
                shard_end = shard_start + fp.data.numel()
                if shard_end > fp._total_numel:
                    valid_size = fp._total_numel - shard_start
                    print(f"       [Rank {rank}] This shard has padding! valid_size={valid_size}/{fp.data.numel()}")
        torch.distributed.barrier()
    
    optimizer = FSDPOptimizer(
        list(model.parameters()),
        optimizer_cls=torch.optim.SGD,
        lr=0.1
    )
    
    # SAME input on all ranks
    torch.manual_seed(100)
    x = torch.randn(2, 4).to(device)
    y = torch.randn(2, 2).to(device)
    
    if rank == 0:
        print(f"\nInput x sum: {x.sum().item():.10f}")
        print(f"Target y sum: {y.sum().item():.10f}")
    
    # Forward
    optimizer.zero_grad()
    out = model(x)
    loss = nn.MSELoss()(out, y)
    
    if rank == 0:
        print(f"\nForward:")
        print(f"  Output sum: {out.sum().item():.10f}")
        print(f"  Loss: {loss.item():.10f}")
    
    # Backward
    loss.backward()
    
    # Print gradients from each rank
    for r in range(world_size):
        if rank == r:
            print(f"\n[Rank {rank}] After backward (local grad shard):")
            for i, fp in enumerate(flat_params):
                if fp.grad is not None:
                    print(f"  FP{i} grad sum: {fp.grad.sum().item():.10f}")
                    print(f"  FP{i} grad mean: {fp.grad.mean().item():.10f}")
                    print(f"  FP{i} grad numel: {fp.grad.numel()}")
                    
                    # Check padding region
                    shard_start = fp.rank * fp.data.numel()
                    shard_end = shard_start + fp.data.numel()
                    if shard_end > fp._total_numel:
                        valid_size = fp._total_numel - shard_start
                        print(f"       Padding region grad sum: {fp.grad[valid_size:].sum().item():.10f}")
        torch.distributed.barrier()
    
    # Step
    optimizer.step()
    
    if rank == 0:
        print(f"\nAfter optimizer step (need to all-gather to see full params):")
        for i, fp in enumerate(flat_params):
            # All-gather to see full parameter
            fp.all_gather()
            print(f"  FP{i} full param sum: {fp.full_param.sum().item():.10f}")
    
    from fsdp.utils import cleanup_distributed
    cleanup_distributed()
    
    if rank == 0:
        return loss.item(), [fp.full_param.sum().item() for fp in flat_params]
    return None, None


if __name__ == "__main__":
    if "RANK" in os.environ:
        loss, params = test_multi_gpu()
        if get_rank() == 0:
            print(f"\nFinal result: loss={loss:.10f}, params={params}")
    else:
        loss, params = test_single_gpu()
        print(f"\nFinal result: loss={loss:.10f}, params={params}")

