"""
Test FSDP vs DDP equivalence in真实data parallel scenario.

Each GPU processes different data, gradients are averaged.
FSDP and DDP should produce the same results.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import sys

sys.path.insert(0, '/root/cs336-assignment2.5-fsdp')

from fsdp.api import fully_shard, get_flat_parameters, clear_fsdp_registry
from fsdp.optimizer import FSDPOptimizer
from fsdp.utils import setup_distributed, cleanup_distributed, get_rank, get_world_size


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16, bias=False)
        self.fc2 = nn.Linear(16, 4, bias=False)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def train_with_ddp(num_steps=3, seed=42):
    """Train with DDP (baseline for data parallel)."""
    rank = get_rank()
    world_size = get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print("="*80)
        print(f"DDP Training ({world_size} GPUs)")
        print("="*80)
    
    # Same model initialization across all ranks
    torch.manual_seed(seed)
    model = SimpleModel().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Different data for each rank (data parallel)
    torch.manual_seed(seed + 100 + rank)  # Different seed per rank
    data = [(torch.randn(2, 8).to(device), torch.randn(2, 4).to(device)) for _ in range(num_steps)]
    
    losses = []
    for step, (x, y) in enumerate(data):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if rank == 0:
            print(f"  Step {step}: Loss = {loss.item():.10f}")
    
    # Get final parameters
    if rank == 0:
        param_sum = sum(p.sum().item() for p in model.parameters())
        print(f"\n  Final param sum: {param_sum:.10f}")
        return losses, param_sum
    
    return None, None


def train_with_fsdp(num_steps=3, seed=42):
    """Train with FSDP."""
    rank = get_rank()
    world_size = get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print("\n" + "="*80)
        print(f"FSDP Training ({world_size} GPUs)")
        print("="*80)
    
    # Same model initialization
    torch.manual_seed(seed)
    clear_fsdp_registry()
    model = SimpleModel().to(device)
    
    # Apply FSDP
    model.fc1 = fully_shard(model.fc1)
    model.fc2 = fully_shard(model.fc2)
    
    optimizer = FSDPOptimizer(
        list(model.parameters()),
        optimizer_cls=torch.optim.SGD,
        lr=0.01
    )
    criterion = nn.MSELoss()
    
    # Different data for each rank (same as DDP)
    torch.manual_seed(seed + 100 + rank)
    data = [(torch.randn(2, 8).to(device), torch.randn(2, 4).to(device)) for _ in range(num_steps)]
    
    losses = []
    for step, (x, y) in enumerate(data):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if rank == 0:
            print(f"  Step {step}: Loss = {loss.item():.10f}")
    
    # Get final parameters (all-gather)
    if rank == 0:
        flat_params = get_flat_parameters(model)
        for fp in flat_params:
            fp.all_gather()
        param_sum = sum(fp.full_param.sum().item() for fp in flat_params)
        print(f"\n  Final param sum: {param_sum:.10f}")
        return losses, param_sum
    
    return None, None


def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size == 1:
        print("This test requires multiple GPUs!")
        print("Usage: torchrun --nproc_per_node=N test_fsdp_vs_ddp.py")
        return
    
    setup_distributed(rank=rank, world_size=world_size)
    
    # Train with DDP
    ddp_losses, ddp_params = train_with_ddp(num_steps=5, seed=42)
    
    # Train with FSDP
    fsdp_losses, fsdp_params = train_with_fsdp(num_steps=5, seed=42)
    
    # Compare (only rank 0)
    if rank == 0:
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        
        print(f"\n{'Step':<10} {'DDP Loss':<20} {'FSDP Loss':<20} {'Diff':<15}")
        print("-" * 65)
        
        max_loss_diff = 0.0
        for i, (dl, fl) in enumerate(zip(ddp_losses, fsdp_losses)):
            diff = abs(dl - fl)
            max_loss_diff = max(max_loss_diff, diff)
            print(f"{i:<10} {dl:<20.10f} {fl:<20.10f} {diff:<15.2e}")
        
        param_diff = abs(ddp_params - fsdp_params)
        
        print(f"\nParameter sum diff: {param_diff:.2e}")
        print(f"Max loss diff:      {max_loss_diff:.2e}")
        
        # Verdict
        threshold = 1e-5
        if max_loss_diff < threshold and param_diff < threshold:
            print(f"\n✅ FSDP and DDP are EQUIVALENT!")
            print(f"   (differences < {threshold:.2e})")
        else:
            print(f"\n✗ FSDP and DDP differ!")
            print(f"   Max diff {max(max_loss_diff, param_diff):.2e} exceeds {threshold:.2e}")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()

