"""
Strict Multi-GPU Equivalence Test

Verify that FSDP on 1, 2, 4, and 8 GPUs produces EXACTLY the same results
when all GPUs process the SAME data (not data parallel scenario).

This is the true test of mathematical correctness:
- Same input data
- Same model initialization  
- Should produce EXACT same gradients and parameter updates
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/root/cs336-assignment2.5-fsdp')

from fsdp.api import fully_shard, get_flat_parameters, clear_fsdp_registry
from fsdp.optimizer import FSDPOptimizer
from fsdp.utils import setup_distributed, cleanup_distributed, get_rank, get_world_size


class TinyModel(nn.Module):
    """Tiny model for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20, bias=True)
        self.fc2 = nn.Linear(20, 5, bias=True)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def train_single_gpu(num_steps=3, seed=42):
    """Train with FSDP on single GPU."""
    print("="*80)
    print("Training on SINGLE GPU (world_size=1)")
    print("="*80)
    
    torch.manual_seed(seed)
    clear_fsdp_registry()
    
    # Create model
    model = TinyModel()
    
    # Apply FSDP (world_size=1, so no actual sharding)
    model.fc1 = fully_shard(model.fc1)
    model.fc2 = fully_shard(model.fc2)
    
    # Optimizer
    optimizer = FSDPOptimizer(
        list(model.parameters()),
        optimizer_cls=torch.optim.SGD,
        lr=0.01
    )
    
    # Fixed data
    torch.manual_seed(seed + 100)
    data = [(torch.randn(4, 10), torch.randn(4, 5)) for _ in range(num_steps)]
    
    # Training
    losses = []
    criterion = nn.MSELoss()
    
    for step, (x, y) in enumerate(data):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"Step {step}: Loss = {loss.item():.10f}")
    
    # Get final parameters
    flat_params = get_flat_parameters(model)
    param_sums = [fp.data.sum().item() for fp in flat_params]
    
    print(f"\nFinal parameter sums: {param_sums}")
    print(f"Final losses: {losses}")
    
    return losses, param_sums


def train_multi_gpu(num_steps=3, seed=42):
    """Train with FSDP on multiple GPUs (all processing SAME data)."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    setup_distributed(rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print("="*80)
        print(f"Training on {world_size} GPUs (all using SAME data)")
        print("="*80)
    
    torch.manual_seed(seed)
    clear_fsdp_registry()
    
    # Create model
    model = TinyModel().to(device)
    
    # Apply FSDP
    model.fc1 = fully_shard(model.fc1)
    model.fc2 = fully_shard(model.fc2)
    
    if rank == 0:
        flat_params = get_flat_parameters(model)
        print(f"\nFlatParameters info:")
        for i, fp in enumerate(flat_params):
            print(f"  FlatParam {i}: total={fp._total_numel}, padded={fp._padded_total_numel}, " +
                  f"shard_size={fp.data.numel()}, rank={fp.rank}")
    
    # Optimizer
    optimizer = FSDPOptimizer(
        list(model.parameters()),
        optimizer_cls=torch.optim.SGD,
        lr=0.01
    )
    
    # CRITICAL: All GPUs use SAME data (not data parallel)
    torch.manual_seed(seed + 100)  # Same seed = same data
    data = [(torch.randn(4, 10).to(device), torch.randn(4, 5).to(device)) for _ in range(num_steps)]
    
    # Training
    losses = []
    criterion = nn.MSELoss()
    
    for step, (x, y) in enumerate(data):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        losses.append(loss_val)
        
        if rank == 0:
            print(f"Step {step}: Loss = {loss_val:.10f}")
    
    # Get final parameters (only from rank 0)
    if rank == 0:
        # All-gather all parameters to verify
        flat_params = get_flat_parameters(model)
        
        # Force all-gather to get full parameters
        for fp in flat_params:
            fp.all_gather()
        
        param_sums = [fp.full_param.sum().item() for fp in flat_params]
        
        print(f"\nFinal parameter sums: {param_sums}")
        print(f"Final losses: {losses}")
        
        cleanup_distributed()
        return losses, param_sums
    else:
        cleanup_distributed()
        return None, None


def compare_results(results_1gpu, results_2gpu, results_4gpu, results_8gpu):
    """Compare results from different GPU counts."""
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    losses_1, params_1 = results_1gpu
    losses_2, params_2 = results_2gpu  
    losses_4, params_4 = results_4gpu
    losses_8, params_8 = results_8gpu
    
    # Compare losses
    print("\nLoss Comparison:")
    print(f"{'Step':<10} {'1 GPU':<20} {'2 GPU':<20} {'4 GPU':<20} {'8 GPU':<20}")
    print("-" * 90)
    
    max_loss_diff = 0.0
    for i in range(len(losses_1)):
        l1, l2, l4, l8 = losses_1[i], losses_2[i], losses_4[i], losses_8[i]
        print(f"{i:<10} {l1:<20.10f} {l2:<20.10f} {l4:<20.10f} {l8:<20.10f}")
        
        # Check differences
        diff_2 = abs(l1 - l2)
        diff_4 = abs(l1 - l4)
        diff_8 = abs(l1 - l8)
        max_loss_diff = max(max_loss_diff, diff_2, diff_4, diff_8)
    
    # Compare final parameters
    print("\nFinal Parameter Sum Comparison:")
    print(f"{'Param':<10} {'1 GPU':<20} {'2 GPU':<20} {'4 GPU':<20} {'8 GPU':<20}")
    print("-" * 90)
    
    max_param_diff = 0.0
    for i in range(len(params_1)):
        p1, p2, p4, p8 = params_1[i], params_2[i], params_4[i], params_8[i]
        print(f"{i:<10} {p1:<20.10f} {p2:<20.10f} {p4:<20.10f} {p8:<20.10f}")
        
        # Check differences  
        diff_2 = abs(p1 - p2)
        diff_4 = abs(p1 - p4)
        diff_8 = abs(p1 - p8)
        max_param_diff = max(max_param_diff, diff_2, diff_4, diff_8)
    
    # Final verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    print(f"Max loss difference:      {max_loss_diff:.2e}")
    print(f"Max parameter difference: {max_param_diff:.2e}")
    
    threshold = 1e-6
    if max_loss_diff < threshold and max_param_diff < threshold:
        print(f"\n✅ ALL GPU COUNTS ARE EXACTLY EQUIVALENT!")
        print(f"   (differences < {threshold:.2e})")
        return True
    else:
        print(f"\n✗ DIFFERENCES DETECTED!")
        print(f"   Max diff {max(max_loss_diff, max_param_diff):.2e} exceeds threshold {threshold:.2e}")
        print(f"\n⚠️  This indicates a bug in the FSDP implementation!")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Compare all results")
    args = parser.parse_args()
    
    if args.compare:
        # Load saved results
        results_1 = torch.load("/tmp/fsdp_1gpu.pt")
        results_2 = torch.load("/tmp/fsdp_2gpu.pt")
        results_4 = torch.load("/tmp/fsdp_4gpu.pt")
        results_8 = torch.load("/tmp/fsdp_8gpu.pt")
        
        success = compare_results(results_1, results_2, results_4, results_8)
        sys.exit(0 if success else 1)
    
    elif "RANK" in os.environ:
        # Multi-GPU mode
        losses, params = train_multi_gpu(num_steps=5, seed=42)
        
        if get_rank() == 0:
            world_size = get_world_size()
            torch.save((losses, params), f"/tmp/fsdp_{world_size}gpu.pt")
            print(f"\nResults saved to /tmp/fsdp_{world_size}gpu.pt")
    else:
        # Single GPU mode
        losses, params = train_single_gpu(num_steps=5, seed=42)
        torch.save((losses, params), "/tmp/fsdp_1gpu.pt")
        print(f"\nResults saved to /tmp/fsdp_1gpu.pt")

