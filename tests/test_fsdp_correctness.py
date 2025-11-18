"""
真正的正确性验证：FSDP vs Non-FSDP在真实data parallel场景下的等价性

场景设置：
- 8 GPU FSDP: 每个GPU处理batch_size=4的数据，8个GPU并行
- 1 GPU Non-FSDP: 处理batch_size=32的数据（4×8=32）

期望：两种方式应该产生相同的梯度和参数更新（因为处理的总数据量相同）
"""

import os
import torch
import torch.nn as nn
import sys

sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from fsdp.api import fully_shard, get_flat_parameters, clear_fsdp_registry
from fsdp.optimizer import FSDPOptimizer
from fsdp.utils import setup_distributed, cleanup_distributed, get_rank, get_world_size


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32, bias=False)
        self.fc2 = nn.Linear(32, 8, bias=False)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def train_non_fsdp_single_gpu(batch_size=32, num_steps=5, seed=42):
    """在单GPU上训练，batch_size=32（模拟8个GPU各处理4个samples）"""
    print("="*70)
    print(f"Non-FSDP Training (1 GPU, batch_size={batch_size})")
    print("="*70)
    
    torch.manual_seed(seed)
    model = TestModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Fixed target
    torch.manual_seed(seed + 1000)
    target = torch.randn(1, 8)  # Same target for all samples
    target = target.expand(batch_size, -1)
    
    losses = []
    for step in range(num_steps):
        # Different data each step
        torch.manual_seed(seed + 100 + step)
        x = torch.randn(batch_size, 16)
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"  Step {step}: Loss = {loss.item():.10f}")
    
    # Get final parameters
    params = [p.detach().clone() for p in model.parameters()]
    param_sum = sum(p.sum().item() for p in params)
    
    print(f"\nFinal param sum: {param_sum:.15f}")
    
    return losses, params


def train_fsdp_multi_gpu(batch_size_per_gpu=4, num_steps=5, seed=42):
    """在多GPU上训练，每个GPU的batch_size=4（总共等于32）"""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    setup_distributed(rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print("="*70)
        print(f"FSDP Training ({world_size} GPUs, batch_size={batch_size_per_gpu}/GPU)")
        print(f"Total effective batch_size: {batch_size_per_gpu * world_size}")
        print("="*70)
    
    # Same model initialization
    torch.manual_seed(seed)
    clear_fsdp_registry()
    model = TestModel().to(device)
    
    # Apply FSDP
    model.fc1 = fully_shard(model.fc1)
    model.fc2 = fully_shard(model.fc2)
    
    optimizer = FSDPOptimizer(
        list(model.parameters()),
        optimizer_cls=torch.optim.SGD,
        lr=0.01
    )
    criterion = nn.MSELoss()
    
    # Same target as non-FSDP
    torch.manual_seed(seed + 1000)
    target = torch.randn(1, 8).to(device)
    target = target.expand(batch_size_per_gpu, -1)
    
    losses = []
    all_rank_losses = []  # Store losses from all ranks for averaging
    
    for step in range(num_steps):
        # CRITICAL: Each rank gets different data from the same overall distribution
        # This simulates splitting a batch_size=32 batch across 8 GPUs
        # Rank 0 gets samples [0:4], Rank 1 gets [4:8], etc.
        torch.manual_seed(seed + 100 + step)
        # Generate full batch, then take this rank's slice
        full_x = torch.randn(batch_size_per_gpu * world_size, 16)
        x = full_x[rank * batch_size_per_gpu : (rank + 1) * batch_size_per_gpu].to(device)
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        
        # Store per-rank loss
        rank_loss = loss.item()
        losses.append(rank_loss)
        
        # Collect losses from all ranks for averaging
        import torch.distributed as dist
        loss_tensor = torch.tensor(rank_loss, device=device)
        gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_losses, loss_tensor)
        avg_loss = sum(l.item() for l in gathered_losses) / world_size
        all_rank_losses.append(avg_loss)
        
        if rank == 0:
            print(f"  Step {step}: Rank-0 Loss = {rank_loss:.10f}, Avg Loss = {avg_loss:.10f}")
    
    # Get final parameters
    # CRITICAL: All ranks must participate in all_gather!
    flat_params = get_flat_parameters(model)
    for fp in flat_params:
        fp.all_gather()  # Collective operation - all ranks must call this!
    
    # Only rank 0 collects and prints results
    if rank == 0:
        # Unflatten parameters to match Non-FSDP format
        params = []
        for fp in flat_params:
            views = fp.create_views()  # Get unflattened views
            params.extend([v.detach().cpu().clone() for v in views])
        
        param_sum = sum(p.sum().item() for p in params)
        
        print(f"\nFinal param sum: {param_sum:.15f}")
        
        cleanup_distributed()
        # Return averaged losses (across all ranks) for comparison with Non-FSDP
        return all_rank_losses, params
    
    cleanup_distributed()
    return None, None


def compare_results():
    """比较Non-FSDP和FSDP的结果"""
    # Load results
    nonfsdp = torch.load("/tmp/nonfsdp_result.pt")
    fsdp = torch.load("/tmp/fsdp_result.pt")
    
    print("\n" + "="*70)
    print("COMPARISON: Non-FSDP (1 GPU, BS=32) vs FSDP (8 GPU, BS=4/GPU)")
    print("="*70)
    
    print(f"\n{'Step':<10} {'Non-FSDP Loss':<20} {'FSDP Loss':<20} {'Diff':<15}")
    print("-" * 65)
    
    max_loss_diff = 0.0
    for i, (l1, l2) in enumerate(zip(nonfsdp['losses'], fsdp['losses'])):
        diff = abs(l1 - l2)
        max_loss_diff = max(max_loss_diff, diff)
        print(f"{i:<10} {l1:<20.10f} {l2:<20.10f} {diff:<15.2e}")
    
    # Compare parameters
    print(f"\nParameter Comparison:")
    max_param_diff = 0.0
    for i, (p1, p2) in enumerate(zip(nonfsdp['params'], fsdp['params'])):
        diff = (p1 - p2).abs().max().item()
        max_param_diff = max(max_param_diff, diff)
        print(f"  Param {i}: max_diff = {diff:.2e}")
    
    print(f"\n{'='*70}")
    print(f"Max loss diff:  {max_loss_diff:.2e}")
    print(f"Max param diff: {max_param_diff:.2e}")
    
    threshold = 1e-5
    if max_param_diff < threshold:
        print(f"\n✅ FSDP和Non-FSDP在data parallel场景下等价！")
        print(f"   (diff < {threshold:.2e})")
        return True
    else:
        print(f"\n✗ FSDP和Non-FSDP有差异")
        print(f"   (diff {max_param_diff:.2e} > threshold {threshold:.2e})")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["nonfsdp", "fsdp", "compare"], required=True)
    args = parser.parse_args()
    
    if args.mode == "nonfsdp":
        # Non-FSDP: 1 GPU, batch_size=32
        losses, params = train_non_fsdp_single_gpu(batch_size=32, num_steps=5, seed=42)
        torch.save({'losses': losses, 'params': params}, "/tmp/nonfsdp_result.pt")
        print("\nSaved to /tmp/nonfsdp_result.pt")
        
    elif args.mode == "fsdp":
        # FSDP: 8 GPUs, batch_size=4/GPU (total=32)
        rank = int(os.environ.get("RANK", 0))
        losses, params = train_fsdp_multi_gpu(batch_size_per_gpu=4, num_steps=5, seed=42)
        if rank == 0 and losses is not None:
            torch.save({'losses': losses, 'params': params}, "/tmp/fsdp_result.pt")
            print("\nSaved to /tmp/fsdp_result.pt")
        
    elif args.mode == "compare":
        success = compare_results()
        sys.exit(0 if success else 1)

