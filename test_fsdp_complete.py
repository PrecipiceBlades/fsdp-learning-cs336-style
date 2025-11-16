"""Complete FSDP distributed training test."""

import torch
import torch.nn as nn
import torch.distributed as dist
import sys
sys.path.append('/root/cs336-assignment2-systems/cs336-basics')

from cs336_basics.model import BasicsTransformerLM
from fsdp import fully_shard, get_flat_parameters, clear_fsdp_registry
from fsdp.optimizer import FSDPOptimizer


def main():
    """Main training function."""
    # Initialize
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    clear_fsdp_registry()
    
    # Model config
    config = {
        "vocab_size": 1000,
        "context_length": 128,
        "d_model": 512,
        "num_layers": 8,
        "num_heads": 8,
        "d_ff": 2048,
        "rope_theta": 10000.0,
    }
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"FSDP Distributed Training Test")
        print(f"{'='*80}")
        print(f"World size: {world_size} GPUs")
    
    # Create model
    torch.manual_seed(42)
    model = BasicsTransformerLM(**config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    
    if rank == 0:
        print(f"Model: {total_params/1e6:.2f}M parameters")
    
    # Apply FSDP
    for layer in model.layers:
        fully_shard(layer, reshard_after_forward=True, rank=rank, world_size=world_size)
    fully_shard(model.token_embeddings, reshard_after_forward=True, rank=rank, world_size=world_size)
    fully_shard(model.lm_head, reshard_after_forward=True, rank=rank, world_size=world_size)
    fully_shard(model.ln_final, reshard_after_forward=True, rank=rank, world_size=world_size)
    
    flat_params = get_flat_parameters(model)
    
    if rank == 0:
        print(f"FlatParameters: {len(flat_params)}")
        shard_size = sum(fp.data.numel() for fp in flat_params)
        print(f"Shard size per device: {shard_size} ({shard_size / total_params * 100:.1f}% of total)")
    
    # Create optimizer
    optimizer = FSDPOptimizer(
        flat_params,
        optimizer_cls=torch.optim.Adam,
        lr=1e-3
    )
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Reset memory
    torch.cuda.reset_peak_memory_stats(device)
    
    # Training
    if rank == 0:
        print(f"\n{'='*80}")
        print("Training")
        print(f"{'='*80}")
    
    for epoch in range(5):
        # Same data on all ranks
        torch.manual_seed(42 + epoch)
        x = torch.randint(0, config["vocab_size"], (4, 32)).to(device)
        y = torch.randint(0, config["vocab_size"], (4, 32)).to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        
        # Gather loss
        loss_tensor = torch.tensor([loss.item()], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / world_size
        
        if rank == 0:
            print(f"Epoch {epoch+1}/5: loss={avg_loss:.6f}")
    
    # Memory stats
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024**3)
    
    memory_tensor = torch.tensor([peak_memory], device=device)
    memory_list = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(memory_list, memory_tensor)
    
    if rank == 0:
        print(f"\n{'='*80}")
        print("Memory Usage")
        print(f"{'='*80}")
        total_memory = sum(m.item() for m in memory_list)
        for i, mem in enumerate(memory_list):
            print(f"  GPU {i}: {mem.item():.2f} GB")
        
        print(f"\nTotal: {total_memory:.2f} GB")
        print(f"Per device: {total_memory/world_size:.2f} GB")
        
        # Expected memory (params + grads + optimizer for Adam)
        expected = (4 * total_params * 4) / (1024**3)
        expected_per_device = expected / world_size
        print(f"\nExpected per device: {expected_per_device:.2f} GB")
        print(f"Actual per device: {total_memory/world_size:.2f} GB")
        print(f"Overhead ratio: {(total_memory/world_size) / expected_per_device:.2f}x")
        
        # Check balance
        memories = [m.item() for m in memory_list]
        imbalance = (max(memories) - min(memories)) / max(memories) * 100
        if imbalance < 10:
            print(f"\n✅ Memory well balanced (imbalance: {imbalance:.1f}%)")
        else:
            print(f"\n⚠️  Memory imbalance: {imbalance:.1f}%")
        
        print(f"\n{'='*80}")
        print("✅ Test Completed Successfully!")
        print(f"{'='*80}")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

