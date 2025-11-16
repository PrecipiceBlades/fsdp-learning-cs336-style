"""Test FSDP with larger model (GPT-2 scale) on 8 GPUs."""

import torch
import torch.nn as nn
import torch.distributed as dist
import sys
sys.path.append('/root/cs336-assignment2-systems/cs336-basics')

from cs336_basics.model import BasicsTransformerLM
from fsdp import fully_shard, get_flat_parameters, clear_fsdp_registry
from fsdp.optimizer import FSDPOptimizer


def main():
    """Main function."""
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    clear_fsdp_registry()
    
    # GPT-2 Medium scale
    config = {
        "vocab_size": 50257,
        "context_length": 512,
        "d_model": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "d_ff": 4096,
        "rope_theta": 10000.0,
    }
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"FSDP Multi-GPU Test: GPT-2 Medium Scale")
        print(f"{'='*80}")
        print(f"World size: {world_size} H100 GPUs")
    
    # Create model
    torch.manual_seed(42)
    model = BasicsTransformerLM(**config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    
    if rank == 0:
        print(f"Model: {total_params/1e6:.2f}M parameters ({total_params/1e9:.3f}B)")
    
    # Apply FSDP
    for layer in model.layers:
        fully_shard(layer, reshard_after_forward=True, rank=rank, world_size=world_size)
    fully_shard(model.token_embeddings, reshard_after_forward=True, rank=rank, world_size=world_size)
    fully_shard(model.lm_head, reshard_after_forward=True, rank=rank, world_size=world_size)
    fully_shard(model.ln_final, reshard_after_forward=True, rank=rank, world_size=world_size)
    
    flat_params = get_flat_parameters(model)
    
    if rank == 0:
        shard_size = sum(fp.data.numel() for fp in flat_params)
        print(f"Shard size per device: {shard_size} ({shard_size / total_params * 100:.1f}% of total)")
    
    # Create optimizer
    optimizer = FSDPOptimizer(
        flat_params,
        optimizer_cls=torch.optim.Adam,
        lr=1e-4
    )
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Reset memory
    torch.cuda.reset_peak_memory_stats(device)
    
    # Training
    if rank == 0:
        print(f"\n{'='*80}")
        print("Training (5 epochs)")
        print(f"{'='*80}")
    
    for epoch in range(5):
        torch.manual_seed(42 + epoch)
        x = torch.randint(0, config["vocab_size"], (2, 128)).to(device)
        y = torch.randint(0, config["vocab_size"], (2, 128)).to(device)
        
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
        print("Memory Analysis")
        print(f"{'='*80}")
        
        # Per-device memory
        print(f"\nPer-device memory:")
        total_memory = sum(m.item() for m in memory_list)
        for i, mem in enumerate(memory_list):
            print(f"  GPU {i}: {mem.item():.2f} GB")
        
        memories = [m.item() for m in memory_list]
        imbalance = (max(memories) - min(memories)) / max(memories) * 100
        
        print(f"\nTotal across all GPUs: {total_memory:.2f} GB")
        print(f"Average per device: {total_memory/world_size:.2f} GB")
        print(f"Memory balance: {imbalance:.1f}% imbalance")
        
        # Theoretical calculation
        bytes_per_param = 4  # FP32
        expected_total = (4 * total_params * bytes_per_param) / (1024**3)  # params + grads + 2*optimizer
        expected_per_device = expected_total / world_size
        
        print(f"\n{'='*80}")
        print("Memory Scaling Verification")
        print(f"{'='*80}")
        print(f"Expected total memory: {expected_total:.2f} GB")
        print(f"Expected per device: {expected_per_device:.2f} GB")
        print(f"Actual per device: {total_memory/world_size:.2f} GB")
        print(f"Overhead: {(total_memory/world_size) / expected_per_device:.2f}x")
        
        # Verify scaling
        print(f"\nScaling verification:")
        print(f"  memory_per_device × world_size = {total_memory/world_size:.2f} × {world_size} = {total_memory:.2f} GB")
        print(f"  Expected total = {expected_total:.2f} GB")
        
        if abs(total_memory - expected_total) / expected_total < 0.5:
            print(f"\n✅ Memory scaling is correct (within 50% of theory)")
        else:
            print(f"\n⚠️  Memory overhead is higher than expected")
        
        if imbalance < 10:
            print(f"✅ Memory is well balanced")
        
        print(f"\n{'='*80}")
        print(f"✅ Multi-GPU FSDP Test Completed Successfully!")
        print(f"{'='*80}")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

