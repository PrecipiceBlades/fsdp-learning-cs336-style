"""Memory profiling script for FSDP.

This script profiles memory usage of FSDP vs. baseline training.
"""

import argparse
import torch
import torch.nn as nn
from tests.models import SimpleTransformer
import torch.multiprocessing as mp
from fsdp.utils import setup_distributed, cleanup_distributed


def profile_baseline(model_config, batch_size, seq_len, device):
    """Profile memory usage of baseline training."""
    model = SimpleTransformer(**model_config).to(device)
    
    # Generate dummy data
    input_ids = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len), device=device)
    labels = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len), device=device)
    
    # Track memory
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    
    # Forward pass
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = model(input_ids)
    loss = nn.functional.cross_entropy(
        output.view(-1, model_config['vocab_size']),
        labels.view(-1)
    )
    
    if device.type == 'cuda':
        forward_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
    
    # Backward pass
    loss.backward()
    
    if device.type == 'cuda':
        backward_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
    
    optimizer.step()
    
    if device.type == 'cuda':
        total_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
    else:
        forward_memory = backward_memory = total_memory = 0.0
    
    return {
        'forward_memory_gb': forward_memory,
        'backward_memory_gb': backward_memory,
        'total_memory_gb': total_memory,
        'num_parameters': sum(p.numel() for p in model.parameters()),
    }


def profile_fsdp(model_config, batch_size, seq_len, rank, world_size):
    """Profile memory usage of FSDP training."""
    # TODO: Implement FSDP memory profiling
    # Students will implement this
    raise NotImplementedError("Students will implement FSDP memory profiling")


def main():
    parser = argparse.ArgumentParser(description="Profile memory usage")
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--world_size", type=int, default=1)
    
    args = parser.parse_args()
    
    model_config = {
        'n_layers': args.n_layers,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'vocab_size': args.vocab_size,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("Memory Profiling Results")
    print("=" * 60)
    print(f"Model config: {model_config}")
    print(f"Batch size: {args.batch_size}, Seq len: {args.seq_len}")
    print()
    
    # Profile baseline
    print("Profiling baseline...")
    baseline_stats = profile_baseline(model_config, args.batch_size, args.seq_len, device)
    print(f"Baseline - Parameters: {baseline_stats['num_parameters']:,}")
    if device.type == 'cuda':
        print(f"Baseline - Forward memory: {baseline_stats['forward_memory_gb']:.2f} GB")
        print(f"Baseline - Backward memory: {baseline_stats['backward_memory_gb']:.2f} GB")
        print(f"Baseline - Total memory: {baseline_stats['total_memory_gb']:.2f} GB")
    print()
    
    # TODO: Profile FSDP
    # print("Profiling FSDP...")
    # fsdp_stats = profile_fsdp(model_config, args.batch_size, args.seq_len, 0, args.world_size)
    # print(f"FSDP - Forward memory: {fsdp_stats['forward_memory_gb']:.2f} GB")
    # print(f"FSDP - Backward memory: {fsdp_stats['backward_memory_gb']:.2f} GB")
    # print(f"FSDP - Total memory: {fsdp_stats['total_memory_gb']:.2f} GB")
    # print(f"FSDP - Memory reduction: {baseline_stats['total_memory_gb'] / fsdp_stats['total_memory_gb']:.2f}x")


if __name__ == "__main__":
    main()

