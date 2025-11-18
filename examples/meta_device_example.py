"""Example: Using FSDP with Meta Device Initialization

This example demonstrates how to use meta device initialization with FSDP,
which is memory-efficient for large models.

Key benefits:
1. Model initialization doesn't allocate memory
2. Only shards are materialized on each device
3. Supports custom initialization functions
"""

import torch
import torch.nn as nn
from fsdp import fully_shard, get_flat_parameters
from fsdp.optimizer import FSDPOptimizer


class TinyTransformer(nn.Module):
    """Tiny transformer for demonstration."""
    def __init__(self, vocab_size=1000, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


class TransformerBlock(nn.Module):
    """Single transformer block."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        # Feedforward
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x


def count_parameters(model):
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def main():
    print("=" * 70)
    print("FSDP with Meta Device Initialization Example")
    print("=" * 70)
    
    # Method 1: Standard initialization (allocates memory immediately)
    print("\n[Method 1] Standard Initialization")
    print("-" * 70)
    
    model_standard = TinyTransformer()
    n_params = count_parameters(model_standard)
    print(f"Model parameters: {n_params:,}")
    print(f"Memory (FP32): {n_params * 4 / 1024**2:.2f} MB")
    
    # Apply FSDP
    for layer in model_standard.layers:
        fully_shard(layer)
    fully_shard(model_standard.embed)
    fully_shard(model_standard.output)
    fully_shard(model_standard)
    
    # Test forward pass
    x = torch.randint(0, 1000, (2, 10))
    y = model_standard(x)
    print(f"Output shape: {y.shape}")
    print(f"✓ Standard initialization works!")
    
    # Method 2: Meta device initialization (memory efficient)
    print("\n[Method 2] Meta Device Initialization (Memory Efficient)")
    print("-" * 70)
    
    # Initialize on meta device (no memory allocated)
    with torch.device("meta"):
        model_meta = TinyTransformer()
    
    print(f"Model parameters: {count_parameters(model_meta):,}")
    print(f"Memory allocated: 0 MB (model is on meta device)")
    print(f"All parameters are on meta device: {all(p.is_meta for p in model_meta.parameters())}")
    
    # Apply FSDP (automatically materializes only shards)
    print("\nApplying FSDP...")
    for layer in model_meta.layers:
        fully_shard(layer)
    fully_shard(model_meta.embed)
    fully_shard(model_meta.output)
    fully_shard(model_meta)
    
    # Check that parameters are now materialized
    flat_params = get_flat_parameters(model_meta)
    print(f"Number of FlatParameters: {len(flat_params)}")
    total_shard_size = sum(fp.numel() for fp in flat_params)
    print(f"Total shard size: {total_shard_size:,} parameters")
    print(f"Memory (FP32): {total_shard_size * 4 / 1024**2:.2f} MB")
    
    # Test forward pass
    x = torch.randint(0, 1000, (2, 10))
    y = model_meta(x)
    print(f"Output shape: {y.shape}")
    print(f"✓ Meta device initialization works!")
    
    # Method 3: Meta device with custom initialization
    print("\n[Method 3] Meta Device with Custom Initialization")
    print("-" * 70)
    
    # Custom initialization function
    def custom_init(tensor):
        """Initialize with small values."""
        nn.init.normal_(tensor, mean=0, std=0.02)
    
    # Initialize on meta device
    with torch.device("meta"):
        model_custom = TinyTransformer()
    
    # Apply FSDP with custom init
    for layer in model_custom.layers:
        fully_shard(layer, param_init_fn=custom_init)
    fully_shard(model_custom.embed, param_init_fn=custom_init)
    fully_shard(model_custom.output, param_init_fn=custom_init)
    fully_shard(model_custom, param_init_fn=custom_init)
    
    # Check parameter values
    flat_params = get_flat_parameters(model_custom)
    for i, fp in enumerate(flat_params):
        if fp.numel() > 0:
            print(f"FlatParam {i}: mean={fp.mean().item():.6f}, std={fp.std().item():.6f}")
    
    print(f"✓ Custom initialization works!")
    
    # Training example
    print("\n[Training Example]")
    print("-" * 70)
    
    # Create optimizer
    optimizer = FSDPOptimizer(
        get_flat_parameters(model_meta),
        optimizer_cls=torch.optim.Adam,
        lr=1e-3
    )
    
    # Training loop
    print("Running 5 training steps...")
    for step in range(5):
        # Forward
        x = torch.randint(0, 1000, (2, 10))
        y = model_meta(x)
        target = torch.randint(0, 1000, (2, 10))
        
        # Loss (simple example)
        loss = torch.nn.functional.cross_entropy(
            y.view(-1, y.size(-1)),
            target.view(-1)
        )
        
        # Backward
        loss.backward()
        
        # Update
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"  Step {step + 1}: loss = {loss.item():.4f}")
    
    print(f"✓ Training works!")
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print("✓ Standard initialization: Simple, allocates full model")
    print("✓ Meta device initialization: Memory efficient, only allocates shards")
    print("✓ Custom initialization: Full control over parameter initialization")
    print("✓ All methods support training with FSDP")


if __name__ == "__main__":
    # Import here to avoid circular imports in test
    from fsdp import clear_fsdp_registry
    clear_fsdp_registry()
    
    main()

