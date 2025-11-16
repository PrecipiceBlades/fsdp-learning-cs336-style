"""Strict equivalence test with GPT-2 XL model and more complex training."""

import torch
import torch.nn as nn
import sys
import os
sys.path.append('/root/cs336-assignment2-systems/cs336-basics')

from cs336_basics.model import BasicsTransformerLM
from fsdp.flat_param import flatten_module_params
from fsdp.forward_pass import register_forward_hooks
from fsdp.backward_pass import register_backward_hooks
from fsdp.optimizer import FSDPOptimizer


def get_gpt2_xl_config():
    """GPT-2 XL configuration."""
    return {
        "vocab_size": 50257,
        "context_length": 1024,
        "d_model": 1600,
        "num_layers": 48,
        "num_heads": 25,
        "d_ff": 6400,
        "rope_theta": 10000.0,
    }


def get_gpt2_large_config():
    """GPT-2 Large configuration (for faster testing)."""
    return {
        "vocab_size": 50257,
        "context_length": 1024,
        "d_model": 1280,
        "num_layers": 36,
        "num_heads": 20,
        "d_ff": 5120,
        "rope_theta": 10000.0,
    }


def get_gpt2_medium_config():
    """GPT-2 Medium configuration."""
    return {
        "vocab_size": 50257,
        "context_length": 1024,
        "d_model": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "d_ff": 4096,
        "rope_theta": 10000.0,
    }


def calculate_model_memory(config, world_size=1):
    """Calculate expected memory usage for FSDP."""
    vocab_size = config["vocab_size"]
    d_model = config["d_model"]
    num_layers = config["num_layers"]
    d_ff = config["d_ff"]
    
    # Token embeddings: vocab_size * d_model
    token_emb_params = vocab_size * d_model
    
    # Per-layer parameters
    # Attention: 4 * (d_model * d_model) for q, k, v, output
    attn_params = 4 * d_model * d_model
    # FFN: 3 * (d_model * d_ff) for w1, w2, w3
    ffn_params = d_model * d_ff + d_ff * d_model + d_model * d_ff
    # LayerNorm: 2 * d_model for ln1, ln2
    ln_params = 2 * d_model
    
    layer_params = attn_params + ffn_params + ln_params
    total_layer_params = num_layers * layer_params
    
    # Final layer norm + lm_head (shares with token embeddings in actual GPT-2, but separate here)
    ln_final_params = d_model
    lm_head_params = vocab_size * d_model
    
    total_params = token_emb_params + total_layer_params + ln_final_params + lm_head_params
    
    # Memory calculation (FP32)
    # FSDP: params (N/W) + grads (N/W) + optimizer states (2N/W for Adam) = 4N/W per device
    bytes_per_param = 4  # FP32
    
    # Non-FSDP (single GPU): 4N
    non_fsdp_memory_gb = (4 * total_params * bytes_per_param) / (1024**3)
    
    # FSDP: 4N/W per device
    fsdp_memory_per_device_gb = non_fsdp_memory_gb / world_size
    
    return {
        "total_params": total_params,
        "total_params_m": total_params / 1e6,
        "non_fsdp_memory_gb": non_fsdp_memory_gb,
        "fsdp_memory_per_device_gb": fsdp_memory_per_device_gb,
        "world_size": world_size,
    }


def test_strict_equivalence(
    config_name="gpt2-medium",
    num_epochs=10,
    batch_size=4,
    seq_len=128,
    lr=1e-4,
    seed=42,
):
    """Test strict equivalence between Non-FSDP and FSDP."""
    
    # Select configuration
    if config_name == "gpt2-xl":
        config = get_gpt2_xl_config()
    elif config_name == "gpt2-large":
        config = get_gpt2_large_config()
    elif config_name == "gpt2-medium":
        config = get_gpt2_medium_config()
    else:
        raise ValueError(f"Unknown config: {config_name}")
    
    # Adjust context_length for testing
    config["context_length"] = seq_len
    
    print("=" * 80)
    print(f"Strict Equivalence Test: {config_name}")
    print("=" * 80)
    print(f"Config: d_model={config['d_model']}, num_layers={config['num_layers']}")
    print(f"Training: {num_epochs} epochs, batch_size={batch_size}, seq_len={seq_len}")
    print(f"Optimizer: Adam (lr={lr})")
    
    # Memory calculation
    mem_info = calculate_model_memory(config, world_size=1)
    print(f"\n📊 Memory Calculation:")
    print(f"  Total parameters: {mem_info['total_params_m']:.2f}M")
    print(f"  Expected memory (Non-FSDP): {mem_info['non_fsdp_memory_gb']:.2f} GB")
    print(f"  Expected memory (FSDP, 1 GPU): {mem_info['fsdp_memory_per_device_gb']:.2f} GB")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate more complex dataset
    torch.manual_seed(seed)
    num_samples = 32
    # Use more realistic data distribution
    dataset = []
    for i in range(num_samples):
        # Vary sequence lengths
        actual_seq_len = seq_len if i % 3 == 0 else seq_len // 2
        x = torch.randint(0, config["vocab_size"], (batch_size, actual_seq_len))
        y = torch.randint(0, config["vocab_size"], (batch_size, actual_seq_len))
        dataset.append((x, y))
    
    print(f"\n📦 Dataset: {len(dataset)} batches with varying sequence lengths")
    
    # ========================================================================
    # Non-FSDP Baseline
    # ========================================================================
    print(f"\n{'='*80}")
    print("1️⃣ Training Non-FSDP Baseline")
    print(f"{'='*80}")
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    model_nonfsdp = BasicsTransformerLM(**config).to(device)
    opt_nonfsdp = torch.optim.Adam(model_nonfsdp.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    # Track memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    losses_nonfsdp = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(dataset):
            x, y = x.to(device), y.to(device)
            opt_nonfsdp.zero_grad()
            logits = model_nonfsdp(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            opt_nonfsdp.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataset)
        losses_nonfsdp.append(avg_loss)
        print(f"  Epoch {epoch+1:2d}: loss={avg_loss:.6f}")
    
    if torch.cuda.is_available():
        peak_memory_nonfsdp = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"\n  Peak GPU memory: {peak_memory_nonfsdp:.2f} GB")
    
    # ========================================================================
    # FSDP
    # ========================================================================
    print(f"\n{'='*80}")
    print("2️⃣ Training FSDP")
    print(f"{'='*80}")
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    model_fsdp = BasicsTransformerLM(**config).to(device)
    
    # Wrap all modules
    flat_params = []
    
    # Wrap transformer blocks
    for i, block in enumerate(model_fsdp.layers):
        flat_param = flatten_module_params(block, rank=0, world_size=1)
        register_forward_hooks(block, flat_param, reshard_after_forward=True)
        register_backward_hooks(block, flat_param, reshard_after_forward=True)
        flat_params.append(flat_param)
    
    # Wrap token embeddings
    flat_emb = flatten_module_params(model_fsdp.token_embeddings, rank=0, world_size=1)
    register_forward_hooks(model_fsdp.token_embeddings, flat_emb, reshard_after_forward=True)
    register_backward_hooks(model_fsdp.token_embeddings, flat_emb, reshard_after_forward=True)
    flat_params.append(flat_emb)
    
    # Wrap lm_head
    flat_lm = flatten_module_params(model_fsdp.lm_head, rank=0, world_size=1)
    register_forward_hooks(model_fsdp.lm_head, flat_lm, reshard_after_forward=True)
    register_backward_hooks(model_fsdp.lm_head, flat_lm, reshard_after_forward=True)
    flat_params.append(flat_lm)
    
    # Wrap ln_final
    flat_ln = flatten_module_params(model_fsdp.ln_final, rank=0, world_size=1)
    register_forward_hooks(model_fsdp.ln_final, flat_ln, reshard_after_forward=True)
    register_backward_hooks(model_fsdp.ln_final, flat_ln, reshard_after_forward=True)
    flat_params.append(flat_ln)
    
    opt_fsdp = FSDPOptimizer(flat_params, optimizer_cls=torch.optim.Adam, lr=lr)
    
    # Track memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    losses_fsdp = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(dataset):
            x, y = x.to(device), y.to(device)
            opt_fsdp.zero_grad()
            logits = model_fsdp(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            opt_fsdp.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataset)
        losses_fsdp.append(avg_loss)
        print(f"  Epoch {epoch+1:2d}: loss={avg_loss:.6f}")
    
    if torch.cuda.is_available():
        peak_memory_fsdp = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"\n  Peak GPU memory: {peak_memory_fsdp:.2f} GB")
    
    # ========================================================================
    # Comparison
    # ========================================================================
    print(f"\n{'='*80}")
    print("3️⃣ Equivalence Comparison")
    print(f"{'='*80}")
    
    print(f"\n{'Epoch':<10} {'Non-FSDP Loss':<20} {'FSDP Loss':<20} {'Diff':<20}")
    print("-" * 70)
    
    max_loss_diff = 0
    all_exact = True
    for epoch, (loss_nf, loss_f) in enumerate(zip(losses_nonfsdp, losses_fsdp), 1):
        diff = abs(loss_nf - loss_f)
        max_loss_diff = max(max_loss_diff, diff)
        exact = diff < 1e-10
        all_exact = all_exact and exact
        status = "✅" if exact else "⚠️"
        print(f"{epoch:<10} {loss_nf:<20.15f} {loss_f:<20.15f} {diff:<20.15e} {status}")
    
    print("-" * 70)
    print(f"Max loss diff: {max_loss_diff:.15e}")
    
    if all_exact:
        print("\n✅ EXACTLY EQUIVALENT (all epochs diff = 0.0)")
    elif max_loss_diff < 1e-6:
        print(f"\n✅ PRACTICALLY EQUIVALENT (max diff = {max_loss_diff:.2e})")
    else:
        print(f"\n❌ NOT EQUIVALENT (max diff = {max_loss_diff:.2e})")
    
    # Parameter comparison
    print(f"\n{'='*80}")
    print("4️⃣ Final Parameter Comparison")
    print(f"{'='*80}")
    
    state_nonfsdp = model_nonfsdp.state_dict()
    state_fsdp = model_fsdp.state_dict()
    
    max_param_diff = 0
    for key in sorted(state_nonfsdp.keys()):
        if key in state_fsdp:
            diff = (state_nonfsdp[key] - state_fsdp[key]).abs().max().item()
            max_param_diff = max(max_param_diff, diff)
    
    print(f"Max parameter diff: {max_param_diff:.15e}")
    
    if max_param_diff < 1e-10:
        print("✅ Parameters EXACTLY THE SAME")
    elif max_param_diff < 1e-6:
        print("✅ Parameters PRACTICALLY THE SAME")
    else:
        print(f"❌ Parameters DIFFERENT: {max_param_diff:.2e}")
    
    # Memory comparison
    if torch.cuda.is_available():
        print(f"\n{'='*80}")
        print("5️⃣ Memory Usage Comparison")
        print(f"{'='*80}")
        print(f"Expected (Non-FSDP): {mem_info['non_fsdp_memory_gb']:.2f} GB")
        print(f"Actual   (Non-FSDP): {peak_memory_nonfsdp:.2f} GB")
        print(f"Expected (FSDP):     {mem_info['fsdp_memory_per_device_gb']:.2f} GB")
        print(f"Actual   (FSDP):     {peak_memory_fsdp:.2f} GB")
        print(f"\nMemory ratio (Non-FSDP / FSDP): {peak_memory_nonfsdp / peak_memory_fsdp:.2f}x")
        
        # For world_size=1, FSDP and Non-FSDP should use similar memory
        # (FSDP might use slightly more due to infrastructure overhead)
        if abs(peak_memory_nonfsdp - peak_memory_fsdp) / peak_memory_nonfsdp < 0.2:
            print("✅ Memory usage is similar (as expected for world_size=1)")
        else:
            print(f"⚠️  Memory usage differs more than expected")
    
    return all_exact and max_param_diff < 1e-10


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="gpt2-medium", 
                        choices=["gpt2-medium", "gpt2-large", "gpt2-xl"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    success = test_strict_equivalence(
        config_name=args.config,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
    )
    
    if success:
        print("\n" + "🎉" * 40)
        print("SUCCESS: FSDP is EXACTLY equivalent to Non-FSDP!")
        print("🎉" * 40)
    else:
        print("\n" + "⚠️ " * 40)
        print("WARNING: Some differences detected")
        print("⚠️ " * 40)

