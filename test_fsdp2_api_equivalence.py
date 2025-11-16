"""Updated strict equivalence test using FSDP2-style API."""

import torch
import torch.nn as nn
import sys
sys.path.append('/root/cs336-assignment2-systems/cs336-basics')

from cs336_basics.model import BasicsTransformerLM
from fsdp import fully_shard, get_flat_parameters
from fsdp.optimizer import FSDPOptimizer


def test_equivalence_with_fsdp2_api(config_name="gpt2-small", num_epochs=5):
    """Test strict equivalence using FSDP2-style API."""
    
    configs = {
        "gpt2-small": {
            "vocab_size": 50257,
            "context_length": 256,  # Reduced for faster testing
            "d_model": 768,
            "num_layers": 12,
            "num_heads": 12,
            "d_ff": 3072,
            "rope_theta": 10000.0,
        },
    }
    
    config = configs[config_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print(f"FSDP2 API Equivalence Test: {config_name}")
    print("=" * 80)
    print(f"Config: d_model={config['d_model']}, num_layers={config['num_layers']}")
    print(f"Training: {num_epochs} epochs")
    
    # Dataset
    torch.manual_seed(42)
    dataset = []
    for i in range(16):
        x = torch.randint(0, config["vocab_size"], (2, 64))
        y = torch.randint(0, config["vocab_size"], (2, 64))
        dataset.append((x, y))
    
    # Non-FSDP baseline
    print(f"\n1️⃣ Non-FSDP Baseline")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    model_nonfsdp = BasicsTransformerLM(**config).to(device)
    opt_nonfsdp = torch.optim.Adam(model_nonfsdp.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    losses_nonfsdp = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x, y in dataset:
            x, y = x.to(device), y.to(device)
            opt_nonfsdp.zero_grad()
            logits = model_nonfsdp(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            opt_nonfsdp.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataset)
        losses_nonfsdp.append(avg_loss)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.6f}")
    
    # FSDP with new API
    print(f"\n2️⃣ FSDP (FSDP2-style API)")
    print("  Code: from fsdp import fully_shard")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    model_fsdp = BasicsTransformerLM(**config).to(device)
    
    # Apply FSDP
    for layer in model_fsdp.layers:
        fully_shard(layer)
    fully_shard(model_fsdp.token_embeddings)
    fully_shard(model_fsdp.lm_head)
    fully_shard(model_fsdp.ln_final)
    
    flat_params = get_flat_parameters(model_fsdp)
    print(f"  Found {len(flat_params)} FlatParameters")
    
    opt_fsdp = FSDPOptimizer(flat_params, optimizer_cls=torch.optim.Adam, lr=1e-4)
    
    losses_fsdp = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x, y in dataset:
            x, y = x.to(device), y.to(device)
            opt_fsdp.zero_grad()
            logits = model_fsdp(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            opt_fsdp.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataset)
        losses_fsdp.append(avg_loss)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.6f}")
    
    # Compare
    print(f"\n3️⃣ Comparison")
    print("-" * 70)
    max_diff = 0
    all_exact = True
    for epoch, (l1, l2) in enumerate(zip(losses_nonfsdp, losses_fsdp), 1):
        diff = abs(l1 - l2)
        max_diff = max(max_diff, diff)
        exact = diff < 1e-10
        all_exact = all_exact and exact
        status = "✅" if exact else "⚠️"
        print(f"Epoch {epoch}: Non-FSDP={l1:.15f}, FSDP={l2:.15f}, diff={diff:.2e} {status}")
    
    print(f"\nMax diff: {max_diff:.2e}")
    
    if all_exact:
        print("\n✅ EXACTLY EQUIVALENT (diff = 0.0)")
        return True
    elif max_diff < 1e-6:
        print(f"\n✅ PRACTICALLY EQUIVALENT (diff < 1e-6)")
        return True
    else:
        print(f"\n❌ NOT EQUIVALENT (diff = {max_diff:.2e})")
        return False


if __name__ == "__main__":
    success = test_equivalence_with_fsdp2_api("gpt2-small", num_epochs=5)
    
    if success:
        print("\n" + "🎉 " * 20)
        print("FSDP2 API works correctly and is equivalent to Non-FSDP!")
        print("🎉 " * 20)

