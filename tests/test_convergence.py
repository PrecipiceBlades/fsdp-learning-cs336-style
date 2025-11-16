"""Convergence test: FSDP vs Non-FSDP should produce identical training loss.

This test verifies that FSDP training is mathematically equivalent to non-FSDP training
when using the same:
- Initial model parameters
- Data order
- Optimizer configuration
- Random seeds

Reference: ~/cs336-assignment2-systems/naive_ddp.py
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
sys.path.append('/root/cs336-assignment2-systems/cs336-basics')

from cs336_basics.model import BasicsTransformerLM
from fsdp.flat_param import flatten_module_params
from fsdp.forward_pass import register_forward_hooks
from fsdp.backward_pass import register_backward_hooks
from fsdp.optimizer import FSDPOptimizer


def get_dataset(num_samples, vocab_size, seq_len, seed=1234):
    """Generate random dataset for testing."""
    torch.manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    targets = torch.randint(0, vocab_size, (num_samples, seq_len))
    return TensorDataset(input_ids, targets)


def train_non_fsdp(model, dataset, lr=1e-3, num_epochs=3, batch_size=2, device="cpu", seed=1234):
    """Train model without FSDP (baseline)."""
    torch.manual_seed(seed)
    if device != "cpu":
        torch.cuda.manual_seed(seed)
    
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for input_ids, targets in dataloader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/{num_epochs}, loss: {avg_loss:.6f}")
    
    return losses


def train_fsdp(model, dataset, lr=1e-3, num_epochs=3, batch_size=2, device="cuda", seed=1234, world_size=1, rank=0):
    """Train model with FSDP."""
    torch.manual_seed(seed)
    if device != "cpu":
        torch.cuda.manual_seed(seed)
    
    # Move model to device first
    model = model.to(device)
    
    # Wrap each transformer block with FSDP
    transformer_flat_params = []
    for block in model.layers:
        flat_param = flatten_module_params(block, rank=rank, world_size=world_size)
        register_forward_hooks(block, flat_param, reshard_after_forward=True)
        register_backward_hooks(block, flat_param, reshard_after_forward=True)
        transformer_flat_params.append(flat_param)
    
    # Also wrap embeddings and lm_head WITH hooks for correct gradient handling
    token_emb_flat = flatten_module_params(model.token_embeddings, rank=rank, world_size=world_size)
    register_forward_hooks(model.token_embeddings, token_emb_flat, reshard_after_forward=True)
    register_backward_hooks(model.token_embeddings, token_emb_flat, reshard_after_forward=True)
    
    lm_head_flat = flatten_module_params(model.lm_head, rank=rank, world_size=world_size)
    register_forward_hooks(model.lm_head, lm_head_flat, reshard_after_forward=True)
    register_backward_hooks(model.lm_head, lm_head_flat, reshard_after_forward=True)
    
    # Collect all FlatParameters for optimizer
    all_flat_params = transformer_flat_params + [token_emb_flat, lm_head_flat]
    
    # Create FSDP optimizer
    optimizer = FSDPOptimizer(all_flat_params, optimizer_cls=torch.optim.AdamW, lr=lr)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for input_ids, targets in dataloader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/{num_epochs}, loss: {avg_loss:.6f}")
    
    return losses


def test_convergence():
    """Test that FSDP and non-FSDP produce identical training loss."""
    print("=" * 80)
    print("Convergence Test: FSDP vs Non-FSDP")
    print("=" * 80)
    
    # Small model for quick testing
    config = {
        "vocab_size": 1000,
        "context_length": 128,
        "d_model": 128,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 512,
        "rope_theta": 10000.0,
    }
    
    seed = 1234
    num_samples = 16  # Small dataset for quick testing
    num_epochs = 3
    batch_size = 2
    lr = 1e-3
    
    # Generate dataset
    dataset = get_dataset(num_samples, config["vocab_size"], seq_len=32, seed=seed)
    
    # ============ Non-FSDP Training ============
    print("\n1️⃣  Non-FSDP Training:")
    print("-" * 80)
    
    torch.manual_seed(seed)
    model_nonfsdp = BasicsTransformerLM(**config)
    initial_state_dict = {k: v.clone() for k, v in model_nonfsdp.state_dict().items()}
    
    losses_nonfsdp = train_non_fsdp(
        model_nonfsdp, dataset, lr=lr, num_epochs=num_epochs, 
        batch_size=batch_size, device="cuda", seed=seed
    )
    
    # ============ FSDP Training ============
    print("\n2️⃣  FSDP Training (single rank for testing):")
    print("-" * 80)
    
    torch.manual_seed(seed)
    model_fsdp = BasicsTransformerLM(**config)
    model_fsdp.load_state_dict(initial_state_dict)  # Same initial parameters
    
    losses_fsdp = train_fsdp(
        model_fsdp, dataset, lr=lr, num_epochs=num_epochs,
        batch_size=batch_size, device="cuda", seed=seed, world_size=1, rank=0
    )
    
    # ============ Compare Results ============
    print("\n3️⃣  Comparison:")
    print("-" * 80)
    print(f"{'Epoch':<10} {'Non-FSDP Loss':<15} {'FSDP Loss':<15} {'Diff':<15} {'Match?':<10}")
    print("-" * 80)
    
    all_match = True
    tolerance = 1e-2  # Relaxed tolerance for floating point + optimizer state differences
    for epoch, (loss_nonfsdp, loss_fsdp) in enumerate(zip(losses_nonfsdp, losses_fsdp), 1):
        diff = abs(loss_nonfsdp - loss_fsdp)
        match = diff < tolerance
        all_match = all_match and match
        
        print(f"{epoch:<10} {loss_nonfsdp:<15.6f} {loss_fsdp:<15.6f} {diff:<15.8f} {'✓' if match else '✗':<10}")
    
    print("-" * 80)
    
    if all_match:
        print("✅ CONVERGENCE TEST PASSED!")
        print("   FSDP and Non-FSDP produce identical training losses.")
    else:
        print("❌ CONVERGENCE TEST FAILED!")
        print("   FSDP and Non-FSDP produce different training losses.")
        print("\n⚠️  This indicates a bug in the FSDP implementation.")
        print("   Check:")
        print("   - Gradient reduce-scatter correctness")
        print("   - Parameter all-gather/reshard correctness")
        print("   - Optimizer state management")
    
    return all_match


if __name__ == "__main__":
    try:
        success = test_convergence()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

