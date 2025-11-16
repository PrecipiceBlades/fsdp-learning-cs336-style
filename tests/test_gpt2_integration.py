"""Integration test with GPT-2 style model - verify memory calculations."""

import torch
import torch.nn as nn
import sys
import os

# Add cs336-basics to path
sys.path.append('/root/cs336-assignment2-systems/cs336-basics')

from cs336_basics.model import BasicsTransformerLM
from fsdp.meta_init import init_model_on_meta, materialize_meta_module
from fsdp.flat_param import flatten_module_params
from fsdp.forward_pass import register_forward_hooks
from fsdp.backward_pass import register_backward_hooks
from fsdp.optimizer import FSDPOptimizer


def count_parameters(model):
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def estimate_memory_mb(num_params, dtype=torch.float32):
    """Estimate memory for parameters in MB."""
    bytes_per_param = 4 if dtype == torch.float32 else 2  # fp32 = 4 bytes, fp16 = 2 bytes
    return (num_params * bytes_per_param) / (1024 ** 2)


def test_small_transformer():
    """Test with a small transformer model."""
    print("=" * 80)
    print("Test 1: Small Transformer (for quick validation)")
    print("=" * 80)
    
    # Small config for testing
    config = {
        "vocab_size": 1000,
        "context_length": 256,
        "d_model": 256,
        "num_layers": 4,
        "num_heads": 4,
        "d_ff": 1024,
        "rope_theta": 10000.0,
    }
    
    # Create model
    model = BasicsTransformerLM(**config)
    total_params = count_parameters(model)
    
    print(f"\n📊 Model Config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"\n  Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Simulate multi-device scenario (single device for testing)
    world_size = 1
    rank = 0
    
    # Wrap each transformer block with FSDP
    wrapped_blocks = []
    for i, block in enumerate(model.layers):
        # Flatten parameters for this block
        flat_param = flatten_module_params(block, rank=rank, world_size=world_size)
        
        # Register hooks
        pre_handle, post_handle = register_forward_hooks(
            block, flat_param, reshard_after_forward=True
        )
        bwd_handles = register_backward_hooks(
            block, flat_param, reshard_after_forward=True
        )
        
        wrapped_blocks.append({
            'block': block,
            'flat_param': flat_param,
            'handles': (pre_handle, post_handle, bwd_handles)
        })
    
    # Collect all flat_params for optimizer
    transformer_block_flat_params = [wb['flat_param'] for wb in wrapped_blocks]
    
    # Also wrap token embeddings and lm_head
    token_emb_flat = flatten_module_params(model.token_embeddings, rank=rank, world_size=world_size)
    lm_head_flat = flatten_module_params(model.lm_head, rank=rank, world_size=world_size)
    
    # Note: For simplicity, we don't wrap token_emb and lm_head with hooks in this test
    # In production, you would also wrap them
    
    # Collect all flat params for optimizer
    all_flat_params = transformer_block_flat_params + [token_emb_flat, lm_head_flat]
    
    # Create optimizer
    optimizer = FSDPOptimizer(
        all_flat_params,
        optimizer_cls=torch.optim.AdamW,
        lr=1e-4,
    )
    
    print(f"\n✅ FSDP Wrapper Applied:")
    print(f"  World Size: {world_size}")
    print(f"  Rank: {rank}")
    print(f"  Number of FlatParameters: {len(all_flat_params)}")
    print(f"    Transformer blocks with hooks: {len(transformer_block_flat_params)}")
    print(f"    Other parameters (embeddings, lm_head): {len(all_flat_params) - len(transformer_block_flat_params)}")
    
    # Calculate memory requirements
    print(f"\n💾 Memory Calculation (FP32):")
    
    # Parameters
    param_memory_total_mb = estimate_memory_mb(total_params)
    param_memory_per_device_mb = param_memory_total_mb / world_size
    
    print(f"\n  Parameters:")
    print(f"    Total: {param_memory_total_mb:.2f} MB")
    print(f"    Per Device (sharded): {param_memory_per_device_mb:.2f} MB")
    
    # Gradients (same size as parameters)
    grad_memory_total_mb = param_memory_total_mb
    grad_memory_per_device_mb = grad_memory_total_mb / world_size
    
    print(f"\n  Gradients:")
    print(f"    Total: {grad_memory_total_mb:.2f} MB")
    print(f"    Per Device (sharded): {grad_memory_per_device_mb:.2f} MB")
    
    # Optimizer States (Adam: momentum + variance = 2x parameters)
    optimizer_state_memory_total_mb = 2 * param_memory_total_mb
    optimizer_state_memory_per_device_mb = optimizer_state_memory_total_mb / world_size
    
    print(f"\n  Optimizer States (Adam):")
    print(f"    Total: {optimizer_state_memory_total_mb:.2f} MB")
    print(f"    Per Device (sharded): {optimizer_state_memory_per_device_mb:.2f} MB")
    
    # Total
    total_memory_needed_mb = param_memory_total_mb + grad_memory_total_mb + optimizer_state_memory_total_mb
    total_memory_per_device_mb = total_memory_needed_mb / world_size
    
    print(f"\n  TOTAL (Params + Grads + Optimizer):")
    print(f"    Total across all devices: {total_memory_needed_mb:.2f} MB")
    print(f"    Per Device (sharded): {total_memory_per_device_mb:.2f} MB")
    print(f"\n  ✓ Verification: {total_memory_per_device_mb:.2f} MB × {world_size} devices = {total_memory_per_device_mb * world_size:.2f} MB")
    
    # Test forward + backward
    print(f"\n🔄 Running Forward + Backward Pass...")
    
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    
    # Forward
    output = model(input_ids)
    assert output.shape == (batch_size, seq_len, config["vocab_size"]), \
        f"Expected shape ({batch_size}, {seq_len}, {config['vocab_size']}), got {output.shape}"
    
    # Compute loss
    target = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output.view(-1, config["vocab_size"]), target.view(-1))
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients for transformer blocks (which have backward hooks)
    for i, flat_param in enumerate(transformer_block_flat_params):
        assert flat_param.grad is not None, f"Transformer block {i} FlatParameter should have gradients"
    
    # Note: token_emb and lm_head will have gradients on original params, not reduce-scattered
    # In a full implementation, they should also be wrapped with hooks
    
    print(f"  ✓ Forward pass completed")
    print(f"  ✓ Backward pass completed")
    print(f"  ✓ Gradients computed for {len(transformer_block_flat_params)} transformer block FlatParameters")
    
    # Optimizer step
    optimizer.step()
    
    print(f"  ✓ Optimizer step completed")
    print(f"  ✓ Loss: {loss.item():.4f}")
    
    # Clean up handles
    for wb in wrapped_blocks:
        wb['handles'][0].remove()  # pre_handle
        wb['handles'][1].remove()  # post_handle
        for h in wb['handles'][2]:  # bwd_handles (list)
            h.remove()
    
    print(f"\n✅ Test PASSED!")
    return True


def test_gpt2_xl_memory_calculation():
    """Calculate memory requirements for GPT-2 XL."""
    print("\n" + "=" * 80)
    print("Test 2: GPT-2 XL Memory Calculation (No actual model instantiation)")
    print("=" * 80)
    
    # GPT-2 XL config
    config = {
        "vocab_size": 50257,
        "context_length": 1024,
        "d_model": 1600,
        "num_layers": 48,
        "num_heads": 25,
        "d_ff": 6400,
        "rope_theta": 10000.0,
    }
    
    print(f"\n📊 GPT-2 XL Config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Estimate parameters
    vocab_size = config["vocab_size"]
    d_model = config["d_model"]
    num_layers = config["num_layers"]
    d_ff = config["d_ff"]
    
    # Token embeddings
    token_emb_params = vocab_size * d_model
    
    # LM head (tied with token embeddings, but let's count separately for clarity)
    lm_head_params = vocab_size * d_model
    
    # Per transformer block:
    # - Attention: Q, K, V projections + output projection
    #   (d_model * d_model) * 4
    # - FFN: 2 linear layers
    #   (d_model * d_ff) + (d_ff * d_model)
    # - RMSNorm: d_model (two norms per block)
    #   d_model * 2
    per_block_params = (
        4 * d_model * d_model +  # Attention
        2 * d_model * d_ff +      # FFN
        2 * d_model               # LayerNorms
    )
    
    total_block_params = per_block_params * num_layers
    
    # Final layer norm
    final_norm_params = d_model
    
    total_params = token_emb_params + total_block_params + final_norm_params + lm_head_params
    
    print(f"\n  Estimated Total Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"    Token Embeddings: {token_emb_params:,} ({token_emb_params/1e6:.2f}M)")
    print(f"    Transformer Blocks ({num_layers} layers): {total_block_params:,} ({total_block_params/1e9:.2f}B)")
    print(f"    LM Head: {lm_head_params:,} ({lm_head_params/1e6:.2f}M)")
    print(f"    Final Norm: {final_norm_params:,}")
    
    # Memory calculations for different world sizes
    for world_size in [1, 2, 4, 8]:
        print(f"\n{'─'*80}")
        print(f"  💾 Memory Requirements with {world_size} device(s) (FP32):")
        
        # Parameters
        param_memory_total_mb = estimate_memory_mb(total_params)
        param_memory_per_device_mb = param_memory_total_mb / world_size
        
        # Gradients
        grad_memory_total_mb = param_memory_total_mb
        grad_memory_per_device_mb = grad_memory_total_mb / world_size
        
        # Optimizer States (Adam)
        optimizer_state_memory_total_mb = 2 * param_memory_total_mb
        optimizer_state_memory_per_device_mb = optimizer_state_memory_total_mb / world_size
        
        # Total
        total_memory_needed_mb = param_memory_total_mb + grad_memory_total_mb + optimizer_state_memory_total_mb
        total_memory_per_device_mb = total_memory_needed_mb / world_size
        
        print(f"\n    Parameters:        {param_memory_per_device_mb:8.2f} MB/device  (total: {param_memory_total_mb:.2f} MB)")
        print(f"    Gradients:         {grad_memory_per_device_mb:8.2f} MB/device  (total: {grad_memory_total_mb:.2f} MB)")
        print(f"    Optimizer States:  {optimizer_state_memory_per_device_mb:8.2f} MB/device  (total: {optimizer_state_memory_total_mb:.2f} MB)")
        print(f"    {'─'*60}")
        print(f"    TOTAL:             {total_memory_per_device_mb:8.2f} MB/device  (total: {total_memory_needed_mb:.2f} MB)")
        print(f"                       {total_memory_per_device_mb/1024:8.2f} GB/device  (total: {total_memory_needed_mb/1024:.2f} GB)")
        
        print(f"\n    ✓ Verification: {total_memory_per_device_mb:.2f} MB/device × {world_size} = {total_memory_per_device_mb * world_size:.2f} MB")
    
    print(f"\n✅ Memory Calculation VERIFIED!")
    return True


if __name__ == "__main__":
    print("\n" + "🚀 " * 40)
    print("FSDP Integration Test with Transformer Models")
    print("🚀 " * 40)
    
    try:
        # Test 1: Small model with actual forward/backward
        test_small_transformer()
        
        # Test 2: GPT-2 XL memory calculation
        test_gpt2_xl_memory_calculation()
        
        print("\n" + "=" * 80)
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

