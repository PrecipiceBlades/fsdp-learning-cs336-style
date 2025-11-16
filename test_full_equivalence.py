"""Test: Wrap ALL modules and compare with Non-FSDP."""

import torch
import torch.nn as nn
import sys
sys.path.append('/root/cs336-assignment2-systems/cs336-basics')

from cs336_basics.model import BasicsTransformerLM
from fsdp.flat_param import flatten_module_params
from fsdp.forward_pass import register_forward_hooks
from fsdp.backward_pass import register_backward_hooks
from fsdp.optimizer import FSDPOptimizer


def test_full_wrap():
    """Test with ALL modules wrapped."""
    
    config = {
        "vocab_size": 100,
        "context_length": 16,
        "d_model": 32,
        "num_layers": 1,
        "num_heads": 2,
        "d_ff": 64,
        "rope_theta": 10000.0,
    }
    
    seed = 42
    
    # Non-FSDP
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model1 = BasicsTransformerLM(**config).cuda()
    opt1 = torch.optim.SGD(model1.parameters(), lr=0.01)
    
    # FSDP - wrap ALL modules
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model2 = BasicsTransformerLM(**config).cuda()
    
    flat_params = []
    
    # Wrap transformer blocks
    for block in model2.layers:
        flat_param = flatten_module_params(block, rank=0, world_size=1)
        register_forward_hooks(block, flat_param, reshard_after_forward=True)
        register_backward_hooks(block, flat_param, reshard_after_forward=True)
        flat_params.append(flat_param)
    
    # Wrap token embeddings
    flat_emb = flatten_module_params(model2.token_embeddings, rank=0, world_size=1)
    register_forward_hooks(model2.token_embeddings, flat_emb, reshard_after_forward=True)
    register_backward_hooks(model2.token_embeddings, flat_emb, reshard_after_forward=True)
    flat_params.append(flat_emb)
    
    # Wrap lm_head
    flat_lm = flatten_module_params(model2.lm_head, rank=0, world_size=1)
    register_forward_hooks(model2.lm_head, flat_lm, reshard_after_forward=True)
    register_backward_hooks(model2.lm_head, flat_lm, reshard_after_forward=True)
    flat_params.append(flat_lm)
    
    # Wrap ln_final
    flat_ln = flatten_module_params(model2.ln_final, rank=0, world_size=1)
    register_forward_hooks(model2.ln_final, flat_ln, reshard_after_forward=True)
    register_backward_hooks(model2.ln_final, flat_ln, reshard_after_forward=True)
    flat_params.append(flat_ln)
    
    opt2 = FSDPOptimizer(flat_params, optimizer_cls=torch.optim.SGD, lr=0.01)
    
    loss_fn = nn.CrossEntropyLoss()
    
    for iteration in range(1, 6):
        print(f"\n{'='*80}")
        print(f"Iteration {iteration}")
        print(f"{'='*80}")
        
        # Same data
        torch.manual_seed(seed + iteration)
        x = torch.randint(0, config["vocab_size"], (2, 8)).cuda()
        y = torch.randint(0, config["vocab_size"], (2, 8)).cuda()
        
        # Non-FSDP
        opt1.zero_grad()
        logits1 = model1(x)
        loss1 = loss_fn(logits1.view(-1, logits1.size(-1)), y.view(-1))
        loss1.backward()
        opt1.step()
        
        # FSDP
        opt2.zero_grad()
        logits2 = model2(x)
        loss2 = loss_fn(logits2.view(-1, logits2.size(-1)), y.view(-1))
        loss2.backward()
        opt2.step()
        
        diff = abs(loss1.item() - loss2.item())
        print(f"Loss1: {loss1.item():.15f}")
        print(f"Loss2: {loss2.item():.15f}")
        print(f"Diff:  {diff:.15e}")
        
        if diff < 1e-10:
            print("✅ EXACTLY EQUIVALENT")
        elif diff < 1e-6:
            print("✅ PRACTICALLY EQUIVALENT")
        else:
            print(f"❌ HAS DIFFERENCE")
    
    # Final parameter comparison
    print(f"\n{'='*80}")
    print("Final Parameter Comparison")
    print(f"{'='*80}")
    
    state1 = model1.state_dict()
    state2 = model2.state_dict()
    
    max_diff = 0
    for key in sorted(state1.keys()):
        if key in state2:
            diff = (state1[key] - state2[key]).abs().max().item()
            max_diff = max(max_diff, diff)
            print(f"  {key}: {diff:.15e}")
    
    print(f"\nMax parameter diff: {max_diff:.15e}")
    
    if max_diff < 1e-10:
        print("✅ EXACTLY EQUIVALENT")
    elif max_diff < 1e-6:
        print("✅ PRACTICALLY EQUIVALENT")
    else:
        print(f"❌ HAS DIFFERENCE: {max_diff:.15e}")


if __name__ == "__main__":
    test_full_wrap()

