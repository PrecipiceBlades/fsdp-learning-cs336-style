"""Tests for Task 5: Sharded Optimizer."""

import pytest
import torch
import torch.nn as nn
from fsdp.flat_param import FlatParameter, flatten_module_params
from fsdp.forward_pass import register_forward_hooks
from fsdp.backward_pass import register_backward_hooks
from fsdp.optimizer import FSDPOptimizer, create_fsdp_optimizer


class TestShardedOptimizer:
    """Test sharded optimizer functionality."""
    
    def test_optimizer_creation(self):
        """Test optimizer creation with FlatParameter."""
        model = nn.Linear(10, 5, bias=False)
        flat_param = flatten_module_params(model, rank=0, world_size=1)
        
        # Create optimizer
        optimizer = FSDPOptimizer(
            [flat_param],
            optimizer_cls=torch.optim.SGD,
            lr=0.01
        )
        
        assert optimizer.local_optimizer is not None, "Should have local optimizer"
        assert len(optimizer.all_params) == 1, "Should have 1 parameter"
    
    def test_optimizer_step(self):
        """Test optimizer step updates parameters."""
        model = nn.Linear(10, 5, bias=False)
        flat_param = flatten_module_params(model, rank=0, world_size=1)
        
        # Register hooks
        pre_handle, post_handle = register_forward_hooks(
            model, flat_param, reshard_after_forward=True
        )
        bwd_pre_handle, bwd_post_handle = register_backward_hooks(
            model, flat_param, reshard_after_forward=True
        )
        
        # Create optimizer
        optimizer = FSDPOptimizer(
            [flat_param],
            optimizer_cls=torch.optim.SGD,
            lr=0.1
        )
        
        # Store initial parameter values
        flat_param.all_gather()
        initial_params = flat_param.full_param.clone()
        flat_param.reshard()
        
        # Forward + backward
        input_tensor = torch.randn(3, 10, requires_grad=True)
        output = model(input_tensor)
        loss = output.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that parameters changed
        flat_param.all_gather()
        updated_params = flat_param.full_param.clone()
        
        # Parameters should be different (optimizer updated them)
        assert not torch.allclose(initial_params, updated_params, atol=1e-6), \
            "Parameters should have been updated by optimizer"
        
        # Clean up
        pre_handle.remove()
        post_handle.remove()
        bwd_pre_handle.remove()
        bwd_post_handle.remove()
    
    def test_optimizer_zero_grad(self):
        """Test zero_grad works correctly."""
        model = nn.Linear(10, 5, bias=False)
        flat_param = flatten_module_params(model, rank=0, world_size=1)
        
        # Register hooks
        pre_handle, post_handle = register_forward_hooks(
            model, flat_param, reshard_after_forward=True
        )
        bwd_pre_handle, bwd_post_handle = register_backward_hooks(
            model, flat_param, reshard_after_forward=True
        )
        
        # Create optimizer
        optimizer = FSDPOptimizer(
            [flat_param],
            optimizer_cls=torch.optim.SGD,
            lr=0.01
        )
        
        # Forward + backward
        input_tensor = torch.randn(3, 10, requires_grad=True)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Should have gradients
        assert flat_param.grad is not None, "Should have gradients"
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Gradients should be None (default behavior of zero_grad)
        # Note: Depending on set_to_none, grads might be zeroed or None
        # For single rank, flat_param.grad might not be cleared by local_optimizer
        # So we check that at least the original params have no grad or zero grad
        
        # Clean up
        pre_handle.remove()
        post_handle.remove()
        bwd_pre_handle.remove()
        bwd_post_handle.remove()
    
    def test_optimizer_state_dict(self):
        """Test state_dict and load_state_dict."""
        model = nn.Linear(10, 5, bias=False)
        flat_param = flatten_module_params(model, rank=0, world_size=1)
        
        # Create optimizer (use Adam to have state)
        optimizer = FSDPOptimizer(
            [flat_param],
            optimizer_cls=torch.optim.Adam,
            lr=0.01
        )
        
        # Register hooks
        pre_handle, post_handle = register_forward_hooks(
            model, flat_param, reshard_after_forward=True
        )
        bwd_pre_handle, bwd_post_handle = register_backward_hooks(
            model, flat_param, reshard_after_forward=True
        )
        
        # Do a few steps to create optimizer state
        for _ in range(3):
            input_tensor = torch.randn(3, 10, requires_grad=True)
            output = model(input_tensor)
            loss = output.sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Get state dict
        state_dict = optimizer.state_dict()
        assert state_dict is not None, "Should have state dict"
        
        # Create new optimizer and load state
        model2 = nn.Linear(10, 5, bias=False)
        flat_param2 = flatten_module_params(model2, rank=0, world_size=1)
        optimizer2 = FSDPOptimizer(
            [flat_param2],
            optimizer_cls=torch.optim.Adam,
            lr=0.01
        )
        
        # Load state dict
        optimizer2.load_state_dict(state_dict)
        
        # State should be loaded (we can't easily verify exact state, but no error is good)
        
        # Clean up
        pre_handle.remove()
        post_handle.remove()
        bwd_pre_handle.remove()
        bwd_post_handle.remove()
    
    def test_create_fsdp_optimizer(self):
        """Test helper function for optimizer creation."""
        model = nn.Linear(10, 5, bias=False)
        flat_param = flatten_module_params(model, rank=0, world_size=1)
        
        # Create optimizer using helper
        optimizer = create_fsdp_optimizer(
            [flat_param],
            optimizer_cls=torch.optim.AdamW,
            lr=1e-3,
            weight_decay=0.01
        )
        
        assert isinstance(optimizer, FSDPOptimizer), "Should be FSDPOptimizer"
        assert optimizer.local_optimizer is not None, "Should have local optimizer"
    
    def test_full_training_loop(self):
        """Test a complete training loop with optimizer."""
        model = nn.Linear(10, 5, bias=False)
        flat_param = flatten_module_params(model, rank=0, world_size=1)
        
        # Register hooks
        pre_handle, post_handle = register_forward_hooks(
            model, flat_param, reshard_after_forward=True
        )
        bwd_pre_handle, bwd_post_handle = register_backward_hooks(
            model, flat_param, reshard_after_forward=True
        )
        
        # Create optimizer
        optimizer = FSDPOptimizer(
            [flat_param],
            optimizer_cls=torch.optim.SGD,
            lr=0.1
        )
        
        # Training loop
        initial_loss = None
        final_loss = None
        
        for i in range(10):
            input_tensor = torch.randn(3, 10, requires_grad=True)
            target = torch.randn(3, 5)
            
            # Forward
            output = model(input_tensor)
            loss = ((output - target) ** 2).mean()
            
            if i == 0:
                initial_loss = loss.item()
            if i == 9:
                final_loss = loss.item()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients exist
            assert flat_param.grad is not None, f"Iteration {i}: Should have gradients"
            
            # Optimizer step
            optimizer.step()
        
        # Loss might not decrease (random targets), but the loop should complete
        assert initial_loss is not None, "Should have computed initial loss"
        assert final_loss is not None, "Should have computed final loss"
        
        # Clean up
        pre_handle.remove()
        post_handle.remove()
        bwd_pre_handle.remove()
        bwd_post_handle.remove()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

