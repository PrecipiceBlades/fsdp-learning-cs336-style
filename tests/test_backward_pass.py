"""Tests for Task 4: Backward Pass with All-Gather and Reduce-Scatter."""

import pytest
import torch
import torch.nn as nn
from fsdp.flat_param import FlatParameter, flatten_module_params
from fsdp.forward_pass import register_forward_hooks
from fsdp.backward_pass import (
    reduce_scatter_grads,
    register_backward_hooks,
)


class TestBackwardPass:
    """Test backward pass functionality."""
    
    def test_reduce_scatter_grads(self):
        """Test gradient reduce-scatter."""
        # Create parameters
        param1 = nn.Parameter(torch.randn(10, 5))
        param2 = nn.Parameter(torch.randn(10))
        
        # Create FlatParameter (single rank)
        flat_param = FlatParameter([param1, param2], rank=0, world_size=1)
        
        # Simulate having a full gradient
        flat_param.all_gather()
        flat_param.full_param.grad = torch.randn_like(flat_param.full_param)
        
        # Reduce-scatter
        reduce_scatter_grads(flat_param)
        
        # Should have gradient on local shard
        assert flat_param.grad is not None, "Should have gradient after reduce-scatter"
        assert flat_param.grad.shape == flat_param.data.shape, "Gradient shape should match local shard"
    
    def test_backward_with_hooks(self):
        """Test backward pass with hooks."""
        # Create model
        model = nn.Linear(10, 5, bias=False)
        
        # Create FlatParameter
        flat_param = flatten_module_params(model, rank=0, world_size=1)
        
        # Register forward and backward hooks
        pre_handle, post_handle = register_forward_hooks(
            model, flat_param, reshard_after_forward=True
        )
        bwd_pre_handle, bwd_post_handle = register_backward_hooks(
            model, flat_param, reshard_after_forward=True
        )
        
        # Forward pass
        input_tensor = torch.randn(3, 10, requires_grad=True)
        output = model(input_tensor)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Should have gradients on flat_param
        assert flat_param.grad is not None, "Should have gradients after backward"
        
        # Should be resharded after backward
        assert flat_param._is_sharded, "Should be resharded after backward"
        
        # Clean up
        pre_handle.remove()
        post_handle.remove()
        bwd_pre_handle.remove()
        bwd_post_handle.remove()
    
    def test_backward_no_reshard_forward(self):
        """Test backward when forward didn't reshard."""
        model = nn.Linear(10, 5, bias=False)
        flat_param = flatten_module_params(model, rank=0, world_size=1)
        
        # Register hooks with reshard_after_forward=False
        pre_handle, post_handle = register_forward_hooks(
            model, flat_param, reshard_after_forward=False
        )
        bwd_pre_handle, bwd_post_handle = register_backward_hooks(
            model, flat_param, reshard_after_forward=False
        )
        
        # Forward
        input_tensor = torch.randn(3, 10, requires_grad=True)
        output = model(input_tensor)
        
        # Parameters should still be full after forward
        assert not flat_param._is_sharded, "Should not be resharded after forward"
        
        # Backward
        loss = output.sum()
        loss.backward()
        
        # Should have gradients
        assert flat_param.grad is not None, "Should have gradients"
        
        # Should be resharded after backward
        assert flat_param._is_sharded, "Should be resharded after backward"
        
        # Clean up
        pre_handle.remove()
        post_handle.remove()
        bwd_pre_handle.remove()
        bwd_post_handle.remove()
    
    def test_gradient_correctness(self):
        """Test that gradients are correct."""
        # Create model with known parameters
        model = nn.Linear(3, 2, bias=False)
        
        # Set known weights
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        
        # Create FlatParameter
        flat_param = flatten_module_params(model, rank=0, world_size=1)
        
        # Register hooks
        pre_handle, post_handle = register_forward_hooks(
            model, flat_param, reshard_after_forward=True
        )
        bwd_pre_handle, bwd_post_handle = register_backward_hooks(
            model, flat_param, reshard_after_forward=True
        )
        
        # Forward
        input_tensor = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)
        output = model(input_tensor)  # [6.0, 15.0]
        
        # Backward
        loss = output.sum()  # 21.0
        loss.backward()
        
        # Expected gradient: d(loss)/d(weight) = input.T @ d(loss)/d(output)
        # d(loss)/d(output) = [1.0, 1.0]
        # input.T = [[1.0], [1.0], [1.0]]
        # Expected grad = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        expected_grad = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]).flatten()
        
        # Check gradient (should be in flat_param.grad now)
        assert flat_param.grad is not None, "Should have gradients"
        assert torch.allclose(flat_param.grad, expected_grad, atol=1e-5), \
            f"Gradient {flat_param.grad} doesn't match expected {expected_grad}"
        
        # Clean up
        pre_handle.remove()
        post_handle.remove()
        bwd_pre_handle.remove()
        bwd_post_handle.remove()
    
    def test_multiple_backward_passes(self):
        """Test multiple forward-backward passes."""
        model = nn.Linear(10, 5, bias=False)
        flat_param = flatten_module_params(model, rank=0, world_size=1)
        
        # Register hooks
        pre_handle, post_handle = register_forward_hooks(
            model, flat_param, reshard_after_forward=True
        )
        bwd_pre_handle, bwd_post_handle = register_backward_hooks(
            model, flat_param, reshard_after_forward=True
        )
        
        # Multiple forward-backward cycles
        for i in range(5):
            # Forward
            input_tensor = torch.randn(3, 10, requires_grad=True)
            output = model(input_tensor)
            
            # Backward
            loss = output.sum()
            loss.backward()
            
            # Check gradients exist
            assert flat_param.grad is not None, f"Iteration {i}: Should have gradients"
            
            # Should be resharded
            assert flat_param._is_sharded, f"Iteration {i}: Should be resharded"
            
            # Zero gradients for next iteration
            flat_param.grad = None
        
        # Clean up
        pre_handle.remove()
        post_handle.remove()
        bwd_pre_handle.remove()
        bwd_post_handle.remove()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

