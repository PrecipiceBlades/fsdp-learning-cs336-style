"""Tests for Task 3: Forward Pass with All-Gather and Reshard."""

import pytest
import torch
import torch.nn as nn
from fsdp.flat_param import FlatParameter, flatten_module_params
from fsdp.forward_pass import (
    all_gather_params,
    reshard_params,
    register_forward_hooks,
)


class TestForwardPass:
    """Test forward pass functionality."""
    
    def test_all_gather_and_reshard(self):
        """Test all-gather and reshard cycle."""
        # Create parameters
        param1 = nn.Parameter(torch.randn(10, 5))
        param2 = nn.Parameter(torch.randn(10))
        
        # Create FlatParameter (single rank for simplicity)
        flat_param = FlatParameter([param1, param2], rank=0, world_size=1)
        
        # Initially should be sharded
        assert flat_param._is_sharded, "Should start sharded"
        
        # All-gather
        all_gather_params(flat_param)
        assert not flat_param._is_sharded, "Should not be sharded after all-gather"
        assert flat_param.full_param is not None, "Full param should exist"
        
        # Reshard
        reshard_params(flat_param)
        assert flat_param._is_sharded, "Should be sharded after reshard"
        assert flat_param.full_param is None, "Full param should be freed"
    
    def test_forward_with_hooks(self):
        """Test forward pass with hooks."""
        # Create a simple model
        model = nn.Linear(10, 5, bias=False)
        
        # Create FlatParameter
        flat_param = flatten_module_params(model, rank=0, world_size=1)
        
        # Register hooks (with reshard_after_forward=True)
        pre_handle, post_handle = register_forward_hooks(
            model, flat_param, reshard_after_forward=True
        )
        
        # Test forward pass
        input_tensor = torch.randn(3, 10)
        output = model(input_tensor)
        
        assert output.shape == (3, 5), f"Output shape should be (3, 5), got {output.shape}"
        
        # After forward with reshard=True, should be sharded again
        assert flat_param._is_sharded, "Should be resharded after forward"
        
        # Clean up hooks
        pre_handle.remove()
        post_handle.remove()
    
    def test_forward_no_reshard(self):
        """Test forward pass without resharding."""
        # Create model
        model = nn.Linear(10, 5, bias=False)
        
        # Create FlatParameter
        flat_param = flatten_module_params(model, rank=0, world_size=1)
        
        # Register hooks (with reshard_after_forward=False)
        pre_handle, post_handle = register_forward_hooks(
            model, flat_param, reshard_after_forward=False
        )
        
        # Test forward pass
        input_tensor = torch.randn(3, 10)
        output = model(input_tensor)
        
        assert output.shape == (3, 5), "Output shape should be (3, 5)"
        
        # After forward with reshard=False, should NOT be resharded
        assert not flat_param._is_sharded, "Should NOT be resharded after forward"
        assert flat_param.full_param is not None, "Full param should still exist"
        
        # Clean up
        pre_handle.remove()
        post_handle.remove()
    
    def test_forward_correctness(self):
        """Test that forward pass produces correct results."""
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
        
        # Test input
        input_tensor = torch.tensor([[1.0, 1.0, 1.0]])  # Sum will be 6 and 15
        output = model(input_tensor)
        
        expected = torch.tensor([[6.0, 15.0]])
        assert torch.allclose(output, expected, atol=1e-5), \
            f"Output {output} doesn't match expected {expected}"
        
        # Clean up
        pre_handle.remove()
        post_handle.remove()
    
    def test_multiple_forward_passes(self):
        """Test multiple forward passes work correctly."""
        model = nn.Linear(10, 5, bias=False)
        flat_param = flatten_module_params(model, rank=0, world_size=1)
        
        # Register hooks
        pre_handle, post_handle = register_forward_hooks(
            model, flat_param, reshard_after_forward=True
        )
        
        # Multiple forward passes
        for i in range(5):
            input_tensor = torch.randn(3, 10)
            output = model(input_tensor)
            assert output.shape == (3, 5), f"Forward pass {i} failed"
            # Should be resharded after each forward
            assert flat_param._is_sharded, f"Should be resharded after forward pass {i}"
        
        # Clean up
        pre_handle.remove()
        post_handle.remove()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

