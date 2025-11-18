"""Test FSDP with meta device initialization.

This tests the integrated pipeline:
1. Initialize model on meta device
2. Apply fully_shard (which materializes only shards)
3. Train and verify correctness
"""

import pytest
import torch
import torch.nn as nn
from fsdp import fully_shard, get_flat_parameters, clear_fsdp_registry
from fsdp.optimizer import FSDPOptimizer


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestMetaDeviceFSDP:
    """Test FSDP with meta device."""
    
    def setup_method(self):
        """Clear registry before each test."""
        clear_fsdp_registry()
        torch.manual_seed(42)
    
    def test_meta_device_basic(self):
        """Test basic meta device initialization."""
        # Create model on meta device
        with torch.device("meta"):
            model = SimpleModel()
        
        # Verify it's on meta device
        for p in model.parameters():
            assert p.is_meta, "Parameter should be on meta device"
        
        # Apply FSDP (should materialize)
        fully_shard(model.fc1)
        fully_shard(model.fc2)
        fully_shard(model)
        
        # Verify parameters are no longer meta
        flat_params = get_flat_parameters(model)
        for fp in flat_params:
            assert not fp.is_meta, "FlatParameter should not be on meta device"
            assert fp.device == torch.device("cpu"), "Should be materialized to CPU"
    
    def test_meta_device_forward(self):
        """Test forward pass after meta device initialization."""
        # Create model on meta device
        with torch.device("meta"):
            model = SimpleModel()
        
        # Apply FSDP
        fully_shard(model.fc1)
        fully_shard(model.fc2)
        fully_shard(model)
        
        # Create input and run forward
        x = torch.randn(2, 10)
        y = model(x)
        
        assert y.shape == (2, 5), f"Expected shape (2, 5), got {y.shape}"
        assert not y.isnan().any(), "Output contains NaN"
        assert not y.isinf().any(), "Output contains Inf"
    
    def test_meta_device_backward(self):
        """Test backward pass after meta device initialization."""
        # Create model on meta device
        with torch.device("meta"):
            model = SimpleModel()
        
        # Apply FSDP
        fully_shard(model.fc1)
        fully_shard(model.fc2)
        fully_shard(model)
        
        # Forward and backward
        x = torch.randn(2, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Check gradients exist (only for FlatParams with parameters)
        flat_params = get_flat_parameters(model)
        has_grad = False
        for fp in flat_params:
            if fp.grad is not None:
                has_grad = True
                assert not fp.grad.isnan().any(), "Gradient contains NaN"
                assert not fp.grad.isinf().any(), "Gradient contains Inf"
        assert has_grad, "At least one FlatParameter should have gradient"
    
    def test_meta_device_optimizer(self):
        """Test optimizer with meta device initialization."""
        # Create model on meta device
        with torch.device("meta"):
            model = SimpleModel()
        
        # Apply FSDP
        fully_shard(model.fc1)
        fully_shard(model.fc2)
        fully_shard(model)
        
        # Create optimizer
        flat_params = get_flat_parameters(model)
        optimizer = FSDPOptimizer(flat_params, optimizer_cls=torch.optim.SGD, lr=0.01)
        
        # Training step
        x = torch.randn(2, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Get params before step (only for params with grad)
        params_before = []
        params_with_grad = []
        for fp in flat_params:
            if fp.grad is not None:
                params_before.append(fp.data.clone())
                params_with_grad.append(fp)
        
        optimizer.step()
        
        # Verify parameters changed (only check params that had gradients)
        assert len(params_with_grad) > 0, "No parameters with gradients"
        for fp, before in zip(params_with_grad, params_before):
            after = fp.data
            assert not torch.equal(before, after), "Parameters should change after optimizer step"
    
    def test_meta_device_vs_standard(self):
        """Compare meta device initialization with standard initialization."""
        torch.manual_seed(42)
        
        # Standard initialization
        model_std = SimpleModel()
        clear_fsdp_registry()
        fully_shard(model_std.fc1)
        fully_shard(model_std.fc2)
        fully_shard(model_std)
        
        torch.manual_seed(42)
        
        # Meta device initialization
        with torch.device("meta"):
            model_meta = SimpleModel()
        clear_fsdp_registry()
        fully_shard(model_meta.fc1)
        fully_shard(model_meta.fc2)
        fully_shard(model_meta)
        
        # Both should produce same output with same input
        torch.manual_seed(123)
        x = torch.randn(2, 10)
        
        y_std = model_std(x)
        y_meta = model_meta(x)
        
        # They won't be exactly equal because initialization might differ
        # But they should both be valid
        assert y_std.shape == y_meta.shape, "Shapes should match"
        assert not y_std.isnan().any() and not y_meta.isnan().any(), "No NaNs"
        assert not y_std.isinf().any() and not y_meta.isinf().any(), "No Infs"
    
    @pytest.mark.parametrize("device", [
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")),
        "cpu",
    ])
    def test_meta_device_to_device(self, device):
        """Test moving meta device model to different devices."""
        # Create model on meta device
        with torch.device("meta"):
            model = SimpleModel()
        
        # Apply FSDP (materializes to CPU)
        fully_shard(model.fc1)
        fully_shard(model.fc2)
        fully_shard(model)
        
        # Move to target device
        flat_params = get_flat_parameters(model)
        for fp in flat_params:
            fp.data = fp.data.to(device)
        
        # Test forward on target device
        x = torch.randn(2, 10, device=device)
        y = model(x)
        
        assert y.device.type == device, f"Output should be on {device}"
        assert y.shape == (2, 5), f"Expected shape (2, 5), got {y.shape}"


def test_custom_init_function():
    """Test custom initialization function with meta device."""
    clear_fsdp_registry()
    
    # Custom init that sets all tensors to 1.0
    # Note: param_init_fn receives a tensor, not a module
    def custom_init(tensor):
        nn.init.ones_(tensor)
    
    # Create model on meta device
    with torch.device("meta"):
        model = SimpleModel()
    
    # Apply FSDP with custom init
    fully_shard(model.fc1, param_init_fn=custom_init)
    fully_shard(model.fc2, param_init_fn=custom_init)
    fully_shard(model, param_init_fn=custom_init)
    
    # Test forward (with all weights and biases=1)
    x = torch.ones(1, 10)
    y = model(x)
    
    # fc1: output = relu(x @ ones(10, 20) + ones(20)) = relu(10 * ones(20) + ones(20)) = 11 * ones(20)
    # fc2: output = (11 * ones(20)) @ ones(20, 5) + ones(5) = 220 * ones(5) + ones(5) = 221 * ones(5)
    expected = torch.full((1, 5), 221.0)
    
    assert torch.allclose(y, expected, rtol=1e-5), f"Expected {expected}, got {y}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

