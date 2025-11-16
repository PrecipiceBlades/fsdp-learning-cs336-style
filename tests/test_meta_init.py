"""Tests for Task 1: Meta Device & Deferred Initialization."""

import pytest
import torch
import torch.nn as nn
from fsdp.meta_init import (
    is_meta_device,
    init_model_on_meta,
    materialize_meta_tensor,
    materialize_meta_module,
)


class TestMetaDevice:
    """Test meta device functionality."""
    
    def test_is_meta_device(self):
        """Test checking if tensor is on meta device."""
        # Create tensor on meta device
        with torch.device("meta"):
            meta_tensor = torch.randn(10, 10)
        
        assert is_meta_device(meta_tensor), "Should detect meta device"
        
        # Create regular tensor
        cpu_tensor = torch.randn(10, 10)
        assert not is_meta_device(cpu_tensor), "Should not detect CPU as meta device"
    
    def test_init_model_on_meta(self):
        """Test initializing model on meta device."""
        # Define model factory
        def create_model():
            return nn.Sequential(
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 50)
            )
        
        model = init_model_on_meta(create_model)
        
        # Check all parameters are on meta device
        for param in model.parameters():
            assert is_meta_device(param), f"Parameter should be on meta device"
            assert param.numel() > 0, "Should have shape information"
            # In PyTorch 2.x, meta device shows size but doesn't actually allocate
            assert param.device.type == "meta", "Should be on meta device"
    
    def test_materialize_meta_tensor(self, device):
        """Test materializing meta tensor to real device."""
        # Create meta tensor
        with torch.device("meta"):
            meta_tensor = torch.randn(10, 10)
        
        # Materialize to real device
        real_tensor = materialize_meta_tensor(meta_tensor, device)
        
        assert real_tensor.device == device, f"Should be on {device}"
        assert real_tensor.shape == meta_tensor.shape, "Shape should match"
        assert real_tensor.storage().size() > 0, "Should have allocated storage"
    
    def test_materialize_meta_tensor_with_init(self, device):
        """Test materializing with custom initialization."""
        with torch.device("meta"):
            meta_tensor = torch.randn(10, 10)
        
        # Materialize with custom init
        def custom_init(t):
            torch.nn.init.constant_(t, 42.0)
        
        real_tensor = materialize_meta_tensor(meta_tensor, device, init_fn=custom_init)
        
        assert torch.allclose(real_tensor, torch.full((10, 10), 42.0, device=device)), \
            "Should use custom initialization"
    
    def test_materialize_meta_module(self, device):
        """Test materializing entire module."""
        # Create model on meta device
        with torch.device("meta"):
            model = nn.Sequential(
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, 5)
            )
        
        # Materialize module
        materialize_meta_module(model, device)
        
        # Check all parameters are on real device
        for param in model.parameters():
            assert param.device == device, f"Parameter should be on {device}"
            assert param.storage().size() > 0, "Should have allocated storage"
        
        # Check model is functional
        input_tensor = torch.randn(3, 10, device=device)
        output = model(input_tensor)
        assert output.shape == (3, 5), "Model should be functional"
    
    def test_memory_savings(self):
        """Test that meta device saves memory."""
        # Create large model on meta device
        with torch.device("meta"):
            large_model = nn.Sequential(*[
                nn.Linear(1000, 1000) for _ in range(10)
            ])
        
        # Total parameters
        total_params = sum(p.numel() for p in large_model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Check all params are on meta device (not actually in memory)
        for param in large_model.parameters():
            assert is_meta_device(param), "All params should be on meta device"
        
        print(f"All {total_params:,} parameters are on meta device (not in actual memory)")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

