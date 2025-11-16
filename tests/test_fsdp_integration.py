"""Tests for Task 7: Full FSDP Integration.

This test validates that FSDP-trained models match non-distributed training.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.models import ToyModel, SimpleTransformer
from tests.conftest import spawn_for_test, assert_tensors_equal, assert_gradients_equal
from fsdp.utils import setup_distributed, cleanup_distributed
from fsdp.config import FSDPConfig


def train_baseline(model, data, labels, num_steps=10, lr=0.01):
    """Train model without FSDP (baseline)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return losses, model.state_dict()


def train_fsdp(model, data, labels, rank, world_size, num_steps=10, lr=0.01):
    """Train model with FSDP."""
    # TODO: This will be implemented by students
    # Steps:
    # 1. Wrap model with FSDP
    # 2. Create FSDP optimizer
    # 3. Train for num_steps
    # 4. Return losses and final state_dict
    raise NotImplementedError("Students will implement FSDP training")


def _test_fsdp_correctness(rank, world_size):
    """Test that FSDP training matches baseline."""
    setup_distributed(rank, world_size, backend="gloo")
    
    try:
        # Set seed for reproducibility
        torch.manual_seed(42 + rank)
        
        # Create model and data
        model = ToyModel()
        data = torch.randn(8, 10)
        labels = torch.randn(8, 5)
        
        # TODO: Implement FSDP training and compare with baseline
        # if rank == 0:
        #     baseline_losses, baseline_state = train_baseline(
        #         model, data, labels, num_steps=10, lr=0.01
        #     )
        
        # fsdp_losses, fsdp_state = train_fsdp(
        #     model, data, labels, rank, world_size, num_steps=10, lr=0.01
        # )
        
        # # Compare losses
        # if rank == 0:
        #     for step, (bl, fl) in enumerate(zip(baseline_losses, fsdp_losses)):
        #         assert abs(bl - fl) < 1e-4, \
        #             f"Step {step}: baseline loss {bl:.6f} vs FSDP loss {fl:.6f}"
        
    finally:
        cleanup_distributed()


class TestFSDPIntegration:
    """Integration tests for FSDP."""
    
    def test_single_gpu_training(self, device):
        """Test FSDP on single GPU (should work like normal training)."""
        model = ToyModel().to(device)
        
        # TODO: Uncomment when implemented
        # from fsdp.fsdp_module import FSDPModule
        # config = FSDPConfig(reshard_after_forward=True, enable_prefetch=False)
        # fsdp_model = FSDPModule(model, config=config)
        
        # # Test forward pass
        # input_tensor = torch.randn(4, 10, device=device)
        # output = fsdp_model(input_tensor)
        # assert output.shape == (4, 5), f"Output shape should be (4, 5), got {output.shape}"
        
        # # Test backward pass
        # loss = output.sum()
        # loss.backward()
        
        # # Check gradients exist
        # for param in fsdp_model.parameters():
        #     assert param.grad is not None, "All parameters should have gradients"
    
    def test_fsdp_vs_baseline_distributed(self):
        """Test that FSDP training matches baseline in distributed setting."""
        # TODO: Uncomment when implemented
        # spawn_for_test(_test_fsdp_correctness, world_size=2)
        pass
    
    def test_memory_savings(self, device):
        """Test that FSDP reduces memory usage."""
        # TODO: Implement memory profiling test
        # Should show that FSDP uses less memory than baseline
        pass
    
    def test_transformer_training(self, device):
        """Test FSDP with Transformer model."""
        model = SimpleTransformer(n_layers=4, d_model=64).to(device)
        
        # TODO: Test FSDP with Transformer
        # This tests more complex architecture with nested modules
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

