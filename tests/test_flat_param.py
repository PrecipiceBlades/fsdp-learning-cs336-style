"""Tests for Task 2: FlatParameter Implementation."""

import pytest
import torch
import torch.nn as nn
from fsdp.flat_param import FlatParameter, flatten_module_params
from tests.conftest import spawn_for_test, assert_tensors_equal
from fsdp.utils import setup_distributed, cleanup_distributed


class TestFlatParameter:
    """Test FlatParameter functionality."""
    
    def test_flat_parameter_creation(self):
        """Test creating FlatParameter from list of parameters."""
        # Create some parameters
        param1 = nn.Parameter(torch.randn(10, 5))
        param2 = nn.Parameter(torch.randn(10))
        
        # Create FlatParameter
        flat_param = FlatParameter([param1, param2], rank=0, world_size=2)
        
        # Check metadata
        assert len(flat_param._param_shapes) == 2, "Should store 2 parameter shapes"
        assert flat_param._param_shapes[0] == torch.Size([10, 5]), "First param shape"
        assert flat_param._param_shapes[1] == torch.Size([10]), "Second param shape"
        
        # Check sharding
        total_numel = param1.numel() + param2.numel()  # 50 + 10 = 60
        shard_size = total_numel // 2  # 30
        assert flat_param.local_shard.numel() == shard_size, \
            f"Local shard should have {shard_size} elements"
    
    def test_all_gather(self):
        """Test all-gather operation."""
        # Create parameters
        param1 = nn.Parameter(torch.randn(10, 5))
        param2 = nn.Parameter(torch.randn(10))
        
        # Create FlatParameter
        flat_param = FlatParameter([param1, param2], rank=0, world_size=1)
        
        # For single rank, all-gather should return full parameter
        full_param = flat_param.all_gather()
        assert full_param.numel() == 60, "Full parameter should have 60 elements"
        assert not flat_param._is_sharded, "Should be marked as not sharded after all-gather"
    
    def test_reshard(self):
        """Test resharding operation."""
        # Create parameters
        param1 = nn.Parameter(torch.randn(10, 5))
        param2 = nn.Parameter(torch.randn(10))
        
        # Create FlatParameter (single rank for simplicity)
        flat_param = FlatParameter([param1, param2], rank=0, world_size=1)
        
        # All-gather first
        full_param = flat_param.all_gather()
        assert not flat_param._is_sharded, "Should not be sharded after all-gather"
        
        # Now reshard
        flat_param.reshard()
        assert flat_param._is_sharded, "Should be sharded after reshard"
        assert flat_param._full_param is None, "Full parameter should be freed"
    
    def test_create_views(self):
        """Test creating parameter views."""
        # Create parameters with known values
        param1 = nn.Parameter(torch.ones(3, 2))  # 6 elements
        param2 = nn.Parameter(torch.ones(4) * 2)  # 4 elements
        
        # Create FlatParameter (single rank)
        flat_param = FlatParameter([param1, param2], rank=0, world_size=1)
        
        # All-gather to get full parameter
        flat_param.all_gather()
        
        # Create views
        views = flat_param.create_views()
        
        assert len(views) == 2, "Should create 2 views"
        assert views[0].shape == torch.Size([3, 2]), "First view shape"
        assert views[1].shape == torch.Size([4]), "Second view shape"
        
        # Check that views share storage with flat parameter
        views[0][0, 0] = 999
        assert flat_param._full_param[0] == 999, "View should share storage with flat param"
    
    def test_flatten_module_params(self):
        """Test flattening module parameters."""
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 5)
        )
        
        flat_param = flatten_module_params(model, rank=0, world_size=2)
        
        # Check that parameters were flattened
        assert flat_param is not None, "Should create FlatParameter"
        # Total params: Linear1 (10*10=100 + 10 bias) + Linear2 (10*5=50 + 5 bias) = 165
        total_numel = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_numel}")
        
        # With 2 ranks, each should have ~82-83 elements
        expected_shard = total_numel // 2
        assert abs(flat_param.local_shard.numel() - expected_shard) <= 1, \
            f"Shard size should be ~{expected_shard}"


def _test_distributed_flat_parameter(rank, world_size):
    """Test FlatParameter in distributed setting."""
    setup_distributed(rank, world_size, backend="gloo")
    
    try:
        # Create same parameters on all ranks
        torch.manual_seed(42)
        param1 = nn.Parameter(torch.randn(10, 5))
        param2 = nn.Parameter(torch.randn(10))
        
        # TODO: Uncomment when implemented
        # # Create FlatParameter
        # flat_param = FlatParameter([param1, param2], rank=rank, world_size=world_size)
        
        # # Each rank should have different shard
        # total_numel = param1.numel() + param2.numel()
        # expected_shard_size = total_numel // world_size
        # if rank < total_numel % world_size:
        #     expected_shard_size += 1
        
        # assert flat_param.local_shard.numel() == expected_shard_size, \
        #     f"Rank {rank} shard size mismatch"
        
        # # Test all-gather
        # full_param = flat_param.all_gather()
        # assert full_param.numel() == total_numel, \
        #     f"Full parameter should have {total_numel} elements"
        
        # # All ranks should have same full parameter after all-gather
        # if rank == 0:
        #     expected = full_param.clone()
        # else:
        #     # In real test, would use broadcast to compare
        #     pass
        
    finally:
        cleanup_distributed()


class TestFlatParameterDistributed:
    """Test FlatParameter in distributed setting."""
    
    def test_distributed_sharding(self):
        """Test that parameters are correctly sharded across ranks."""
        # TODO: Uncomment when implemented
        # spawn_for_test(_test_distributed_flat_parameter, world_size=2)
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

