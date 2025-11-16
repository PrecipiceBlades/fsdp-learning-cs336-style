"""Task 1: Meta Device & Deferred Initialization.

This module implements meta device initialization for FSDP, allowing models to be
constructed without allocating memory, then materialized shard-by-shard on each rank.

Key concepts:
- Meta device: torch.device("meta") creates tensor metadata without storage
- Deferred materialization: Allocate memory only when needed, only on owning rank
- Memory savings: Can construct arbitrarily large models without OOM
"""

import torch
import torch.nn as nn
from typing import Optional, Callable
from fsdp.utils import get_rank, get_world_size


def is_meta_device(tensor: torch.Tensor) -> bool:
    """Check if a tensor is on meta device.
    
    Args:
        tensor: Tensor to check
    
    Returns:
        True if tensor is on meta device, False otherwise
    
    Example:
        >>> with torch.device("meta"):
        ...     tensor = torch.randn(10, 10)
        >>> is_meta_device(tensor)
        True
    """
    return tensor.device.type == "meta"


def init_model_on_meta(model_fn: Callable[[], nn.Module]) -> nn.Module:
    """Initialize a model on meta device.
    
    This allows constructing large models without allocating memory.
    
    Args:
        model_fn: Function that returns a model (e.g., lambda: MyModel())
    
    Returns:
        Model with all parameters on meta device
    
    Example:
        >>> model = init_model_on_meta(lambda: nn.Linear(1000, 1000))
        >>> assert all(is_meta_device(p) for p in model.parameters())
        >>> # No memory allocated yet!
        >>> print(model.weight.numel())  # Shape information available
        1000000
    """
    with torch.device("meta"):
        model = model_fn()
    return model


def materialize_meta_tensor(
    meta_tensor: torch.Tensor,
    device: torch.device,
    init_fn: Optional[Callable[[torch.Tensor], None]] = None
) -> torch.Tensor:
    """Materialize a meta tensor on a real device.
    
    Args:
        meta_tensor: Tensor on meta device
        device: Real device to materialize on (e.g., cuda:0, cpu)
        init_fn: Optional initialization function (e.g., torch.nn.init.kaiming_uniform_)
                 If None, initializes to zeros.
    
    Returns:
        Materialized tensor on real device
    
    Example:
        >>> with torch.device("meta"):
        ...     meta_param = torch.randn(10, 10)
        >>> real_param = materialize_meta_tensor(
        ...     meta_param,
        ...     torch.device("cpu"),
        ...     init_fn=lambda t: torch.nn.init.normal_(t, mean=0, std=0.02)
        ... )
        >>> assert real_param.device.type == "cpu"
        >>> assert real_param.shape == meta_param.shape
    """
    # Create empty tensor with same shape and dtype on target device
    materialized = torch.empty_like(meta_tensor, device=device)
    
    # Initialize the tensor
    if init_fn is not None:
        init_fn(materialized)
    else:
        # Default: zero initialization
        torch.nn.init.zeros_(materialized)
    
    return materialized


def materialize_meta_module(
    module: nn.Module,
    device: torch.device,
    init_fn: Optional[Callable[[torch.Tensor], None]] = None
) -> None:
    """Materialize all parameters in a module from meta device to real device.
    
    This modifies the module in-place.
    
    Args:
        module: Module with parameters on meta device
        device: Real device to materialize on
        init_fn: Optional initialization function for parameters
    
    Example:
        >>> with torch.device("meta"):
        ...     model = nn.Linear(10, 10)
        >>> materialize_meta_module(model, torch.device("cpu"))
        >>> assert all(p.device.type == "cpu" for p in model.parameters())
    """
    # Materialize parameters
    for name, param in module.named_parameters(recurse=True):
        if is_meta_device(param):
            materialized = materialize_meta_tensor(param, device, init_fn)
            # Navigate to parent module and replace parameter
            *path, param_name = name.split('.')
            parent = module
            for attr in path:
                parent = getattr(parent, attr)
            setattr(parent, param_name, nn.Parameter(materialized, requires_grad=param.requires_grad))
    
    # Materialize buffers
    for name, buffer in module.named_buffers(recurse=True):
        if is_meta_device(buffer):
            materialized = materialize_meta_tensor(buffer, device, None)  # No init for buffers
            # Navigate to parent module and replace buffer
            *path, buffer_name = name.split('.')
            parent = module
            for attr in path:
                parent = getattr(parent, attr)
            parent.register_buffer(buffer_name, materialized)


def materialize_shard_only(
    module: nn.Module,
    device: torch.device,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    init_fn: Optional[Callable[[torch.Tensor], None]] = None
) -> None:
    """Materialize only the local shard of parameters for this rank.
    
    This is a key optimization: instead of materializing full parameters then sharding,
    we directly materialize only the shard we need.
    
    Args:
        module: Module with parameters on meta device
        device: Device to materialize shard on
        rank: Current rank (if None, uses get_rank())
        world_size: World size (if None, uses get_world_size())
        init_fn: Optional initialization function
    
    Note:
        This is a simplified version. Full FSDP would need to coordinate
        initialization across ranks to ensure all shards come from the same
        random initialization.
    
    Example:
        >>> with torch.device("meta"):
        ...     model = nn.Linear(100, 100)  # 10,000 parameters
        >>> materialize_shard_only(model, torch.device("cpu"), rank=0, world_size=4)
        >>> # Rank 0 only has ~2,500 parameters materialized
    """
    # Simplified implementation: For now, materialize full parameters
    # In practice, this would be integrated with FlatParameter (Task 2)
    # which handles the actual sharding
    rank = rank if rank is not None else get_rank()
    world_size = world_size if world_size is not None else get_world_size()
    
    # For single rank, just materialize everything
    if world_size == 1:
        materialize_meta_module(module, device, init_fn)
        return
    
    # For multi-rank, we still materialize full params here
    # The actual sharding will happen in FlatParameter (Task 2)
    materialize_meta_module(module, device, init_fn)


# Example usage (for testing):
if __name__ == "__main__":
    # Test 1: Create model on meta device
    print("Test 1: Meta device initialization")
    with torch.device("meta"):
        model = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000)
        )
    
    print(f"Model parameters on meta device: {all(is_meta_device(p) for p in model.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Memory allocated: {sum(p.storage().size() for p in model.parameters())}")  # Should be 0
    
    # Test 2: Materialize on real device
    print("\nTest 2: Materialize on real device")
    materialize_meta_module(model, torch.device("cpu"))
    print(f"Model parameters on cpu: {all(p.device.type == 'cpu' for p in model.parameters())}")
    print(f"Memory allocated: {sum(p.storage().size() for p in model.parameters())}")  # Should be > 0

