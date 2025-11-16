"""Task 7: Full FSDP Module Integration.

This module integrates all FSDP components into a unified FSDPModule wrapper.

Key concepts:
- Module wrapping: Wrap nn.Module to add FSDP functionality
- Hook management: Register and manage forward/backward hooks
- Nested wrapping: Support wrapping submodules (e.g., each Transformer layer)
- Configuration: Use FSDPConfig for behavior control

Components integrated:
1. Meta device initialization (Task 1)
2. FlatParameter (Task 2)
3. Forward pass hooks (Task 3)
4. Backward pass hooks (Task 4)
5. Sharded optimizer (Task 5)
6. Prefetching (Task 6)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Callable, Any
from fsdp.config import FSDPConfig
from fsdp.flat_param import FlatParameter, flatten_module_params
from fsdp.forward_pass import register_forward_hooks
from fsdp.backward_pass import register_backward_hooks
from fsdp.prefetch import PrefetchManager
from fsdp.utils import get_rank, get_world_size, is_distributed


class FSDPModule(nn.Module):
    """FSDP-wrapped module with parameter sharding and communication management.
    
    This class wraps an nn.Module to add FSDP functionality:
    - Parameters are flattened and sharded across ranks
    - Forward pass: all-gather before computation
    - Backward pass: reduce-scatter after computation
    - Optional prefetching for performance
    
    Attributes:
        module: Wrapped module
        flat_param: Flattened and sharded parameter
        config: FSDP configuration
        rank: Current rank
        world_size: World size
        _hook_handles: List of registered hook handles
    
    Example:
        >>> model = nn.Sequential(
        ...     nn.Linear(100, 100),
        ...     nn.ReLU(),
        ...     nn.Linear(100, 50)
        ... )
        >>> fsdp_model = FSDPModule(model, config=FSDPConfig(reshard_after_forward=True))
        >>> output = fsdp_model(input)
        >>> loss.backward()
    """
    
    def __init__(
        self,
        module: nn.Module,
        config: Optional[FSDPConfig] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        """Initialize FSDP module.
        
        Args:
            module: Module to wrap with FSDP
            config: FSDP configuration (if None, uses default)
            rank: Current rank (if None, uses get_rank())
            world_size: World size (if None, uses get_world_size())
        """
        super().__init__()
        
        # TODO: Implement FSDP module initialization
        # Steps:
        # 1. Store configuration
        # 2. Store rank and world_size
        # 3. Store wrapped module
        # 4. Flatten and shard module parameters
        # 5. Register forward and backward hooks
        # 6. Initialize prefetch manager if enabled
        #
        # Hints:
        # - Use flatten_module_params() to create FlatParameter
        # - Use register_forward_hooks() and register_backward_hooks()
        # - Store hook handles for later removal
        # - Handle case where module has no parameters
        
        self.config = config if config is not None else FSDPConfig()
        self.rank = rank if rank is not None else get_rank()
        self.world_size = world_size if world_size is not None else get_world_size()
        
        self.module = module
        self.flat_param: Optional[FlatParameter] = None
        self._hook_handles: List[Any] = []
        self.prefetch_manager: Optional[PrefetchManager] = None
        
        raise NotImplementedError("TODO: Task 7 - Initialize FSDP module")
    
    def forward(self, *args, **kwargs):
        """Forward pass through wrapped module.
        
        The registered hooks will handle:
        - All-gathering parameters before forward
        - Optionally resharding after forward
        """
        # TODO: Implement forward pass
        # Simply call self.module(*args, **kwargs)
        # Hooks will handle all-gather and reshard automatically
        raise NotImplementedError("TODO: Task 7 - Forward pass")
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        # TODO: Remove all hooks
        # Iterate through self._hook_handles and call handle.remove()
        raise NotImplementedError("TODO: Task 7 - Remove hooks")
    
    def __del__(self):
        """Cleanup when module is deleted."""
        self._remove_hooks()


def wrap_fsdp(
    module: nn.Module,
    config: Optional[FSDPConfig] = None,
    wrap_policy: Optional[Callable[[nn.Module], bool]] = None,
) -> nn.Module:
    """Wrap a module and optionally its submodules with FSDP.
    
    Args:
        module: Module to wrap
        config: FSDP configuration
        wrap_policy: Optional function that returns True for modules to wrap
                     If None, only wraps the top-level module
    
    Returns:
        FSDP-wrapped module
    
    Example:
        >>> # Wrap entire model
        >>> fsdp_model = wrap_fsdp(model)
        
        >>> # Wrap each Linear layer separately
        >>> def should_wrap(module):
        ...     return isinstance(module, nn.Linear)
        >>> fsdp_model = wrap_fsdp(model, wrap_policy=should_wrap)
    """
    # TODO: Implement module wrapping with policy
    # Steps:
    # 1. If wrap_policy is None, wrap only top-level module
    # 2. If wrap_policy provided:
    #    a. Recursively traverse module tree
    #    b. For each submodule, check if wrap_policy(submodule) returns True
    #    c. If yes, wrap that submodule with FSDPModule
    #    d. Replace submodule in parent with wrapped version
    # 3. Finally wrap top-level module
    # 4. Return wrapped module
    #
    # Hint: Use module.named_children() to iterate through submodules
    # Hint: Use setattr(parent, name, wrapped_child) to replace submodules
    raise NotImplementedError("TODO: Task 7 - Wrap module with policy")


# Convenience function for common case: wrap each layer
def wrap_transformer_layers(
    model: nn.Module,
    layer_cls: type,
    config: Optional[FSDPConfig] = None,
) -> nn.Module:
    """Wrap each Transformer layer in a model with FSDP.
    
    Args:
        model: Model containing Transformer layers
        layer_cls: Class of layer to wrap (e.g., TransformerBlock)
        config: FSDP configuration
    
    Returns:
        Model with each layer wrapped
    
    Example:
        >>> class TransformerBlock(nn.Module):
        ...     ...
        >>> model = nn.Sequential(*[TransformerBlock() for _ in range(12)])
        >>> fsdp_model = wrap_transformer_layers(model, TransformerBlock)
    """
    wrap_policy = lambda m: isinstance(m, layer_cls)
    return wrap_fsdp(model, config=config, wrap_policy=wrap_policy)


# Example usage (for testing):
if __name__ == "__main__":
    print("Test: FSDP Module Integration")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 5)
    )
    
    # Wrap with FSDP
    config = FSDPConfig(
        reshard_after_forward=True,
        enable_prefetch=False  # Disable for simple test
    )
    fsdp_model = FSDPModule(model, config=config)
    
    print(f"FSDP model created: {fsdp_model}")
    print(f"Has flat parameter: {fsdp_model.flat_param is not None}")
    
    # Test forward pass
    input_tensor = torch.randn(3, 10)
    output = fsdp_model(input_tensor)
    print(f"Output shape: {output.shape}")
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    print("Backward pass completed!")
    
    print("\nTest: Wrapping with policy")
    
    # Wrap only Linear layers
    model2 = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 5)
    )
    
    fsdp_model2 = wrap_fsdp(
        model2,
        config=config,
        wrap_policy=lambda m: isinstance(m, nn.Linear)
    )
    
    print(f"Model with selective wrapping: {fsdp_model2}")

