"""Task 3: Forward Pass with All-Gather and Optional Reshard.

This module implements the forward pass logic for FSDP:
1. All-gather full parameters before forward computation
2. Run forward pass
3. Optional: Reshard parameters immediately after forward to save memory

Key concepts:
- All-gather: Gather parameter shards from all ranks into full parameter
- Forward hooks: register_forward_pre_hook and register_forward_hook
- Reshard after forward: Trade-off between memory (lower with reshard) and
  communication (lower without reshard, since no re-gather needed in backward)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Any
from fsdp.flat_param import FlatParameter
from fsdp.utils import all_gather_tensor


def all_gather_params(
    flat_param: FlatParameter,
    async_op: bool = False
) -> Optional[torch.distributed.Work]:
    """All-gather parameter from all ranks.
    
    Args:
        flat_param: FlatParameter to all-gather
        async_op: If True, return work handle for async operation
    
    Returns:
        Work handle if async_op=True, else None
    
    Example:
        >>> flat_param = FlatParameter([weight, bias], rank=0, world_size=2)
        >>> all_gather_params(flat_param)
        >>> # Now flat_param.full_param contains the complete parameter
    """
    # For now, we don't support async operations
    if async_op:
        raise NotImplementedError("Async all-gather not yet supported")
    
    # Call flat_param's all_gather method
    flat_param.all_gather()
    return None


def reshard_params(flat_param: FlatParameter) -> None:
    """Reshard parameters back to local shard only.
    
    This frees memory by discarding the full parameter, keeping only local shard.
    
    Args:
        flat_param: FlatParameter to reshard
    
    Example:
        >>> all_gather_params(flat_param)
        >>> # ... use full parameter ...
        >>> reshard_params(flat_param)
        >>> # Now only local shard remains in memory
    """
    flat_param.reshard()


def create_forward_pre_hook(
    flat_param: FlatParameter,
    reshard_after_forward: bool = True
):
    """Create a forward pre-hook that all-gathers parameters before forward.
    
    Args:
        flat_param: FlatParameter to manage
        reshard_after_forward: Whether to reshard after forward pass
    
    Returns:
        Hook function to be registered with module.register_forward_pre_hook()
    
    Example:
        >>> module = nn.Linear(10, 10)
        >>> flat_param = flatten_module_params(module)
        >>> hook = create_forward_pre_hook(flat_param, reshard_after_forward=True)
        >>> handle = module.register_forward_pre_hook(hook)
    """
    def forward_pre_hook(module: nn.Module, inputs: Tuple[Any, ...]) -> None:
        """Hook called before forward pass.
        
        This hook all-gathers the full parameter before forward computation.
        """
        # All-gather parameters
        all_gather_params(flat_param)
        
        # Update parameter views to point to full parameter
        flat_param.use_full_param()
    
    return forward_pre_hook


def create_forward_post_hook(
    flat_param: FlatParameter,
    reshard_after_forward: bool = True
):
    """Create a forward post-hook that optionally reshards after forward.
    
    Args:
        flat_param: FlatParameter to manage
        reshard_after_forward: If True, reshard parameters after forward
    
    Returns:
        Hook function to be registered with module.register_forward_hook()
    
    Example:
        >>> module = nn.Linear(10, 10)
        >>> flat_param = flatten_module_params(module)
        >>> hook = create_forward_post_hook(flat_param, reshard_after_forward=True)
        >>> handle = module.register_forward_hook(hook)
    """
    def forward_post_hook(
        module: nn.Module,
        inputs: Tuple[Any, ...],
        outputs: Any
    ) -> None:
        """Hook called after forward pass.
        
        This hook optionally reshards parameters to save memory.
        """
        if reshard_after_forward:
            # Reshard parameters
            reshard_params(flat_param)
            # Note: We don't call use_sharded_param() here because parameters
            # should not be accessed between forward and backward
    
    return forward_post_hook


def register_forward_hooks(
    module: nn.Module,
    flat_param: FlatParameter,
    reshard_after_forward: bool = True
) -> Tuple[torch.utils.hooks.RemovableHandle, torch.utils.hooks.RemovableHandle]:
    """Register forward hooks on a module.
    
    Args:
        module: Module to register hooks on
        flat_param: FlatParameter for this module
        reshard_after_forward: Whether to reshard after forward
    
    Returns:
        Tuple of (pre_hook_handle, post_hook_handle) for removal if needed
    
    Example:
        >>> module = nn.Linear(10, 10)
        >>> flat_param = flatten_module_params(module)
        >>> pre_handle, post_handle = register_forward_hooks(
        ...     module, flat_param, reshard_after_forward=True
        ... )
        >>> # Forward pass will now automatically all-gather and reshard
        >>> output = module(input)
    """
    # Create hooks
    pre_hook = create_forward_pre_hook(flat_param, reshard_after_forward)
    post_hook = create_forward_post_hook(flat_param, reshard_after_forward)
    
    # Register hooks
    pre_handle = module.register_forward_pre_hook(pre_hook)
    post_handle = module.register_forward_hook(post_hook)
    
    return pre_handle, post_handle


# Example usage (for testing):
if __name__ == "__main__":
    print("Test: Forward pass with all-gather and reshard")
    
    # Create a simple module
    module = nn.Linear(10, 5)
    
    # Create FlatParameter
    from fsdp.flat_param import flatten_module_params
    flat_param = flatten_module_params(module, rank=0, world_size=2)
    
    # Register forward hooks
    pre_handle, post_handle = register_forward_hooks(
        module, flat_param, reshard_after_forward=True
    )
    
    # Test forward pass
    input_tensor = torch.randn(3, 10)
    output = module(input_tensor)
    
    print(f"Output shape: {output.shape}")
    print(f"After forward, flat_param is sharded: {flat_param._is_sharded}")

