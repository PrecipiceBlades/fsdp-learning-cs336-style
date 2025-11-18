"""Task 2: FlatParameter Implementation.

This module implements parameter flattening for FSDP. Flattening reduces communication
overhead by combining many small parameters into one large parameter.

Key concepts:
- Flatten: Combine multiple parameters into one contiguous tensor
- Shard: Split flattened parameter across ranks
- Views: Create parameter views back to original shapes
- Synchronization: Keep views in sync with flat storage

Why FlatParameter?
- Reduces communication overhead: 1 all-gather instead of N all-gathers
- Better memory locality: contiguous storage
- Matches PyTorch FSDP's actual implementation
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from fsdp.utils import (
    get_rank,
    get_world_size,
    flatten_params,
    all_gather_tensor,
)


class FlatParameter(nn.Parameter):
    """A parameter that represents a flattened and sharded view of multiple parameters.
    
    This class:
    1. Flattens multiple parameters into one contiguous tensor
    2. Shards the flat parameter across ranks
    3. Maintains views back to original parameter shapes
    4. Handles all-gather and reshard operations
    
    Attributes:
        _param_numels: List of number of elements in each original parameter
        _param_shapes: List of original parameter shapes
        _param_infos: List of (name, numel, shape) tuples
        _local_shard: The local shard stored on this rank
        _full_param: Optional full parameter after all-gather
        _is_sharded: Whether parameter is currently sharded
        rank: Current rank
        world_size: World size
    """
    
    def __new__(cls, params: List[nn.Parameter], rank: Optional[int] = None, world_size: Optional[int] = None):
        """Create new FlatParameter.
        
        Note: We override __new__ to properly initialize the Parameter base class.
        """
        rank = rank if rank is not None else get_rank()
        world_size = world_size if world_size is not None else get_world_size()
        
        # Flatten all parameters
        flat_param_full = flatten_params(params)
        total_numel = flat_param_full.numel()
        
        # Compute uniform shard size (pad if necessary for all_gather compatibility)
        # all_gather requires output_size == world_size * input_size
        shard_size = (total_numel + world_size - 1) // world_size  # Ceiling division
        padded_total_numel = shard_size * world_size
        
        # Pad the full parameter if necessary
        if padded_total_numel > total_numel:
            padding_size = padded_total_numel - total_numel
            flat_param_full = torch.cat([
                flat_param_full,
                torch.zeros(padding_size, dtype=flat_param_full.dtype, device=flat_param_full.device)
            ])
        
        # Compute shard for this rank (now uniform)
        start = rank * shard_size
        end = start + shard_size
        local_shard_data = flat_param_full[start:end].clone().detach()
        
        # Create Parameter with local shard
        instance = super().__new__(cls, local_shard_data, requires_grad=True)
        
        # Store metadata on instance (will be used in __init__)
        instance._init_params = params
        instance._init_rank = rank
        instance._init_world_size = world_size
        instance._init_total_numel = total_numel  # Unpadded size
        instance._init_padded_numel = padded_total_numel  # Padded size for all_gather
        instance._init_shard_offset = start
        instance._init_shard_numel = shard_size  # Uniform shard size
        
        return instance
    
    def __init__(
        self,
        params: List[nn.Parameter],
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        """Initialize FlatParameter from list of parameters.
        
        Args:
            params: List of parameters to flatten
            rank: Current rank (if None, uses get_rank())
            world_size: World size (if None, uses get_world_size())
        """
        # Get metadata from __new__
        self.rank = self._init_rank
        self.world_size = self._init_world_size
        self._shard_offset = self._init_shard_offset
        self._shard_numel = self._init_shard_numel
        self._total_numel = self._init_total_numel  # Unpadded
        self._padded_total_numel = self._init_padded_numel  # Padded
        
        # Store parameter metadata
        self._param_numels: List[int] = [p.numel() for p in params]
        self._param_shapes: List[torch.Size] = [p.shape for p in params]
        self._param_infos: List[Tuple[str, int, torch.Size]] = [
            (f"param_{i}", p.numel(), p.shape) for i, p in enumerate(params)
        ]
        
        # Full parameter storage (only allocated during all-gather)
        self._full_param: Optional[torch.Tensor] = None
        self._is_sharded: bool = True
        
        # Original parameters for reference (we'll replace these with views later)
        self._orig_params = params
        
        # Clean up temporary attributes
        del self._init_params
        del self._init_rank
        del self._init_world_size
        del self._init_total_numel
        del self._init_padded_numel
        del self._init_shard_offset
        del self._init_shard_numel
    
    @property
    def local_shard(self) -> torch.Tensor:
        """Get the local shard of the parameter."""
        if self._is_sharded:
            return self.data
        else:
            # When not sharded, extract the local shard from full parameter
            if self._full_param is not None:
                return self._full_param[self._shard_offset:self._shard_offset + self._shard_numel]
            return self.data
    
    @property
    def full_param(self) -> Optional[torch.Tensor]:
        """Get the full parameter (only available after all-gather)."""
        return self._full_param
    
    def create_views(self) -> List[torch.Tensor]:
        """Create views back to original parameter shapes.
        
        Returns:
            List of tensors (views into flat parameter) with original shapes
        
        Example:
            >>> # Original params: Linear.weight [10, 5], Linear.bias [10]
            >>> flat_param = FlatParameter([weight, bias])
            >>> weight_view, bias_view = flat_param.create_views()
            >>> assert weight_view.shape == (10, 5)
            >>> assert bias_view.shape == (10,)
            >>> # Modifying view modifies flat_param
            >>> weight_view[0, 0] = 999
            >>> assert flat_param.full_param[0] == 999
        """
        views = []
        
        # Use full parameter if available, otherwise use local shard
        source_tensor = self._full_param if self._full_param is not None else self.data
        
        offset = 0
        for shape in self._param_shapes:
            numel = torch.Size(shape).numel()
            
            # Check if this parameter is within our tensor
            if offset + numel <= source_tensor.numel():
                view = source_tensor[offset:offset + numel].view(shape)
                views.append(view)
            else:
                # This parameter is not fully in our shard
                # Return a placeholder or partial view
                # For now, we'll skip it (this happens when we only have a shard)
                break
            
            offset += numel
        
        return views
    
    def all_gather(self) -> torch.Tensor:
        """All-gather parameter shards from all ranks into full parameter.
        
        Returns:
            Full parameter tensor (unpadded, original size)
        
        Example:
            >>> flat_param = FlatParameter([weight, bias], rank=0, world_size=4)
            >>> full_param = flat_param.all_gather()
            >>> assert full_param.shape[0] == total_numel (unpadded)
        """
        # If already gathered, return cached result
        if not self._is_sharded and self._full_param is not None:
            return self._full_param
        
        # If world_size=1, no need to gather
        # Just use data directly (no clone to avoid creating separate tensors)
        if self.world_size == 1:
            self._full_param = self.data  # Use data directly, don't clone!
            self._is_sharded = False
            return self._full_param
        
        # Allocate storage for padded full parameter
        padded_full_param = torch.empty(
            self._padded_total_numel,
            dtype=self.data.dtype,
            device=self.data.device
        )
        
        # All-gather from all ranks (using uniform shard sizes)
        all_gather_tensor(
            output_tensor=padded_full_param,
            input_tensor=self.data,
            async_op=False
        )
        
        # Slice to get unpadded version
        self._full_param = padded_full_param[:self._total_numel]
        
        self._is_sharded = False
        return self._full_param
    
    def reshard(self) -> None:
        """Reshard parameter back to local shard only.
        
        This frees the full parameter storage, keeping only the local shard.
        
        Important: We DON'T copy from full_param back to self.data, because:
        1. self.data already contains the local shard
        2. If the full_param was modified (e.g., by optimizer), we need to preserve those changes
        3. The local shard in self.data is what the optimizer updates directly
        """
        # If already sharded, nothing to do
        if self._is_sharded:
            return
        
        # DON'T copy from full_param back to self.data!
        # The optimizer updates self.data directly, so copying would overwrite optimizer updates.
        # self.data already contains the correct local shard.
        
        # Free full parameter storage
        self._full_param = None
        self._is_sharded = True
    
    def use_full_param(self) -> None:
        """Switch parameter views to point to full parameter.
        
        Call this after all-gather to make parameter views point to full param.
        """
        if self._full_param is None:
            raise RuntimeError("Cannot use full parameter - not gathered yet")
        
        # Create views from full parameter and update original parameters
        views = self.create_views()
        for orig_param, view in zip(self._orig_params, views):
            orig_param.data = view
    
    def use_sharded_param(self) -> None:
        """Switch parameter views to point to local shard.
        
        Call this after resharding to make parameter views point to shard only.
        """
        if not self._is_sharded:
            raise RuntimeError("Cannot use sharded parameter - not sharded yet")
        
        # When sharded, we can't provide full views to original parameters
        # This is expected behavior - parameters should not be accessed when sharded
        # In practice, FSDP will all-gather before any forward/backward pass
    
    def __repr__(self) -> str:
        """String representation of FlatParameter."""
        status = "sharded" if self._is_sharded else "full"
        return (
            f"FlatParameter(rank={self.rank}, world_size={self.world_size}, "
            f"status={status}, numel={self.numel()}, "
            f"n_params={len(self._param_shapes)})"
        )


def flatten_module_params(
    module: nn.Module,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> FlatParameter:
    """Flatten all parameters in a module into a FlatParameter.
    
    Args:
        module: Module whose parameters to flatten
        rank: Current rank
        world_size: World size
    
    Returns:
        FlatParameter containing all module parameters
    
    Example:
        >>> module = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 5))
        >>> flat_param = flatten_module_params(module, rank=0, world_size=2)
        >>> # Rank 0 now has ~half of the parameters
    """
    params = list(module.parameters(recurse=True))  # Recurse to get all nested parameters
    if not params:
        raise ValueError("Module has no parameters to flatten")
    
    return FlatParameter(params, rank=rank, world_size=world_size)


# Example usage (for testing):
if __name__ == "__main__":
    print("Test: FlatParameter creation and operations")
    
    # Create some test parameters
    weight = nn.Parameter(torch.randn(10, 5))
    bias = nn.Parameter(torch.randn(10))
    
    # Create FlatParameter
    flat_param = FlatParameter([weight, bias], rank=0, world_size=2)
    
    print(f"FlatParameter: {flat_param}")
    print(f"Local shard size: {flat_param.local_shard.numel()}")
    
    # Test all-gather
    full_param = flat_param.all_gather()
    print(f"Full param size after all-gather: {full_param.numel()}")
    
    # Test views
    views = flat_param.create_views()
    print(f"Created {len(views)} views")
    
    # Test reshard
    flat_param.reshard()
    print(f"After reshard: {flat_param}")

