"""Task 5: Sharded Optimizer (ZeRO-1 Style).

This module implements an optimizer that shards optimizer states across ranks,
achieving significant memory savings for optimizers like Adam.

Key concepts:
- Optimizer state sharding: Each rank only stores states for its local parameter shard
- Memory savings: For Adam (momentum + variance), 2N → 2N/W per rank
- Parameter updates: Each rank updates only its owned parameters
- Broadcast after step: Ensure all ranks have updated parameters

Why shard optimizer states?
- Adam optimizer stores: momentum (1st moment) + variance (2nd moment) = 2× parameters
- Total memory per rank: params (N) + grads (N) + optimizer states (2N) = 4N
- With FSDP: params (N/W) + grads (N/W) + optimizer states (2N/W) = 4N/W
- Memory savings: 4N → 4N/W, enabling W× larger models
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Iterable
from fsdp.flat_param import FlatParameter
from fsdp.utils import broadcast_tensor, get_rank, get_world_size, is_distributed


class FSDPOptimizer(torch.optim.Optimizer):
    """Optimizer with sharded optimizer states.
    
    This optimizer:
    1. Only stores optimizer states for local parameter shards
    2. Updates only local shards during step()
    3. Broadcasts updated shards to all ranks after step()
    
    Attributes:
        base_optimizer_cls: Base optimizer class (e.g., torch.optim.AdamW)
        rank: Current rank
        world_size: World size
        local_optimizer: Optimizer instance for local parameters only
    """
    
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        optimizer_cls: type = torch.optim.AdamW,
        **optimizer_kwargs
    ):
        """Initialize sharded optimizer.
        
        Args:
            params: Parameters to optimize (should be FlatParameters)
            optimizer_cls: Base optimizer class to use
            **optimizer_kwargs: Keyword arguments for base optimizer (lr, betas, etc.)
        
        Example:
            >>> model = nn.Linear(100, 100)
            >>> flat_param = flatten_module_params(model)
            >>> optimizer = FSDPOptimizer(
            ...     [flat_param],
            ...     optimizer_cls=torch.optim.AdamW,
            ...     lr=1e-3,
            ...     betas=(0.9, 0.999),
            ...     weight_decay=0.01
            ... )
        """
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.base_optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        
        # Store all parameters (including non-local ones for broadcast)
        self.all_params: List[nn.Parameter] = list(params)
        
        # Mapping from parameter to owner rank
        self.param_to_rank: Dict[int, int] = {}
        
        # Collect local parameters to optimize
        local_params = []
        for param in self.all_params:
            if isinstance(param, FlatParameter):
                # For FlatParameter, we only optimize the local shard
                # Note: the local shard is stored in param.data (which is an nn.Parameter)
                local_params.append(param)  # param.data is the local shard
                # Store the rank that owns this shard
                self.param_to_rank[id(param)] = param.rank
            else:
                # For regular parameters, optimize as normal
                local_params.append(param)
                self.param_to_rank[id(param)] = self.rank
        
        # Create local optimizer with local parameters only
        self.local_optimizer = optimizer_cls(local_params, **optimizer_kwargs)
    
    def step(self, closure=None):
        """Perform optimization step and broadcast updated parameters.
        
        Args:
            closure: Optional closure that reevaluates the model and returns loss
        
        Returns:
            Loss if closure is provided
        
        Process:
        1. Local optimizer updates local parameter shards
        2. Broadcast updated shards from owner rank to all other ranks
        3. Now all ranks have the updated parameters
        """
        # Step 1: Update local parameters
        loss = None
        if closure is not None:
            loss = self.local_optimizer.step(closure)
        else:
            self.local_optimizer.step()
        
        # Step 2: CRITICAL - Zero out padding in FlatParameters after optimizer update
        # This ensures padding doesn't accumulate non-zero values over time
        from fsdp.flat_param import FlatParameter
        for param in self.all_params:
            if isinstance(param, FlatParameter):
                # Check if this rank's shard contains padding
                shard_start = param._shard_offset
                shard_end = shard_start + param._shard_numel
                
                if shard_end > param._total_numel:
                    # This shard extends into padding region
                    valid_size = param._total_numel - shard_start
                    if valid_size < param._shard_numel and valid_size >= 0:
                        # Zero out the padding portion of parameters
                        with torch.no_grad():
                            param.data[valid_size:] = 0.0
        
        # NOTE: For FSDP, we DON'T need to broadcast after step!
        # Each rank updates its own local shard, and the next all-gather
        # will automatically get the updated shards from all ranks.
        
        return loss
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients.
        
        Args:
            set_to_none: If True, set gradients to None instead of zero
        """
        # Zero gradients for flat parameters
        self.local_optimizer.zero_grad(set_to_none=set_to_none)
        
        # Also zero gradients for original parameters (views)
        # This is critical because gradients accumulate in orig_param.grad
        for param in self.all_params:
            from fsdp.flat_param import FlatParameter
            if isinstance(param, FlatParameter):
                # Zero gradients for original parameters
                for orig_param in param._orig_params:
                    if set_to_none:
                        orig_param.grad = None
                    elif orig_param.grad is not None:
                        orig_param.grad.zero_()
    
    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer state dict.
        
        Note: This returns only the local optimizer's state.
        For full checkpoint, would need to gather states from all ranks.
        """
        return self.local_optimizer.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state dict.
        
        Args:
            state_dict: State dict to load
        """
        self.local_optimizer.load_state_dict(state_dict)
    
    @property
    def param_groups(self):
        """Get parameter groups from local optimizer."""
        if not hasattr(self, 'local_optimizer') or self.local_optimizer is None:
            return []
        return self.local_optimizer.param_groups


# Helper function for easy creation
def create_fsdp_optimizer(
    params: Iterable[nn.Parameter],
    optimizer_cls: type = torch.optim.AdamW,
    **optimizer_kwargs
) -> FSDPOptimizer:
    """Create FSDP optimizer with standard interface.
    
    Args:
        params: Parameters to optimize
        optimizer_cls: Optimizer class to use
        **optimizer_kwargs: Optimizer arguments (lr, betas, weight_decay, etc.)
    
    Returns:
        FSDPOptimizer instance
    
    Example:
        >>> optimizer = create_fsdp_optimizer(
        ...     model.parameters(),
        ...     optimizer_cls=torch.optim.AdamW,
        ...     lr=1e-3,
        ...     weight_decay=0.01
        ... )
    """
    return FSDPOptimizer(params, optimizer_cls, **optimizer_kwargs)


# Example usage (for testing):
if __name__ == "__main__":
    print("Test: Sharded optimizer")
    
    # Create simple model
    model = nn.Linear(100, 50)
    
    # Create FlatParameter
    from fsdp.flat_param import flatten_module_params
    flat_param = flatten_module_params(model, rank=0, world_size=4)
    
    # Create sharded optimizer
    optimizer = FSDPOptimizer(
        [flat_param],
        optimizer_cls=torch.optim.AdamW,
        lr=1e-3,
        weight_decay=0.01
    )
    
    print(f"Optimizer created for rank {optimizer.rank}")
    print(f"Local optimizer has {len(list(optimizer.local_optimizer.param_groups[0]['params']))} parameters")
    
    # Test forward, backward, step
    input_tensor = torch.randn(10, 100)
    output = model(input_tensor)
    loss = output.sum()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Optimizer step completed successfully!")

