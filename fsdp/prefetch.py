"""Task 6: Prefetching & Communication-Computation Overlap.

This module implements prefetching to overlap communication with computation,
which is essential for FSDP performance.

Key concepts:
- Prefetching: Start all-gather for next module while computing current module
- Async operations: Use async_op=True in distributed collectives
- Double-buffering: Need space for current + next module's parameters
- Synchronization: Wait for async operation before using prefetched data

Why is prefetching critical?
- Without prefetch: Compute → Wait for communication → Compute → Wait...
- With prefetch: Compute (while next layer communicates) → Compute → ...
- Can hide communication latency behind computation
- Speedup can be 2-3× for communication-bound workloads
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict
from fsdp.flat_param import FlatParameter
from fsdp.utils import all_gather_tensor
import torch.distributed as dist


class PrefetchManager:
    """Manages prefetching of parameters for communication-computation overlap.
    
    Attributes:
        modules: List of FSDP modules in execution order
        flat_params: List of FlatParameters corresponding to modules
        enable_prefetch: Whether prefetching is enabled
        current_handle: Current async operation handle
        prefetch_buffer: Buffer for prefetched parameters
    """
    
    def __init__(
        self,
        modules: List[nn.Module],
        flat_params: List[FlatParameter],
        enable_prefetch: bool = True
    ):
        """Initialize prefetch manager.
        
        Args:
            modules: List of modules to manage
            flat_params: List of FlatParameters for each module
            enable_prefetch: Whether to enable prefetching
        """
        # TODO: Initialize prefetch manager
        # Steps:
        # 1. Store modules and flat_params
        # 2. Create mapping from module to flat_param
        # 3. Initialize async operation handles dict
        # 4. Initialize prefetch buffers if needed
        
        self.modules = modules
        self.flat_params = flat_params
        self.enable_prefetch = enable_prefetch
        
        # Mapping from module to its FlatParameter
        self.module_to_flat_param: Dict[nn.Module, FlatParameter] = {}
        
        # Async operation handles for prefetching
        self.async_handles: Dict[nn.Module, Optional[dist.Work]] = {}
        
        raise NotImplementedError("TODO: Task 6 - Initialize prefetch manager")
    
    def prefetch_next(self, current_idx: int) -> Optional[dist.Work]:
        """Prefetch parameters for next module.
        
        Args:
            current_idx: Index of current module being computed
        
        Returns:
            Work handle for async operation, or None
        
        Example:
            >>> manager = PrefetchManager(modules, flat_params)
            >>> # Before computing module i, prefetch module i+1
            >>> handle = manager.prefetch_next(i)
            >>> output = modules[i](input)
            >>> # By the time we need module i+1, prefetch should be done
        """
        # TODO: Implement prefetching
        # Steps:
        # 1. Check if prefetching is enabled
        # 2. Check if there's a next module (current_idx + 1 < len(modules))
        # 3. Get next module's FlatParameter
        # 4. Start async all-gather for next module's parameters
        # 5. Store async handle for later synchronization
        # 6. Return handle
        #
        # Hint: Use all_gather_tensor() with async_op=True
        # Hint: flat_param.all_gather() can be modified to support async
        raise NotImplementedError("TODO: Task 6 - Prefetch next module")
    
    def wait_for_prefetch(self, module: nn.Module) -> None:
        """Wait for prefetch operation to complete for a module.
        
        Args:
            module: Module whose prefetch to wait for
        
        Example:
            >>> manager.prefetch_next(i)  # Start prefetch for module i+1
            >>> # ... do some work ...
            >>> manager.wait_for_prefetch(modules[i+1])  # Wait before using module i+1
            >>> output = modules[i+1](input)  # Now safe to use
        """
        # TODO: Implement wait for prefetch
        # Steps:
        # 1. Check if there's an async handle for this module
        # 2. If yes, call handle.wait() to synchronize
        # 3. Clear the handle after waiting
        # 4. Update module's flat_param state (mark as gathered)
        #
        # Hint: handle.wait() blocks until async operation completes
        raise NotImplementedError("TODO: Task 6 - Wait for prefetch")


def forward_with_prefetch(
    modules: List[nn.Module],
    flat_params: List[FlatParameter],
    input_tensor: torch.Tensor,
    enable_prefetch: bool = True
) -> torch.Tensor:
    """Forward pass through modules with prefetching.
    
    Args:
        modules: List of modules to execute
        flat_params: List of FlatParameters for each module
        input_tensor: Input to first module
        enable_prefetch: Whether to enable prefetching
    
    Returns:
        Output tensor from last module
    
    Example:
        >>> modules = [layer1, layer2, layer3]
        >>> flat_params = [fp1, fp2, fp3]
        >>> output = forward_with_prefetch(modules, flat_params, input, enable_prefetch=True)
    
    Process:
    - For module i:
      1. Wait for prefetch of module i (if prefetched)
      2. Start prefetch of module i+1
      3. Compute module i
      4. Reshard module i (optional)
    """
    # TODO: Implement forward with prefetch
    # Steps:
    # 1. Create PrefetchManager
    # 2. For each module:
    #    a. If prefetch enabled, wait for this module's prefetch
    #    b. If not first module, all-gather parameters (or rely on prefetch)
    #    c. Start prefetch for next module
    #    d. Compute forward pass for current module
    #    e. Optionally reshard current module
    # 3. Return final output
    #
    # Hint: This orchestrates the prefetching across all modules
    # Hint: Need to handle first module (no prefetch to wait for)
    # Hint: Need to handle last module (no next module to prefetch)
    raise NotImplementedError("TODO: Task 6 - Forward with prefetch")


def backward_with_prefetch(
    modules: List[nn.Module],
    flat_params: List[FlatParameter],
    grad_output: torch.Tensor,
    enable_prefetch: bool = True
) -> torch.Tensor:
    """Backward pass through modules with prefetching.
    
    Backward pass goes in reverse order, so we prefetch in reverse.
    
    Args:
        modules: List of modules (in forward order)
        flat_params: List of FlatParameters (in forward order)
        grad_output: Gradient from loss
        enable_prefetch: Whether to enable prefetching
    
    Returns:
        Gradient w.r.t. input
    
    Process:
    - For module i (in reverse order):
      1. Wait for prefetch of module i
      2. Start prefetch of module i-1
      3. Compute backward pass for module i
      4. Reduce-scatter gradients for module i
      5. Reshard parameters for module i
    """
    # TODO: Implement backward with prefetch
    # Steps:
    # 1. Create PrefetchManager
    # 2. Reverse iteration through modules (backward order)
    # 3. For each module:
    #    a. Wait for prefetch (if enabled)
    #    b. All-gather parameters (if needed for backward)
    #    c. Start prefetch for previous module
    #    d. Compute backward pass
    #    e. Reduce-scatter gradients
    #    f. Reshard parameters
    # 4. Return gradient w.r.t. input
    #
    # Hint: Similar to forward_with_prefetch but in reverse
    # Hint: Use reversed(list(enumerate(modules))) for backward iteration
    raise NotImplementedError("TODO: Task 6 - Backward with prefetch")


# Example usage (for testing):
if __name__ == "__main__":
    print("Test: Prefetching")
    
    # Create simple sequential model
    modules = [
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 5)
    ]
    
    # Create FlatParameters for each module (only for Linear layers)
    from fsdp.flat_param import flatten_module_params
    flat_params = []
    for module in modules:
        if isinstance(module, nn.Linear):
            flat_param = flatten_module_params(module, rank=0, world_size=2)
            flat_params.append(flat_param)
        else:
            flat_params.append(None)  # No params for ReLU
    
    # Create prefetch manager
    linear_modules = [m for m in modules if isinstance(m, nn.Linear)]
    linear_flat_params = [fp for fp in flat_params if fp is not None]
    
    manager = PrefetchManager(linear_modules, linear_flat_params, enable_prefetch=True)
    
    # Test prefetching
    print("Testing prefetch...")
    handle = manager.prefetch_next(0)
    if handle:
        print("Prefetch started for next module")
        manager.wait_for_prefetch(linear_modules[1])
        print("Prefetch completed!")

