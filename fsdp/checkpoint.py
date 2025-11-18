"""Activation Checkpointing for FSDP.

This module provides utilities for gradient checkpointing to reduce memory usage.
Checkpointing trades compute for memory by recomputing activations during backward
instead of storing them during forward.

Key benefits:
- Reduces activation memory by ~70-80%
- Enables training larger models or batch sizes
- Integrates seamlessly with FSDP

Usage:
    # Option 1: Wrap individual modules
    from fsdp.checkpoint import checkpoint_wrapper
    
    for layer in model.layers:
        layer = checkpoint_wrapper(layer)
        fully_shard(layer)
    
    # Option 2: Use apply_activation_checkpointing helper
    from fsdp.checkpoint import apply_activation_checkpointing
    
    apply_activation_checkpointing(model, check_fn=lambda m: isinstance(m, TransformerBlock))
"""

import torch
import torch.nn as nn
from typing import Any, Callable, Optional
from torch.utils.checkpoint import checkpoint


def checkpoint_wrapper(
    module: nn.Module,
    *,
    preserve_rng_state: bool = True,
    use_reentrant: bool = False,
) -> nn.Module:
    """Wrap a module to use gradient checkpointing.
    
    This wrapper modifies the module's forward method to use torch.utils.checkpoint,
    which saves memory by not storing intermediate activations and recomputing them
    during backward pass.
    
    Args:
        module: Module to wrap
        preserve_rng_state: Whether to preserve RNG state for reproducibility
        use_reentrant: Whether to use reentrant checkpoint (legacy mode).
            False is recommended for better memory efficiency.
    
    Returns:
        The same module with checkpointing enabled
    
    Example:
        >>> layer = TransformerBlock(d_model=512)
        >>> layer = checkpoint_wrapper(layer)
        >>> # Forward pass will now use checkpointing
        >>> output = layer(input)
    
    Memory Savings:
        - Without checkpointing: O(L × B × S × D) activation memory
        - With checkpointing: O(B × S × D) activation memory  
        - Savings: ~(L-1)/L reduction (e.g., 47/48 = 97.9% for 48 layers)
    
    Trade-off:
        - ✅ Memory: -70~80% for large models
        - ❌ Compute: +33% (one forward + one recompute)
        - ⚖️ Overall: Net win for memory-bound training
    """
    # Store original forward method
    original_forward = module.forward
    
    def checkpointed_forward(*args, **kwargs):
        """Forward with gradient checkpointing."""
        # Only use checkpointing during training
        if not module.training:
            return original_forward(*args, **kwargs)
        
        # Create a wrapper function for checkpoint
        # checkpoint requires a function that takes tensors as positional args
        def run_function(*inputs):
            # Reconstruct args from inputs
            # checkpoint passes all tensor inputs as positional args
            return original_forward(*inputs, **kwargs)
        
        # Use checkpoint for forward pass
        # This will not save intermediate activations during forward
        # Instead, it will recompute them during backward
        return checkpoint(
            run_function,
            *args,
            use_reentrant=use_reentrant,
            preserve_rng_state=preserve_rng_state,
        )
    
    # Replace forward method
    module.forward = checkpointed_forward
    
    # Mark module as checkpointed (useful for debugging)
    module._is_checkpointed = True
    
    return module


def apply_activation_checkpointing(
    model: nn.Module,
    *,
    check_fn: Optional[Callable[[nn.Module], bool]] = None,
    preserve_rng_state: bool = True,
    use_reentrant: bool = False,
) -> None:
    """Apply activation checkpointing to all modules matching check_fn.
    
    This is a convenience function that applies checkpoint_wrapper to all
    submodules that match the check_fn predicate.
    
    Args:
        model: Root model to apply checkpointing to
        check_fn: Function that returns True for modules to checkpoint.
            If None, does nothing (you must specify check_fn).
        preserve_rng_state: Whether to preserve RNG state
        use_reentrant: Whether to use reentrant checkpoint
    
    Example:
        >>> # Checkpoint all TransformerBlock modules
        >>> apply_activation_checkpointing(
        ...     model,
        ...     check_fn=lambda m: isinstance(m, TransformerBlock)
        ... )
        
        >>> # Checkpoint layers by name pattern
        >>> apply_activation_checkpointing(
        ...     model,
        ...     check_fn=lambda m: "layer" in m._get_name().lower()
        ... )
    
    Typical Patterns:
        # For Transformer models - checkpoint each layer
        check_fn = lambda m: isinstance(m, (TransformerBlock, TransformerLayer))
        
        # For ResNets - checkpoint each residual block
        check_fn = lambda m: isinstance(m, ResidualBlock)
        
        # For any sequential model - checkpoint every N layers
        check_fn = lambda m: hasattr(m, 'layer_id') and m.layer_id % 2 == 0
    """
    if check_fn is None:
        raise ValueError(
            "check_fn must be specified. Example: "
            "check_fn=lambda m: isinstance(m, TransformerBlock)"
        )
    
    # Track number of modules checkpointed
    num_checkpointed = 0
    
    # Iterate through all submodules
    for name, module in model.named_modules():
        # Skip the root module itself
        if module is model:
            continue
        
        # Check if this module should be checkpointed
        if check_fn(module):
            # Apply checkpointing
            checkpoint_wrapper(
                module,
                preserve_rng_state=preserve_rng_state,
                use_reentrant=use_reentrant,
            )
            num_checkpointed += 1
    
    if num_checkpointed == 0:
        print(f"Warning: No modules matched check_fn. No checkpointing applied.")
    else:
        print(f"Applied activation checkpointing to {num_checkpointed} module(s)")


def is_checkpointed(module: nn.Module) -> bool:
    """Check if a module has checkpointing enabled.
    
    Args:
        module: Module to check
    
    Returns:
        True if module is checkpointed
    
    Example:
        >>> layer = checkpoint_wrapper(TransformerBlock())
        >>> assert is_checkpointed(layer)
    """
    return hasattr(module, '_is_checkpointed') and module._is_checkpointed


# Export public API
__all__ = [
    'checkpoint_wrapper',
    'apply_activation_checkpointing',
    'is_checkpointed',
]

