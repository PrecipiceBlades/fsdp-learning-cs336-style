"""FSDP (Fully Sharded Data Parallel) implementation.

This package provides a PyTorch FSDP implementation following the FSDP2 API design.

Quick start:
    from fsdp import fully_shard, get_flat_parameters
    from fsdp.optimizer import FSDPOptimizer
    
    # Standard usage
    model = Transformer()
    for layer in model.layers:
        fully_shard(layer)
    fully_shard(model)
    
    # Meta device usage (memory efficient for large models)
    with torch.device("meta"):
        model = Transformer()
    for layer in model.layers:
        fully_shard(layer)  # Materializes only this layer's shard
    fully_shard(model)
    
    flat_params = get_flat_parameters(model)
    optimizer = FSDPOptimizer(flat_params, optimizer_cls=torch.optim.Adam, lr=1e-3)

Components:
    - fully_shard: Apply FSDP to a module (FSDP2-style API, supports meta device)
    - FlatParameter: Flattened and sharded parameter
    - FSDPOptimizer: Optimizer with sharded states
    - init_model_on_meta: Initialize model on meta device
    - materialize_meta_module: Materialize meta tensors to real device
"""

from fsdp.api import (
    fully_shard,
    get_flat_parameters,
    is_fsdp_managed,
    clear_fsdp_registry,
)
from fsdp.flat_param import FlatParameter, flatten_module_params
from fsdp.optimizer import FSDPOptimizer, create_fsdp_optimizer
from fsdp.meta_init import (
    init_model_on_meta,
    materialize_meta_tensor,
    materialize_meta_module,
)

__all__ = [
    # FSDP2-style API
    'fully_shard',
    'get_flat_parameters',
    'is_fsdp_managed',
    'clear_fsdp_registry',
    # Core classes
    'FlatParameter',
    'flatten_module_params',
    'FSDPOptimizer',
    'create_fsdp_optimizer',
    # Meta device utilities
    'init_model_on_meta',
    'materialize_meta_tensor',
    'materialize_meta_module',
]

__version__ = '0.1.0'
