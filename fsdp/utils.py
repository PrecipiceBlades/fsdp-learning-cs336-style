"""Utility functions for FSDP implementation."""

import torch
import torch.distributed as dist
from typing import Optional, List, Tuple


def get_rank() -> int:
    """Get current process rank. Returns 0 if not distributed."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get world size (total number of processes). Returns 1 if not distributed."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_initialized() and dist.get_world_size() > 1


def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """Initialize distributed process group.
    
    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        backend: Backend to use ('nccl' for GPU, 'gloo' for CPU)
    """
    import os
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        
        # Set device if using CUDA
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def get_device(rank: Optional[int] = None) -> torch.device:
    """Get device for given rank.
    
    Args:
        rank: Process rank. If None, uses current rank.
    
    Returns:
        torch.device: CUDA device if available, else CPU
    """
    if rank is None:
        rank = get_rank()
    
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def broadcast_tensor(tensor: torch.Tensor, src: int = 0, group=None) -> None:
    """Broadcast tensor from src rank to all other ranks.
    
    Args:
        tensor: Tensor to broadcast (modified in-place on all ranks)
        src: Source rank that holds the data
        group: Process group (None = default group)
    """
    if is_distributed():
        dist.broadcast(tensor, src=src, group=group)


def all_gather_tensor(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group=None,
    async_op: bool = False
) -> Optional[dist.Work]:
    """All-gather tensor shards into full tensor.
    
    Args:
        output_tensor: Output tensor of shape [world_size * shard_size, ...]
        input_tensor: Input tensor (local shard) of shape [shard_size, ...]
        group: Process group
        async_op: If True, return Work handle for async operation
    
    Returns:
        Work handle if async_op=True, else None
    """
    if not is_distributed():
        output_tensor.copy_(input_tensor)
        return None
    
    return dist.all_gather_into_tensor(
        output_tensor=output_tensor,
        input_tensor=input_tensor,
        group=group,
        async_op=async_op
    )


def reduce_scatter_tensor(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    op=dist.ReduceOp.SUM,
    group=None,
    async_op: bool = False
) -> Optional[dist.Work]:
    """Reduce-scatter tensor: reduce and scatter across ranks.
    
    Args:
        output_tensor: Output tensor (local shard) of shape [shard_size, ...]
        input_tensor: Input tensor (full) of shape [world_size * shard_size, ...]
        op: Reduce operation (usually SUM for gradients)
        group: Process group
        async_op: If True, return Work handle for async operation
    
    Returns:
        Work handle if async_op=True, else None
    """
    if not is_distributed():
        output_tensor.copy_(input_tensor)
        return None
    
    return dist.reduce_scatter_tensor(
        output=output_tensor,
        input=input_tensor,
        op=op,
        group=group,
        async_op=async_op
    )


def compute_shard_range(
    total_numel: int,
    rank: int,
    world_size: int
) -> Tuple[int, int]:
    """Compute the [start, end) range for a rank's shard.
    
    Args:
        total_numel: Total number of elements
        rank: Current rank
        world_size: Total number of ranks
    
    Returns:
        (start, end): Start and end indices for this rank's shard
    """
    shard_size = total_numel // world_size
    remainder = total_numel % world_size
    
    # Distribute remainder elements to first ranks
    if rank < remainder:
        start = rank * (shard_size + 1)
        end = start + shard_size + 1
    else:
        start = rank * shard_size + remainder
        end = start + shard_size
    
    return start, end


def compute_shard_size(total_numel: int, rank: int, world_size: int) -> int:
    """Compute shard size for a given rank.
    
    Args:
        total_numel: Total number of elements
        rank: Current rank
        world_size: Total number of ranks
    
    Returns:
        Shard size for this rank
    """
    start, end = compute_shard_range(total_numel, rank, world_size)
    return end - start


def flatten_params(params: List[torch.nn.Parameter]) -> torch.Tensor:
    """Flatten list of parameters into single contiguous tensor.
    
    Args:
        params: List of parameters to flatten
    
    Returns:
        Flattened tensor containing all parameters
    """
    if not params:
        return torch.tensor([])
    
    flat_param = torch.cat([p.detach().flatten() for p in params])
    return flat_param


def unflatten_params(
    flat_tensor: torch.Tensor,
    param_shapes: List[torch.Size]
) -> List[torch.Tensor]:
    """Unflatten tensor back into list of tensors with original shapes.
    
    Args:
        flat_tensor: Flattened tensor
        param_shapes: List of original parameter shapes
    
    Returns:
        List of tensors with original shapes (views into flat_tensor)
    """
    tensors = []
    offset = 0
    
    for shape in param_shapes:
        numel = torch.Size(shape).numel()
        tensor = flat_tensor[offset:offset + numel].view(shape)
        tensors.append(tensor)
        offset += numel
    
    return tensors


def free_storage(tensor: torch.Tensor) -> None:
    """Free storage of a tensor to save memory.
    
    Args:
        tensor: Tensor whose storage to free
    """
    if tensor.storage().size() > 0:
        tensor.set_(tensor.storage(), 0, (0,), (1,))

