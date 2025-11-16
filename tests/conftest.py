"""pytest configuration and fixtures for FSDP tests."""

import os
import pytest
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from fsdp.utils import setup_distributed, cleanup_distributed


def pytest_configure(config):
    """Configure pytest."""
    # Set environment variables for distributed testing
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")


@pytest.fixture
def device():
    """Get device for testing (CPU or CUDA)."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture
def rank():
    """Get current rank (0 if not distributed)."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


@pytest.fixture
def world_size():
    """Get world size (1 if not distributed)."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def spawn_for_test(test_func, world_size=2, backend="gloo"):
    """Spawn multiple processes for distributed testing.
    
    Args:
        test_func: Test function with signature (rank, world_size, *args)
        world_size: Number of processes to spawn
        backend: Backend to use ('nccl' for GPU, 'gloo' for CPU)
    
    Example:
        >>> def test_distributed(rank, world_size):
        ...     setup_distributed(rank, world_size)
        ...     # ... test code ...
        ...     cleanup_distributed()
        >>> 
        >>> spawn_for_test(test_distributed, world_size=2)
    """
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    mp.spawn(
        test_func,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


# Test utilities
def assert_tensors_equal(t1: torch.Tensor, t2: torch.Tensor, atol=1e-5, rtol=1e-4):
    """Assert two tensors are equal within tolerance."""
    assert t1.shape == t2.shape, f"Shape mismatch: {t1.shape} vs {t2.shape}"
    assert torch.allclose(t1, t2, atol=atol, rtol=rtol), \
        f"Tensors not equal. Max diff: {(t1 - t2).abs().max()}"


def assert_gradients_equal(model1, model2, atol=1e-5, rtol=1e-4):
    """Assert gradients of two models are equal."""
    params1 = list(model1.parameters())
    params2 = list(model2.parameters())
    
    assert len(params1) == len(params2), "Different number of parameters"
    
    for p1, p2 in zip(params1, params2):
        if p1.grad is None and p2.grad is None:
            continue
        assert p1.grad is not None and p2.grad is not None, "Gradient mismatch (None vs not None)"
        assert_tensors_equal(p1.grad, p2.grad, atol=atol, rtol=rtol)

