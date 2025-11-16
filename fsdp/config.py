"""FSDP configuration class."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FSDPConfig:
    """Configuration for FSDP module.
    
    Attributes:
        reshard_after_forward: If True, reshard parameters immediately after forward pass
            to save memory. If False, keep full parameters until backward pass.
            Trade-off: True saves memory but requires extra all-gather in backward.
        
        enable_prefetch: If True, prefetch next module's parameters during current
            module's computation to overlap communication with computation.
        
        limit_all_gathers: Optional limit on number of modules that can have full
            parameters at once. Helps control peak memory usage.
        
        optimizer_state_sharding: If True, shard optimizer states across ranks.
            This is the key to ZeRO-1/2/3 memory savings.
        
        mixed_precision: If True, use mixed precision training (FP16/BF16).
            Note: This is intentionally simplified - production FSDP has complex
            mixed precision handling.
    """
    
    reshard_after_forward: bool = True
    enable_prefetch: bool = True
    limit_all_gathers: Optional[int] = None
    optimizer_state_sharding: bool = True
    mixed_precision: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if self.limit_all_gathers is not None and self.limit_all_gathers < 1:
            raise ValueError("limit_all_gathers must be >= 1")

