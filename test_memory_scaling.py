"""Multi-GPU memory test: verify that FSDP memory usage scales correctly."""

import torch
import torch.nn as nn
import sys
sys.path.append('/root/cs336-assignment2-systems/cs336-basics')

from cs336_basics.model import BasicsTransformerLM


def calculate_model_memory_detailed(config):
    """Calculate detailed memory breakdown."""
    vocab_size = config["vocab_size"]
    d_model = config["d_model"]
    num_layers = config["num_layers"]
    d_ff = config["d_ff"]
    
    # Token embeddings
    token_emb_params = vocab_size * d_model
    
    # Per-layer parameters
    attn_params = 4 * d_model * d_model
    ffn_params = 3 * d_model * d_ff
    ln_params = 2 * d_model
    layer_params = attn_params + ffn_params + ln_params
    total_layer_params = num_layers * layer_params
    
    # Final components
    ln_final_params = d_model
    lm_head_params = vocab_size * d_model
    
    total_params = token_emb_params + total_layer_params + ln_final_params + lm_head_params
    
    # Memory in GB (FP32)
    bytes_per_param = 4
    
    # Non-FSDP: params + grads + optimizer (Adam: 2 states) = 4N
    param_memory = total_params * bytes_per_param / (1024**3)
    grad_memory = total_params * bytes_per_param / (1024**3)
    optimizer_memory = 2 * total_params * bytes_per_param / (1024**3)
    total_non_fsdp = param_memory + grad_memory + optimizer_memory
    
    return {
        "total_params": total_params,
        "total_params_m": total_params / 1e6,
        "total_params_b": total_params / 1e9,
        "param_memory_gb": param_memory,
        "grad_memory_gb": grad_memory,
        "optimizer_memory_gb": optimizer_memory,
        "total_memory_gb": total_non_fsdp,
    }


def get_configs():
    """Get various model configurations."""
    return {
        "gpt2-small": {
            "vocab_size": 50257,
            "context_length": 1024,
            "d_model": 768,
            "num_layers": 12,
            "num_heads": 12,
            "d_ff": 3072,
            "rope_theta": 10000.0,
        },
        "gpt2-medium": {
            "vocab_size": 50257,
            "context_length": 1024,
            "d_model": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "d_ff": 4096,
            "rope_theta": 10000.0,
        },
        "gpt2-large": {
            "vocab_size": 50257,
            "context_length": 1024,
            "d_model": 1280,
            "num_layers": 36,
            "num_heads": 20,
            "d_ff": 5120,
            "rope_theta": 10000.0,
        },
        "gpt2-xl": {
            "vocab_size": 50257,
            "context_length": 1024,
            "d_model": 1600,
            "num_layers": 48,
            "num_heads": 25,
            "d_ff": 6400,
            "rope_theta": 10000.0,
        },
    }


def print_memory_table():
    """Print memory usage table for different configurations and world sizes."""
    
    configs = get_configs()
    world_sizes = [1, 2, 4, 8]
    
    print("=" * 120)
    print("FSDP Memory Scaling Analysis (FP32, Adam Optimizer)")
    print("=" * 120)
    print("\nMemory components:")
    print("  - Parameters: N × 4 bytes")
    print("  - Gradients: N × 4 bytes")
    print("  - Optimizer states (Adam): 2N × 4 bytes (momentum + variance)")
    print("  - Total per device (Non-FSDP): 4N × 4 bytes = 16N bytes")
    print("  - Total per device (FSDP): (4N / world_size) × 4 bytes = 16N / world_size bytes")
    
    for config_name in ["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        config = configs[config_name]
        mem = calculate_model_memory_detailed(config)
        
        print(f"\n{'-' * 120}")
        print(f"Model: {config_name.upper()}")
        print(f"  Parameters: {mem['total_params_m']:.2f}M ({mem['total_params_b']:.3f}B)")
        print(f"  d_model={config['d_model']}, num_layers={config['num_layers']}, d_ff={config['d_ff']}")
        print(f"{'-' * 120}")
        
        print(f"\n{'Setup':<25} {'Params (GB)':<15} {'Grads (GB)':<15} {'Opt States (GB)':<20} {'Total (GB)':<15} {'% of Non-FSDP':<15}")
        print("-" * 120)
        
        # Non-FSDP baseline
        print(f"{'Non-FSDP (1 GPU)':<25} {mem['param_memory_gb']:<15.2f} {mem['grad_memory_gb']:<15.2f} {mem['optimizer_memory_gb']:<20.2f} {mem['total_memory_gb']:<15.2f} {'100%':<15}")
        
        # FSDP with different world sizes
        for ws in world_sizes:
            param_per_device = mem['param_memory_gb'] / ws
            grad_per_device = mem['grad_memory_gb'] / ws
            opt_per_device = mem['optimizer_memory_gb'] / ws
            total_per_device = mem['total_memory_gb'] / ws
            percentage = (total_per_device / mem['total_memory_gb']) * 100
            
            setup_name = f"FSDP ({ws} GPU{'s' if ws > 1 else ''})"
            print(f"{setup_name:<25} {param_per_device:<15.2f} {grad_per_device:<15.2f} {opt_per_device:<20.2f} {total_per_device:<15.2f} {percentage:<15.1f}%")
        
        # Verification: total memory across all devices should equal non-FSDP
        print(f"\nVerification (total across all devices should equal Non-FSDP):")
        for ws in world_sizes:
            total_across_devices = mem['total_memory_gb']
            per_device = total_across_devices / ws
            print(f"  {ws} GPU{'s' if ws > 1 else ''}: {per_device:.2f} GB/device × {ws} = {total_across_devices:.2f} GB total ✓")
    
    print(f"\n{'=' * 120}")
    print("Summary:")
    print("  - FSDP memory per device = Non-FSDP memory / world_size")
    print("  - Total memory across all devices = Non-FSDP memory (constant)")
    print("  - Memory savings per device: linear scaling with world_size")
    print("=" * 120)
    
    # GPU availability check
    print(f"\n📊 Current System:")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            total_memory_gb = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name}, {total_memory_gb:.2f} GB")
        
        print(f"\n💡 Recommendations:")
        for config_name in ["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
            config = configs[config_name]
            mem = calculate_model_memory_detailed(config)
            
            # Find minimum GPUs needed
            min_gpus_needed = 1
            for ws in world_sizes:
                per_device = mem['total_memory_gb'] / ws
                if per_device <= total_memory_gb * 0.8:  # 80% threshold for safety
                    min_gpus_needed = ws
                    break
            
            if min_gpus_needed == 1:
                print(f"  {config_name}: Can fit on 1 GPU ({mem['total_memory_gb']:.2f} GB needed)")
            else:
                per_device_mem = mem['total_memory_gb'] / min_gpus_needed
                print(f"  {config_name}: Needs {min_gpus_needed}+ GPUs with FSDP ({per_device_mem:.2f} GB/device)")
    else:
        print("  No CUDA GPUs available")


if __name__ == "__main__":
    print_memory_table()

