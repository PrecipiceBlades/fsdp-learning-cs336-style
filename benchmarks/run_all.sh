#!/bin/bash
# Run all benchmarks

echo "=========================================="
echo "Running FSDP Benchmarks"
echo "=========================================="
echo ""

# Memory profiling
echo "1. Memory Profiling"
echo "-------------------"
python benchmarks/memory_profile.py \
    --n_layers 12 \
    --d_model 512 \
    --n_heads 8 \
    --d_ff 2048 \
    --batch_size 4 \
    --seq_len 512 \
    --world_size 4

echo ""
echo ""

# Communication profiling
echo "2. Communication Profiling"
echo "-------------------------"
python benchmarks/comm_profile.py --world_size 4

echo ""
echo ""

# Scaling analysis
echo "3. Scaling Analysis"
echo "-------------------"
echo "Testing weak scaling (model size grows with # GPUs)..."
for world_size in 1 2 4 8; do
    echo "World size: $world_size"
    # TODO: Add scaling benchmark
done

echo ""
echo "=========================================="
echo "Benchmarks Complete!"
echo "=========================================="

