#!/bin/bash
# Run GPT-2 XL equivalence test (Single GPU, DDP, FSDP)

set -e

CONFIG=${1:-small}  # small, medium, or gpt2xl

echo "=========================================="
echo "GPT-2 XL Equivalence Test"
echo "Config: $CONFIG"
echo "=========================================="

# Step 1: Single GPU
echo ""
echo "Step 1: Running Single GPU baseline..."
uv run python tests/test_gpt2xl_equivalence.py --mode single --config $CONFIG

# Step 2: DDP
echo ""
echo "Step 2: Running DDP (8 GPUs)..."
uv run torchrun --nproc_per_node=8 tests/test_gpt2xl_equivalence.py --mode ddp --config $CONFIG 2>&1 | grep -v "W1116"

# Step 3: FSDP
echo ""
echo "Step 3: Running FSDP (8 GPUs)..."
uv run torchrun --nproc_per_node=8 tests/test_gpt2xl_equivalence.py --mode fsdp --config $CONFIG 2>&1 | grep -v "W1116"

# Step 4: Compare
echo ""
echo "Step 4: Comparing results..."
uv run python tests/test_gpt2xl_equivalence.py --mode compare

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="


