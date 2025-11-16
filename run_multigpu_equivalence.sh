#!/bin/bash

# Test strict equivalence across 1, 2, 4, and 8 GPUs
# All GPUs process the SAME data to verify mathematical correctness

set -e

echo "================================================================================"
echo "FSDP Multi-GPU Equivalence Test"
echo "Testing that 1, 2, 4, and 8 GPUs produce EXACTLY the same results"
echo "when all GPUs process the SAME data"
echo "================================================================================"

cd /root/cs336-assignment2.5-fsdp

# Clean up old results
rm -f /tmp/fsdp_*gpu.pt

# Test 1: Single GPU
echo ""
echo "Step 1: Testing 1 GPU..."
echo "--------------------------------------------------------------------------------"
uv run python test_multigpu_equivalence.py

# Test 2: 2 GPUs
echo ""
echo "Step 2: Testing 2 GPUs..."
echo "--------------------------------------------------------------------------------"
uv run torchrun --nproc_per_node=2 test_multigpu_equivalence.py

# Test 3: 4 GPUs  
echo ""
echo "Step 3: Testing 4 GPUs..."
echo "--------------------------------------------------------------------------------"
uv run torchrun --nproc_per_node=4 test_multigpu_equivalence.py

# Test 4: 8 GPUs
echo ""
echo "Step 4: Testing 8 GPUs..."
echo "--------------------------------------------------------------------------------"
uv run torchrun --nproc_per_node=8 test_multigpu_equivalence.py

# Compare all results
echo ""
echo "Step 5: Comparing all results..."
echo "--------------------------------------------------------------------------------"
uv run python test_multigpu_equivalence.py --compare

echo ""
echo "================================================================================"
echo "Test Complete!"
echo "================================================================================"

