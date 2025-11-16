#!/bin/bash
# Test and create submission for CS336 Assignment 2.5: FSDP

set -e  # Exit on error

echo "=========================================="
echo "CS336 Assignment 2.5: FSDP"
echo "Testing and Submission Script"
echo "=========================================="
echo ""

# Check if writeup exists
if [ ! -f "WRITEUP.md" ]; then
    echo "ERROR: WRITEUP.md not found!"
    echo "Please create your writeup using WRITEUP_TEMPLATE.md as a guide."
    exit 1
fi

echo "Step 1: Installing dependencies..."
echo "-----------------------------------"
pip install -e . > /dev/null 2>&1
pip install -e ".[dev]" > /dev/null 2>&1
echo "✓ Dependencies installed"
echo ""

echo "Step 2: Running unit tests..."
echo "-----------------------------"
pytest tests/ -v --tb=short || {
    echo "ERROR: Some tests failed!"
    echo "Please fix failing tests before submitting."
    exit 1
}
echo "✓ All tests passed!"
echo ""

echo "Step 3: Running integration tests..."
echo "-------------------------------------"
# Run distributed tests if possible
if command -v torchrun &> /dev/null; then
    echo "Running distributed integration test..."
    torchrun --nproc_per_node=2 tests/test_fsdp_integration.py || {
        echo "WARNING: Distributed tests failed or not fully implemented"
        echo "Continuing with submission..."
    }
else
    echo "torchrun not found, skipping distributed tests"
fi
echo ""

echo "Step 4: Creating submission archive..."
echo "---------------------------------------"

# Create submission directory
SUBMISSION_DIR="submission_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SUBMISSION_DIR"

# Copy necessary files
echo "Copying files..."
cp -r fsdp "$SUBMISSION_DIR/"
cp -r tests "$SUBMISSION_DIR/"
cp -r benchmarks "$SUBMISSION_DIR/"
cp WRITEUP.md "$SUBMISSION_DIR/"
cp README.md "$SUBMISSION_DIR/"
cp pyproject.toml "$SUBMISSION_DIR/"

# Create tarball
TARBALL="submission.tar.gz"
tar -czf "$TARBALL" "$SUBMISSION_DIR"

# Cleanup
rm -rf "$SUBMISSION_DIR"

echo "✓ Created $TARBALL"
echo ""

# Show submission contents
echo "Submission contents:"
tar -tzf "$TARBALL" | head -20
echo "..."
echo ""

# Show file size
SIZE=$(du -h "$TARBALL" | cut -f1)
echo "Submission size: $SIZE"
echo ""

echo "=========================================="
echo "✓ Submission ready!"
echo "=========================================="
echo ""
echo "Please submit: $TARBALL"
echo ""
echo "Before submitting, ensure:"
echo "  1. All tests pass"
echo "  2. WRITEUP.md is complete"
echo "  3. Code is well-documented"
echo "  4. You've answered all interview questions"
echo ""
echo "Good luck!"

