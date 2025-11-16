# CS336 Assignment 2.5: FSDP - Writeup

**Name:** [Your Name]  
**Date:** [Submission Date]

---

## 1. Overview (1-2 pages)

Provide a high-level summary of your FSDP implementation.

### 1.1 What is FSDP?

Explain in your own words:
- What FSDP is and why it's important
- How it differs from DDP (from Assignment 2)
- Key memory savings achieved

### 1.2 Architecture Diagram

Include a diagram showing the flow of data through your FSDP implementation:
```
[Initialization] → [Forward Pass] → [Backward Pass] → [Optimizer Step]
```

Show where all-gather, reduce-scatter, and broadcast operations happen.

### 1.3 Key Design Decisions

Explain your major design decisions:
- How did you structure FlatParameter?
- How did you manage hook registration?
- Any optimizations you implemented?

---

## 2. Implementation Details (3-4 pages)

### 2.1 Task 1: Meta Device & Deferred Initialization

**Implementation approach:**
- Describe how you implemented meta device initialization
- How did you materialize parameters shard-by-shard?

**Key code snippets:**
```python
# Include key code snippets (2-5 lines each)
```

**Challenges:**
- What was tricky about this task?
- How did you debug issues?

### 2.2 Task 2: FlatParameter

**Implementation approach:**
- How did you flatten parameters into a single tensor?
- How did you handle parameter views?
- How did you compute shard ranges?

**Key code snippets:**
```python
# Your flattening logic
```

**Challenges:**
- How did you ensure views stay synchronized with flat storage?

### 2.3 Task 3: Forward Pass - All-Gather & Reshard

**Implementation approach:**
- Where did you hook into the forward pass?
- How did you implement all-gather?
- How did you implement optional resharding?

**Key code snippets:**
```python
# Your forward hook
```

**Trade-off analysis:**
- Compare reshard_after_forward=True vs False
- Memory usage difference
- Communication overhead difference

### 2.4 Task 4: Backward Pass - All-Gather, Compute, Reduce-Scatter

**Implementation approach:**
- How did you detect when to all-gather in backward?
- How did you implement reduce-scatter?
- When do your hooks fire?

**Key code snippets:**
```python
# Your backward hook
```

**Why reduce-scatter instead of all-reduce?**
- Memory comparison: N vs N/W gradient storage
- Communication volume comparison

### 2.5 Task 5: Sharded Optimizer

**Implementation approach:**
- How did you shard optimizer states?
- How did you implement the broadcast after step?

**Key code snippets:**
```python
# Your optimizer step logic
```

**Memory savings:**
- Calculate memory savings from optimizer state sharding
- For Adam with N parameters: Before vs After

### 2.6 Task 6: Prefetching & Communication-Computation Overlap

**Implementation approach:**
- How did you implement prefetching?
- How did you handle synchronization?

**Key code snippets:**
```python
# Your prefetch logic
```

**Performance impact:**
- Speedup from prefetching (include numbers)

### 2.7 Task 7: Full Integration

**Implementation approach:**
- How did you integrate all components?
- How did you handle nested modules?

**Challenges:**
- What integration bugs did you encounter?

---

## 3. Correctness Validation (2 pages)

### 3.1 Unit Test Results

Show results of your unit tests:
```bash
$ pytest tests/ -v
...
```

Include pass/fail status for each test.

### 3.2 Integration Test: FSDP vs Baseline

Compare FSDP training with non-distributed baseline:

**Model:** ToyModel (150 parameters)  
**Data:** Random data, 100 steps  
**Learning rate:** 0.01

**Loss curves:**
```
Step | Baseline Loss | FSDP Loss | Difference
-----|---------------|-----------|------------
0    | 1.234         | 1.234     | 0.000
10   | 0.856         | 0.856     | 0.000
...
100  | 0.123         | 0.123     | 0.000
```

**Analysis:**
- Maximum loss difference: ___
- Are they within acceptable tolerance? (Yes/No)

### 3.3 Parameter & Gradient Verification

After training, verify that parameters match:
- Final parameter L2 difference: ___
- Final gradient L2 difference: ___

---

## 4. Performance Analysis (3-4 pages)

### 4.1 Memory Profiling

**Setup:**
- Model: SimpleTransformer (N layers, d_model)
- Batch size: ___
- Sequence length: ___

**Results:**

| Configuration | Parameters | Memory (GB) | Reduction |
|---------------|------------|-------------|-----------|
| Baseline (1 GPU) | N | X.XX | - |
| FSDP (2 GPUs) | N | Y.YY | 2.0× |
| FSDP (4 GPUs) | N | Z.ZZ | 4.0× |

**Analysis:**
- Does memory scale as expected (1/W)?
- Where is memory being used? (params, grads, optimizer states, activations)

Include a breakdown:
```
Baseline memory breakdown:
- Parameters: X GB
- Gradients: X GB
- Optimizer states: X GB
- Activations: X GB

FSDP memory breakdown (per rank):
- Parameters: X/W GB
- Gradients: X/W GB
- Optimizer states: X/W GB
- Activations: X GB (not sharded)
```

### 4.2 Communication Profiling

**Results:**

| Operation | Tensor Size | Time (ms) | Bandwidth (GB/s) |
|-----------|-------------|-----------|------------------|
| All-gather | 1 MB | ___ | ___ |
| All-gather | 10 MB | ___ | ___ |
| Reduce-scatter | 1 MB | ___ | ___ |
| Reduce-scatter | 10 MB | ___ | ___ |

**Analysis:**
- How does communication time scale with tensor size?
- Which operation is faster: all-gather or reduce-scatter?

### 4.3 End-to-End Training Performance

**Setup:**
- Model: SimpleTransformer (__n_layers__)
- Training steps: 100
- Measure: time per step

**Results:**

| Configuration | Time/Step (ms) | Speedup | Communication % |
|---------------|----------------|---------|-----------------|
| Baseline | ___ | 1.0× | 0% |
| FSDP (no prefetch) | ___ | ___× | ___% |
| FSDP (with prefetch) | ___ | ___× | ___% |

**Analysis:**
- Why is FSDP slower/faster than baseline?
- How much does prefetching help?
- What is the communication overhead?

### 4.4 Scaling Analysis

**Weak scaling:** Model size grows with # GPUs

| # GPUs | Model Size (M params) | Memory/GPU (GB) | Time/Step (ms) |
|--------|----------------------|-----------------|----------------|
| 1 | M | ___ | ___ |
| 2 | 2M | ___ | ___ |
| 4 | 4M | ___ | ___ |

**Analysis:**
- Does memory per GPU stay constant? (It should!)
- How does time per step scale?

---

## 5. Interview Questions (2-3 pages)

Answer these questions as if in an interview:

### 5.1 Memory

**Q1: Walk through the memory breakdown of FSDP. Why does it achieve W× memory reduction?**

Your answer:

**Q2: What's the trade-off of `reshard_after_forward=True` vs `False`?**

Your answer:

**Q3: In FSDP, which tensors are sharded and which are replicated?**

Your answer:

### 5.2 Communication

**Q4: Why do we all-gather in the forward pass?**

Your answer:

**Q5: Why do we all-gather again in backward pass (if resharded)?**

Your answer:

**Q6: Why reduce-scatter instead of all-reduce for gradients?**

Your answer:

**Q7: What's the total communication volume per training step in FSDP?**

Your answer (derive formula):

### 5.3 Performance

**Q8: Why is prefetching essential for FSDP performance?**

Your answer:

**Q9: What's the peak memory overhead of prefetching?**

Your answer:

**Q10: When does prefetching NOT help?**

Your answer:

### 5.4 Architecture

**Q11: Why use FlatParameter instead of individual parameters?**

Your answer:

**Q12: How do you choose FSDP unit granularity (per-layer vs whole model)?**

Your answer:

**Q13: What happens with tied weights in FSDP?**

Your answer:

---

## 6. Bonus Tasks (if completed)

### 6.1 Hybrid Sharding (HSDP)

[If you implemented this bonus, explain your approach]

### 6.2 CPU Offloading

[If you implemented this bonus, explain your approach]

### 6.3 Other Optimizations

[Any other optimizations you implemented]

---

## 7. Reflection (1 page)

### 7.1 What did you learn?

- Key insights from implementing FSDP
- Connections to course material
- Relevance to real-world systems

### 7.2 Challenges

- What was most difficult?
- How did you overcome obstacles?

### 7.3 Future improvements

- If you had more time, what would you improve?
- What production features would be important?

---

## 8. Appendix

### 8.1 Commands to Reproduce Results

```bash
# Unit tests
pytest tests/ -v

# Memory profiling
python benchmarks/memory_profile.py --world_size 4

# Communication profiling
python benchmarks/comm_profile.py --world_size 4

# Integration test
torchrun --nproc_per_node=2 tests/test_fsdp_integration.py
```

### 8.2 Environment

- Python version: ___
- PyTorch version: ___
- CUDA version: ___
- Hardware: ___

### 8.3 Additional Notes

[Any additional notes, observations, or comments]

---

**Total Pages:** ___ (target: 10-15 pages)

**Submission Date:** ___

