# Day 5 — GPU Memory Hierarchy and Bandwidth

## Experiment

Compared two CUDA copy kernels:

1. Direct global-memory copy
2. Global copy staged through shared memory

Each element performs one global read and one global write.

Effective bandwidth:

    bandwidth =
        (bytes read + bytes written) / kernel time

For N float32 elements:

    useful bytes = 8N

## Observed regimes

### 1. Small inputs: launch-bound

From 1 KiB through roughly tens of KiB, execution time stayed
near 0.002 ms.

Increasing input size therefore increased calculated bandwidth
without substantially increasing runtime.

Conclusion:

Small copies are dominated by fixed kernel execution/launch costs,
not sustained DRAM bandwidth.

### 2. Medium inputs: cache-dominated

Bandwidth rose above the GPU's nominal DRAM bandwidth:

    1 MiB  : 408 GB/s
    4 MiB  : 577 GB/s
    16 MiB : 632 GB/s

Since this exceeds nominal GDDR6 bandwidth, repeated accesses are
being substantially served from the cache hierarchy.

The benchmark repeatedly accesses the same input/output buffers.

### 3. Cache-capacity transition

The largest discontinuity occurred between:

    16 MiB : 631.98 GB/s
    17 MiB : 221.59 GB/s

The copy has both an input and output buffer.

Therefore:

    16 MiB + 16 MiB ≈ 32 MiB working set
    17 MiB + 17 MiB ≈ 34 MiB working set

This transition is consistent with exhausting the GPU's roughly
32 MiB L2 cache and shifting toward DRAM traffic.

This is evidence, not a claim that cache behavior is a perfect
binary capacity boundary.

### 4. Large inputs: DRAM-bandwidth plateau

Large sizes stabilized around:

    220–235 GB/s

At 256 MiB:

    global copy = 235.45 GB/s

This is close to the GPU's nominal 256 GB/s memory bandwidth.

Therefore approximately 64–256 MiB represents the sustained
DRAM-bandwidth regime for this experiment.

## Shared-memory result

Shared staging did not provide a consistent benefit.

Direct:

    global read
    global write

Shared staging:

    global read
    shared write
    synchronization
    shared read
    global write

The shared-memory version adds work while removing no required
global-memory traffic.

Shared memory is useful when it enables reuse or cooperation,
not simply because shared memory itself is fast.

## Prediction versus measurement

Predicted:

- tiny workloads would be launch-bound
- bandwidth would increase with size
- large workloads would reach a bandwidth plateau
- one-use shared-memory staging would not consistently help

Observed:

All four predictions held.

The unexpected result was the extremely sharp transition between
16 MiB and 17 MiB.

That transition revealed a cache-capacity regime that was much
clearer than expected.

## Main lesson

Kernel performance depends on where the bytes actually come from.

A measured "bandwidth" number is not automatically DRAM bandwidth.

Before interpreting performance, determine whether the workload is:

    launch-bound
    cache-dominated
    DRAM-bandwidth-bound
    compute-bound