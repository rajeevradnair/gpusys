# GPU Systems Lab — Benchmark Contract

## 1. Correctness before performance

Never benchmark an implementation until it passes its declared
correctness tests against a trusted reference.

Primary reference:
- PyTorch implementation where available.

A performance result from an incorrect kernel is invalid.

## 2. Inputs

For every benchmark record:

- tensor shape
- dtype
- device
- implementation name
- random seed
- layout/contiguity assumptions

Default seed:

    1234

Set seeds before generating benchmark inputs.

## 3. Warm-up

Default warm-up iterations:

    10

Warm-up executions are discarded.

Purpose:
- initialize CUDA runtime state
- remove one-time setup effects
- allow caches/runtime paths to stabilize
- exclude compilation where steady-state performance is being measured

Compilation/setup time must be reported separately when relevant.

## 4. Repetitions

Default measured repetitions:

    50

Do not report latency from one execution.

Each repetition must use the same benchmark contract unless the experiment
explicitly studies changing inputs or configuration.

## 5. GPU timing

Preferred method:

    CUDA events

Timing region:

    start event
        ↓
    GPU operation
        ↓
    end event
        ↓
    synchronize
        ↓
    elapsed_time

Unsynchronized CPU wall-clock timing is not valid for CUDA kernel latency.

Synchronized wall-clock timing may be used for experiments involving
end-to-end CPU-visible latency, but it must be labeled explicitly.

## 6. Summary statistics

Report:

- median latency
- p20 latency
- p80 latency

Median is the primary latency statistic.

Do not select the fastest observed run as the headline result.

## 7. Numerical validation

Compare candidate output against the reference using:

- max absolute error
- max relative error
- worst mismatch index
- expected value at worst mismatch
- actual value at worst mismatch

Initial dtype tolerances:

| dtype | atol | rtol |
|---|---:|---:|
| float32 | 1e-5 | 1e-5 |
| float16 | 1e-3 | 1e-3 |
| bfloat16 | 1e-2 | 1e-2 |

These are baseline defaults, not universal constants.

Operators such as reductions, softmax, and matmul may require
operation-specific tolerances justified by numerical evidence.

Never loosen a tolerance merely to make a failing implementation pass.

## 8. Baseline

Every optimization must have a named baseline.

Examples:

- torch.add
- torch.sum
- torch.softmax
- torch.matmul
- previous known-correct kernel

Do not claim "X% faster" without identifying the exact baseline.

## 9. Throughput

When useful, derive throughput from the operation contract.

For memory-oriented kernels:

    effective bandwidth =
        useful bytes moved / median execution time

The byte-count assumption must be written down.

For compute-oriented kernels, FLOP/s may be reported when the operation
count is well defined.

## 10. Result record

Every benchmark row should contain at least:

- shape
- dtype
- device
- implementation
- median_ms
- p20_ms
- p80_ms

And where applicable:

- throughput_GBps
- FLOPs/s
- warmup
- repetitions
- seed
- commit

## 11. Benchmark discipline

Before running an optimization experiment:

1. Freeze the baseline.
2. Freeze the shape/dtype matrix.
3. Run correctness.
4. Predict the expected result.
5. Benchmark.
6. Investigate unexpected results.
7. Preserve both winning and losing cases.

Do not:

- remove unfavorable shapes after seeing results
- change tolerances to hide errors
- compare different timing methods
- compare warmed-up code against cold-start code
- report only the fastest run
- hide regressions