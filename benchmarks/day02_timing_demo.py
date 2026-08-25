import torch
from numba import cuda
import time
import statistics
from bench_utils import benchmark_cuda_pure_gpu, summarize, make_result

def fn(x:torch.Tensor):
    if x.is_cuda:
        return x + x
    else:
        raise RuntimeError("Expected CUDA tensor, but found on CPU")     

N = 10_000_000

# x is on cpu
print("Tensor is on CPU")
x = torch.randn(N)
try:
    times = benchmark_cuda_pure_gpu(
        fn,
        x,
        warmup=10,
        repetitions=50,
    )
    stats = summarize(times)
    print(stats)
except RuntimeError as e:
    print(str(e))

# x is on gpu
print()
print("Tensor is on GPU")
x = torch.randn(N, device="cuda")
# although gpu is being called, since the tensor is on CPU, the work of adding the tensors happens on the CPU 
# and when the CPU work is happening, the GPU is just blocked waiting
# hence measuring the CPU elapsed time and not the GPU elapsed time
times = benchmark_cuda_pure_gpu(
    fn,
    x,
    warmup=10,
    repetitions=50,
)
stats = summarize(times)

print(stats)

bytes_moved = N * 3 * 4

result = make_result(
    shape=(N,),
    dtype=x.dtype,
    device=x.device,
    implementation="torch_add",
    stats=stats,
    bytes_moved=bytes_moved,
)

print(result)


print()
N = 20_000_000
REPETITIONS = 50

x = torch.randn(N, device="cuda")

for _ in range(10):
    y = x + x

torch.cuda.synchronize()

naive_times = []

for _ in range(REPETITIONS):
    start = time.perf_counter()

    y = x + x

    end = time.perf_counter()

    naive_times.append((end - start) * 1000)


sync_times = []

for _ in range(REPETITIONS):
    torch.cuda.synchronize()

    start = time.perf_counter()

    y = x + x

    torch.cuda.synchronize()

    end = time.perf_counter()

    sync_times.append((end - start) * 1000)

event_times = []

for _ in range(REPETITIONS):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    y = x + x

    end.record()

    end.synchronize()

    event_times.append(start.elapsed_time(end))

print(
    "Naive wall clock with no synchronization:",
    statistics.median(naive_times),
    "ms",
)

print(
    "Synchronized wall clock, but with CPU overhead:",
    statistics.median(sync_times),
    "ms",
)

print(
    "CUDA events: (CUDA runtime & more accurate as it measures GPU timing)",
    statistics.median(event_times),
    "ms",
)
