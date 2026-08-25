import torch
import statistics


def cuda_time_ms(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    fn()

    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end)


def benchmark_cuda_pure_gpu(fn, x, warmup:int =10, repetitions:int =50):
    for _ in range(warmup):
        fn(x)

    torch.cuda.synchronize()

    times = []

    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # place start event on stream
        start.record()
        fn(x)
        # place end event on stream
        end.record()

        # wait for gpu to finish executing upto end event in the stream
        end.synchronize()

        # measure the true gpu elapsed time
        times.append(start.elapsed_time(end))

    return times



def summarize(times):
    sorted_times = sorted(times)

    n = len(sorted_times)

    return {
        "p20_ms": sorted_times[int(0.20 * (n - 1))],
        "median_ms": statistics.median(sorted_times),
        "p80_ms": sorted_times[int(0.80 * (n - 1))],
    }

def make_result(
    *,
    shape,
    dtype,
    device,
    implementation,
    stats,
    bytes_moved=None,
):
    result = {
        "shape": shape,
        "dtype": str(dtype),
        "device": str(device),
        "implementation": implementation,
        "median_ms": stats["median_ms"],
        "p20_ms": stats["p20_ms"],
        "p80_ms": stats["p80_ms"],
    }

    if bytes_moved is not None:
        seconds = stats["median_ms"] / 1000.0
        result["throughput_GBps"] = bytes_moved / (seconds * 1e9)

    return result