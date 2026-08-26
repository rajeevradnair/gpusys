#include <cstdio>
#include <cuda_runtime.h>


__global__ void global_copy(
    const float* input,
    float* output,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        output[i] = input[i];
    }
}


__global__ void shared_copy(
    const float* input,
    float* output,
    int N
) {
    extern __shared__ float shared[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        shared[threadIdx.x] = input[i];
    }

    __syncthreads();

    if (i < N) {
        output[i] = shared[threadIdx.x];
    }
}


float time_global_copy(
    float* input,
    float* output,
    int N,
    int blocks,
    int threads,
    int repetitions
) {
    for (int i = 0; i < 10; i++) {
        global_copy<<<blocks, threads>>>(input, output, N);
    }

    cudaDeviceSynchronize();

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    for (int i = 0; i < repetitions; i++) {
        global_copy<<<blocks, threads>>>(input, output, N);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms;

    cudaEventElapsedTime(
        &ms,
        start,
        end
    );

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return ms / repetitions;
}


float time_shared_copy(
    float* input,
    float* output,
    int N,
    int blocks,
    int threads,
    int repetitions
) {
    for (int i = 0; i < 10; i++) {
        shared_copy<<<
            blocks,
            threads,
            threads * sizeof(float) // this is the size of the shared memory
        >>>(
            input,
            output,
            N
        );
    }

    cudaDeviceSynchronize();

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    for (int i = 0; i < repetitions; i++) {
        shared_copy<<<
            blocks,
            threads,
            threads * sizeof(float)
        >>>(
            input,
            output,
            N
        );
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms;

    cudaEventElapsedTime(
        &ms,
        start,
        end
    );

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return ms / repetitions;
}


int main() {
    const int threads = 256;
    const int repetitions = 100;

    const size_t sizes[] = {
        1ULL * 1024,
        4ULL * 1024,
        16ULL * 1024,
        64ULL * 1024,
        256ULL * 1024,
        1ULL * 1024 * 1024,
        4ULL * 1024 * 1024,
        16ULL * 1024 * 1024,
        17ULL * 1024 * 1024,
        18ULL * 1024 * 1024,
        19ULL * 1024 * 1024,
        20ULL * 1024 * 1024,
        32ULL * 1024 * 1024,
        64ULL * 1024 * 1024,
        256ULL * 1024 * 1024
    };

    const int num_sizes =
        sizeof(sizes) / sizeof(sizes[0]);


    printf(
        "%12s %12s %15s %15s\n",
        "Size MiB",
        "Global ms",
        "Global GB/s",
        "Shared GB/s"
    );


    for (int s = 0; s < num_sizes; s++) {

        size_t bytes = sizes[s];

        int N =
            bytes / sizeof(float);

        int blocks =
            (N + threads - 1) / threads;


        float* d_input;
        float* d_output;

        cudaMalloc(
            &d_input,
            bytes
        );

        cudaMalloc(
            &d_output,
            bytes
        );


        // Input values do not matter for bandwidth measurement.
        cudaMemset(
            d_input,
            1,
            bytes
        );


        float global_ms =
            time_global_copy(
                d_input,
                d_output,
                N,
                blocks,
                threads,
                repetitions
            );


        float shared_ms =
            time_shared_copy(
                d_input,
                d_output,
                N,
                blocks,
                threads,
                repetitions
            );


        // Copy = one read + one write.
        double moved_bytes =
            2.0 * bytes;


        double global_seconds =
            global_ms / 1000.0;

        double shared_seconds =
            shared_ms / 1000.0;


        double global_gbps =
            moved_bytes
            / global_seconds
            / 1e9;


        double shared_gbps =
            moved_bytes
            / shared_seconds
            / 1e9;


        printf(
            "%12.3f %12.4f %15.2f %15.2f\n",
            bytes / 1024.0 / 1024.0,
            global_ms,
            global_gbps,
            shared_gbps
        );


        cudaFree(d_input);
        cudaFree(d_output);
    }


    return 0;
}