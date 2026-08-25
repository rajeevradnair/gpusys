#include <cstdio>
#include <cuda_runtime.h>

int ceil_div(int n, int d) {
    return (n + d - 1) / d;
}

__global__ void fill_matrix(int* out, int height, int width) {

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < height && col < width) {
        int index = row * width + col;
        out[index] = index;
    }

}

void run_matrix_case(
    int height,
    int width,
    int block_x,
    int block_y
) {
    printf(
        "\nmatrix=%dx%d block=%dx%d\n",
        height,
        width,
        block_x,
        block_y
    );

    // Empty input: do not launch a kernel.
    if (height == 0 || width == 0) {
        printf("no kernel launch\n");
        return;
    }

    dim3 block(block_x, block_y);

    dim3 grid(
        ceil_div(width, block.x),
        ceil_div(height, block.y)
    );

    printf(
        "grid=%dx%d\n",
        grid.x,
        grid.y
    );

    int size = height * width;

    int* d_out;
    cudaMalloc(
        &d_out,
        size * sizeof(int)
    );

    fill_matrix<<<grid, block>>>(
        d_out,
        height,
        width
    );

    cudaDeviceSynchronize();

    int* h_out = new int[size];

    cudaMemcpy(
        h_out,
        d_out,
        size * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    // Validate every matrix element.
    bool passed = true;

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {

            int index =
                row * width + col;

            if (h_out[index] != index) {
                printf(
                    "FAIL row=%d col=%d expected=%d actual=%d\n",
                    row,
                    col,
                    index,
                    h_out[index]
                );

                passed = false;
                break;
            }
        }

        if (!passed) {
            break;
        }
    }

    if (passed) {
        printf("PASS\n");
    }

    cudaFree(d_out);
    delete[] h_out;
}
int main() {

    printf("=== ODD AND TINY SHAPES ===\n");

    run_matrix_case(1,   1,   16, 16);
    run_matrix_case(1,   17,  16, 16);
    run_matrix_case(17,  1,   16, 16);
    run_matrix_case(17,  17,  16, 16);
    run_matrix_case(31,  33,  16, 16);
    run_matrix_case(100, 257, 16, 16);

    // Empty work
    run_matrix_case(0, 10, 16, 16);
    run_matrix_case(10, 0, 16, 16);


    printf("\n=== BLOCK SHAPE COMPARISON ===\n");

    run_matrix_case(
        100,
        257,
        32,
        1
    );

    run_matrix_case(
        100,
        257,
        16,
        16
    );

    run_matrix_case(
        100,
        257,
        32,
        8
    );


    return 0;
}