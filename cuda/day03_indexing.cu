#include <cstdio>
#include <cuda_runtime.h>

__global__ void fn_1d() {

    int bi = blockIdx.x;
    int bd = blockDim.x;
    int ti = threadIdx.x;

    printf("\nblock index=%d, block dim=%d, thread index=%d", bi, bd, ti);

}

__global__ void fn_2d() {

    printf("block id = %d, row id=%d, col id =%d\n", blockIdx.x, threadIdx.y, threadIdx.x);
    
}

__global__ void fn_3d(int x_dim, int y_dim, int z_dim, int num_elements_to_process) {
    int global_idx = 
                    blockIdx.x * x_dim * y_dim * z_dim + 
                    threadIdx.x  + 
                    threadIdx.y * x_dim + 
                    threadIdx.z * x_dim * y_dim;

    if (global_idx < num_elements_to_process) {
        printf("global_idx=%d, block id = %d, x=%d, y=%d, z=%d, blockdimensions (%d, %d, %d)\n", global_idx, blockIdx.x, threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y, blockDim.z);
    }

}

int main() {

    //fn_1d<<<1,2>>>();

    //dim3 threads_per_block(3, 4);
    //fn_2d<<<1, threads_per_block>>>();

    int x_dim=3;
    int y_dim=4;
    int z_dim=5;
    dim3 threads_per_block(x_dim, y_dim, z_dim);
    fn_3d<<<2, threads_per_block>>>(x_dim, y_dim, z_dim, 100);
    cudaDeviceSynchronize();

}