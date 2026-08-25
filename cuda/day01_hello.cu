#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>

__global__ void f1(int* a) {
    *a=41;
}

int main() {
    int* a;
    int b;
    
    cudaMalloc(&a, sizeof(int));


    f1<<<1,2>>>(a);

    cudaMemcpy(
        &b,
        a,
        sizeof(int),
        cudaMemcpyDeviceToHost
    );

    std::cout<<b<<"\n";
    printf("b=%d\n",b);

    cudaFree(a);
}