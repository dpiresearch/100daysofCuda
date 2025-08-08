#include <iostream>
#include <cuda_runtime.h>

__constant__ char d_message[20];

// cuda kernel to print message from thread 
__global__ void helloGPU(){
    printf("Hello from gpu thread %d! \n", threadIdx.x);
}

__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    // Kernel invocation with N threads
    // VecAdd<<<1, N>>>(A, B, C);
    helloGPU<<<1,1>>>();
}
