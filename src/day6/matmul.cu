#include <cuda_runtime.h>
#include <cstdio>

__global__ void matmul(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row*k + i] * B[i*n + col];
        }
        C[row*n + col] = sum;
    }

}

int main() {

    // M = m x k, N = k x n, P = m x n
    int m = 128;
    int n = 256;
    int k = 16; // k = n

    // Host variables
    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;

    // Allocate host memory
    A_h = (float*)malloc(m*k*sizeof(float));
    B_h = (float*)malloc(k*n*sizeof(float));
    C_h = (float*)malloc(m*n*sizeof(float));

    // Allocate device memory
    cudaMalloc((void**)&A_d, m*k*sizeof(float));
    cudaMalloc((void**)&B_d, k*n*sizeof(float));
    cudaMalloc((void**)&C_d, m*n*sizeof(float));

    // Initialize host memory
    for (int i = 0; i < m*k; i++) {
        A_h[i] = i;
    }
    for (int i = 0; i < k*n; i++) {
        B_h[i] = i;
    }
    for (int i = 0; i < m*n; i++) {
        C_h[i] = 0;
    }

    // Copy host memory to device
    cudaMemcpy(A_d, A_h, m*k*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, k*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, m*n*sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    matmul<<<grid, block>>>(A_d, B_d, C_d, m, n, k);

    // Copy device memory to host
    cudaMemcpy(C_h, C_d, m*n*sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (j == 0 && i == 0) {
                printf("at 0,0 %f ", C_h[i*n + j]);
                printf("\n");
            }
            if (j == n-1 && i == m-1) {
                printf("at max %f ", C_h[i*n + j]);
                printf("\n");
            }
        }
    }
}