#include <cstdio>
#include <cstdlib>

__global__ void matrixAdd(float *A, float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < n && idy < n) {
        C[idx * n + idy] = A[idx * n + idy] + B[idx * n + idy];
    }
}

int main(int argc, char** argv) {
    int n = 1024;
    int block_dim=16;

    size_t size = n * n * sizeof(float);

    float *A, *B, *C;

    cudaMalloc((void**)&A, size);
    cudaMalloc((void**)&B, size);
    cudaMalloc((void**)&C, size);


    float *A_h = (float*)malloc(size);
    float *B_h = (float*)malloc(size);
    float *C_h = (float*)malloc(size);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_h[i * n + j] = i * n + j;
            B_h[i * n + j] = (i * n + j) * 2;
        }
    }
    
    cudaMemcpy(A, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_h, size, cudaMemcpyHostToDevice);

    dim3 block(block_dim, block_dim);
    dim3 grid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);

    matrixAdd<<<grid, block>>>(A, B, C, n);
    
    float first = 0.f, last = 0.f;
    cudaMemcpy(&first, C, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last, C + (n * n - 1), sizeof(float), cudaMemcpyDeviceToHost);
    printf("n=%d block=%d \n", n, block_dim);
    printf("C[0]=%f C[n*n-1]=%f\n", first, last);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    free(A_h);
    free(B_h);
    free(C_h);
}