#include <cuda_runtime.h>
#include <cstdio>

__global__ void matrixAdd_1D(float *A, float *B, float *C, int m, int n) {
    //__shared__ float shared_A[16][16];


    const int total = m * n;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total) {
        C[idx] = A[idx] + B[idx];
    }
    //if (dataidx < total) {
    //    shared_A[threadIdx.x][threadIdx.y] = A[dataidx];
    //}

    //__syncthreads();
    // for (int i = 0; i < total; i++) {
    //     C[i] = A[i] + B[i];
    // }
}

__global__ void matrixAdd_1D_shared(float *A, float *B, float *C, int m, int n) {

    const int total = m*n;

    extern __shared__ float shared_data[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total ) {
        shared_data[threadIdx.x] = A[idx];
    }

    __syncthreads();

    if (idx < total) {
        C[idx] = shared_data[threadIdx.x] + B[idx];
    }


        
}

int main(int argc, char** argv) {
    const int M = 128;
    const int N = 128;
    const int TOTAL = M * N;
    int block_dim = 256;
    int grid_dim = (TOTAL + block_dim - 1) / block_dim;

    // host variables
    float *A_h = (float*)malloc(TOTAL * sizeof(float));
    float *B_h = (float*)malloc(TOTAL * sizeof(float));
    float *C_h = (float*)malloc(TOTAL * sizeof(float));

    // device variables
    float *A_d, *B_d, *C_d;

    cudaMalloc((void**)&A_d, TOTAL * sizeof(float));    
    cudaMalloc((void**)&B_d, TOTAL * sizeof(float));
    cudaMalloc((void**)&C_d, TOTAL * sizeof(float));
    
    // initialize host variables
    for (int i = 0; i < TOTAL; i++) {
        A_h[i] = i;
        B_h[i] = i;
    }  

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);

    cudaMemcpy(A_d, A_h, TOTAL * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, TOTAL * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Time taken cudamemcpy: %f ms\n", elapsed_time);

    matrixAdd_1D<<<grid_dim, block_dim>>>(A_d, B_d, C_d, M, N);

    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);

    cudaEventElapsedTime(&elapsed_time, stop, start);
    printf("Time taken matrixAdd_1D: %f ms\n", elapsed_time);

    float first = 0.0f, last = 0.0f;

    cudaMemcpy(&first, C_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last, C_d + TOTAL-1, sizeof(float), cudaMemcpyDeviceToHost);

    printf("c[0]=%f c[TOTAL-1]=%f\n", first, last);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Time taken cudamemcpy first last: %f ms\n", elapsed_time);

    int shared_mem_size = M*N*sizeof(float);

    matrixAdd_1D_shared<<<grid_dim, block_dim, TOTAL*sizeof(float)>>>(A_d, B_d, C_d, M, N);

    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);

    cudaEventElapsedTime(&elapsed_time, stop, start);
    printf("Time taken matrixAdd_1D_shared: %f ms\n", elapsed_time);

}
