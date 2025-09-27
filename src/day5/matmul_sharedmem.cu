// nvcc -O3 -arch=native gemm_tiled.cu -o gemm_tiled
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

#ifndef TILE
#define TILE 16
#endif

// Baseline: no shared memory (naive O(N^3) loads from global)
__global__ void matMul_naive(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N
    if (row >= M || col >= N) return;

    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

// Tiled GEMM: reuse tiles from shared memory
__global__ void matMul_tiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int K, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;                 // 0..TILE-1
    int ty = threadIdx.y;                 // 0..TILE-1
    int row = blockIdx.y * TILE + ty;     // output row
    int col = blockIdx.x * TILE + tx;     // output col

    float acc = 0.f;

    // Number of tiles along K dimension
    int tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < tiles; ++t) {
        int aCol = t * TILE + tx;         // column in A tile
        int bRow = t * TILE + ty;         // row in B tile

        // Load A tile (row, aCol)
        if (row < M && aCol < K) {
            As[ty][tx] = A[row * K + aCol];
        } else {
            As[ty][tx] = 0.f;
        }

        // Load B tile (bRow, col)
        if (bRow < K && col < N) {
            Bs[ty][tx] = B[bRow * N + col];
        } else {
            Bs[ty][tx] = 0.f;
        }

        __syncthreads();

        // Multiply the tile
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

static inline void ck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(e));
        std::exit(1);
    }
}

float time_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.f;
    ck(cudaEventElapsedTime(&ms, start, stop), "evtElapsed");
    return ms;
}

int main() {
    // Problem sizes (square for simplicity). Try 1024, 2048, etc.
    const int M = 1024, K = 1024, N = 1024;
    const size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;

    std::vector<float> hA(szA), hB(szB), hC(szC), hRef(szC);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (size_t i = 0; i < szA; ++i) hA[i] = dist(rng);
    for (size_t i = 0; i < szB; ++i) hB[i] = dist(rng);

    // Reference on host (slow O(N^3) — keep sizes modest if you use this)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) acc += (double)hA[i*K + k] * (double)hB[k*N + j];
            hRef[i*N + j] = (float)acc;
        }
    }

    float *dA, *dB, *dC;
    ck(cudaMalloc(&dA, szA * sizeof(float)), "malloc dA");
    ck(cudaMalloc(&dB, szB * sizeof(float)), "malloc dB");
    ck(cudaMalloc(&dC, szC * sizeof(float)), "malloc dC");
    ck(cudaMemcpy(dA, hA.data(), szA * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
    ck(cudaMemcpy(dB, hB.data(), szB * sizeof(float), cudaMemcpyHostToDevice), "H2D B");

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    // Warm up
    matMul_naive<<<grid, block>>>(dA, dB, dC, M, K, N);
    matMul_tiled<<<grid, block>>>(dA, dB, dC, M, K, N);
    ck(cudaDeviceSynchronize(), "warmup sync");

    // Events
    cudaEvent_t s1, e1, s2, e2;
    cudaEventCreate(&s1); cudaEventCreate(&e1);
    cudaEventCreate(&s2); cudaEventCreate(&e2);

    // Time naive
    const int iters = 10;
    cudaEventRecord(s1);
    for (int i = 0; i < iters; ++i) {
        matMul_naive<<<grid, block>>>(dA, dB, dC, M, K, N);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms_naive = time_ms(s1, e1) / iters;

    // Validate
    ck(cudaMemcpy(hC.data(), dC, szC * sizeof(float), cudaMemcpyDeviceToHost), "D2H naive");
    // spot check few elements
    for (int tries = 0; tries < 5; ++tries) {
        int i = (rng() % M), j = (rng() % N);
        float got = hC[i*N + j], ref = hRef[i*N + j];
        if (fabs(got - ref) > 1e-2f) {
            fprintf(stderr, "Naive mismatch at (%d,%d): got %f ref %f\n", i, j, got, ref);
            return 2;
        }
    }

    // Time tiled (shared memory)
    cudaEventRecord(s2);
    for (int i = 0; i < iters; ++i) {
        matMul_tiled<<<grid, block>>>(dA, dB, dC, M, K, N);
    }
    cudaEventRecord(e2);
    cudaEventSynchronize(e2);
    float ms_tiled = time_ms(s2, e2) / iters;

    // Validate again
    ck(cudaMemcpy(hC.data(), dC, szC * sizeof(float), cudaMemcpyDeviceToHost), "D2H tiled");
    for (int tries = 0; tries < 5; ++tries) {
        int i = (rng() % M), j = (rng() % N);
        float got = hC[i*N + j], ref = hRef[i*N + j];
        if (fabs(got - ref) > 1e-2f) {
            fprintf(stderr, "Tiled mismatch at (%d,%d): got %f ref %f\n", i, j, got, ref);
            return 3;
        }
    }

    printf("TILE=%d  grid=(%d,%d) block=(%d,%d)\n", TILE, grid.x, grid.y, block.x, block.y);
    printf("Avg kernel time  naive: %.3f ms\n", ms_naive);
    printf("Avg kernel time tiled: %.3f ms\n", ms_tiled);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaEventDestroy(s1); cudaEventDestroy(e1);
    cudaEventDestroy(s2); cudaEventDestroy(e2);
    return 0;
}
