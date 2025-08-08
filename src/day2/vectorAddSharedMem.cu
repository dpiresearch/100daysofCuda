#include <iostream>
#include <vector>

// CUDA kernel to perform a sliding-window addition using shared memory.
// Each thread calculates C[i] = A[i] + A[i+1].
__global__ void slidingWindowKernel(const int* d_A, int* d_C, int N) {
    // Dynamically sized shared memory array for one block's data.
    // It's sized to hold blockDim.x elements, plus one "halo" element
    // from the adjacent block for the overlapping computation.
    extern __shared__ int s_data[];

    // Calculate the global index for the current thread.
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate the total number of threads.
    int total_threads = gridDim.x * blockDim.x;

    // --- Cooperative Loading into Shared Memory ---
    // The shared memory size is blockDim.x + 1.
    // Each thread loads one element from global memory into shared memory.
    // The last thread in the block loads the "ghost" element for the next block's
    // calculation (or handles the boundary case for the last block).
    if (global_id < N) {
        s_data[threadIdx.x] = d_A[global_id];
    }
    
    // For the last thread in the block, load the "halo" element.
    if (threadIdx.x == blockDim.x - 1 && (global_id + 1) < N) {
        s_data[threadIdx.x + 1] = d_A[global_id + 1];
    }

    // Wait for all threads in the block to finish loading from global memory.
    // This is crucial to ensure all necessary data is available in shared memory
    // before any thread begins its calculation.
    __syncthreads();

    // --- Perform Computation from Shared Memory ---
    // Each thread calculates its result using the fast shared memory.
    // The calculation for a thread at index `i` requires `s_data[i]` and `s_data[i+1]`.
    if (global_id < N-1) { // Only calculate up to N-1 to avoid out-of-bounds access for C[i+1]
        d_C[global_id] = s_data[threadIdx.x] + s_data[threadIdx.x + 1];
    }
}

int main() {
    const int N = 100;
    const int threadsPerBlock = 64;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Host-side vectors
    std::vector<int> h_A(N);
    std::vector<int> h_C(N - 1);
    
    // Initialize host input data
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
    }

    // Device-side pointers
    int *d_A = nullptr;
    int *d_C = nullptr;

    // Allocate memory on the device
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_C, (N - 1) * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with the specified grid, block size, and shared memory size.
    // The shared memory size is `(threadsPerBlock + 1) * sizeof(int)` to
    // accommodate the "halo" element for each block.
    slidingWindowKernel<<<blocksPerGrid, threadsPerBlock, (threadsPerBlock + 1) * sizeof(int)>>>(d_A, d_C, N);

    // Check for any CUDA errors after the kernel launch.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error after kernel launch: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Copy the results from the device back to the host
    cudaMemcpy(h_C.data(), d_C, (N - 1) * sizeof(int), cudaMemcpyDeviceToHost);

    // Print a few sample results to verify
    std::cout << "Original array A:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << h_A[i] << " ";
    }
    std::cout << "..." << std::endl;
    
    std::cout << "Calculated results C (from a sliding window sum):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << h_C[i] << " ";
    }
    std::cout << "..." << std::endl;
    
    // Verify a few results
    std::cout << "\nVerification check:" << std::endl;
    std::cout << "h_C[0] should be 0 + 1 = 1.  Result: " << h_C[0] << std::endl;
    std::cout << "h_C[1] should be 1 + 2 = 3.  Result: " << h_C[1] << std::endl;
    std::cout << "h_C[9] should be 9 + 10 = 19. Result: " << h_C[9] << std::endl;


    // Free device memory
    cudaFree(d_A);
    cudaFree(d_C);

    return 0;
}


