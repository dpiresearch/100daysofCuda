## Matrix Addition

*Code for Exercise 1 from [here](../../exercises/README.MD)*

---

The host code:

- Allocates memory for the input and output matrices on host and initializes the memory.
- Allocates memory for the input and output matrices on device and copys the input matrices to the device.
- Launches the kernel.
- Copys the output matrix from the device to the host and prints the results.
- Frees the memory on the device and host.


Kernel **matrix_addition_B** uses 2D blocks and uses one thread per element for addition. The kernel utilizes coalesced memory accesses because the j value is the fastest changing index which is computed using the *threadIdx.x* value.

Kernels **matrix_addition_C** and **matrix_addition_D** use 1D blocks for the matrix addition.

**matrix_addition_C** kernel uses a *for-loop* so each thread will add elements of the same row. Thus, in each iteration, threads in the same block will load a column of the input matrices. These loads are not coalesced and the kernel will be inefficient.

**matrix_addition_D** kernel uses a *for-loop* so each thread will add elements of the same column. Thus, in each iteration, threads in the same block will load a row of the input matrices. These loads are coalesced.

Output: ( Nvidia gtx titan x). Similar numbers for GV100, QV100, 2060 Super, and RTX 3070   

Running in FUNCTIONAL mode...
Compiling...
Executing...
Time taken for malloc and assignment: 5.893649 ms
Time taken for transfer: 2.188359 ms
Time taken for kernel run B: 377.422638 ms
Time taken for kernel run C: 1.358459 ms
Time taken for kernel run D: 94.369804 ms
Exit status: 0

For Nvidia Titan V, we have much smaller numbers for kernel D
Time taken for kernel run B: 341.455078 ms
Time taken for kernel run C: 1.725652 ms
Time taken for kernel run D: 7.154977 ms


TODO:

Q: Run C is supposed to be inefficient - why is the number so low?
Ans:  Per research, it's because of cache locality having a larger influence than data coalescing.  In Kernel C, each thread access consecutive elements: A[i*n+0], A[i*n+1], A[i*n+2]...  Kernel D has bad cache locality as each thread accesses elements with stride n:A[0*n+j], A[1*n+j], A[2*n+j]...

Q: Is cache locality dependent on the hardware/firmware used?
Yes, look to roofline models for each GPU to explain memory performance differences.
PMPP Section 5.1

