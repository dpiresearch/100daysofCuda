# Matmul with tiling, shared memory, and timings

Followup to day4

We implement matmul two ways - once with shared memory and the other without

Timing results for Nvidia Titan V

Running in FUNCTIONAL mode...
Compiling...
Executing...
TILE=32  grid=(1,1) block=(32,32)
Avg kernel time  naive: 0.253 ms
Avg kernel time tiled: 0.006 ms
Exit status: 0

However, for most of the other GPUs, the non-shared memory version was still faster.

Also, problem size constrained to 256 x 256 x 256, otherwise LeetGPU timed out
TILE of 32 x 32 seemed to work best.


