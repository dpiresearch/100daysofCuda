# Matrix add two ways with shared memory and timings

Matrix add implementation to illustrate that using x,y,and z does not impact performance and is only used for ease of impl or readability

-- Output --
Running in FUNCTIONAL mode...
Compiling...
Executing...
Time taken cudamemcpy: 0.470821 ms
Time taken matrixAdd_1D: 1519.588745 ms
c[0]=0.000000 c[TOTAL-1]=32766.000000
Time taken cudamemcpy first last: 0.879468 ms
Time taken matrixAdd_1D_shared: 3506.687012 ms
Exit status: 0

-- questions --
Why did shared mem take longer here?

Ans: There were many problems with this impl

1. Threads not taking advantage of shared memory
2. Each thread operated on the entire dataset.
3. Shared memory allocation too large.


