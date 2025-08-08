# Vector Add with a twist

The inspiration for this code came from the following question from PMPP
https://github.com/R100001/Programming-Massively-Parallel-Processors/blob/master/Chapters/Ch02%20-%20Data%20Parallel%20Computing/exercises/README.md

2. Assume that we want to use each thread to calculate two (adjacent) elements of a vector addition. What would be the expression for mapping the thread/block indices to i, the data index of the first element to be processed by a thread?

A. i=blockIdx.x*blockDim.x + threadIdx.x +2;
B. i=blockIdx.xthreadIdx.x2;
C. i=(blockIdx.x*blockDim.x + threadIdx.x)*2;
D. i=blockIdx.xblockDim.x2 + threadIdx.x;
Correct answer: C

The problem assumes that by adjacent, they mean there are 2 independent elements in the data array from each thread


| Thread 1      | Thread 2      | Thread 3 | ...
| Data1 | Data2 | Data3 | Data4 |
| Ouput 1       | Ouptut2       |

In other words, Thread 1 -> data1 + data2, thread 2 => data3 + data4

But I had intepreted the question as

| Thread 1      | Thread 3      | Thread 3 | ...
        | Thread 2      | Thread 4       |
| Data1 | Data2 | Data3 | Data4 | Data 5 | Data 6 | 

| Ouput 1       | Ouptut 3      |
        | Ouptut 2      | Output 4       |

or thread 1 -> data1 + data2, thread2 -> data2 + data3

So the answer for me would have been i=(blockIdx.x*blockDim.x + threadIdx.x) + 1

In any case, this would make the data mapping a little trickier because you have to make sure the data is there for two threads intead of for one thread ( thread 1 and thread 2 depend on data2 being there ).  Hence the use of __syncthreads()

