---
layout: post
author: Nishant Kelkar
title: Cuda Constructs Part 3 - Memory Accesses
tags: computer-science
---

In this blog post, we will dive deeper into how memory accesses work on a GPU. Particularly, we will discuss:
- How memory is tiered, and how fast accesses are to each of these tiers.
- How CUDA coalesces global memory accesses, and how to structure kernels to optimize accesses by coalescing them.
- What shared memory banks are, how bank conflicts can occur, and how to remedy them.

TABLE OF CONTENTS
* This will become a table of contents (this text will be scrapped).
{:toc}

## Memory tiers in GPUs

In general, GPUs have 4 kinds of memory; registers, L1/shared memory, L2 cache, and "global" memory, also known as "High Bandwidth Memory" (HBM) in some specific GPU microarchitectures. The figure below shows these memories as they are laid out along-side execution units in streaming multiprocessors (SM).

<figure class="blog-fig">
  <img src="/assets/images/cuda-gpu-memory-arch.png">
  <figcaption>Figure 1. Standard Nvidia GPU memory architectural layout</figcaption>
</figure>

### Registers

Registers are small memory banks private to each GPU execution unit, and so thereby to each Thread that runs on this unit. These are primarily used to store arrays, variables, and any other temporary data initialized within the CUDA kernel. Cumulatively, registers add up to a very small amount of memory. For example, for all compute capabilities 7.5 (T4) - 9.0 (H100), each register is `4-bytes` long, and each SM can have at most `65,536` registers, for a total of `256KiB` of register memory per SM. On top of this, each Thread may at most use `255` registers, for a total of `~1KiB` of register memory access. You can thus imagine that this memory is to be used for only the most frequently used, but temporarily available variables per-Thread.

When all the registers of a Thread are filled up, CUDA stores the spill-over data of arrays and variables into the much larger -- but slower -- "global" memory (see below).

Because registers are considered "on-chip" i.e. physically right next to the GPU execution units as shown in figure (1), they also have extremely low latency, i.e. they are the fastest kind of memory to access.

### L1 data cache and shared memory

The L1 data cache and shared memory are two separate kinds of memory, but co-located in the GPU memory hierarchy, i.e. they both share the same physical "on-chip" space. A single Silicon block is used for both these memories, and partitioning into L1 cache and shared memory is done at the software level. Each L1 data cache/shared memory is specific to a SM.

L1 cache is the top-level cache on the read/write path to global memory. Data frequently read from/written to in global memory gets stored in the L1 cache.
This also means that when multiple SMs (each with their individual L1 cache) make write operations to locations in global memory for which they have data in their L1 caches, we could run into consistency problems. Nvidia does **NOT** guarantee cache consistency among the per-SM L1 caches. Because scheduling on SMs is on a per-Block basis, and because for a Grid launch for a given CUDA kernel we could have multiple Blocks, requiring that the L1 caches be consistent would amount to requiring an implicit dependency between the Blocks determined at runtime, which is against the whole GPU philosophy. With GPUs, we want to be able to truly execute all Blocks in parallel, independent of each other.

Shared memory as referenced here, is a common pool of memory accessible to each Thread in a Block. Threads outside of the current Block are not able to access shared memory of the current Block. Shared memory accesses are slower than register accesses, but are still many times faster than global memory access. Latencies to access shared memory and the L1 cache must be identical, as they both share the same physical characteristics. In fact, CUDA actually lets you provide hints to the runtime as to what proportion of the overall L1 data/shared memory space you would like to use as shared memory, the leftover being the L1 data cache. See the `cudaFuncAttributePreferredSharedMemoryCarveout` attribute [here](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g317e77d2657abf915fd9ed03e75f3eb0).

The following table shows various statistics w.r.t. memory sizes, hit latency (time in clock cycles to read/write from L1 cache/shared memory) and bandwidth (GiB/s). These were obtained by the Citadel study on T4 GPUs (see references below).

|  GPU (microarch)  |       Clock frequency (MHz)        | L1 data cache size (KiB)        |       L1 data hit latency (cycles)        |       Shared memory hit latency (cycles)        |       Shared memory obs. bandwidth (GiB/s)        |
| :---: | :------------: | :------------: | :------------: | :--------------: | :--------------: |
| T4 (Turing) | 1,590 | ≤64 | 32 | 19 | 3,662 |
| V100 (Volta) | 1,380 | ≤128 | 28 | 19 | 12,080 |
| P100 (Pascal) | 1,328 | 24 | 82 | 24 | 7,763 |
| P4 (Pascal) | 1,531 | 24 | 82 | 23 | 3,555 |

### L2 cache

The L2 cache is shared across all SMs in a GPU. All Threads in any Block can read/write from/to L2 cache. Writes to L2 cache happen when (a) a Thread intends to write to global memory, and the L2 cache is written to as part of this write operation, or (b) if some data is being read from global memory that is not in L2 cache yet (e.g. data that has been newly copied over from CPU main memory to global memory via `cudaMemcpy`).

Also note from figure (1) that the L2 cache is a _unified cache_ for data, instructions, and constants. By this, we mean that data, program instructions, or constants in global memory are all copied into the same L2 cache memory space. This is true for all the following microarchitectures: Maxwell, Pascal, Volta, Turing, and Ampere.

Here are the same statistics for performance as we noted above for L1 data/shared memory:

|  GPU (microarch)  |       Clock frequency (MHz)        | L2 data cache size (KiB)        |       L2 data hit latency (cycles)        |
| :---: | :------------: | :------------: | :------------: |
| T4 (Turing) | 1,590 | 4,096 | 188 |
| V100 (Volta) | 1,380 | 6,144 | 193 |
| P100 (Pascal) | 1,328 | 4,096 | 234 |
| P4 (Pascal) | 1,531 | 2,048 | 216 |

Looking at the T4 as an example to compare across L1 and L2 data caches, the hit latency is ~6x larger (i.e. slower) for L2 compared to L1. However, the size of the L2 data cache is at least **64x** larger than the L1 data cache! This shows that for memory design, some principles in memory physical layout that apply for regular CPU-based memories also apply for GPU-based designs - memory that is closer to the execution unit physically is usually faster to access than memory that is farther away.

### Global memory

The last kind of memory is the global memory. This memory is accessible from both host program (that runs on CPU) and device program (i.e. kernel code). A typical CUDA kernel launch involves first allocating enough space in the global memory via `cudaMalloc` and then copying over data from host memory (RAM) to this memory via `cudaMemcpy`, both operations being run in the _host_ program.

Global memory is also the largest kind of memory w.r.t. size. However, it is also the 'slowest' w.r.t. bandwidth, i.e. the number of bytes that can be moved from this memory to a higher level memory (L2 data cache) per second is the lowest compared to other memories (e.g. L1 data cache). Here are the memory size and bandwidth numbers:

|  GPU (microarch)  |       Clock frequency (MHz)        |       Memory size (MiB)        |    Obs. bandwidth (GiB/s)     |
| :---: | :------------: | :------------: | :------------: |
| T4 (Turing) |  5,001 |  15,079 |  220 |
| V100 (Volta) |  877 | 16,152 | 750 |
| P100 (Pascal) |  715 | 16,276 | 510 |
| P4 (Pascal) |  3,003 | 8,115 | 162 |

Reading the number for the T4 GPU again as an example, the global memory bandwidth is ~16.5x lesser than that for the L1 data cache, a massive difference!

Also note another curious difference compared to the L1 data cache table - the clock frequencies are different. This is because each GPU has _two_ clocks. One clock is for instruction execution, which is shown in the L1 data cache table. This is the graphics card clock rate. The other is for data back/forth between the GPU registers and global memory; this is the data transfer clock rate, shown in the table immediately above.

## Coalesced global memory access

<figure class="blog-fig">
  <img src="/assets/images/cuda-globalmemory-coalesce.png">
  <figcaption>Figure 2. A coalesced global memory access pattern</figcaption>
</figure>

<figure class="blog-fig">
  <img src="/assets/images/cuda-globalmemory-non-coalesce.png">
  <figcaption>Figure 3. What happens when global memory access isn't coalesced</figcaption>
</figure>

## Shared memory banks and bank conflicts

Shared memory is divided up into what is known as "banks". Each contiguous sequence of 4-bytes in shared memory is assigned to a bank. Consecutive sequences of 4-bytes are assigned to consecutive banks, with wraparound as there are only _32_ banks (for GPUs with compute capability ≥ 2.0).

Reads and writes to each bank are serialized. Reads/writes _across_ banks can happen in parallel. This means that if you have multiple threads in a Warp trying to read (or write) to different data all within the same bank, these reads (or writes) will be serialized, thus leading to a slower execution of the overall warp compared to if these reads (or writes) were happening on different banks. If you have `b` banks, and if you can structure your shared memory accesses such that your Warps read (or write) across these `b` banks, then effectively you can get a `b`x boost to your shared memory bandwidth.

<figure class="blog-fig">
  <img src="/assets/images/cuda-sharedmem-bank-conflicts.png">
  <figcaption>Figure 4. Shared memory bank conflicts toy example</figcaption>
</figure>

Let us take a toy example as shown in figure (4), where we have only 3 Threads trying to access elements of a `3x3` matrix, in an architecture which supports only 3 shared memory banks. The matrix being accessed is shown on the left; however, it is physically stored in shared memory as a sequential array of size 9, as shown on the write.

Because data is written to the shared memory banks in sequential row-major order, the value "1" is written to the YELLOW bank (Y), the value "2" is written to the BLUE bank (B), the value "3" is written to the GREEN bank (G), the value "4" is written to Y again since we have only 3 banks, and so on so forth.

Imagine the 3 Threads accessing this data over 3 "time steps", i.e. the CUDA kernel has a for-loop that loops from `i=1 to 3`, and for each Thread, accesses consecutive elements along the _rows_. This access can be for whatever goal the kernel wants to achieve (as a simple example, imagine that we are trying to square the numbers in the matrix). Also assume that in between each "time step" we have a `__syncthreads()` call.

At `t=0`, all 3 Threads will access the values in the first column (1, 4, and 7). Notice how all of these values are part of the Y bank. Because these 3 Threads are part of the same Warp and Block, the memory access for reading these 3 values will be serialized.

Similarly, at time `t=1`, all 3 Threads will access the values in the second column (2, 5, 8), which are part of the B bank. These reads will also be serialized by the hardware.

Finally, at time `t=2`, all 3 Threads will access the values in the last column (3, 6, 9), which are part of the G bank. Like the previous 2 cases, these reads will also be serialized by the hardware.

Assuming each bank $b$'s access requires $t_b$ time, across all 3 time steps, we have paid the cost:

$$T = 3*t_Y + 3*t_B + 3*t_G$$

for all these memory accesses. This is what we would call a shared memory bank conflict. Let us see what we can do about this now.

<figure class="blog-fig">
  <img src="/assets/images/cuda-sharedmem-bank-conflicts-resolved.png">
  <figcaption>Figure 5. Shared memory bank conflicts resolved for toy example</figcaption>
</figure>

Figure (5) shows how to remedy the situation. We know that the Threads access the columns of the shared memory matrix in between 2 `__syncthreads()` calls. To force the values of these columns in the _linearized_ physical layout to use different banks, we can add an extra column to the shared memory matrix representation, with some dummy value we do not care about (e.g. 0). Now we see that along each column, the bank colors are different! For example, whereas earlier column 1 (1, 4, 7) were all part of the Y bank, now 1 is part of Y, 4 is part of B, and 7 is part of G banks.

While the value => shared memory bank assigment has changed, the Thread's access pattern has not changed (i.e. we are still running the same CUDA kernel logic we had before). We can see that at each time step, the 3 Threads now access _different_ banks even though they access the same values in thes same order as before. Because the 3 bank accesses can happen in parallel, the total cost of accessing the shared memory is now:

$$T = 3*\max(t_Y, t_B, t_G)$$

This is much better than what we saw before!

## References

See [this NASA article](https://www.nas.nasa.gov/hecc/support/kb/basics-on-nvidia-gpu-hardware-architecture_704.html) for a great introduction to the details of these GPU memories.

The following 2 microbenchmarking studies by Citadel are also great in diving deeper into how memory performance is measured, and how popular "data center" GPUs in the various Nvidia microarchitectures compare with each other on various aspects:

- T4 GPU: <https://arxiv.org/pdf/1903.07486>
- A100 GPU (need a Developer Account and can download slide deck as PDF): <https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s33322/>

The following table gives compute capability related technical specifications. Each GPU has a compute capability, so to look up a GPU's technical specifications you need to know it's compute capability number.

- [Compute capability to GPU model mapping](https://developer.nvidia.com/cuda-gpus)
- [Technical specs per compute capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications)

For example, if I want to know the maximum number of Threads runnable in parallel per SM, from the first link I see that the T4 GPU has a compute capability of `7.5`. From the technical specification link, I can tell that for this compute capability, a maximum of `1024` Threads per SM can run in parallel.

In addition to this, the following resources were in-general helpful for understanding what happens under the hood w.r.t. memory accesses:

- Bob Crovella's answer [here](https://forums.developer.nvidia.com/t/difference-between-l2-read-write-transactions-and-l2-l1-read-write-transactions/80777/2).
- Bob Crovella's answer on inter-memory latencies, and microbenchmarking results [here](https://forums.developer.nvidia.com/t/why-reg-shared-global-is-faster-than-reg-global/215759/4).
