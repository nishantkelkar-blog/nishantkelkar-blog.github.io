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


## Memory tiers in GPUs

In general, GPUs have 4 kinds of memory; registers, L1/shared memory, L2 cache, and "global" memory, also known as "High Bandwidth Memory" (HBM). The figure below shows these memories as they are laid out along-side execution units in streaming multiprocessors (SM).

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

The following table shows various statistics w.r.t memory sizes, hit latency (time in clock cycles to read/write from L1 cache/shared memory) and bandwidth (GiB/s). These were obtained by the Citadel study on T4 GPUs (see references below).

|  GPU (microarch)  |       Clock frequency (MHz)        | L1 data cache size (KiB)        |       L1 data hit latency (cycles)        |       Shared memory hit latency (cycles)        |       Shared memory obs. bandwidth (GiB/s)        |
| :---: | :------------: | :------------: | :------------: | :--------------: |
| T4 (Turing) | 1,590 | ≤64 | 32 | 19 | 3,662 |
| V100 (Volta) | 1,380 | ≤128 | 28 | 19 | 12,080 |
| P100 (Pascal) | 1,328 | 24 | 82 | 24 | 7,763 |
| P4 (Pascal) | 1,531 | 24 | 82 | 23 | 3,555 |

### L2 cache

### Global memory (or HBM)

### References

See [this NASA article](https://www.nas.nasa.gov/hecc/support/kb/basics-on-nvidia-gpu-hardware-architecture_704.html) for a great introduction to the details of these GPU memories.

The following 2 microbenchmarking studies by Citadel are also great in diving deeper into how memory performance is measured, and how popular "data center" GPUs in the various Nvidia microarchitectures compare with each other on various aspects:

- T4 GPU: <https://arxiv.org/pdf/1903.07486>
- A100 GPU (need a Developer Account and can download slide deck as PDF): <https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s33322/>

In addition to this, the following resources were in-general helpful for understanding what happens under the hood w.r.t. memory accesses:
- Bob Crovella's answer here: <https://forums.developer.nvidia.com/t/difference-between-l2-read-write-transactions-and-l2-l1-read-write-transactions/80777/2>
- Bob Crovella's answer on inter-memory latencies, and microbenchmarking results here: <https://forums.developer.nvidia.com/t/why-reg-shared-global-is-faster-than-reg-global/215759/4>
