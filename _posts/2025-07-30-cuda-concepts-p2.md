---
layout: post
author: Nishant Kelkar
title: Cuda Constructs Part 2 - Vector Addition
tags: computer-science
---

In [the part-1 blog post]({% post_url 2025-07-30-cuda-concepts-p1 %}) on this topic, we reviewed what a Grid, Block, and Thread were. We also saw how these are arranged in CUDA for facilitating parallelism of computation (which ultimately runs in each Thread). In this post, we will continue our review of key CUDA constructs. Particularly, we will dive into:
1. How to write a CUDA kernel in C
2. How to run the CUDA kernel as part of a C++ program (on a T4 GPU)

We will build upon our findings from part 1 of the series.

* This will become a table of contents (this text will be scrapped).
{:toc}


## CUDA programs in C

A basic CUDA kernel function signature looks like so:

```c++
__global__ void kernelFn(...) {
    // Logic of CUDA kernel function.
    // ...
    // ...
    // No return from this function.
}
```

It is important to clarify that every C++ program is also a CUDA program. And so as such, it can be compiled with the CUDA compiler `nvcc`.

There are a couple of peculiarities in the above snippet. First, the `__global__` keyword. This keyword tells the CUDA compiler that this is a function that can be called from both the host (CPU) and device (GPU) machines. Every function intended to run from within a Nvidia GPU kernel function must either have a `__global__` or `__device__` keyword associated with it. Some functions (e.g. methods defined on classes) may be required to be called from both the host and device (GPU) machines; in that case it is prefixed with the `__host__ __device__` keywords.

Second, note that the CUDA kernel function does not have a return value. All CUDA kernel functions always have a `void` return value. If anything is to be "returned" from the kernel function, it has to be written to a data object/pointer passed in as an argument, and then later extracted and accessed in the host code.

Next, we will look at an example that show a CUDA kernel in action.

### A CUDA kernel "Hello World" - vector addition

{% highlight c++ linenos %}
__global__ void vectorAdd(float* va, float* vb, float* vr, int n) {
    // Since these are vectors, we simply need a 1-D Block.
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (x >= n) {
        return;
    }
    vr[x] = va[x] + vb[x];
}
{% endhighlight %}

This is what the kernel looks like for a vector addition use-case. Here, `va` and `vb` are two vectors (represented as data arrays in C) of pre-defined length `n`. In this kernel, we assume that each Thread on the GPU computes the output value for 1 element in the output vector `vr` ("v-result").

First, we compute the index in the output data array `vr` for which the current Thread's execution of this kernel is responsible for. To do this, we "skip ahead" $$(\text{blockIdx.x} \times \text{blockDim.x})$$ elements, since they are for Threads in CUDA Blocks that is not this current Thread's Block. Note that we only have need for a 1-D Blocks here, as vectors are inherently 1-D constructs.

To this, we add the value $$\text{threadIdx.x}$$ which brings us to the element in the output data array we will write to. The following visual shows how the 1-D Blocks "overlay" on top of the 2 input arrays `va` and `vb`.

<figure class="blog-fig">
  <img src="/assets/images/cuda-vecadd-overlap.png">
  <figcaption>Figure 1. Overlap of CUDA Block and Thread constructs on vector data elements</figcaption>
</figure>

Then, in our CUDA kernel on line 4, we check whether `x` does not exceed the value `n` which is the length of the input vectors. This is for handling the case where the last Block may have threads that go __beyond__ the length of the input arrays. This is also shown in the visual above, where the final Thread at $\text{x} = 2B$ does not overlap over any elements of the input arrays, and so it ends up doing no work.

Finally, on line 6, we set `vr[x]` to be the sum of the values in `va` and `vb` at index `x`.

### Driver program for testing our kernel

The main function that runs the above kernel, is given below:

```c++
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include "kernel.h"
#include "util.h"


void vectorAddOnCPU(float* vah, float* vbh, float* vrh, int l) {
    for (int i = 0; i < l; i++) {
      vrh[i] = vah[i] + vbh[i];
    }
}

void vectorAddOnGPU(float* vah, float* vbh, float* vrdh, int l) {
  int p_l = l * sizeof(float);

  int nx = 32;  // Num threads in x-direction.
  int gd_x = (l / nx) + 1;

  dim3 block_dim(nx, 1, 1);
  dim3 grid_dim(gd_x, 1, 1);

  float* vad;
  float* vbd;
  float* vrd;
  cudaMalloc((void **)&vad, p_l);
  cudaMalloc((void **)&vbd, p_l);
  cudaMalloc((void **)&vrd, p_l);

  cudaMemcpy(vad, vah, p_l, cudaMemcpyHostToDevice);
  cudaMemcpy(vbd, vbh, p_l, cudaMemcpyHostToDevice);

  vectorAdd<<<grid_dim, block_dim>>>(vad, vbd, vrd, l);
  
  cudaDeviceSynchronize();

  cudaMemcpy(vrdh, vrd, p_l, cudaMemcpyDeviceToHost);
    
  cudaFree(vad);
  cudaFree(vbd);
  cudaFree(vrd);
}

int main(int argc, char *argv[]) {
  int l = 100000000;
  int p_l = l * sizeof(float);

  // Host input and output arrays.
  float* vah = (float*) malloc(p_l);
  float* vbh = (float*) malloc(p_l);
  float* vrh = (float*) malloc(p_l);

  srand(42);

  // Randomly fill in input values.
  double min_val = 1.00;
  double max_val = 100.00;
  for (int i = 0; i < l; i++) {
    vah[i] = min_val + ((double)rand() / RAND_MAX) * (max_val - min_val);
    vbh[i] = min_val + ((double)rand() / RAND_MAX) * (max_val - min_val);
  }

  /* CPU based implementation. */
  timeit(vectorAddOnCPU, vah, vbh, vrh, l);

  /* GPU based implementation. */
  float* vrdh = (float*) malloc(p_l);
  timeit(vectorAddOnGPU, vah, vbh, vrdh, l);

  int offending_idx = are_arrays_equal(vrh, vrdh, l, 6);

  if (offending_idx > 0) {
    printf("Mismatch between 'GPU' and 'CPU' modes found at %d!\n", offending_idx);      
  }
  
  /* Free malloc'd arrays. */
  free(vah);
  free(vbh);
  free(vrh);
  free(vrdh);
}
```

Assume for the purpose of this blog that a templated function `timeit` exists that can take in another function and it's arguments as input, and runs the function and prints out the runtime for the function. Also assume that an `are_arrays_equal` function exists that can be used to compare the values within 2 arrays, to `decimals` places (in the above case, `decimals=6`).

The above code does 3 things:

1. It runs vector addition on CPU, without any GPU kernels.
2. It runs vector addition on GPU, e.g. Nvidia's T4 GPU.
3. __[Optional]__ It compares the outputs of the 2 methods and ensures that the results are the same.

`vectorAddOnGPU` is the interesting function here. This function does 5 things, that all CUDA programs have to do in some form or another:

1. It runs `cudaMalloc` for allocating space to hold the inputs and outputs in GPU "global" memory (nowadays called High-Bandwidth Memory or HBM).
2. It copies the inputs from the _host_ to the device, i.e. the GPU "global" memory via `cudaMemcpy`.
3. It runs the `vectorAdd` CUDA kernel that we examined at the top, and waits for all the threads to finish via the `cudaDeviceSynchronize` blocking function.
4. It copies the results from the _device_ back to an array on the host, using `cudaMemCpy`. This is so that non-kernel code may later use/examine these results. This is the `vrdh` array ("dh" suffix intended to convey "device to host").
5. It frees all the memory that was allocated on the device in the GPU global memory for holding inputs and outputs using `cudaFree`.

### Executing the driver program

To run the above code, first put together this Makefile (preserve the tabs!):

```Makefile
objects = main.o kernel.o

all: $(objects)
	nvcc -arch=sm_75 $(objects) -o app

%.o: %.cc
	nvcc -arch=sm_75 -I. -dc $< -o $@ -x cu

%.o: %.cu
	nvcc -arch=sm_75 -I. -dc $< -o $@

clean:
	rm -f *.o app
```

`nvcc` is the Nvidia compiler for CUDA programs.

Once you create this Makefile, you can run `make all` to build a binary called `app`. Finally, run it -- `./app` -- to see the results being printed out. Note that the above architecture `sm_75` targets T4 GPUs, you will need to change this based on your GPU variant.

For completeness, the assumption is that your project directory looks like this:

```bash
.
├── Makefile
├── kernel.cu
├── kernel.h
├── main.cc
└── util.h
```

The kernel code lives in `kernel.cu` and the main function code above in `main.cc`. The contents of `util.h` are:

```c++
#ifndef UTIL_H
#define UTIL_H
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <functional>   // For std::function
#include <utility>      // For std::forward
#include <limits>       // For std::numeric_limits
#include <type_traits>  // For std::is_floating_point_v

// Defining the function implementation here because the linker is unable to find a 
// specialized version of this function suited for float, uint8_t, etc. types.
template <typename T>
int are_arrays_equal(const T* array1, const T* array2, size_t size, int decimals = 0) {
    if (array1 == nullptr || array2 == nullptr) {
        return 0; // Return index 0 as an error state for invalid pointers
    }

    if (decimals < 0) {
        std::cerr << "Error: 'decimals' cannot be negative. Using default of 0." << std::endl;
        decimals = 0;
    }

    // Calculate the tolerance value.
    // The tolerance is a small value that accounts for floating-point inaccuracies.
    T tolerance;
    if constexpr (std::is_floating_point_v<T>) {
        tolerance = std::pow(10.0, -decimals) * std::numeric_limits<T>::epsilon();
    } else {
        // For integer types, a tolerance of 0 means strict equality.
        tolerance = 0;
    }
    
    // Check if the numbers are equal up to 'decimals' decimal places.
    for (size_t i = 0; i < size; ++i) {
        // Use std::abs to handle both positive and negative differences
        if (std::abs(array1[i] - array2[i]) > tolerance) {
            return i; // Return the first index where a difference is found
        }
    }

    return -1; // All elements are equal within the tolerance
}

// Defining the function implementation here because the linker is unable to find a 
// specialized version of this function suited for float, uint8_t, etc. types.
template <typename Func, typename... Args>
void timeit(Func&& func, Args&&... args) {
    // Get the starting time point
    auto start_time = std::chrono::high_resolution_clock::now();

    // Call the function with its arguments
    // Use std::forward to preserve the value category (lvalue/rvalue) of the arguments
    std::forward<Func>(func)(std::forward<Args>(args)...);

    // Get the ending time point
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // Print the duration
    std::cout << "Function took " << duration.count() / 1e6 << " seconds to execute." << std::endl;

    return;
}

#endif // UTIL_H
```

Feel free to use your favorite GPT implementation to understand these templated functions in more detail :)

The contents of `kernel.h` are:

```c++
#ifndef KERNEL_H
#define KERNEL_H
#include <stdint.h>

__global__ void vectorAdd(float* va, float* vb, float* vr, int l);

#endif // KERNEL_H
```

## Separating out CUDA and C++ codes

[This blog](https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/) post from Nvidia is a great resource to understand in greater detail what is happening behind the scenes when using `nvcc` for compilation and a separate tool like `g++` for linking.
