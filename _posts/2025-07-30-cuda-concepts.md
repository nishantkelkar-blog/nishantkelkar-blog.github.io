---
layout: post
author: Nishant Kelkar
title: Cuda Constructs - Part 1
tags: computer-science
---

Today we will review some key CUDA programming concepts. Particularly, we will dive into:

- What are the key constructs to think about when working with CUDA?
- How does a pixel (x, y) map to the CUDA constructs for a 2D image?
- How does linearization of images in memory work? What is the math behind this?

## Introduction

GPU programming with CUDA has gotten popular within the past decade or so. After the [Attention is All You Need](https://arxiv.org/pdf/1706.03762) paper, general interest in fast implementations of these $O(n^2)$ attention operations picked up really quickly. This was only accelerated by the arrivial of the [Flash Attention](https://arxiv.org/pdf/2205.14135) paper in 2022, and it's follow-ups [Flash Attention-2](https://arxiv.org/pdf/2307.08691) and [Flash Attention-3](https://arxiv.org/pdf/2407.08608) in the past ~2 years.


All these papers say 1 important thing -- that having a fast implementation of the attention operation at the CUDA kernel level is critical to being able to serve the long (128K, sometimes even larger) context lengths and blazing fast time-to-first-token (TTFT) we see in modern DL models.

Here we will explore the basic concepts in CUDA GPU programming (the most popular GPU programming framework out there as of now) to see how computation is organized so as to achieve these blazing fast results.

## Key Constructs

When working with CUDA, there are 3 key constructs: a `Grid`, a `Block`, and a `Thread`.

- A `Grid` is a _spatially 3-dimensional_ organization of `Block`s.
- A `Block` is a _spatially 3-dimensional_ organization of `Thread`s.
- A `Thread` is what runs your GPU CUDA kernel function. It is the fundamental unit of work in a GPU.

A Grid or Block does not have to strictly be 3-dimensional, it could be lower-dimensional too. Grid and Block dimensions are always specified in the format `(z_dim, y_dim, x_dim)`.

Here are some examples of different legal Grid/Block dimensions: `(0, 0, 2)` (linear), `(0, 1, 2)` (2-dimensional), `(2, 4, 6)` (3-dimensional). You cannot have 4 or higher dimensional Grids or Blocks, for example something like `(1, 1, 2, 3)` is not a valid dimension.


CUDA exposes a few datastructures for accessing these various dimensions and objects within them. A builtin variable `gridDim` has 3 member variables `x`, `y`, and `z`. `gridDim.x` gives the Grid dimension in the x-direction. Similar definitions hold for `gridDim.y` and `gridDim.z`.

Within a Block in a Grid, `blockDim` gives the dimensions of the Block. `blockDim.x` gives the Block dimension -- i.e. the number of Threads -- in the x-direction. Similar definitions hold for `blockDim.y` and `blockDim.z`.

**Note:** Each Block within a Grid must have the same dimensions.


<figure class="blog-fig">
  <img src="/assets/images/cuda-gridblockthread.png">
  <figcaption>Figure 1. CUDA Grid, Block, and Thread Example</figcaption>
</figure>

Figure 1 above shows an example of what a CUDA Grid looks like. The positive x, y, and z directions are shown in the image. In this example, the Grid dimension is `(1, 2, 3)` going by the standard `(z, y, x)` specification format.

Within each 'cell' of this Grid is a Block. Each Block has dimension `(3, 2, 4)`. Each 'cell' in a Block is a Thread. There is no further sub-division of a Thread, each Thread is what ultimately runs the CUDA kernel function.

For this example then,

- `gridDim.x = 3`
- `gridDim.y = 2`
- `gridDim.z = 1`
- `blockDim.x = 4`
- `blockDim.x = 2`
- `blockDim.x = 3`

Similarly, CUDA exposes a way to index into a specific Block or Thread, by exposing builtins `blockIdx` and `threadIdx`. These variables are only available for use **within** a CUDA kernel.

They function similar to their `...Dim` counterparts. As examples, `blockIdx.x`, gives the x-coordinate index of the Block that the CUDA kernel is running in. The same definition applies for `blockIdx.y` and `blockIdx.z`. `threadIdx.y` gives the y-coordinate of the Thread that the CUDA kernel is running in. The same definition applies for `threadIdx.x` and `threadIdx.z`.

## 2-D images: mapping pixels to threads and vice-versa

Alright, so now what can we do with all these constructs? Let us take a 2-D image as an example of arbitrary size $$m \times n$$ (rows x columns).
Suppose we overlay a **2-D Grid** over this image. The Grid has dimensions `(0, gridDim.y, gridDim.x)`. Each Block within this Grid is also 2-D, with shape `(0, blockDim.y, blockDim.x)`. Further let's assume, that this Grid overlay covers the entire image, and that each resulting Thread ends up processing just 1 unique pixel.

A few questions we can ask:

1. Given a specific Thread identified by the 4-tuple (`blockIdx.y`, `blockIdx.x`, `threadIdx.y`, `threadIdx.x`) - what is the image pixel that it will process?

2. Given the `(x, y)` coordinates of an image pixel, what Block and specific Thread -- identified via the 4-tuple (`blockIdx.y`, `blockIdx.x`, `threadIdx.y`, `threadIdx.x`) -- will this pixel be processed by?

3. How can we generalize this to the 3-D case where the input is a 3-D tensor (e.g. a color image) and consequently so is the Grid/Block?


To find the pixel `(x, y)` that would be processed by a specific 4-tuple (`blockIdx.y`, `blockIdx.x`, `threadIdx.y`, `threadIdx.x`), let us first try to think about just the y-coordinate. This is findable by first skipping ahead to the image-row for the current block, by ignoring the first $$blockDim.y * blockIdx.y $$ rows. Within the current Block, the y-coordinate is `threadIdx.y` so we need to move ahead those many rows too. Thus in total we moved `` from the origin in the y-direction, so 

$$
\begin{align}
    \text{y} = (\text{blockDim.y} \times \text{blockIdx.y}) + \text{threadIdx.y}
\end{align}
$$

Similarly,

$$
\begin{align}
    \text{x} = (\text{blockDim.x} \times \text{blockIdx.x}) + \text{threadIdx.x}
\end{align}
$$

**Note:** when you do this math inside of your CUDA kernel function, it is important to check that `y < m` and `x < n` before proceeding with computations, or else you will run into a segmentation fault as your memory access will be outside of the image bounds. This can happen if the Grid overlay on the 2-D image isn't perfect, i.e. the Blocks on the borders of the input image may have some Threads that are outside the boundaries of the image. In this case, for such Threads, your CUDA kernel function should do nothing and exit.

Conversely, suppose we know the `(x, y)` coordinates of a pixel P. How do we find out the id of the Thread that will process this pixel?

Take the y-coordinate as an example. We need to skip-ahead the first `blockIdx.y` blocks each of size `blockDim.y` in this direction, until we reach the Block that contains P. This is

$$
\begin{align}
    \text{blockIdx.y} = \left\lfloor\frac{\text{y}}{\text{blockDim.y}}\right\rfloor
\end{align}
$$

Similarly, we have for the x-direction,

$$
\begin{align}
    \text{blockIdx.x} = \left\lfloor\frac{\text{x}}{\text{blockDim.x}}\right\rfloor
\end{align}
$$

Computing which Thread will process P within the Block with indexes `(blockIdx.y, blockIdx.x)` is simple now (remember: `blockIdx.z = 0` because this is a 2-D case). We know that each Thread processes at most 1 pixel. This gives us

$$
\begin{align}
    \text{threadIdx.y} = \text{y} - (\text{blockIdx.y} \times \text{blockDim.y})
\end{align}
$$

I.e. subtracting $(\text{blockIdx.y} \times \text{blockDim.y})$ from $\text{y}$ has the effect of shifting the origin from `(0, 0)` to `(blockIdx.y, 0)`.
Similarly, for the x-coordinate, we have

$$
\begin{align}
    \text{threadIdx.x} = \text{x} - (\text{blockIdx.x} \times \text{blockDim.x})
\end{align}
$$


What if the input was a color image, which meant that it also had a color-channel dimension, and thus was a 3-D instead of 2-D tensor? In this case, the math is the same for the z-coordinate. The following gives the pixel `z` coordinate for the pixel processed by a Thread with z-coordinate `threadIdx.z` within a Block with z-coordinate `blockIdx.z` within a Grid with `gridDim.z` Blocks.

$$
\begin{align}
    \text{z} = (\text{blockDim.z} \times \text{blockIdx.z}) + \text{threadIdx.z}
\end{align}
$$

Conversely, and similar to the x and y coordinate cases, the `blockIdx.z` and `threadIdx.z` values for a Thread that processes a pixel P with coordinates `(x, y, z)` are given by

$$
\begin{align}
    \text{blockIdx.z} = \left\lfloor\frac{\text{z}}{\text{blockDim.z}}\right\rfloor
\end{align}
$$

$$
\begin{align}
    \text{threadIdx.z} = \text{z} - (\text{blockIdx.z} \times \text{blockDim.z})
\end{align}
$$

## Image linearization

It would be great if 2 or 3 dimensional arrays existed in the C programming language that would let us index into specific elements by specifying the x, y, and z coordinates of an input pixel e.g. like `C[x][y][z]`. We would reference the inputs directly then, once we know the `(x, y, z)` coordinates to use. but unfortunately such an indexing mechanism does not exist in C. Instead, 2 (or 3) dimensional inputs have to be stored as 1-dimensional arrays, and we have to resort to "indexing magic" to ensure that our computations continue to remain fast and correct.

We can consider storing 2-dimensional arrays in "row-major" format. This means that we store each row of elements contiguously, and append rows together to form a 1-dimensional array.

For example, the following matrix `M`

$$
\begin{align}
    \text{M} = \begin{pmatrix}
    10 & 20 & 30 & 40 \\
    -1 & -2 & -3 & -4
    \end{pmatrix}
\end{align}
$$


would be stored into a 1-dimensional array by concatenating the rows along the columnar direction, so like this:

$$
\begin{align}
    \text{M_vec} = \begin{pmatrix}
    10 & 20 & 30 & 40 & -1 & -2 & -3 & -4
    \end{pmatrix}
\end{align}
$$

Alright, so now that we can store a 2-dimensional matrix into a 1-dimensional array, let us assume we have an image of size $(m \times n)$ ($m$ rows, $n$ columns). How do we convert a pixel P with `(x, y)` coordinates in the 2-dimensional input image tensor to an index `i` in the 1-dimensional counterpart?

This is fairly easy actually. Assuming 0-based indexing, there are `y` rows that come before the row P is in. So we can 'skip' these many elements in 1 jump, i.e. we will skip `y * n` elements. Now the x-coordinate of P `x` gives us index the because the row that P is in is the 'current' row being looked at in the 1-dimensional array.

So the index of P in the 1-dimensional array is then $i = (y \times n) + x$.

But $\text{y} = (\text{blockDim.y} \times \text{blockIdx.y}) + \text{threadIdx.y}$ and $\text{x} = (\text{blockDim.x} \times \text{blockIdx.x}) + \text{threadIdx.x}$ themselves, so substituting these in,

$$
\begin{align}
    \text{i}_{2D}\ =\ &(m \cdot \text{blockDim.y} \cdot \text{blockIdx.y}) + (m \cdot \text{threadIdx.y})\ + \\
    &(\text{blockDim.x} \cdot \text{blockIdx.x}) + \text{threadIdx.x}
\end{align}
$$

This is generalizable in 3-dimensions. In 3-dimensions, we first skip to the right 'plane' in the z-direction that contains the pixel P of interest with coordinates `(x, y, z)`. We can do this by ignoring the first `z * (m * n)` since there are `z` 2-D planes that come before the plane that P is in. Once we are in the correct 'plane', the formula we have for the 2-D case still holds. So now, the index `i` is given by


$i = (z \times m \times n) + (y \times n) + x$.

And substituting using the formulas we have for `x`, `y`, and `z` using the Thread/Block indexing builtins, we get


$$
\begin{align}
    \text{i}_{3D}\ =\ &(m \cdot n \cdot \text{blockDim.z} \cdot \text{blockIdx.z}) + (m \cdot n \cdot \text{threadIdx.z}) &&+ \\
    &(m \cdot \text{blockDim.y} \cdot \text{blockIdx.y}) + (m \cdot \text{threadIdx.y}) &&+ \\
    &(\text{blockDim.x} \cdot \text{blockIdx.x}) + \text{threadIdx.x}
\end{align}
$$

And that's it! Let us recap what we studied in this blog:

1. We understood what the fundamental constructs -- `Grid`, `Block`, and `Thread` -- in any CUDA programming setup are (even though we did not write any C-programs).
2. We understood what `gridDim`, `blockDim`, `blockIdx`, and `threadIdx` mean and do within a CUDA kernel.
3. For 2-dimensional inputs, we understood how to map a Thread to a pixel and vice-versa.
4. For 3-dimensional inputs, we understood how to map a Thread to a pixel and vice-versa.
5. We discussed how 2 and 3 dimensional arrays are actually stored in 1-dimensional arrays in C, and discussed the 'row-major' format.
6. For 2 and 3 dimensional arrays stored in 'row-major' format, we discussed how we could find the index of the pixel a Thread is to process within the 1-dimensional array representation, within a CUDA kernel, i.e. the $$\text{i}_{2D}$$ and $$\text{i}_{3D}$$ index values.
