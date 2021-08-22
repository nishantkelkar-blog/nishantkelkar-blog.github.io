---
layout: post
author: Nishant Kelkar
title: Vector calculus for deep learning
tags: machine-learning deep-learning
---

This blog is an introduction to taking derivatives of scalar valued functions with respect to (w.r.t) other scalars, vectors, and matrices.
It assumes that you have a basic background in calculus and linear algebra (i.e. know what a derivative is, what tensors are, how vectors and matrices are defined, how they multiply, ...).

In general, we have [scalars, vectors, and tensors] (a matrix is just a tensor of rank = 2).
3 _classes_ in total, if you will.
Whenever you take a derivative of a quantity, you have to take it with respect to (w.r.t.) another quantity.
With a total of 3 classes, there are thus $3 \times 3 = 9$ possible ways to get a derivative (e.g. 'scalar w.r.t. scalar', 'matrix w.r.t. matrix', ...).

Here is a table of all the possible combinations and the _shape_ of the results.
By "shape" here, we are referring to what type the result is.
That is, is it a scalar? Or a vector, or a matrix? Or yet still, a tensor of rank = 3?
After all, the result itself must belong to one of the 3 classes listed above.

<figure class="blog-fig">
  <img src="/assets/images/derivative-classes.jpg">
  <figcaption>Figure 1. All possible combinations of derivative shapes with {scalar, vector, matrix} variable classes</figcaption>
</figure>

Of course, for each of these, to be certain of the shapes we are claiming we use, we have to define where each individual element in the result comes from (w.r.t. the original quantities).
So when we say $D = \frac{\partial A}{\partial B}$ when $A$ and $B$ are matrices is a tensor of shape $(p \times q \times n \times m)$, we have to define precisely what each element of $D$ looks like to us.
It is possible that in a different kind of definition of $D$, the result may have shape $(p \times q \times \bf{m} \times \bf{n})$, for example.

As we shall see in later blogs, to get good at setting up Deep Learning pipelines and training models, it isn't required that we know so much about tensor calculus, but it is good to know this because (a) you can show off this kind of stuff in front of your friends :) and (b) it really gives you a deeper understanding of what is happening under the hood, which makes it easier to also read DL literature.
Below we list the combinations in the first row of the table above, along with some explanations of the cases and examples to that you can "see" the math better.

## Derivative of a scalar w.r.t. a scalar

This is an easy one.
Suppose we have a function $f(x)$ where $x$ is a scalar, and the output of $f$ is also a scalar.
Then, the derivative $\frac{df}{dx}$ is also a scalar.
A concrete example of this is as follows: suppose $f(x) = x^2$ where $x$ is a scalar.
Then, $\frac{df}{dx} = 2x$ which when evaluated for a particular value of $x$, is also scalar.

## Derivative of a scalar w.r.t. a vector

This is where things start to get interesting.
We define the derivative of a scalar valued function $f(\bm{x})$ where $\bm{x}$ is a $(n \times 1)$ vector $[x_1, x_2, \ldots, x_n]^\top$, to be the _column vector_

$$
\frac{df}{d\bm{x}} = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \ldots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}
$$

This vector of partial derivatives w.r.t. the elements of $\bm{x}$ is also called "the _gradient_ of $f$ w.r.t. $\bm{x}$".
As an example, suppose you have the following setup:

$$
\begin{align*}
    f(\bm{x}) &= \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} \cdot \bm{x} \\
    \bm{x_0}  &= \begin{bmatrix} 5\\ 8\\ -1\\ \end{bmatrix}
\end{align*}
$$

This is the dot product of the constant vector $\begin{bmatrix} 1 & 2 & 3 \end{bmatrix}$ with $\bm{x}$.
Therefore, $f$ can be defined as $f(\bm{x}) = 1x_1 + 2x_2 + 3x_3$.
The value of $f(\bm{x_0})$ is then $(1 \times 5) + (2 \times 8) + (3 \times -1) = 18$, which is a scalar.

The gradient of $f$, $\frac{df}{d\bm{x}}$, is now given by

$$
\begin{align*}
    \frac{df}{d\bm{x}} &= \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \frac{\partial f}{\partial x_3} \end{bmatrix} = \begin{bmatrix} \frac{\partial (1x_1 + 2x_2 + 3x_3)}{\partial x_1} \\ \frac{\partial (1x_1 + 2x_2 + 3x_3)}{\partial x_2} \\ \frac{\partial (1x_1 + 2x_2 + 3x_3)}{\partial x_3} \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}
\end{align*}
$$

## Derivative of a scalar w.r.t. a matrix

This one is not too difficult to think of given we've solved for the case above.
Because the quantity we are taking a derivative of evaluates to a scalar, there aren't any multiple 'components' of it that we have to evaluate individually.
This means that all (partial) derivatives taken will always be for a single scalar quantity.
Since we are taking a derivative w.r.t. a matrix, we have to take the derivative of the scalar w.r.t. each individual element of the matrix, and arrange all those partial derivatives in a matrix itself.

Let $f(\bm{X})$ be a scalar function of the $(m \times n)$ sized matrix $\bm{X}$.
Then, we define the derivative of $f$ w.r.t. $\bm{X}$ as a matrix,

$$
\begin{align*}
    \bm{X} &= \begin{bmatrix} x_{11}, & x_{12}, & \ldots, & x_{1n} \\ x_{21}, & x_{22}, & \ldots, & x_{2n} \\ \ldots \\ x_{m1}, & x_{m2}, & \ldots, & x_{mn} \end{bmatrix} \\

    \frac{df}{d\bm{X}} &= \begin{bmatrix} \frac{\partial f}{\partial x_{11}}, & \frac{\partial f}{\partial x_{12}}, & \ldots, & \frac{\partial f}{\partial x_{1n}} \\ \frac{\partial f}{\partial x_{21}}, & \frac{\partial f}{\partial x_{22}}, & \ldots, & \frac{\partial f}{\partial x_{2n}} \\ \ldots \\ \frac{\partial f}{\partial x_{m1}}, & \frac{\partial f}{\partial x_{m2}}, & \ldots, & \frac{\partial f}{\partial x_{mn}} \end{bmatrix}
\end{align*}
$$

Let's look at a small example here.
Suppose $\bm{X_0}$ is a $(2 \times 2)$ matrix defined as follows:

$$
\begin{align*}
    \bm{X_0} = \begin{bmatrix} x_{11}, & x_{12} \\ x_{21}, & x_{22} \end{bmatrix} = \begin{bmatrix} 3, & 2 \\ 6, & -4 \end{bmatrix}
\end{align*}
$$

And let $f(\bm{X})$ be a scalar function defined on $(2 \times 2)$ matrices as follows:

$$
\begin{align*}
    f(\bm{X}) = x_{11}x_{22} - x_{12}x_{21}
\end{align*}
$$

You may realize that this function is actually the determinant of $(2 \times 2)$ matrices 😃
The derivative of this function w.r.t. $\bm{X}$ can be computed as,

$$
\begin{align*}
    \frac{df}{d\bm{X}} &= \begin{bmatrix} \frac{\partial f}{\partial x_{11}}, & \frac{\partial f}{\partial x_{12}} \\ \frac{\partial f}{\partial x_{21}}, & \frac{\partial f}{\partial x_{22}} \end{bmatrix} \\
    &= \begin{bmatrix} \frac{\partial (x_{11}x_{22} - x_{12}x_{21})}{\partial x_{11}}, & \frac{\partial (x_{11}x_{22} - x_{12}x_{21})}{\partial x_{12}} \\ \frac{\partial (x_{11}x_{22} - x_{12}x_{21})}{\partial x_{21}}, & \frac{\partial (x_{11}x_{22} - x_{12}x_{21})}{\partial x_{22}} \end{bmatrix} \\
    &= \begin{bmatrix} x_{22}, & -x_{21} \\ -x_{12}, & x_{11} \end{bmatrix}
\end{align*}
$$

When we specifically evaluate this on $\bm{X_0}$ we get,

$$
\begin{align*}
    \frac{df}{d\bm{X_0}} &= \begin{bmatrix} -4, & -6 \\ -2, & 3 \end{bmatrix}
\end{align*}
$$

## General tip on derivatives of scalars

In general, it is important to remember: when taking a derivative of a scalar function $L$ w.r.t. any class $\bm{Q}$, the shape of $\frac{\partial L}{\partial \bm{Q}}$ will always be the same as the shape of $\bm{Q}$.
This has to be the case, because in Deep Learning we are trying to find derivatives so that we can change $\bm{Q}$ according to the gradient update rule,

$$
\begin{align*}
    \bm{Q} = \bm{Q} - \alpha \frac{\partial L}{\partial \bm{Q}}
\end{align*}
$$

$\alpha$ is a scalar here, so it has no shape.
This subtraction on the right hand side will only work if $\bm{Q}$ and $\frac{\partial L}{\partial \bm{Q}}$ have the same dimensions.
$L$ is a scalar function here, that usually computes the _loss_ of the current model using the current value of $\bm{Q}$, a quantity we are trying to minimize.

That is all for now!
In later blogs, we will explore rows 2 and 3 of figure 1 above.
