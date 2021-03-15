---
layout: post
author: Nishant Kelkar
title: An introduction to traditional feature selection
tags: machine-learning
---

Traditional machine learning (i.e. not deep learning) has to deal with feature selection as a problem when the number of features per data point is too large.
What happens with an explosion of features (think tens of thousands) is that we hit something called **The curse of dimensionality**.
A large number of features can also potentially lead to lesser model prediction efficacy due to the potential to introduce a lot of noise.
And finally, you do not want too many features simply because it makes the training phase of your model building slower.

## The need for feature selection

Feature selection is concerned with keeping the least amount of features that give the "best" performance.
Performance in this context, can mean many things.
For example, performance may refer to training time, or to prediction accuracy, or perhaps human readability and understanding of the model.

Consider a dataset with each row containing **100** features, and only **10** data points.
The data file, if it were a CSV file, would probably look something like this:

```text
index,f_00,f_01,f_02,...,f_99     # Header row
0,10.23,11.21,-1.05,...,8.723
1,12.11,15.00,-4.14,...,10.66
...
9,30.91,9.830,-10.02,...,6.49
```

Note how the number of data points is much lesser than the amount of information we have per data point.
Each row in this dataset can be imagined as a vector in a 100--dimensional vector space.
The number of points that we need to demarcate a cuboidal region in a 100--dimensional space (i.e. 2 points to demarcate the range of values along any dimension) would be $2^{100}$.

Why do we need these points in this higher dimensional space?
Well, imagine a 3D space (a cuboid) that you know your entire training (and testing) data lives in.
How many points mark the boundary of such a cuboid in 3D space?
They would be exactly be 8 -- or, $2^3$.
Similarly, in a 2D space (i.e. the plane) you would need 4 points to mark off a rectangular area, which is also exactly $2^2$.
In general, to mark a "cuboidal" area in an $n$D space, you would need $2^n$ points.
If a minimum of $2^n$ points isn't provided, all hope of a perfect training set classification is lost because we just don't know enough about the entire space of possibilities where the unseen (i.e. test) data may lie.
For example, if I only provide 4 data points on the `xz` plane in a 3D space, you would not be able to get a good classifier that predicts the class of unseen data points in 3D, because you do not have enough data -- since all 4 data points simply lie on a single 2D plane, and so anything off of that 2D plane is unseen and something the classifier has no idea about.
A similar phenomenon happens when we go from 3 dimensions, to in general $n$ dimensions.

<figure class="blog-fig">
  <img src="/assets/images/classif-2d.jpeg">
  <figcaption>Figure 1. Classifying 4 points in 3 dimensions all lying in the same plane</figcaption>
</figure>

Figure 1 above shows this "degenerate" situation, where a lower dimensional data is embedded in a higher dimensional space.
The input data isn't providing us any information along the y--axis in this example.
Therefore, what the classifier learns is probably a single dividing line between the 4 points that lie in the `xz` plane, which can be thought of as a 3D plane with no y--component in its definition.
Of course, this then ends up misclassifying the 2 blue points above the `xz` plane and the 2 red points below the `xz` plane.

If 8 points (4 blue, 4 red) are provided, a slanted discriminating plane like that shown in figure 2 may be learned.
In this situation, the plane correctly classifies all 8 data points, and it was able to do so because it had information about the distribution of points along the y--axis as well.

<figure class="blog-fig">
  <img src="/assets/images/classif-3d.jpeg">
  <figcaption>Figure 1. Classifying 8 points in 3 dimensions utilizing all 3 dimensions</figcaption>
</figure>

For various reasons (mostly cost related), if we have an $n$ dimensional space, collecting ~$2^n$ data points for a reasonably large value of $n$ is not practical.
Therefore, what we _can_ do is reduce the dimensionality of the dataset to $m$ where $m \ll n$ and also such that if we have $d$ data points, $d >= 2^m$.
Such a dimensionality reduction amounts to two things: (a) some features are kept and others have to be thrown away, and (b) since we are going from "more data" to "less data", we are inherently throwing away some information.
And so there is bound to be some data loss.

Such high--dimensional, low cardinality data is not uncommon in practice.
These are usually situations where data collection is expensive, but once you are doing it, you are trying to maximize the amount of information you collect.
For example, data collection about the various preferences of a university campus community with regards to library hours, books, etc.
Or, thermal + pressure time series data collected on two groups of engines -- one which function as expected, and one which have some sort of combustion chamber degradation.
Or yet still, classifying documents based on the words that they contain.
In all these examples, the number of data points (#campus students, #engines, #documents) is far far lesser than the data collected per point (library preferences, time series of temperature + pressure, and #possible distinct words in a single document).
So the motivation to reduce the dimensionality of such small datasets to get better model accuracies certainly exists.

## Feature selection methods

TODO: write about the wrapper method, filter method here

## Feature ranking riteria

TODO: write about what metrics are commonly used to rank features

## Further reading and references

TODO: scikit-learn references are good to put here
