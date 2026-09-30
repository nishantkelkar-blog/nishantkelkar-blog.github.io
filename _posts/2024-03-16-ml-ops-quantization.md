---
layout: post
author: Nishant Kelkar
title: Model Quantization
tags: ml-ops
---

Today we dive deeper into ML model quantization. We examine multiple aspects of this:

- What is model quantization? What is it _not_?
- How does model quantization work?
- What are some experimental results with model quantization?

## Introduction

Model quantization refers to reducing the representation of the floating point weights of a deep learning model. This helps reduce both the memory requirements, as well as speed up arithmetic operations required as part of inferencing.

Importantly, model quantization is **not** about rounding down floating point weights to integers or bytes.
This would significantly affect the quality of the final model, and one of the goals of quantization is to achieve reduction in model size without affecting the quality metrics of a model (or at least not 'significantly').
