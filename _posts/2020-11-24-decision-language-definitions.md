---
layout: post
author: Nishant Kelkar
title: Decision problems and languages
---

This post goes into the definitions of decision problems and their relationship to languages, as defined in the "holy grail" 
book by Garey and Johnson.

# Introduction
[Garey and Johnson's](https://www.amazon.com/Computers-Intractability-NP-Completeness-Mathematical-Sciences/dp/0716710455) 
book on computational complexity is a great read, especially for beginners and newcomers to the world of complexity theory.
Contrary to what most would think, this book is actually very beginner friendly, and does not assume advanced knowledge of
any particular niche subject like the theory of languages, graph theory, etc.

This blog post is meant to solidify the definitions of the terms `decision problem` and `language`.

<!--
There onward, we define what a `deterministic Turing machine (DTM)` is, and then define how a program defined for a DTM 
can be thought of as a language.
We look at the key characteristics of a program that solves a decision problem, and finally using the analogy between a 
program and a language, we define the class P in terms of a language.
-->

# Decision problems

A _**decision problem**_ $\Pi$ consists of two parts: a definition of the problem, followed by a posed question based on 
this definition, with a "yes/no" type answer. 
For example, the following is a well-formed decision problem:

<hr />
<div class="problem">SUBGRAPH-ISOMORPHISM</div>
**Definition:** Given two graphs $G = (V_G, E_G)$ and $H = (V_H, E_H)$<br/>
**Question:** Is there a subgraph in $H$, say $H'$, that is isomorphic with $G$?
<hr /><br/>
This is a decision problem since it has a definition section that clearly defines the inputs (and/or parameters), and 
a question section that asks a "yes/no" question based on the inputs.

We denote by $D_\Pi$ the set of all valid _instances_ of the problem $\Pi$ (that satisfy the "Definition" part) and by 
$Y_\Pi \subset D_\Pi$ the set of all valid instances that also answer the "Question" part with a "yes".
Consequently, the set $D_\Pi - Y_\Pi$ of instances are those valid instances that are either (a) instances that answer 
the "Question" part with a "no", or (b) instances that run forever -- they produce neither a "yes", nor a "no" answer.

Consider the following decision problem $\Pi$:

<hr />
<div class="problem">EVEN-SUM</div>
**Definition:** Given a tuple of two natural numbers $(a, b)$<br/>
**Question:** Is the sum $(a + b)$ divisible by 2?
<hr /><br/>
Here are a few statements that are true:
1. $(0, 1) \in D_{\Pi}$
2. $(-1, 1) \notin D_{\Pi}$ (-1 is not a natural number)
3. $(2, 3) \notin L_{\Pi}$
4. $(2, 3) \in D_{\Pi} - L_{\Pi}$
5. $(0, 1) \notin L_{\Pi}$
6. $(0, 1) \in D_{\Pi} - L_{\Pi}$
7. $(1, 1) \in L_{\Pi}$
8. $(202, 101) \in L_{\Pi}$

From the above examples, note that if $x \in L_{\Pi}$ then implicitly we also know that $x \in D_{\Pi}$.

# Languages

For any finite set of symbols $\Sigma$, $\Sigma^\*$ is the set of strings made by concatenating symbols from $\Sigma$, 
each string having finite length.
We call $\Sigma$ an _alphabet_.
A few examples of strings from $\Sigma^\*$ where $\Sigma = \\{A, B\\}$ are: "" (empty string, often denoted as $\epsilon$), 
"A", "AB", "BA", "ABA", "ABB".
Clearly, $\Sigma^\*$ has an infinite number of strings in it, but each string in it is of finite length.
We define a _**language**_ $L$ _over_ the symbols $\Sigma$, if $L \subset \Sigma^\*$.

A really neat analogy here is...well, the English "language"!
Here, $\Sigma = \\{ A, a, B, b, ..., Z, z \\}$, $\Sigma^\* = \\{ \text{all strings possible with the 26 $\times$ 2 = 52 
letters} \\}$ and $L_{Oxford} = \\{$all English language words as defined in the Oxford dictionary$\\}$. 

You can see that the definition of an alphabet is really generic, and a language can be seen as a restricted set of strings 
formed by concatenating symbols from this alphabet.
For example, the string "gibberish" is in $L_{Oxford}$, but the string "gibberish**e**" is not in $L_{Oxford}$, even though 
it _is_ in $\Sigma^\*$.
