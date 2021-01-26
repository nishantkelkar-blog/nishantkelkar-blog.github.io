---
layout: post
author: Nishant Kelkar
title: Deterministic Turing machines
tags: computer-science turing
---

This post goes into the definition of a deterministic Turing machine (DTM) as defined in the "holy grail" book by Garey and Johnson.
This machine is the central computational model around which the complexity class P is defined.

# Definitions

A deterministic Turing machine (DTM) sits at the center of all of computational complexity theory.
It is critical to the definition of the computational class of problems known as $\cal{P}$.
Roughly speaking, problems in $\cal{P}$ are those that we know algorithms for, that run in time polynomial in the "input size".
Out there in the vast, _most_ problems that exist are either undecidable (i.e. we cannot solve them), or are too hard
currently for a polynomial time algorithm to solve them.
In such a bleak arid wasteland of problems, the problems in $\cal{P}$ are somewhat of an oasis, a shining example of the progress
that we have been able to achieve as humans.

Let's get to it.
A DTM is a computation model, which means it has a computing unit, a mechanism to specify an input, and a mechanism to
read input.
We define each of these, starting in reverse order.
A DTM uses a "read/write head" to read in input.
This is basically a pointer to a specific location in the input.
The data at this specific location can then be read, or written to.
The input to a DTM is a horizontal "tape" that extends infinitely in both directions.
This tape has sequentially arranged "squares" on it left-to-right, each square holding some data.
Thus, each square becomes a location for data and so the read/write head at any point of time is pointing to _some_ square
on the tape.
Finally, the DTM has a "Finite State Control" module.
This module has 3 functions:

1. to read data from the square currently being pointed to by the read/write head (current square), and
2. to decide, if it has to, what to write to the current square, and
3. to decide, if it has to, whether to move to the square immediately to the left or immediately to the right of the current square

Figure (1) below shows all these components on an input tape with the squares, and each square with its index number shown
contained with it.
Each program that runs on a DTM starts with the read/write head pointing to square 0.
In the picture, you can tell that the DTM is in the process of executing some program, as the read/write head is currently
at square 1.

<figure class="blog-fig">
  <img src="/assets/images/dtm.jpg">
  <figcaption>Figure 1. A DTM tape with a computer reading from it</figcaption>
</figure><br/>

We now formally define what a _program_ input to a DTM looks like.
It consists of the following components:

1. A finite set of legal tape symbols $\rm T$
2. A subset of **input** symbols $\Sigma \subset \rm T$
3. A special "blank" symbol $b$ such that $b \in \rm T$ and $b \notin \Sigma$
4. A finite set of states $Q$ including 3 distinguished states: $q_0$ initial state, $q_Y$ "yes" state, and $q_N$ "no" state
5. A transition function $f: (Q - \\{ q_Y, q_N \\}) \times \rm T \to Q \times \rm T \times \\{ -1, +1 \\}$

The set of symbols $\rm T$ that can be accepted on the DTM tape need not be the same as the program symbols $\Sigma$,
but the program symbols are required to be a subset of the allowable tape symbols.
The special blank symbol $b$ can be used as an indicator in a program agnostic way -- to signal the beginning, or the
end of certain events.
This is why it is required to not be in $\Sigma$.

The "Finite State Control" (FSC) computer from figure (1) transitions from one state to another from the set of states $Q$.
We have a special "start state" $q_0$ in which every program on the DTM will always start in.
In the beginning, the FSC is in the state $q_0$, and the read/write head points to square 0 on the tape.
We also have two "terminal states" $q_Y$ and $q_N$.
The program terminates if it ever enters either of these two states.
$q_Y$ is the state we enter when the answer to our decision problem is "yes".
$q_N$ is the state we enter when the answer to our decision problem is "no".

The stage is almost set, except we need a way for the FSC to transition from one state to another.
The transition function $f$ defines the rules for making these state transitions.
Suppose that the FSC is in a non--terminal state $q$, and the read/write head is pointing to a square with a symbol $s$.
Suppose also that the location of this tape square is given by $l$.
The tuple $(q, s)$ forms an input to our transition function $f$.
$f(q, s)$ then returns a _triple_ $(q', s', \delta)$.
$q'$ is the new state which the FSC has transitioned to.
$s'$ is the new symbol that the read/write head wrote to the tape square which previously had the symbol $s$ in it.
Finally, the value of $\delta$ gives where the read/write head is currently at.
If $\delta = -1$ then the head has moved to the location $l' = l - 1$.
If $\delta = +1$ then the head has moved to the location $l' = l + 1$.
Note that since $\delta \in \\{ -1, +1 \\}$, we are guaranteed that the read/write head will move to a neighboring tape
square relative to the location $l$.

# Example walkthrough

The above section introduces the formal definitions of what a DTM is, along with what we mean by an input _program_ to
such a DTM.
In this section, we will look at a concrete example of such a program that solves a real--life decision problem.
The decision problem is given as follows:

<hr />
<div class="problem">EVEN-NUMBER</div>
<span class="problem-headers">Definition:</span> Given an integer number $a$<br/>
<span class="problem-headers">Question:</span> Is $a$ even (divisible by 2)?
<hr /><br/>
Let us go through the steps of designing a program for this.
Suppose that our computer can only understand binary, and so our tape symbols are $\rm T = \\{ 0, 1, b \\}$, and since 
our program also has to be in binary, $\Sigma = \\{ 0, 1 \\}$.
This means that the input number $a$ must be specified in binary to us, for us to run it on the DTM.
We already have the special states $q_0$, $q_Y$, and $q_N$.
Here, $q_Y =$ the terminal state we reach if $a$ is even, and $q_N =$ the terminal state we reach if $a$ is not even.
$q_0$ is the special start state that we must start in, and when we start, we always start at square 0 on the tape.

To check if a number is divisible by 2 in binary is easy: simply check its least--significant--bit (LSB) and see if it is
set to 0.
If it is, the number is even, otherwise it is odd.
This will be our main strategy then.
We will assume that the number $a$ is written out on our tape starting with its most--significant--bit (MSB) located at
square 0, and that it extends to the right.
All the squares to the _left_ of location 0 are irrelevant for our program, and we can simply assume them to be filled
in with $b$ so that we may ignore them.
All the squares to the _right_ of the LSB are also assumed to be irrelevant, and filled in with a $b$ symbol.
For example, figure (2) below shows the number $5 = (0101)_2$ with its MSB "0" at square 0, and its LSB "1" at square 3.

<figure class="blog-fig">
  <img src="/assets/images/dtm-example.jpg">
  <figcaption>Figure 2. Finding if the number $(0101)_2\ (= 5)$ is even or not using a DTM</figcaption>
</figure>

The general idea of the above algorithm then is: starting at square 0, keep moving rightwards until we hit the first $b$
after the LSB.
Once we encounter this symbol, we move 1 square to the left and inspect the value of the symbol in that square.
If it is "0", we transition to the $q_Y$ state, or else if it is "1" then we transition to the $q_N$ state.
It can only ever be either 0 or 1, since the LSB is part of the binary representation of the input integer $a$.
The state transition "table" that encapsulates this strategy is shown below:

|  q\s  |       0        |       1        |       $b$        |
| :---: | :------------: | :------------: | :--------------: |
| $q_0$ | ($q_0$, 0, +1) | ($q_0$, 1, +1) | ($q_1$, $b$, -1) |
| $q_1$ | ($q_Y$, 0, +1) | ($q_N$, 1, +1) |        -         |

This table shows the transitions clearly.
We have one row per non--terminal state in our program, and one column per legal tape symbol encountered.
In state $q_0$, as long as we encounter any binary symbols (0 or 1), we keep staying in state $q_0$ and we keep moving
to the right with $\delta = +1$.
As soon as we hit a $b$ symbol in $q_0$, we transition to the new state $q_1$ within the FSC and move the read/write head
back to the previous tape square (i.e. the LSB of the input integer $a$) with $\delta = -1$.

Our time in $q_1$ is short--lived.
We read in the bit value at the read/write head (which is pointing to the LSB now).
If it is a 0, we know that $a$ is even, and so we immediately transition to state $q_Y$ and we are done.
Otherwise, if it is a 1, we know that $a$ is odd, and so we transition to state $q_N$ and we are also done.
Note that in state $q_1$, we cannot encounter a $b$ symbol because if we did, then the $b$ symbol encountered in state
$q_0$ would not be the first non--binary symbol beyond the LSB.
Therefore, in the table above the corresponding cell for $(q_1, b)$ is not filled in.

This algorithm is also seen in action in the figure (2) above.
The read/write head starts at square 0 and keeps going toward the right until it hits the first $b$ symbol after the
LSB.
At that point, it backtracks to the LSB location, and checks if there is a 0 or a 1 in that square.
In the example from the figure, since $a = 5$, we know that its LSB position will have a 1, and therefore our algorithm
above would exit in state $q_N$ indicating that $a = 5$ is not even.
