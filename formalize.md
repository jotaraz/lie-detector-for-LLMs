Consider a circumstance $c$ (e.g., a situation a human or LLM can be in). 
And a reaction $y = f(c)$ to this circumstance (e.g., what the human or LLM then says).
Together they are an event $x = (c, y)$.
Consider a specific effect (e.g., does the human or LLM lie?).
There is a 'ground truth/label' function t(x) which maps events that have the effect to 1 and events that do not to -1.
Consider a list $C_1, ..., C_N$  of sets of different circumstances in which this effect can appear or not.
Here $C_i$ is a set containing circumstances of type $i$ (different types of circumstances could be: human is instructed to lie, human decides to lie because they consider it advantageous for themselves, human lies about their own past misbehavior, ...). 
For a specific, deterministic reaction function $f$ we can split $C_i$ into $C_i^+$ and $C_i^-$ such that 
$C_i^+ = {c \in C_i | t((c,f(c)))=+1}$, and
$C_i^- = {c \in C_i | t((c,f(c)))=-1}$.

Lets say we now have a representation of the reaction-process given a circumstance, thus encoding the event as $z(x) \in \mathbb{R}^n$ somehow.
We can then collect all these representations for all the different circumstance types and the different circumstances in the types. 

We now want to build a detector $d : \mathbb{R}^n \to {-1, +1}$ such that $d(z(x)) = t(x)$.
Detectors are 'trained' on sets, so we could train a detector on a set of circumstances and the reactions $X_i := (C_i, f(C_i))$, or on the union of some. 
And we can then measure the accuracy on some set (could be the same as the training set or a different one) by checking for how many circumstances the detector's label $d(z)$ matches the ground truth.
If we train on some events $X_a$ and then test on $X_b$ and we then observe that the accuracy on $X_b$ is very low there could be a few sources of errors.
1. The representation $z(x)$ might be lossy.
2. The training of the detector might have failed. 
3. The representations of the effect on $X_a$ and on $X_b$ are too different. 

First questions: Are there any other reasons for failure?
Second question: In practice, how does one know which reason dominates?


