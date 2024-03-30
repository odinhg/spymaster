# Codenames Spymaster

A Codenames spymaster self-supervised machine learning model based on word embeddings and set neural networks.

## Definitions 

Let $\mathcal{V}$ be our complete vocabulary and $\mathcal{W}\subseteq \mathcal{V}$ be the Codenames vocabulary. Morover, let $f\colon V\to\mathbb{R}^d$ be some fixed word embedding. For two words $w$ and $w'$, we write $d(w, w')=\Vert f(w) - f(w')\Vert_2$. By laziness, we sometimes write $w$ for $f(w)$ to simplify notation and maximize confusion.

A *game* consists of three subsets $T, E, C\subseteq \mathcal{W}$, and a word $a\in \mathcal{W}$. The sets $T$, $E$ and $C$ corresponds to our team's words, our enemy team's words and the civilian words, respectively. The word $a$ is the word corresponding to the assassin. Given a game $G=(T,E,C,a)$, we write $W(G)$ for the union $T\cup E\cup C\cup \{a\}$.

## Learning Objective

For a given game $G=(T,E,C,a)$, we want to find a clue word $c\in\mathcal{V}\setminus W(G)$ 

- minimizing $d(c, w)$ for all $w\in T$ and 
- maximizing $d(c, w)$ for all $w\in E\cup\{a\}$.

Let $g_T, g_E\colon\mathcal{P}(\mathbb{R}^d)\to\mathbb{R}^k$ be set nerual networks, i.e., permutation invariant neural networks. These networks will give us embeddings of our team's words and our enemy team's words, respectively. Finally, we use an MLP $h\colon\mathbb{R}^{2k+d}\to\mathbb{R}^d$ to combine these representations together with the embedding of $a$.


Given a subset $t\in\binom{T}{k}$ for some $1\leq k\leq |T|$, we define the function $c\colon\mathcal{P}(T)\to\mathbb{R}^d$ by letting

$$
c(t) = h(g_T(t), g_E(E), a).
$$

In other words, $c$ assigns a clue word to every subset of our team's words.

Scoring a clue word for a given subset $t\in\binom{T}{k}$:

- $|t|$ should be maximized, minimize $1/|t|$
- but $c(t)$ should be close to all words in $t$
- and $c(t)$ should be far away from $a$
- and $c(t)$ should be far away from all words in $E$
$\max_{w\in t}d(c(t), w) \leq \min_{w\in E} d(c(t), w)$
$\max_{w\in t}d(c(t), w) \leq d(c(t), a)$ 


Picking the final clue: For the best scoring word $c(t)$ pick the closest word to $c(t)$ from $\mathcal{V}$ and the number $|t|$.


Print the words in $t$.


Loss function:

$$
L(T,E,C,a) = \frac{1}{2^{|T|}-1}\sum_{t\subseteq T}\frac{1}{|t|}\text{ReLU}(\alpha+2\max_{w\in t} d(c(t), w)-\min_{w\in E}d(c(t), w)-d(c(t), a))
$$

We split the loss into three parts:

#### 1. Enemy loss $L_E$
$L_E=\text{ReLU}(\alpha_E+\max_{w\in t}d(c(t), w) - \min_{w\in E} d(c(t), w))$
#### 2. Civilian loss $L_C$
$L_C=\text{ReLU}(\alpha_C+\max_{w\in t}d(c(t), w) - \min_{w\in C} d(c(t), w))$
#### 3. Assassin loss $L_A$
$L_A=\text{ReLU}(\alpha_A + \max_{w\in t}d(c(t), w) - d(c(t), a))$

$$
L(T,E,C,a) = \frac{1}{2^{|T|}-1}\sum_{t\subseteq T}\frac{1}{|t|}(L_E(t)+L_C(t))+L_A(t)
$$

### Online Mining of Games (OMG)

Given a set of $25$ words, we can easily mine all possible game position with these cards.
Furhtermore, given the entire deck $\mathcal{W}$, we can mine all possible starting position.
Hence, we can simply push $\mathcal{W}$ to GPU and waste no time on data transfers.
