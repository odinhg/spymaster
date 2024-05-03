# Codenames Spymaster 🕵️
## Generating clues from pre-trained word embeddings using a contrastive score function

This project uses a pre-trained word embedding (English FastText as default) and a fixed spymaster vocabulary (from the file `vocab.txt`) to generate clues for the game Codenames.

We specify three sets of words: $W_t, W_e$ and $W_b$ corresponding to our team's words, the enemy team's words and the neutral (bystander) words, respectively. We also specify an assassin word $w_a$.

For simplicity, we write $d(w, w') := d(F(w), F(w'))$ where $F$ is the word embedding and $d$ is the Euclidean or cosine distance function.

To score a candidate clue word $w_c$ targeting a subset $W\subseteq W_t$, we compute the expression

$$
\begin{align*}
S(w_c) = \frac{1}{|W|}\sum_{w\in W}d(w, w_c)
&+ \text{ReLU}\left(\max_{w\in W}d(w, w_c) - \min_{w\in W_e} d(w, w_c) + \alpha_e \right)\\
&+ \text{ReLU}\left(\max_{w\in W}d(w, w_c) - \min_{w\in W_b} d(w, w_c) + \alpha_b \right)\\
&+ \text{ReLU}\left(\max_{w\in W}d(w, w_c) - d(w_a, w_c) + \alpha_a \right)
\end{align*}
$$

where $\alpha_e, \alpha_b, \alpha_a>0$ are margin parameters.

The score function $S$ is inspired by contrastive loss functions and the value of $S(w_c)$ is minimized when $w_c$ is close to the words in $W$ and at a distance from the words in $W_e\cup W_b\cup \{w_a\}$. The margin parameters determines the amount of separation we desire for the different word categories. In Codenames, we typically want $\alpha_b\leq\alpha_e\leq\alpha_a$.

### Example usage

Run `python main.py` (or `python main.py | less` when the output is long) to generate clues targeting different subsets of the specified set of team words.

Example output:

```
Team words:
	['india', 'flower', 'bowl', 'beer', 'surf']
Enemy words:
	['cast', 'engine', 'bee', 'death', 'stick']
Assassin word:
	knife
Bystander words:
	['pine', 'hit', 'princess', 'car']
1 target subsets for "cashew":
	2	1.817	['india', 'flower']
3 target subsets for "bermuda":
	2	1.758	['india', 'bowl']
	3	1.758	['india', 'bowl', 'surf']
	4	2.069	['india', 'flower', 'bowl', 'surf']
1 target subsets for "chutney":
	2	1.844	['india', 'beer']
1 target subsets for "australia":
	2	1.591	['india', 'surf']
2 target subsets for "tangerine":
	2	1.565	['flower', 'bowl']
	3	1.901	['flower', 'bowl', 'surf']
2 target subsets for "tea":
	2	1.515	['flower', 'beer']
	4	2.064	['india', 'flower', 'bowl', 'beer']
...
```

