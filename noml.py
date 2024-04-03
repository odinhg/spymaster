import torch
import numpy as np
from torchtext.vocab.vectors import GloVe, FastText
import itertools

def load_vocab(filename: str, allowed_words: set[str]) -> list[str]:
    with open(filename, "r") as f:
        words = f.read().lower().splitlines()
    print(f"Read {len(words)} words from {filename} ", end="")
    words = list(set(words))
    print(f"({len(words)} unique).")
    vocab = [w for word in words if (w := word.split()[0]) in allowed_words]
    return vocab


P = ["dog", "horse", "cat"]
N = ["fish", "australia"]
distance_function = "cosine"
top_k_clues = 10
margin = 0.1

#word_embedding = GloVe(name="6B", dim=300)
word_embedding = FastText(language="en")
words_in_embedding = set(word_embedding.itos)
W = load_vocab("nounlist.txt", words_in_embedding)
D = load_vocab("codename_vocab.txt", words_in_embedding)

assert len(set(P).intersection(set(N))) == 0, "One or more words appears in both sets."
assert all(p in D for p in P), f"One or more of the words {P} not in the deck."
assert all(n in D for n in N), f"One or more of the words {N} not in the deck."

P_emb = word_embedding.get_vecs_by_tokens(P, lower_case_backup=True)
N_emb = word_embedding.get_vecs_by_tokens(N, lower_case_backup=True)
W_emb = word_embedding.get_vecs_by_tokens(W, lower_case_backup=True)

if distance_function == "cosine":
    dist_WP = 1 - torch.nn.functional.cosine_similarity(W_emb[:,:,None], P_emb.t()[None,:,:])
    dist_WN = 1 - torch.nn.functional.cosine_similarity(W_emb[:,:,None], N_emb.t()[None,:,:])
elif distance_function == "euclidean":
    dist_WP = torch.cdist(W_emb, P_emb, p=2)
    dist_WN = torch.cdist(W_emb, N_emb, p=2)
else:
    raise ValueError(f"Unknown distance function {distance_function} given.")

min_dist_WN, _ = torch.min(dist_WN, dim=-1)

n_p = len(P)
W_numpy = np.array(W, dtype="object")
P_numpy = np.array(P, dtype="object")
illegal_words = set(P + N)

for r in range(2, n_p + 1):
    print(f"{r} word combinations:")
    for I_p in itertools.combinations(range(n_p), r):
        p_emb = P_emb[list(I_p)]
        diam_p = torch.cdist(p_emb, p_emb).max()
        print(f"Diam: {diam_p:.4f}")
        dist_Wp = dist_WP[:, list(I_p)]
        #mean_dist_Wp = torch.mean(dist_Wp, dim=-1)
        max_dist_Wp, _ = torch.max(dist_Wp, dim=-1)
        #score = mean_dist_Wp + torch.nn.functional.relu(max_dist_Wp - min_dist_WN + margin)
        score = max_dist_Wp + torch.nn.functional.relu(max_dist_Wp - min_dist_WN + margin)
        top_k_scores, top_k_idx = torch.topk(score, 2 * top_k_clues + r, largest=False)
        print(f"\tTarget: {P_numpy[list(I_p)]}\n\tAvoid: {N}")
        c = 1
        for i, (score, word) in enumerate(zip(top_k_scores, W_numpy[top_k_idx])):
            if word in illegal_words:
                continue
            if any(word in p or p in word for p in P):
                continue
            if any(word in n or n in word for n in N):
                continue
            print(f"\t{c:02}: {score:.04}\t{word}")
            c += 1
            if c > top_k_clues:
                break
        print("-"*50)
