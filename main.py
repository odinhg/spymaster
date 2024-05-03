import torch
import numpy as np
from torchtext.vocab.vectors import FastText
import itertools


def load_vocab(filename: str, allowed_words: set[str] | None = None) -> list[str]:
    with open(filename, "r") as f:
        words = list(set(f.read().lower().splitlines()))
    if allowed_words is None:
        return words
    return [w for w in words if w in allowed_words]


def compute_distances(
    x: torch.Tensor, y: torch.Tensor, method: str = "cosine"
) -> torch.Tensor:
    if method == "cosine":
        return 1 - torch.nn.functional.cosine_similarity(
            x[:, :, None], y.t()[None, :, :]
        )
    elif method == "euclidean":
        return torch.cdist(x, y, p=2)
    else:
        raise ValueError(f"Unknown distance function {distance_function}.")


team_words = ["india", "flower", "bowl", "beer", "surf"]
enemy_words = ["cast", "engine", "bee", "death", "stick"]
assassin_word = "knife"
bystander_words = ["pine", "hit", "princess", "car"]


print(f"Team words:\n\t{team_words}")
print(f"Enemy words:\n\t{enemy_words}")
print(f"Assassin word:\n\t{assassin_word}")
print(f"Bystander words:\n\t{bystander_words}")

# Minimum and maximum team words subset size
min_n, max_n = (2, 6)
distance_function = "cosine" 
top_k_clues = 1
margin_bystander = 0.10
margin_enemy = 0.40
margin_assassin = 1.00

spymaster_vocab = "vocab.txt"
word_embedding = FastText(language="en")
words_in_embedding = set(word_embedding.itos)
vocabulary = load_vocab(spymaster_vocab)

team_embeddings = word_embedding.get_vecs_by_tokens(team_words, lower_case_backup=True)
enemy_embeddings = word_embedding.get_vecs_by_tokens(
    enemy_words, lower_case_backup=True
)
assassin_embedding = word_embedding.get_vecs_by_tokens(
    [assassin_word], lower_case_backup=True
)
bystander_embeddings = word_embedding.get_vecs_by_tokens(
    bystander_words, lower_case_backup=True
)
vocabulary_embeddings = word_embedding.get_vecs_by_tokens(
    vocabulary, lower_case_backup=True
)

dist_to_team = compute_distances(
    vocabulary_embeddings, team_embeddings, method=distance_function
)
dist_to_enemy = compute_distances(
    vocabulary_embeddings, enemy_embeddings, method=distance_function
)
dist_to_assassin = compute_distances(
    vocabulary_embeddings, assassin_embedding, method=distance_function
)
dist_to_bystanders = compute_distances(
    vocabulary_embeddings, bystander_embeddings, method=distance_function
)

min_dist_to_enemy, _ = torch.min(dist_to_enemy, dim=-1)
min_dist_to_bystanders, _ = torch.min(dist_to_bystanders, dim=-1)
dist_to_assassin, _ = torch.min(dist_to_assassin, dim=-1)

n_team = len(team_words)
illegal_words = set(team_words + enemy_words + bystander_words + [assassin_word])
vocabulary_numpy = np.array(vocabulary, dtype="object")
team_words_numpy = np.array(team_words, dtype="object")

candidates = {}
for r in range(min_n, min(max_n, n_team) + 1):
    for idxs in map(list, itertools.combinations(range(n_team), r)):
        target_embeddings = team_embeddings[idxs]
        dist_to_targets = dist_to_team[:, idxs]
        mean_dist_to_targets = torch.mean(dist_to_targets, dim=-1)
        max_dist_to_targets, _ = torch.max(dist_to_targets, dim=-1)
        targets_diameter = torch.cdist(target_embeddings, target_embeddings).max()
        score = (
            mean_dist_to_targets
            + torch.nn.functional.relu(
                max_dist_to_targets - min_dist_to_enemy + margin_enemy
            )
            + torch.nn.functional.relu(
                max_dist_to_targets - min_dist_to_bystanders + margin_bystander
            )
            + torch.nn.functional.relu(
                max_dist_to_targets - dist_to_assassin + margin_assassin
            )
        )
        top_k_scores, top_k_idx = torch.topk(score, 2 * top_k_clues + r, largest=False)
        target_words = list(team_words_numpy[idxs])
        c = 1
        for i, (score, clue_word) in enumerate(
            zip(top_k_scores, vocabulary_numpy[top_k_idx])
        ):
            if any(clue_word in word or word in clue_word for word in illegal_words):
                continue

            if clue_word in candidates:
                candidates[clue_word]["scores"].append(score.item())
                candidates[clue_word]["targets"].append(target_words)
            else:
                candidates[clue_word] = {
                    "scores": [score.item()],
                    "targets": [target_words],
                }

            c += 1
            if c > top_k_clues:
                break

for clue_word, clue_dict in candidates.items():
    targets_list, score_list = clue_dict["targets"], clue_dict["scores"]
    print(f"{len(targets_list)} target subsets for \"{clue_word}\":")
    for targets, score in zip(targets_list, score_list):
        n_clue = len(targets)
        print(f"\t{n_clue}\t{score:.3f}\t{targets}")
