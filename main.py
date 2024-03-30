import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import numpy as np
import random
import itertools
from torchtext.vocab.vectors import GloVe


def load_deck(filename: str = "codename_vocab.txt") -> list[str]:
    with open(filename, "r") as f:
        deck = f.read().lower().splitlines()
    return deck


def words2vec(
    word_embeddings: torchtext.vocab.Vectors, words: list[str]
) -> torch.Tensor:
    """
    Embed a list of words using a fixed word embedding.
    """
    vecs = word_embeddings.get_vecs_by_tokens(words, lower_case_backup=True)
    return vecs


def generate_random_position(deck: np.ndarray, embedded_deck: torch.Tensor):
    # Generate a random Codenames game position
    n_t = random.randint(2, 9)
    n_e = random.randint(1, 9)
    n_b = random.randint(1, 7)
    n = n_t + n_e + n_b
    idx = np.random.choice(len(deck), n + 1, replace=False)
    board = deck[idx]
    board_embedded = embedded_deck[idx]
    words_t = board[:n_t]
    vecs_t = board_embedded[:n_t]
    words_e = board[n_t : n_t + n_e]
    vecs_e = board_embedded[n_t : n_t + n_e]
    words_b = board[n_t + n_e : n_t + n_e + n_b]
    vecs_b = board_embedded[n_t + n_e : n_t + n_e + n_b]
    word_a = board[-1]
    vec_a = board_embedded[-1]
    position = {
        "team_words_emb": vecs_t,
        "team_words": words_t,
        "enemy_words_emb": vecs_e,
        "enemy_words": words_e,
        "bystanders_words_emb": vecs_b,
        "bystanders_words": words_b,
        "assassin_word_emb": vec_a,
        "assassin_word": word_a,
    }
    return position


class PermEquivLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xm = torch.mean(x, dim=-2, keepdim=True)
        out = self.linear(x - xm)
        return out


class Set2Vec(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()

        self.layers = nn.Sequential(
            PermEquivLayer(in_dim, hidden_dim // 4),
            nn.GELU(),
            PermEquivLayer(hidden_dim // 4, out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        out, _ = torch.max(out, dim=0)
        return out


class SpyMasterModel(nn.Module):
    def __init__(self, emb_dim: int, expand_factor: int = 2):
        super().__init__()
        hidden_dim = emb_dim * expand_factor
        self.set_nn_1 = Set2Vec(emb_dim, hidden_dim, emb_dim)
        self.set_nn_2 = Set2Vec(emb_dim, hidden_dim, emb_dim)
        self.set_nn_3 = Set2Vec(emb_dim, hidden_dim, emb_dim)
        self.linear = nn.Sequential(
            nn.Linear(4 * emb_dim, emb_dim),
            #nn.GELU(),
            #nn.Linear(2 * hidden_dim, emb_dim),
        )

    def forward(
        self, x_t: torch.Tensor, x_e: torch.Tensor, x_b: torch.Tensor, x_a: torch.Tensor
    ) -> torch.Tensor:
        h_t = self.set_nn_1(x_t)
        h_e = self.set_nn_1(x_e)
        h_b = self.set_nn_1(x_b)
        h = torch.cat([h_t, h_e, h_b, x_a], dim=0)
        out = self.linear(h)
        return out

class ClueLoss(nn.Module):
    def __init__(self, enemy_margin: float, bystander_margin: float, assassin_margin: float):
        super().__init__()
        self.enemy_margin = torch.tensor(enemy_margin)
        self.bystander_margin = torch.tensor(bystander_margin)
        self.assassin_margin = torch.tensor(assassin_margin)

    def forward(self, c_t, x_t, x_e, x_b, x_a):
        n_words_to_guess = x_t.shape[0]
        dist_to_team = torch.linalg.norm(x_t - c_t.unsqueeze(0), dim=1)
        max_dist_to_team = dist_to_team.max() 
        min_dist_to_enemy = torch.linalg.norm(x_e - c_t.unsqueeze(0), dim=1).min()
        min_dist_to_bystanders = torch.linalg.norm(x_b - c_t.unsqueeze(0), dim=1).min()
        dist_to_assassin = torch.linalg.norm(x_a - c_t)

        enemy_loss = F.relu(self.enemy_margin + max_dist_to_team - min_dist_to_enemy)
        bystander_loss = F.relu(self.bystander_margin + max_dist_to_team - min_dist_to_bystanders)
        assassin_loss = F.relu(self.assassin_margin + max_dist_to_team - dist_to_assassin)

        loss = 1 / n_words_to_guess * (enemy_loss + bystander_loss) + assassin_loss + dist_to_team.mean() 

        return loss

steps = 5000 
emb_dim = 300# 50, 100, 200 or 300
lr = 0.01
enemy_margin, bystander_margin, assassin_margin = 0.05, 0.01, 0.1

model = SpyMasterModel(emb_dim)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {num_params}")
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
loss_function = ClueLoss(enemy_margin, bystander_margin, assassin_margin)
#word_embeddings = GloVe(name="6B", dim=emb_dim)
#word_embeddings = GloVe(name="twitter.27B", dim=emb_dim)
word_embeddings = GloVe(name="42B", dim=emb_dim)
deck = load_deck()

embedded_deck = words2vec(word_embeddings, deck)
deck_numpy = np.array(deck, dtype="object")

model.train()

mean_losses = []
min_losses = []
max_losses = []

for step in range(steps):
    print(f"Step {step + 1:05}/{steps:05}", end="")
    # Generate random game position
    position = generate_random_position(deck_numpy, embedded_deck)
    x_e, x_b, x_a = (
        position["enemy_words_emb"],
        position["bystanders_words_emb"],
        position["assassin_word_emb"],
    )

    # Generate clue for every subset of the team's words.
    n_team_words = len(position["team_words_emb"])
    step_losses = []
    for j in range(1, n_team_words):
        for combination in itertools.combinations(np.arange(n_team_words), r=(j + 1)):
            # TODO: Maybe many of the examples are too difficult
            # For each n, pick the k examples with smallest diameter
            # Start with k=1 so consider only the two closest words, three closest words and so on.
            idx = list(combination)
            #words = position["team_words"][idx]
            x_t = position["team_words_emb"][idx]
            optimizer.zero_grad()
            c_t = model(x_t, x_e, x_b, x_a)
            loss = loss_function(c_t, x_t, x_e, x_b, x_a)
            loss.backward()
            optimizer.step()
            step_losses.append(loss.item())
            #print(c_t)
    mean_step_loss = np.mean(step_losses)
    min_step_loss = np.min(step_losses)
    max_step_loss = np.max(step_losses)
    mean_losses.append(mean_step_loss)
    min_losses.append(min_step_loss)
    max_losses.append(max_step_loss)
    print(f"\tMean = {mean_step_loss:.4f}\tMin = {min_step_loss:.4f}\tMax = {max_step_loss:.4f}")

import matplotlib.pyplot as plt

plt.plot(min_losses)
plt.plot(max_losses)
plt.plot(mean_losses)
plt.show()
