import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import numpy as np
import random
import itertools
from torchtext.vocab.vectors import GloVe
from tqdm import tqdm

from layers import SpyMaster


class BatchGenerator:
    def __init__(
        self,
        deck_filename: str = "codename_vocab.txt",
        emb_name: str = "42B",
        emb_dim: int = 300,
    ):
        self.emb_name = emb_name
        self.emb_dim = emb_dim
        with open(deck_filename, "r") as f:
            deck = f.read().lower().splitlines()
            self.preprocess_words(deck)
        self.deck = deck
        self.deck_numpy = np.array(deck, dtype="object")
        self.deck_size = len(deck)
        self.word_embedding = GloVe(name=emb_name, dim=emb_dim)
        self.deck_emb = self.word_embedding.get_vecs_by_tokens(
            self.deck, lower_case_backup=True
        )
        _, self.sorted_idx = torch.sort(
            torch.cdist(self.deck_emb, self.deck_emb), dim=1
        )

    def preprocess_words(self, words: list[str]):
        for i in range(len(words)):
            word = words[i]
            if " " in word:
                words[i] = word.split()[0]

    def get_batch(
        self, batch_size: int, n_positive: int, max_difficulty: int, n_negative: int
    ) -> torch.Tensor:
        max_n_neighbours = n_positive + max_difficulty
        # Each subset of team words starts with a single word
        centroids_idx = torch.randint(self.deck_size, (batch_size,))

        # Add words to the subsets depending on maximum difficulty
        neighbours = self.sorted_idx[centroids_idx][:, 1 : max_n_neighbours + 1]
        neighbour_indices = torch.stack(
            [
                torch.randperm(max_n_neighbours)[: n_positive - 1]
                for _ in range(batch_size)
            ]
        )
        positive_idx = torch.cat(
            [
                centroids_idx.unsqueeze(1),
                torch.gather(neighbours, 1, neighbour_indices),
            ],
            dim=1,
        )
        batch_positive = self.deck_emb[positive_idx]

        # Choose negative words
        unused_idx = self.sorted_idx[centroids_idx][:, max_n_neighbours + 1 :]
        random_idx = torch.stack(
            [torch.randperm(len(unused_idx))[:n_negative] for _ in range(batch_size)]
        )
        negative_idx = torch.gather(unused_idx, 1, random_idx)
        batch_negative = self.deck_emb[negative_idx]

        return batch_positive, batch_negative


class ClueLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = torch.tensor(margin)

    def forward(self, clues, positives, negatives):
        dist_to_positives = 1 - F.cosine_similarity(
            positives, clues.unsqueeze(1), dim=2
        )
        dist_to_negatives = 1 - F.cosine_similarity(
            negatives, clues.unsqueeze(1), dim=2
        )
        max_dist_to_positives, _ = dist_to_positives.max(-1)
        min_dist_to_negatives, _ = dist_to_negatives.min(-1)
        loss = dist_to_positives.mean() + F.relu(
            max_dist_to_positives.mean() - min_dist_to_negatives.mean() + self.margin
        )
        return loss


steps = 500 
batch_size = 256 
lr = 0.001
max_n_positive_words = 6
difficulties = [0, 1, 2, 4, 8, 16, 32]
max_n_negative_words = 17

batch_generator = BatchGenerator()
model = SpyMaster(emb_dim=batch_generator.emb_dim, hidden_dim=100)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {num_params}")
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = ClueLoss()
model.train()

train_losses = []

hyperparams = itertools.product(
    range(2, max_n_positive_words + 1),
    difficulties,
    range(1, max_n_negative_words + 1),
)
n_params = (
    (max_n_positive_words - 1) * len(difficulties) * max_n_negative_words
)

for i, (n_positive, max_difficulty, n_negative) in enumerate(hyperparams):
    print(
        f"[{i + 1:04}/{n_params:04}] Training on {n_positive} positive words with max difficulty {max_difficulty} and {n_negative} negative words."
    )
    for step in (pbar := tqdm(range(steps))):
        # print(f"Step {step + 1:05}/{steps:05}", end="\t")
        batch_positive, batch_negative = batch_generator.get_batch(
            batch_size, n_positive, max_difficulty, n_negative
        )
        optimizer.zero_grad()
        clues = model(batch_positive, batch_negative)
        loss = loss_function(clues, batch_positive, batch_negative)
        loss.backward()
        train_losses.append(loss.item())
        pbar.set_description(f"Loss: {loss.item():.4f}")
        optimizer.step()

torch.save(model.state_dict(), "spymaster.pt")

import matplotlib.pyplot as plt

plt.plot(train_losses)
plt.show()


# Inference
def load_vocabulary(
    word_embedding, vocab_filename: str = "english_vocab.txt"
) -> list[str]:
    with open(vocab_filename, "r") as f:
        vocab = f.read().lower().splitlines()
    cleaned_vocab = []
    for word in vocab:
        word = word.strip()
        if len(word) <= 2 or word not in word_embedding.stoi:
            continue
        cleaned_vocab.append(word)
    vocab_emb = word_embedding.get_vecs_by_tokens(cleaned_vocab, lower_case_backup=True)
    return cleaned_vocab, vocab_emb


# vocab, vocab_emb = load_vocabulary(batch_generator.word_embedding)
