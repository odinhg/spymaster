import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import numpy as np
import random
import itertools
from torchtext.vocab.vectors import GloVe
from tqdm import tqdm

from layers import SpyMasterModel

class BatchGenerator():
    def __init__(self, deck_filename: str = "codename_vocab.txt", emb_name: str= "42B", emb_dim: int = 300):
        self.emb_name = emb_name
        self.emb_dim = emb_dim
        with open(deck_filename, "r") as f:
            deck = f.read().lower().splitlines()
            self.preprocess_words(deck)
        self.deck = deck
        self.deck_numpy = np.array(deck, dtype="object")
        self.deck_size = len(deck)
        self.word_embedding = GloVe(name=emb_name, dim=emb_dim)
        self.deck_emb = self.word_embedding.get_vecs_by_tokens(self.deck, lower_case_backup=True)
        _, self.sorted_idx = torch.sort(torch.cdist(self.deck_emb, self.deck_emb), dim=1)

    def preprocess_words(self, words: list[str]):
        for i in range(len(words)):
            word = words[i]
            if " " in word:
                words[i] = word.split()[0]

    def get_batch(self, batch_size: int, n_t: int, max_difficulty: int) -> torch.Tensor:
        max_n_neighbours = n_t + max_difficulty
        centroids_idx = torch.randint(self.deck_size, (batch_size,))
        neighbours = self.sorted_idx[centroids_idx][:, 1:max_n_neighbours+1]
        neighbour_indices = torch.stack([torch.randperm(max_n_neighbours)[:n_t - 1] for _ in range(batch_size)])
        batch_idx = torch.cat([centroids_idx.unsqueeze(1), torch.gather(neighbours, 1, neighbour_indices)], dim=1)
        #batch_words = self.deck_numpy[batch_idx]
        batch_emb = self.deck_emb[batch_idx]
        return batch_emb # (batch_size, n_t, emb_dim)


class ClueLoss(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, clues, targets):
        #dist_to_team = torch.linalg.norm(targets - clues.unsqueeze(1), dim=2)
        dist_to_team = 1 - F.cosine_similarity(targets, clues.unsqueeze(1), dim=2)
        loss = dist_to_team.mean()
        return loss


steps = 400
batch_size = 128
lr = 0.01
max_n_target_words = 6
max_target_difficulty = 10

batch_generator = BatchGenerator()
model = SpyMasterModel(emb_dim=batch_generator.emb_dim)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {num_params}")
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = ClueLoss()
model.train()

train_losses = []

for n_t, max_difficulty in itertools.product(range(2, max_n_target_words + 1), range(0, max_target_difficulty + 1)):
    print(f"Training on {n_t} target words with max difficulty {max_difficulty}")
    for step in tqdm(range(steps)):
        #print(f"Step {step + 1:05}/{steps:05}", end="\t")
        targets = batch_generator.get_batch(batch_size, n_t, max_difficulty) 
        optimizer.zero_grad()
        clues = model(targets)
        loss = loss_function(clues, targets)
        loss.backward()
        train_losses.append(loss.item())
        #print(f"Loss: {loss.item()}")
        optimizer.step()

import matplotlib.pyplot as plt
plt.plot(train_losses)
plt.show()

# Inference
def load_vocabulary(word_embedding, vocab_filename: str="english_vocab.txt") -> list[str]:
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

#vocab, vocab_emb = load_vocabulary(batch_generator.word_embedding)
