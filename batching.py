import torch
import numpy as np
from torchtext.vocab.vectors import GloVe

class BatchGenerator:
    def __init__(
        self,
        deck_filename: str,
        emb_name: str,
        emb_dim: int,
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

