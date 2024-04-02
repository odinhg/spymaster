import torch
import torch.nn.functional as F
from torchtext.vocab.vectors import GloVe
import numpy as np
import random

from layers import SpyMaster
from clueloss import ClueLoss
import config

def load_vocabulary(
    word_embedding, vocab_filename: str
) -> list[str]:
    with open(vocab_filename, "r") as f:
        vocab = f.read().lower().splitlines()
    cleaned_vocab = []
    for word in vocab:
        word = word.strip()
        if len(word) <= 2 or word not in word_embedding.stoi:
            continue
        cleaned_vocab.append(word)
    cleaned_vocab = list(set(cleaned_vocab))
    vocab_emb = word_embedding.get_vecs_by_tokens(cleaned_vocab, lower_case_backup=True)
    return np.array(cleaned_vocab), vocab_emb

def get_random_words(deck_filename: str, n_positive: int, n_negative: int) -> tuple[list[str]]:
    with open(deck_filename, "r") as f:
        deck = f.read().lower().splitlines()
    words = random.sample(deck, n_positive + n_negative)
    return words[:n_positive], words[n_positive:]


word_embedding = GloVe(name=config.emb_name, dim=config.emb_dim)
vocab, vocab_emb = load_vocabulary(word_embedding, vocab_filename=config.vocab_filename)

model = SpyMaster(emb_dim=config.emb_dim, hidden_dim=config.hidden_dim)
model.load_state_dict(torch.load(config.checkpoint_filename))
model.eval()

loss_function = ClueLoss()
#loss_function.eval()

positive_words = ["apple"]
negative_words = ["europe"]
#positive_words, negative_words = get_random_words(config.deck_filename, 4, 9)

positive_emb = word_embedding.get_vecs_by_tokens(positive_words, lower_case_backup=True).unsqueeze(0)
negative_emb = word_embedding.get_vecs_by_tokens(negative_words, lower_case_backup=True).unsqueeze(0)

clue = model(positive_emb, negative_emb).squeeze(0)

loss = loss_function(clue.unsqueeze(0), positive_emb, negative_emb)
print(loss.item())

#dist_to_vocab = 1 - F.cosine_similarity(vocab_emb, clue.unsqueeze(0), dim=1)
dist_to_vocab = torch.linalg.norm(vocab_emb - clue.unsqueeze(0), dim=1)
_, closest_idx = torch.topk(dist_to_vocab, 10, largest=False)
clues = []
illegal_words = set(positive_words + negative_words)
for idx in closest_idx.detach().tolist():
    word = vocab[idx]
    if word in illegal_words:
        continue
    if any(word in positive_word or positive_word in word for positive_word in positive_words):
        continue
    if any(word in negative_word or negative_word in word for negative_word in negative_words):
        continue
    print(word)

print(f"Positive words: {positive_words}")
print(f"Negative words: {negative_words}")
