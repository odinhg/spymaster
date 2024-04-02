import torch
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt

from layers import SpyMaster
from batching import BatchGenerator
from clueloss import ClueLoss
import config

batch_generator = BatchGenerator(emb_dim=config.emb_dim, emb_name=config.emb_name, deck_filename=config.deck_filename)
model = SpyMaster(emb_dim=config.emb_dim, hidden_dim=config.hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
loss_function = ClueLoss()
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"# Parameters: {num_params}")

hyperparams = itertools.product(
    range(2, config.max_n_positive_words + 1),
    config.difficulties,
    range(1, config.max_n_negative_words + 1),
)
n_params = (
    (config.max_n_positive_words - 1) * len(config.difficulties) * config.max_n_negative_words
)

train_losses = []
model.train()

for i, (n_positive, max_difficulty, n_negative) in enumerate(hyperparams):
    print(
        f"[{i + 1:04}/{n_params:04}] {n_positive} positive words of max difficulty {max_difficulty} and {n_negative} negative words."
    )
    for step in (pbar := tqdm(range(config.steps))):
        batch_positive, batch_negative = batch_generator.get_batch(
            config.batch_size, n_positive, max_difficulty, n_negative
        )
        optimizer.zero_grad()
        clues = model(batch_positive, batch_negative)
        loss = loss_function(clues, batch_positive, batch_negative)
        loss.backward()
        train_losses.append(loss.item())
        pbar.set_description(f"ClueLoss: {loss.item():.4f}")
        optimizer.step()

# Save model weights
torch.save(model.state_dict(), config.checkpoint_filename)

# Plot training loss
plt.plot(train_losses)
plt.show()
