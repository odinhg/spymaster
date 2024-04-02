# Word embedding configuration
emb_name = "42B"
emb_dim = 300
deck_filename = "codename_vocab.txt"
vocab_filename = "english_vocab.txt"
hidden_dim = 100

# Model hyperparameters
max_n_positive_words = 6
difficulties = [0, 1, 2, 4, 8, 16, 32, 64, 128]
max_n_negative_words = 6#17

# Training configuration
steps = 1000 
batch_size = 256 
lr = 0.001
checkpoint_filename = "spymaster.pt"
