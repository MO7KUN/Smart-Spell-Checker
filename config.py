# src/config.py

# Shared parameters
VOCAB_SIZE = 48          # Set dynamically by Data Prep teammate
MAX_ENC_LEN = 25         # Max French word length
MAX_DEC_LEN = 27         # Max French word length + start/end tokens

# Architecture parameters (can be adjusted by Training teammate)
LATENT_DIM = 128         
EMBEDDING_DIM = 64

# Training parameters
BATCH_SIZE = 64
EPOCHS = 30