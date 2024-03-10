import torch
from torch.utils.data import Dataset

class JokeDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        # Total number of sequences
        return len(self.tokens) - (self.seq_len + 1)

    def __getitem__(self, idx):
        # get sequence of tokens starting at index
        sequence = self.tokens[idx:idx + self.seq_len]
        next_token = self.tokens[idx + self.seq_len]
        # Return X and y
        return torch.tensor(sequence), torch.tensor(next_token)