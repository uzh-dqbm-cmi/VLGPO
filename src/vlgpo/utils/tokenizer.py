import numpy as np
import torch

class Encoder:
    """
    Encoder class to convert between amino acid sequences and their one-hot representations.
    """
    def __init__(self, alphabet: str = 'ARNDCQEGHILKMFPSTWYV'):
        self.alphabet = alphabet
        self.a_to_t = {a: i for i, a in enumerate(self.alphabet)}  # Amino acid to token
        self.t_to_a = {i: a for i, a in enumerate(self.alphabet)}  # Token to amino acid

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet)

    @property
    def vocab(self) -> np.ndarray:
        return np.array(list(self.alphabet))

    @property
    def tokenized_vocab(self) -> np.ndarray:
        return np.array([self.a_to_t[a] for a in self.alphabet])

    def onehotize(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of tokenized sequences into one-hot encoded format.

        Args:
            batch (torch.Tensor): Tensor of tokenized sequences.

        Returns:
            torch.Tensor: One-hot encoded representation.
        """
        onehot = torch.zeros(batch.size(0), self.vocab_size)
        onehot.scatter_(1, batch.unsqueeze(1), 1)
        return onehot

    def encode(self, seq_or_batch, return_tensor=True):
        """
        Encode a sequence or batch of sequences into tokenized format.

        Args:
            seq_or_batch (str or list): Sequence or list of sequences.
            return_tensor (bool): Whether to return a PyTorch tensor (default=True).

        Returns:
            torch.Tensor or list: Tokenized sequence(s).
        """
        if isinstance(seq_or_batch, str):
            encoded = [self.a_to_t[a] for a in seq_or_batch]
        else:
            encoded = [[self.a_to_t[a] for a in seq] for seq in seq_or_batch]

        return torch.tensor(encoded) if return_tensor else encoded

    def decode(self, x) -> str or list:
        """
        Decode tokenized sequences back to amino acid sequences.

        Args:
            x (np.ndarray, list, or torch.Tensor): Tokenized sequence(s).

        Returns:
            str or list: Decoded amino acid sequence(s).
        """
        if isinstance(x, (np.ndarray, torch.Tensor)):
            x = x.tolist()

        if isinstance(x[0], list):
            return [''.join(self.t_to_a[t] for t in seq) for seq in x]
        else:
            return ''.join(self.t_to_a[t] for t in x)
