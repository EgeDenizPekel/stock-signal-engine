"""
PyTorch LSTM model and rolling-window Dataset for binary classification.

Architecture
------------
- Input:  (batch, sequence_length, n_features)
- LSTM:   hidden_size=64, num_layers=2, dropout=0.2
- Output: single sigmoid unit → probability of label=1

Dataset
-------
SequenceDataset wraps a 2D feature matrix and produces overlapping windows
of length `sequence_length`. The label for each window is the target on the
last day of the window.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


SEQUENCE_LENGTH = 20
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2


class SequenceDataset(Dataset):
    """
    Sliding-window dataset for time-series classification.

    Given a feature matrix X of shape (T, n_features) and a label vector y
    of shape (T,), produces samples (window, label) where:
        window = X[i : i + sequence_length]   shape: (sequence_length, n_features)
        label  = y[i + sequence_length - 1]   the target on the last day of the window
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int = SEQUENCE_LENGTH,
    ) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = sequence_length

    def __len__(self) -> int:
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx: int):
        x_window = self.X[idx : idx + self.seq_len]          # (seq_len, n_features)
        label = self.y[idx + self.seq_len - 1]               # scalar
        return x_window, label


class LSTMClassifier(nn.Module):
    """
    Two-layer LSTM followed by a linear layer with sigmoid output.
    Predicts probability that the next-5-day return exceeds the threshold.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, sequence_length, input_size)

        Returns
        -------
        (batch,) — predicted probabilities in [0, 1]
        """
        lstm_out, _ = self.lstm(x)           # (batch, seq_len, hidden_size)
        last_hidden = lstm_out[:, -1, :]     # (batch, hidden_size) — last timestep
        logit = self.head(last_hidden)       # (batch, 1)
        return torch.sigmoid(logit).squeeze(1)  # (batch,)
