"""
@Huy Anh Nguyen, 113094662
Created on Nov 18, 2021 (moved from jupyter notebook)
Last modified on Nov 18, 2021
"""

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, num_class, embedding_dim, hidden_dim, max_sequence_length=32):
        # stacked Bi-GRU
        super().__init__()
        self.LSTMs = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        if self.LSTMs.bidirectional:
            linear_dim = 2*hidden_dim
        else:
            linear_dim = hidden_dim

        self.linear = nn.Linear(linear_dim, 512, bias=True)
        self.output = nn.Linear(512, num_class, bias=True)

    #def forward(self, sequence, mask):
    def forward(self, sequence):
        """
        Expected sequence is a tensor of indices (N, max_seq_length)
        Output: (N, max_seq_length, embedding_dim)
        """

        _, (sen_rep, _) = self.LSTMs(sequence)
        sen_rep = torch.swapaxes(sen_rep, 0, 1)
        if self.LSTMs.bidirectional:
            x = sen_rep[:, -2:, :]
        else:
            x = sen_rep[:, -1, :]

        x = torch.flatten(x, 1, -1)
        x = self.linear(x)
        x = nn.ReLU()(x)
        x = self.output(x)

        return x