"""
@Huy Anh Nguyen, 113094662
Created on Nov 18, 2021 (moved from jupyter notebook)
Last modified on Nov 18, 2021
"""
import torch
import torch.nn as nn
from model.GRU import GRUModel
from model.LSTM import LSTMModel
from model.CNN import CNNModel

class FastTextModel(nn.Module):
    def __init__(self, num_class, model = "GRU", max_sequence_length=32):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.model_type = model
        if model == "GRU":
            self.model = GRUModel(num_class, 300, 300, max_sequence_length=max_sequence_length)
        elif model == "LSTM":
            self.model = LSTMModel(num_class, 300, 300, max_sequence_length=max_sequence_length)
        elif model == "CNN":
            self.model = CNNModel(num_class, 300)

    def forward(self, seq, mask):
        if self.model_type in ["GRU", "LSTM"]:
            seq = nn.utils.rnn.pack_padded_sequence(seq, mask, batch_first=True, enforce_sorted=False)
            
        x = self.model(seq)

        return x