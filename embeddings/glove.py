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

class GloveModel(nn.Module):
    def __init__(self, num_class, embedding_matrix, model = "GRU", max_sequence_length=32):
        super().__init__()
        embedding_size = embedding_matrix.shape[-1]
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding.from_pretrained(embeddings=embedding_matrix, padding_idx=1)
        if model == "GRU":
            self.model = GRUModel(num_class, embedding_size, 300, max_sequence_length=max_sequence_length)
        elif model == "LSTM":
            self.model = LSTMModel(num_class, embedding_size, 300, max_sequence_length=max_sequence_length)
        elif model == "CNN":
            self.model = CNNModel(num_class, embedding_size)

    def forward(self, sequence):
        """
        Input: list of tokens
        Output: logits
        """
        x = self.embedding(sequence)
        x = self.model(x)

        return x