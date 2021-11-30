"""
@Huy Anh Nguyen, 113094662
Created on Nov 18, 2021 (moved from jupyter notebook)
Last modified on Nov 18, 2021
"""
import torch
import torch.nn as nn
from transformers import BertModel
from clf_model.GRU import GRUModel
from clf_model.LSTM import LSTMModel
from clf_model.CNN import CNNModel
class BERTEmbModel(nn.Module):
    def __init__(self, num_class, model = "GRU", max_sequence_length=32):
        super().__init__()
        
        self.max_sequence_length = max_sequence_length
        #self.embedding = BertModel.from_pretrained('bert-base-uncased').embeddings.word_embeddings
        self.embedding = BertModel.from_pretrained('bert-base-uncased').embeddings
        if model == "GRU":
            self.model = GRUModel(num_class, 768, 300, max_sequence_length=max_sequence_length)
        elif model == "LSTM":
            self.model = LSTMModel(num_class, 768, 300, max_sequence_length=max_sequence_length)
        elif model == "CNN":
            self.model = CNNModel(num_class, 768)

    def forward(self, sequence):
        """
        Input: list of tokens
        Output: logits
        """
        x = self.embedding(sequence)
        x = self.model(x)

        return x