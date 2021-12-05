"""
@Huy Anh Nguyen, 113094662
Created on Nov 29, 2021 
Last modified on Nov 29, 2021
"""


import torch
import torch.nn as nn
from transformers import BertTokenizer, BertLMHeadModel, BertConfig

class BertSeq(nn.Module):
    def __init__(self, word2idx):
        super().__init__()
        config = BertConfig.from_pretrained("bert-base-uncased")
        config.is_decoder = True

        self.model = BertLMHeadModel.from_pretrained('bert-base-uncased', config=config)

        in_features = self.model.cls.predictions.decoder.in_features
        out_features = len(word2idx)

        self.model.cls.predictions.decoder = nn.Linear(in_features, out_features, bias=True)

    def forward(self, input_seq):
        outputs = self.model(**input_seq)
        return outputs.logits