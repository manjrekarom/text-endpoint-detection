"""
@Huy Anh Nguyen, 113094662
Created on Nov 18, 2021 (moved from jupyter notebook)
Last modified on Nov 18, 2021
"""
import torch.nn as nn
from transformers import BertForSequenceClassification

class BERTModel(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_class, output_attentions=False, output_hidden_states=False)


    def forward(self, inputs):
        logits = self.model(**inputs)['logits']
        return logits