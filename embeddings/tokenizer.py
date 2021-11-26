"""
@Huy Anh Nguyen, 113094662
Created on Nov 18, 2021 (moved from jupyter notebook)
Last modified on Nov 18, 2021
"""
import numpy as np
import spacy
import torch

class SpacyTokenizer():
    def __init__(self, model="en_core_web_sm", word_to_idx=None, unk_idx=0, padding_idx=1, max_sequence_length=32):
        self.tokenizer = spacy.load(model)
        if word_to_idx is not None:
            self.word_to_idx = word_to_idx
            self.padding_idx = padding_idx
            self.unk_idx = unk_idx
        self.max_sequence_length = max_sequence_length

    def __call__(self, text):
        # Tokenizer and convert to list of indices
        # Invoke batch mode if the input is not string type
        if type(text) is str:
            return self.get_idx(text)
        else:
            return self.get_idx_batch(text)
            
    def tokenize(self, text):
        return [token.text for token in self.tokenizer(text)]

    def tokenize_batch(self, list_text):
        res = []
        for text in list_text:
            res.append(self.tokenize(text))

        return res

    def get_idx(self, text):
        assert self.word_to_idx is not None, "Method's not supported for default Spacy"
        tokens = self.tokenize(text)
        idx = []
        for token in tokens:
            if token in self.word_to_idx:
                idx.append(self.word_to_idx[token])
            else:
                idx.append(self.unk_idx)

        if len(idx) < self.max_sequence_length:
            idx += [self.padding_idx]*(self.max_sequence_length - len(idx))

        else:
            idx = idx[:self.max_sequence_length]

        return torch.tensor(idx)

    def get_idx_batch(self, list_text):
        assert self.word_to_idx is not None, "Method's not supported for default Spacy"
        res = []
        for text in list_text:
            res.append(self.get_idx(text))

        return torch.stack(res)