"""
@Huy Anh Nguyen, 113094662
Created on Nov 18, 2021 (moved from jupyter notebook)
Last modified on Nov 18, 2021
"""

from torch.utils.data import Dataset

class SentenceDataset(Dataset):
    def __init__(self, sentences, labels=None):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        if self.labels is None:
            return self.sentences[idx]
        else:
            return self.sentences[idx], self.labels[idx]