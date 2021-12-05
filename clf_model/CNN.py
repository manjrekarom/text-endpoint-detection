"""
@Huy Anh Nguyen, 113094662
Created on Nov 18, 2021 (moved from jupyter notebook)
Last modified on Nov 18, 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, num_class=2, embedding_dim=300, filter_sizes = [2,3,4,5], num_filters=[100, 100, 100, 100], stride=1, dropout=0.3):
        super().__init__()

        self.conv1d_list = nn.ModuleList([nn.Conv1d(
        in_channels=embedding_dim,
        out_channels=num_filters[i],
        kernel_size=filter_sizes[i],
        stride=stride) 
        for i in range(len(filter_sizes))])

        self.output = nn.Linear(sum(num_filters), num_class)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence):
        x = torch.swapaxes(sequence, 1, 2)
        list_x = [F.relu(conv1d(x)) for conv1d in self.conv1d_list]
        list_x = [F.max_pool1d(x, kernel_size=x.shape[2]) for x in list_x]
        x = torch.cat([x.squeeze(dim=2) for x in list_x], dim=1)
        x = self.output(self.dropout(x))

        return x