import os
import random
import numpy as np
from pandas import concat

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .transformers import TransformerEncoder, TransformerEncoderLayer


class Baseline(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(65730, 2000),
            nn.LeakyReLU(0.3),
            nn.BatchNorm1d(2000),
            nn.Dropout(0.4),
            nn.Linear(2000, 300),
        )
        self.decoder = None

        self.classifier = nn.Linear(300, 180)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        latent = self.encoder(x)

        y_pred = self.softmax(self.classifier(latent))

        return latent, y_pred


class ROIBaseline(nn.Module):
    def __init__(self, num_rois=333):
        super().__init__()

        self.num_rois = num_rois

        self.sizes = np.load(
            "/home/kanishk/cogsci_project/pytorch_code/data/look_ups/sizes.npy"
        )
        self.reduced = np.load(
            "/home/kanishk/cogsci_project/pytorch_code/data/look_ups/reduced_sizes.npy"
        )

        linear_layers = []
        concat_dim = 0
        for i in range(self.sizes.shape[0]):
            linear_layers.append(
                nn.Sequential(
                    nn.Linear(self.sizes[i], self.reduced[i]),
                    nn.BatchNorm1d(self.reduced[i]),
                    nn.LeakyReLU(0.3),
                )
            )
            concat_dim += self.reduced[i]

        self.encoder = nn.ModuleList(linear_layers)

        self.regressor = nn.Sequential(
            nn.Linear(concat_dim, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.3),
            nn.Linear(1024, 300), 
        )
        self.classifier = nn.Linear(300, 180)

        self.dropout = nn.Dropout(0.4)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        outputs = []

        index = 0
        for i in range(self.num_rois):
            new_index = index + self.sizes[i]
            outputs.append(self.encoder[i](x[:, index:new_index]))
            index = new_index

        concat_out = torch.cat(outputs, dim=-1)

        reg_out = self.regressor(self.dropout(concat_out))
        y_pred = self.softmax(self.classifier(reg_out))

        return reg_out, y_pred


class SelfAttnROI(nn.Module):
    def __init__(self, num_rois=333):
        super().__init__()
        
        self.num_rois = num_rois

        self.sizes = np.load(
            "/home/kanishk/cogsci_project/pytorch_code/data/look_ups/sizes.npy"
        )
        self.reduced = np.load(
            "/home/kanishk/cogsci_project/pytorch_code/data/look_ups/reduced_sizes.npy"
        )

        linear_layers = []
        concat_dim = 0
        for i in range(self.sizes.shape[0]):
            linear_layers.append(
                nn.Sequential(
                    nn.Linear(self.sizes[i], self.reduced[i]),
                    nn.BatchNorm1d(self.reduced[i]),
                    nn.LeakyReLU(0.3),
                )
            )
            concat_dim += self.reduced[i]

        # print(concat_dim)

        self.encoder = nn.ModuleList(linear_layers)
        
        self.multi_head_attn = nn.MultiheadAttention(concat_dim+3, 8)
        
        self.regressor = nn.Sequential(
            nn.Linear(concat_dim+3, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.1),
            nn.Linear(1024, 300), 
        )
        self.classifier = nn.Linear(300, 180)

        self.dropout = nn.Dropout(0.2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        
        outputs = []

        index = 0
        for i in range(self.num_rois):
            new_index = index + self.sizes[i]
            outputs.append(self.encoder[i](x[:, index:new_index]))
            index = new_index

        concat_out = torch.cat(outputs, dim=-1)
        concat_out = F.pad(concat_out, (1, 2), "constant", 0)
        concat_out = concat_out.unsqueeze(0)
        
        attn_output, _ = self.multi_head_attn(concat_out, concat_out, concat_out)
        
        reg_out = self.regressor(self.dropout(attn_output[0]))
        y_pred = self.softmax(self.classifier(reg_out))

        return reg_out, y_pred
