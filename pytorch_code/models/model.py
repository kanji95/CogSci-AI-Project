import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.LeakyReLU(0.3),
            nn.BatchNorm1d(300),
            nn.Dropout(0.4)
        )
        self.decoder = None
        
        # self.regressor = nn.Sequential(
        #     nn.Linear(200, 300),
        #     nn.BatchNorm1d(300),                                                                                                                                                                                                                                               
        #     nn.LeakyReLU(0.3),
        #     nn.Dropout(0.4),
        # )
        self.classifier = nn.Linear(300, 180)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):

        # x = F.normalize(x, p=2, dim=1)
        latent = self.encoder(x)
        
        # recon_x = self.decoder(latent)
        # reg_out = self.relu(self.regressor(latent))
        # reg_out = self.regressor(latent)
        y_pred = self.softmax(self.classifier(latent))
        
        return y_pred

class ROIBaseline(nn.Module):
    def __init__(self, num_rois=333):
        super().__init__()
        
        self.num_rois = num_rois
        
        self.sizes = np.load('/home/kanishk/cogsci_project/pytorch_code/data/look_ups/sizes.npy')
        self.reduced = np.load('/home/kanishk/cogsci_project/pytorch_code/data/look_ups/reduced_sizes.npy')
        
        linear_layers = []
        concat_dim = 0
        for i in range(self.sizes.shape[0]):
            linear_layers.append(
                nn.Sequential(
                    nn.Linear(self.sizes[i], self.reduced[i]),
                    nn.BatchNorm1d(self.reduced[i]),
                    nn.LeakyReLU(0.3)
                )
            )
            concat_dim += self.reduced[i]
            
        self.encoder = nn.ModuleList(linear_layers)
        
        self.regressor = nn.Linear(concat_dim, 300)
        self.classifier = nn.Linear(300, 180)
        
        self.dropout = nn.Dropout(0.4)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        
        # x = F.normalize(x, p=2, dim=1)

        outputs = []

        if random.random() < 0.05:
            import pdb; pdb.set_trace()
        
        index = 0
        for i in range(self.num_rois):
            new_index = index + self.sizes[i]
            outputs.append(self.encoder[i](x[:, index:new_index]))
            index = new_index
        
        concat_out = torch.cat(outputs, dim=-1)     
        
        reg_out = self.regressor(self.dropout(concat_out))
        y_pred = self.softmax(self.classifier(reg_out))
        
        return y_pred
        # return concat_out, reg_out, y_pred
        
    
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        pass
