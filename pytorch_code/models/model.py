import os
import numpy as np

import torch
import torch.nn as nn

from .transformers import TransformerEncoder, TransformerEncoderLayer

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(65370, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.4),
            
            nn.Linear(512, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.4),
        )
        self.decoder = None
        
        self.regressor = nn.Linear(200, 300)
        self.classifier = nn.Linear(300, 180)
        
    def forward(self, x):
        latent = self.encoder(x)
        
        # recon_x = self.decoder(latent)
        reg_out = self.regressor(latent)
        y_pred = self.classifier(reg_out)
        
        return reg_out, y_pred

class ROIBaseline(nn.Module):
    def __init__(self, num_rois=333):
        super().__init__()
        
        self.num_rois = num_rois
        
        self.sizes = np.load(str(os.path.dirname(os.path.abspath(__file__))) + '../data/look_ups/sizes.npy')
        self.reduced = np.load(str(os.path.dirname(os.path.abspath(__file__))) + '../data/look_ups/reduced.npy')
        
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
        
        outputs = []
        
        index = 0
        for i in range(self.num_rois):
            new_index = index + self.sizes[i]
            outputs.append(self.encoder[i](x[index:new_index]))
            index = new_index
        
        concat_out = torch.cat(outputs)     
        
        reg_out = self.regressor(self.dropout(concat_out))
        y_pred = self.softmax(self.classifier(reg_out))
        
        return concat_out, reg_out, y_pred
        
    
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        pass