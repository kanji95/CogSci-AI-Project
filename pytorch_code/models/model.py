from turtle import forward
import torch
import torch.nn as nn


class Brain2Word(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.ecnoder = None
        self.decoder = None
        
        self.regressor = None
        self.classifier = None
        
    def forward(self, x):
        recon_x = None
        reg_out = None
        y_pred = None
        return recon_x, reg_out, y_pred