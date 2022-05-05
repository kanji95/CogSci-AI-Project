
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import Normalizer


class FmriDataset(Dataset):
    def __init__(self, data_root="/ssd_scratch/cvit/kanishk/simplified_data", split="train"):
        
        self.data_root = data_root
        
        self.data_fine = np.load(f'{data_root}/data_fine_{split}.npy')
        self.glove_fine = np.load(f'{data_root}/glove_fine_{split}.npy')
        self.class_fine = np.load(f'{data_root}/class_fine_{split}.npy')

        # print(self.data_fine)
        print(self.class_fine)
    
    def __len__(self):
        return len(self.data_fine)

    def __getitem__(self, idx):
        fmri_scan = self.data_fine[idx]
        glove_emb = self.glove_fine[idx]
        word_label = self.class_fine[idx]
        
        transformer = Normalizer().fit(fmri_scan)
        fmri_scan = transformer.transform(fmri_scan)
        
        return torch.tensor(fmri_scan).float(), glove_emb, word_label
