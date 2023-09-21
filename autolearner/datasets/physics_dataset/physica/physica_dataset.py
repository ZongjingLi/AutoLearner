import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import gym
import pygame 


class PhysicaDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config

        dataset_root = config.dataset_root + "physica"

        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # states: [T,N,D] [T,N,N,D]
        return {"states":self.data[idx]}