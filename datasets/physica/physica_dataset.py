import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import gym
import pygame 


class PhysicaDataset(Dataset):
    def __init__(self, config):
        super().__init__()

    