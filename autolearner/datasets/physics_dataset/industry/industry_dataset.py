import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

class IndustryDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dataset_root = config.dataset_root

    def __len__(self):return 1

    def __getitem__(self, idx):
        return idx