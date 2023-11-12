import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import os

class PhysicaDataset(Dataset):
    def __init__(self,split = "train", root = None):
        super().__init__()
        if root is None:
            data_root = "/Users/melkor/Documents/datasets/physica"
        data_dir     = os.path.join(data_root, split)
        all_files = os.listdir(data_dir)
        all_indices = [int(a.split(".")[0]) for a in all_files]
        self.data_root = data_root
        self.size = max(all_indices)

    def __len__(self):return self.size - 1

    def __getitem__(self, idx):
        npfile = np.load(self.data_root + "/train/{}.npy".format(idx + 1), allow_pickle = True)
        states = []
        id_map = {"brick_dwarven":1.0,"planks2_chestnut":2.0,"unholy":3.0}
        for state in npfile:
            add_state = []
            for obj in state:
                obj[0] = id_map[obj[0]]
                add_state.append([float(a) for a in obj])
            states.append(add_state)
        return {"states":states}