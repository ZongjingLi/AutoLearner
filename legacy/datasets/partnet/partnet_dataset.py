'''
 # @ Author: Yiqi Sun
 # @ Create Time: 2023-03-14 13:31:26
 # @ Modified by: Yiqi Sun
 # @ Modified time: 2023-03-14 13:32:10
 # @ Description: This file is distributed under the MIT license.
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class PartNet(Dataset):
    def __init__(self,split = "train",data_path = None):
        super().__init__()
        
        assert(split in ["train","test","val"]),print("Unknown split for the dataset: {}.".format(split))
        
        self.split = split
        self.root_dir = ""

    def __getitem__(self,index):
        return torch.zeros([128,128,4])

    def __len__(self):return 6

if __name__ == "__main__":
    import h5py
    filename = "/Users/melkor/Documents/datasets/sem_seg_h5/Table-2/train-00.h5"

    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]

        # get the object type for a_group_key: usually group or dataset
        print(type(f[a_group_key])) 

        # If a_group_key is a group name, 
        # this gets the object names in the group and returns as a list
        data = list(f[a_group_key])

        # If a_group_key is a dataset name, 
        # this gets the dataset values and returns as a list
        data = list(f[a_group_key])
        # preferred methods to get dataset values:
        ds_obj = f[a_group_key]      # returns as a h5py dataset object
        ds_arr = f[a_group_key][()]  # returns as a numpy array
        print("datashape",f["data"].shape)