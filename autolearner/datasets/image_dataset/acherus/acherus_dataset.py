'''
 # @ Author: Yiqi Sun
 # @ Create Time: 2023-03-14 13:31:26
 # @ Modified by: Yiqi Sun
 # @ Modified time: 2023-03-14 13:32:10
 # @ Description: This file is distributed under the MIT license.
'''
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
from utils import *

class AcherusImageDataset(Dataset):
    def __init__(self,split = "train",path = "",resolution = (128,128)):
        super().__init__()
        self.resolution = resolution
        self.path = "/Users/melkor/Documents/datasets/acherus/train/{}.jpg"
        #self.images = sorted(glob(self.path))
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.file_names = os.listdir("/Users/melkor/Documents/datasets/acherus/{}/".format(split,split))


    def __len__(self):
        return 399

    def __getitem__(self, index):
        image = Image.open(self.path.format(index))
        image = image.convert("RGB").resize(self.resolution)
        image = self.img_transform(image)
        sample = {"image":image.permute(1,2,0)}
        return sample

class AcherusDataset(Dataset):
    def __init__(self, split = "train", resolution = (128,128)):
        super().__init__()
        self.resolution = resolution
        self.path = "/Users/melkor/Documents/datasets/acherus/train/{}.jpg"
        #self.images = sorted(glob(self.path))
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.file_names = os.listdir("/Users/melkor/Documents/datasets/acherus/{}/".format(split,split))

        self.questions = load_json("/Users/melkor/Documents/datasets/acherus/{}_questions.json".format(split))

    def __len__(self): return 399

    def __getitem__(self, index):
        image = Image.open(self.path.format(index))
        image = image.convert("RGB").resize(self.resolution)
        image = self.img_transform(image)

        sample = {"image":image.permute(1,2,0),"question":self.questions[index]}
        return sample
