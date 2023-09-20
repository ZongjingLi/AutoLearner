import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image

class Clevr4(Dataset):
    def __init__(self, config):
        self.path = "/Users/melkor/Documents/datasets/clevr4/CLEVR_new_{}.png"
        #self.images = sorted(glob(self.path))
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.config = config

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        config = self.config
        index = "000000" + str(index)
        image = Image.open(self.path.format(index[-6:]))
        image = image.convert("RGB").resize([config.imsize,config.imsize])
        image = self.img_transform(image) 
        sample = {"image":image.permute([1,2,0])}
        return sample