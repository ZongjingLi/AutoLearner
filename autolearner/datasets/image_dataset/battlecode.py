import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


class BattlecodeImageData(Dataset):
    def __init__(self,split = "train",data_path = None, resolution = (128,128)):
        super().__init__()
        self.resolution = resolution
        
        assert split in ["train","val","test"]
        self.split = split
        self.root_dir = "/Users/melkor/Documents/datasets/battlecode2"
        self.files = os.listdir(os.path.join(self.root_dir,split))
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        #self.question_file = load_json(os.path.join(self.root_dir,"{}_sprite_qa.json".format(split)))

    def __len__(self): return 200#len(self.files) - 4000

    def __getitem__(self,index):
        index = index + 2
        image = Image.open(os.path.join(self.root_dir,self.split,"{}.jpg".format(index)))
        image = image.convert("RGB").resize(self.resolution)
        image = self.img_transform(image).permute([1,2,0])
        return {"image":image}