import os
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader

from PIL import Image
from utils import *

class SpriteWithQuestions(Dataset):
    def __init__(self,split = "train",resolution = (64,64)):
        super().__init__()
        
        assert split in ["train","val","test"]
        self.split = split
        self.resolution = resolution
        self.root_dir = "/Users/melkor/Documents/datasets/sprites"
        self.questions = load_json(os.path.join(self.root_dir,"sprite_qa.json"))
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )
    def __len__(self): return 500#len(self.files)

    def __getitem__(self,index):

        image = Image.open(os.path.join(self.root_dir,self.split,"{}_{}.png".format(self.split,index)))
        image = image.convert("RGB").resize(self.resolution) 
        image = self.img_transform(image).permute([1,2,0]) 

        #question = [q["question"] for q in self.questions[index]]
        programs  = [p["program"] for p in self.questions[index]]
        answers   = [a["answer"] for a in self.questions[index]]

        questions = []
        for i in range(len(programs)):
            questions.append({"program":programs[i], "answer":answers[i]})

        sample = {"image":image,"question":questions}
        return sample


class SpriteData(Dataset):
    def __init__(self,split = "train",resolution = (128,128)):
        super().__init__()
        
        assert split in ["train","val","test"]
        self.split = split
        self.resolution = resolution
        self.root_dir = "/Users/melkor/Documents/datasets/sprites"
        self.files = os.listdir(os.path.join(self.root_dir,split))
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )
    def __len__(self): return 100#len(self.files)

    def __getitem__(self,index):
        path = self.files[index]
        image = Image.open(os.path.join(self.root_dir,self.split,"{}_{}.png".format(self.split,index)))
        image = image.convert("RGB").resize(self.resolution) 
        image = self.img_transform(image).permute([1,2,0])
        sample = {"image":image}
        return sample