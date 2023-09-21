import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class HearthDataset(Dataset):
    def __init__(self,split = "train",path = "",resolution = (128,128)):
        super().__init__()
        self.resolution = resolution
        self.path = "/Users/melkor/Documents/datasets/heathstone/{}.jpg"
        #self.images = sorted(glob(self.path))
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __len__(self):
        return 430

    def __getitem__(self, index):
        image = Image.open(self.path.format(index))
        image = image.convert("RGB").resize(self.resolution)
        image = self.img_transform(image).permute(1,2,0)
        sample = {"image":image}
        return sample