from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import os

class PTRData(Dataset):
    def __init__(self,split="train",resolution = (128,128)):
        super().__init__()
        assert split in ["train","test","val"]
        self.split = split
        self.root_dir = "/Users/melkor/Documents/datasets/ptr_data/PTR"
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )

        self.resolution = resolution
        self.file_names = os.listdir("/Users/melkor/Documents/datasets/ptr_data/PTR/{}/{}_images/".format(split,split))

        if split == "train": # 518110
            pass
            #self.ptr_data = load_json("/Users/melkor/Documents/datasets/ptr_data/PTR/train_questions.json")["questions"]
        elif split == "val":
            pass
            #self.ptr_data = load_json("/Users/melkor/Documents/datasets/ptr_data/PTR/val_questions.json")["questions"]
    def __getitem__(self,index): # 91720
        #data_bind = self.ptr_data[index]
        #idx = ("000000" + str(index))[-6:]
        image_file_name = os.path.join(self.root_dir,self.split,"{}_images".format(self.split),self.file_names[index])
        #question = data_bind['question']
        #program = data_bind["program"]
        #answer = data_bind["answer"]
        image = Image.open(image_file_name).convert()
        image = image.convert("RGB").resize(self.resolution) 
        image = self.img_transform(image).permute([1,2,0]) 
        #return torch.tensor(np.array(image)).float()/256
        return {"image":image * 1.0}#"question":question,"answer":answer,"program":program}

    def __len__(self):
        return len(self.file_names)