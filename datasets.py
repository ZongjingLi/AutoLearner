import functools
import tensorflow as tf
import numpy as np
from utils import *
# Modified from https://github.com/deepmind/multi_object_datasets to
# work in TensorFlow 2

COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
IMAGE_SIZE = [64, 64]
# The maximum number of foreground and background entities in each variant
# of the provided datasets. The values correspond to the number of
# segmentation masks returned per scene.
MAX_NUM_ENTITIES = {
    'train': 7,
    'six_objects': 10,
    'empty_room': 4,
    'identical_color': 10
}
BYTE_FEATURES = ['mask', 'image']


def feature_descriptions(max_num_entities):
  """Create a dictionary describing the dataset features.
  Args:
    max_num_entities: int. The maximum number of foreground and background
      entities in each image. This corresponds to the number of segmentation
      masks returned per scene.
  Returns:
    A dictionary which maps feature names to `tf.Example`-compatible shape and
    data type descriptors.
  """
  return {
      'image': tf.io.FixedLenFeature(IMAGE_SIZE+[3], tf.string),
      'mask': tf.io.FixedLenFeature([max_num_entities]+IMAGE_SIZE+[1], tf.string),
  }


def _decode(example_proto, features):
  # Parse the input `tf.Example` proto using a feature description dictionary.
  single_example = tf.io.parse_single_example(example_proto, features)
  for k in BYTE_FEATURES:
    single_example[k] = tf.squeeze(tf.io.decode_raw(single_example[k], tf.uint8),
                                   axis=-1)
  return single_example


def dataset(tfrecords_path, dataset_variant, read_buffer_size=None,
            map_parallel_calls=None):
  """Read, decompress, and parse the TFRecords file.
  Args:
    tfrecords_path: str. Path to the dataset file.
    dataset_variant: str. One of ['train', 'six_objects', 'empty_room',
      'identical_color']. This is used to identify the maximum number of
      entities in each scene. If an incorrect identifier is passed in, the
      TFRecords file will not be read correctly.
    read_buffer_size: int. Number of bytes in the read buffer. See documentation
      for `tf.data.TFRecordDataset.__init__`.
    map_parallel_calls: int. Number of elements decoded asynchronously in
      parallel. See documentation for `tf.data.Dataset.map`.
  Returns:
    An unbatched `tf.data.TFRecordDataset`.
  """
  if dataset_variant not in MAX_NUM_ENTITIES:
    raise ValueError('Invalid `dataset_variant` provided. The supported values'
                     ' are: {}'.format(list(MAX_NUM_ENTITIES.keys())))
  max_num_entities = MAX_NUM_ENTITIES[dataset_variant]
  raw_dataset = tf.data.TFRecordDataset(
      tfrecords_path, compression_type=COMPRESSION_TYPE,
      buffer_size=read_buffer_size)
  features = feature_descriptions(max_num_entities)
  partial_decode_fn = functools.partial(_decode, features=features)
  return raw_dataset.map(partial_decode_fn,
                         num_parallel_calls=map_parallel_calls)

import torch
import torchvision
from torchvision import transforms

from torch.utils.data import Dataset,DataLoader

from PIL import Image

import os

class Tetrominoes(Dataset):
    def __init__(self,mode = "images"):
        super().__init__()
        self.path = "/Users/melkor/Documents/datasets/tetrominoes"
        self.resolution = [35,35]
        self.resolution = [64,64]
        self.mode = mode

        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __len__(self): return 200
        #return len(self.question_file)

    def __getitem__(self,index):
        image = Image.open(os.path.join(self.path,self.mode,"{}.png".format(index)))
        image = image.convert("RGB").resize(self.resolution) 
        image = self.img_transform(image) * 255.0

        return {"image":image.permute([1,2,0])}



class Clevr4(Dataset):
    def __init__(self, stage=0,path = "/Users/melkor/Documents/datasets/clevr4/CLEVR_new_{}.png"):
        super().__init__()
        self.path = path
        #self.images = sorted(glob(self.path))
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __len__(self):
        return 200

    def __getitem__(self, index):
        index = "000000" + str(index)
        image = Image.open(self.path.format(index[-6:]))
        image = image.convert("RGB").resize([64,64])
        image = self.img_transform(image)
        sample = {"image":image.permute([1,2,0]) * 255}
        return sample

class BattlecodeImageData(Dataset):
    def __init__(self,split = "train",data_path = None):
        super().__init__()
        
        assert split in ["train","val","test"]
        self.split = split
        self.root_dir = "/Users/melkor/Documents/datasets/battlecode2"
        self.files = os.listdir(os.path.join(self.root_dir,split))
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        #self.question_file = load_json(os.path.join(self.root_dir,"{}_sprite_qa.json".format(split)))

    def __len__(self): return 2#len(self.files) - 4000

    def __getitem__(self,index):
        index = index + 2
        image = Image.open(os.path.join(self.root_dir,self.split,"{}.jpg".format(index)))
        image = image.convert("RGB").resize([64,64])
        image = self.img_transform(image).permute([1,2,0]) *256
        return {"image":image}

  
class SpriteData(Dataset):
    def __init__(self,split = "train",data_path = None):
        super().__init__()
        
        assert split in ["train","val","test"]
        self.split = split
        self.root_dir = "/Users/melkor/Documents/datasets/sprites"
        self.files = os.listdir(os.path.join(self.root_dir,split))
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )
    def __len__(self): return 100#len(self.files)

    def __getitem__(self,index):
        path = self.files[index]
        image = Image.open(os.path.join(self.root_dir,self.split,"{}_{}.png".format(self.split,index)))
        image = image.convert("RGB").resize([64,64]) 
        image = self.img_transform(image).permute([1,2,0]) * 255
        sample = {"image":image}
        return sample

class SpriteQA(Dataset):
    def __init__(self,split = "train",data_path = None):
        super().__init__()
        
        assert split in ["train","val","test"]
        self.split = split
        self.root_dir = "/Users/melkor/Documents/datasets/sprites"
        self.questions = load_json(os.path.join(self.root_dir,"sprite_qa.json"))
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )
    def __len__(self): return 500#len(self.files)

    def __getitem__(self,index):

        image = Image.open(os.path.join(self.root_dir,self.split,"{}_{}.png".format(self.split,index)))
        image = image.convert("RGB").resize([64,64]) 
        image = self.img_transform(image).permute([1,2,0]) * 255

        question = [q["question"] for q in self.questions[index]]
        programs  = [p["program"] for p in self.questions[index]]
        answers   = [a["answer"] for a in self.questions[index]]

        sample = {"image":image,"question":question,"program":programs,"answer":answers}
        return sample
    
class MixSpriteData(Dataset):
    def __init__(self,split = "train",data_path = None):
        super().__init__()
        
        assert split in ["train","val","test"]
        self.split = split
        self.root_dir = "/Users/melkor/Documents/datasets/sprites_mix"

        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )
    def __len__(self): return 500#len(self.files)

    def __getitem__(self,index):

        image = Image.open(os.path.join(self.root_dir,"{}.png".format(index)))
        image = image.convert("RGB").resize([64,64]) 
        image = self.img_transform(image).permute([1,2,0]) * 255
        sample = {"image":image}
        return sample

class DynamicSprite(Dataset):
    def __init__(self,split = "train",data_path = None):
        super().__init__()
        
        assert split in ["train","val","test"]
        self.split = split
        self.root_dir = "/Users/melkor/Documents/datasets/DynamicSprites"

        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )
    def __len__(self): return 90#len(self.files)

    def __getitem__(self,index):
        video = np.load(os.path.join(self.root_dir,"{}.npy".format(index)))
        sample = {"video":video}
        return sample

class MineClip(Dataset):
    def __init__(self,split = "train",path = "",resolution = (128,128)):
        super().__init__()
        self.resolution = resolution
        self.path = "/Users/melkor/Documents/GitHub/Melkor/data/memo_work/{}.jpg"
        #self.images = sorted(glob(self.path))
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __len__(self):
        return 40

    def __getitem__(self, index):
        image = Image.open(self.path.format(index))
        image = image.convert("RGB").resize(self.resolution)
        image = self.img_transform(image).permute(1,2,0)* 255
        sample = {"image":image}
        return sample


class MineOut(Dataset):
    def __init__(self,split = "train",path = "",resolution = (128,128)):
        super().__init__()
        self.resolution = resolution
        self.path = "/Users/melkor/Documents/GitHub/Melkor/data/heathstone/{}.jpg"
        #self.images = sorted(glob(self.path))
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __len__(self):
        return 430

    def __getitem__(self, index):
        image = Image.open(self.path.format(index))
        image = image.convert("RGB").resize(self.resolution)
        image = self.img_transform(image).permute(1,2,0)* 255
        sample = {"image":image}
        return sample

class MineCrazy(Dataset):
    def __init__(self,split = "train",path = "",resolution = (128,128)):
        super().__init__()
        self.resolution = resolution
        self.path = "/Users/melkor/Documents/GitHub/Melkor/data/crazy/{}.jpg"
        #self.images = sorted(glob(self.path))
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __len__(self):
        return 3200

    def __getitem__(self, index):
        image = Image.open(self.path.format(index))
        image = image.convert("RGB").resize(self.resolution)
        image = self.img_transform(image).permute(1,2,0)* 255
        sample = {"image":image}
        return sample

class MineLM(Dataset):
    def __init__(self,split = "train",path = "",resolution = (128,128)):
        super().__init__()
        self.resolution = resolution
        self.path = "/Users/melkor/Documents/GitHub/Melkor/data/clattonia/{}.jpg"
   
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __len__(self):
        return 30

    def __getitem__(self, index):
        # Output Index mean whether there is a Valinor Set
        image = Image.open(self.path.format(index))
        image = image.convert("RGB").resize(self.resolution)
        image = self.img_transform(image).permute(1,2,0)* 255
        sample = {"image":image,"gt":index >= 20}
        return sample


class StaticPhase(Dataset):
    def __init__(self,split = "train"):
        super().__init__()
        
        assert split in ["train","val","test"]
        self.split = split
        self.root_dir = "/Users/melkor/Documents/datasets/static_phase"
        self.files = os.listdir(os.path.join(self.root_dir,split))
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.size = {"train":6000,"test":1500}
    def __len__(self): return self.size[self.split]#len(self.files)

    def __getitem__(self,index):
        path = self.files[index]
        image = Image.open(os.path.join(self.root_dir,self.split,"{}.png".format(index)))
        image = image.convert("RGB").resize([128,128]) 
        image = self.img_transform(image).permute([1,2,0]) * 255
        sample = {"image":image}
        return sample


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

class ToyData(Dataset):
    def __init__(self,split = "train",resolution = (128,128)):
        super().__init__()
        
        assert split in ["train","val","test"]
        self.split = split
        self.resolution = resolution
        self.root_dir = "/Users/melkor/Documents/datasets/toy/images"

        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )
    def __len__(self): return 400#len(self.files)

    def __getitem__(self,index):
        image = Image.open(os.path.join(self.root_dir,"{}.png".format(index)))
        image = image.convert("RGB").resize(self.resolution) 
        image = self.img_transform(image).permute([1,2,0]) * 256
        sample = {"image":image}
        return sample

class ToyDataWithQuestions(nn.Module):
    def __init__(self, split = "train", resolution = (128,128)):
        super().__init__()

        assert split in ["train","val","test"]
        self.split = split
        self.resolution = resolution
        self.root_dir = "/Users/melkor/Documents/datasets/toy/"
        self.questions = load_json(self.root_dir + "train_questions.json")
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __len__(self): return 200#len(self.files)

    def __getitem__(self,index):
        image = Image.open(os.path.join(self.root_dir,"images","{}.png".format( 1 + index)))
        image = image.convert("RGB").resize(self.resolution) 
        image = self.img_transform(image).permute([1,2,0]) 
        sample = {"image":image,"question":self.questions[index]}
        return sample


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
        return {"image":image * 255.0}#"question":question,"answer":answer,"program":program}

    def __len__(self):
        return 500 #len(self.ptr_data)