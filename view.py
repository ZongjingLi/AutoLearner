# This file is somewhat hackish for now, will refactor later

import os,sys


import numpy as np

import torch

import torchvision
import torch_sparse

print(torch.__version__)

import torch_scatter
print(torch_scatter.__version__)

import torch_sparse

print(torch_sparse.__version__)

import torch_geometric
print(torch_geometric.__version__)

from torch_geometric.utils import to_dense_batch


import matplotlib.pyplot as plt
import psgnet
import datasets



batch_size = 1
imsize     = 128

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Create train dataloader 
tf_records_path = '/Users/melkor/Documents/GitHub/PSGNet/datasets/objects_room_train.tfrecords'

dataset = objectsRoomLoader.Tetrominoes()
train_dataloader = torch.utils.data.DataLoader(dataset,batch_size = 2,shuffle = True)

dataset = objectsRoomLoader.Clevr4()
#dataset = ObjectsRoomLoader.BattlecodeImageData()
#dataset = ObjectsRoomLoader.SpriteData()
#dataset = ObjectsRoomLoader.MineClip()

dataset1 = objectsRoomLoader.MineClip()#BattlecodeImageData()
dataset2 = objectsRoomLoader.MineOut()
dataset3 = objectsRoomLoader.MineCrazy()

dataset = torch.utils.data.ConcatDataset([dataset2,dataset1,dataset3])
#dataset = ObjectsRoomLoader.Clevr4()
train_dataloader = torch.utils.data.DataLoader(dataset,batch_size = batch_size, shuffle = False)



model=psgnet.PSGNetQTR(imsize)


model = torch.load("checkpoints/qtr_model5.ckpt",map_location = device)

model = model.to(device)
model.device = device

# Training loop

while True:
    for model_input in train_dataloader:
        
        gt_img = torch.tensor(model_input["image"].numpy()).float().to(device)/255
        
        outputs = model(gt_img)
        
        recons, clusters, all_losses = outputs["recons"],outputs["clusters"],outputs["losses"]

  
        plt.imshow(recons[-1][0].detach().reshape([imsize,imsize,3]).clip(0.0,1.0))

        plt.pause(0.001)
