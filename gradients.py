import torch
import torch.nn as nn

from config import *
from models import *
from datasets import *

model = SceneLearner(config)
model = torch.load("checkpoints/Elbon_perception_toy_psgnet.ckpt", map_location=config.device)

train_dataset = PTRData("train", resolution = config.resolution)
dataloader = DataLoader(train_dataset, batch_size = 2)

counter = 0
n = 4

import matplotlib.pyplot as plt

for sample in dataloader:
    ims = sample["image"]
    image = ims.permute([0,3,1,2])
    b,c,w,h = image.shape
    # encoder model: extract visual feature map from the image

    feature_map = model.scene_perception.backbone.rdn(image)
    #recons = model.scene_perception(ims)["recons"][-1]
    
    for i in range(feature_map.shape[1]):
        grad_map = feature_map[0,i,:,:]
        plt.subplot(121)
        plt.imshow(grad_map.detach().numpy(),cmap="bone")
        plt.subplot(122)
        plt.imshow(ims[0,:,:,:])

        plt.show()

    counter += 1
    if counter >= n:break


    