# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-21 00:45:46
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-28 22:52:05


from autolearner.model import *
from autolearner.config import *

scenenet = SceneNet(config)

from autolearner.datasets import SpriteData, DynamicSpriteData
from torch.utils.data import DataLoader


dataset = SpriteData(config, "train",resolution=config.resolution)
dataset = DynamicSpriteData(config, "train",resolution = config.resolution)

loader = DataLoader(dataset, shuffle = True)

for sample in loader:
	#print(sample["image"].shape)
	break;

import matplotlib.pyplot as plt

# Frame
vidarr = sample["video"]
frames = torch.tensor(vidarr)
img1_batch = torch.stack([frames[0,0,...], frames[0,50,...]]).float().permute(0,3,1,2)
img2_batch = torch.stack([frames[0,5,...], frames[0,1,...]]).float().permute(0,3,1,2)

#frames = "/Users/melkor/Desktop/test/mov"

sepr = 1
img1_batch = frames[0,:-sepr,...].float().permute(0,3,1,2)
img2_batch = frames[0,sepr:,...].float().permute(0,3,1,2)
print(img1_batch.shape)
print(img2_batch.shape)

weights = Raft_Large_Weights.DEFAULT
transforms = weights.transforms()

#img1_batch += torch.randn([3,64,64]) * 0.1
#img2_batch += torch.randn([3,64,64]) * 0.1

def preprocess(img1_batch, img2_batch):
    img1_batch = torchvision.transforms.Resize(size=[128, 128], antialias=False)(img1_batch)
    img2_batch = torchvision.transforms.Resize(size=[128, 128], antialias=False)(img2_batch)
    return transforms(img1_batch, img2_batch)


img1_batch, img2_batch = preprocess(img1_batch, img2_batch)

#plt.ion()
#for i in range(vidarr.shape[1]):
#	plt.clf()
#	plt.imshow(vidarr[0,i,:,:,:])
#	plt.pause(0.1)
#plt.ioff()
#img1_batch, img2_batch = preprocess(img1_batch,img2_batch)

flow_maps = compute_optical_flow(img1_batch,img2_batch)[-1]

flow_maps = flow_to_image(flow_maps)
flow_maps = flow_maps /255

#print(flow_maps.shape)
#print(flow_maps.detach()[0,...].permute(1,2,0))
plt.imshow(flow_maps.detach()[0].permute(1,2,0))
plt.show()

plt.ion()
for i in range(vidarr.shape[1] - sepr):
	plt.subplot(121)
	plt.cla()
	plt.imshow(vidarr[0,i,:,:,:])
	plt.pause(0.1)
	plt.subplot(122)
	plt.cla()
	plt.imshow(flow_maps.detach()[i].permute(1,2,0))
	plt.pause(0.01)
plt.ioff()

outputs = scenenet(sample["image"])
masks = outputs["masks"]

num_masks = masks.shape[1]
for i in range(num_masks):
	plt.subplot(num_masks // 5, 5, i+1)
	plt.imshow(masks[0,i,:,:].detach())
plt.show()
