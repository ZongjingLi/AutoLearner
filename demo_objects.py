# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-21 00:45:46
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-21 02:45:55


from autolearner.model import *
from autolearner.config import *

scenenet = SceneNet(config)

from autolearner.datasets import SpriteData
from torch.utils.data import DataLoader

dataset = SpriteData("train",resolution=(64,64))
loader = DataLoader(dataset)

for sample in loader:
	print(sample["image"].shape)
	break;

import matplotlib.pyplot as plt

def optical_flow(image_sequence):
	return 

outputs = scenenet(sample["image"])