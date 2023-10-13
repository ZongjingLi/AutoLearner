# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-05 07:09:56
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-12 09:52:22

import torch
import torch.nn as nn

from karanir.dklearn.nn import FCBlock
from karanir.utils import save_json

#lender --background --python render_images_partnet.py -- [args]

import matplotlib.pyplot as plt

from autolearner.datasets import SpriteWithQuestions
from torch.utils.data import DataLoader
from karanir.utils.tokens import *

data = SpriteWithQuestions(split = "trainn")
loader = DataLoader(data, batch_size = 1, shuffle = True)

for sample in loader:
	im = sample["image"][0]
	for data in sample["question"]:
		print(data["program"][0],end='->')
		print(data["answer"][0])
	save_json(sample["question"],"outputs/test_dict.json")
	break

print(im.shape)

from PIL import Image
plt.figure(frameon = False)
plt.imshow(im)
plt.savefig("outputs/test_im.png")



