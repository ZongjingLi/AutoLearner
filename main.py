# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-05 07:09:56
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-05 10:47:45

print("This is a new Sublime Editor");

import torch
import torch.nn as nn

class Mifafa(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x

mifafa = Mifafa()

print(mifafa("MiracleFaFa_OvO_"))

from Karanir.dklearn.nn import FCBlock
from Karanir.utils import save_json

test_file = {"A":1,"Z":26,"question":"what is the color of the chair?"}
save_json(test_file,"outputs/test_dict.json")

import matplotlib.pyplot as plt

from autolearner.datasets import BattlecodeImageData
from torch.utils.data import DataLoader

data = BattlecodeImageData(split = "train")
loader = DataLoader(data, batch_size = 1)

for sample in loader:
	im = sample["image"][0]
	break

print(im.shape)

from PIL import Image
plt.figure(frameon = False)
plt.imshow(im)
plt.savefig("outputs/test_im.png")
plt.imsave()
