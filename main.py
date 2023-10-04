# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-05 07:09:56
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-05 07:21:48

print("this is a new funcion");

import torch
import torch.nn as nn

class Mifafa(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x

mifafa = Mifafa()

print(mifafa("MiracleFaFa_OvO_"))