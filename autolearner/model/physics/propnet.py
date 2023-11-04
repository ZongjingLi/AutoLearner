# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-29 14:03:09
# @Last Modified by:   Melkor
# @Last Modified time: 2023-11-03 06:17:00

import torch
import torch.nn as nn

class ParticleEncoder(nn.Module):
	def __init__(self, input_dim, output_dim = 32):
		super().__init__()
		self.linear1 = nn.Linear(input_dim, output_dim)

	def forward(self, x):
		return x

class PropNet(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x