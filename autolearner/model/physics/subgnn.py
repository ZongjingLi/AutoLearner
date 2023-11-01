# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-29 14:03:49
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-29 14:04:32

import torch
import torch.nn as nn

class SubGNN(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x, features, edges):
		return x