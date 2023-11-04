# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-11-03 06:15:44
# @Last Modified by:   Melkor
# @Last Modified time: 2023-11-03 06:18:50
import torch
import torch.nn as nn

class NeuroParticleFilter(nn.Module):
	def __init__(self, config):
		super().__init__()

	def step(self, action):
		return action