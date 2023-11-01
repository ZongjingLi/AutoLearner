from karanir import *

import torch
import torch.nn as nn

class AgtNet(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

	def forward(self, x, features, edegs):
		return