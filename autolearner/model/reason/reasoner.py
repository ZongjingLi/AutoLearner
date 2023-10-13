# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-13 13:43:08
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-13 13:58:37

import torch
import torch.nn as nn

import numpy as np

class NeuroReasoner(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		# path to the domain knowledge file
		struct_path = "autolearner/domain/{}_action.icstruct".format(config.domain)
		print(struct_path)

		self.parse_domain(struct_path)

	def parse_domain(self, path):
		return 


	def forward_reason(self, x):
		return x