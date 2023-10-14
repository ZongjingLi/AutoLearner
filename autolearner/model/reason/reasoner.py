# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-13 13:43:08
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-13 14:04:00

import torch
import torch.nn as nn

import numpy as np

from model.knowledge.predicates import *

class NeuroReasoner(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		# path to the domain knowledge file
		struct_path = "autolearner/domain/{}_action.icstruct".format(config.domain)
		print(struct_path)

		self.neuro_actions = {}
		self.parse_domain(struct_path)

	def parse_domain(self, path):
		self.neuro_actions["v1"] = NeuroAction("v1")
		return 


	def forward_reason(self, x):
		return x