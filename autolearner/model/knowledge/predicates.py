# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-13 13:07:56
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-13 13:24:39
import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt

from karanir.dklearn.nn import FCBlock

class NeuroPredicate(nn.Module):
	def __init__(self, name, args_type = ["vector"],return_type = "boolean", latent_dim = 128):
		super().__init__()
		"""
		all the concept with boolean return value follows the following grammar:
		is-concept: where concept is a known measure (box, cone, plane)
		"""
		self.name = name
		latent_dim = latent_dim
		self.predicate_net = None

		assert return_type in ["boolean","vector"],print("not a valid return type")

		if return_type == "boolean":
			self.predicate_net = FCBlock(128,3,latent_dim, 1, "nn.Sigmoid()")
		if return_type == "vector":
			self.predicate_net = FCBlock(128,3,latent_dim, latent_dim, "nn.Sigmoid()")

	def forward(self,x):
		return self.__call__()

	def __call__(self, args, executor):
		if self.return_type == "boolean":
			# use the program executor to get the result
			return executor(executor.parse("filter(scene(),{})"))

		return self.predicate_net(*args)

class NeuroAction(nn.Module):
	def __init__(self,name):
		super().__init__()

	@staticmethod
	def parse_action(self,action_script):
		return 

