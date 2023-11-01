# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-24 11:07:30
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-25 06:39:11

import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

class AcherusEnv:
	def __init__(self):
		super().__init__()
		self.background = 0.0

	def step(self, action):
		if action == None or action == "None":
			pass

		return {"observation":img}

if __name__ == "__main__":
	pass