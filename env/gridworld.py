# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-05 10:56:50
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-05 11:23:12

import torch
import numpy as np
import matplotlib.pyplot as plt

from karanir.algs.graph import GridGraph
from base_env import BaseEnv

class GridWorld(BaseEnv):
	def __init__(self, size=(32,32)):
		self.size = size
		self.grid_graph = GridGraph(*size)

	def render(self):
		return self.grid_graph.render()

if __name__ == "__main__":
	test_gridworld = GridWorld((32,32))

	print(test_gridworld)

	print("test functionality done.")