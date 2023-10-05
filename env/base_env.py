# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-05 11:11:49
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-05 11:18:30

import numpy as np

class BaseEnv:
	def __init__(self):
		self.action_space = None
		self.state_space = None

	def step(self, action):
		"""
		step the env with action provided, return the s,r,d
		return:(keys)
		--state: the next state observation
		--reward: reward value of current action on previous state
		--done: whether this state should be stopped.
		"""
		output_state = {
		"state":None,
		"reward":None,
		"done":True,
		}
		return output_state

	def reset(self):
		"""this method should reset environment to the initial state"""
		return