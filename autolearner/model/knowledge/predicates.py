# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-13 13:07:56
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-19 04:42:09
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

class Precondition(object):
	def __init__(self, boolean_expression):
		super().__init__()
		self.boolean_expr = boolean_expression

class Effect(object):
	def __init__(self):
		super().__init__()
		self.boolean_expr = boolean_expression
		self.effect = None

	def available(self):
		return this.effect is not None

	def parse_canonical_effect(effect):
		return 

class NeuroAction(nn.Module):
	def __init__(self,name, params, precond, effect):
		super().__init__()
		self.name = name
		self.params = params
		self.precond = precond
		self.effect = effect

		"""
		all the concept with boolean return value follows the following grammar:
		is-concept: where concept is a known measure (box, cone, plane)
		"""

	def __str__(self):
		return "name:{}\n--params:{}\n--precond:{}\n--effect:{}".format(self.name,self.params,self.precond,self.effect) 

	@staticmethod
	def parse_action(action_script):
		assert action_script[0] == "(",print("not a valid action!")
		assert action_script[-1] == ")",print("not a valid action!")
		assert action_script[1] == ":",print("not start with : action!")
		# Parse the Action Name
		action_name = action_script[9:action_script.index("\n\t")]
		action_script = action_script[action_script.index("\n\t")+2:-1]
		
		# Parse the Parameters of Action
		parameters = action_script[action_script.index("(")+1: action_script.index(")")]
		parameters = parameters.split(" ")
		action_script = action_script[action_script.index(")")+3:]

		# Parse the Precond of the Action (if any)
		precond = None
		if action_script[1:8] == "precond":
			precond =  first_braket(action_script).replace("\n","").replace("\t","")
			action_script = action_script[action_script.index(")")+3:]

		# Parse the Effect of the Action
		effect = first_braket(action_script).replace("\n","").replace("\t","")
		
		return action_name, parameters, precond, effect

	@staticmethod
	def parse_predicate(predicate_script):
		assert predicate_script[0] == "(",print("not a valid action!")
		assert predicate_script[-1] == ")",print("not a valid action!")
		assert predicate_script[1] == ":",print("not start with : predicate!")
		predicate_script = predicate_script[predicate_script[1:-1].index("(")+1:-1]
		#print(predicate_script)
		name = predicate_script[1:predicate_script.index(" ")]
		params = []
		types = []
		return name, params, types

	@staticmethod
	def parse_derived(derived_script):
		assert derived_script[0] == "(",print("not a valid action!")
		assert derived_script[-1] == ")",print("not a valid action!")
		assert derived_script[1] == ":",print("not start with : derived!")
		
		derived_script = derived_script[derived_script[1:-1].index("(")+1:-1]
		name = derived_script[1:derived_script.index(" ")]
		
		param, derived_effect = inbraket(derived_script)
		effect = derived_effect
		params = param[param.index("?"):-1].split(" ")
		types = []
		return name, params, effect, types

	def forward(self,x):
		return self.__call__()

	def __call__(self, args, executor):
		if self.return_type == "boolean":
			# use the program executor to get the result
			return executor(executor.parse("filter(scene(),{})"))

		return self.predicate_net(*args)

def first_braket(sequence):
	outputs = None
	bra_num = 0
	hold_pos = None
	for i,v in enumerate(sequence):
		if v == "(":
			if bra_num == 0:hold_pos = i
			bra_num += 1
		if v == ")":
			bra_num -= 1
			if bra_num == 0:
				outputs = sequence[hold_pos:i+1]
				return outputs
	return outputs

def inbraket(sequence):
	outputs = []
	bra_num = 0
	hold_pos = None
	for i,v in enumerate(sequence):
		if v == "(":
			if bra_num == 0:hold_pos = i
			bra_num += 1
		if v == ")":
			bra_num -= 1
			if bra_num == 0:
				outputs.append(sequence[hold_pos:i+1])
	return outputs

if __name__ == "__main__":
	path = "/Users/melkor/Documents/GitHub/AutoLearner/autolearner/domain/demo_action.icstruct"
	ics = ""
	with open(path,"r") as file:
		lines = file.readlines()
		for line in lines:ics+=line#.replace(" ","")

	parse_output = inbraket(ics[1:-2])

	for comp in parse_output:
		if comp[1:8] == ":action":
			effect = NeuroAction.parse_action(comp)
			#print(effect)

		if comp[1:11] == ":predicate":
			predicate = NeuroAction.parse_predicate(comp)

		if comp[1:9] == ":derived":
			#print("derived:",comp)
			derived = NeuroAction.parse_derived(comp)