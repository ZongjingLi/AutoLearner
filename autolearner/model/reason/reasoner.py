# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-13 13:43:08
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-25 02:20:09

import torch
import torch.nn as nn

import numpy as np

from karanir.dklearn.nn import FCBlock, ConvolutionUnits
from autolearner.model.knowledge.predicates import *
from .quasi_symbolic import *

class ConvEncoder(nn.Module):
	def __init__(self,input_channel, output_channel = 64):
		super().__init__()
		self.convs = ConvolutionUnits(input_channel, 132)

		self.final_linear = nn.Linear(132, output_channel)

	def forward(self, x):
		x = x.permute(0,3,1,2)
		x = self.convs(x)
		x = x.permute(0,2,3,1)
		x = self.final_linear(x)
		x = x.mean(1).mean(1)
		return x

class NeuroReasoner(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		# path to the domain knowledge file
		struct_path = config.root + "/domain/{}_action.icstruct".format(config.domain)

		self.neuro_actions = {}
		self.action_signatures = {}

		# predicate signatures
		self.derived = {}
		self.derived_signatures = {}

		# all the predicats available
		self.predicates = {}
		self.predicate_signatures = {}

		# Neuro Components of Predicates, Actions
		self.neuro_components = nn.ModuleList()

		self.type_specs = self.parse_type_specs(struct_path)

		self.parse_domain(struct_path)


	def parse_type_specs(self, path):
		scripts = ""
		with open(path,"r") as file:
			lines = file.readlines()
			for line in lines:scripts+=line#.replace(" ","")
		parse_scripts = inbraket(scripts[1:-2])

		specs = {}
		# Parse Domain Definion and Neuro Actions
		for comp in parse_scripts:
			if comp[1:11] == ":predicate":
				# infer the predicate type
				name, params, types= NeuroAction.parse_predicate(comp)
				output_type = inbraket(comp[1:-1])[0].split(" ")[1].split("=")[1]
				specs[name] = output_type
			
			if comp[1:9] == ":derived":
				name, params, effect, types = NeuroAction.parse_derived(comp)

				output_type = inbraket(comp[1:-1])[0].split(" ")[1].split("=")[1]
				neuro_comp_names, infer_types = find_all_unknown(effect,False)

				specs[name] = output_type
		return specs

	def observe_predicates(self, state):
		for deriv_name in self.derived:
			# [Unary Predicates]
			if len(self.derived_signatures[deriv_name]["params"]) == 1:
				for obj_name in state:
					deriv_effect = self.derived[deriv_name]


					# [Generate the Derived Script]
					formal_params = self.derived_signatures[deriv_name]["params"]

					knowledge_based = not "?" in deriv_effect
					# [Calculate the Effect Value]
					if knowledge_based:
						return
					else:
						eff_predicate = QuasiSymbolic.parse(deriv_effect)
						eff_args = {}
						eff_args["global_name"] = deriv_name
						for fparam in formal_params:
							eff_args[fparam] = state[obj_name]
						eff_output = eff_predicate(eff_args,self)
						state[obj_name][deriv_name] = eff_output
					
					
			
			# [Binary Predicates]
			if len(self.derived_signatures[deriv_name]["params"]) == 2:
				for obj_name in state:
					for obj_name in state:
						pass
		return state

	def parse_domain(self, path):
		scripts = ""
		with open(path,"r") as file:
			lines = file.readlines()
			for line in lines:scripts+=line#.replace(" ","")
		parse_scripts = inbraket(scripts[1:-2])

		# Parse Domain Definion and Neuro Actions
		for comp in parse_scripts:
			# Parse if [Action]
			if comp[1:8] == ":action":
				name,paras,precond,effect = NeuroAction.parse_action(comp)
				types = []

				neuro_comp_names, infer_types = find_all_unknown(effect, True)

				# make the local neuro component of action
				for i,local_name in enumerate(neuro_comp_names):
					comp_name = name + "-" + local_name
					substitute = make_by_types(comp_name,infer_types[i]["input"],infer_types[i]["output"],self.type_specs)
					self.neuro_components.append(substitute)

				self.neuro_actions[name] = NeuroAction(name,paras,precond,effect)
				self.action_signatures[name] = {"params":params,"types":types}

			if comp[1:11] == ":predicate":
				# infer the predicate type
				name, params, types= NeuroAction.parse_predicate(comp)
				self.predicates[name] = params
				self.predicate_signatures[name] = {"params":params,"types":types}
			
			if comp[1:9] == ":derived":
				name, params, effect, types = NeuroAction.parse_derived(comp)

				neuro_comp_names, infer_types = find_all_unknown(effect, False)

				# make the local neuro component accordingly
				for i,local_name in enumerate(neuro_comp_names):
					comp_name = name + "-" + local_name
					substitute = make_by_types(comp_name,infer_types[i]["input"],self.type_specs[name],self.type_specs)
					self.neuro_components.append(substitute)

				self.derived[name] = effect
				self.derived_signatures[name] = {"params":params,"types":types}
		return True

	def apply(self, name, args, executor):
		# [Choose Action] Make applicable arguments
		action = self.neuro_actions[name]
		precond = action.precond
		effect = action.effect
		formal_names = [c for c in action.params]
		print(effect)

		kwargs = {}
		# [Get the formal Parameters of the Action]
		
		entities = args["entities"]
		for i,fname in enumerate(formal_names):kwargs[fname] = entities[i]
		
		# [Check Entites have All Predicates]
		observed = False
		if not observed:
			kwargs["entities"] = self.observe_predicates(kwargs)

		# [Evaluate the Precondition]
		if precond is not None:
			precond_prob = 1.0
		else: precond_prob = 1.0
		kwargs["precond"] = precond_prob
		kwargs["global_name"] = name


		# [Perform Action on the Formal Parameters]
		q_effect = QuasiSymbolic.parse(effect)
		
		return q_effect(kwargs,self)

	def forward_reason(self, x):
		return x

class UniversalMap(nn.Module):
	def __init__(self, name, input_dims, output_dim, fc_dim = 64, conv_dim = 64):
		super().__init__()
		self.name = name
		fc_dim = fc_dim
		conv_dim = conv_dim

		# Add FCBlock or Convs to the Encoder Layer
		self.encoder_nets = nn.ModuleList([])
		joint_dim = 0
		for dim in input_dims:
			if len(dim) == 1:
				joint_dim += fc_dim
				self.encoder_nets.append(FCBlock(132,2,dim[0],fc_dim,"nn.CELU()"))
			if len(dim) == 3:
				joint_dim += conv_dim
				self.encoder_nets.append(ConvEncoder(dim[-1],conv_dim))

		# Create the Joint Map Decoder
		self.decoder_net = FCBlock(132,1,joint_dim, output_dim[0],"nn.CELU()")

	def forward(self,inputs):
		joint_features = []
		for i,encoder in enumerate(self.encoder_nets):
			joint_features.append(encoder(inputs[i]))
		joint_features = torch.cat(joint_features, dim = -1)

		output = self.decoder_net(joint_features)
		output = torch.sigmoid(output)
		return output

def tensor_shape(seq):return [int(v) for v in seq[1:-1].split(",")[1:]]

def make_by_types(name,input_types, output_type, type_specs):
	"""
	make the universal mapping that matches input_types and output_types
	@name: the name of the predicate to use
	"""
	input_dims = [tensor_shape(type_specs[t]) for t in input_types if t in type_specs]
	if output_type in type_specs:
		output_type = type_specs[output_type]
	output_dim = tensor_shape(output_type)

	unknown_net = UniversalMap(name,input_dims, output_dim)

	return unknown_net

def infer_io_types(sequence, name, ifo = True):
	start_pos = sequence.index("??{}".format(name))
	inner_subseq = ""
	bra_num = 0
	for i,v in enumerate(sequence[start_pos:]):
		if v == "(":bra_num += 1
		if v == ")":bra_num -= 1
		if bra_num == -1:
			break
		inner_subseq += v
	input_types = inbraket(inner_subseq)
	input_types = [c[1:c.index(" ")] for c in input_types]
	
	bra_count = 0
	parent_pos = None
	for i in range(start_pos - 1):
		char_ = sequence[start_pos - i - 1]
		if char_ == "(":bra_count += 1
		if char_ == ")":bra_count -= 1

		if bra_count == 2:
			parent_pos = start_pos - i
			break

	if ifo:
		output_head = sequence[parent_pos:]
		output_type = output_head[:output_head.index(":")]
	else: output_type = None
	return input_types, output_type

def find_all_unknown(sequence, ifo = True):
	outputs = []
	infer_types = []
	for i,v in enumerate(sequence):
		if v == "?" and sequence[i+1] == "?":
			loc = sequence[i:].index(" ")
			outputs.append(sequence[i+2:i+loc])
	for local_name in outputs:
		input_types, output_type = infer_io_types(sequence,local_name, ifo)
		infer_types.append({"input":input_types,"output":output_type})
	return outputs, infer_types
