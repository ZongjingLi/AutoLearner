# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-18 21:46:31
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-20 04:26:13
import torch
import torch.nn as nn

class QuasiSymbolic(object):
	def __init__(self):
		super().__init__()

	def __call__(self):
		return 

	def __str__(self):
		return ""

	def parse(self,sequence):
		return 

class AndExecute(QuasiSymbolic):
	def __init__(self,left_block,right_block):
		super().__init__()
		self.left_block = left_block
		self.right_block = right_block

	def __call__(self, kwargs, reasoner):
		self.left_block(executor, reasoner)

class Entity(QuasiSymbolic):
	def __init__(self,name):
		self.name = name

	def __call__(self, kwargs, reasoner):
		return kwargs[self.name][predicate]

class If(QuasiSymbolic):
	def __init__(self, condition, code_block):
		super().__init__()
		self.name = "if"
		self.condition = condition
		self.code_block = code_block

	def __call__(self,kwargs, reasoner):
		boolean_prob = self.condition()["end"]
		return self.code_block(boolean_prob, kwargs, executor, reasoner)

class Assign(QuasiSymbolic):
	def __init__(self, predicate, target, value):
		super().__init__()
		self.name = "assign"
		self.predicate = predicate
		self.target = target
		self.value = value

	def __call__(self, kwargs, reasoner):
		# value asssign to
		target = self.target(self.predicate,kwargs,executor,reasoner)
		value1 = target[self.predicate]

		# value to assign from other
		value2 = self.value(kwargs, executor, reasoner)
		
		# diff Godel-t norm of block code
		prob = soft_boolean

		update = value1 * (1 - prob) + value2 * prob
		return {}

class Predicate(QuasiSymbolic):
	def __init__(self, entity):
		super().__init__()
		self.entity = entity

	def __call__(self, kwargs, reasoner):
		reasoner.neuro_predicates
		return 

class NeuroComp(QuasiSymbolic):
	def __init__(self,local_name, global_name, child):
		super().__init__()
		self.local_name = local_name
		self.global_name = global_name
		self.child = child

	def __call__(self, kwargs, reasoner):
		comp = reasoner.neuro_components["{}-{}".format(self.global_name,self.local_name)]
		return comp(self.child())