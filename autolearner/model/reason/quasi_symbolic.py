# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-18 21:46:31
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-21 00:40:49
import torch
import torch.nn as nn
from model.knowledge.predicates import *

def segindex(seq,chars):
	min_idx = len(seq) + 1
	for c in chars:min_idx = min(min_idx,seq.index(c))
	return min_idx

def is_predicate(seq):
	flag = True
	comps = seq.split(" ")
	if "?" not in seq:flag = False
	if "?" in comps[0]: flag = False
	for c in comps[1:]:
		if c!="" and c[0] != "?":flag = False
	if "(" == seq[0] or ")" == seq[-1]:
		flag = False
	return flag

class QuasiSymbolic(object):
	def __init__(self):
		super().__init__()

	def __call__(self):
		return 


	@staticmethod
	def parse(sequence):

		if is_predicate(sequence):
			comps = sequence.split(" ")
			name = comps[0]
			args = [QuasiSymbolic.parse(arg) for arg in comps[1:]]
			return Predicate(name,args)

		if "??" == sequence[:2]:
			func_name = sequence[:sequence.index("(")]
			comp_args = inbraket(sequence)
			args = [QuasiSymbolic.parse(a) for a in comp_args]
			return NeuroComp(func_name, args)

		if len(sequence)>=2 and "?" == sequence[0] and sequence[1] != "?":
			return Entity(sequence[:].replace(" ",""))

		if "::" in sequence[:sequence.index("(")]:
			#print(sequence[:sequence.index("(")].split("::"))

			predicate_name = sequence[:sequence.index("(")].split("::")[0]
			assert sequence[:sequence.index("(")].split("::")[1][:6] == "assign",print("not a valid assign")

			assignment = sequence[segindex(sequence,["?"," ","("]):]

			bra_loc = assignment.index("(")
			target, value = assignment[:bra_loc],assignment[bra_loc:]
			target = QuasiSymbolic.parse(target)

			value = QuasiSymbolic.parse(value)

			return Assign(predicate_name,target,value)

		if sequence[:2] == "??":
			func_name = sequence[2:sequence.index("("," ","?")]

		if sequence[:3] == "and":
			comps = sequence.split("and")
			if len(comps) == 2:
				comps = inbraket(comps[1])
				left_comp = QuasiSymbolic.parse(comps[0])
				right_comp = QuasiSymbolic.parse(comps[1])
				return AndExecute([left_comp,right_comp])
			else:
				curr_codeblock = comps[1]
				rest_codeblock = sequence[3+len(comps[1]):]
				childs = []
				childs.append(QuasiSymbolic.parse(curr_codeblock))
				childs.append(QuasiSymbolic.parse(rest_codeblock))
				return AndExecute(childs)
		comps = inbraket(sequence)

		if comps[0] == sequence:
			return QuasiSymbolic.parse(sequence[1:-1])
		return sequence

	def __str__(self):
		return "N/A"

class AndExecute(QuasiSymbolic):
	def __init__(self,childs):
		super().__init__()
		self.childs = childs

	def __call__(self, kwargs, reasoner):
		for child in self.childs:
			child(kwargs, reasoner)

	def __str__(self):
		return "{} and {}".format(str(self.childs[0]),str(self.childs[1]))

class Entity(QuasiSymbolic):
	def __init__(self,name):
		self.name = name

	def __call__(self, kwargs, reasoner):
		return self.name

	def __str__(self):return self.name

class If(QuasiSymbolic):
	def __init__(self, condition, code_block):
		super().__init__()
		self.name = "if"
		self.condition = condition
		self.code_block = code_block

	def __call__(self,kwargs, reasoner):
		boolean_prob = self.condition()["end"]
		return self.code_block(boolean_prob, kwargs, executor, reasoner)

	def __str__(self):return "if"

class Assign(QuasiSymbolic):
	def __init__(self, predicate, target, value):
		super().__init__()
		self.name = "assign"
		self.predicate = predicate
		self.target = target
		self.value = value

	def __call__(self, kwargs, reasoner):

		# value asssign to
		target = self.target(kwargs, reasoner)
		value1 = kwargs[target][self.predicate]

		# value to assign from other
		value2 = self.value(kwargs, reasoner)
		
		# diff Godel-t norm of block code
		prob = kwargs["precond"]

		update = value1 * (1 - prob) + value2 * prob
		kwargs[self.target.name][self.predicate] = update
		return kwargs

	def __str__(self):
		return "{}.{}<-{}".format(self.target,self.predicate,self.value)

class Predicate(QuasiSymbolic):
	def __init__(self, name, args):
		super().__init__()
		self.name = name.replace(" ","")
		self.args = args

	def __call__(self, kwargs, reasoner):
		valid = self.name in reasoner.predicates or self.name in reasoner.derived
		assert valid,print("this is not a valid predicate")

		comp = None
		for ncomp in reasoner.neuro_components:
			if ncomp.name.split("-")[0] == self.name:comp = ncomp;break

		args = [arg(kwargs, reasoner) for arg in self.args]

		if self.name in reasoner.predicates:
			pred_val = kwargs[args[0]]
			for arg in args[1:]:
				pred_val = pred_val[arg]
			pred_val = pred_val[self.name]
			return pred_val
		elif self.name in reasoner.derived:
			effect = reasoner.derived[self.name]
			formal_params = reasoner.derived_signatures[self.name]["params"]

			knowledge_based = False
			if knowledge_based:
				return
			else:
				q_predicate = QuasiSymbolic.parse(effect)
				eff_args = {}
				eff_args["global_name"] = self.name
				for i,fparam in enumerate(formal_params):
					eff_args[fparam] = kwargs[self.args[i](kwargs,reasoner)]
				q_output = q_predicate(eff_args,reasoner)
				return q_output
		else:
			assert False,print("Not a valid Predicate/Derived")

		#return comp(args)

	def __str__(self):
		output = self.name + "("
		for arg in self.args:output+=str(arg);output+" "
		output += ")"
		return output

class NeuroComp(QuasiSymbolic):
	def __init__(self,name, args):
		super().__init__()
		self.name = name.replace(" ","")
		self.args = args

	def __call__(self, kwargs, reasoner):
		lookup_name = "{}-{}".format(kwargs["global_name"],self.name[2:])

		comp = None
		for ncomp in reasoner.neuro_components:
			if ncomp.name == lookup_name:comp = ncomp;break

		input_args = [arg(kwargs, reasoner) for arg in self.args]
		return comp(input_args)

	def __str__(self):
		output = self.name + "("
		for arg in self.args:output+=str(arg);output+=" "
		output += ")"
		return output