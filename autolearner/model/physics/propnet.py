# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-29 14:03:09
# @Last Modified by:   Melkor
# @Last Modified time: 2023-11-03 06:17:00

import torch
import torch.nn as nn

class ParticleEncoder(nn.Module):
	def __init__(self, input_dim, output_dim = 32):
		super().__init__()
		self.linear1 = nn.Linear(input_dim, output_dim)

	def forward(self, x):
		return x

class RelationEncoder(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super().__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim

		self.model = nn.Sequential([
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, output_dim),
			nn.ReLU(),
		])
	
	def forward(self, x):
		"""
		Args:
			x: [batch_size, n_relations, input_dim]
		Returns:
			x': [batch_size, n_relations, output_dim]
		"""
		B, N, D = x.shape
		x = self.model(x.flatten(start_dim = 1, end_dim = 2))
		return x.reshape(B,N,self.output_dim)

class ParticleEncoder(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super().__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim

		self.model = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, output_dim),
			nn.ReLU(),
		)
	def forward(self, x):
		"""
		Args:
			x: [batch_size, n_particles, input_dim]
		Returns:
			x': [batch_size, n_particles, output_dim]
		"""
		B, N, D = x.shape
		x = self.model(x.flatten(start_dim = 1, end_dim = 2))
		return x.reshape([B,N,self.output_dim])

class Propagator(nn.Module):
	def __init__(self, input_dim, output_dim, residual = False):
		super()._init__()

		self.input_dim = input_dim
		self.output_dim = output_dim
		self.residual = residual

		self.linear = nn.Linear(input_dim, output_dim)
		self.relu = nn.ReLU()
	def forward(self, x, res = None):
		B, N, D = x.shape
		if self.residual:
			res = self.linear(x.flatten(start_dim = 1, end_dim = 2))
			x = self.relu(x.flatten(start_dim = 1, end_dim = 2) + res)
		else:
			x = self.relu(self.linear(x.flatten(start_dim = 1, end_dim = 2)))
		return x.reshape(B,N,self.output_dim)

class ParticlePredictor(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super().__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim

		self.linear0 = nn.Linear(input_dim, hidden_dim)
		self.linear1 = nn.Linear(hidden_dim, output_dim)
		self.relu = nn.ReLU()
	
	def forward(self, x):
		B, N, D = x.shape
		x = x.reshape(B * N , D)
		x = self.linear1(self.relu(self.linear0(x)))
		return x.reshape(B,N,D)

class PropModule(nn.Module):
	def __init__(self, input_dim, output_dim, batch = True, residual = False):
		super().__init__()

		hidden_dim = 132

		self.batch = batch
		self.state_dim
		
		self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

		self.particle_encoder = ParticleEncoder(
			input_dim, hidden_dim, output_dim
		)

		self.relation_encoder = RelationEncoder(
			input_dim, hidden_dim, output_dim
		)
		
		# Propagator Modules in Action
		self.particle_propagator = Propagator
	
	def forward(self,state, Rs, Rr, Ra, itrs = 3):
		"""
		Args:
			state: input states of the input
			Rs: the relation feature sender matrix [B,N,N,1]
			Rr: the relation feature reciever matrix [B,N,N,1]
			Ra: the relation attribute matrix [B,N,N,D]
		"""
		B, N, Dx = state.shape
		# calculate the particle effect
		particle_effect = torch.autograd.Variable(
			torch.zeros([B, N, Dx])
		).to(self.device)

		# calculate reciever_states and sender_states
		if self.batch:
			state_r = torch.bmm(Rr, state)
			state_s = torch.bmm(Rs, state)
		# particle encode
		particle_encode = self.particle_encoder(state)

		# calculate the relation encode
		relation_encode = self.relation_encoder(torch.cat([
			state_r, state_s, Ra
		], dim = 2))

		for i in range(itrs):
			if self.batch:
				effect_r = torch.bmm(Rr, particle_effect)
				effect_s = torch.bmm(Rs, particle_effect)
			# calculate the relation effect
			#relation_effect = self.relation_propagator()
	
		return state

class PropNet(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x