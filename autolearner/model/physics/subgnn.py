# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-29 14:03:49
# @Last Modified by:   Melkor
# @Last Modified time: 2023-11-03 06:18:50

import torch
import torch.nn as nn
from karanir.dklearn import FCBlock

def inv_product(z1, z2):
	_, v1 = z1.chunk(2, dim = -1)
	_, v2 = z2.chunk(2, dim = -1)
	return torch.cat([z1-z2,v1,v2], dim = -1)

def mask_avg(values, masks):
	"""
	values [B,N,D]: the values of vector input
	masks [B,N,M]: the soft or hard masks for each avg
	"""
	masks = masks.permute(0,2,1)
	avg_sum = torch.bmm(masks, values)

	return avg_sum / masks.sum(dim = -1, keepdim=True)

def expat(tens, dim , n = 1):
	outputs = tens.unsqueeze(dim)
	target_shape = [1 for d in outputs.shape]
	target_shape[dim] = n
	outputs = outputs.repeat(target_shape)
	return outputs

class SubEquiMLP(nn.Module):
	def __init__(self, hidden_dim, layer_num, input_dim, output_dim, g = None):
		super().__init__()
		device = "cuda:0" if torch.cuda.is_available() else "cpu"
		self.mlp = FCBlock(hidden_dim, layer_num, input_dim, hidden_dim)
		if g is not None:
			self.g = g
		else:
			self.g = torch.tensor([0,0,-1.0]).to(device)

	def forward(self, z, h, g = None):
		if g is None: g = self.g
		B, N, D = z.shape
		g = g.unsqueeze(0).unsqueeze(0)
		g = g.repeat(B,N,1)

		joint_features = torch.cat([z, h], dim = -1)


		aug_features = torch.cat([z, g], dim = -1)
		print(aug_features.shape)
		expat_mat = torch.einsum("bnd,bmd->bnm",aug_features,aug_features).unsqueeze(-1)
		
		h = h.unsqueeze(1).repeat(1,N,1,1)

		subequi_features = torch.cat([expat_mat,h], dim = -1)

		mat = self.mlp(subequi_features).squeeze(-1)

		output_features = torch.einsum("bnno,bmd->bnd",mat, z)

		return output_features


class SubGNN(nn.Module):
	def __init__(self, x_dim = 2, h_dim = 32, hidden_dim = 43):
		super().__init__()

		self.encode1 = SubEquiMLP(128,3,h_dim*4+1, hidden_dim)
		self.encode2 = SubEquiMLP(128,3,h_dim*4+1, hidden_dim)

		#print(x_dim*16 , 12*x_dim + h_dim*2)
		self.decode1 = SubEquiMLP(128,3,1 + 12*x_dim + h_dim*2, x_dim * 2)
		self.decode1 = SubEquiMLP(128,3,1 + 12*x_dim + h_dim*2, h_dim)

	def forward(self, z, h, edges, masks = None):
		"""
		z [BxNxD2]: the SE(3) convariant feature
		h [BxNxH]: the categorical invariant feature
		mask: [BxNxM] mask information of the local
		edges:[BxNxN] could be a sparse matrix.
		"""
		outputs = {}
		# [Initalize Representation]
		B, N, D = z.shape
		B, N, H = h.shape
		B, N, M = masks.shape

		assert D % 2 == 0, print("dimension d must be divisible")
		D = D // 2
		x, v = z.chunk(2, dim = -1)

		# [Gather Invariant-Covariant Features]
		masked_z_feature = mask_avg(z, masks) # [BxMxD]
		masked_h_feature = mask_avg(h, masks) # [BxMxH]

		cent_z = torch.bmm(masks,masked_z_feature) #[BxNxD]
		cent_h = torch.bmm(masks, masked_h_feature) # [BxNxH]
		
		# [Propagation]
		if isinstance(edges, torch.Tensor):
			cat_z_features = torch.cat([
				inv_product(expat(z,2,N), expat(cent_z,2,N)),
				inv_product(expat(z,1,N), expat(cent_z,1,N)),
				inv_product(expat(z,1,N), expat(z,2,N))
				], dim = -1).flatten(start_dim = 1, end_dim = 2)
			cat_h_features = torch.cat([
				expat(h,1,N),expat(h,2,N),
				expat(cent_h,1,N),expat(cent_h,2,N)
				], dim = -1).flatten(start_dim = 1, end_dim = 2)
			
			# Encode Joint Feature of Co-Invariant

			encode1 = self.encode1(cat_z_features, cat_h_features) # [BxNxNxE]
			encode2 = self.encode2(cat_z_features, cat_h_features) # [BxNxNxE]

			# Make the Final Decoder Features
			encode1 = encode1.reshape([B,N,N,-1])
			encode2 = encode2.reshape([B,N,N,-1])
			covar_feature = torch.cat([
				encode1.sum(2),
				inv_product(z,cent_z),
				], dim = -1)
			invar_feature = torch.cat([
				encode2.sum(2),
				h,
				cent_h,
				], dim = -1)
			decode1 = self.decode1(covar_feature, invar_feature)
			decode2 = self.decode1(covar_feature, invar_feature)

		return decode1

class HierarchySubGNN(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

	def forward(self, z, h, masks, edges):
		return 

def R(t):
	t = torch.tensor(t)
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	return torch.tensor([
		[torch.cos(t), torch.sin(t), 0],
		[torch.sin(t), torch.cos(t), 0],
		[0, 0, 1]]).to(device)

if __name__ == "__main__":
	N = 3; M = 2; B = 1
	x = torch.randn([B,N,3])
	v = torch.randn([B,N,3])

	masks = torch.softmax(torch.randn([B,N,M]) * 5, dim = -1)
	edges = torch.ones([B,N,N])
	subgnn = SubGNN(3, 32)

	# [Actual Input]
	z = torch.cat([x,v], dim = -1)
	h = torch.randn([B,N,32])

	rz =torch.cat([R(0.5) * x, R(0.4) * v], dim = -1)

	o = subgnn(z, h, edges, masks)
	ro = subgnn(rz, h, edges, masks)

	loss = torch.nn.functional.mse_loss(ro, o)
	print("loss:{}".format(loss.float().detach().numpy()))
	