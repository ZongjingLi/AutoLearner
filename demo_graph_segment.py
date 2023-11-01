# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-25 15:42:04
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-30 02:29:34

from autolearner.config import *
from autolearner.model  import *

import random
from tqdm import tqdm
import sys

# [Visualize]

# [Generate Data] (demo)
eps = 1e-6
blue_feature = torch.cat(
	[torch.randn([1,2]), eps * torch.ones(1,2)] , dim = -1)
blue_feature.requires_grad = True

red_feature = torch.cat(
	[torch.randn([1,2]), eps * torch.ones(1,2)] , dim = -1)
red_feature.requires_grad = True

def R(t):
	return torch.tensor([[torch.cos(t), -torch.sin(t)],[torch.sin(t), torch.cos(t)]])

def make_house(loc, pos):
	return loc, pos

def make_ship(loc, pos):
	return loc, pos

def make_scene(n_tri, n_squ, n_house = 0, n_ship = 0):
	data = []
	edges = []
	cats_count = [n_tri, n_squ, n_house, n_ship]
	for _ in range(n_tri):
		loc = torch.randn(2).clip(-.9, 0.9) / 2 + 0.5
		pos = torch.randn(2).clip(-.9, 0.9) / 2 + 0.5
		data.append(["triangle",loc,pos])
	for _ in range(n_squ):
		loc = torch.randn(2).clip(-.9, 0.9) / 2 + 0.5
		pos = torch.randn(2).clip(-.9, 0.9) / 2 + 0.5
		data.append(["square",loc,pos])
	for _ in range(n_house):
		loc = torch.randn(2).clip(-.9, 0.9) / 2 + 0.5
		pos = 0.0 * torch.randn(2).clip(-.9, 0.9) / 2 + 0.5
		data.append(
			["triangle",loc + torch.tensor([0.0,0.1]),pos]
			)
		data.append(["square",loc,pos])
		edges.append([len(data)-2, len(data)-1])
	for _ in range(n_ship):
		loc = torch.randn(2).clip(-.9, 0.9) / 2 + 0.5
		pos = 0.0 * torch.randn(2).clip(-.9, 0.9) / 2 + 0.5
		data.append(["triangle",loc,pos])
		data.append(["square",loc,pos])
		edges.append([len(data)-2, len(data)-1])
	programs = []
	answers = []
	for i,cat in enumerate(["triangle","square","house","ship"]):
		flag = cats_count[i] > 0
		programs.append("exist(filter(scene(),{}))".format(cat))
		if flag:answers.append("yes")
		else:answers.append("no")

	all_data = {"scene":data,"program":programs,"answer":answers, "edges":edges}
	return all_data

def generate_gc_dataset(num_samples):
	scenes = []
	for i in range(num_samples):
		n_t = int(random.random() + 0.5)
		n_s = int(random.random() + 0.5)
		n_house = int(random.random() + 1.5)
		n_ship = int(random.random() + 0.5)
		compose_scene = make_scene(n_t, n_s, n_house, n_ship)
		scenes.append(compose_scene)
	return scenes

def render_scene(objects):
	return 

# [Module]
class GraphPoolLayer(nn.Module):
	def __init__(self, input_dim, output_dim, latent_dim = 128):
		super().__init__()
		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.output_dim = output_dim

	def forward(self, x, edges):
		return x

class ObjectFeatureEncoder(nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()
		self.linear = nn.Linear(input_dim, output_dim)
		self.linear = FCBlock(128, 2 ,input_dim, output_dim)

	def forward(self, x):
		return self.linear(x)

class GraphCoarsenLearner(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		self.executor = SceneProgramExecutor(config)
		self.object_feature_map = ObjectFeatureEncoder(5,config.concept_dim)

		self.graph_coarsen = GraphPoolLayer(config.concept_dim, config.concept_dim)

	def forward(self, scene, programs, answers, test = False):
		all_loss = 0.00
		object_features = []

		for obj in scene:
			if obj[0] == "house":
				cat = torch.ones([1])
			if obj[0] == "ship":
				cat = torch.ones([1]) * 2
			if obj[0] == "square":
				cat = torch.ones([1]) * -1
			if obj[0] == "triangle":
				cat = torch.ones([1]) * -2
			infeat = torch.cat([cat,obj[1],obj[2]])
			obj_feat = self.object_feature_map(infeat)
			obj_feat = torch.cat([obj_feat, torch.ones_like(obj_feat) * eps] , dim = -1).unsqueeze(0)

			object_features.append(obj_feat)
		object_features = torch.cat(object_features, dim = 0)
		kwargs = {"end":[torch.ones(feat.shape[0]) for feat in [object_features]],"features":[object_features]}

		acc = 0
		total = 0
		for i,program in enumerate(programs):
			#program = "filter(scene(),ship)"
			program = self.executor.parse(program)
			o = self.executor(program, **kwargs)
			
			if answers[i] == "yes":
				all_loss -= o["end"].sigmoid().log()
				if test and o["end"].sigmoid() > 0.5:acc += 1
			else: 
				all_loss -= (1-o["end"].sigmoid()).log()
				if test and o["end"].sigmoid() < 0.5:acc += 1
			if test:total += 1

		outputs = {"loss":all_loss,"acc":acc,"total":total}
		return outputs

	def eval(self, scene, program):
		object_features = []

		for obj in scene:
			if obj[0] == "house":
				cat = torch.ones([1])
			if obj[0] == "ship":
				cat = torch.ones([1]) * 2
			if obj[0] == "square":
				cat = torch.ones([1]) * -1
			if obj[0] == "triangle":
				cat = torch.ones([1]) * -2

			infeat = torch.cat([cat,obj[1],obj[2]])
			obj_feat = self.object_feature_map(infeat)
			obj_feat = torch.cat([obj_feat, torch.ones_like(obj_feat) * eps] , dim = -1).unsqueeze(0)

			object_features.append(obj_feat)
		object_features = torch.cat(object_features, dim = 0)
		kwargs = {"end":[torch.ones(feat.shape[0]) for feat in [object_features]],"features":[object_features]}

		program = self.executor.parse(program)
		o = self.executor(program, **kwargs)
		return o

config.concept_dim = 50
gcl = GraphCoarsenLearner(config)

def train(model, dataset, epochs = 100):
	loss_history = []
	acc_history = []
	params = [{"params":model.parameters()},]#{"params":blue_feature},{"params":features}]
	optimizer = torch.optim.Adam(params, lr = 1e-3)
	for epoch in range(epochs):
		epoch_loss = 0.0
		for data in dataset:
			programs = data["program"]
			answers = data["answer"]

			outputs = model(data["scene"], programs, answers)

			loss = outputs["loss"]

			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

			epoch_loss += loss.detach()
		epoch_loss /= (len(dataset) * 2)
		loss_history.append(epoch_loss)
		acc,total = evaluate(model,dataset)
		acc_history.append(acc*1.0 /total)
		if True or visualize:
			print("loss:{}".format(epoch_loss))
			plt.figure("visualize",figsize = (12,6))
			plt.subplot(121)
			plt.cla();plt.plot(loss_history)
			plt.pause(0.01)
			plt.subplot(122)
			plt.cla();plt.plot(acc_history)
			plt.pause(0.01)

	return model

def evaluate(model, dataset):
	acc = 0
	total = 0
	for data in dataset:
		programs = data["program"]
		answers = data["answer"]

		outputs = model(data["scene"], programs, answers, True)
		acc += outputs["acc"]
		total += outputs["total"]

		loss = outputs["loss"]
	print("acc:{}/{}".format(acc,total))
	return acc, total
# [Train the Model]
gc_dataset = generate_gc_dataset(num_samples = 100)

evaluate(gcl, gc_dataset)

model = train(gcl, gc_dataset)

evaluate(gcl, gc_dataset)

scene = gc_dataset[0]["scene"]
for obj in scene:print(obj[0], end= " ")

print("")

for cat in ["triangle","square","ship","house"]:
	o = gcl.eval(scene, "filter(scene(),{})".format(cat))
	print(cat,":",o["end"][0].sigmoid().detach().numpy())