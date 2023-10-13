# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-13 13:24:24
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-13 13:56:47

from config import *
from model  import *

config.concept_dim = 2
executor = SceneProgramExecutor(config)

p = "exist(filter(scene(),red))"
q = executor.parse(p)
EPS = 1e-6

if config.concept_type == "box":
    features = [torch.tensor([
            [0.4,-0.4,EPS,EPS],
            [0.3,0.4,EPS,EPS],])]
    r = 0.5

kwargs = {"end":[torch.ones(feat.shape[0]) for feat in features],
             "features":features}

o = executor(q, **kwargs)
print(o["end"])

q = executor.parse("filter(scene(),red)")
o = executor(q, **kwargs)
print(o["end"])

red_feat = torch.tensor([[0.4, -0.4, EPS, EPS]])
print(executor.entail_prob(red_feat,"red"))

neuro_move = NeuroAction("move")
print(neuro_move)

neuro_planner = NeuroReasoner(config)
print(neuro_planner)