# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-13 13:24:24
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-25 02:20:27

from autolearner.config import *
from autolearner.model  import *

config.concept_dim = 2
executor = SceneProgramExecutor(config)

test_concept = "red"
p = "exist(filter(scene(),{}))".format(test_concept)
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
#print("exists {} prob:{}".format(test_concept,o["end"].sigmoid().detach().numpy()))

q = executor.parse("filter(scene(),red)")
o = executor(q, **kwargs)
#print(o["end"][0].sigmoid().detach().numpy())

red_feat = torch.tensor([[0.4, -0.4, EPS, EPS]])
#print(executor.entail_prob(red_feat,"red")[0].sigmoid().detach().numpy())


neuro_planner = NeuroReasoner(config)

for pred in neuro_planner.predicates:
	pass
	print(pred, neuro_planner.predicates[pred])

for deriv in neuro_planner.derived:
	pass
	print(deriv, neuro_planner.derived[deriv], neuro_planner.derived_signatures[deriv]["params"])

for name in neuro_planner.neuro_actions:
	pass
	print(neuro_planner.neuro_actions[name])

for comp in neuro_planner.neuro_components:pass;#print(comp)

autolearner = AutoLearner(config)

# Must at least include input predicate
o1 = {"image":torch.randn(1,64,64,3)}
o2 = {"image":torch.randn(1,64,64,3)}

entities = [o1,o2]
state = {"entities":entities}

obs_state = {"name1":o1,"name2":o2}

obs_state = neuro_planner.observe_predicates(obs_state)

for i,fn in enumerate(state["entities"]):
	print("fn:{}".format(i))
	print("is-yellow:",fn["is-yellow"].detach().numpy())
	print("pos:",fn["pos"].detach().numpy())
print("")
neuro_planner.apply("demo-act",state,executor) 
for i,fn in enumerate(state["entities"]):
	print("fn:{}".format(i))
	print("is-yellow:",fn["is-yellow"].detach().numpy())
	print("pos:",fn["pos"].detach().numpy())
print("")
neuro_planner.apply("move-in",state,executor) 
for i,fn in enumerate(state["entities"]):
	print("fn:{}".format(i))
	print("is-yellow:",fn["is-yellow"].detach().numpy())
	print("pos:",fn["pos"].detach().numpy())
print("")
neuro_planner.apply("turnleft",state,executor) 
for i,fn in enumerate(state["entities"]):
	print("fn:{}".format(i))
	print("is-yellow:",fn["is-yellow"].detach().numpy())
	print("pos:",fn["pos"].detach().numpy())
#state = neuro_planner.apply("turnleft",state,2)
#print(state)