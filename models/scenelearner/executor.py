import itertools

import torch
from torch import nn
from torch.nn import functional as F

from models.nn import build_entailment, build_box_registry
from utils import freeze
from utils.misc import *
from utils import *

class UnknownArgument(Exception):
    def __init__(self):super()

class UnknownConceptError(Exception):
    def __init__(self):super()

class SceneGraphRepresentation(nn.Module):
    def __init__(self):
        super().__init__()

        self.effective_level = 1
        self.max_level = 4

    @property
    def top_objects(self):
        return 0

class SceneProgramExecutor(nn.Module):
    NETWORK_REGISTRY = {}

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.entailment = build_entailment(config)
        self.concept_registry = build_box_registry(config)

        # [Word Vocab]
        concept_vocab = []
        with open("knowledge/{}_concept_vocab.txt".format(config.domain)) as vocab:
            for concept_name in vocab:concept_vocab.append(concept_name.strip())

        self.concept_vocab = concept_vocab
        
        # args during the execution
        self.kwargs = None 

        # Hierarchy Representation
        self.hierarchy = 0

        self.translator = config.translator

    def get_concept_embedding(self,concept):
        try:
            concept_index = self.concept_vocab.index(concept)
            idx = torch.tensor(concept_index).unsqueeze(0).to(self.config.device)
            return self.concept_registry(idx)
        except:
            print(concept_index)
            raise UnknownConceptError

    def forward(self, q, **kwargs):
        self.kwargs = kwargs

        return q(self)

    def parse(self,string, translator = None):
        string = string.replace(" ","")
        if translator == None: translator = self.translator
        def chain(p):
            head, paras = head_and_paras(p)
            if paras == None:
                q = head
            elif '' in paras:
                q = translator[head]()
            else:
                args = [chain(o) for o in paras]
                if head in translator: q = translator[head](*args)
                else: raise UnknownArgument
            return q
        program = chain(string)
        return program


class MetaLearner(nn.Module):

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = MetaLearner.get_name(cls.__name__)
        SceneProgramExecutor.NETWORK_REGISTRY[name] = cls
        cls.name = name

    @staticmethod
    def get_name(name):
        return name[:-len('Learner')]

    def forward(self, p):
        return {}

    def compute_logits(self, p, **kwargs):
        return p.evaluate_logits(self, **kwargs)


class PipelineLearner(nn.Module):
    def __init__(self, network, entailment):
        super().__init__()
        self.network = network
        self.entailment = entailment

    def forward(self, p):
        shots = []
        for q in p.train_program:
            end = q(self)["end"]
            index = end.squeeze(0).max(0).indices
            shots.append(q.object_collections[index])
        shots = torch.stack(shots)
        if not p.is_fewshot:
            shots = shots[0:0]
        fewshot = p.to_fewshot(shots)
        return fewshot(self)

    def compute_logits(self, p, **kwargs):
        return self.network.compute_logits(p, **kwargs)