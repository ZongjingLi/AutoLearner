import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Singleton
from .measure import Measure

class Entailment(nn.Module, metaclass = Singleton):
    rep = "box"

    def __init__(self,config):
        super().__init__()
        self.measure = Measure(config)

    def forward(self, premise, consequence):
        return self.measure.entailment(premise, consequence)
    
class PlaneEntailment(nn.Module, metaclass = Singleton):
    rep = "plane"

    def __init__(self, config):
        super().__init__()
        self.margin = 0.2

    def forward(self, premise, consequence):
        logit_pr = (premise * consequence - self.margin).mean(-1).clamp(-1, 1) * 8.
        return logit_pr

class ConeEntailment(nn.Module, metaclass = Singleton):
    rep = "cone"

    def __init__(self, config):
        super().__init__()
        self.weight = 8.0
        self.margin = 0.8

    def forward(self, premise, consequence):
        logit_pr = self.weight / self.margin * (F.cosine_similarity(premise, consequence, -1) - 1 + self.margin)

        return logit_pr

REP2ENTAILMENT = {"box":Entailment, "plane":PlaneEntailment, "cone": ConeEntailment}

def build_entailment(config): return REP2ENTAILMENT[config.concept_type](config)