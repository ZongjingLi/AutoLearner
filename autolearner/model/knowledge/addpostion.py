import torch
import torch.nn as nn

class Adpositions(nn.Module):
    def __init__(self, name, arity):
        super().__init__()
        self.arity = arity
    
    def forward(self, x):
        return x