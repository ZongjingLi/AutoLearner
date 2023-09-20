import torch
import torch.nn as nn
import torchvision
import numpy as np

class BaseAttributes(object):
    
    @staticmethod
    def cum_sum(sequence):
        r,s = [0], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r

    def __init__(self, attributes_config):
        super(BaseAttributes, self).__init__()
        

class AttributeSpectrum(nn.Module):
    def __init__(self, keys, vals, normalize = True):
        self.keys = keys
        self.vals = vals 

    def braket(self,other):
        for e in self.keys: assert e in other.keys
        return (self.vals * other.vals).reshape(1)
    
    def sample(self, rand = True):
        pdf = np.array(self.vals) / np.sum(np.array(self.vals))
        if rand:return np.random.choice(self.keys, p = pdf)
        else:return self.keys(np.argmax(pdf))

    def sample_with_prob(self, rand = True):
        pdf = np.array(self.vals) / np.sum(np.array(self.vals))
        if rand:return np.random.choice(self.keys, p = pdf)
        else:return self.keys(np.argmax(pdf))
