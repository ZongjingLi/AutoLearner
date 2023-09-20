from copy import deepcopy
import numpy as np
import math

from .sim.step_sim import step, reverse_step
from .sim.propnet import *

class Stepper(object):
    def __init__(self, config):
        perturbation_config = 0
        self.to_perturb = perturbation_config

        self.use_magic = True

        if self.use_magic:
            self.disappear_probability = 0.0
            self.disappear_penalty = 1.0
            
            self.stop_probability = 0.0
            self.stop_penalty = 1.0

            self.accelerate_probability = 0.0
            self.accelerate_penalty = 1.0

            self.accelerate_lambda = 0.5
    
class NeuroStepper(object):
    def __init__(self, config):
        perturbation = 0

        self.dynamic_model = None
        if config.dynamic_model == "propnet":
            self.dynamic_model = PropNet(config)

    def step(self, states):
        new_states = [] # evolve new states in the scene
        for state in states:
            new_states.append()
    
    def step_single_state(states):
        return 

    def perturbation(self, state, scale = 0.1):
        if type(state) == list:
            perturb_states = []
            
        return perturb_states