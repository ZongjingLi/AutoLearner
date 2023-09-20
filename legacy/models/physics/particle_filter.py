import random
import time
import os

import logging
from copy import deepcopy, copy
from statistics import mean

import pycocotools.mask as mask_util
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import pybullet as p
import pybullet_data
import numpy as np

from utils import * 
from .match import Matcher
from .step import Stepper
from utils.geometry import iou

from .sim import *

class _ParticleUpdater(object):
    def __init__(self, config, t, belief, camera, observation_history = None):
        self.camera = camera
        self.belief = deepcopy(belief)

        if observation_history is None:
            self.observation_history = [deepcopy(belief)]
        else:
            self.observation_history = copy(observation_history)
        
        self.n_objects = 0

        # TODO: Matcher and Stepper
        self.matcher = Matcher(config)
        self.stepper = Stepper(config)

        self.t = t
        self.area_threhold = config.area_threshold
        #TODO: config.area_threshold

        samples_mass = config.mass
        #TODO: config.mass

        self.to_sample_mass = config.to_sample_mass
        #TODO: to sample mass

    
    def step(self):
        "Belief Step"
        self.belief, magic_penalty = self.stepper.step(self.belief)
        self.magic_penalty += magic_penalty
        self.t += 1

class FilterUpdate(object):
    def __init__(self, config, belief, case_name, camera, n_filter):
        self.t = 0
        self.config = config


class NeuroParticleFilter(nn.Module):
    def __init__(self, config):
        super().__init__()

        # [Stepper] aka prediction module
        self.stepper = PropNet(config)

        # [Matcher] ask observation module
        self.matcher = Matcher(config)
