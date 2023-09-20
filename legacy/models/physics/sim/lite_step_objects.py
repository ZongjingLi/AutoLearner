import pybullet as p
import pybullet_data
import os
from utils.geometry import reverse_euler, reverse_xyz

class LiteObjectManager(object):
    def __init__(self, config, obj_dir, forward = True):
        p.resetSimulation()
        self.obj_dir = obj_dir


class NeuroObjectManager(object):
    def __init__(self, config, forward = True):
        super().__init__()
