import pybullet as p
import os

from .lite_step_objects import LiteObjectManager

step_time = 4
sim_time_step = 0.01

def step(objects, forward = True):
    """Step a simulation"""
    content_folder = ""
    om = LiteObjectManager(objects, os.path.joint(content_folder,"physics/data"), forward = forward)
    p.setTimeStep(sim_time_step)

    for i in range(step_time):
        p.stepSimulation()
    new_objects = []
    for object in om.object_ids:
        new_objects.append(om.get_object_motion(object))
    
    return new_objects

def reverse_step(config):
    """Reverse step a simulation"""
    return step(config, False)

def neuro_step(objects, forward = True):
    new_objects = []
    return new_objects

def neuro_reverse_step(config):
    return step(config, False)