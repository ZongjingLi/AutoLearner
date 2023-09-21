import torch
import torch.nn as nn

import numpy as np

def get_box(box_params):
    center = box_params[0: 3]
    lengths = box_params[3: 6]
    dir_1 = box_params[6: 9]
    dir_2 = box_params[9:]