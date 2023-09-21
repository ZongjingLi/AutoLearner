import argparse
from Karanir import *

local = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"

root_path = "/Users/melkor/Documents/GitHub/SceneGraphLearner" if local else "SceneGraphLearner"

parser = argparse.ArgumentParser()
parser.add_argument("--device",                     default = device)
parser.add_argument("--root",                       default = root_path)
parser.add_argument("--dataset_root",               default = "/Users/melkor/Documents/datasets")
parser.add_argument("--name",                       default = "AutoLearner")

# [Knowledge]
parser.add_argument("--object_dim",                 default = 100)
parser.add_argument("--concept_dim",                default = 100)

# [Perception]
parser.add_argument("--channels",                   default = 3)
parser.add_argument("--resolution",                 default = [128,128])
parser.add_argument("--conv_dim",                   default = 132)
parser.add_argument("--perception_size",            default = 1)
parser.add_argument("--spatial_dim",                default = 2)

# [Physics]

# [Reasoning]


config = parser.parse_args(args = [])