import argparse
from Karanir import *

from model.knowledge.symbolic import *
translator = {"scene":Scene,"exist":Exist,"filter":Filter,"union":Union,"unique":Unique,"count":Count,
              "parents":Parents,"subtree":Subtree}

local = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"

root_path = "/Users/melkor/Documents/GitHub/AutoLearner/AutoLearner" if local else "AutoLearner"

parser = argparse.ArgumentParser()
parser.add_argument("--device",                     default = device)
parser.add_argument("--root",                       default = root_path)
parser.add_argument("--dataset_root",               default = "/Users/melkor/Documents/datasets")
parser.add_argument("--name",                       default = "AutoLearner")
parser.add_argument("--latent_dim",                 default = 128)

# [Knowledge]
parser.add_argument("--concept_type",               default = "box")
parser.add_argument("--object_dim",                 default = 100)
parser.add_argument("--concept_dim",                default = 100)
parser.add_argument("--temperature",                default = 0.1)
parser.add_argument("--entries",                    default = 100)
parser.add_argument("--method",                     default = "uniform")
parser.add_argument("--center",                     default = [-0.25,0.25])
parser.add_argument("--offset",                     default = [-0.0,0.0])
parser.add_argument("--domain",                     default = "demo")
parser.add_argument("--translator",                 default = translator)

# [Perception]
parser.add_argument("--channels",                   default = 3)
parser.add_argument("--resolution",                 default = [128,128])
parser.add_argument("--conv_dim",                   default = 132)
parser.add_argument("--perception_size",            default = 1)
parser.add_argument("--spatial_dim",                default = 2)

# [Physics]

# [Reasoning]


config = parser.parse_args(args = [])