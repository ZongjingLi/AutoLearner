# [config setup]

import argparse 
parser = argparse.ArgumentParser()

parser.add_argument("--name",               default = "Morgoth")
parser.add_argument("--imsize",             default = 64)
parser.add_argument("--node_size",          default = 32)
parser.add_argument("--concept_dim",        default = 100)
parser.add_argument("--predicate_dim",      default = 100)
parser.add_argument("--spatial_dim",        default = 7 * 2 + 1)
parser.add_argument("--visual_path",        default = "checkpoints/constant_qtr_model.ckpt")

config = parser.parse_args(args = [])

