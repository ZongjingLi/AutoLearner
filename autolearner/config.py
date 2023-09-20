import argparse
from Karanir import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--device",         default = device)
parser.add_argument("--conv_dim",       default = 132)

config = parser.parse_args()