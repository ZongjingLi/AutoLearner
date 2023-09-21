from config import *
from model import *
from train import *

argparser = argparse.ArgumentParser()
argparser.add_argument("--mode",        default = "scenelearner")
argparser.add_argument("--epochs",      default = 1132)

args = argparser.parse_args()

model = AutoLearner(config)

if args.checkpoint_dir is not None:
    model.restore_parameters(args.checkpoint_dir)

model = model.to(device)

if args.mode == "scenelearner":
    train_scenelearner(model, config, args)