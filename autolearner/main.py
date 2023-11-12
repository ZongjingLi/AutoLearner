from config import *
from model import *
from train import *

weights = {"reconstruction":1.0,"color_reconstruction":1.0,"occ_reconstruction":1.0,"localization":1.0,"chamfer":1.0,"equillibrium_loss":1.0}

argparser = argparse.ArgumentParser()
argparser.add_argument("--name",                    default = "WLK")
argparser.add_argument("--mode",                    default = "scenelearner")
argparser.add_argument("--epochs",                  default = 1132)
argparser.add_argument("--optimizer",               default = "Adam")
argparser.add_argument("--lr",                      default = 1e-3)
argparser.add_argument("--batch_size",              default = 1)
argparser.add_argument("--dataset",                 default = "toy")
argparser.add_argument("--category",                default = ["vase"])
argparser.add_argument("--freeze_perception",       default = False)
argparser.add_argument("--concept_type",            default = False)
argparser.add_argument("--domain",                  default = "sprites")

# [perception and language grounding training]
argparser.add_argument("--perception",              default = "psgnet")
argparser.add_argument("--training_mode",           default = "joint")
argparser.add_argument("--alpha",                   default = 1.00)
argparser.add_argument("--beta",                    default = 1.0)
argparser.add_argument("--loss_weights",            default = weights)

# [additional training details]
argparser.add_argument("--warmup",                  default = True)
argparser.add_argument("--warmup_steps",            default = 300)
argparser.add_argument("--decay",                   default = False)
argparser.add_argument("--decay_steps",             default = 20000)
argparser.add_argument("--decay_rate",              default = 0.99)
argparser.add_argument("--shuffle",                 default = True)

# [checkpoint details]
argparser.add_argument("--checkpoint_dir",          default = None)

args = argparser.parse_args()
args.batch_size = int(args.batch_size)
args.lr = float(args.lr)

model = AutoLearner(config)

if args.checkpoint_dir is not None:
    model.restore_parameters(args.checkpoint_dir)

model = model.to(device)

if args.mode == "scenelearner":
    if args.dataset in ["Sprites","Acherus","Toys","PTR","Hearth","Battlecode"]:
        print("start the image domain training session.")
        train_scenelearner(model, config, args)

if args.mode == "nerf":
    if args.dataset in ["PAC"]:
        print("start the training of the nerf.")
        train_nerf(model, config,args)