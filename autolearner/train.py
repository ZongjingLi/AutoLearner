import karanir
from .datasets import *
import time
import datetime
from torch.utils.tensorboard import SummaryWriter

def train_nerf(train_model, config, args):
    return 

def train_segment(train_model, config, args, query = False):
    # [Create the Dataloader]
    if args.dataset == "Hearth":
        train_dataset = HearthDataset("train", resolution = config.resolution)
    if args.dataset == "Battlecode":
        train_dataset = BattlecodeImageData("train", resolution = config.resolution)
    if args.dataset == "PTR":
        train_dataset = PTRData("train", resolution = config.resolution)
        val_dataset =  PTRData("val", resolution = config.resolution)
    if args.dataset == "Toys" :
        if query:
            train_dataset = ToyDataWithQuestions("train", resolution = config.resolution)
        else:
            train_dataset = ToyData("train", resolution = config.resolution)
    if args.dataset == "Sprites":
        if query:
            train_dataset = SpriteWithQuestions("train", resolution = config.resolution)
        else:
            train_dataset = SpriteData("train", resolution = config.resolution)
    if args.dataset == "Acherus":
        train_dataset = AcherusDataset("train")

    if args.training_mode == "query":
        karanir.utils.tensor.freeze(train_model.perception)

    dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = args.shuffle); B = args.batch_size

    # [setup the optimizer and lr schedulr]
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(train_model.parameters(), lr = args.lr)
    if args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(train_model.parameters(), lr = args.lr)

    # [start the training process recording]
    itrs = 0
    start = time.time()
    logging_root = "./tf-logs"
    ckpt_dir     = os.path.join(logging_root, 'checkpoints')
    events_dir   = os.path.join(logging_root, 'events')
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    if not os.path.exists(events_dir): os.makedirs(events_dir)
    writer = SummaryWriter(events_dir)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for sample in dataloader:
            itrs += 1

            # [Compute Optical Flow Segment]

            # [Afffinity Coercion]
            outputs = model(sample)

            # [Calculate the Working Loss]
            working_loss = 0.0

            # [Log the Epoch Loss]
            epoch_loss += working_loss

            if itrs % config.checkpoint_itrs:
                pass

def train_scenelearner(model, args, config):
    # [Create the Dataloader]
    if args.dataset == "Hearth":
        train_dataset = HearthDataset("train", resolution = config.resolution)
    if args.dataset == "Battlecode":
        train_dataset = BattlecodeImageData("train", resolution = config.resolution)
    if args.dataset == "PTR":
        train_dataset = PTRData("train", resolution = config.resolution)
        val_dataset =  PTRData("val", resolution = config.resolution)
    if args.dataset == "Toys" :
        if query:
            train_dataset = ToyDataWithQuestions("train", resolution = config.resolution)
        else:
            train_dataset = ToyData("train", resolution = config.resolution)
    if args.dataset == "Sprites":
        if query:
            train_dataset = SpriteWithQuestions("train", resolution = config.resolution)
        else:
            train_dataset = SpriteData("train", resolution = config.resolution)
    if args.dataset == "Acherus":
        train_dataset = AcherusDataset("train")

    if args.training_mode == "query":
        Karanir.utils.tensor.freeze(train_model.perception)

    dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = args.shuffle); B = args.batch_size

    # [setup the optimizer and lr schedulr]
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(train_model.parameters(), lr = args.lr)
    if args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(train_model.parameters(), lr = args.lr)

    # [start the training process recording]
    itrs = 0
    start = time.time()
    logging_root = "./tf-logs"
    ckpt_dir     = os.path.join(logging_root, 'checkpoints')
    events_dir   = os.path.join(logging_root, 'events')
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    if not os.path.exists(events_dir): os.makedirs(events_dir)
    writer = SummaryWriter(events_dir)
    for epoch in range(args.epochs):
        pass

def train_physics(model, config, args):
    # suppose this model already have a good enough perception moduloe
    for epoch in range(config.epochs):
        pass

def train_reasoning(model, config, args):
    pass